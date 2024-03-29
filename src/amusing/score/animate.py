import copy
import cv2
import logging
import numpy as np
import os
from time import sleep
from threading import Thread
from typing import Iterator, Sequence
from xml.etree.ElementTree import ElementTree

from ..printing import ProgressPrinter
from .mscx import MElement, parse_custom_etree, Note

IntPage = int
StrPath = str


def frame_index_to_path(index: int, page: int | None = None) -> StrPath:
    if page is None:
        return f'frm{index:04d}.png'
    else:
        return f'frm{index:04d}-{page}.png'


class Amusing:
    first_measure_num: int = 1
    musescore_executable_path: str = 'MuseScore3.exe'
    temp_filename: str = '.amusing_thread'
    tempdir = '__temp__'

    def __init__(self, width: int, outdir: str = 'frames', *, 
                       threads: int = 8, log_file: str | None = None,
                       delete_temp: bool = True, print_progress: bool = True,
                       first_emtpy_frame: bool = True, frame0: int = 0) -> None:
        self.width = width
        self.outdir = outdir
        self.threads = threads
        self.delete_temp = delete_temp
        self.print_progress = print_progress
        self.first_emtpy_frame = first_emtpy_frame
        self.frame0 = frame0
        
        self.jobs: dict[int, Note] = {}
        self._progress = ProgressPrinter()
        self._tree: ElementTree = None
        
        self._measures_num: int = None
        self._score_width: float = None
        self._timesigs: np.ndarray = None
        self._page_num: int = None

        self._filepath: tuple[str, str] = None

        logging.basicConfig(filename=log_file,
                            level=logging.WARNING,
                            format='[%(levelname)s:%(filename)s:%(lineno)d] %(message)s')

    @classmethod
    def convert(cls, from_path: str, to_path: str, dpi: int | None = None) -> None:
        cmd = f'{cls.musescore_executable_path} {from_path} --export-to {to_path}'
        if dpi is not None:
            cmd += f' -r {dpi}'
        os.system(cmd)

    @staticmethod
    def _last_element_in_measure(duration: float, max_duration) -> bool:
        return max_duration < duration - 0.01

    @staticmethod
    def remove_file(filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f'removed {filepath!r}')
            return True
        else:
            logging.warning(f'tried to remove {filepath!r} but not found')
            return False

    def temp_path(self, thread_index: int) -> str:
        return f'{self.tempdir}\\{self.temp_filename}_Thread-{thread_index:02d}.mscx'

    def _print_progress(self, *args, **kwargs) -> None:
        if self.print_progress: self._progress.string(*args, **kwargs)
    
    def _sort_jobs(self) -> None:
        self.jobs = dict(sorted(self.jobs.items()))

    def _convert(self, index: int, frame: int, page: int, tree: ElementTree, alpha_only: bool) -> None:
        temp_path = self.temp_path(index)
        self._write(tree, temp_path)

        to_file = os.path.join(self.outdir, frame_index_to_path(frame)) 
        self._export(temp_path, to_file)
        for i in range(1, self._page_num + 1):
            if i == page:
                if os.path.exists(to_file): os.remove(to_file)
                os.rename(os.path.join(self.outdir, frame_index_to_path(frame, page)), to_file)
            else:
                os.remove(os.path.join(self.outdir, frame_index_to_path(frame, i)))
        
        if alpha_only:
            alpha = cv2.imread(os.path.join(self.outdir, frame_index_to_path(frame)), -1)[:, :, -1]
            cv2.imwrite(os.path.join(self.outdir, frame_index_to_path(frame)), alpha)

    def _export(self, from_musescore_path: str, to_path: str) -> None:
        type(self).convert(from_musescore_path, to_path, self.width / self._score_width)
        logging.info(f'exported {from_musescore_path=!r} to {to_path=!r}')
    
    def _read_timesigs(self) -> None:
        root = self._tree.getroot()
        staves = root.findall('Score/Staff')
        self._timesigs = np.empty((len(staves[0].findall('Measure')), len(staves)))

        for j, staff in enumerate(staves):
            for i, measure in enumerate(staff.findall('Measure')):
                for element in measure.find('voice'):
                    if element.tag == 'TimeSig':
                        timesig = Note(1).value \
                                * int(element.find('sigN').text) \
                                / int(element.find('sigD').text) 
                    elif element.tag == 'Chord': break
                self._timesigs[i, j] = timesig
                if 'len' in measure.attrib:
                    self._timesigs[i, j] = Note(1).value * eval(measure.attrib['len'])

    def _write(self, tree: ElementTree, to_temp_path: str) -> None:
        tree.write(to_temp_path, encoding='UTF-8', xml_declaration=True)
        logging.info(f'wrote tree to {to_temp_path=!r}')

    def _visibilize_tremolo(self, voice: MElement, element: MElement, element_index: int,
                            duration: float, duration_type: float, chord_duration: float,
                            next_tremolo_element_index: int, max_duration: float,
                            max_tremolo: Note) -> int:
        tremolo: MElement = element.find('Tremolo')
        if tremolo is not None:
            subtype_c, tremolo_timediff = tremolo.tremolo_subtype(duration_type)
            if subtype_c == 'c':
                next_element, next_tremolo_element_index = MElement.get_next_chord(voice, element_index)
                if self._last_element_in_measure(duration + chord_duration, max_duration):
                    if tremolo_timediff.value >= max_tremolo.value - 0.01:
                        tremolo.set_visible()
                        
                        if ((max_duration - duration) % (2 * tremolo_timediff.value)) / tremolo_timediff.value < 1:
                            element.set_visible_chord()
                            next_element.set_invisible_chord()
                        else:
                            element.set_invisible_chord()
                            next_element.set_visible_chord()
                    else:
                        next_element.set_visible_all()
                else:
                    element.set_visible()
                    next_element.set_visible()
        
        return next_tremolo_element_index

    def _visibilize_measure(self, measures: list[MElement], measure_index: int, time_sig: float, max_duration: float, max_tremolo: Note) -> None:
        for staff_index, measure in enumerate(measures):
            for voice in measure:
                voice: MElement

                if voice.is_unprintable():
                    continue
                elif voice.tag != 'voice':
                    logging.warning(f'element with tag={voice.tag!r} in {measure_index=}, {staff_index=}')

                next_tremolo_element_index: int = None
                duration, tuplet, dotted = 0, 1, 1
                for element_index, element in enumerate(voice):
                    element: MElement

                    if element_index == next_tremolo_element_index:
                        next_tremolo_element_index = None
                        element.set_visible_all()
                        continue
                    
                    element.set_visible_all()

                    if element.is_gracenote():
                        pass

                    elif element.tag == 'location':
                        duration += element.duration_offset()
                            
                    elif element.is_tuplet():
                        tuplet = element.tuplet_value()

                    elif element.tag == 'Chord' or element.tag == 'Rest':
                        dotted, duration_type = element.duration_value(time_sig)
                        chord_duration = tuplet * dotted * duration_type

                        next_tremolo_element_index = self._visibilize_tremolo(voice, element, element_index, duration, duration_type,
                                                                              chord_duration, next_tremolo_element_index,
                                                                              max_duration, max_tremolo)
                        
                        duration += chord_duration

                    if self._last_element_in_measure(duration, max_duration):
                        break

    def _get_trees(self, max_tremolo: Note) -> Iterator[tuple[IntPage, ElementTree]]:
        root = self._tree.getroot()

        page: IntPage = 1
        newpage: bool = False

        self._progress.start()
        self._print_progress(0)

        if self.first_emtpy_frame:
            yield 1, copy.deepcopy(self._tree)
            
        staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
        for measure_index, measures in enumerate(zip(*staves)):  
            measures: tuple[MElement]
            for measure in measures:
                for voice in measure:
                    if voice.tag == 'LayoutBreak':
                        if voice.find('subtype').text == 'page':
                            newpage = True
                            break
            
            if measure_index in self.jobs:
                subdivision = self.jobs[measure_index]
                time_sig = self._timesigs[measure_index, 0]
                frame_count: int = round(time_sig / subdivision.value)

                for duration_index, max_duration in enumerate(np.linspace(0, time_sig, frame_count, endpoint=False)):
                    self._visibilize_measure(measures, measure_index, time_sig, max_duration, max_tremolo)
                    yield page, copy.deepcopy(self._tree)
                    self._print_progress((list(self.jobs.keys()).index(measure_index) + duration_index / frame_count) / len(self.jobs))

                for measure in measures:
                    measure.set_visible_all()
            else:
                for measure in measures:
                    measure.set_visible_all()

            if newpage:
                page += 1
                yield page, copy.deepcopy(self._tree)
                newpage = False
        
        for staff in staves:
            if not staff[-1].contains('BarLine'):
                barline = MElement.new_element('BarLine', visible=True)
                barline.append(MElement.new_element('subtype', text='end'))
                staff[-1].append(barline)
                
        yield page, copy.deepcopy(self._tree)
 
    def _invisibilize_measures(self) -> None:
        root = self._tree.getroot()

        staves = [staff.findall('Measure') for staff in root.findall('Score/Staff')]
        for measure_index, measures in enumerate(zip(*staves)):  
            measures: tuple[MElement]
            for measure in measures:
                if measure_index != len(staves[0]) - 1 and not measure[0].contains('BarLine'):
                    measure[0].append_new('BarLine')

                for element in measure.iter():
                    element: MElement
                    if not element.is_visible():
                        element.lock_visibility()
                    else:
                        element.add_implied_children()

                for element in measure.iter():
                    element: MElement
                    element.invisibilize()
                
        self._convert(0, 0, 1, self._tree, True)

    def read_score(self, filepath: str) -> None:
        self._filepath = os.path.splitext(filepath)
        if self._filepath[1] not in ('.mscx', '.mscz'):
            return
        elif self._filepath[1] == '.mscz':
            type(self).convert(filepath, self.tempdir + '.score.mscx')
            self._tree = parse_custom_etree(self.tempdir + '.score.mscx')
        else:
            self._tree = parse_custom_etree(filepath)
        baseroot = self._tree.getroot()

        self._score_width = float(baseroot.find('Score/Style/pageWidth').text)
        self._page_num = 1
        for element in baseroot.iter():
            if element.tag == 'LayoutBreak':
                if element.find('subtype').text == 'page':
                    self._page_num += 1

        self._measures_num = len(baseroot.find('Score/Staff').findall('Measure'))
        self._read_timesigs()

    def add_job(self, measures: int | Sequence[int] | None, subdivision: Note) -> None:
        if isinstance(measures, int):
            self.jobs.update({measures - self.first_measure_num: subdivision})
            logging.info('added job with 1 measure')
        else:
            if isinstance(measures, range):
                if measures.stop == -1:
                    measures = range(measures.start, self._measures_num + self.first_measure_num)
            self.jobs.update({measure - self.first_measure_num: subdivision for measure in measures})
            logging.info(f'added job with {len(measures)} measures')

    def add_job_all_measures(self, subdivision: Note) -> None:
        self.jobs.update({i: subdivision for i in range(self._measures_num)})

    def delete_jobs(self) -> None:
        self.jobs = {}
    
    def generate_frames(self, max_tremolo: Note | None = None, *, alpha_only: bool = True) -> None:
        if max_tremolo is None: max_tremolo = Note(32)
        logging.info(f'generate frames using {self.threads} process{"es" if self.threads != 1 else ""}')
    
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            logging.info(f'create outdir={self.outdir}')

        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
            logging.info(f'create tempdir={self.tempdir}')

        self._progress.start()
        self._sort_jobs()

        self._invisibilize_measures()

        threads: list[Thread] = [Thread() for _ in range(self.threads)]
        for frame, (page, tree) in enumerate(self._get_trees(max_tremolo), start=self.frame0):
            while (free_thread := next((t for t in threads if not t.is_alive()), None)) is None:
                sleep(0.1)
            thread_index = threads.index(free_thread)
            
            threads[thread_index] = Thread(target=self._convert, name=f'Thread {thread_index}', args=(thread_index, frame, page, tree, alpha_only))
            threads[thread_index].start()
        
        for thread in threads:
            thread.join()

        if self.delete_temp:
            logging.info('remove temp files')
            for thread_index in range(self.threads):
                self.remove_file(self.temp_path(thread_index))
            if self._filepath[1] == '.mscz':
                self.remove_file(self.tempdir + '.score.mscx')
        
        if len(os.listdir(self.tempdir)) == 0:
            os.rmdir(self.tempdir)
            logging.info('remove tempdir')

        self._print_progress(1)
