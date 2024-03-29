from xml.etree.ElementTree import XMLParser, Element, TreeBuilder, parse as parse_etree
from typing import Self, Iterator, Iterable

CHORD_SUB: list[str] = ['Accidental', 'Stem', 'NoteDot', 'Note', 'Hook']
GRACENOTE: set[str] = {'grace4', 'acciaccatura', 'appoggiatura', 'grace8after', 'grace16', 'grace16after', 'grace32', 'grace32after'}
INVISIBILIZE: set[str] = {'Accidental', 'Articulation',
                          'BarLine', 'Beam', 'Clef', 'Dynamic', 'Fermata', 'Fingering',
                          'HairPin', 'Hook', 'KeySig', 'Note', 'Ottava', 'Pedal', 'Rest',
                          'Segment', 'Slur', 'SlurSegment', 'StaffText', 'Stem', 'StemSlash', 'SystemText',
                          'Tempo', 'TextLine', 'Tie', 'TieSegment', 'TimeSig', 'Tremolo', 'Trill'}
UNPRINTABLE: set[str] = {'visible', 'irregular', 'stretch', 'startRepeat', 'endRepeat', 'MeasureNumber', 'LayoutBreak', 'noOffset', 'vspacerUp', 'vspacerDown', 'vspacerFixed'}


class Note:
    def __init__(self, note_type: int) -> None:
        self._value: float = 1024 / note_type

    @property
    def ntype(self) -> float:
        return 1024 / self._value
    
    @property
    def value(self) -> float:
        return self._value

    @classmethod
    def from_text(cls, text: str) -> Self:
        if text == 'breve':
            return cls(0.5)
        elif text == 'whole':
            return cls(1)
        elif text == 'half':
            return cls(2)
        elif text == 'quarter':
            return cls(4)
        elif text == 'eighth':
            return cls(8)
        else:
            return cls(int(text[:-2]))
    
    def half(self) -> Self:
        self._value /= 2
        return self

    def dot(self) -> Self:
        self._value *= 1.5
        return self
    
    def n_dot(self, dots: int) -> Self:
        self._value *= sum(1/2**i for i in range(dots + 1))
        return self

    def triplet(self) -> Self:
        self._value *= 2/3
        return self
    
    def n_tuplet(self, actual_notes: int, normal_notes: int) -> Self:
        self._value *= normal_notes/actual_notes
        return self


class MElementTagError(Exception):
    """Raises when MElement targeted to edit has the wrong Tag"""

    @classmethod
    def from_tags(cls, actual_tag: str, possible_tags: str | Iterable[str], msg: str | None = None):
        error_msg = f"MElement has tag {actual_tag!r}, should be "

        if isinstance(possible_tags, str):
            error_msg += repr(possible_tags)
        else:
            error_msg += ' or '.join(repr(tag) for tag in possible_tags)

        if msg:
            error_msg += f': {msg}'

        return cls(error_msg)


class MElement(Element):
    def __init__(self, tag: str, attrib: dict[str, str] = {}, **extra: str) -> None:
        super().__init__(tag, attrib, **extra)
        self.visibility_locked: bool = tag == 'Measure'

    @classmethod
    def new_element(cls, tag: str, text: str | None = None, visible: bool = True) -> Self:
        elem = cls(tag)
        if text is not None:
            elem.text = text
        if not visible:
            elem.set_invisible()
        return elem

    @staticmethod
    def get_next_chord(voice: Self, index: int) -> tuple['MElement', int]:
        if voice.tag != 'voice':
            raise MElementTagError.from_tags(voice.tag, 'voice', 'cannot get next chord')
        for i, subelement in enumerate(voice[index + 1:]):
            if subelement.tag == 'Chord':
                return subelement, i + index + 1

    def append_new(self, tag: str, text: str | None = None, visible: bool = True) -> None:
        self.append(type(self).new_element(tag, text, visible))

    def contains(self, tag: str) -> bool:
        return self.find(tag) is not None

    def add_implied_children(self) -> None:
        if self.tag == 'Chord':
            for subtag in ('Stem', 'Beam', 'Hook'):
                if not self.contains(subtag):
                    self.append_new(subtag)

        if self.tag == 'Tie':
            if not self.contains('TieSegment'):
                self.append_new('TieSegment')
        
        if self.contains('acciaccatura') and not self.contains('StemSlash'):
            self.append_new('StemSlash')

    def invisibilize(self) -> None:
        if self.tag in UNPRINTABLE:
            return

        if self.tag in INVISIBILIZE:
            self.set_invisible()

    def _is_visible(self) -> tuple[Self, bool]:
        visible = self.find('visible')
        if visible is None:
            return None, True
        else:
            return visible, visible.text != '0'

    def lock_visibility(self) -> None:
        self.visibility_locked = True

    def is_protected(self) -> bool:
        return self.visibility_locked

    def tremolo_subtype(self, duration_type: float) -> tuple[str, Note]:
        if self.tag != 'Tremolo':
            raise MElementTagError.from_tags(self.tag, 'Tremolo', 'cannot get tremolo subtype')

        subtype_text = self.find('subtype').text
        tremolo_timediff = int(subtype_text[1:])

        return subtype_text[0], Note(tremolo_timediff * max(1, Note(4).value/duration_type))

    def chord_subelements(self) -> set:
        subelems = set()
        for elem in self.get_chord_subelements():
            subelems.add(elem)
        return subelems

    def get_chord_subelements(self) -> Iterator[Self]:
        if self.tag != 'Chord':
            raise MElementTagError.from_tags(self.tag, 'Chord', 'cannot get chord chord')
        for tag in CHORD_SUB:
            for element in self.iter(tag):
                yield element

    def duration_value(self, timesig: float) -> tuple[float, float]:
        if self.tag not in ('Chord', 'Rest'):
            raise MElementTagError.from_tags(self.tag, ['Chord', 'Rest'], 'cannot get duration')

        if (dots := self.find('dots')) is None:
            dotted = 1
        else:
            dotted = sum(1/2**i for i in range(int(dots.text) + 1))
        
        duration_type_text = self.find('durationType').text
        if duration_type_text == 'measure':
            duration_type = timesig
        else:
            duration_type = Note.from_text(duration_type_text).value

        return dotted, duration_type

    def is_tuplet(self) -> bool:
        return self.tag == 'Tuplet' or self.tag == 'endTuplet'

    def tuplet_value(self) -> float:
        if self.tag == 'Tuplet':
            return int(self.find('normalNotes').text)/int(self.find('actualNotes').text)
        elif self.tag == 'endTuplet':
            return 1.0
        else:
            raise MElementTagError.from_tags(self.tag, ['Tuplet', 'endTuplet'], 'cannot get tuplet value')

    def duration_offset(self) -> float:
        if self.tag != 'location':
            raise MElementTagError.from_tags(self.tag, 'location', 'cannot get offset')
        return eval(self.find('fractions').text) * Note(1).value

    def has_arpeggio(self) -> bool:
        if self.tag != 'Chord':
            raise MElementTagError.from_tags(self.tag, 'Chord')
        return self.find('Arpeggio') is None

    def is_unprintable(self) -> bool:
        return self.tag in UNPRINTABLE

    def is_gracenote(self) -> bool:
        return any(subelem.tag in GRACENOTE for subelem in self)

    def is_visible(self) -> bool:
        _, _visible = self._is_visible()
        return _visible
    
    def set_visible(self) -> None:
        if self.is_protected():
            return

        e, is_visible = self._is_visible()
        if not is_visible:
            self.remove(e)
    
    def set_invisible(self) -> None:
        if self.is_protected():
            return

        is_visible = self.is_visible()
        if not is_visible: return

        invisible = type(self)('visible', {})
        invisible.text = '0'
        self.append(invisible)
    
    def set_visible_all(self, tag: str | None = None, protected: set[Self] | None = None) -> None:
        if protected is None:
            protected = set()

        for subelement in self.iter(tag):
            subelement: Self
            if subelement in protected:
                continue
            subelement.set_visible()
    
    def set_invisible_all(self, tag: str | None = None) -> None:
        for subelement in self.iter(tag):
            subelement: Self
            subelement.set_invisible()
    
    def set_visible_chord(self) -> None:
        for element in self.get_chord_subelements():
            element.set_visible()
    
    def set_invisible_chord(self) -> None:
        for element in self.get_chord_subelements():
            element.set_invisible()


def parse_custom_etree(source: str):
    treebuilder = TreeBuilder(element_factory=MElement)
    parser = XMLParser(target=treebuilder)
    tree = parse_etree(source, parser)
    return tree
