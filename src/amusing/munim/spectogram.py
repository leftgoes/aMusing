import cv2
import numpy as np
from scipy.ndimage import zoom

from .munim import Munim, NoteInfo
from .colormaps import ColorMap


class Spectrum(Munim):
    a4_hz: float = 440
    _note_offset: dict[str, int] = {'C': -9, 'D': -7, 'E': -5, 'F': -4,
                                    'G': -2, 'A': 0, 'B': 2}
    _accidental_offset: dict[str, int] = {'bb': -2, 'b': -1, '#': 1, 'x': 2}  

    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str | ColorMap = 'magma') -> None:
        super().__init__(fps, width, height, cmap)

        self._transformed: np.ndarray = None

    def hz(self, note: NoteInfo) -> float:
        if isinstance(note, float) or isinstance(note, int): return note
        elif not isinstance(note, str): raise ValueError(f'cannot get frequency from {note!r}')

        pitch = self._note_offset[note[0].upper()]
        if len(note) == 1:
            return self.a4_hz * 2**(pitch/12)
        elif len(note) == 2:
            if note[1] in self._accidental_offset:
                return self.a4_hz * 2**((pitch + self._accidental_offset[note[1]])/12)
            else:
                return self.a4_hz * 2**(pitch/12 + int(note[1]) - 4)
        else:
            accidental, octave = note[1:-1], note[-1]
            return self.a4_hz * 2**((pitch + self._accidental_offset[accidental])/12 + int(octave) - 4)

    def normalized(self, gamma: float, highlight_clip: float, invert_intensity: bool) -> np.ndarray:
        freq = self._transformed - self._transformed.min()
        freq = np.clip(highlight_clip * freq/freq.max(), 0, 1)**gamma
        return 1 - freq if invert_intensity else freq

    def render_video(self, filepath: str, fourcc: str = 'mp4v', gamma: float = 1, highlight_clip: float = 1,
                     colorize: bool = True, invert_intensity: bool = False, invert_color: bool = False, show_frames: bool = False) -> None:
        self._progress.start()
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height), colorize)

        frames = self.normalized(gamma, highlight_clip, invert_intensity)
        for i, frame_data in enumerate(frames):
            zoomed = frame_data if frame_data.shape[0] == self.width else zoom(frame_data, self.width / frame_data.shape[0])
            
            if colorize:
                colored = self._cmap(zoomed)[:, :3][:, ::-1]
                if invert_color: colored = 1 - colored
                frame = np.uint8(255 * np.tile(colored, (self.height, 1, 1)))
            else:
                frame = np.uint8(255 * np.tile(zoomed, (self.height, 1)))

            if show_frames:
                cv2.imshow(f'frames', frame)
                cv2.waitKey(0)

            videoout.write(frame)
            self._progress.string((i + 1)/len(frames))
        videoout.release()
    
    def show_spectrum(self, gamma: float = 1) -> None:
        cv2.imshow(self.__class__.__name__, np.uint8(255 * (self._transformed/self._transformed.max())**gamma))
        cv2.waitKey(0)

    def save_spectrum(self, filepath: str, gamma: float = 1, dtype: np.dtype | str = 'uint16') -> None:
        cv2.imwrite(filepath, (np.iinfo(dtype).max * (self._transformed/self._transformed.max())**gamma).astype(dtype))

    def transform(self, *args, **kwargs) -> None:
        return


class STFT(Spectrum):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str | ColorMap = 'magma') -> None:
        super().__init__(fps, width, height, cmap)

    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', snippet_frames_num: float = 20) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)
        length = round(snippet_frames_num * self.frame_length)

        stft_x = np.fft.fftfreq(length, 1 / self.sample_rate)
        freqs = np.exp(np.linspace(np.log(min_hz), np.log(max_hz), length))
        indices = np.array([np.argmin(np.abs(stft_x - freq)) for freq in freqs])
        freq_sample_num = indices.shape[0]

        self._transformed = np.empty((int(self._audio.shape[0] / self.frame_length), freq_sample_num))
        for i, t0 in enumerate(self.frames()):
            if t0 + length > self._audio.shape[0]:
                break
            snippet = self._audio[t0:t0 + length]
            stft_y = np.fft.fft(snippet)
            self._transformed[i, :] = np.abs(stft_y[indices])


class Wavelet(Spectrum):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str | ColorMap = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
    
    @staticmethod
    def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-x**2/(2 * sigma**2))

    def psi(self, x: np.ndarray, f: float, sigma: float) -> np.ndarray:
        return

    def transform(self, lowest: NoteInfo = 'A0', highest: NoteInfo = 'C8', *, sigma_seconds: float = 0.1, radius: float = 4) -> None:
        min_hz, max_hz = self.hz(lowest), self.hz(highest)

        sigma = round(self.sample_rate * sigma_seconds)  # sigma in samples
        length = round(2*radius*sigma)

        freqs = 2**np.linspace(np.log2(min_hz), np.log2(max_hz), self.width)
        x = np.arange(-radius*sigma, radius*sigma)

        self._progress.start()
        self._transformed = np.empty((int(self._audio.shape[0] / self.frame_length) + 1, freqs.shape[0]))
        for j, f in enumerate(freqs):
            wavelet = self.psi(x, f/self.sample_rate, sigma)
            for i, t0 in enumerate(self.frames()):
                if t0 + length > self._audio.shape[0] - 1:
                    snippet = self._audio[t0:]
                    self._transformed[i, j] = np.abs(np.dot(wavelet[:snippet.shape[0]], snippet)) * np.log(f)
                else:
                    snippet = self._audio[t0:t0 + length]
                    self._transformed[i, j] = np.abs(np.dot(wavelet, snippet)) * np.log(f)
                self._progress.string(((i + 1)/self._transformed.shape[0] + j)/freqs.shape[0])
        self._progress.string(1)


class Morlet(Wavelet):
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: str | ColorMap = 'magma') -> None:
        super().__init__(fps, width, height, cmap)
    
    def psi(self, x: np.ndarray, f: float, sigma: float) -> np.ndarray:
        return self.gaussian(x, sigma) * np.exp(1j * 2*np.pi*f * x)
