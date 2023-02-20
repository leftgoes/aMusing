import numpy as np
from typing import Iterator

from .colormaps import ColorMap
from ..printing import ProgressPrinter
from . import audiomanip

NoteInfo = str | float
T = float | np.ndarray


def linmap(x: T, from_range: tuple[T, T], to_range: tuple[T, T]) -> T:
    return to_range[0] + (to_range[1] - to_range[0]) * (x - from_range[0])/(from_range[1] - from_range[0])


class Munim:
    def __init__(self, fps: int = 30, width: int = 1920, height: int = 1080, cmap: ColorMap | None = None) -> None:
        self.fps = fps
        self.width = width
        self.height = height

        self.sample_rate: int = None

        self._audio: np.ndarray = None
        self._progress = ProgressPrinter()

        if cmap is None:
            self._cmap = ColorMap('viridis')
        else:
            self._cmap = cmap
            
    @property
    def frame_length(self) -> int:
        return round(self.sample_rate / self.fps)

    def frames(self) -> Iterator[int]:
        for t0 in range(0, self._audio.shape[0] - self.frame_length, self.frame_length):
            yield t0

    def read_audio(self, filepath: str, start: float = 0, end: float = 1) -> None:
        data, self.sample_rate = audiomanip.open_audio(filepath)
        self._audio = audiomanip.audio2mono_start_end(data, start, end)

    def render_video(self, *args, **kwargs) -> None:
        return
