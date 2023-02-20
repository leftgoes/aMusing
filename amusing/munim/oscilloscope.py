import cv2
import numpy as np

from .munim import Munim, linmap
from . import audiomanip


class Oscilloscope(Munim):
    def __init__(self, fps: int = 30, width: int = 1080, cmap: str = 'magma') -> None:
        super().__init__(fps, width, -1, cmap)
    
    def cmap(self, x: np.ndarray) -> np.ndarray:
        return self._cmap(x)[:,:,:3][:,:,::-1]

    def read_audio(self, filepath: str, start: float = 0, end: float = 1) -> None:
        data, self.sample_rate = audiomanip.open_audio(filepath)

        if len(data.shape) in (1, 2):
            self._audio = data[round(start * data.shape[0]):round(end * data.shape[0]) - 1]/np.abs(data).max()

    def render_video(self, filepath: str, fourcc: str = 'mp4v', darken: float = 0.05) -> None:
        self._progress.start()
        videoout = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.width), True)
        length = self.frame_length

        frame = np.zeros((self.width, self.width))
        if len(self._audio.shape) == 1:
            pass
        else:
            for i, (right, left) in enumerate(self._audio):
                x = int(linmap(right, (-1, 1), (0, self.width - 1)))
                y = int(linmap(left, (-1, 1), (0, self.width - 1)))

                frame[y, x] += (1 - (i % length) / length) * 0.07
                if i % length == 0:
                    frame *= 1 - darken
                    videoout.write(np.uint8(255 * self.cmap(np.clip(frame, 0, 1))))
                    self._progress.string(i/self._audio.shape[0])
            
        self._progress.string(1)
        videoout.release()
