from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class SingleColorMap:
    def __init__(self, color_rgb: tuple[float], t0: float = 0.5, colormax: float = 1) -> None:
        self.t0 = t0
        self.color_rgb = (color_rgb[0] / colormax, color_rgb[1] / colormax, color_rgb[2] / colormax)

    def get(self, arr: np.ndarray) -> np.ndarray:
        if len(arr.shape) == 1:
            return self._get_1darray(arr)
        else:
            raise NotImplemented

    def _get_single(self, t: float) -> tuple[float]:
        if t < self.t0:
            a = t / self.t0
            return (a * self.color_rgb[2], a * self.color_rgb[1], a * self.color_rgb[0])
        else:
            a = (t - self.t0)/(1 - self.t0)
            return (self.color_rgb[2] + a * (1 - self.color_rgb[2]),
                    self.color_rgb[1] + a * (1 - self.color_rgb[1]),
                    self.color_rgb[0] + a * (1 - self.color_rgb[0]))
    
    def _get_1darray(self, arr: np.ndarray) -> np.ndarray:
        colorized = np.empty((arr.shape[0], 3))
        for i, x in enumerate(arr):
            r, g, b = self._get_single(x)
            colorized[i, 0] = r
            colorized[i, 1] = g
            colorized[i, 2] = b
        return colorized


class ColorMap:
    def __init__(self, cmap: str | SingleColorMap = 'viridis'):
        if isinstance(cmap, SingleColorMap):
            self._cmap = cmap.get
        elif cmap in plt.colormaps():
            self._cmap = cm.get_cmap(cmap)
        else:
            raise NotImplemented
    
    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return self.get(arr)

    def get(self, arr: np.ndarray) -> np.ndarray:
        return self._cmap(arr)

