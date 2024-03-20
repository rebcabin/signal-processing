"""motivated by https://github.com/rebcabin/Digital-Image-Processing"""

import numpy as np
from dataclasses import dataclass
# import matplotlib.pyplot as plt
# import PIL.Image as Image
# import math
# import cmath
# import time
# import csv
#
# from numpy import binary_repr


def return_true() -> bool:
    return True


def _is_power_of_two(n: int) -> bool:
    return n & (n - 1) == 0


@dataclass
class ImageU8:
    img: np.ndarray

    def _is_suitable(self) -> bool:
        N = self.img.shape[0]
        is_square = (N == self.img.shape[1])
        assert is_square  # is square
        is_p_of_2 = _is_power_of_two(N)
        assert is_p_of_2
        return is_p_of_2 and is_p_of_2

    @staticmethod
    def bw_square(N: int) -> "ImageU8":
        """factory"""
        assert (N % 4 == 0)
        it = ImageU8(np.zeros((N, N), dtype=np.uint8))
        assert it._is_suitable()
        # Start a quarter of the way from the left and top.
        i1: int = N // 4  # e.g. 8 for N=32
        # End three quarters of the way from the right and bottom.
        i2: int = i1 + (N // 2)  # 8 + 16 = 24 for N=32
        N2 = N // 2
        it.img[i1:i2, i1:i2] = np.ones((N2, N2), dtype=np.uint8)
        return it

    def bw_square_resized_white_portions(self) -> list["ImageU8"]:
        assert self._is_suitable()
        result = []
        N = self.img.shape[0]  # e.g., 32
        s = N // 2  # 16, 8, 4
        while s >= 4:
            s = s // 2        #  8,  4,  2
            j = (N - s) // 2  #  4,  6,  7  (16-8)//2=4, (16-4)//2=6, (16-2)//2=7
            k = j + s         # 12, 10,  9  4+8=12, 6+4=10, 7+2=9
                              # The sum is always 16.
            temp = np.zeros((N, N), dtype=np.uint8)
            temp[j:k, j:k] = np.ones((s, s), dtype=np.uint8)
            result.append(ImageU8(temp))
        return result

    def plot(self, ax, idx: int) -> None:
        vmin = np.iinfo(np.uint8).min
        vmax = 1  # remap [0..255] to [0..1]
        ax[idx].imshow(self.img, cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].set_title('Original image')



