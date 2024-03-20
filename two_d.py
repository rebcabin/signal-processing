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
class Image:
    img: np.ndarray


def bw_square_u8_image(N: int) -> np.ndarray:
    assert (N % 4 == 0)

    result = np.zeros((N, N), dtype=np.uint8)
    assert _is_suitable_u8_img(result)
    # Start a quarter of the way from the left and top.
    i1: int = N // 4  # e.g. 8 for N=32
    # End three quarters of the way from the right and bottom.
    i2: int = i1 + (N // 2)  # 8 + 16 = 24 for N=32
    N2 = N // 2
    result[i1:i2, i1:i2] = np.ones((N2, N2), dtype=np.uint8)
    return result


def _is_suitable_u8_img(img: np.ndarray) -> bool:
    N = img.shape[0]
    is_square = (N == img.shape[1])
    assert is_square  # is square
    is_p_of_2 = _is_power_of_two(N)
    assert is_p_of_2
    return is_p_of_2 and is_p_of_2


def bw_square_u8_resized_white_portions(img: np.ndarray) -> list[np.ndarray]:
    assert _is_suitable_u8_img(img)
    result = []
    N = img.shape[0]  # e.g., 32
    s = N // 2  # 16, 8, 4
    while s >= 4:
        s = s // 2        #  8,  4,  2
        j = (N - s) // 2  #  4,  6,  7  (16-8)//2=4, (16-4)//2=6, (16-2)//2=7
        k = j + s         # 12, 10,  9  4+8=12, 6+4=10, 7+2=9
                          # The sum is always 16.
        temp = np.zeros((N, N), dtype=np.uint8)
        temp[j:k, j:k] = np.ones((s, s), dtype=np.uint8)
        result.append(temp)
    return result
