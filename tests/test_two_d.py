import matplotlib.pyplot as plt
import PIL.Image as Image
import math
import cmath
import time
import csv

import numpy as np
from numpy import binary_repr

import pytest

from two_d import (return_true, ImageU8)


def test_test_itself():
    assert return_true()


def test_64_u8_img():
    fig, ax = plt.subplots(1, 5, figsize=(17, 17))
    # N = 64
    # img: ImageU8 = ImageU8.bw_square_inset_white(N=N)
    # img.plot(ax, 0)
    images = ImageU8.images()
    for i, image in enumerate(images):
        image.plot(ax, i)
    plt.show()
