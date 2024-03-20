import matplotlib.pyplot as plt
import PIL.Image as Image
import math
import cmath
import time
import csv

import numpy as np
from numpy import binary_repr

import pytest

import two_d


def test_test_itself():
    assert two_d.return_true()


def test_64_u8_img():
    fig, ax = plt.subplots(1, 5, figsize=(17, 17))
    N = 64
    img: two_d.ImageU8 = two_d.ImageU8.bw_square(N=N)
    img.plot(ax, 0)
    plt.show()
