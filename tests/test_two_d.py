import matplotlib.pyplot as plt
import PIL.Image as Image
import math
import cmath
import time
import csv

import numpy as np
from numpy import binary_repr

import pytest

from two_d import (return_true, ImageU8, SqMatC32,
                   )


def test_test_itself():
    assert return_true()


def test_64_u8_img():
    fig, ax = plt.subplots(1, 5, figsize=(17, 17))
    images = ImageU8.images()
    for i, image in enumerate(images):
        image.plot(ax, i)
    plt.show()


def test_conjugate():
    a = SqMatC32(
        np.array([[1+1j, 2], [4, 5]], dtype=np.complexfloating))
    b = a.conjugate()
    result = np.all(np.equal(b.data, [[1.-1.j, 2], [4, 5]]))
    orig_not_changed = np.all(np.equal(a.data, [[1+1j, 2], [4, 5]]))
    assert result and orig_not_changed


def test_mul():
    a = SqMatC32(
        np.array([[1+1j, 2], [4, 5]], dtype=np.complexfloating))
    b = a.conjugate()
    result = np.all(np.equal(a.mul(b).data, [[2, 4], [16, 25]]))
    orig_not_changed = np.all(np.equal(a.data, [[1+1j, 2], [4, 5]]))
    assert result and orig_not_changed


def test_dft():
    fig, ax = plt.subplots(4, 5, figsize=(17, 17))
    images = ImageU8.images()  # [3:]  # the others are too slow
    cplxs = [i.to_Complex() for i in images]
    dfts = [c.dft().chop() for c in cplxs]
    idfts = [d.dft(inverse=True).chop() for d in dfts]
    for i, img in enumerate(images):
        img.plot(ax, (0, i))
        cplxs[i].plot(ax, (1, i), 'Original as complex')
        dfts[i].plot(ax, (2, i), 'DFT')
        idfts[i].plot(ax, (3, i), 'InvDFT(DFT)')
    plt.show()

def test_cast():
    temp = ImageU8.bw_square_inset_white(N=64)
    tmp1 = temp.to_Complex()
    pass