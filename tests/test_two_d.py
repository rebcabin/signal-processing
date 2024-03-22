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


@pytest.mark.skip(reason="unneeded")
def test_64_u8_img():
    fig, ax = plt.subplots(1, 5, figsize=(17, 17))
    images = ImageU8.images()
    for i, image in enumerate(images):
        image.plot_abs(ax, i)
    plt.show()


def test_conjugate():
    a = SqMatC32(
        np.array([[1 + 1j, 2], [4, 5]], dtype=np.complexfloating))
    b = a.conjugate()
    result = np.all(np.equal(b.data, [[1. - 1.j, 2], [4, 5]]))
    orig_not_changed = np.all(np.equal(a.data, [[1 + 1j, 2], [4, 5]]))
    assert result and orig_not_changed


def test_mul():
    a = SqMatC32(
        np.array([[1 + 1j, 2], [4, 5]], dtype=np.complexfloating))
    b = a.conjugate()
    result = np.all(np.equal(a.elementwise_mul(b).data, [[2, 4], [16, 25]]))
    orig_not_changed = np.all(np.equal(a.data, [[1 + 1j, 2], [4, 5]]))
    assert result and orig_not_changed


@pytest.mark.skip(reason="known bugs")
def test_dft():
    fig, ax = plt.subplots(4, 5, figsize=(17, 17))
    images = ImageU8.images()  # [3:]  # the others are too slow
    cplxs = [i.to_Complex() for i in images]
    dfts = [c.dft().chop() for c in cplxs]
    idfts = [d.dft(inverse=True).chop() for d in dfts]
    for i, img in enumerate(images):
        img.plot_abs(ax, (0, i))
        cplxs[i].plot_abs(ax, (1, i), '|complex img|')
        dfts[i].plot_abs(ax, (2, i), '|DFT|')
        idfts[i].plot_abs(ax, (3, i), '|InvDFT(DFT)|')
    plt.show()


def test_fft2():
    fig, ax = plt.subplots(5, 5, figsize=(17, 17))
    images = ImageU8.images()  # [3:]  # the others are too slow
    cplxs = [i.to_Complex() for i in images]
    ffts = [c.fft() for c in cplxs]
    iffts = [f.ifft().chop() for f in ffts]
    for i, img in enumerate(images):
        cplxs[i].plot_abs(ax, (0, i), '|complex img|')
        ffts[i].plot_abs(ax, (1, i), '|DFT|')
        ffts[i].plot_arg(ax, (2, i), '/_DFT')
        iffts[i].plot_abs(ax, (3, i), '|IDFT(DFT)|')
        iffts[i].plot_arg(ax, (4, i), '/_IDFT(DFT)')
    plt.show()


# https://setosa.io/ev/image-kernels/

kernel_emboss = \
    SqMatC32(np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]],
        dtype=np.complexfloating))

kernel_blur = \
    SqMatC32(np.array([
        [1 / 16., 1 / 8., 1 / 16.],
        [1 / 8., 1 / 4., 1 / 8.],
        [1 / 16., 1 / 8., 1 / 16.]],
        dtype=np.complexfloating))

kernel_wipe = \
    SqMatC32(np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        dtype=np.complexfloating))


def test_convolution():
    images = ImageU8.images()
    cplxs = list(reversed([i.to_Complex() for i in images]))
    convos = [c.convolve_brute_force(kernel_emboss) for c in cplxs
              for c in cplxs]
    thrms = [c.convolution_theorem(kernel_emboss) for c in cplxs]
    fig, ax = plt.subplots(5, 5, figsize=(17, 17))
    for i, img in enumerate(images):
        cplxs[i].plot_abs(ax, (0, i), '|complex img|')
        convos[i].plot_abs(ax, (1, i), '|convo emboss|')
        convos[i].chop().plot_arg(ax, (2, i), '/_convo emboss')
        thrms[i].plot_abs(ax, (3, i), '|theorem emboss|')
        thrms[i].chop().plot_arg(ax, (4, i), '/_theorem emboss')
    plt.show()


def test_cast():
    temp = ImageU8.bw_square_inset_white(N=64)
    tmp1 = temp.to_Complex()
    pass
