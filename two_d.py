"""motivated by https://github.com/rebcabin/Digital-Image-Processing"""
from unittest import result

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


def is_power_of_two(n: int) -> bool:
    return n & (n - 1) == 0


def divides(small: int, big: int) -> bool:
    assert small <= big
    # e.g., 12 % 4
    result: bool = big % small == 0
    return result


@dataclass
class SqMat:
    data: np.ndarray

    def is_square(self) -> bool:
        result = self.data.shape[0] == self.data.shape[1]
        return result

    def is_p2(self) -> bool:
        result = is_power_of_two(self.data.shape[0])
        return result

    def H(self) -> int:
        result: int = self.data.shape[0]
        return result

    def W(self) -> int:
        result: int = self.data.shape[1]
        return result

    def N(self) -> int:
        assert self.is_square()
        return self.H()

    def is_square_power_of_two(self) -> bool:
        return self.is_square() and self.is_p2()

    def plot_abs(self, ax, idx, title: str) -> None:
        raise NotImplementedError


@dataclass
class SqMatC32(SqMat):

    @staticmethod
    def zeros(N: int) -> "SqMatC32":
        result = SqMatC32(np.zeros((N, N),
                                   dtype=np.complexfloating))
        return result

    def conjugate(self):
        temp = np.conj(self.data)
        result = SqMatC32(temp)
        return result

    def elementwise_mul(self, other: "SqMatC32") -> "SqMatC32":
        """Element-wise multiplication (Hadamard product)."""
        result = SqMatC32(self.data * other.data)
        return result

    def fft(self) -> "SqMatC32":
        """Why the shifts?"""
        fft = np.fft.ifftshift(self.data)
        fft = np.fft.fft2(fft)
        fft = np.fft.fftshift(fft)
        result = SqMatC32(fft)
        return result

    def ifft(self) -> "SqMatC32":
        """Why the shifts?"""
        ifft = np.fft.ifftshift(self.data)
        ifft = np.fft.ifft2(ifft)
        ifft = np.fft.fftshift(ifft)
        result = SqMatC32(ifft)
        return result

    def convolve_brute_force(self, kernel: "SqMatC32") -> "SqMatC32":
        # TODO: tested only for 3x3 for now.
        # ML-style -- reduces the size of self by K // 2 on
        # top, bottom, left, and right.
        N = self.N()
        K = kernel.N()
        assert K == 3
        R = K // 2
        output = SqMatC32.zeros(N)
        for h in range(R, self.H() - R):
            for w in range(R, self.W() - R):
                acc = 0. + 0.j
                for hk in range(kernel.H()):
                    for wk in range(kernel.W()):
                        acc += kernel.data[hk, wk] \
                            * self.data[h + hk - R, w + wk - R]
                output.data[h, w] = acc
        return output

    def convolution_theorem(self, kernel: "SqMatC32") -> "SqMatC32":

        def pad_kernel(kernel: "SqMatC32") -> "SqMatC32":
            """TODO Tested only for 3x3 kernel."""
            N = self.N()
            K = kernel.N()
            # If N == 32, the half-way index is N // 2 == 16.
            # That index begins the second half of the array.
            # An odd-order kernel should stride that index.
            #
            # Looks like the starting index for the centered
            # boxcar is (N // 2) - (K // 2).
            #
            #   0         1         2         3
            #   012345678901234567890123456789012
            #                   V
            #  .---=---=---=---=---=---=---=---=.
            #  |              .-=-.             |
            #  | <----15----> |K=3| <--- 14 --> |
            #  |              `-=-'             |
            # ...                              ...
            #  |                                |
            #  |                                |
            #  `--------------------------------'
            temp = SqMatC32.zeros(N)
            L = (N // 2) - (K // 2)
            temp.data[L: L + K, L: L + K] = kernel.data
            return temp

        temp = pad_kernel(kernel)
        kfft = temp.fft()
        sorc = self.fft()
        temp2 = kfft.elementwise_mul(sorc)
        result = temp2.ifft()
        return result

    def dft(self, inverse: bool = False) -> "SqMatC32":
        """TODO: broken
        https://en.wikipedia.org/wiki/Discrete_Fourier_transform
        O(n^2), slow."""
        assert self.is_square() and self.is_p2()
        N: int = self.data.shape[0]
        it = np.fft.ifftshift(self.data)
        if not inverse:
            it = np.fft.fft2(it)
        else:
            it = np.fft.ifft2(it)
        it = np.fft.fftshift(it)
        result = SqMatC32(it)
        # result = SqMatC32(np.zeros((N, N), dtype=np.complexfloating))
        # for ky in range(N):
        #     for kx in range(N):
        #         acc = 0. + 0.j
        #         for ny in range(N):
        #             for nx in range(N):
        #                 nn = np.array([nx / N, ny / N], dtype=np.complexfloating)
        #                 kk = np.array([kx, ky], dtype=np.complexfloating)
        #                 f = (0 + 1j) if inverse else (0 - 1j)
        #                 e = f * (2 * np.pi * np.dot(nn, kk))
        #                 v = np.exp(e, dtype=np.complexfloating)
        #                 acc += (v / (N * N)) if inverse else v
        #         result.data[kx, ky] = acc
        return result

    def chop(self, epsilon=1.e-10) -> "SqMatC32":
        assert self.is_square() and self.is_p2()
        moduli = np.abs(self.data)
        N: int = self.data.shape[0]
        temp = SqMatC32.zeros(N)
        for ky in range(N):
            for kx in range(N):
                temp.data[kx, ky] = \
                    (0. + 0.j) if moduli[kx, ky] < epsilon else moduli[kx, kx]
        return temp

    def plot_abs(self, ax, idx, title='modulus') -> None:
        it = np.abs(self.data)
        it = np.log(1 + it)
        vmin = np.min(it)
        vmax = np.max(it)
        ax[idx].imshow(it, cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].set_title(title)

    def plot_arg(self, ax, idx, title='argument') -> None:
        it = np.angle(self.data)
        vmin = np.min(it)
        vmax = np.max(it)
        ax[idx].imshow(it, cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].set_title(title)


@dataclass
class ImageU8(SqMat):

    def to_Complex(self):
        assert (np.can_cast(self.data, np.complexfloating))
        result = SqMatC32(self.data.astype(np.complexfloating))
        return result

    @staticmethod
    def black_square(N: int) -> "ImageU8":
        """factory"""
        it = ImageU8(ImageU8._black_square_nda(N))
        assert it.is_square_power_of_two()
        return it

    @staticmethod
    def _white_square_nda(N: int) -> np.ndarray:
        result = np.iinfo(np.uint8).max \
                 * np.ones((N, N), dtype=np.uint8)
        return result

    @staticmethod
    def _black_square_nda(N: int) -> np.ndarray:
        result = np.iinfo(np.uint8).min \
                 * np.zeros((N, N), dtype=np.uint8)
        return result

    @staticmethod
    def bw_square_inset_white(N: int) -> "ImageU8":
        """factory"""
        assert (N % 4 == 0)
        it = ImageU8.black_square(N)
        # Start a quarter of the way from top-left (tl).
        tl: int = N // 4  # e.g. 8 for N=32
        # End three quarters of the way from bottom-right (br).
        br: int = tl + (N // 2)  # 8 + 16 = 24 for N=32
        N2 = N // 2
        it.data[tl:br, tl:br] = ImageU8._white_square_nda(N2)
        return it

    def bw_square_resized_white_portions(self) -> list["ImageU8"]:
        """factory"""
        assert self.is_square_power_of_two()
        result = []
        N = self.N()  # e.g., 32
        side = N // 2  # 16, 8, 4
        while side >= 4:
            side = side // 2  # 8,  4,  2
            j = (N - side) // 2  # 4,  6,  7  (16-8)//2=4, (16-4)//2=6, (16-2)//2=7
            k = j + side  # 12, 10,  9  4+8=12, 6+4=10, 7+2=9
            # The sum is always 16.
            temp = ImageU8.black_square(N)
            temp.data[j:k, j:k] = ImageU8._white_square_nda(side)
            result.append(temp)
        return result

    def avg_pool(self, new_size: int) -> "ImageU8":
        assert self.is_square_power_of_two()
        N = self.N()
        assert divides(new_size, N)
        step = N // new_size
        result = ImageU8.black_square(new_size)
        for h in range(result.H()):
            for w in range(result.W()):
                result.data[h, w] = \
                    np.mean(self.data[
                            h * step: (h + 1) * step,
                            w * step: (w + 1) * step])
        return result

    @staticmethod
    def images(sizes: tuple[int] = (128, 64, 32, 16, 8)) \
            -> list["ImageU8"]:
        result = []
        temp = ImageU8.bw_square_inset_white(sizes[0])
        result.append(temp)
        for size in sizes[1:]:
            result.append(temp.avg_pool(size))
        return result

    def plot_abs(self, ax, idx, title='Original image') -> None:
        vmin = self.data.min()
        vmax = self.data.max()
        ax[idx].imshow(self.data, cmap='gray', vmin=vmin, vmax=vmax)
        ax[idx].set_title(title)
