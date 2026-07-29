"""
conv.py — 2D Convolutional layer via im2col.

Conv2D is implemented using the im2col algorithm, which transforms the
convolution operation into a single matrix multiplication. This is exactly
how cuDNN and PyTorch implement it internally for efficiency.

im2col  : Unrolls each receptive field into a column — (N,C,H,W) -> (N, C*kH*kW, out_H*out_W)
col2im  : The reverse operation for the backward pass
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


def im2col(x: np.ndarray, kernel_h: int, kernel_w: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    im2col: Convert image patches to columns for efficient convolution.

    Input:  x shape (N, C, H, W)
    Output: cols shape (C * kH * kW, N * out_H * out_W)

    Each column represents one receptive field patch, flattened.
    This lets us compute convolution as a single matrix multiply.
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # Pad the input
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    else:
        x_padded = x

    # Extract sliding windows using stride tricks
    shape = (N, C, kernel_h, kernel_w, out_h, out_w)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2] * stride,
        x_padded.strides[3] * stride,
        x_padded.strides[2],
        x_padded.strides[3],
    )

    # Use numpy stride tricks to create a view — no memory copy
    windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    # Reshape to (N, C*kH*kW, out_H*out_W)
    cols = windows.reshape(N, C * kernel_h * kernel_w, out_h * out_w)

    return cols


def col2im(cols: np.ndarray, x_shape: Tuple[int, ...], kernel_h: int, kernel_w: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    col2im: Reverse of im2col. Scatter column gradients back to image positions.

    Input:  cols shape (N, C*kH*kW, out_H*out_W)
    Output: dx shape (N, C, H, W)

    Uses vectorized advanced indexing with np.add.at instead of nested Python
    loops over kernel dimensions, avoiding O(kH*kW) Python-level iterations.
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # Reshape columns back to window view: (N, C, kH, kW, out_h, out_w)
    windows = cols.reshape(N, C, kernel_h, kernel_w, out_h, out_w)

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    dx_padded = np.zeros((N, C, H_padded, W_padded), dtype=np.float64)

    # Build vectorized index arrays for all kernel positions at once
    # ky_idx, kx_idx: kernel offsets (kH, kW)
    ky_idx, kx_idx = np.mgrid[0:kernel_h, 0:kernel_w]  # each (kH, kW)
    # oh_idx, ow_idx: output spatial positions (out_h, out_w)
    oh_idx, ow_idx = np.mgrid[0:out_h, 0:out_w]         # each (out_h, out_w)

    # row_idx[ky, kx, oh, ow] = ky + oh * stride
    row_idx = ky_idx[:, :, None, None] + oh_idx[None, None, :, :] * stride  # (kH, kW, out_h, out_w)
    # col_idx[ky, kx, oh, ow] = kx + ow * stride
    col_idx = kx_idx[:, :, None, None] + ow_idx[None, None, :, :] * stride  # (kH, kW, out_h, out_w)

    # Scatter-add using np.add.at to handle overlapping indices correctly
    # windows shape: (N, C, kH, kW, out_h, out_w) — matches index dims on axes 2,3,4,5
    np.add.at(dx_padded, (slice(None), slice(None), row_idx, col_idx), windows)

    # Remove padding if present
    if padding > 0:
        return dx_padded[:, :, padding:-padding, padding:-padding]
    return dx_padded


class Conv2D(Module):
    """
    2D Convolutional layer.

    Args:
        in_channels:  Number of channels in the input
        out_channels: Number of channels produced by the convolution
        kernel_size:  Size of the convolving kernel (int or tuple)
        stride:       Stride of the convolution (default: 1)
        padding:      Zero-padding added to both sides (default: 0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.kernel_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.stride = stride
        self.padding = padding

        # Kaiming He initialization for convolutional layers
        # std = sqrt(2 / (in_channels * kH * kW))
        fan_in = in_channels * self.kernel_h * self.kernel_w
        scale = np.sqrt(2.0 / fan_in)

        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, self.kernel_h, self.kernel_w) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using im2col.

        Args:
            x: Input tensor of shape (N, C, H, W)
        Returns:
            Output tensor of shape (N, out_channels, out_H, out_W)
        """
        N, C, H, W = x.data.shape
        out_ch = self.out_channels

        # Compute output spatial dimensions
        out_h = (H + 2 * self.padding - self.kernel_h) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_w) // self.stride + 1

        # im2col: (N, C, H, W) -> (N, C*kH*kW, out_H*out_W)
        cols = im2col(x.data, self.kernel_h, self.kernel_w, self.stride, self.padding)

        # Reshape weight: (out_ch, in_ch, kH, kW) -> (out_ch, in_ch*kH*kW)
        w_col = self.weight.data.reshape(out_ch, -1)

        # Matrix multiplication: (out_ch, in_ch*kH*kW) @ (N, in_ch*kH*kW, out_H*out_W)
        # We need to handle batch dimension: do it per sample or batch-matmul trick
        # (out_ch, C*kH*kW) @ (N, C*kH*kW, out_H*out_W) -> we want (N, out_ch, out_H*out_W)
        out = np.zeros((N, out_ch, out_h * out_w))
        for i in range(N):
            out[i] = w_col @ cols[i]  # (out_ch, C*kH*kW) @ (C*kH*kW, out_H*out_W)

        # Add bias and reshape
        out = out + self.bias.data.reshape(1, -1, 1)  # (N, out_ch, out_H*out_W)
        out = out.reshape(N, out_ch, out_h, out_w)

        requires_grad = x.requires_grad or self.weight.requires_grad or self.bias.requires_grad
        result = Tensor(out, requires_grad=requires_grad, _children=(x, self.weight, self.bias), _op="conv2d")

        def _backward() -> None:
            # dout shape: (N, out_ch, out_h, out_w)
            dout = result.grad.reshape(N, out_ch, out_h * out_w)

            # Gradient w.r.t. bias: sum over N, spatial
            if self.bias.requires_grad:
                self.bias.grad += dout.sum(axis=(0, 2))

            # Gradient w.r.t. weight
            if self.weight.requires_grad:
                # (out_ch, N*out_H*out_W) @ (N*out_H*out_W, C*kH*kW) -> (out_ch, C*kH*kW)
                dw = np.zeros((out_ch, self.in_channels * self.kernel_h * self.kernel_w))
                for i in range(N):
                    dw += dout[i] @ cols[i].T  # (out_ch, out_H*out_W) @ (out_H*out_W, C*kH*kW)
                self.weight.grad += dw.reshape(out_ch, self.in_channels, self.kernel_h, self.kernel_w)

            # Gradient w.r.t. input
            if x.requires_grad:
                # dout: (N, out_ch, out_H*out_W)
                # w_col: (out_ch, C*kH*kW)
                # dx_col: (N, C*kH*kW, out_H*out_W)
                dx_col = np.zeros((N, self.in_channels * self.kernel_h * self.kernel_w, out_h * out_w))
                for i in range(N):
                    dx_col[i] = w_col.T @ dout[i]  # (C*kH*kW, out_ch) @ (out_ch, out_H*out_W)

                # col2im to get back image shape
                x.grad += col2im(dx_col, (N, C, H, W), self.kernel_h, self.kernel_w, self.stride, self.padding)

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.in_channels}, {self.out_channels}, "
            f"kernel_size=({self.kernel_h}, {self.kernel_w}), "
            f"stride={self.stride}, padding={self.padding})"
        )
