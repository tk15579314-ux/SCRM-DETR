"""Compatibility helpers for torchvision datapoints/tv_tensors changes."""

from __future__ import annotations

import torchvision
import torchvision.transforms.v2 as T
import torch

torchvision.disable_beta_transforms_warning()

try:
    from torchvision import datapoints
except ImportError:
    from torchvision import tv_tensors as datapoints


if hasattr(datapoints, "BoundingBox"):
    BoundingBox = datapoints.BoundingBox
else:
    BoundingBox = datapoints.BoundingBoxes
BoundingBoxFormat = datapoints.BoundingBoxFormat
Image = datapoints.Image
Video = datapoints.Video
Mask = datapoints.Mask

if hasattr(T, "ToImageTensor"):
    ToImageTensor = T.ToImageTensor
else:
    class ToImageTensor(T.ToImage):
        pass


if hasattr(T, "ConvertDtype"):
    ConvertDtype = T.ConvertDtype
else:
    class ConvertDtype(T.ToDtype):
        def __init__(self, dtype=torch.float32, scale=True):
            super().__init__(dtype=dtype, scale=scale)


if hasattr(T, "SanitizeBoundingBox"):
    SanitizeBoundingBox = T.SanitizeBoundingBox
else:
    class SanitizeBoundingBox(T.SanitizeBoundingBoxes):
        pass


def wrap_bounding_box(data, format, spatial_size):
    """Create a bounding-box tensor on old and new torchvision versions."""
    if hasattr(datapoints, "BoundingBox"):
        return datapoints.BoundingBox(data, format=format, spatial_size=spatial_size)
    return datapoints.BoundingBoxes(data, format=format, canvas_size=spatial_size)


def get_spatial_size(inpt):
    """Return (height, width) for old datapoints or new tv_tensors boxes."""
    if hasattr(inpt, "spatial_size"):
        return inpt.spatial_size
    return inpt.canvas_size
