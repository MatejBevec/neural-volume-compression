import os 
import sys
import time
import struct
import math

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import k3d
import pyvista as pv
import raster_geometry as rg


# UTILS


def load_volume(pth, size, dtype=np.uint8, rgb=False):
    "Load .raw file into a volume array."

    with open(pth, "rb") as f:
        raw_data = f.read()

    size = [size, size, size] if isinstance(size, int) else size
    size = size[::-1]
    data = np.frombuffer(raw_data, dtype=dtype)
    if rgb: size += tuple([3])
    data = data.reshape(size).astype(dtype)

    return data
    

def save_volume(data, pth, dtype=np.uint8, rgb=False):
    """Save grid values representing volume to file."""

    if len(data.shape) > 1:
        data = data.flatten(order="F")
    #data = data.astype(dtype)
    data = to_dtype(data, dtype)

    with open(pth, "wb") as f:
        f.write(data)


def show_volume(data, origin=(0, 0, 0), spacing=(1, 1, 1),
                colormap="hot", screenshot=False, wsize=None, zoom=1.0, rot=0):
    """Visualize a 3D volume using PyVista."""

    grid = pv.UniformGrid()
    grid.dimensions = data.shape
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data["data"] = data.flatten(order="F")
    plotter = pv.Plotter(off_screen=screenshot)
    clim = [0, 1] if screenshot else None
    if wsize:
        plotter.window_size = wsize
    plotter.set_background("white")
    plotter.add_volume(grid, cmap=colormap, show_scalar_bar=not screenshot, clim=clim)
    plotter.camera.zoom(zoom)
    plotter.camera.azimuth = rot
    plotter.show()
    if screenshot:
        return np.array(plotter.screenshot())
    


def load_compressed(pth):
    pass

def save_compressed(data, pth):
    pass


def blockify(data, size=16, margin=0):
    """Split a volume into blocks for compression.
        Returns 3D array of size*size*size blocks."""
    
    # TODO: NO LOOPS
    # TODO: consider padding by half on both sides, so model gets better distribution

    h, w, d = data.shape[:3]
    yblocks = math.ceil(h / size)
    xblocks = math.ceil(w / size)
    zblocks= math.ceil(d / size)

    ypad = yblocks*size - h
    xpad = xblocks*size - w
    zpad = zblocks*size - d
    padded = np.pad(data, ((0, ypad), (0, xpad), (0, zpad)))
    m = margin
    padded = np.pad(padded, ((m, m), (m, m), (m, m)), constant_values=0)

    blocks = np.ndarray((yblocks, xblocks, zblocks, size+2*m, size+2*m, size+2*m))
    blocks = blocks.astype(data.dtype)
    for i,y in enumerate(np.arange(m, h, size)):
        for j,x in enumerate(np.arange(m, w, size)):
            for k,z in enumerate(np.arange(m, d, size)):
                blocks[i, j, k, :, :, :] = padded[y-m:y+size+m, x-m:x+size+m, z-m:z+size+m]

    return blocks


def _smooth_kernel2(s, m):

    sz = s+2*m
    ker = np.zeros((s*3+2*m, s*3+2*m, s*3+2*m))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ones = np.ones((sz, sz, sz))
                ker[i*s:(i+1)*s+2*m, j*s:(j+1)*s+2*m, k*s:(k+1)*s+2*m] += ones
    ker = ker[s:2*s+2*m, s:2*s+2*m, s:2*s+2*m]
    ker = 1/ker
    return ker


def unblockify(blocks, margin=0, interpol=False):
    """Combine blocks back into a full volume."""

    m = margin
    size = blocks.shape[3] - 2*m
    ybl, xbl, zbl = blocks.shape[0:3]
    
    vol = np.ndarray((ybl*size+2*m, xbl*size+2*m, zbl*size+2*m))
    #vol = vol.astype(np.float32)

    if interpol:
        ker = _smooth_kernel2(size, m)
        #print(ker)
    else:
        ker = np.zeros((size+2*m, size+2*m, size+2*m))
        ker[m:size+m, m:size+m, m:size+m] = 1

    #print(blocks.shape)
    #print(vol.shape)

    for i in range(ybl):
        for j in range(xbl):
            for k in range(zbl):
                # vol[i*size:(i+1)*size, j*size:(j+1)*size, k*size:(k+1)*size] \
                #     = blocks[i, j, k, m:size+m, m:size+m, m:size+m]
                #print(blocks[i, j, k, :, :, 4])
                block = (blocks[i, j, k, :, :, :] * ker)
                #print(block[:, :, 4])
                vol[i*size : (i+1)*size+2*m, j*size : (j+1)*size+2*m, k*size : (k+1)*size+2*m] += block
    vol = vol[m:vol.shape[0]-m, m:vol.shape[1]-m, m:vol.shape[2]-m].astype(blocks.dtype)
    return vol


def to_mono(data):
    """Convert volume to single-channel if in RGB."""

    if (len(data.shape) > 3 and data.shape[-1] == 3):
        dtype = data.dtype
        data = np.mean(data, axis=-1).astype(dtype)
    if (data.shape[-1] == 1):
        data = np.squeeze(data)
    return data

def to_dtype(data, dtype=np.uint8):
    """Convert volume to given dtype, assuming full range of vals is used."""

    from_max = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    to_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

    ratio = to_max / from_max
    return (data * ratio).astype(dtype)

def to_standard(data):
    """Shorthand to convert to mono uint8 for training."""

    return to_dtype(to_mono(data), dtype=np.uint8)


if __name__ == "__main__":


    # vol = load_volume("data/virgo_256x256x256_1x1x1_uint8x3.raw", (256, 256, 256), rgb=True, dtype=np.uint8)
    # vol = load_volume("data/stag_beetle_832x832x494_1x1x1_uint16.raw", (832, 832, 494), dtype=np.uint16)
    vol = load_volume("data/tacc_turbulence_256x256x256_1x1x1_uint8.raw", (256, 256, 256), dtype=np.uint8)

    blocks = blockify(vol, size=8, margin=1)

    rec = unblockify(blocks, margin=1, interpol=True)
    show_volume(rec)

    #ker = _smooth_kernel2(8, 1)



    # dct = sp.fftpack.dctn(vol)
    # print(dct)
    # print(dct.shape)
    # #dct[np.abs(dct) < 1000] = 0
    # rec = sp.fftpack.idctn(dct)

    # show_volume(dct)
    # show_volume(rec)

    
