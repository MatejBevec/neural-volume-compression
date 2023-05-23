import numpy as np
import scipy as sp
import torch
import time

from utils import load_volume, save_volume, show_volume
from models import load_model
from utils import blockify, unblockify, to_standard, to_dtype
from models import DenseVAE



# 3D VOLUME ENCODER-DECODER CLASSES

class Codec():
    """Base class for all compression codecs (i.e. encoder-decoders)."""

    def __init__(self):
        self.blocksize = 8
        self.nblocks = None
        self.m = 0

    def _encode_blocks(self, batch):
        """Should implement encoding a batch of blocks."""

        raise NotImplementedError

    def encodes(self, data):
        """Encode given volume (split into blocks and encode blocks)."""

        t0 = time.time()
        bs = self.blocksize
        vol = to_standard(data)
        self.shape = data.shape
        blocks = blockify(vol, size=bs, margin=self.m)
        self.nblocks = blocks.shape[:3]
        in_batch = blocks.reshape(-1, bs+self.m*2, bs+self.m*2, bs+self.m*2)
        #print("in batch", in_batch.shape)
        enc_batch = self._encode_blocks(in_batch)
        print(f"encoding took {time.time()-t0}s")

        return enc_batch
    
    def encode(self, loadpth, size, dtype=np.uint8, rgb=False):
        """Encode volume from given file path. TODO: other was around"""

        vol = load_volume(loadpth, size, dtype, rgb)
        return self.encodes(vol)

    def _decode_blocks(self, batch):
        """Should implement decoding a batch of blocks."""

        raise NotImplementedError

    def decodes(self, data, size=None, interpol=False):
        """Decode given encoded volume (decode blocks and reconstruct)"""

        bs = self.blocksize
        if size is not None:
            self.nblocks = size/bs
        else: assert self.nblocks is not None

        t0 = time.time()
        dec_batch = self._decode_blocks(data)
        h, w, d = self.nblocks
        dec_blocks = dec_batch.reshape(h, w, d, bs+self.m*2, bs+self.m*2, bs+self.m*2)
        dec_vol = unblockify(dec_blocks, margin=self.m, interpol=interpol)

        if size is None: size = self.shape
        dec_vol = dec_vol[:size[0], :size[1], :size[2]]

        print(f"decoding took {time.time()-t0}s")
        
        return dec_vol
    
    def decode(self, data, savepth, size=None, interpol=False):
        """Decodes volume to file. TODO: other was around"""

        dec_vol = self.decodes(data, size=size, interpol=interpol)
        save_volume(dec_vol, savepth)


class CodecDCT(Codec):
    """Codec based on blockwise discrete fourier transform (TBA)."""

    def __init__(self, blocksize=8, thr=1):
        self.blocksize = blocksize
        self.thr = thr
        self.m = 0
        pass

    def _encode_blocks(self, batch):

        enc_list = []
        for block in batch:
            dct = sp.fftpack.dctn(block)
            dct[np.abs(dct) < self.thr] = 0
            enc_list.append(dct)
        return np.stack(enc_list, axis=0)

    def _decode_blocks(self, batch):
        
        dec_list = []
        for block in batch:
            dec = sp.fftpack.idctn(block)
            dec_list.append(dec)
        return np.stack(dec_list, axis=0)


class CodecDenseVAE(Codec):
    """Codec based on a pretrained blockwise VAE."""

    def __init__(self, size=8, margin=2,
                 model_pth="weights/dense_vae_s8_m2_l64_h256_b64_e8.pt"):
        
        model_pth = model_pth
        self.model = DenseVAE.load_model(model_pth)
        self.blocksize = size
        self.m = margin

    def _encode_blocks(self, batch):
        batch = torch.from_numpy(to_dtype(batch, np.float32))
        mu, var = self.model.encode(batch)
        return mu.detach().numpy()

    def _decode_blocks(self, batch):
        batch = torch.from_numpy(to_dtype(batch, np.float32))
        y = self.model.decode(batch)
        return y.detach().numpy()



if __name__ == "__main__":

    #codec = CodecDCT(blocksize=8, thr=1)
    #codec = CodecDenseVAEOld()
    codec = CodecDenseVAE(size=8, margin=2, model_pth="weights/dense_vae_s8_m2_l64_h256_b64_e8.pt")
    #codec = CodecDenseVAE(size=16, margin=0, model_pth="weights/dense_vae_s16_m0_l32_h256_b64_e10.pt")
    #vol = load_volume("data/tacc_turbulence_256x256x256_1x1x1_uint8.raw", size=256)
    vol = load_volume("data/stag_beetle_832x832x494_1x1x1_uint16.raw", size=(832, 832, 494), dtype=np.uint16)
    #vol = load_volume("data/miranda_512x512x512_1x1x1_uint8.raw", size=512)
    #vol = load_volume("data/porsche_280x512x174_1x1x1_uint8.raw", size=(280, 512, 174))
    #vol = load_volume("data/shockwave_64x64x512_1x1x1_uint8.raw", size=(64, 64, 512))
    #vol = load_volume("data/vismale_128x256x256_1577740x995861x1007970_uint8.raw", size=(128, 256, 256), dtype=np.uint8)
    #vol = load_volume("data/engine_256x256x256_1x1x1_uint8.raw", size=256)
    #vol = load_volume("data/clouds_512x512x32_1x1x1_uint8.raw", size=(512, 512, 32))
    #vol = load_volume("data/daisy_192x180x168_1x1x1_uint8.raw", size=(192, 180, 168))
    #vol = load_volume("data/skull_256x256x256_1x1x1_uint8.raw", size=256)
    #vol = load_volume("data_test/sheep_352x352x256_1x1x1_uint8.raw", size=(352, 352, 256), dtype=np.uint8)
    #vol = load_volume("data_test/bonsai2_lo_512x512x189_50293x50293x125000_uint8.raw", size=(512, 512, 189), dtype=np.uint8)
    
    vol = to_standard(vol)

    show_volume(vol)
    encoded = codec.encodes(vol)
    decoded = codec.decodes(encoded, interpol=False)
    show_volume(decoded)



    # encoded = codec.encode("data/miranda_512x512x512_1x1x1_uint8.raw", size=512)
    # size = 4*len(encoded.flatten())
    # print(size)
    # decoded = codec.decode(encoded, "data/miranda_decoded.raw")
    


    