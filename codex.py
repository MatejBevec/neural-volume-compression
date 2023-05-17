import numpy as np
import scipy as sp
import torch

from utils import load_volume, save_volume, show_volume
from models import load_model
from utils import blockify, unblockify, to_standard, to_dtype
from models import DenseVAE


# 3D VOLUME ENCODER-DECODER CLASSES

class Codec():

    def __init__(self):
        self.blocksize = 8
        self.nblocks = None
        self.m = 0

    def _encode_blocks(self, batch):
        raise NotImplementedError

    def encodes(self, data):
        bs = self.blocksize
        vol = to_standard(data)
        blocks = blockify(vol, size=bs, margin=self.m)
        self.nblocks = blocks.shape[:3]
        in_batch = blocks.reshape(-1, bs+self.m*2, bs+self.m*2, bs+self.m*2)
        print("in batch", in_batch.shape)
        enc_batch = self._encode_blocks(in_batch)

        return enc_batch
    
    def encode(self, loadpth, size, dtype=np.uint8, rgb=False):
        vol = load_volume(loadpth, size, dtype, rgb)
        return self.encodes(vol)

    def _decode_blocks(self, batch):
        raise NotImplementedError

    def decodes(self, data, size=None, interpol=False):
        bs = self.blocksize
        if size is not None:
            self.nblocks = size/bs
        else: assert self.nblocks is not None

        dec_batch = self._decode_blocks(data)
        h, w, d = self.nblocks
        dec_blocks = dec_batch.reshape(h, w, d, bs+self.m*2, bs+self.m*2, bs+self.m*2)
        dec_vol = unblockify(dec_blocks, margin=self.m, interpol=interpol)
        
        return dec_vol
    
    def decode(self, data, savepth, size=None, interpol=False):
        dec_vol = self.decodes(data, size=size, interpol=interpol)
        save_volume(dec_vol, savepth)


class CodecDCT(Codec):

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


class CodecDenseVAEOld(Codec):

    def __init__(self):
        
        model_pth = "weights/dense_vae_d.pt"
        b = 8
        self.blocksize = b
        latents = 64
        hidden = [256]
        self.m = 1
        
        self.model = DenseVAE([b+self.m*2, b+self.m*2, b+self.m*2], latent_dim=latents, hidden_dims=hidden)
        load_model(self.model, model_pth)

    def _encode_blocks(self, batch):
        batch = torch.from_numpy(to_dtype(batch, np.float32))
        mu, var = self.model.encode(batch)
        return mu.detach().numpy()

    def _decode_blocks(self, batch):
        batch = torch.from_numpy(to_dtype(batch, np.float32))
        y = self.model.decode(batch)
        return y.detach().numpy()
    

class CodecDenseVAE(Codec):

    def __init__(self, size=8, margin=0,
                 model_pth="weights/dense_vae_s8_m0_l32_h128_b64_e1.pt"):
        
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
    codec = CodecDenseVAE()
    #codec = CodecDenseVAE(size=16, margin=0, model_pth="weights/dense_vae_s16_m0_l32_h256_b64_e10.pt")
    #vol = load_volume("data/tacc_turbulence_256x256x256_1x1x1_uint8.raw", size=256)
    #vol = load_volume("data/stag_beetle_832x832x494_1x1x1_uint16.raw", size=(832, 832, 494), dtype=np.uint16)
    vol = load_volume("data/miranda_512x512x512_1x1x1_uint8.raw", size=512)
    #vol = load_volume("data/porsche_280x512x174_1x1x1_uint8.raw", size=(280, 512, 174))
    #vol = load_volume("data/shockwave_64x64x512_1x1x1_uint8.raw", size=(64, 64, 512))
    #vol = load_volume("data/vismale_128x256x256_1577740x995861x1007970_uint8.raw", size=(128, 256, 256), dtype=np.uint8)
    vol = to_standard(vol)

    scr = show_volume(vol[100:116, 100:116, 100:116], screenshot=False, wsize=[128, 128])
    print(scr.shape)
    #encoded = codec.encodes(vol)
    #decoded = codec.decodes(encoded, interpol=False)
    #show_volume(decoded)



    # encoded = codec.encode("data/miranda_512x512x512_1x1x1_uint8.raw", size=512)
    # size = 4*len(encoded.flatten())
    # print(size)
    # decoded = codec.decode(encoded, "data/miranda_decoded.raw")
    


    