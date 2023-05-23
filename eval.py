import os
import sys
import time
import datetime
import json
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import pandas as pd

from models import save_model, load_model
from utils import to_dtype, to_standard, show_volume, load_volume

from models import DenseVAE
from dataset import BlockDataset
from codex import CodecDenseVAE


def eval_model(model, dataset):
    """Compute MSE loss for a given autoencoder model on a volume block dataset."""

    bs = 64
    dl = DataLoader(dataset, batch_size=bs, num_workers=0)
    n = len(dataset); nb = int(n/bs)
    loss_func = torch.nn.MSELoss()
    model.eval()
    mean_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i%1000 == 0: print(i)
            y = model(batch)[0]
            loss = loss_func(batch, y)
            mean_loss += loss

    mean_loss /= nb
    return mean_loss


def eval_multiple_models(models, datasets):
    """Compute MSE loss for multiple autoencoder models and datasets."""

    results = {}
    for mname in models:
        model_results = {}
        for dname in datasets:
            print(f"Evaluating {mname} on {dname}...")
            loss = eval_model(models[mname], datasets[dname]).item()
            model_results[dname] = loss
        results[mname] = model_results

    return results



def eval_codec_on_volume(codec, volume):
    """Compute a codec's MSE error on an entire volume."""

    volume = to_dtype(volume, np.float32)
    encoded = codec.encodes(volume)
    decoded = codec.decodes(encoded)
    mse = np.mean((volume - decoded) ** 2)
    print(np.min(volume), np.max(volume), np.min(decoded), np.max(decoded))
    return mse

def eval_codec_on_folder(codec, volumes_dir):
    """Compute a codec's MSE error on all volumes in a directory."""

    df = pd.read_csv(os.path.join(volumes_dir, "index.csv"), sep=",", header=None)

    vols = []
    names = []
    isrgb = lambda x: True if x == 3 else False
    getdtp = lambda x: np.uint16 if x == 16 else np.uint8

    for i, row in df.iterrows():
        r = list(row)
        pth = os.path.join(volumes_dir, r[0])
        rgb = isrgb(r[5])
        dtp = getdtp(r[4])
        vol = load_volume(pth, r[1:4], dtype=dtp, rgb=rgb)
        vol = to_standard(vol)
        vols.append(vol)
        names.append(r[0])

    loss = 0
    for i, vol in enumerate(vols):
        mse = eval_codec_on_volume(codec, vol)
        print(names[i], mse)
        loss += mse

    print("Mean loss: ", loss/len(vols))


def viz_codecs(codecs, volume, wsize=256, zoom=1.0, rot=0, interpol=True):
    """Visualize reconstructions of a given volume with different codecs."""

    n = len(codecs)
    names = list(codecs.keys())
    codecs = list(codecs.values())
    
    for i in range(n):
        plt.subplot(2, n, i+1)
        input_img = show_volume(volume, screenshot=True, wsize=(wsize, wsize), 
                                zoom=zoom, rot=rot, colormap="magma")
        plt.imshow(input_img); plt.title(names[i]); plt.axis("off")
        plt.subplot(2, n, n+i+1)
        encoded = codecs[i].encodes(volume)
        output = codecs[i].decodes(encoded, interpol=interpol)
        output_img = show_volume(output, screenshot=True, wsize=(wsize, wsize), 
                                 zoom=zoom, rot=rot, colormap="magma")
        plt.imshow(output_img); plt.axis("off")
    plt.show()

def viz_codecs2(codecs, volumes, zooms, rots, wsize=256, interpol=True):
    """Visualize reconstructions of a given volume with different codecs."""

    n = len(codecs)
    names = list(codecs.keys())
    codecs = list(codecs.values())
    
    for j in range(len(volumes)):
        plt.subplot(len(volumes), n+1, j*(n+1) + 1)
        volume = volumes[j]
        input_img = show_volume(volume, screenshot=True, wsize=(wsize, wsize), 
                                zoom=zooms[j], rot=rots[j], colormap="magma")
        plt.imshow(input_img); plt.axis("off")

        for i in range(n):
            plt.subplot(len(volumes), n+1, j*(n+1) + i + 2)
            encoded = codecs[i].encodes(volume)
            output = codecs[i].decodes(encoded, interpol=interpol)
            output_img = show_volume(output, screenshot=True, wsize=(wsize, wsize), 
                                    zoom=zooms[j], rot=rots[j], colormap="magma")
            plt.imshow(output_img); plt.axis("off")
            if j == 0: plt.title(names[i])

    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":


    # EVAL MODELS BLOCK-WISE

    # models = {}
    # for fn in os.listdir("weights"):
    #     model = DenseVAE.load_model(os.path.join("weights", fn))
    #     models[fn.rsplit(".")[0]] = model

    # datasets = {
    #     "train": BlockDataset("data", blocksize=8, margin=0, save_load=False, split="train"),
    #     "test": BlockDataset("data", blocksize=8, margin=0, save_load=False, split="test")
    # }

    # results = eval_multiple_models(models, datasets)
    # print(results)
    # with open("results.json", "w") as f:
    #     json.dump(results, f)


    # EVAL MODELS VOLUME-WISE

    codecs = {
        "zero margin": CodecDenseVAE(size=8, margin=0, model_pth="weights/dense_vae_s8_m0_l32_h128_b64_e8.pt"),
        "two-wide margin": CodecDenseVAE(size=8, margin=2, model_pth="weights/dense_vae_s8_m2_l32_h128_b64_e8.pt"),
        "more parameters": CodecDenseVAE(size=8, margin=2, model_pth="weights/dense_vae_s8_m2_l64_h256_b64_e8.pt")
    }

    vols = [
        #to_standard(load_volume("data/daisy_192x180x168_1x1x1_uint8.raw", size=(192, 180, 168))),
        #to_standard(load_volume("data/clouds_512x512x32_1x1x1_uint8.raw", size=(512, 512, 32))),
        #to_standard(load_volume("data/tacc_turbulence_256x256x256_1x1x1_uint8.raw", size=256)),
        #to_standard(load_volume("data/porsche_280x512x174_1x1x1_uint8.raw", size=(280, 512, 174))),
        #to_standard(load_volume("data/miranda_512x512x512_1x1x1_uint8.raw", size=512)),
        #to_standard(load_volume("data/stag_beetle_832x832x494_1x1x1_uint16.raw", size=(832, 832, 494), dtype=np.uint16))
    ]

    #viz_codecs2(codecs, vols, [1, 1], [0, 0], interpol=True)
    #viz_codecs2(codecs, vols, [2, 2.1, 1, 1.2], [0, 110, 0, 85], interpol=True)
    #viz_codecs2(codecs, vols, [4, 5, 6, 8], [0, 110, 10, 85], interpol=True)

    eval_codec_on_folder(codecs["two-wide margin"], "data")
