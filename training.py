import os
import sys
import time
import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from models import save_model, load_model
from utils import to_dtype, to_standard, show_volume

def train_model(dataset, model,
                optimizer="adam", loss="mse",
                lr=2e-4, epochs=3, batch_size=32,
                scheduler=None, log=True, run_name=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    n = len(dataset); nb = int(n/batch_size)
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    if log:
        if run_name is None: run_name = datetime.datetime.now()
        wandb.config = {"learning_rate": lr, "epochs": epochs, "batch_size": batch_size}
        wandb.init(project='volume-compression', name=run_name)
        wandb.watch(model, log="all", log_freq=10, log_graph=True)

    for ep in range(epochs):
        
        pbar = tqdm(total=nb)
        pbar.update(1)
        ep_loss = []
        last_batch = None

        for i, batch in enumerate(dl):
            #[print(torch.min(a), torch.max(a)) for a in batch]
            batch = batch.to(device)
            loss = train_step(batch, model, optimizer, loss_func)
            if i%20 == 0:
                pbar.update(20); pbar.set_description(f"Loss = {loss:.8f}")
            ep_loss.append(loss.detach().item())

            if log:
                last_batch = batch
                wandb.log({'Train Loss': loss})
        
        if log:
            #wandb.log({'End of Epoch': None}, step=((ep+1) * len(dl)))
            wsize = 150; n = 16
            outs = model(last_batch)[0].cpu().detach().numpy()
            ins = last_batch.cpu().detach().numpy()

            ins = ins[:n]; outs = outs[:n]
            ins[:, 0, 0, 0] += 1e-4
            print([np.min(a) for a in ins])
            print([np.max(a) for a in ins])
            #print("VALS", [np.max(a) for a in ins])
            optimizer.zero_grad()
            img_ins = [show_volume(a, screenshot=True, wsize=(wsize, wsize)) for a in ins]
            img_outs = [show_volume(a, screenshot=True, wsize=(wsize, wsize)) for a in outs]
            img_pairs = [np.concatenate((a, b), axis=0) for a, b in zip(img_ins, img_outs)]
            examples = []
            for pair in img_pairs:
                examples.append(wandb.Image(pair, ""))
            wandb.log({"Examples": examples})
        
        print(f"Epoch {ep+1} done. Loss = {np.mean(ep_loss)}")

    return model

def train_step(batch, model, optim, loss_func):
    y = model(batch)[0]
    loss = loss_func(batch, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


if __name__ == "__main__":


    from models import DenseVAE
    from dataset import BlockDataset

    # model = DenseVAE([10, 10, 10], latent_dim=64)

    # dataset = BlockDataset("data", blocksize=8, margin=1, save_load=False)


    # model = train_model(dataset, model, batch_size=64, epochs=10,
    #                     run_name="dense_vae_d_s8_l64_m1_e10")
    # save_model(model, "weights/dense_vae_d.pt")


    CONF_DIM = {
        "dense_vae_s8_m0_h128_l32_e10_b64": {
            "size": 8,
            "margin": 0,
            "hidden": [128],
            "latent": 32,
            "epochs": 10,
            "batch_size": 64
        }
    }




    # TRAINING ON REMOTE MACHINE

    # 8x8x8 variations with zero margin - comparing training params:

    dataset = BlockDataset("data", blocksize=8, margin=0, save_load=False)

    name = "dense_vae_s8_m0_l32_h128_b64_e8"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[128])
    model = train_model(dataset, model, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l32_h128_b64_e20"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[128])
    model = train_model(dataset, model, batch_size=64, epochs=20, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l32_h128_b16_e8"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[128])
    model = train_model(dataset, model, batch_size=16, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l32_h128_b64_e8_l3_lr2-5"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[128])
    model = train_model(dataset, model, lr=2e-5, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l32_h128_b64_e8_l3_lr1-3"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[128])
    model = train_model(dataset, model, lr=1e-3, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    # 8x8x8 variations with zero margin - comparing model dimensionality:

    name = "dense_vae_s8_m0_l16_h128_b64_e8"
    model = DenseVAE([8, 8, 8], latent_dim=16, hidden_dims=[128])
    model = train_model(dataset, model, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l64_h256_b64_e8"
    model = DenseVAE([8, 8, 8], latent_dim=64, hidden_dims=[256])
    model = train_model(dataset, model, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")

    name = "dense_vae_s8_m0_l32_h256-128_b64_e8"
    model = DenseVAE([8, 8, 8], latent_dim=32, hidden_dims=[256, 128])
    model = train_model(dataset, model, batch_size=64, epochs=8, run_name=name)
    model.save_model(f"weights/{name}.pt")



