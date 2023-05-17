import os
import shutil
import zipfile

import numpy as np
import torch
import pandas as pd
import gdown

from utils import load_volume, save_volume, show_volume, \
                    blockify, unblockify, to_mono, to_dtype

EMPTY_THR = 0.00001
PACKET_SIZE = 10000

class BlockDataset():

    def __init__(self, datadir, blocksize=8, margin=0, shuffle=True, use_float=True,
                split="train", ratio=0.7, idx=None, save_load=True):
        
        if not os.path.isdir(datadir): os.makedirs(datadir)
        if len(list(os.listdir(datadir))) == 0:
            gdown.download_folder("https://drive.google.com/drive/folders/1eas1dzKSzVaFfZlp1rBxRoCEvrfd_p0J")
            shutil.move("Volumes", "data")
        
        self.dir = datadir
        self.size = blocksize
        self.margin = margin
        self.float = use_float
        
        savedir = os.path.join(datadir, "saved")
        issaved = os.path.isdir(savedir) and len(list(os.listdir(savedir))) > 0 

        dfdir = os.path.join(datadir, "index.csv")
        self.df = pd.read_csv(dfdir, sep=",", header=None)

        if not save_load or not issaved:
            self.vols = self._loadvols(self.df)
            self.blocks = self._blockify(self.vols, self.size, self.margin)
            self.blocks = np.array(self.blocks)

            if save_load:
                self._save_blocks(savedir, self.blocks)
        else:
            self.blocks = self._load_blocks(savedir, len(self.df))
        


        print(self.blocks.shape)

        if shuffle:
            perm = np.random.RandomState(69).permutation(len(self.blocks))
            self.blocks = self.blocks[perm]

        if idx:
            self.blocks = self.blocks[idx]
        elif split is not None:
            limit = int(ratio * len(self.blocks))
            self.blocks = self.blocks[:limit] if split == "train" else self.blocks[limit:]
        

        
    def _loadvols(self, df):
        vols = []
        isrgb = lambda x: True if x == 3 else False
        getdtp = lambda x: np.uint16 if x == 16 else np.uint8

        for i, row in df.iterrows():
            r = list(row)
            pth = os.path.join(self.dir, r[0])
            rgb = isrgb(r[5])
            dtp = getdtp(r[4])
            vol = load_volume(pth, r[1:4], dtype=dtp, rgb=rgb)
            vol = to_dtype(to_mono(vol), np.uint8)
            vols.append(vol)
        return vols

    def _blockify(self, vols, size, margin, remove_empty=True):
        
        blocks = []
        for vol in vols:
            block_grid = blockify(vol, size, margin)
            fullsz = size+margin*2
            block_arr = block_grid.reshape(-1, fullsz, fullsz, fullsz)
            block_arr = block_arr.astype(np.uint8)
            if self.float: block_arr = to_dtype(block_arr, np.float32)
            print(block_arr.shape)
            blocks.extend(block_arr)
        
        if remove_empty:
            blocks = [bl for bl in blocks if np.mean(bl) > EMPTY_THR]

        return blocks
    
    def _save_blocks(self, dir, blocks):
        if not os.path.isdir(dir): os.mkdir(dir)
        for b in range(0, len(blocks), PACKET_SIZE):
            pth = os.path.join(dir, f"{b}.npy")
            packet = self.blocks[b:b+PACKET_SIZE, :, :, :]
            np.save(pth, packet)

    def _load_blocks(self, dir, n):
        blocks = []
        for b in range(0, n, PACKET_SIZE):
            pth = os.path.join(dir, f"{b}.npy")
            blocks.append(np.load(pth))
        return np.concatenate(blocks, axis=0)

    
    def _parse_index(self, idx):
        if idx is None:
            return np.arange(0, len(self.blocks))
        if isinstance(idx, int):
            return [idx]
        return idx

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return torch.from_numpy(self.blocks[idx])

    def get_batch_numpy(self, idx=None):
        return self.blocks[self._parse_index(idx)]
    
    def get_batch_torch(self, idx=None):
        np_batch = self.get_batch_numpy(idx)
        return torch.from_numpy(np_batch)


if __name__ == "__main__":

    ds = BlockDataset("data", blocksize=8, shuffle=False)
    vol0 = ds.vols[0]
    vol0bl0 = vol0[100:116, 100:116, 100:116]

    print(vol0bl0)

    blocks = ds.get_batch_numpy([4000, 4001, 4002])

    print(blocks[2])
