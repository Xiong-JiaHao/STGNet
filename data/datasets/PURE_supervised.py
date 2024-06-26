import torch
import numpy as np
from data.datasets.PURE import PURE
import data.datasets.transforms as transforms


class PURESupervised(PURE):
    def __init__(self, split, arg_obj):
        """
        Initialize the PURESupervised dataset.

        Args:
            split (str): The data split to load, e.g., 'train', 'test', or 'valid'.
            arg_obj: An object containing the arguments passed to the program.
        """
        super().__init__(split, arg_obj)

    def set_augmentations(self):
        """
        Set the augmentations based on the specified arguments.
        """
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        if self.split == 'train':
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False
            self.aug_speed = True if 's' in self.aug else False
            self.aug_resizedcrop = True if 'c' in self.aug else False
        self.aug_reverse = False ## Don't use this with supervised

    def __getitem__(self, idx):
        """
        Get the item (clip, wave, mask, subject ID, speed) at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the clip, wave, mask, subject ID, and speed.
        """
        subj, start_idx = self.samples[idx]
        idcs = np.arange(start_idx, start_idx + self.frames_per_clip, dtype=int)
        clip = self.data[subj]['video'][idcs] # [T,H,W,C]
        clip = transforms.prepare_clip(clip, self.channels) # [C,T,H,W]
        clip, speed_idcs, speed = self.apply_transformations(clip, subj, idcs)
        if speed != 1.0:
            min_idx = int(speed_idcs[0])
            max_idx = int(speed_idcs[-1])+1
            orig_x = np.arange(min_idx, max_idx, dtype=int)
            if self.split == "train":
                orig_wave = self.data[subj]['wave'][orig_x]
                wave = np.interp(speed_idcs, orig_x, orig_wave)
                mask = -1
            else:
                orig_wave = self.waves[subj][orig_x]
                wave = np.interp(speed_idcs, orig_x, orig_wave)
                orig_mask = self.masks[subj][orig_x]
                mask = np.interp(speed_idcs, orig_x, orig_mask)
                mask = mask > 0.5
        else:
            if self.split == "train":
                wave = self.data[subj]['wave'][idcs]
                mask = -1
            else:
                wave = self.waves[subj][idcs]
                mask = self.masks[subj][idcs]
                mask = torch.from_numpy(mask).bool()

        wave = (wave - wave.mean()) / np.std(wave)
        wave = torch.from_numpy(wave).float()

        return clip, wave, mask, subj, speed
