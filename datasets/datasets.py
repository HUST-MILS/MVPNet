from ctypes import sizeof
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from utils.projection import projection
from utils.preprocess_data import prepare_data, compute_mean_and_std
from utils.utils import load_files 

class RangeDatasetRaw(Dataset):

    def __init__(self,cfg,split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("PATH_TO_PROCESSED_DATA")
        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]

        self.BEV_height = (self.cfg["DATA_CONFIG"]["FWD_RANGE"]-self.cfg["DATA_CONFIG"]["BACK_RANGE"])/ 0.2 
        self.BEV_width = (self.cfg["DATA_CONFIG"]["RIGHT_RANGE"]-self.cfg["DATA_CONFIG"]["RIGHT_RANGE"])/ 0.2 

        self.n_channels = 5

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]

        self.projection = projection(self.cfg)

        if split == "train":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        elif split == "val":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        elif split == "test":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")
        
        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames_range = {}
        self.filenames_xyz = {}
        self.filenames_intensity = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0

        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))

            scan_path_range = os.path.join(self.root_dir, seqstr, "Range_processed", "range")
            self.filenames_range[seq] = load_files(scan_path_range)

            scan_path_xyz = os.path.join(self.root_dir, seqstr, "Range_processed", "xyz")
            self.filenames_xyz[seq] = load_files(scan_path_xyz)
            assert len(self.filenames_range[seq]) == len(self.filenames_xyz[seq])

            scan_path_intensity = os.path.join(self.root_dir, seqstr, "Range_processed", "intensity")
            self.filenames_intensity[seq] = load_files(scan_path_intensity)
            assert len(self.filenames_range[seq]) == len(self.filenames_intensity[seq])

            # Get number of sequences based on number of past and future steps
            n_samples_sequence = max(
                0,
                len(self.filenames_range[seq])
                - self.n_past_steps
                - self.n_future_steps
                + 1,
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.n_past_steps + sample_idx - 1
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence

        def __len__(self):
            return self.dataset_size

        def __getitem__(self,idx):
            """Load and concatenate range image channels

            Args:
                idx (int): Sample index

            Returns:
                item: Dataset dictionary item
            """
        seq, scan_idx = self.idx_mapper[idx]
        # Load past data
        past_data = torch.empty(
            [self.n_channels, self.n_past_steps, self.height, self.width]
        )


        #return item_RV,iten_BEV

        
        def load_range(self, filename):
            """Load .npy range image as (1,height,width) tensor"""
            rv = torch.Tensor(np.load(filename)).float()
            return rv

        def load_xyz(self, filename):
            """Load .npy xyz values as (3,height,width) tensor"""
            xyz = torch.Tensor(np.load(filename)).float()[:, :, :3]
            xyz = xyz.permute(2, 0, 1)
            return xyz

        def load_intensity(self, filename):
            """Load .npy intensity values as (1,height,width) tensor"""
            intensity = torch.Tensor(np.load(filename)).float()
            return intensity