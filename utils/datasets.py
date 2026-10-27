import os
import sys
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *
import pickle
import csv
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

# hello


class IXIDataSet(Dataset):
    def __init__(self, config, mode):
        super(IXIDataSet, self).__init__()
        self.config = config
        self.mode = mode
        self.image_size = config.data.image_size
        self.root = getattr(
            config.data,
            "ixi_root",
            os.path.join("dataset", "IXI"),
        )

        split = "train" if mode == "training" else "test"
        self.t1_root = os.path.join(self.root, split, "T1")
        self.t2_root = os.path.join(self.root, split, "T2")
        self.samples = self._build_samples()
        print(f"IXI {split}: found {len(self.samples)} paired slices")

    def _build_samples(self):
        samples = []
        if not os.path.isdir(self.t1_root) or not os.path.isdir(self.t2_root):
            raise FileNotFoundError(
                f"IXI split folders not found: {self.t1_root}, {self.t2_root}"
            )

        subject_ids = sorted(
            set(os.listdir(self.t1_root)).intersection(os.listdir(self.t2_root)),
            key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
        )
        for subject_id in subject_ids:
            t1_dir = os.path.join(self.t1_root, subject_id)
            t2_dir = os.path.join(self.t2_root, subject_id)
            if not os.path.isdir(t1_dir) or not os.path.isdir(t2_dir):
                continue
            t1_files = {
                name
                for name in os.listdir(t1_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            }
            t2_files = {
                name
                for name in os.listdir(t2_dir)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            }
            for file_name in sorted(t1_files.intersection(t2_files)):
                samples.append(
                    (
                        os.path.join(t2_dir, file_name),  # target modality
                        os.path.join(t1_dir, file_name),  # auxiliary modality
                    )
                )
        return samples

    def _load_image(self, path):
        image = Image.open(path).convert("F")
        image = np.asarray(image, dtype=np.float32)
        image = image - np.min(image)
        max_value = np.max(image)
        if max_value > 0:
            image = image / max_value

        height, width = image.shape
        if height > self.image_size or width > self.image_size:
            start_y = max((height - self.image_size) // 2, 0)
            start_x = max((width - self.image_size) // 2, 0)
            image = image[
                start_y : start_y + min(height, self.image_size),
                start_x : start_x + min(width, self.image_size),
            ]
            height, width = image.shape

        padded = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        start_y = (self.image_size - height) // 2
        start_x = (self.image_size - width) // 2
        padded[start_y : start_y + height, start_x : start_x + width] = image
        return padded

    def _image_to_kspace(self, image):
        image = image[np.newaxis, np.newaxis, :, :].astype(np.complex64)
        kspace = FFT2c(image)
        return np.squeeze(kspace, axis=0)

    def __getitem__(self, idx):
        t2_path, t1_path = self.samples[idx]
        t2_image = self._load_image(t2_path)
        t1_image = self._load_image(t1_path)

        kspace_t2 = self._image_to_kspace(t2_image)
        kspace_t1 = self._image_to_kspace(t1_image)
        csm = np.ones((1, self.image_size, self.image_size), dtype=np.complex64)

        if self.mode != "training":
            subject_id = os.path.basename(os.path.dirname(t2_path))
            slice_name = os.path.basename(t2_path)
            return kspace_t2, csm, kspace_t1, subject_id, slice_name

        return kspace_t2, csm, kspace_t1

    def __len__(self):
        return len(self.samples)

class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIKneeDataSet, self).__init__()
        self.config = config
        
        # Default paths and pairing
        self.pd_kspace_dir = None
        self.pair_csv = None
        self.pair_map = {}

        if mode == "training":
            self.kspace_dir = "data/train/pdfs/kspace/"
            self.maps_dir = "data/train/pdfs/map/"
            self.pd_kspace_dir = "data/train/pd/kspace/"
            self.pair_csv = "data/train/knee-pair.csv"
            input_pkl = "data/train/data_slice.pkl"
        elif mode == "test":
            self.kspace_dir = "data/test/pdfs/kspace/"
            self.maps_dir = "data/test/pdfs/map/"
            self.pd_kspace_dir = "data/test/pd/kspace/"
            self.pair_csv = "data/test/test-pair.csv"
            input_pkl = "data/test/data_slice.pkl"
        elif mode == "sample":
            self.kspace_dir = "data/photom/kspace/"
            self.maps_dir = "data/photom/map/"
            input_pkl = "data/photom/data_slice.pkl"
        elif mode == "photom":
            self.kspace_dir = "data/photom/kspace/"
            self.maps_dir = "data/photom/map/"
            input_pkl = "data/photom/data_slice.pkl"
        elif mode == "datashift":
            self.kspace_dir = "data/brain_T2/kspace/"
            self.maps_dir = "data/brain_T2/map/"
            input_pkl = "data/brain_T2/data_slice.pkl"
        else:
            raise NotImplementedError

        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir)
        print(f"Found {len(self.file_list)} files")

        self.num_slices = np.zeros((len(self.file_list),), dtype=int)

        # Keep compatibility with older installations, but never accept an
        # incomplete index: that previously made every missing scan count as a
        # single slice and hid broken datasets.
        if not os.path.exists(input_pkl):
            legacy_pkl = "data/data_slice.pkl"
            if os.path.exists(legacy_pkl):
                input_pkl = legacy_pkl
            else:
                raise FileNotFoundError(
                    f"Slice index not found: {input_pkl}. Run save_data_slice.py "
                    f"for {self.kspace_dir} first."
                )

        with open(input_pkl, "rb") as pkl_file:
            data_dict = pickle.load(pkl_file)

        indexed_files = set(data_dict)
        dataset_files = {os.path.basename(file) for file in self.file_list}
        missing_files = sorted(dataset_files - indexed_files)
        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise ValueError(
                f"Slice index {input_pkl} is incomplete: {len(missing_files)} of "
                f"{len(dataset_files)} files are missing ({preview}). Regenerate "
                "the index after validating the HDF5 files."
            )

        for idx, file in enumerate(self.file_list):
            # Exclude the first 8 frames for training
            if self.mode != "sample" and self.mode != "photom":
                self.num_slices[idx] = int(max(data_dict[os.path.basename(file)] - 8, 0))
            else:
                self.num_slices[idx] = int(data_dict[os.path.basename(file)])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1

        # Load pairing info
        if self.pair_csv and os.path.exists(self.pair_csv):
            try:
                with open(self.pair_csv, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    # Simple heuristic to skip header if present
                    # Check if the 3rd column is a float number (score)
                    if rows and len(rows[0]) >= 3:
                        try:
                            float(rows[0][2])
                        except ValueError:
                            # If conversion fails, assume it's a header
                            print(f"Skipping header: {rows[0]}")
                            rows = rows[1:]
                    for row in rows:
                        if len(row) >= 2:
                            # CSV format: PD, PDFS, Score
                            # We map PDFS (row[1]) -> PD (row[0])
                            self.pair_map[row[1]] = row[0]
                print(f"Loaded {len(self.pair_map)} pairs from {self.pair_csv}")
            except Exception as e:
                print(f"Error loading pair csv: {e}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Subject number
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Frame number of the subject's scan
        slice_idx = (
            int(idx)
            if scan_idx == 0
            else int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        )

        # Determine file path
        filename = os.path.basename(self.file_list[scan_idx])
        maps_file = os.path.join(self.maps_dir, filename)
        
        # Calculate real slice index (skipping first 8 frames for training)
        real_slice_idx = slice_idx
        if self.mode != "sample" and self.mode != "photom":
             real_slice_idx = slice_idx + 8

        # Load maps
        with h5py.File(maps_file, "r") as data:
            num_slices = data["s_maps"].shape[0]
            current_slice = min(num_slices - 1, real_slice_idx)
            
            maps_idx = data["s_maps"][current_slice]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = crop(maps_idx, cropx=320, cropy=320)
            maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)

        # Load raw data (PDFS)
        raw_file = os.path.join(self.kspace_dir, filename)
        with h5py.File(raw_file, "r") as data:
            ksp_idx = data["kspace"][current_slice]
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(IFFT2c(ksp_idx), cropx=320, cropy=320)
            ksp_idx = FFT2c(ksp_idx)
            ksp_idx = np.squeeze(ksp_idx, 0)
            
            if self.config.data.normalize_type == "minmax":
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
            elif self.config.data.normalize_type == "std":
                minv = np.std(ksp_idx)
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)
            elif self.config.data.normalize_type == "img_std":
                ksp_idx = np.expand_dims(ksp_idx, 0)
                ksp_idx = IFFT2c(ksp_idx)
                ksp_idx = normalize_l2(ksp_idx)
                ksp_idx = FFT2c(ksp_idx)
                ksp_idx = np.squeeze(ksp_idx, 0)

            kspace = np.asarray(ksp_idx)

        # Load PD k-space
        kspace_pd = np.zeros((1, 1, 1)) # Default if missing
        name_no_ext = os.path.splitext(filename)[0]
        if self.pd_kspace_dir and name_no_ext in self.pair_map:
            pd_filename = self.pair_map[name_no_ext] + ".h5"
            pd_path = os.path.join(self.pd_kspace_dir, pd_filename)
            if os.path.exists(pd_path):
                try:
                    with h5py.File(pd_path, "r") as pd_h5:
                        if "kspace" in pd_h5:
                            pd_num_slices = pd_h5["kspace"].shape[0]
                            if current_slice < pd_num_slices:
                                pd_data = pd_h5["kspace"][current_slice]
                                pd_data = np.expand_dims(pd_data, 0)
                                # Apply same crop as PDFS to ensure matching dimensions
                                pd_data = crop(IFFT2c(pd_data), cropx=320, cropy=320)
                                pd_data = FFT2c(pd_data) 
                                pd_data = np.squeeze(pd_data, 0)
                                kspace_pd = np.asarray(pd_data)
                except Exception as e:
                    print(f"Error reading PD file {pd_filename}: {e}")

        return kspace, maps, kspace_pd

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == "fastMRI_knee":
        dataset = FastMRIKneeDataSet(config, mode)
    elif config.data.dataset_name == "IXI":
        dataset = IXIDataSet(config, mode)
    else:
        raise ValueError(f"Dataset {config.data.dataset_name} not recognized.")

    if mode == "training":
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        data = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=8,
            pin_memory=False,
        )
        # data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        from utils.utils import worker_init_fn

        data = DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            num_workers=0,
        )

    print(mode, "data loaded")

    return data
