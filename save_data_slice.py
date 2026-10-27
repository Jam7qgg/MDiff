# -*- coding: utf-8 -*-
"""Build a slice-count index for a fastMRI k-space directory.

The index is written only when every HDF5 file passes validation, so a partial
or corrupt dataset cannot silently turn into a tiny training dataset.
"""

import argparse
import csv
import os
import pickle
import tempfile

import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        default="data/train/pdfs/kspace",
        help="Directory containing PDFS k-space HDF5 files.",
    )
    parser.add_argument(
        "--output",
        default="data/train/data_slice.pkl",
        help="Output pickle containing {filename: number_of_slices}.",
    )
    parser.add_argument(
        "--maps-directory",
        default="data/train/pdfs/map",
        help="Directory containing PDFS sensitivity-map HDF5 files.",
    )
    parser.add_argument(
        "--pd-directory",
        default="data/train/pd/kspace",
        help="Directory containing paired PD k-space HDF5 files.",
    )
    parser.add_argument(
        "--pairs",
        default="data/train/knee-pair.csv",
        help="CSV in PD,PDFS,score format.",
    )
    return parser.parse_args()


def load_pair_map(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pair CSV not found: {path}")
    pair_map = {}
    with open(path, "r", newline="") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue
            pd_name, pdfs_name = row[0].strip(), row[1].strip()
            if pd_name.lower() == "pd" and pdfs_name.lower() == "pdfs":
                continue
            pair_map[pdfs_name] = pd_name
    return pair_map


def read_shape(path, dataset_name):
    with h5py.File(path, "r") as h5f:
        if dataset_name not in h5f:
            raise KeyError(f"missing '{dataset_name}' dataset")
        shape = h5f[dataset_name].shape
        if not shape or shape[0] <= 0:
            raise ValueError(f"invalid {dataset_name} shape: {shape}")
        return shape


def main():
    args = parse_args()
    directory = os.path.normpath(args.directory)
    output_pkl = os.path.normpath(args.output)
    maps_directory = os.path.normpath(args.maps_directory)
    pd_directory = os.path.normpath(args.pd_directory)
    pair_map = load_pair_map(os.path.normpath(args.pairs))

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(maps_directory):
        raise FileNotFoundError(f"Maps directory not found: {maps_directory}")
    if not os.path.isdir(pd_directory):
        raise FileNotFoundError(f"PD directory not found: {pd_directory}")

    h5_files = sorted(
        filename for filename in os.listdir(directory) if filename.endswith(".h5")
    )
    if not h5_files:
        raise RuntimeError(f"No HDF5 files found in: {directory}")

    data_dict = {}
    invalid_files = []
    for filename in h5_files:
        full_path = os.path.join(directory, filename)
        try:
            pdfs_name = os.path.splitext(filename)[0]
            if pdfs_name not in pair_map:
                raise KeyError("missing PDFS entry in pair CSV")
            pd_filename = pair_map[pdfs_name] + ".h5"

            kspace_shape = read_shape(full_path, "kspace")
            maps_shape = read_shape(os.path.join(maps_directory, filename), "s_maps")
            pd_shape = read_shape(os.path.join(pd_directory, pd_filename), "kspace")

            if maps_shape[0] < kspace_shape[0]:
                raise ValueError(
                    f"s_maps has fewer slices ({maps_shape[0]}) than PDFS "
                    f"kspace ({kspace_shape[0]})"
                )
            data_dict[filename] = int(kspace_shape[0])
            print(
                f"OK {filename}: PDFS={kspace_shape[0]}, maps={maps_shape[0]}, "
                f"PD={pd_shape[0]}",
                flush=True,
            )
        except Exception as exc:
            invalid_files.append((filename, str(exc)))
            print(f"BAD {filename}: {exc}", flush=True)

    if invalid_files:
        details = "\n".join(f"  {name}: {error}" for name, error in invalid_files)
        raise RuntimeError(
            f"Dataset validation failed for {len(invalid_files)}/{len(h5_files)} files. "
            f"The old index was left unchanged:\n{details}"
        )

    output_dir = os.path.dirname(output_pkl) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix="data_slice_", suffix=".pkl.tmp", dir=output_dir
    )
    try:
        with os.fdopen(fd, "wb") as pkl_file:
            pickle.dump(data_dict, pkl_file)
        os.replace(temporary_path, output_pkl)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise

    counts = list(data_dict.values())
    print(f"Index saved to: {output_pkl}")
    print(f"Files: {len(counts)}")
    print(f"Total slices: {sum(counts)}")
    print(f"Slice range: {min(counts)}..{max(counts)}")


if __name__ == "__main__":
    main()
