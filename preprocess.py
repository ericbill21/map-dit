import argparse
import os

import torch
import numpy as np


def main(args):
    print(f"Loading data from \"{args.data_path}\"...")
    data, labels = [], []

    # Iterate through all .npz files in the data path
    for file in os.listdir(args.data_path):
        if not file.endswith(".npz"):
            continue
            
        print(f"\tReading \"{file}\"...")
        data_batch = np.load(os.path.join(args.data_path, file))
        data.append(data_batch["data"])
        labels.append(data_batch["labels"])

    if len(data) == 0:
        ValueError(f"No \".npz\" files found in \"{args.data_path}\"")
    
    print(f"Saving data to \"{args.output_dir}\"...")
    output_dir = os.makedirs(args.output_dir, exist_ok=True)
    
    # Concatenate all data and labels, cast to torch tensors, and convert to 3x32x32
    data = torch.from_numpy(np.concatenate(data, axis=0)).byte().reshape(-1, 3, 32, 32)
    labels = torch.from_numpy(np.concatenate(labels, axis=0)).long()

    # Save data and labels
    torch.save(data, os.path.join(args.output_dir, "features.pt"))
    torch.save(labels, os.path.join(args.output_dir, "labels.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True, help="Path to directory containing .npz files")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to directory to save features.pt and labels.pt")

    args = parser.parse_args()
    main(args)