from __future__ import division, print_function

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

# from .configs import paths
# from .configs import constants
from util import debug_print
import json

word_len_padding = 8

def load_csv_labels(csv_path) -> dict[str, str]:

    with open(csv_path, 'r', encoding="utf-8") as file:
        labels = json.load(file)

    return labels


def load_filepaths_and_labels(json_file, data_dir, data_dir_ref) -> tuple[list, list]:
    sample_paths: list[str] = []
    sample_paths_ref: list[str] = []
    labels: list[str] = []

    label_dict = load_csv_labels(json_file)

    for file_name in os.listdir(data_dir):
        path = os.path.join(data_dir, file_name)
        # print('data_dir_ref', data_dir_ref)
        # print('file_name', file_name)
        path_ref = os.path.join(data_dir_ref, file_name)

        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # assert file_name in label_dict, f"No label for image '{file_name}'"
            if file_name not in label_dict: continue
            label = label_dict[file_name]

            sample_paths.append(path)
            sample_paths_ref.append(path_ref)
            labels.append(label)

    debug_print(f"Loaded {len(sample_paths)} samples from {data_dir}")
    assert len(sample_paths) == len(labels)
    return sample_paths, sample_paths_ref, labels


class HCRDataset(Dataset):
    def __init__(self, json_file, data_dir, data_dir_ref, processor: TrOCRProcessor):
        self.image_name_list, self.image_name_list_ref, \
        self.label_list = load_filepaths_and_labels(json_file, data_dir, data_dir_ref)
        self.processor = processor
        self._max_label_len = max([word_len_padding] + [len(label) for label in self.label_list])

        # new lines
        self.losses = torch.zeros(len(self.image_name_list))
        self.indices = list(range(len(self.image_name_list)))

    # new lines
    def update_losses(self, new_losses):
        self.losses = torch.tensor(new_losses)  # Store updated losses
        self.indices = sorted(range(len(self.losses)), key=lambda i: self.losses[i])  # Sort by loss

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_ref = Image.open(self.image_name_list_ref[idx]).convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]
        image_tensor_ref: torch.Tensor = self.processor(image_ref, return_tensors="pt").pixel_values[0]

        label = self.label_list[idx]
        label_tensor = self.processor.tokenizer(
            label,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self._max_label_len,
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "input_ref":image_tensor_ref, "label": label_tensor}

    def get_label(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.label_list[idx]

    def get_path(self, idx) -> str:
        assert 0 <= idx < len(self.label_list), f"id {idx} outside of bounds [0, {len(self.label_list)}]"
        return self.image_name_list[idx]


class MemoryDataset(Dataset):
    def __init__(self, images: list[Image.Image], processor: TrOCRProcessor):
        self.images = images
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor: torch.Tensor = self.processor(image, return_tensors="pt").pixel_values[0]

        # create fake label
        label_tensor: torch.Tensor = self.processor.tokenizer(
            "",
            return_tensors="pt",
        ).input_ids[0]

        return {"idx": idx, "input": image_tensor, "label": label_tensor}
