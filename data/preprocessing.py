import torch
import numpy as np
import json
from typing import Tuple, Union
from .transforms import NormalizeByRegion
import random
from collections import defaultdict


class DataSplitter:
    """A class for splitting a dataset into training, validation, and test sets based on subjects.
    """

    def __init__(
        self,
        data_tensor: torch.Tensor,
        val_split: float,
        test_split: float,
        seed: int,
        index_to_name: dict,
        imageID_to_labels: dict,
    ):
        self.data = data_tensor
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.index_to_name = index_to_name
        self.imageID_to_labels = imageID_to_labels

        # Build mapping from subject to indices
        self.subject_to_indices = self._build_subject_to_indices()

        # Get shuffled list of subjects
        self.subjects = list(self.subject_to_indices.keys())
        random.seed(self.seed)
        random.shuffle(self.subjects)

        # Calculate target sizes based on total samples
        total_samples = len(self.data)
        self.target_test_size = int(total_samples * self.test_split)
        self.target_val_size = int(total_samples * self.val_split)
        self.target_train_size = (
            total_samples - self.target_test_size - self.target_val_size
        )

    def _build_subject_to_indices(self):
        """Build a mapping from subject IDs to data indices."""
        subject_to_indices = defaultdict(list)
        for idx in range(len(self.data)):
            image_id = self.index_to_name[idx]
            subject_id = self.imageID_to_labels[image_id]["subject_id"]
            subject_to_indices[subject_id].append(idx)
        return dict(subject_to_indices)

    def _assign_subjects_to_splits(
        self,
    ):
        """Assign subjects to train/val/test splits to approximate target sizes.

        Uses a greedy approach that assigns subjects to test first, then val, then train,
        trying to get as close as possible to target sizes without exceeding them too much.
        """
        test_subjects = []
        val_subjects = []
        train_subjects = []

        test_count = 0
        val_count = 0

        for subject in self.subjects:
            subject_size = len(self.subject_to_indices[subject])

            # Assign to test if we haven't reached target yet
            if test_count < self.target_test_size:
                # Check if adding this subject gets us closer to target
                if test_count + subject_size <= self.target_test_size or abs(
                    test_count + subject_size - self.target_test_size
                ) < abs(test_count - self.target_test_size):
                    test_subjects.append(subject)
                    test_count += subject_size
                    continue

            # Assign to val if we haven't reached target yet (and val_split > 0)
            if self.val_split > 0 and val_count < self.target_val_size:
                if val_count + subject_size <= self.target_val_size or abs(
                    val_count + subject_size - self.target_val_size
                ) < abs(val_count - self.target_val_size):
                    val_subjects.append(subject)
                    val_count += subject_size
                    continue

            # Otherwise assign to train
            train_subjects.append(subject)

        return train_subjects, val_subjects, test_subjects

    def split_data(
        self,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Splits data indices based on subjects.

        Returns sorted arrays of indices for each split.
        """
        train_subjects, val_subjects, test_subjects = self._assign_subjects_to_splits()

        # Collect indices for each split
        train_indices = []
        val_indices = []
        test_indices = []

        for subject in train_subjects:
            train_indices.extend(self.subject_to_indices[subject])
        for subject in val_subjects:
            val_indices.extend(self.subject_to_indices[subject])
        for subject in test_subjects:
            test_indices.extend(self.subject_to_indices[subject])

        # Calculate actual sizes for reporting
        total = len(self.data)
        train_size = len(train_indices)
        val_size = len(val_indices)
        test_size = len(test_indices)

        print(
            f"Train samples: {train_size} ({train_size / total:.2%}) "
            f"[target: {self.target_train_size} ({1 - self.val_split - self.test_split:.2%})]"
        )
        if val_size > 0:
            print(
                f"Validation samples: {val_size} ({val_size / total:.2%}) "
                f"[target: {self.target_val_size} ({self.val_split:.2%})]"
            )
        print(
            f"Test samples: {test_size} ({test_size / total:.2%}) "
            f"[target: {self.target_test_size} ({self.test_split:.2%})]"
        )
        print(
            f"Subjects - Train: {len(train_subjects)}, "
            f"Val: {len(val_subjects)}, Test: {len(test_subjects)}"
        )

        train_indices = np.sort(np.array(train_indices))
        test_indices = np.sort(np.array(test_indices))

        if val_size > 0:
            val_indices = np.sort(np.array(val_indices))
            return train_indices, val_indices, test_indices
        else:
            return train_indices, test_indices


class TimeWindowSplitter:
    """Splits time-series data into windows."""

    def __init__(self, data: torch.Tensor, window_size: int):
        self.data = data
        self.window_size = window_size

    def split(self) -> torch.Tensor:
        """Performs the split."""
        n_samples, h, w, d, t = self.data.shape
        if t % self.window_size != 0:
            raise ValueError(
                f"Time dimension ({t}) must be divisible by window size ({self.window_size})."
            )

        num_windows = t // self.window_size

        # Reshape and permute to create windows
        # [N, H, W, D, T] -> [N, H, W, D, num_windows, window_size]
        data = self.data.view(n_samples, h, w, d, num_windows, self.window_size)
        # -> [N, num_windows, H, W, D, window_size]
        data = data.permute(0, 4, 1, 2, 3, 5).contiguous()
        # -> [N * num_windows, H, W, D, window_size]
        data = data.view(-1, h, w, d, self.window_size)

        return data

    @staticmethod
    def update_info_dict(info_dict: dict, original_len: int, window_size: int) -> dict:
        """Updates the info dictionary to reflect the windowed data."""
        num_windows = original_len // window_size
        new_info_dict = {}
        original_keys = sorted(info_dict.keys())

        new_idx = 0
        for old_idx in original_keys:
            info = info_dict[old_idx]
            for i in range(num_windows):
                new_info = info.copy()
                new_info["window_index"] = i
                new_info_dict[new_idx] = new_info
                new_idx += 1
        return new_info_dict


def load_and_process_data(config):
    """Main function to load, preprocess, and split the data."""

    with open(f"{config.BASE_DATA_PATH}/index_to_name.json", "r") as f:
        index_to_info = json.load(f)
    index_to_info = {int(k): v for k, v in index_to_info.items()}

    with open(f"{config.BASE_DATA_PATH}/imageID_to_labels.json", "r") as f:
        imageID_to_labels = json.load(f)

    all_data_4d = torch.load(f"{config.BASE_DATA_PATH}/data/all_4d_downsampled.pt")
    schaefer_atlas = torch.load(
        f"{config.BASE_DATA_PATH}/data/time_regions_tensor_not_normalized_schaefer.pt"
    )
    schaefer_atlas = schaefer_atlas.permute(0, 2, 1)  # samples, regions, time

    std_data = np.std(all_data_4d.numpy(), axis=tuple(range(1, all_data_4d.data.ndim)))
    top_k_std_scans = np.argsort(std_data)[-config.REMOVE_TOP_K_STD :]
    mask_bad = np.isin(np.arange(all_data_4d.size(0)), top_k_std_scans)

    clean_indices = np.where(~mask_bad)[0]
    all_data_4d = all_data_4d[clean_indices]
    schaefer_atlas = schaefer_atlas[clean_indices]

    index_to_info = {
        i: index_to_info[clean_idx] for i, clean_idx in enumerate(clean_indices)
    }

    splitter = DataSplitter(
        all_data_4d,
        config.VAL_SPLIT,
        config.TEST_SPLIT,
        config.SEED,
        index_to_info,
        imageID_to_labels,
    )
    train_indices, val_indices, test_indices = splitter.split_data()

    train_data, val_data, test_data = (
        all_data_4d[train_indices],
        all_data_4d[val_indices],
        all_data_4d[test_indices],
    )
    regions_train, regions_val, regions_test = (
        schaefer_atlas[train_indices],
        schaefer_atlas[val_indices],
        schaefer_atlas[test_indices],
    )

    index_to_info_tr = {i: index_to_info[idx] for i, idx in enumerate(train_indices)}
    index_to_info_val = {i: index_to_info[idx] for i, idx in enumerate(val_indices)}
    index_to_info_test = {i: index_to_info[idx] for i, idx in enumerate(test_indices)}

    region_normalize_4d = NormalizeByRegion(all_data_4d)
    region_normalize_atlas = NormalizeByRegion(schaefer_atlas)

    # Apply windowing
    original_time_len = train_data.shape[-1]
    train_data = TimeWindowSplitter(train_data, config.WINDOW_SIZE).split()
    val_data = TimeWindowSplitter(val_data, config.WINDOW_SIZE).split()
    test_data = TimeWindowSplitter(test_data, config.WINDOW_SIZE).split()
    regions_train = (
        TimeWindowSplitter(regions_train.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )
    regions_val = (
        TimeWindowSplitter(regions_val.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )
    regions_test = (
        TimeWindowSplitter(regions_test.unsqueeze(1).unsqueeze(1), config.WINDOW_SIZE)
        .split()
        .squeeze()
    )

    index_to_info_tr = TimeWindowSplitter.update_info_dict(
        index_to_info_tr, original_time_len, config.WINDOW_SIZE
    )
    index_to_info_val = TimeWindowSplitter.update_info_dict(
        index_to_info_val, original_time_len, config.WINDOW_SIZE
    )
    index_to_info_test = TimeWindowSplitter.update_info_dict(
        index_to_info_test, original_time_len, config.WINDOW_SIZE
    )

    return (
        (train_data, regions_train, index_to_info_tr),
        (val_data, regions_val, index_to_info_val),
        (test_data, regions_test, index_to_info_test),
        imageID_to_labels,
        (all_data_4d, schaefer_atlas),
        (region_normalize_4d, region_normalize_atlas),
    )
