import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
from tqdm import tqdm
import os
import pandas as pd

import logging
from itertools import chain
import re
from musint.datasets.babel_dataset import BabelDataset
from musint.utils.dataframe_utils import trim_mint_dataframe_v2
from musint.datasets.mint_dataset import MintDataset
from musint.benchmarks.muscle_sets import MUSCLE_SUBSETS
import cProfile

from utils.profiler_utils import ProfilerContext

# Configure basic logging
logging.basicConfig(level=logging.INFO)
# Get the root logger
logger = logging.getLogger()

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

DO_PROFILING = False

def find_strings(data):
    strings = []

    def recurse(item):
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                recurse(sub_item)
        elif isinstance(item, str):
            strings.append(item)

    recurse(data)
    return strings

class MSMotionDataset(data.Dataset):

    @ProfilerContext.profile_function()
    def __init__(
        self,
        dataset_name,
        mode="train",
        feat_bias=5,
        max_text_len=20,
        window_size=64,
        unit_length=4,
        fps=20,
        motion_fps=20.0,
        clean_data=True,
        label_required=False,
        muscle_subset=None,  # e.g. "LAI_ARNOLD_LOWER_BODY_8", defined in musint.benchmarks.muscle_sets
        max_frame_gap=1 / 8,
        mint_root="$MINT_DATA",  # "$LSDF/data/activity/MuscleSim/musclesim_dataset",
        motion_root="$MOTION_DATA",
        babel_root="$BABEL_DATA",
    ):
        self.motion_root = os.path.expandvars(motion_root)
        self.mint_root = os.path.expandvars(mint_root)
        self.babel_root = os.path.expandvars(babel_root)

        print(f"Motion root: {self.motion_root}")

        assert self.motion_root != "$MOTION_DATA"
        assert self.mint_root != "$MINT_DATA"
        assert self.babel_root != "$BABEL_DATA"

        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.max_text_len = max_text_len
        self.feat_bias = feat_bias

        self.motion_dir = pjoin(self.motion_root, "new_joint_comp_vecs_flat")
        self.text_dir = pjoin(self.motion_root, "texts")

        self.max_motion_length = 196
        self.fps = fps
        self.motion_fps = motion_fps
        self.mode = mode
        self.label_required = label_required

        self.max_frame_gap = max_frame_gap

        self.muscle_subset = MUSCLE_SUBSETS[muscle_subset] if muscle_subset is not None else None

        self.data = []
        self.lengths = []

        self.mean = np.load(os.path.expandvars(pjoin(self.motion_root, "Mean.npy")))
        self.std = np.load(os.path.expandvars(pjoin(self.motion_root, "Std.npy")))

        with open(pjoin(self.motion_root, "train.txt"), "r") as f:
            self.train_split_ids = [line.strip() for line in f]

        with open(pjoin(self.motion_root, "val.txt"), "r") as f:
            self.val_split_ids = [line.strip() for line in f]

        with open(pjoin(self.motion_root, "test.txt"), "r") as f:
            self.test_split_ids = [line.strip() for line in f]

        # Load Babel Datasets
        babel_paths = [
            os.path.join(self.babel_root, "train.json"),
            os.path.join(self.babel_root, "val.json"),
            os.path.join(self.babel_root, "test.json"),
            os.path.join(self.babel_root, "extra_train.json"),
            os.path.join(self.babel_root, "extra_val.json"),
        ]

        self.babel_dataset = BabelDataset.from_datasets([BabelDataset.from_json_file(bp) for bp in babel_paths])

        # Load Mint Dataset
        self.mint_dataset = MintDataset(
            self.mint_root, keep_in_memory=True, pyarrow=False, load_humanml3d_names=False  #
        )

        index_removal_ids = []

        non_train_ids = self.val_split_ids + self.test_split_ids

        if DO_PROFILING:
            pr = cProfile.Profile()
            pr.enable()

        for idx in tqdm(range(len(self.mint_dataset))):
            id_name = self.mint_dataset.path_ids[idx]

            exclude_motion = False # TODO make this a parameter

            if exclude_motion:


                data_path = self.mint_dataset[idx].data_path

                babel_info = self.babel_dataset.by_feat_p(data_path, raise_missing=False)

                if babel_info is None:
                    logger.info(f"Excluding motion for {id_name}")
                    index_removal_ids.append(id_name)
                    continue

                babel_labels = list(set(find_strings(babel_info.clip_actions())))
                babel_labels.extend(babel_info.sequence_actions()[2])


                # Exclude motion if it has no Babel label or has the Babel label "jumping jack"
                if babel_labels == [] or "jumping jacks" in babel_labels:
                    logger.info(f"Excluding motion for {id_name}")
                    index_removal_ids.append(id_name)
                    continue

            if self.mode == "train":
                if any(id_name in split_id for split_id in non_train_ids):
                    index_removal_ids.append(id_name)
                    continue
            elif self.mode == "val":
                if not any(id_name in split_id for split_id in self.val_split_ids):
                    index_removal_ids.append(id_name)
                    continue
            elif self.mode == "test":
                if not any(id_name in split_id for split_id in self.test_split_ids):
                    index_removal_ids.append(id_name)
                    continue
            else:
                raise ValueError(f"Invalid mode {self.mode}")

            motion_path = pjoin(self.motion_dir, id_name + "_poses.npy")

            try:
                # Load MintData
                mint_data = self.mint_dataset[idx]

                # Load motion data
                motion = np.load(motion_path)

                motlen = motion.shape[0] / self.fps  # in seconds

                if motion.shape[0] < self.window_size:
                    raise ValueError(
                        f"Pose Motion with {motion.shape[0] / self.fps:.2f} "
                        f"shorter than window size of {self.window_size / self.fps:.2f} seconds"
                    )

                # Load muscle activation data using MintData
                muscle_df = mint_data.muscle_activations
                muslen = muscle_df.index[-1] - muscle_df.index[0]  # Index is timestamps.

                # Validate muscle timestamp
                if not self.rand_valid_muscle_ts(muscle_df):
                    # logger.info(f"No valid muscle activation window within {muslen:.2f} seconds.")
                    raise ValueError(f"No full muscle activation window within {muslen:.2f} seconds.")

                if muslen < motlen - 0.4:  # We cut off some activation data at the edges for stability reasons.
                    logger.debug(
                        f"Muscle activation data shorter than motion data for {id_name}. "
                        f"Muscle: {muslen:.2f}, Motion: {motlen:.2f} seconds."
                    )

                # Data is valid, append to lists
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)

            except Exception as e:
                # Log error and mark for removal
                logger.debug(f"Problem with {id_name}: {str(e)}")
                index_removal_ids.append(id_name)

        if DO_PROFILING:
            pr.disable()

            # Create a profile file
            profile_filename = "ms_motion_init_profile.prof"
            pr.dump_stats(profile_filename)

        # Drop invalid entries
        old_len = len(self.mint_dataset.metadata)
        self.mint_dataset.metadata = self.mint_dataset.metadata.drop(index=index_removal_ids)
        logger.info(
            f"Removed {old_len - len(self.mint_dataset.metadata)} invalid entries, {len(self.mint_dataset.metadata)} remaining entries."
        )

        self.id_list = self.mint_dataset.metadata.index

        # Add babel_durs and babel_acts to the mint_dataset metadata
        self.mint_dataset.metadata["babel_durs"] = [
            (
                self.babel_dataset.by_babel_sid(int(mus_meta["babel_sid"])).dur
                if self.babel_dataset.by_babel_sid(int(mus_meta["babel_sid"]), raise_missing=False) is not None
                else None
            )
            for _, mus_meta in self.mint_dataset.metadata.iterrows()
        ]
        self.mint_dataset.metadata["babel_acts"] = [
            (
                self.babel_dataset.by_babel_sid(int(mus_meta["babel_sid"])).clip_actions()
                if self.babel_dataset.by_babel_sid(int(mus_meta["babel_sid"]), raise_missing=False) is not None
                else None
            )
            for _, mus_meta in self.mint_dataset.metadata.iterrows()
        ]

        logger.info("Total number of motions in mode {}: {}".format(self.mode, len(self.data)))

        if self.mode in ["val", "test"]:
            logger.info("Validation/Test mode, building valid start indices.")
            self.valid_start_indices = self.build_valid_start_indices()
            logger.info("Total number of valid start indices: {}".format(len(self.valid_start_indices)))
        else:
            self.valid_start_indices = None

    def build_valid_start_indices(self):
        valid_starts = []
        for idx, _ in enumerate(self.data):  # Use _ if motion is not used within the loop
            mint_data = self.mint_dataset[idx]
            try:
                muscle_df = mint_data.muscle_activations
                valid_timestamps = self.get_valid_timestamps_v2(muscle_df, window_size_seconds=self.window_size / self.motion_fps)

                # Ensure non-overlapping windows by only adding timestamps spaced by at least the window size
                last_ts = 0
                for ts in valid_timestamps:
                    if (
                        ts - last_ts >= self.window_size / self.motion_fps
                    ):  # Window size divided by motion fps gives the time duration of the window
                        valid_starts.append((idx, ts))
                        last_ts = ts

            except Exception as e:
                continue
        return valid_starts

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):

        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    

    def get_ordered_actions(self, actions):
        # Sort actions based on the start timestamp
        sorted_actions = sorted(actions, key=lambda x: x[0])

        ordered_actions = []
        for _, _, action_list in sorted_actions:
            # Extend the ordered list with actions from the current tuple
            if len(action_list) == 0:
                continue

            if isinstance(action_list[0], list):
                action_list = list(chain.from_iterable(action_list))

            ordered_actions.extend(action_list)

        return ordered_actions

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.valid_start_indices)
        else:
            return len(self.data)

    @ProfilerContext.profile_function(save_after=10000)
    def __getitem__(self, index, iters=0):

        if iters > 100:
            raise ValueError(f"Could not find any valid timestamp after {iters} resamplings")
        if self.mode in ["val", "test"]:
            sample_idx, window_start_timestamp = self.valid_start_indices[index]
        else:
            sample_idx, window_start_timestamp = index, None

        sample = {}

        meta_index = self.id_list[sample_idx]

        sample["name"] = meta_index

        mint_data = self.mint_dataset[sample_idx]

        try:
            muscle_df = mint_data.muscle_activations  #
        except BaseException as be:
            if self.mode == "test":
                raise ValueError("Could not open test sample {}".format(meta_index))

            logger.info("Could not open muscle data, sampling iteration {}".format(iters))

            new_index = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_index, iters=iters + 1)

        if window_start_timestamp is None:  # We are in train mode.
            window_start_timestamp = self.rand_valid_muscle_ts(muscle_df, max_frame_gap=self.max_frame_gap)

        if window_start_timestamp is None:  # At this point we need something.
            logger.info("Could not find any valid timestamp, sampling iteration {}".format(iters))
            new_index = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_index, iters=iters + 1)

        window_end_timestamp = window_start_timestamp + self.window_size * 1 / self.motion_fps
        window_end_timestamp = round(window_end_timestamp, 2)
        time_window = np.array((window_start_timestamp, window_end_timestamp))

        if self.muscle_subset is not None:
            muscle_df = muscle_df[self.muscle_subset]

        sample["muscle_activation"] = trim_mint_dataframe_v2(
            df=muscle_df,
            time_window=(window_start_timestamp, window_end_timestamp),
            target_frame_count=28,
        )

        # get corresponding babel meta from tran, val, test datasets if it exists, else None.
        babel_meta = self.babel_dataset.by_babel_sid(mint_data.babel_sid, raise_missing=False)
        filtered_actions = []
        if babel_meta is not None:
            filtered_actions = babel_meta.clip_actions_in_range(time_window[0], time_window[1])
        # filtered_actions = self.filter_actions_by_window(muscle_activation_meta["babel_acts"], time_window)
        sample["actions"] = filtered_actions

        ordered_actions = self.get_ordered_actions(filtered_actions)
        flattened_list = [word for sentence in ordered_actions for word in re.split(r"[\s\\]", sentence)]

        # get corresponding motion window
        motion = self.data[sample_idx]
        idx = int(window_start_timestamp * self.motion_fps)
        motion = motion[idx : idx + self.window_size]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        sample["motion"] = motion
        sample["time_start"] = idx / self.motion_fps
        sample["time_end"] = (idx + self.window_size) / self.motion_fps

        logger.debug(f"Start timestamp: {window_start_timestamp}, end timestamp: {window_end_timestamp}")
        logger.debug(f"Motion shape {sample['motion'].shape}, muscle shape: { sample['muscle_activation'].shape}")

        if np.isnan(sample["motion"]).any() is None or np.isnan(sample["muscle_activation"]).any() is None:
            logger.warning("Motion or muscle activation contains NaN, skipping {}".format(meta_index))
            new_index = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_index, iters=iters + 1)

        sample["unique_name"] = f"{sample['name']}_{sample['time_start']:3.2f}_{sample['time_end']:3.2f}"

        return sample

    def rand_valid_muscle_ts(self, muscle_activations: pd.DataFrame, max_frame_gap=1 / 8):
        """
        Returns a random valid timestamp from the given muscle activations DataFrame.

        Parameters:
        - muscle_activations (pd.DataFrame): DataFrame containing muscle activation data.

        Returns:
        - random_timestamp (str): Random valid timestamp from the DataFrame.
        - None if no valid ts exists
        """
        try:
            valid_timestamps = self.get_valid_timestamps_v2(muscle_activations, max_frame_gap=max_frame_gap, window_size_seconds=self.window_size / self.motion_fps)
        except ValueError:
            return None

        random_timestamp = random.choice(valid_timestamps)

        return random_timestamp

    def get_valid_timestamps(self, muscle_activations: pd.DataFrame, window_size_seconds=64 / 20, max_frame_gap=1 / 8):
        timestamps = muscle_activations.index.to_numpy()
        valid_start_times = []

        # Initialize the start index of the first block
        block_start_idx = 0
        for i in range(1, len(timestamps)):
            frame_interval = timestamps[i] - timestamps[i - 1]

            # Check if the current frame is outside the continuous block
            if frame_interval > max_frame_gap:
                # End of the current block, check if it can accommodate a 3.2-second window
                if timestamps[i - 1] - timestamps[block_start_idx] >= window_size_seconds:
                    # Add all valid start times within this block from the original timestamps
                    for start_time in timestamps[block_start_idx:i]:
                        if start_time + window_size_seconds <= timestamps[i - 1]:
                            valid_start_times.append(start_time)

                # Start a new block from the current frame
                block_start_idx = i

        # Check the last block
        if timestamps[-1] - timestamps[block_start_idx] >= window_size_seconds:
            for start_time in timestamps[block_start_idx:]:
                if start_time + window_size_seconds <= timestamps[-1]:
                    valid_start_times.append(start_time)

        if not valid_start_times:
            raise ValueError("No valid timestamps found.")

        return valid_start_times

    def get_valid_timestamps_v2(
        self, muscle_activations: pd.DataFrame, window_size_seconds=64 / 20, max_frame_gap=1 / 8
    ):
        timestamps = muscle_activations.index.to_numpy()
        frame_intervals = np.diff(timestamps, prepend=timestamps[0])

        # Identify the start of new blocks
        block_starts = np.where(frame_intervals > max_frame_gap)[0]
        block_starts = np.concatenate(([0], block_starts))
        block_ends = np.concatenate((block_starts[1:], [len(timestamps)])) - 1

        valid_start_times = []

        is_valid_block = timestamps[block_ends] - timestamps[block_starts] >= window_size_seconds
        valid_starts = block_starts[is_valid_block]
        valid_ends = block_ends[is_valid_block]

        for bs, be in zip(valid_starts, valid_ends):
            block_timestamps = timestamps[bs:be]

            valid_starts_in_block = block_timestamps[block_timestamps + window_size_seconds <= block_timestamps[-1]]
            valid_start_times.extend(valid_starts_in_block.tolist())

        if not valid_start_times:
            raise ValueError("No valid timestamps found.")

        return valid_start_times


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


from typing import List

from torch.utils.data._utils.collate import default_collate


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for a batch of dictionaries. For keys 'motion' and 'muscle_activation',
    it applies default collate behavior. For other keys, it concatenates values to lists without creating tensors.
    """
    collated = {}
    special_keys = ["motion", "muscle_activation"]  # These are always present.

    # Separate items for default collation and custom concatenation
    default_collation_items = {key: [] for key in special_keys}
    custom_concatenation_items = {}

    for item in batch:
        for key, value in item.items():
            # Handle special keys with default collate
            if key in special_keys:
                default_collation_items[key].append(value)
            else:
                # Custom concatenation for other keys
                if key not in custom_concatenation_items:
                    custom_concatenation_items[key] = []
                custom_concatenation_items[key].append(value)

    # Apply default collation to special keys
    for key, values in default_collation_items.items():
        collated[key] = default_collate(values)

    # Custom concatenation for other keys
    for key, values in custom_concatenation_items.items():
        collated[key] = values

    return collated


def DATALoader(
    dataset_name,
    batch_size,
    mode="train",
    num_workers=8,
    window_size=64,
    unit_length=4,
    label_required=False,
    use_profiling=False,
):

    def create_worker_init_fn(use_profiling):
        # Outer function defines a variable to be used in the inner function
        def worker_init_fn(worker_id):
            # Inner function has access to variables defined in the outer function
            ProfilerContext.is_profiling_active = use_profiling and worker_id == 0

        return worker_init_fn  # Return the inner function

    dataset = MSMotionDataset(
        dataset_name,
        mode=mode,
        window_size=window_size,
        unit_length=unit_length,
        label_required=label_required,
    )
    if mode == "train" or mode == "test":
        prob = dataset.compute_sampling_prob()
        sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples=len(dataset) * 10000, replacement=True)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            collate_fn=collate_fn,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=create_worker_init_fn(use_profiling),
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=16,
            collate_fn=collate_fn,
            drop_last=False,
            worker_init_fn=create_worker_init_fn(use_profiling),
        )

    return train_loader
