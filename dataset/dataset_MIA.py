import ast
import cProfile
import logging
import os
import random
import re
from itertools import chain
from os.path import join as pjoin
from typing import Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from musint.benchmarks.muscle_sets import MUSCLE_SUBSETS
from musint.datasets.mint_dataset import MintDataset
from musint.utils.dataframe_utils import frame_to_time, time_to_frame, trim_mint_dataframe, trim_mint_dataframe_v2

from scipy.interpolate import interp1d
from torch.utils import data
from tqdm import tqdm

from utils.profiler_utils import ProfilerContext

from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure basic logging
logging.basicConfig(level=logging.INFO)
# Get the root logger
logger = logging.getLogger()

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

DO_PROFILING = False


MIA_ACTION_ENCODING = {
    "ElbowPunch": 0,
    "FrontKick": 1,
    "FrontPunch": 2,
    "HighKick": 3,
    "HookPunch": 4,
    "JumpingJack": 5,
    "KneeKick": 6,
    "LegBack": 7,
    "LegCross": 8,
    "RonddeJambe": 9,
    "Running": 10,
    "Shuffle": 11,
    "SideLunges": 12,
    "SlowSkater": 13,
    "Squat": 14,
}

MIA_MUSCLES = [
    "quadriceps_femoris_l",
    "hamstring_l",
    "lateral_l",
    "biceps_l",
    "quadriceps_femoris_r",
    "hamstring_r",
    "lateral_r",
    "biceps_r",
]

MIA_MUSCLE_ENDCODING = {muscle: i for i, muscle in enumerate(MIA_MUSCLES)}
MIA_MUSCLE_DECODING = {i: muscle for i, muscle in enumerate(MIA_MUSCLES)}


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


class MIAData:
    """
    A class to represent the muscle activations of a sample from the MIA dataset
    """

    def __init__(
        self,
        path_id: str,  # e.g. "train/Subject5/ElbowPunch/972"
        dataset_path: str,
        pyarrow: bool = False,
        fps=10,
        lazy_loading=True,
        **kwargs,
    ):
        self.path_id = path_id
        self.dataset_path = dataset_path
        self.full_data_path = pjoin(dataset_path, self.path_id, "emgvalues.npy")

        pattern = re.compile(r"(train|val)/Subject(\d+)/(\w+)/(\d+)")
        match = pattern.match(self.path_id)

        if not match:
            raise ValueError(f"Could not match {self.path_id} against regex.")

        self.split = match.group(1)
        self.subject = int(match.group(2))
        self.action = match.group(3)
        self.recording = int(match.group(4))

        self.fps = fps

        self.pyarrow = pyarrow
        self.lazy_loading = lazy_loading

        self.start_time = 0.0
        self._muscle_activations = None
        self._end_time = None
        self._num_frames = None

        if not lazy_loading:
            self._load_muscle_activations()

    def _load_muscle_activations(self):
        self._muscle_activations = np.load(self.full_data_path)  # shape (30,8)

        self._end_time = self._muscle_activations.shape[0] / 10.0  # All the data is recorded with 10 fps.
        num_frames = self._muscle_activations.shape[0]

        if self.fps != 10:  # All the data is recorded with 10 fps.
            self._muscle_activations, num_frames = self._interpolate_activations(self.fps)

        self._muscle_activations = pd.DataFrame(self._muscle_activations, columns=MIA_MUSCLES)

        muscle_activations_index = pd.Index(np.linspace(self.start_time, self.end_time, num_frames), name="Time")
        self._muscle_activations.index = muscle_activations_index

        if self.pyarrow:
            self._muscle_activations = self._muscle_activations.convert_dtypes(dtype_backend="pyarrow")

    @property
    def muscle_activations(self):
        if self._muscle_activations is None:
            self._load_muscle_activations()
        return self._muscle_activations

    @property
    def end_time(self):
        if self._muscle_activations is None:
            self._load_muscle_activations()
        return self._end_time

    @property
    def shape(self):
        if self._muscle_activations is None:
            self._load_muscle_activations()
        return self._muscle_activations.shape

    def __len__(self):
        if self._muscle_activations is None:
            self._load_muscle_activations()
        return self._muscle_activations.shape[0]

    def _interpolate_activations(self, fps):
        original_times = np.linspace(self.start_time, self.end_time, len(self.muscle_activations))
        new_num_frames = int(self.end_time * fps)
        new_times = np.linspace(self.start_time, self.end_time, new_num_frames)

        interpolator = interp1d(original_times, self.muscle_activations, axis=0, kind="linear")
        muscle_activations = interpolator(new_times)
        num_frames = new_num_frames

        return muscle_activations, num_frames

    def get_muscle_activations(
        self,
        time_window: Tuple[float, float] = None,
        target_fps=20,
        rolling_average=False,
        target_frame_count=None,
        as_numpy=False,
    ):
        """
        Resample muscle activations to the given time window and fps. Returns the values as a numpy array or a dataframe

        Parameters:
        time_window (Tuple[float, float]): The start and end times for the window
        target_fps (float): The target frames per second for resampling
        rolling_average (bool): Whether to apply a rolling average

        Returns:
        np.ndarray/pd.DataFrame: The resampled values as a numpy array or a dataframe
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        return trim_mint_dataframe_v2(
            self.muscle_activations,
            time_window,
            target_frame_count,
            as_numpy,
        )

    def get_valid_indices(self, time_window: Tuple[float, float] = None, target_fps=20.0, as_time=True):
        """
        Gets the valid indices of the muscle activations given the frame indices.
        Returns the indices as frames or time corresponding to the target fps.

        Returns:
        np.ndarray: The valid indices as frames or time
        """
        if time_window is None:
            time_window = (self.start_time, self.end_time)

        trimmed_muscle_activation_index = self.muscle_activations.index[
            self.muscle_activations.index.to_series().between(time_window[0], time_window[1])
        ]

        frame_indices = np.round(trimmed_muscle_activation_index * target_fps, 0).astype(int).unique()

        if as_time:
            frame_times = frame_indices / target_fps
            return np.round(frame_times, 2)
        else:
            return frame_indices

    def get_gaps(self, as_frame=False, target_fps=20.0):
        """
        Gets all the pairs of indices before and after a gap in the muscle activations
        """
        # resample index of the muscle activations to fps
        muscle_activation_index = self.muscle_activations.index
        muscle_activation_index = np.round(muscle_activation_index * self.fps).astype(int)

        differences = pd.Series(muscle_activation_index).diff()

        normal_difference = 1.01
        # Get the indices of the differences that are larger than 0.02s
        gap_indices = muscle_activation_index[differences > normal_difference]

        gap_tuples = []
        for gap_index in gap_indices:
            pos2 = gap_index
            time = frame_to_time(gap_index, self.fps)
            previous_frame = self.muscle_activations.index.get_loc(frame_to_time(pos2, self.fps)) - 1
            previous_time = self.muscle_activations.index[previous_frame]
            if as_frame:
                gap_tuples.append(
                    (
                        time_to_frame(previous_time, target_fps),
                        time_to_frame(time, target_fps),
                    )
                )
            else:
                gap_tuples.append((previous_time, time))

        return gap_tuples

    def __repr__(self):
        attributes = [
            "path_id",
            "split",
            "subject",
            "action",
            "recording",
            "num_frames",
            "fps",
        ]
        info = [f"{attr}={getattr(self, attr)!r}" for attr in attributes]
        return f"{self.__class__.__name__}({', '.join(info)})"


class MotionData:
    def __init__(self, path_id: str, dataset_path: str, lazy_loading=True, fps=20, force_frames=60):

        self.path_id = path_id
        self.dataset_path = dataset_path

        self.full_data_path = pjoin(dataset_path, "motion_vects", path_id + ".npy")

        assert fps == 20, "Currently only fixed 20 fps."

        self.force_frames = force_frames

        self.lazy_loading = lazy_loading

        self._motion_data = None

        if not lazy_loading:
            self._load_motion_data()

    def _load_motion(self):
        self._motion_data = np.load(self.full_data_path)  # shape (58, 263)

        if self.force_frames:
            self._motion_data = self.interpolate_motion(self.force_frames)

    def interpolate_motion(self, target_num_frames):
        if self._motion_data is None:
            self._load_motion()

        original_num_frames = self._motion_data.shape[0]
        original_times = np.linspace(0, 1, original_num_frames)
        target_times = np.linspace(0, 1, target_num_frames)

        interpolator = interp1d(original_times, self._motion_data, axis=0, kind="linear")
        interpolated_motion = interpolator(target_times)

        return interpolated_motion

    @property
    def motion(self):
        if self._motion_data is None:
            self._load_motion()
        return self._motion_data

    @property
    def shape(self):
        if self._motion_data is None:
            self._load_motion()
        return self._motion_data.shape

    def __len__(self):
        if self._motion_data is None:
            self._load_motion()
        return self._motion_data.shape[0]


class MIADataset(data.Dataset):

    @ProfilerContext.profile_function()
    def __init__(
        self,
        dataset_name="MIA",
        mode="train",
        window_size=60,
        fps=20,
        muscle_subset=None,  # e.g. "LAI_ARNOLD_LOWER_BODY_8", defined in musint.benchmarks.muscle_sets
        mia_root="$MIA_DATA",
        mia_motion_root="$MIA_MOTION_DATA",
        lazy_loading=True,
        load_parallel=False,
        data_ratio=1.0,
    ):
        self.dataset_name = dataset_name

        self.muscle_root = os.path.expandvars(mia_root)
        self.motion_root = os.path.expandvars(mia_motion_root)

        print(f"Motion root: {self.motion_root}")

        assert self.muscle_root != "$MIA_DATA"
        assert self.motion_root != "$MIA_MOTION_DATA"

        self.lazy_loading = lazy_loading
        self.window_size = window_size

        self.fps = fps
        self.mode = mode

        self.data_ratio = data_ratio

        self.muscle_subset = MUSCLE_SUBSETS[muscle_subset] if muscle_subset is not None else None
        self.max_frame_gap = 1 / 8  # 0.125 seconds

        self.mean_motion = np.load(os.path.expandvars(pjoin(self.motion_root, "Mean.npy")))
        self.std_motion = np.load(os.path.expandvars(pjoin(self.motion_root, "Std.npy")))

        self.mean_muscle = np.load(os.path.expandvars(pjoin(self.muscle_root, "Mean.npy")))
        self.std_muscle = np.load(os.path.expandvars(pjoin(self.muscle_root, "Std.npy")))
        self.pct99_muscle = np.load(os.path.expandvars(pjoin(self.muscle_root, "99pct.npy")))

        with open(pjoin(self.muscle_root, "train.txt"), "r") as f:
            self.train_split_ids = [line.strip() for line in f]

        with open(pjoin(self.muscle_root, "val.txt"), "r") as f:
            self.val_split_ids = [line.strip() for line in f]

        # Load Dataset
        split_ids = self.val_split_ids if mode == "val" else self.train_split_ids

        mia_kwargs = {"dataset_path": self.muscle_root, "lazy_loading": self.lazy_loading, "fps": 20}
        motion_kwargs = {"dataset_path": self.motion_root, "lazy_loading": self.lazy_loading, "fps": 20}

        if load_parallel:
            self.mia_data = self._load_data_parallel(split_ids=split_ids, load_func=self._load_mia_data)
            self.motion_data = self._load_data_parallel(split_ids=split_ids, load_func=self._load_motion_data)
        else:
            self.mia_data = [MIAData(path_id=pid, **mia_kwargs) for i, pid in tqdm(enumerate(split_ids))]
            self.motion_data = [MotionData(path_id=pid, **motion_kwargs) for i, pid in tqdm(enumerate(split_ids))]

        print(self.mia_data[0].shape)
        print(self.motion_data[0].shape)
        logger.info("Total number of motions in mode {}: {}".format(self.mode, len(self.motion_data)))

        if self.mode in ["val", "test"]:
            logger.info("Validation/Test mode, building valid start indices.")
            self.valid_start_indices = self.build_valid_start_indices()
            logger.info("Total number of valid start indices: {}".format(len(self.valid_start_indices)))
        else:  # Train mode
            if data_ratio < 1.0:
                logger.info(f"Subsampling data with ratio {data_ratio}")

                # Group indices by action category
                category_indices = defaultdict(list)
                for idx, data in enumerate(self.mia_data):
                    category_indices[data.action].append(idx)

                # Subsample within each category
                subsampled_indices = []
                for indices in category_indices.values():
                    sample_size = int(data_ratio * len(indices))
                    subsampled_indices.extend(random.sample(indices, sample_size))

                subsampled_indices = sorted(subsampled_indices)

                self.mia_data = [self.mia_data[i] for i in subsampled_indices]
                self.motion_data = [self.motion_data[i] for i in subsampled_indices]

                logger.info(f"Subsampled data, new number of samples: {len(self.mia_data)}")

            self.valid_start_indices = None

        self.sample_ids = []

        for ms, mo in zip(self.mia_data, self.motion_data):
            assert ms.path_id == mo.path_id
            self.sample_ids.append(ms.path_id)

    def _load_mia_data(self, pid):
        return MIAData(path_id=pid, dataset_path=self.muscle_root, lazy_loading=self.lazy_loading, fps=self.fps)

    def _load_motion_data(self, pid):
        return MotionData(path_id=pid, dataset_path=self.motion_root, lazy_loading=self.lazy_loading, fps=self.fps)

    def _load_data_parallel(self, split_ids, load_func):
        data = {}
        with ThreadPoolExecutor() as executor:
            future_to_id = {executor.submit(load_func, pid): i for i, pid in enumerate(split_ids)}
            for future in tqdm(as_completed(future_to_id), total=len(split_ids)):
                i = future_to_id[future]

                data = future.result()

                data[i] = data
        return data

    def build_valid_start_indices(self):
        valid_starts = []
        for idx, _ in enumerate(self.mia_data):  # Use _ if motion is not used within the loop
            mia_data: MIAData = self.mia_data[idx]
            try:
                muscle_df = mia_data.muscle_activations
                valid_timestamps = self.get_valid_timestamps_v2(
                    muscle_df, window_size_seconds=self.window_size / self.fps
                )

                # Ensure non-overlapping windows by only adding timestamps spaced by at least the window size
                last_ts = -np.inf
                for ts in valid_timestamps:
                    if (
                        ts - last_ts >= self.window_size / self.fps
                    ):  # Window size divided by motion fps gives the time duration of the window
                        valid_starts.append((idx, ts))
                        last_ts = ts

            except Exception as e:
                print(e)
                continue
        return valid_starts

    def inv_motion_transform(self, motion):
        return motion * self.std_motion + self.mean_motion

    def inv_muscle_transform(self, muscle):
        return muscle * self.std_muscle + self.mean_muscle

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.valid_start_indices)
        else:
            return len(self.mia_data)

    @ProfilerContext.profile_function(save_after=10000)
    def __getitem__(self, index, iters=0):

        if iters > 100:
            raise ValueError(f"Could not find any valid timestamp after {iters} resamplings")
        if self.mode in ["val", "test"]:
            sample_idx, window_start_timestamp = self.valid_start_indices[index]
        else:
            sample_idx, window_start_timestamp = index, None

        mia_sample: MIAData = self.mia_data[sample_idx]
        motion_sample: MotionData = self.motion_data[sample_idx]

        sample = {"name": mia_sample.path_id}

        try:
            muscle_df = mia_sample.muscle_activations  #
        except BaseException as be:
            if self.mode == "test":
                raise ValueError("Could not open test sample {}".format(mia_sample.path_id))

            logger.info("Could not open muscle data, sampling iteration {}".format(iters))

            new_index = random.randint(0, len(self.mia_data) - 1)
            return self.__getitem__(new_index, iters=iters + 1)

        if window_start_timestamp is None:  # We are in train mode.
            window_start_timestamp = self.rand_valid_muscle_ts(muscle_df, max_frame_gap=self.max_frame_gap)

        if window_start_timestamp is None:  # At this point we need something.
            logger.info("Could not find any valid timestamp, sampling iteration {}".format(iters))
            new_index = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_index, iters=iters + 1)

        window_end_timestamp = window_start_timestamp + self.window_size * 1 / self.fps
        window_end_timestamp = round(window_end_timestamp, 2)
        time_window = np.array((window_start_timestamp, window_end_timestamp))

        if self.muscle_subset is not None:
            muscle_df = muscle_df[self.muscle_subset]

        sample["muscle_activation"] = trim_mint_dataframe_v2(
            df=muscle_df,
            time_window=(window_start_timestamp, window_end_timestamp),
            target_frame_count=self.window_size,
        )

        # We scale the 99th percentile to 1
        sample["muscle_activation"] = sample["muscle_activation"] / self.pct99_muscle

        sample["actions"] = [mia_sample.action]

        # get corresponding motion window
        idx = int(window_start_timestamp * self.fps)
        motion = motion_sample.motion[idx : idx + self.window_size]

        "Z Normalization"
        motion = (motion - self.mean_motion) / self.std_motion

        sample["motion"] = motion
        sample["time_start"] = idx / self.fps
        sample["time_end"] = (idx + self.window_size) / self.fps

        assert sample["motion"].shape[0] == self.window_size
        assert sample["muscle_activation"].shape[0] == self.window_size

        logger.debug(f"Start timestamp: {window_start_timestamp}, end timestamp: {window_end_timestamp}")
        logger.debug(f"Motion shape {sample['motion'].shape}, muscle shape: { sample['muscle_activation'].shape}")

        if np.isnan(sample["motion"]).any() is None or np.isnan(sample["muscle_activation"]).any() is None:
            logger.warning("Motion or muscle activation contains NaN, skipping {}".format(mia_sample.path_id))
            new_index = random.randint(0, len(self.mia_data) - 1)
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
            valid_timestamps = self.get_valid_timestamps_v2(
                muscle_activations, max_frame_gap=max_frame_gap, window_size_seconds=self.window_size / self.fps
            )
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

            valid_starts_in_block = block_timestamps[
                block_timestamps + window_size_seconds <= block_timestamps[-1]
            ]  # We allow 100 ms mismatch
            valid_start_times.extend(valid_starts_in_block.tolist())

        if not valid_start_times:
            raise ValueError("No valid timestamps found.")

        return valid_start_times


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


from typing import List, Tuple

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


def MIADATALoader(dataset_name, batch_size, mode="train", num_workers=8, window_size=60, **kwargs):

    dataset = MIADataset(dataset_name=dataset_name, mode=mode, window_size=window_size, **kwargs)
    if mode == "train":
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=len(dataset) * 10000)
    elif mode == "val":
        sampler = torch.utils.data.SequentialSampler(dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=sampler,
    )

    return train_loader


if __name__ == "__main__":
    os.environ["MIA_DATA"] = "/lsdf/data/activity/MIADatasetOfficial"
    os.environ["MIA_MOTION_DATA"] = "/lsdf/users/dschneider-kf3609/workspace/HumanML3D/MIAHML3D"

    dataset = MIADataset(mode="train", window_size=60)

    print(dataset[0])

    print("Dataset length: ", len(dataset))

    dataset_val = MIADataset(mode="val", window_size=60)

    print(dataset_val[0])
    print("Validation Dataset length: ", len(dataset_val))

    train_loader = MIADATALoader("MIA", 8, mode="train", num_workers=8, window_size=60)
    print(next(iter(train_loader)))

    val_loader = MIADATALoader("MIA", 8, mode="val", num_workers=8, window_size=60)
    print(next(iter(val_loader)))
