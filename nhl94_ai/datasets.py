"""Read and write existing demonstration datasets and their metadata."""
import glob
import json
import os
import numpy as np


def save_demo_shard(path, arrays, metadata):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    manifest_path = f"{path}.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path


def discover_npz_files(paths):
    files = []
    for path in paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            files.extend(sorted(glob.glob(os.path.join(expanded, "*.npz"))))
        else:
            matches = sorted(glob.glob(expanded))
            files.extend(matches if matches else [expanded])
    files = [path for path in files if path.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No .npz dataset shards found in: {paths}")
    missing = [path for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Dataset shard not found: {missing[0]}")
    return files


def load_demo_arrays(paths, expected=None):
    observations = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []

    next_episode_offset = 0
    for path in discover_npz_files(paths):
        from nhl94_ai.artifacts import read_metadata, validate_schema
        metadata = read_metadata(path)
        if expected is not None and metadata.get('compatibility') != 'unknown':
            sample_expected = expected
            if (metadata.get('sample_interval_frames') == 1
                    and metadata.get('teacher_decision_interval') == expected['schema']['frame_skip']):
                # Reactive teachers expose the actual per-frame labels within
                # the requested policy interval. Checkpoint timing stays strict.
                sample_expected = dict(expected, schema=dict(expected['schema'], frame_skip=1))
            validate_schema(metadata, sample_expected)
        with np.load(path, allow_pickle=False) as data:
            observations.append(np.asarray(data["observations"], dtype=np.float32))
            actions.append(np.asarray(data["actions"], dtype=np.float32))
            if "rewards" in data:
                rewards.append(np.asarray(data["rewards"], dtype=np.float32))
            if "dones" in data:
                dones.append(np.asarray(data["dones"], dtype=np.bool_))
            if "episode_ids" in data:
                shard_episode_ids = np.asarray(data["episode_ids"], dtype=np.int64) + next_episode_offset
                episode_ids.append(shard_episode_ids)
                next_episode_offset = int(shard_episode_ids.max(initial=-1)) + 1

    payload = {
        "observations": np.concatenate(observations, axis=0),
        "actions": np.concatenate(actions, axis=0),
    }
    if rewards:
        payload["rewards"] = np.concatenate(rewards, axis=0)
    if dones:
        payload["dones"] = np.concatenate(dones, axis=0)
    if episode_ids:
        payload["episode_ids"] = np.concatenate(episode_ids, axis=0)
    else:
        payload["episode_ids"] = np.zeros(payload["actions"].shape[0], dtype=np.int64)
    return payload
