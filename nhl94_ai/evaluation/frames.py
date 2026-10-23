from typing import Tuple
import numpy as np

def get_env_frame(env) -> np.ndarray:
    frame = None

    # First try the vectorized render method (works for most VecEnvs)
    try:
        frame = env.render(mode="rgb_array")
        if isinstance(frame, list):
            frame = next((img for img in frame if img is not None), None)
    except TypeError:
        frame = None

    # Some VecEnvs expose get_images instead of render returns
    if frame is None and hasattr(env, "get_images"):
        try:
            images = env.get_images()
            if images:
                frame = next((img for img in images if img is not None), None)
        except (NotImplementedError, TypeError):
            frame = None

    # Fallback to env_method without additional kwargs
    if frame is None:
        try:
            frames = env.env_method("render")
            if isinstance(frames, list):
                frame = next((img for img in frames if img is not None), None)
            else:
                frame = frames
        except (TypeError, AttributeError):
            frame = None

    if frame is None:
        raise RuntimeError("Environment did not return an rgb_array frame. Make sure the env supports rendering.")

    return frame

def prepare_action_info(probabilities, actions, button_count: int) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float).flatten()
    acts = np.asarray(actions, dtype=float).flatten()

    if probs.size < button_count:
        if probs.size == 0:
            probs = np.zeros(button_count, dtype=float)
        else:
            probs = np.pad(probs, (0, button_count - probs.size), constant_values=0.0)
    elif probs.size > button_count:
        probs = probs[:button_count]

    if acts.size < button_count:
        if acts.size == 0:
            acts = np.zeros(button_count, dtype=float)
        else:
            acts = np.pad(acts, (0, button_count - acts.size), constant_values=0.0)
    elif acts.size > button_count:
        acts = acts[:button_count]

    return probs, acts
