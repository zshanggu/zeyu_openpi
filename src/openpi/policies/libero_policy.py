import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Optional per-frame auxiliary classification labels (0/1), only present for
        # datasets converted with annotate_subskill_boundaries.py's --label-phases and
        # convert_libero_subskills_to_lerobot.py's subskill_done/phase features -- absent
        # (and silently skipped) for any other dataset, e.g. the public libero release.
        if "phase" in data:
            inputs["phase_label"] = np.asarray(data["phase"]).squeeze().astype(np.int32)
        if "subskill_done" in data:
            inputs["done_label"] = np.asarray(data["subskill_done"]).squeeze().astype(np.int32)

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # Set for a model/data config pair with aux_signal_mode/subskill_aux_mode=="action_dim" --
    # the sampled "actions" chunk is then 9-dim (7 real + phase + done, see
    # LeRobotLiberoDataConfig.subskill_aux_mode), and this splits the extra 2 back out into
    # phase_prob/done_prob (named to match the classification_head mode's aux output keys),
    # instead of silently sending them to the robot as extra motor commands.
    has_action_dim_aux: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"])
        outputs = {"actions": actions[..., :7]}
        if self.has_action_dim_aux:
            # `actions` here is the full (action_horizon, model_action_dim) chunk (the caller
            # already squeezed the batch dim -- see policy.py's/attention_policy.py's own
            # `x[0, ...]`), so actions[..., 7]/[..., 8] are themselves still one whole
            # action_horizon-length sequence, not a scalar -- unlike the classification_head
            # mode's phase_prob/done_prob, which are a single pooled judgment about the CURRENT
            # frame. Take just the first (soonest) predicted step so the wire contract stays a
            # single float either way, and cast off numpy's scalar type (float32 isn't JSON
            # serializable; plain Python float is). Raw regressed values, not
            # sigmoid-calibrated probabilities -- trained directly against hard 0/1 targets via
            # the flow-matching velocity loss, so they cluster near 0/1 but aren't a true
            # probability. Named the same as the classification_head mode's keys for a common
            # consumer-side interface.
            outputs["phase_prob"] = float(actions[0, 7])
            outputs["done_prob"] = float(actions[0, 8])
        return outputs
