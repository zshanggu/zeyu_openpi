"""Annotate, for every demo in a set of LIBERO HDF5 files, which frames belong
to which sub-skill. Two selectable --task-set configs:

  - "libero_10" (default): the 10 LIBERO-10 long-horizon tasks, each split
    into the 2-4 sub-skills listed in libero_100/libero_10/subskill_prompts.txt.
  - "libero_90_lilo22": LiLo-VLA's 22-skill atomic library (Table III of
    https://yy-gx.github.io/LiLo-VLA/static/pdfs/appendix.pdf), mapped onto 13
    real LIBERO-90 HDF5 files. LIBERO-90's tasks are already short/atomic
    (mostly one pick + one place; a few are a single open/close/turn-on), so
    unlike libero_10 none of these need a "transport" padding skill -- a file
    just gets as many sub-skills as it genuinely has.

This does NOT cut/write any new dataset -- it only records frame-index
boundaries (plus the sub-skill prompts) into one JSON file per task. Actually
segmenting the episodes (e.g. for a LeRobot-format conversion where each
sub-skill becomes its own labeled episode) is a later step that consumes this
JSON (see convert_libero_subskills_to_lerobot.py).

Boundaries are found the same way LIBERO's own evaluation code decides task
state -- not a hand-rolled visual/geometric heuristic:
  - "place"/"open"/"close"/"turn on" boundaries reuse the exact BDDL predicate
    classes (On/In/Open/Close/TurnOn from
    third_party/libero/libero/libero/envs/predicates/base_predicates.py) that
    libero's own `_check_success()` evaluates every step -- see
    Libero_Kitchen_Tabletop_Manipulation._check_success() and _eval_predicate()
    for the reference. A sub-skill's completion frame is simply the first
    frame at which its predicate turns True.
  - "pick up X" boundaries reuse robosuite's own `_check_grasp(gripper,
    object_geoms)` (the same utility robosuite's built-in tasks, e.g. Lift,
    Stack, PickPlace, use for their own reward shaping/success checks) plus a
    simple height-rise check (object must be lifted off whatever it's resting
    on, not just touched), since grasping has no BDDL predicate of its own.
  - Two tasks only have one natural pick-place pair in their goal (no second
    object, no open/close container), so a 4th "transport" sub-skill was
    added when the prompts were written (see subskill_prompts.txt). It has no
    predicate of its own either; its start is approximated as the first
    frame after the pick where the gripper begins to re-open (a "release
    onset" heuristic on the recorded `obs/gripper_states`), since that's the
    natural moment transport ends and placing begins.

Replays each demo exactly the way third_party/libero/scripts/create_dataset.py
does: reset the env with that demo's own randomized `model_file` XML, restore
the recorded initial `states[0]`, then step through the recorded `actions`
sequentially (not teleporting state every frame) so the live MuJoCo state
(contacts, joint qpos, body positions) stays physically consistent for the
predicate/grasp checks. Runs with no renderer at all (no images needed here),
so this doesn't need EGL/GLX or a GPU -- just the same Python 3.8 + LIBERO
environment examples/libero/main.py runs in.

Usage (inside the LIBERO env/Docker image -- see examples/libero/README.md):
    python examples/libero/annotate_subskill_boundaries.py \
        --data-dir /path/to/libero_100/libero_10 \
        --out /path/to/libero_100/libero_10/subskill_boundaries.json

    # LiLo-VLA's 22-skill library, on LIBERO-90 instead:
    python examples/libero/annotate_subskill_boundaries.py \
        --task-set libero_90_lilo22 \
        --data-dir /path/to/libero_100/libero_90 \
        --out /path/to/libero_100/libero_90/subskill_boundaries.json

    # Quick smoke test on a couple of demos before running the full batch:
    python examples/libero/annotate_subskill_boundaries.py \
        --data-dir /path/to/libero_100/libero_10 \
        --out /tmp/smoke.json --task-filter KITCHEN_SCENE3 --max-demos 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import typing

import h5py
import numpy as np

LIFT_THRESHOLD_M = 0.015  # object must rise this much above its resting height to count as "picked up"
# Gripper must re-open by this much (sum of |finger qpos|) to count as
# "releasing". Calibrated against real failed detections on KITCHEN_SCENE3 and
# STUDY_SCENE1: 0.008 missed several demos whose actual release delta was
# 0.004-0.007 (thin/small objects like the book need less finger travel to
# release than a bulky mug or pot), while the holding-phase drift observed in
# those same demos stayed under ~0.002 -- so 0.004 recovers the real signal
# with still-comfortable margin above that noise floor. A genuine remainder
# (recordings that end before any visible release, rise < 0.002) can't be
# fixed by any threshold and correctly falls back to detection_failed=True.
RELEASE_DELTA_M = 0.004

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BDDL_FILES_ROOT = REPO_ROOT / "third_party/libero/libero/libero/bddl_files"


@dataclasses.dataclass
class GraspEvent:
    obj: str


@dataclasses.dataclass
class ContactEvent:
    obj: str


@dataclasses.dataclass
class PredicateEvent:
    name: str
    args: tuple[str, ...]


@dataclasses.dataclass
class ReleaseOnsetEvent:
    after_skill_idx: int  # search for release onset strictly after this skill's completion frame


# Not "GraspEvent | ContactEvent | ..." -- that's a runtime expression, not a
# lazy annotation, and Python 3.8 (this script's target env) doesn't support
# `|` between classes (PEP 604 is 3.10+). typing.Union works on 3.8.
Event = typing.Union[GraspEvent, ContactEvent, PredicateEvent, ReleaseOnsetEvent]


@dataclasses.dataclass
class SkillSpec:
    prompt: str
    event: Event


@dataclasses.dataclass
class TaskConfig:
    hdf5_name: str
    bddl_name: str
    skills: list[SkillSpec]
    aux_events: list[tuple[str, Event]] = dataclasses.field(default_factory=list)


LIBERO_10_CONFIGS: list[TaskConfig] = [
    TaskConfig(
        hdf5_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        bddl_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
        skills=[
            SkillSpec("turn on the stove", PredicateEvent("turnon", ("flat_stove_1",))),
            SkillSpec("pick up the moka pot", GraspEvent("moka_pot_1")),
            SkillSpec("move the moka pot to the stove", ReleaseOnsetEvent(after_skill_idx=2)),
            SkillSpec("place the moka pot on the stove", PredicateEvent("on", ("moka_pot_1", "flat_stove_1_cook_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl",
        skills=[
            SkillSpec("open the bottom drawer of the cabinet", PredicateEvent("open", ("white_cabinet_1_bottom_region",))),
            SkillSpec("pick up the black bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec(
                "place the black bowl in the bottom drawer of the cabinet",
                PredicateEvent("in", ("akita_black_bowl_1", "white_cabinet_1_bottom_region")),
            ),
            SkillSpec("close the bottom drawer of the cabinet", PredicateEvent("close", ("white_cabinet_1_bottom_region",))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        bddl_name="KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl",
        skills=[
            SkillSpec("open the microwave", PredicateEvent("open", ("microwave_1",))),
            SkillSpec("pick up the yellow and white mug", GraspEvent("white_yellow_mug_1")),
            SkillSpec(
                "place the yellow and white mug in the microwave",
                PredicateEvent("in", ("white_yellow_mug_1", "microwave_1_heating_region")),
            ),
            SkillSpec("close the microwave", PredicateEvent("close", ("microwave_1",))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
        bddl_name="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
        # "First"/"second" here means temporal order in the demos, not BDDL
        # object-id order -- checked directly against real demo trajectories
        # (moka_pot_2 is consistently picked up before moka_pot_1 is even
        # touched, confirmed via each object's height-rise trace).
        skills=[
            SkillSpec("pick up the first moka pot", GraspEvent("moka_pot_2")),
            SkillSpec("place the first moka pot on the stove", PredicateEvent("on", ("moka_pot_2", "flat_stove_1_cook_region"))),
            SkillSpec("pick up the second moka pot", GraspEvent("moka_pot_1")),
            SkillSpec("place the second moka pot on the stove", PredicateEvent("on", ("moka_pot_1", "flat_stove_1_cook_region"))),
        ],
        # Not a language-instruction word, but the BDDL goal ALSO requires the
        # stove to be on -- flagged as an auxiliary event (see module
        # docstring / README) rather than silently dropped or forced into one
        # of the 4 fixed labeled sub-skills.
        aux_events=[("turnon(flat_stove_1)", PredicateEvent("turnon", ("flat_stove_1",)))],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl",
        skills=[
            SkillSpec("pick up the alphabet soup", GraspEvent("alphabet_soup_1")),
            SkillSpec("place the alphabet soup in the basket", PredicateEvent("in", ("alphabet_soup_1", "basket_1_contain_region"))),
            SkillSpec("pick up the cream cheese box", GraspEvent("cream_cheese_1")),
            SkillSpec("place the cream cheese box in the basket", PredicateEvent("in", ("cream_cheese_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl",
        skills=[
            SkillSpec("pick up the alphabet soup", GraspEvent("alphabet_soup_1")),
            SkillSpec("place the alphabet soup in the basket", PredicateEvent("in", ("alphabet_soup_1", "basket_1_contain_region"))),
            SkillSpec("pick up the tomato sauce", GraspEvent("tomato_sauce_1")),
            SkillSpec("place the tomato sauce in the basket", PredicateEvent("in", ("tomato_sauce_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl",
        skills=[
            SkillSpec("pick up the cream cheese box", GraspEvent("cream_cheese_1")),
            SkillSpec("place the cream cheese box in the basket", PredicateEvent("in", ("cream_cheese_1", "basket_1_contain_region"))),
            SkillSpec("pick up the butter", GraspEvent("butter_1")),
            SkillSpec("place the butter in the basket", PredicateEvent("in", ("butter_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        skills=[
            SkillSpec("pick up the white mug", GraspEvent("porcelain_mug_1")),
            SkillSpec("place the white mug on the left plate", PredicateEvent("on", ("porcelain_mug_1", "plate_1"))),
            SkillSpec("pick up the yellow and white mug", GraspEvent("white_yellow_mug_1")),
            SkillSpec("place the yellow and white mug on the right plate", PredicateEvent("on", ("white_yellow_mug_1", "plate_2"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
        skills=[
            SkillSpec("pick up the white mug", GraspEvent("porcelain_mug_1")),
            SkillSpec("place the white mug on the plate", PredicateEvent("on", ("porcelain_mug_1", "plate_1"))),
            SkillSpec("pick up the chocolate pudding", GraspEvent("chocolate_pudding_1")),
            SkillSpec(
                "place the chocolate pudding to the right of the plate",
                PredicateEvent("on", ("chocolate_pudding_1", "living_room_table_plate_right_region")),
            ),
        ],
    ),
    TaskConfig(
        hdf5_name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
        bddl_name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl",
        skills=[
            SkillSpec("reach for the book", ContactEvent("black_book_1")),
            SkillSpec("pick up the book", GraspEvent("black_book_1")),
            SkillSpec("move the book to the caddy", ReleaseOnsetEvent(after_skill_idx=2)),
            SkillSpec("place the book in the back compartment of the caddy", PredicateEvent("in", ("black_book_1", "desk_caddy_1_back_contain_region"))),
        ],
    ),
]


# LiLo-VLA's 22-skill atomic library (Table III of
# https://yy-gx.github.io/LiLo-VLA/static/pdfs/appendix.pdf), mapped onto
# real LIBERO-90 HDF5 files rather than LIBERO-10: LIBERO-90's tasks are
# already short/atomic (usually one pick + one place), so most of these need
# no "transport" padding skill the way two LIBERO-10 tasks did -- a file
# simply has as many skills as it genuinely has (1 for a bare open/close/
# turn-on task, 2 for a pick-place one). "Pick Black Bowl" (the paper's S7)
# is reused across three different source files, matching the paper's own
# framing of it as one shared atomic skill with several possible followups
# (place on plate / stack / place in drawer) -- each occurrence here is
# labeled identically but detected independently in its own file.
LIBERO_90_LILO22_CONFIGS: list[TaskConfig] = [
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Alphabet Soup", GraspEvent("alphabet_soup_1")),
            SkillSpec("Place Alphabet Soup in Basket", PredicateEvent("in", ("alphabet_soup_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Cream Cheese", GraspEvent("cream_cheese_1")),
            SkillSpec("Place Cream Cheese in Basket", PredicateEvent("in", ("cream_cheese_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Tomato Sauce", GraspEvent("tomato_sauce_1")),
            SkillSpec("Place Tomato Sauce in Basket", PredicateEvent("in", ("tomato_sauce_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5",
        bddl_name="KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Place Black Bowl on Plate", PredicateEvent("on", ("akita_black_bowl_1", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5",
        bddl_name="KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Stack Black Bowl on Black Bowl", PredicateEvent("on", ("akita_black_bowl_1", "akita_black_bowl_2"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Place Black Bowl in Bottom Drawer", PredicateEvent("in", ("akita_black_bowl_1", "white_cabinet_1_bottom_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl",
        skills=[
            SkillSpec("Close Bottom Drawer", PredicateEvent("close", ("white_cabinet_1_bottom_region",))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5",
        bddl_name="KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl",
        skills=[
            SkillSpec("Pick Moka Pot", GraspEvent("moka_pot_1")),
            SkillSpec("Place Moka Pot on Stove", PredicateEvent("on", ("moka_pot_1", "flat_stove_1_cook_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",
        bddl_name="KITCHEN_SCENE3_turn_on_the_stove.bddl",
        skills=[
            SkillSpec("Turn On Stove", PredicateEvent("turnon", ("flat_stove_1",))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Butter", GraspEvent("butter_1")),
            SkillSpec("Place Butter in Basket", PredicateEvent("in", ("butter_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
        skills=[
            SkillSpec("Pick Chocolate Pudding", GraspEvent("chocolate_pudding_1")),
            SkillSpec(
                "Place Chocolate Pudding Right of Plate",
                PredicateEvent("on", ("chocolate_pudding_1", "living_room_table_plate_right_region")),
            ),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick White Mug", GraspEvent("porcelain_mug_1")),
            SkillSpec("Place White Mug on Plate", PredicateEvent("on", ("porcelain_mug_1", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.bddl",
        skills=[
            SkillSpec("Pick Yellow and White Mug", GraspEvent("white_yellow_mug_1")),
            SkillSpec("Place Yellow and White Mug on Right Plate", PredicateEvent("on", ("white_yellow_mug_1", "plate_2"))),
        ],
    ),
]

# name -> (task configs, bddl_files subdirectory to resolve bddl_name against)
TASK_SET_REGISTRY: dict[str, tuple[list[TaskConfig], str]] = {
    "libero_10": (LIBERO_10_CONFIGS, "libero_10"),
    "libero_90_lilo22": (LIBERO_90_LILO22_CONFIGS, "libero_90"),
}


def _build_env(env_args: dict, bddl_file_path: pathlib.Path):
    """Builds a LIBERO env with no renderer at all -- we only need physics
    (contacts, joint qpos, body positions) for predicate/grasp checks, not
    images, so this needs no EGL/GLX/GPU."""
    from libero.libero.envs import TASK_MAPPING
    import libero.libero.utils.utils as libero_utils

    env_kwargs = dict(env_args["env_kwargs"])
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=str(bddl_file_path),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=False,
        camera_depths=False,
        camera_segmentations=None,
    )
    problem_name = env_args["problem_name"]
    return TASK_MAPPING[problem_name](**env_kwargs)


def _gripper_width(gripper_qpos: np.ndarray) -> float:
    return float(np.abs(gripper_qpos[0]) + np.abs(gripper_qpos[1]))


def _check_event(env, event: Event, initial_heights: dict[str, float]) -> bool:
    if isinstance(event, PredicateEvent):
        from libero.libero.envs.predicates import eval_predicate_fn

        args = [env.object_states_dict[a] for a in event.args]
        return bool(eval_predicate_fn(event.name, *args))
    if isinstance(event, ContactEvent):
        gripper = env.robots[0].gripper
        obj = env.get_object(event.obj)
        return bool(env.check_contact(gripper, obj))
    if isinstance(event, GraspEvent):
        # Not robosuite's own _check_grasp(): its strict requirement that BOTH
        # the left and right fingerpad geom groups individually contact the
        # object turned out too fragile here -- confirmed on real demos (e.g.
        # LIVING_ROOM_SCENE5) where an object visibly lifts several
        # centimeters (a genuine pick) while _check_grasp stays False the
        # entire time, apparently because this object's grasp doesn't land
        # simultaneous two-sided fingerpad contact the way _check_grasp
        # assumes. Plain gripper-object contact plus a real height rise is a
        # looser but much more robust proxy for "the object is being held",
        # and still requires the two conditions to co-occur so a mere brush
        # of contact without lifting (or a lift from an unrelated nudge
        # without contact) doesn't count.
        gripper = env.robots[0].gripper
        obj = env.get_object(event.obj)
        if not env.check_contact(gripper, obj):
            return False
        current_z = float(env.sim.data.body_xpos[env.obj_body_id[event.obj]][2])
        return (current_z - initial_heights[event.obj]) > LIFT_THRESHOLD_M
    raise TypeError(f"_check_event does not handle {type(event)} directly (ReleaseOnsetEvent is post-processed)")


def _grasp_and_contact_objects(skills: list[SkillSpec]) -> set[str]:
    objs = set()
    for skill in skills:
        if isinstance(skill.event, (GraspEvent, ContactEvent)):
            objs.add(skill.event.obj)
    return objs


def _fix_libero_asset_paths(xml_str: str) -> str:
    """Rewrites LIBERO's own (non-robosuite) mesh/texture asset paths.

    Each demo's recorded `model_file` XML has absolute mesh/texture paths
    baked in from whatever machine originally collected the official LIBERO
    benchmark data (e.g. "/home/yifengz/workspace/libero-dev/chiliocosm/
    assets/scenes/fridge/visual/fridge_vis.msh"). libero_utils.
    postprocess_model_xml() already fixes the subset of paths that route
    through the installed `robosuite` package (matching on a "robosuite"
    path segment), but LIBERO's OWN asset library -- everything under an
    "assets" directory in the original collection tree -- isn't a `robosuite`
    path and needs a separate, analogous fix: replace everything up to and
    including that "assets" segment with this checkout's actual asset root
    (from `get_libero_path("assets")`), keeping the relative suffix as-is
    (including the harmless "scenes/../textures/..." style paths some assets
    use, which the filesystem resolves without issue).
    """
    import xml.etree.ElementTree as ET

    from libero.libero import get_libero_path

    assets_root = get_libero_path("assets").rstrip("/")

    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    for elem in asset.findall("mesh") + asset.findall("texture"):
        old_path = elem.get("file")
        if old_path is None or "/assets/" not in old_path or os.path.isfile(old_path):
            continue
        suffix = old_path.split("/assets/", 1)[1]
        elem.set("file", f"{assets_root}/{suffix}")
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def annotate_demo(env, demo_group: h5py.Group, task: TaskConfig) -> dict:
    """Note: this replays the demo by teleporting to each frame's recorded
    ground-truth `states[j]` (sim.set_state_from_flattened + sim.forward()),
    NOT by re-stepping the recorded actions through our own controller. An
    action-replay was tried first (matching third_party/libero/scripts/
    create_dataset.py's pattern) but diverged heavily from frame 0 on this
    checkout's robosuite/mujoco versions (confirmed via the state-playback
    error create_dataset.py itself checks for) -- most likely a controller
    behavior mismatch between whatever robosuite version originally collected
    this data and the one installed here. Since we only need positions/
    contacts/joint qpos for predicate and grasp checks, not images, forward
    kinematics from the recorded ground-truth state is both simpler and more
    faithful than re-deriving an approximation of it via actions.
    """
    import libero.libero.utils.utils as libero_utils

    states = np.asarray(demo_group["states"])
    gripper_qpos_trace = np.asarray(demo_group["obs"]["gripper_states"])  # [T, 2]
    model_xml = demo_group.attrs["model_file"]
    num_frames = states.shape[0]

    env.reset()
    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
    model_xml = _fix_libero_asset_paths(model_xml)
    env.reset_from_xml_string(model_xml)
    env.sim.reset()

    def goto_frame(j: int) -> None:
        env.sim.set_state_from_flattened(states[j])
        env.sim.forward()

    goto_frame(0)
    lift_objs = _grasp_and_contact_objects(task.skills)
    initial_heights = {obj: float(env.sim.data.body_xpos[env.obj_body_id[obj]][2]) for obj in lift_objs}

    # Direct (non-release-onset) events: main skills + aux events, all indexed
    # by a unique key so we can look up "the frame skill i completed at" when
    # resolving a later ReleaseOnsetEvent that refers back to it.
    direct_events: dict[int | str, Event] = {}
    for idx, skill in enumerate(task.skills, start=1):
        if not isinstance(skill.event, ReleaseOnsetEvent):
            direct_events[idx] = skill.event
    for name, event in task.aux_events:
        direct_events[name] = event

    completion_frame: dict[int | str, int | None] = dict.fromkeys(direct_events, None)
    gripper_widths = np.array([_gripper_width(g) for g in gripper_qpos_trace], dtype=np.float32)

    for j in range(num_frames):
        goto_frame(j)
        for key, event in direct_events.items():
            if completion_frame[key] is None and _check_event(env, event, initial_heights):
                completion_frame[key] = j

    # Resolve ReleaseOnsetEvent skills now that direct events are known: first
    # frame, strictly after the referenced skill's completion, where the
    # gripper re-opens beyond RELEASE_DELTA_M from its post-completion minimum.
    for idx, skill in enumerate(task.skills, start=1):
        if not isinstance(skill.event, ReleaseOnsetEvent):
            continue
        after_frame = completion_frame.get(skill.event.after_skill_idx)
        onset_frame = None
        if after_frame is not None and after_frame + 1 < num_frames:
            window = gripper_widths[after_frame + 1 :]
            running_min = np.minimum.accumulate(window)
            candidates = np.nonzero(window - running_min > RELEASE_DELTA_M)[0]
            if candidates.size:
                onset_frame = after_frame + 1 + int(candidates[0])
        completion_frame[idx] = onset_frame

    # --- Assemble contiguous, monotonic [start_frame, end_frame] segments ---
    # The last skill's own detected frame is purely informational: its
    # end_frame is unconditionally forced to the episode's last frame below
    # (a demo may keep going a little past the moment its goal predicate
    # first turns true), so if it happens to fire at/before the previous
    # skill's boundary -- e.g. a "place" predicate like On() can turn True
    # from proximity+contact slightly before or right as the gripper starts
    # to visibly open -- that's an expected near-simultaneous overlap at the
    # very end of a place motion, not a real ordering problem, and shouldn't
    # be flagged as one.
    boundaries = []
    order_anomaly = False
    prev_end = -1
    last_idx = len(task.skills)
    for idx, skill in enumerate(task.skills, start=1):
        detected = completion_frame[idx]
        detection_failed = detected is None
        end_frame = num_frames - 1 if detection_failed else detected
        if end_frame <= prev_end and idx != last_idx:
            order_anomaly = True
            end_frame = min(prev_end + 1, num_frames - 1)
        boundaries.append(
            {
                "skill_idx": idx,
                "prompt": skill.prompt,
                "start_frame": prev_end + 1,
                "end_frame": end_frame,
                "detection_failed": detection_failed,
            }
        )
        prev_end = end_frame
    # Last segment always runs to the end of the episode, regardless of
    # exactly which frame its own completion predicate first fired at.
    boundaries[-1]["end_frame"] = num_frames - 1
    boundaries[-1]["start_frame"] = min(boundaries[-1]["start_frame"], num_frames - 1)

    aux_out = [
        {"event": name, "frame": completion_frame[name]}
        for name, _event in task.aux_events
    ]

    return {
        "num_frames": num_frames,
        "boundaries": boundaries,
        "aux_events": aux_out,
        "order_anomaly": order_anomaly,
    }


def main(args: argparse.Namespace) -> None:
    data_dir = pathlib.Path(args.data_dir)
    results: dict[str, dict] = {}

    task_configs, bddl_subdir = TASK_SET_REGISTRY[args.task_set]
    bddl_root = BDDL_FILES_ROOT / bddl_subdir

    for task in task_configs:
        if args.task_filter and args.task_filter not in task.hdf5_name:
            continue

        hdf5_path = data_dir / task.hdf5_name
        bddl_path = bddl_root / task.bddl_name
        logging.info(f"=== {task.hdf5_name} ===")

        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            env_args = json.loads(data_grp.attrs["env_args"])
            problem_info = json.loads(data_grp.attrs["problem_info"])
            demo_names = sorted(
                (k for k in data_grp.keys() if k.startswith("demo_")),
                key=lambda s: int(s.split("_")[1]),
            )
            if args.max_demos is not None:
                demo_names = demo_names[: args.max_demos]

            env = _build_env(env_args, bddl_path)
            try:
                demo_results = {}
                for demo_name in demo_names:
                    logging.info(f"  {demo_name} ({len(demo_results) + 1}/{len(demo_names)})...")
                    demo_results[demo_name] = annotate_demo(env, data_grp[demo_name], task)
                    if demo_results[demo_name]["order_anomaly"]:
                        logging.warning(f"  {demo_name}: sub-skill order anomaly (see order_anomaly flag)")
            finally:
                env.close()

        results[task.hdf5_name] = {
            "task_instruction": problem_info["language_instruction"],
            "subskills": [s.prompt for s in task.skills],
            "demos": demo_results,
        }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logging.info(f"Wrote {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--task-set",
        choices=sorted(TASK_SET_REGISTRY),
        default="libero_10",
        help=(
            "'libero_10': the 10 LIBERO-10 long-horizon tasks, each split into its own "
            "2-4 sub-skills (--data-dir should point at a libero_10 HDF5 directory). "
            "'libero_90_lilo22': LiLo-VLA's 22-skill atomic library (see "
            "https://yy-gx.github.io/LiLo-VLA/static/pdfs/appendix.pdf Table III), mapped "
            "onto 13 real LIBERO-90 task files, mostly one pick + one place skill per file "
            "(--data-dir should point at a libero_90 HDF5 directory)."
        ),
    )
    parser.add_argument("--task-filter", default=None, help="Only process HDF5 files whose name contains this substring.")
    parser.add_argument("--max-demos", type=int, default=None, help="Only process the first N demos per task (for quick testing).")
    main(parser.parse_args())
