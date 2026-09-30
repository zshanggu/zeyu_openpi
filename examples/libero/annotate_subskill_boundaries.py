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
    # Only used by task sets that span several data folders / suites (see "lilo22_new"): the
    # sub-folder of --data-dir holding this task's HDF5, and the bddl_files sub-folder of its suite.
    data_subdir: str = ""
    bddl_subdir: str = ""
    # True (default): the last skill's segment runs to the end of the episode. False: it ends at the
    # frame its own event fires and the rest of the demo is left unlabeled -- for tasks whose
    # remaining motion is not a LiLo-22 skill (e.g. the place after a "pick black bowl" that is
    # the only library skill in the task).
    label_to_episode_end: bool = True
    # True: when the first skill is a pick, its segment starts AFTER any non-library progress that
    # precedes it -- a drawer or microwave door being moved a long way before the pick (e.g. "close the top
    # drawer, then put the black bowl on top of the cabinet": closing the drawer is not one of the 22
    # skills, so it must not be part of the pick's segment). See _prior_progress_end().
    exclude_prior_progress: bool = False


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
    # --- The remaining 15 of the 28 LIBERO-90 tasks in libero_modified_hdf5/lilo22_tasks.txt
    # (that file's own generator, hdf5-lerobot-visualizer/scripts/list_lilo22_tasks.py, parses
    # each BDDL's goal predicates the same way -- these mirror its S0/S1/... skill lists exactly).
    TaskConfig(
        hdf5_name="KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_demo.hdf5",
        bddl_name="KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_3")),
            SkillSpec("Place Black Bowl on Plate", PredicateEvent("on", ("akita_black_bowl_3", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_demo.hdf5",
        bddl_name="KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Place Black Bowl on Plate", PredicateEvent("on", ("akita_black_bowl_1", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_demo.hdf5",
        bddl_name="KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_2")),
            SkillSpec("Place Black Bowl on Plate", PredicateEvent("on", ("akita_black_bowl_2", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_demo.hdf5",
        bddl_name="KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_2")),
            SkillSpec("Stack Black Bowl on Black Bowl", PredicateEvent("on", ("akita_black_bowl_2", "akita_black_bowl_3"))),
        ],
    ),
    TaskConfig(
        # Goal also requires (On chefmate_8_frypan_1 flat_stove_1_cook_region), which isn't one of
        # the 22 LiLo-VLA skills (no "place frying pan" in the library) -- lilo22_tasks.txt only
        # lists "turn on stove" for this file, so that's the only boundary labeled here too; the
        # demo continues past it (placing the pan) inside this single trailing segment.
        hdf5_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
        bddl_name="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl",
        skills=[
            SkillSpec("Turn On Stove", PredicateEvent("turnon", ("flat_stove_1",))),
        ],
    ),
    TaskConfig(
        # Goal also requires (Open white_cabinet_1_top_region); "open X" isn't one of the 22
        # skills either (only "close"), so only the close-drawer boundary is labeled, matching
        # lilo22_tasks.txt.
        hdf5_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl",
        skills=[
            SkillSpec("Close Bottom Drawer", PredicateEvent("close", ("white_cabinet_1_bottom_region",))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE5_put_the_black_bowl_on_the_plate_demo.hdf5",
        bddl_name="KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Place Black Bowl on Plate", PredicateEvent("on", ("akita_black_bowl_1", "plate_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5",
        bddl_name="KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove.bddl",
        skills=[
            SkillSpec("Pick Moka Pot", GraspEvent("moka_pot_1")),
            SkillSpec("Place Moka Pot on Stove", PredicateEvent("on", ("moka_pot_1", "flat_stove_1_cook_region"))),
            SkillSpec("Turn On Stove", PredicateEvent("turnon", ("flat_stove_1",))),
        ],
    ),
    TaskConfig(
        hdf5_name="KITCHEN_SCENE9_turn_on_the_stove_demo.hdf5",
        bddl_name="KITCHEN_SCENE9_turn_on_the_stove.bddl",
        skills=[
            SkillSpec("Turn On Stove", PredicateEvent("turnon", ("flat_stove_1",))),
        ],
    ),
    TaskConfig(
        # Same "frying pan isn't a LiLo-22 skill" reasoning as the KITCHEN_SCENE3 twin above.
        hdf5_name="KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
        bddl_name="KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl",
        skills=[
            SkillSpec("Turn On Stove", PredicateEvent("turnon", ("flat_stove_1",))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Alphabet Soup", GraspEvent("alphabet_soup_1")),
            SkillSpec("Place Alphabet Soup in Basket", PredicateEvent("in", ("alphabet_soup_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl",
        skills=[
            SkillSpec("Pick Tomato Sauce", GraspEvent("tomato_sauce_1")),
            SkillSpec("Place Tomato Sauce in Basket", PredicateEvent("in", ("tomato_sauce_1", "basket_1_contain_region"))),
        ],
    ),
    TaskConfig(
        # Goal also requires (In akita_black_bowl_2 wooden_tray_1_contain_region); "place bowl in
        # tray" isn't one of the 22 skills, so only the stack boundary is labeled.
        hdf5_name="LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_1")),
            SkillSpec("Stack Black Bowl on Black Bowl", PredicateEvent("on", ("akita_black_bowl_1", "akita_black_bowl_2"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.bddl",
        skills=[
            SkillSpec("Pick Black Bowl", GraspEvent("akita_black_bowl_2")),
            SkillSpec("Stack Black Bowl on Black Bowl", PredicateEvent("on", ("akita_black_bowl_2", "akita_black_bowl_1"))),
        ],
    ),
    TaskConfig(
        hdf5_name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_demo.hdf5",
        bddl_name="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.bddl",
        skills=[
            SkillSpec("Pick White Mug", GraspEvent("porcelain_mug_1")),
            SkillSpec("Place White Mug on Plate", PredicateEvent("on", ("porcelain_mug_1", "plate_1"))),
        ],
    ),
]

NEW_PICK_ONLY_TASKS_JSON = '''
[
[
"libero_object_no_noops",
"libero_object",
"pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
],
[
"libero_goal_no_noops",
"libero_goal",
"put_the_bowl_on_the_stove"
],
[
"libero_goal_no_noops",
"libero_goal",
"open_the_top_drawer_and_put_the_bowl_inside"
],
[
"libero_goal_no_noops",
"libero_goal",
"put_the_bowl_on_top_of_the_cabinet"
],
[
"libero_goal_no_noops",
"libero_goal",
"put_the_cream_cheese_in_the_bowl"
],
[
"libero_10_no_noops",
"libero_10",
"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet"
],
[
"libero90_openvla_256",
"libero_90",
"KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray"
],
[
"libero90_openvla_256",
"libero_90",
"LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate"
],
[
"libero90_openvla_256",
"libero_90",
"STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy"
],
[
"libero90_openvla_256",
"libero_90",
"STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy"
]
]
'''

# --- Tasks that were added to libero_modified_hdf5/lilo22_tasks.txt once "pick X" counted on its own
# (X moved to a destination that is not one of the 11 library place skills). Each has exactly one
# LiLo-22 skill, a pick, so its config is derived from the BDDL goal instead of typed by hand: the
# picked object is the subject of the On/In predicate whose subject is one of the 9 pickable
# categories. (data folder under --data-dir, bddl_files suite folder, task name)
_PICK_CATEGORIES = [
    (r"alphabet_soup_\d+", "Pick Alphabet Soup"),
    (r"cream_cheese_\d+", "Pick Cream Cheese"),
    (r"tomato_sauce_\d+", "Pick Tomato Sauce"),
    (r"butter_\d+", "Pick Butter"),
    (r"akita_black_bowl_\d+", "Pick Black Bowl"),
    (r"moka_pot_\d+", "Pick Moka Pot"),
    (r"chocolate_pudding_\d+", "Pick Chocolate Pudding"),
    (r"porcelain_mug_\d+", "Pick White Mug"),
    (r"white_yellow_mug_\d+", "Pick Yellow and White Mug"),
]
LILO22_NEW_PICK_ONLY_TASKS = json.loads(NEW_PICK_ONLY_TASKS_JSON)


def _pick_only_config(data_subdir: str, bddl_subdir: str, name: str) -> TaskConfig:
    import re

    bddl_path = BDDL_FILES_ROOT / bddl_subdir / f"{name}.bddl"
    goal = bddl_path.read_text()
    goal = goal[goal.index("(:goal"):]
    subjects = []
    for pred, subj, _target in re.findall(r"\(\s*(On|In)\s+(\S+)\s+(\S+)\s*\)", goal):
        for pattern, prompt in _PICK_CATEGORIES:
            if re.fullmatch(pattern, subj) and (subj, prompt) not in subjects:
                subjects.append((subj, prompt))
    assert len(subjects) == 1, f"{name}: expected exactly one picked object in the goal, got {subjects}"
    obj, prompt = subjects[0]
    return TaskConfig(
        hdf5_name=f"{name}_demo.hdf5",
        bddl_name=f"{name}.bddl",
        skills=[SkillSpec(prompt, GraspEvent(obj))],
        data_subdir=data_subdir,
        bddl_subdir=bddl_subdir,
        label_to_episode_end=False,
        exclude_prior_progress=True,
    )


LILO22_NEW_CONFIGS: list[TaskConfig] = [_pick_only_config(*t) for t in LILO22_NEW_PICK_ONLY_TASKS]
for _c in LIBERO_90_LILO22_CONFIGS:
    _c.exclude_prior_progress = True

# LiLo-22 tasks whose goal goes on after their last library skill with steps that are NOT library skills
# (moving the stack into the tray, placing the frying pan, opening the top drawer). Their last skill
# must end where its own event fires, not run to the end of the episode -- that would swallow those steps.
_TRAILING_NON_LIBRARY_STEPS = {
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5",
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray_demo.hdf5",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_demo.hdf5",
}
for _c in LIBERO_90_LILO22_CONFIGS:
    if _c.hdf5_name in _TRAILING_NON_LIBRARY_STEPS:
        _c.label_to_episode_end = False

# --- The two cabinet-swapped LIBERO-90 tasks (libero_modified_hdf5/libero90_openvla_256_wooden_cabinet, made by
# libero_rerender_script/rerender_libero90.py --bddl-dir bddl_wooden_cabinet): same demos and actions, but
# white_cabinet_1 replaced by wooden_cabinet_1 (identical geometry, different texture). The env must be built
# from the SWAPPED BDDL (a directory holding <task>.bddl; bddl_subdir may be absolute), and the cabinet region
# is named wooden_cabinet_1_*. Same skill lists as their white-cabinet originals.
WOODEN_CABINET_BDDL_DIR = os.environ.get("LILO22_WOODEN_CABINET_BDDL_DIR", "/wooden_cabinet_bddl")
LILO22_WOODEN_CABINET_CONFIGS: list[TaskConfig] = [
    TaskConfig(
        hdf5_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_wooden_cabinet_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl",
        skills=[SkillSpec("Close Bottom Drawer", PredicateEvent("close", ("wooden_cabinet_1_bottom_region",)))],
        bddl_subdir=WOODEN_CABINET_BDDL_DIR,
    ),
    TaskConfig(
        # goal continues with (Open top drawer), not a LiLo-22 skill: end at the close, don't run to the end
        hdf5_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_wooden_cabinet_demo.hdf5",
        bddl_name="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl",
        skills=[SkillSpec("Close Bottom Drawer", PredicateEvent("close", ("wooden_cabinet_1_bottom_region",)))],
        bddl_subdir=WOODEN_CABINET_BDDL_DIR,
        label_to_episode_end=False,
    ),
]

# name -> (task configs, bddl_files subdirectory to resolve bddl_name against; "" = per-task)
TASK_SET_REGISTRY: dict[str, tuple[list[TaskConfig], str]] = {
    "libero_10": (LIBERO_10_CONFIGS, "libero_10"),
    "libero_90_lilo22": (LIBERO_90_LILO22_CONFIGS, "libero_90"),
    "lilo22_new": (LILO22_NEW_CONFIGS, ""),
    "lilo22_wooden_cabinet": (LILO22_WOODEN_CABINET_CONFIGS, ""),
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


def _build_env_simple(bddl_file_path: pathlib.Path):
    """Builds a LIBERO env straight from a BDDL file, with no dependency on a
    raw demo file's `env_args`/`model_file` attrs. Used for data that doesn't
    carry those (e.g. libero_modified_hdf5/libero90_openvla_256, produced by
    OpenVLA-style re-rendering, which only keeps what OpenVLA's own pipeline
    needs). Verified equivalent to `_build_env` for predicate/grasp/contact
    checks: `states[j]` is a full flattened MuJoCo state, so
    `sim.set_state_from_flattened` + `sim.forward()` reproduces the recorded
    frame regardless of which XML variant built the sim, as long as the
    object/joint set matches -- which it does, since that's fixed by the BDDL
    file, not by the per-episode initial-state sample."""
    from libero.libero.envs import OffScreenRenderEnv

    wrapper = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file_path),
        camera_heights=64,
        camera_widths=64,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=False,
    )
    return wrapper.env  # the raw TASK_MAPPING instance -- same type _build_env returns


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


# A drawer/door joint must move this far to count as a task step: metres for a sliding drawer, radians for
# a hinged door (a microwave door swings ~1.4 rad; a 0.1 rad wobble from the arm brushing it is not a step).
PRIOR_PROGRESS_MIN_TRAVEL_SLIDE = 0.05
PRIOR_PROGRESS_MIN_TRAVEL_HINGE = 0.5
PRIOR_PROGRESS_SETTLE_TOL = 0.005  # ... and it is "done" once it stays within this of its final position


def _articulated_fixtures(env, cache_attr: str = "_lilo_articulated_fixtures") -> list:
    """object_states_dict keys of articulated fixtures (cabinets, microwave, ...): those whose state can
    report joint positions. Region sites and plain objects have none."""
    keys = getattr(env, cache_attr, None)
    if keys is None:
        keys = []
        for k, st in env.object_states_dict.items():
            try:
                if len(st.get_joint_state()) > 0:
                    keys.append(k)
            except Exception:
                pass
        setattr(env, cache_attr, keys)
    return keys


def _joint_min_travel(env, key: str) -> list:
    """Per-joint travel threshold for the fixture `key`, in the order its get_joint_state() reports."""
    import mujoco

    out = []
    for jname in env.get_object(key).joints:
        jtype = env.sim.model.jnt_type[env.sim.model.joint_name2id(jname)]
        out.append(PRIOR_PROGRESS_MIN_TRAVEL_HINGE if jtype == mujoco.mjtJoint.mjJNT_HINGE else PRIOR_PROGRESS_MIN_TRAVEL_SLIDE)
    return out


def _prior_progress_end(joint_trace: dict, first_skill_done: int, min_travel: dict):
    """Non-library progress made BEFORE the first skill: a drawer/door joint that moved a lot (>=
    its PRIOR_PROGRESS_MIN_TRAVEL_*) before the skill completed -- e.g. closing or opening a drawer. Its
    segment is excluded from the skill; it ends at the frame the joint settles at the position it holds
    from then until the skill completes.

    Travel, not LIBERO's open/close predicates, decides it: those flip at a threshold, so a drawer resting
    just inside it (qpos -0.145 vs a -0.14 threshold) "opens/closes" after an incidental ~1.5 cm bump
    by the arm, and one resting exactly on the `close` range's edge (0.0000) flickers closed/not-closed.
    A real open/close moves the joint ~14 cm.
    Returns (frame, events); events = [{"event": "moved(fixture.jN) q0->q1", "frame": f}]."""
    events = []
    for key, trace in joint_trace.items():
        Q = np.asarray(trace, dtype=float)  # frames x joints
        for jdx in range(Q.shape[1]):
            q = Q[: first_skill_done + 1, jdx]
            if np.max(np.abs(q - q[0])) < min_travel[key][jdx]:
                continue
            far = np.nonzero(np.abs(q - q[-1]) >= PRIOR_PROGRESS_SETTLE_TOL)[0]
            f = int(far[-1]) + 1 if len(far) else 0
            if f > 0:
                events.append({"event": f"moved({key}.j{jdx}) {q[0]:+.3f}->{q[-1]:+.3f}", "frame": f})
    if not events:
        return None, []
    events.sort(key=lambda e: e["frame"])
    return events[-1]["frame"], events


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
    states = np.asarray(demo_group["states"])
    gripper_qpos_trace = np.asarray(demo_group["obs"]["gripper_states"])  # [T, 2]
    num_frames = states.shape[0]

    env.reset()
    if "model_file" in demo_group.attrs:
        # Raw LIBERO demo: replay inside THIS demo's own randomized model (see _build_env_simple's
        # docstring for why this isn't needed -- kept as the original, most-faithful path for data
        # that still carries it).
        import libero.libero.utils.utils as libero_utils

        model_xml = libero_utils.postprocess_model_xml(demo_group.attrs["model_file"], {})
        model_xml = _fix_libero_asset_paths(model_xml)
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
    # else: a regenerated demo (e.g. libero_modified_hdf5) carries no per-demo model XML; `env` was
    # already built from the task's plain BDDL file, and states[0] below puts it in this episode's
    # actual initial layout regardless.

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

    track_prior = task.exclude_prior_progress and task.skills and isinstance(task.skills[0].event, GraspEvent)
    if track_prior:
        art_keys = _articulated_fixtures(env)
        joint_trace: dict = {k: [] for k in art_keys}
        min_travel = {k: _joint_min_travel(env, k) for k in art_keys}

    for j in range(num_frames):
        goto_frame(j)
        for key, event in direct_events.items():
            if completion_frame[key] is None and _check_event(env, event, initial_heights):
                completion_frame[key] = j
        if track_prior:
            for k in art_keys:
                joint_trace[k].append(env.object_states_dict[k].get_joint_state())

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
        if end_frame <= prev_end and (idx != last_idx or not task.label_to_episode_end):
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
    # Last segment runs to the end of the episode, regardless of exactly which frame its own
    # completion predicate first fired at -- unless the task says the rest of the demo is not a
    # labeled skill (label_to_episode_end=False), in which case it stops where it was detected.
    if task.label_to_episode_end:
        boundaries[-1]["end_frame"] = num_frames - 1
        boundaries[-1]["start_frame"] = min(boundaries[-1]["start_frame"], num_frames - 1)

    aux_out = [
        {"event": name, "frame": completion_frame[name]}
        for name, _event in task.aux_events
    ]

    result = {
        "num_frames": num_frames,
        "boundaries": boundaries,
        "aux_events": aux_out,
        "order_anomaly": order_anomaly,
    }
    if track_prior and completion_frame.get(1) is not None:
        prior_end, prior_events = _prior_progress_end(joint_trace, completion_frame[1], min_travel)
        if prior_end is not None:
            boundaries[0]["start_frame"] = prior_end + 1
            result["prior_progress"] = {"end_frame": prior_end, "events": prior_events}
    return result


def _language_from_bddl(bddl_path: pathlib.Path) -> str:
    import re

    m = re.search(r"\(:language\s+(.*?)\)", bddl_path.read_text(), re.S)
    return " ".join(m.group(1).split()) if m else bddl_path.stem.replace("_", " ")


def main(args: argparse.Namespace) -> None:
    data_dir = pathlib.Path(args.data_dir)
    results: dict[str, dict] = {}
    subdir_of: dict[str, str] = {}  # hdf5_name -> data sub-folder it came from

    task_configs, default_bddl_subdir = TASK_SET_REGISTRY[args.task_set]
    if args.shard:
        idx, n = (int(x) for x in args.shard.split("/"))
        task_configs = task_configs[idx::n]

    for task in task_configs:
        if args.task_filter and args.task_filter not in task.hdf5_name:
            continue

        hdf5_path = data_dir / task.data_subdir / task.hdf5_name
        bddl_path = BDDL_FILES_ROOT / (task.bddl_subdir or default_bddl_subdir) / task.bddl_name
        logging.info(f"=== {task.data_subdir + '/' if task.data_subdir else ''}{task.hdf5_name} ===")

        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            # Files that went through OpenVLA-style regeneration (libero_modified_hdf5/*) may carry no
            # attributes at all -- fall back to the BDDL's own :language line.
            if "problem_info" in data_grp.attrs:
                instruction = json.loads(data_grp.attrs["problem_info"])["language_instruction"]
            else:
                instruction = _language_from_bddl(bddl_path)

            # Two group layouts: a raw/OpenVLA-regenerated file has only "data"; a
            # libero_modified_hdf5-style file also has "data_distractor" (randomized-distractor
            # episodes, see libero_rerender_script/rerender_libero90.py) -- both get boundaries,
            # since they run the identical action sequence and only the decor differs.
            groups = [data_grp] + ([f["data_distractor"]] if "data_distractor" in f else [])
            demo_names = sorted(
                (k for g in groups for k in g.keys()),
                key=lambda s: tuple(int(x) for x in s.split("_")[1:] if x.isdigit()),
            )
            if args.max_demos is not None:
                demo_names = demo_names[: args.max_demos]

            has_raw_attrs = "env_args" in data_grp.attrs
            env = _build_env(json.loads(data_grp.attrs["env_args"]), bddl_path) if has_raw_attrs else _build_env_simple(bddl_path)
            try:
                demo_results = {}
                for demo_name in demo_names:
                    logging.info(f"  {demo_name} ({len(demo_results) + 1}/{len(demo_names)})...")
                    group = data_grp if demo_name in data_grp else f["data_distractor"]
                    demo_results[demo_name] = annotate_demo(env, group[demo_name], task)
                    if demo_results[demo_name]["order_anomaly"]:
                        logging.warning(f"  {demo_name}: sub-skill order anomaly (see order_anomaly flag)")
            finally:
                env.close()

        results[task.hdf5_name] = {
            "task_instruction": instruction,
            "subskills": [s.prompt for s in task.skills],
            "data_subdir": task.data_subdir,
            "demos": demo_results,
        }
        subdir_of[task.hdf5_name] = task.data_subdir

    if args.out_per_subdir:
        # One subskill_boundaries.json next to each data folder's HDF5 files (where the viewer looks),
        # merged into any file already there so earlier tasks' labels are kept.
        for subdir in sorted(set(subdir_of.values())):
            out_path = data_dir / subdir / "subskill_boundaries.json"
            merged = json.loads(out_path.read_text()) if out_path.exists() else {}
            merged.update({name: r for name, r in results.items() if subdir_of[name] == subdir})
            out_path.write_text(json.dumps(merged, indent=2))
            logging.info(f"Wrote {out_path} ({len(merged)} tasks)")
        return

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and out_path.exists():
        merged = json.loads(out_path.read_text())
        merged.update(results)
        results = merged
    out_path.write_text(json.dumps(results, indent=2))
    logging.info(f"Wrote {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default=None, help="output JSON (required unless --out-per-subdir)")
    parser.add_argument("--merge", action="store_true", help="with --out: merge into an existing file instead of overwriting it")
    parser.add_argument("--shard", default=None, help="I/N: only every N-th task config starting at I (run N processes in parallel)")
    parser.add_argument(
        "--out-per-subdir",
        action="store_true",
        help="write/merge a subskill_boundaries.json inside each task's data sub-folder of --data-dir "
        "(for task sets that span several folders, e.g. lilo22_new) instead of one --out file",
    )
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
    _args = parser.parse_args()
    if not _args.out and not _args.out_per_subdir:
        parser.error("--out is required unless --out-per-subdir is given")
    main(_args)
