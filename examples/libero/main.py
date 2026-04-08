import collections
import csv
import dataclasses
from typing import List
import logging
import math
import pathlib

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 3  # Number of rollouts per task

    # Extra prompts to evaluate alongside the original task description.
    # Pass multiple times on the CLI: --extra-prompts "Stay still" --extra-prompts "Pick up the apple"
    extra_prompts: List[str] = dataclasses.field(
        default_factory=lambda: [
            # "Stay still and do nothing",
            # "Pick up the apple and put it in the bowl",
            # Add or remove prompts here — no need to recreate the container.
        ]
    )

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # All prompts: original task description (filled in per-task) + fixed extra prompts.
    # We use a sentinel so we can substitute the real task description at runtime.
    ORIGINAL_PROMPT_SENTINEL = "__original__"
    all_prompt_labels = [ORIGINAL_PROMPT_SENTINEL] + list(args.extra_prompts)

    # Per-prompt success tracking across all tasks.
    prompt_total_episodes: dict[str, int] = {p: 0 for p in all_prompt_labels}
    prompt_total_successes: dict[str, int] = {p: 0 for p in all_prompt_labels}

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for prompt_label in tqdm.tqdm(all_prompt_labels, desc="prompts"):
            prompt_text = task_description if prompt_label == ORIGINAL_PROMPT_SENTINEL else prompt_label
            # Slug used in filenames: "original" for the real task, truncated label otherwise.
            prompt_slug = "original" if prompt_label == ORIGINAL_PROMPT_SENTINEL else prompt_label[:40].replace(" ", "_")

            prompt_episodes, prompt_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description} | Prompt: {prompt_text}")

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                attn_buffer = []  # per-inference-call attention info: list of {"lang_attn": [L], "img_attn": [L]}

                logging.info(f"Starting episode {prompt_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": prompt_text,
                            }

                            # Query model to get action (and collect attention info if available)
                            result = client.infer(element)
                            action_chunk = result["actions"]
                            logging.info(f"[attn] result keys: {list(result.keys())}")
                            if "attn_info" in result:
                                attn_buffer.append(result["attn_info"])
                                _lang = result["attn_info"]["lang_attn"]
                                _img  = result["attn_info"]["img_attn"]
                                _per_layer = "  ".join(
                                    f"L{i:02d} lang={_lang[i]:.3f} img={_img[i]:.3f} sum={_lang[i]+_img[i]:.3f}"
                                    for i in range(len(_lang))
                                )
                                logging.info(f"[attn] step={len(attn_buffer):03d}  {_per_layer}")
                            else:
                                logging.warning("[attn] 'attn_info' not in result — server may not have the updated code")
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            prompt_successes += 1
                            prompt_total_successes[prompt_label] += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                prompt_episodes += 1
                prompt_total_episodes[prompt_label] += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_stem = f"rollout_{task_segment}_prompt_{prompt_slug}_ep{episode_idx}_{suffix}"
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"{video_stem}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

                # Save individual frames in a folder with the same name as the video
                frame_dir = pathlib.Path(args.video_out_path) / video_stem
                frame_dir.mkdir(parents=True, exist_ok=True)
                for frame_idx, frame in enumerate(replay_images):
                    imageio.imwrite(frame_dir / f"{frame_idx:04d}.png", np.asarray(frame))

                # Save per-inference-step attention analysis (action expert → language/image tokens)
                # Shape: [num_inference_calls, num_expert_layers]
                logging.info(f"[attn] episode done, attn_buffer size={len(attn_buffer)}, saving to {frame_dir}")
                if attn_buffer:
                    lang_attn  = np.stack([a["lang_attn"] for a in attn_buffer], axis=0)   # [T, L]
                    img_attn   = np.stack([a["img_attn"]  for a in attn_buffer], axis=0)   # [T, L]
                    other_attn = 1.0 - lang_attn - img_attn                                 # [T, L]
                    np.save(frame_dir / "lang_attn_over_time.npy", lang_attn)
                    np.save(frame_dir / "img_attn_over_time.npy", img_attn)

                    # Save CSV log: one row per inference step
                    num_layers = lang_attn.shape[1]
                    with open(frame_dir / "attn_log.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        header = (["step"] +
                                  [f"lang_L{i}"  for i in range(num_layers)] + ["lang_mean"] +
                                  [f"img_L{i}"   for i in range(num_layers)] + ["img_mean"] +
                                  [f"other_L{i}" for i in range(num_layers)] + ["other_mean"])
                        writer.writerow(header)
                        for step_idx, (lang_row, img_row, other_row) in enumerate(
                            zip(lang_attn, img_attn, other_attn)
                        ):
                            writer.writerow(
                                [step_idx] +
                                lang_row.tolist()  + [lang_row.mean()] +
                                img_row.tolist()   + [img_row.mean()] +
                                other_row.tolist() + [other_row.mean()]
                            )

                    steps = np.arange(len(lang_attn))

                    # Figure 1: per-layer curves — 3 rows × 6 cols grid
                    nrows, ncols = 3, 6
                    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(18, 8), sharex=True, sharey=True)
                    for layer_idx in range(num_layers):
                        ax = axes1[layer_idx // ncols, layer_idx % ncols]
                        ax.plot(steps, lang_attn[:, layer_idx],  color="tab:blue",   linewidth=1.2, label="lang")
                        ax.plot(steps, img_attn[:, layer_idx],   color="tab:orange", linewidth=1.2, label="img")
                        ax.plot(steps, other_attn[:, layer_idx], color="tab:green",  linewidth=1.2, label="other")
                        ax.set_title(f"L{layer_idx:02d}", fontsize=8)
                        ax.set_ylim(0, 1)
                        ax.tick_params(labelsize=6)
                        if layer_idx == 0:
                            ax.legend(fontsize=6, loc="upper right")
                    for ax in axes1.flat:
                        ax.set_xlabel("Step", fontsize=7)
                        ax.set_ylabel("Attn", fontsize=7)
                    fig1.suptitle(f"Per-layer attention (lang / img / other)\n{video_stem}\nprompt: {prompt_text}", fontsize=8)
                    fig1.tight_layout()
                    fig1.savefig(frame_dir / "attn_per_layer.png", dpi=120)
                    plt.close(fig1)

                    # Figure 2: mean attention over all layers
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(steps, lang_attn.mean(axis=1),  color="tab:blue",   linewidth=2, label="lang (mean)")
                    ax2.plot(steps, img_attn.mean(axis=1),   color="tab:orange", linewidth=2, label="img (mean)")
                    ax2.plot(steps, other_attn.mean(axis=1), color="tab:green",  linewidth=2, label="other (mean)")
                    ax2.set_xlabel("Inference step")
                    ax2.set_ylabel("Attention score")
                    ax2.set_title("Mean attention over 18 layers")
                    ax2.set_ylim(0, 1)
                    ax2.legend()
                    fig2.suptitle(f"{video_stem}\nprompt: {prompt_text}", fontsize=8)
                    fig2.tight_layout()
                    fig2.savefig(frame_dir / "attn_mean.png", dpi=120)
                    plt.close(fig2)

                    # Figure 3: 2-panel mean attention (lang / img), auto-scaled y-axis
                    fig3, (ax3_lang, ax3_img) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                    lang_mean = lang_attn.mean(axis=1)
                    img_mean  = img_attn.mean(axis=1)

                    ax3_lang.plot(steps, lang_mean, color="tab:blue", linewidth=2)
                    ax3_lang.set_ylabel("Attention score")
                    ax3_lang.set_title("Mean attention → language tokens")
                    margin = (lang_mean.max() - lang_mean.min()) * 0.15 + 1e-6
                    ax3_lang.set_ylim(lang_mean.min() - margin, lang_mean.max() + margin)

                    ax3_img.plot(steps, img_mean, color="tab:orange", linewidth=2)
                    ax3_img.set_xlabel("Inference step")
                    ax3_img.set_ylabel("Attention score")
                    ax3_img.set_title("Mean attention → image tokens")
                    margin = (img_mean.max() - img_mean.min()) * 0.15 + 1e-6
                    ax3_img.set_ylim(img_mean.min() - margin, img_mean.max() + margin)

                    fig3.suptitle(f"{video_stem}\nprompt: {prompt_text}", fontsize=8)
                    fig3.tight_layout()
                    fig3.savefig(frame_dir / "attn_mean_2panel.png", dpi=120)
                    plt.close(fig3)

                # Log current results
                logging.info(f"Success: {done}")
                logging.info(f"Prompt '{prompt_text}' — episodes so far: {prompt_total_episodes[prompt_label]}, "
                             f"successes: {prompt_total_successes[prompt_label]}")

            logging.info(f"Task '{task_description}' | prompt '{prompt_text}' success rate: "
                         f"{float(prompt_successes) / float(prompt_episodes):.3f}")

    # Final summary across all prompts
    logging.info("\n=== Final results by prompt ===")
    for prompt_label in all_prompt_labels:
        prompt_text = "(original task description)" if prompt_label == ORIGINAL_PROMPT_SENTINEL else prompt_label
        n_ep = prompt_total_episodes[prompt_label]
        n_suc = prompt_total_successes[prompt_label]
        logging.info(f"  [{prompt_text}]  {n_suc}/{n_ep} = {n_suc/n_ep*100:.1f}%")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
