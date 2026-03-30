import collections
import csv
import dataclasses
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

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            attn_buffer = []  # per-inference-call attention info: list of {"lang_attn": [L], "img_attn": [L]}

            logging.info(f"Starting episode {task_episodes+1}...")
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
                            "prompt": str(task_description),
                        }

                        # Query model to get action (and collect attention info if available)
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        logging.info(f"[attn] result keys: {list(result.keys())}")
                        if "attn_info" in result:
                            attn_buffer.append(result["attn_info"])
                            logging.info(f"[attn] collected step {len(attn_buffer)}, lang_attn mean={result['attn_info']['lang_attn'].mean():.4f}")
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
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_stem = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            imageio.mimwrite(
                # pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
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
                lang_attn = np.stack([a["lang_attn"] for a in attn_buffer], axis=0)  # [T, L]
                img_attn = np.stack([a["img_attn"] for a in attn_buffer], axis=0)    # [T, L]
                np.save(frame_dir / "lang_attn_over_time.npy", lang_attn)
                np.save(frame_dir / "img_attn_over_time.npy", img_attn)

                # Save CSV log: one row per inference step
                num_layers = lang_attn.shape[1]
                with open(frame_dir / "attn_log.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    header = ["step"] + [f"lang_L{i}" for i in range(num_layers)] + ["lang_mean"] + \
                             [f"img_L{i}" for i in range(num_layers)] + ["img_mean"]
                    writer.writerow(header)
                    for step_idx, (lang_row, img_row) in enumerate(zip(lang_attn, img_attn)):
                        writer.writerow([step_idx] + lang_row.tolist() + [lang_row.mean()] +
                                        img_row.tolist() + [img_row.mean()])

                # Plot attention curves (mean over layers + per-layer)
                steps = np.arange(len(lang_attn))
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for layer_idx in range(num_layers):
                    axes[0].plot(steps, lang_attn[:, layer_idx], alpha=0.3, linewidth=0.8)
                    axes[1].plot(steps, img_attn[:, layer_idx], alpha=0.3, linewidth=0.8)
                axes[0].plot(steps, lang_attn.mean(axis=1), color="black", linewidth=2, label="mean")
                axes[1].plot(steps, img_attn.mean(axis=1), color="black", linewidth=2, label="mean")
                for ax, title in zip(axes, ["Language attention", "Image attention"]):
                    ax.set_xlabel("Inference step")
                    ax.set_ylabel("Attention score")
                    ax.set_title(title)
                    ax.legend()
                fig.suptitle(video_stem, fontsize=9)
                fig.tight_layout()
                fig.savefig(frame_dir / "attn_curve.png", dpi=120)
                plt.close(fig)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


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
