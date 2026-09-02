"""Play entry point for the quadruped HIM AMP heightmap task."""

import os

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import play_quad_arm_him_amp as amp_play


DEFAULT_TASK = "quadruped_arm_him_amp_heightmap"
DEFAULT_LOAD_RUN = None
DEFAULT_CHECKPOINT = -1
DEFAULT_NUM_ENVS = 1

# Playback controls. These are forwarded to the shared AMP play implementation.
USE_NET = True
EXPORT_POLICY = False
RENDER = False
FIX_COMMAND = False
KEYBOARD_ON = True
MOVE_CAMERA = False
HANG_ON = False
RANDOM_ON = False


def main():
    amp_play.USE_NET = USE_NET
    amp_play.EXPORT_POLICY = EXPORT_POLICY
    amp_play.RENDER = RENDER
    amp_play.FIX_COMMAND = FIX_COMMAND
    amp_play.KEYBOARD_ON = KEYBOARD_ON
    amp_play.MOVE_CAMERA = MOVE_CAMERA
    amp_play.HANG_ON = HANG_ON
    amp_play.RANDOM_ON = RANDOM_ON

    args = amp_play.get_args()
    args.play_flag = True
    args.control_test = KEYBOARD_ON and not FIX_COMMAND
    args.task = DEFAULT_TASK
    args.experiment_name = DEFAULT_TASK

    if args.load_run is None:
        raise ValueError(
            "Pass --load_run from a model trained with the WTW-aligned 6x11 "
            "height scan; the existing 11x11 checkpoints are incompatible."
        )
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT
    if args.num_envs is None:
        args.num_envs = DEFAULT_NUM_ENVS

    amp_play.play(args)


if __name__ == "__main__":
    main()
