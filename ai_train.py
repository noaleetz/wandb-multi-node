import argparse
import math
import os
import pathlib
import random
import subprocess
import time

import tqdm

import wandb


def main(
    project: str = "distributed",
    sleep: int = 1,
):
    run = wandb.init(
        # id="e4zkyv64",
        project=project,
        settings=wandb.Settings(
            mode="shared",
            init_timeout=60,
            x_stats_sampling_interval=1,
            x_label="train",
            x_primary=True,
        ),
    )
    print("run_id:", run.id)

    run.define_metric(name="loss", step_metric="train_step")
    # for the eval job:
    run.define_metric(name="eval_accuracy", step_metric="eval_step")

    bar = tqdm.tqdm()
    train_step = 0
    eval_step = 0
    while True:
        try:
            value = math.exp(-train_step / 100) + random.random() / 20
            run.log(
                {
                    "train_step": train_step,
                    "loss": value,
                }
            )
            bar.update(1)
            train_step += 1
            time.sleep(sleep)

            # kick-off evaluation
            if train_step % 10 == 0:
                print("Mesa gonna eval")
                print("eval_step:", eval_step)
                eval_path = pathlib.Path(__file__).parent / "ai_eval.py"
                subprocess.run(
                    [
                        "python",
                        eval_path,
                        "--attach_id",
                        run.id,
                        "--eval_step",
                        str(eval_step),
                        "--num_steps",
                        "1",  # Just log one step per evaluation
                    ],
                    # reset WANDB_SERVICE so that it spins its own wandb-core
                    # this is done to mimic a multi-node scenario.
                    env={**os.environ, **{"WANDB_SERVICE": ""}},
                )
                eval_step += 1

        except KeyboardInterrupt:
            bar.close()
            break

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="distributed")
    parser.add_argument("--sleep", type=int, default=1)

    args = parser.parse_args()

    main(**vars(args))
