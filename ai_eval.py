import argparse
import math
import random
import time

import wandb


def main(attach_id: str, eval_step: int, project: str):
    run = wandb.init(
        id=attach_id,
        project=project,
        settings=wandb.Settings(
            mode="shared",
            x_stats_sampling_interval=0.1,
            x_label="eval",
            x_primary=False,
        ),
    )
    run.define_metric(name="eval_accuracy", step_metric="eval_step")

    value = min(math.log(eval_step + 1) / 5 + random.random() / 20, 1)
    # value = 0.2
    run.log(
        {
            "eval_accuracy": value,
            "eval_step": eval_step,
        },
    )
    print(f"eval_step: {eval_step}, eval_accuracy: {value}")
    time.sleep(1)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--attach_id", type=str, required=True)
    parser.add_argument("--project", type=str, default="distributed")
    parser.add_argument("--eval_step", type=int, required=True)

    args = parser.parse_args()

    main(**vars(args))