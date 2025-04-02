import argparse
import math
import random
import time
import sys
import requests

import wandb


def main(attach_id: str, eval_step: int, project: str, num_steps: int = 10, max_retries: int = 3):
    try:
        # Configure wandb to retry on network errors
        wandb.init(
            id=attach_id,
            project=project,
            settings=wandb.Settings(
                mode="shared",
                x_stats_sampling_interval=0.1,
                x_label="eval",
                x_primary=False,
                # Set up retries for API requests
                _service_wait=5,  # Wait 5 seconds between retries
                _service_retries=max_retries,  # Retry 3 times
            ),
        )
        
        # Define metric relationship
        wandb.define_metric(name="eval_accuracy", step_metric="eval_step")

        current_step = eval_step
        for _ in range(num_steps):
            # value = min(math.log(current_step + 1) / 5 + random.random() / 20, 1)
            value = 0.2
            
            # Use retry logic for logging
            for retry in range(max_retries):
                try:
                    wandb.log({
                        "eval_accuracy": value,
                        "eval_step": current_step,
                    })
                    print(f"Successfully logged eval_step: {current_step}, eval_accuracy: {value}")
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 503 and retry < max_retries - 1:
                        wait_time = 2 ** retry * 2  # Exponential backoff
                        print(f"503 error encountered, retrying in {wait_time} seconds (attempt {retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise  # Re-raise if this was the last retry or a different error
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry * 2  # Exponential backoff
                        print(f"Error during logging: {e}, retrying in {wait_time} seconds (attempt {retry+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to log after {max_retries} attempts: {e}")
                        raise
            
            # Successfully logged this step, move to next
            time.sleep(1)
            current_step += 1

        # Ensure final sync with wait
        wandb.finish(exit_code=0, quiet=False)
        # Small delay to ensure everything syncs
        time.sleep(3)
        print("Evaluation completed successfully")
        
    except wandb.errors.CommError as e:
        print(f"WandB communication error: {e}")
        print("This could be a temporary server issue (503). Try again later.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 503:
            print(f"Service unavailable (503): The W&B service is temporarily unavailable")
        else:
            print(f"HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--attach_id", type=str, required=True)
    parser.add_argument("--project", type=str, default="distributed")
    parser.add_argument("--eval_step", type=int, required=True)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=3)

    args = parser.parse_args()

    main(**vars(args))