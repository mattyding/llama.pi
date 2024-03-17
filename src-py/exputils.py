import re
import numpy as np

NUM_LAYERS = 32


def parse_logs_for_file(filename):
    with open(f"./logs/{filename}") as f:
        time_data, cycle_data, num_tokens = parse_logs(f.readlines())
        return time_data, cycle_data, num_tokens


def compute_stats_for_file(filename):
    print(f"Processing {filename}...")
    with open(f"./logs/{filename}") as f:
        time_data, cycle_data, num_tokens = parse_logs(f.readlines())
        stats = compute_stats(time_data, cycle_data, num_tokens)
        return stats


def parse_logs(logs):
    """
    Args:
        logs: list of strings, each string is a line from the log file. use `open().readlines()`
    Returns:

    """
    num_tokens = len(re.findall(r"generated token", " ".join(logs)))
    time_data = np.zeros((num_tokens, NUM_LAYERS), dtype=np.uint64)
    cycle_data = np.zeros((num_tokens, NUM_LAYERS), dtype=np.uint64)

    curr_line_idx, curr_token_idx = -1, 0

    # always discard first read per token
    last_cycle, last_time = None, None
    # flag to avoid parsing final time print
    in_cycle = False
    uint64_max = np.uint64(2**64 - 1)  # overflow

    for line in logs:
        if "generated token" in line:
            curr_token_idx += 1
            curr_line_idx = -1  # off by one
            in_cycle = False
        if "starting layer" in line:
            in_cycle = True
            curr_line_idx += 1
        elif in_cycle and "Elapsed cycles" in line:
            cycle = np.uint64(re.findall(r"Elapsed cycles: (\d+)", line)[0])
            if last_cycle is not None:
                cycle_diff = (
                    cycle - last_cycle
                    if cycle > last_cycle
                    else uint64_max - last_cycle + cycle
                )
                cycle_data[curr_token_idx, curr_line_idx] = cycle_diff
            last_cycle = cycle
        elif in_cycle and "Elapsed time" in line:
            time = np.uint64(re.findall(r"Elapsed time: (\d+) us", line)[0])
            if last_time is not None:
                time_diff = (
                    time - last_time
                    if time > last_time
                    else uint64_max - last_time + time
                )
                time_data[curr_token_idx, curr_line_idx] = time_diff
            last_time = time

    return time_data, cycle_data, num_tokens


def compute_stats(time_data, cycle_data, num_tokens):
    # computes mean and std dev overall, per-layer, and per-token
    overall_time_mean = np.mean(time_data)
    overall_time_std = np.std(time_data)
    overall_cycle_mean = np.mean(cycle_data)
    overall_cycle_std = np.std(cycle_data)

    time_means = np.mean(time_data, axis=0)
    time_stds = np.std(time_data, axis=0)
    cycle_means = np.mean(cycle_data, axis=0)
    cycle_stds = np.std(cycle_data, axis=0)

    token_time_means = np.mean(time_data, axis=1)
    token_time_stds = np.std(time_data, axis=1)
    token_cycle_means = np.mean(cycle_data, axis=1)
    token_cycle_stds = np.std(cycle_data, axis=1)

    return {
        "overall_time_mean": overall_time_mean,
        "overall_time_std": overall_time_std,
        "overall_cycle_mean": overall_cycle_mean,
        "overall_cycle_std": overall_cycle_std,
        "layer_time_means": time_means,
        "layer_time_stds": time_stds,
        "layer_cycle_means": cycle_means,
        "layer_cycle_stds": cycle_stds,
        "token_time_means": token_time_means,
        "token_time_stds": token_time_stds,
        "token_cycle_means": token_cycle_means,
        "token_cycle_stds": token_cycle_stds,
    }
