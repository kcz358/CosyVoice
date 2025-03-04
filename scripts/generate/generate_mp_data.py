
import argparse
import jsonlines
import os
import torch
import torch.distributed as dist
from itertools import islice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="input file")
    parser.add_argument("--output", "-o", type=str, help="output folder")
    parser.add_argument("--rank", "-r", type=int, help="rank")
    parser.add_argument("--world_size", "-w", type=int, help="world size")

    args = parser.parse_args()

    rank = args.rank
    input_file = args.input
    output_folder = args.output
    world_size = args.world_size
    os.makedirs(output_folder, exist_ok=True)

    with jsonlines.open(input_file) as reader:
        data = list(reader)

    data_iterator = islice(data, rank - 1, None, world_size)

    local_data = list(data_iterator)

    with jsonlines.open(os.path.join(output_folder, f"mp_data_{rank}.jsonl"), "w") as writer:
        writer.write_all(local_data)

