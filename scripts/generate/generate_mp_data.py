
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
    parser.add_argument("--range", type=str, help="Range of the generated data, must in xx-yy format", default=None)

    args = parser.parse_args()

    rank = args.rank
    input_file = args.input
    output_folder = args.output
    world_size = args.world_size
    data_range = args.range
    if data_range is not None:
        data_range_split = data_range.split("-")
        start, end = int(data_range_split[0]), int(data_range_split[1])
        subfolder = f"{start}-{end}"
        output_folder = os.path.join(output_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    with jsonlines.open(input_file) as reader:
        data = list(reader)
    
    if data_range is not None:
        data = data[start:end]

    data_iterator = islice(data, rank - 1, None, world_size)

    local_data = list(data_iterator)

    with jsonlines.open(os.path.join(output_folder, f"mp_data_{rank}.jsonl"), "w") as writer:
        writer.write_all(local_data)

