
import jsonlines
import argparse
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="input folder")
    parser.add_argument("--output", "-o", type=str, help="output file")
    args = parser.parse_args()

    input_folder = args.input
    output_file = args.output

    data_files = glob(f"{input_folder}/*.jsonl")
    concat_data = []
    pbar = tqdm(total=len(data_files), desc="Processing data files")
    for data_file in data_files:
        with jsonlines.open(data_file) as reader:
            data = list(reader)
            concat_data.extend(data)
        pbar.update(1)
    pbar.close()
    with jsonlines.open(output_file, mode="w") as writer:
        writer.write_all(concat_data)