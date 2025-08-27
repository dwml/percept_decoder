import argparse
from pathlib import Path
from percept_decoder.percept_readout import PerceptReadOut
from percept_decoder.conditions import Condition
import random
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

script_description = """
This script will read the file in every row, take it's time domain data, and write it in
a file with an encoded name. The script will also output a csv file that links the
original data to the encoded name.
"""


def _datetime_attribute(readout: PerceptReadOut) -> datetime:
    return datetime.strptime(
        readout.first_packet_date_time, "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def main():
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument(
        "input_file",
        metavar="F",
        type=Path,
        help="A csv file containing the following headers: file, patient, condition "
             "and condition_id",
    )
    args = parser.parse_args()

    filename: Path = args.input_file

    root_path = filename.parent

    df = pd.read_excel(filename)

    number_of_files = len(df)

    random_indices = list(range(number_of_files))
    random.shuffle(random_indices)

    encoding_list = []

    decoded_path = root_path.joinpath("decoded")
    decoded_path.mkdir(parents=True, exist_ok=True)

    encoded_path = root_path.joinpath("encoded")
    encoded_path.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        full_file_path = root_path.joinpath(row.file)
        # rename for uniformity
        with open(full_file_path, "r") as fh:
            json_data = fh.read()

        readout = PerceptReadOut.model_validate_json(json_data)
        new_filename = f"{row.patient}_{Condition(row.condition_id).name}.json"
        shutil.copyfile(
            full_file_path,
            decoded_path.joinpath(new_filename),
        )

        # Split left and right channels
        all_left = [
            time_domain for time_domain in readout.brain_sense_time_domain
            if "LEFT" in time_domain.channel
               and len(time_domain.time_domain_data) > 10_000
        ]
        all_right = [
            time_domain for time_domain in readout.brain_sense_time_domain
            if "RIGHT" in time_domain.channel
               and len(time_domain.time_domain_data) > 10_000
        ]

        # From both left and right sort the list based on datetime and take the newest
        all_left.sort(key=_datetime_attribute)
        newest_left = all_left[-1]

        all_right.sort(key=_datetime_attribute)
        newest_right = all_right[-1]

        random_index = random_indices.pop()
        left_filename = f"{random_index:0>5}_left.npy"
        right_filename = f"{random_index:0>5}_right.npy"
        left_path = encoded_path.joinpath(left_filename)
        right_path = encoded_path.joinpath(right_filename)

        np.save(left_path, np.asarray(newest_left.time_domain_data))
        np.save(right_path, np.asarray(newest_right.time_domain_data))

        encoding_list.append(
            {
                "original_file": new_filename,
                "patient": row.patient,
                "condition": row.condition,
                "condition_id": row.condition_id,
                "left_file": left_filename,
                "left_channel": newest_left.channel,
                "left_first_packet_date_time": newest_left.first_packet_date_time,
                "left_sample_rate_in_hz": newest_left.sample_rate_in_hz,
                "right_file": right_filename,
                "right_channel": newest_right.channel,
                "right_first_packet_date_time": newest_right.first_packet_date_time,
                "right_sample_rate_in_hz": newest_right.sample_rate_in_hz,
            }
        )

    decoding_filepath = root_path.joinpath("./decoding_file.xlsx")

    decoding_file = pd.DataFrame(encoding_list)
    decoding_file.to_excel(decoding_filepath)
