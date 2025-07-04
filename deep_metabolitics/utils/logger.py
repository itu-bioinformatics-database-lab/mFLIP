import json
import logging
import os
import re
import sys
from collections import defaultdict

import pandas as pd

from deep_metabolitics.config import logs_dir


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "lineno": record.lineno,
        }
        return json.dumps(log_record)


def create_logger(name, remove=False):
    log_fpath = logs_dir / f"{name}.log"
    if remove:
        if os.path.exists(log_fpath):
            os.remove(log_fpath)
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    stream_formatter = logging.Formatter(
        fmt=f"%(levelname)-8s %(asctime)s \t line %(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_fpath)
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def load_json_lines(fpath):
    with open(fpath, "r") as file:
        lines = file.readlines()
    return lines


def clean_message(message):
    try:
        message = re.sub(r"\s+", " ", message)
        message = (
            message.replace("array(", "[")
            .replace(")", "]")
            .replace("tensor(", "[")
            .replace(", device='cuda:0'", "")
            .replace(".,", ",")
            .replace("'", '"')
            .replace("\n", "")
            .replace(', dtype="<U2"', "")
            .replace("np.int64(", "[")
            .replace("None", '"None"')
            .replace("\n", "")
            .replace("np.str_(", "[")
            .replace("np.float64(", "[")
            .replace("datetime.datetime(", "[")
            .replace("nan", '"nan"')
            .replace("array(", "[")
            .replace(")", "]")
            .replace("tensor(", "[")
            .replace(", device='cuda:0'", "")
            .replace(".,", ",")
            .replace("'", '"')
            .replace("\n", "")
            .replace(', dtype="<U2"', "")
            .replace("np.int64(", "[")
            .replace("\n", "")
            .replace("np.str_(", "[")
            .replace("np.float64(", "[")
            .replace("np.float32(", "[")
            .replace("datetime.datetime(", "[")
        )

        message = json.loads(message)
        running_for = message.get("running_for")
    except Exception as e:
        print(f"ERROR FOR {message = }, {e = }")
        raise e
        # return message
    return running_for, message


def get_cleaned_message(lines):
    message_map = defaultdict(list)
    for line in lines:
        if (("running_for" in line) and ("mse" in line)) or ("classification" in line):
            message = json.loads(line).get("message")

            running_for, message = clean_message(message)
            message_map[running_for].append(message)

    return message_map


def get_item(value):
    if isinstance(value, list):
        return value[0]
    return value


def read_log_file(experiment_name):
    log_experiment_path = logs_dir / f"{experiment_name}.log"
    lines = load_json_lines(fpath=log_experiment_path)
    message_map = get_cleaned_message(lines=lines)
    train_df = pd.DataFrame(message_map["TRAIN"])
    validation_df = pd.DataFrame(message_map["VALIDATION"])
    test_df = pd.DataFrame(message_map["TEST"])
    classification_df = pd.DataFrame(message_map["classification"])

    for column in train_df.columns:
        train_df[column] = train_df[column].apply(get_item)

    for column in validation_df.columns:
        validation_df[column] = validation_df[column].apply(get_item)

    for column in test_df.columns:
        test_df[column] = test_df[column].apply(get_item)

    for column in classification_df.columns:
        classification_df[column] = classification_df[column].apply(get_item)
    return train_df, validation_df, test_df, classification_df
