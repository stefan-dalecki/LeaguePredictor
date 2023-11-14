import numpy as np
import pandas as pd
from abc import ABC

from pathlib import Path

from src.format_data import DummyData
from src.reader_writer import ParquetRW
from src.train_model import TrainerEvaluator


if __name__ == "__main__":
    # Preprocess/Gather data
    # def gather_data(path: Path | str) -> None:
    #     """
    #     Gather a bunch of the raw data and put it in the path dir (could make path a default relative path)

    #     :param path: _description_
    #     """
    #     pass

    # Gather data (Pull in raw data)
    #   Should just ping Riot for json responses

    data, outcomes = DummyData.format_json({})

    out_df = Path(__file__).parent / "data" / "processed" / "first_df"
    ParquetRW.write_data(data, out_df)

    dnn_model = TrainerEvaluator(data.values, outcomes)
    dnn_model.train()
    accuracy = dnn_model.evaluate()
