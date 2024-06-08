import json
import numpy as np
import pandas as pd
from abc import ABC

from pathlib import Path

from leaguepy.src.format_data import DummyData, TeamAggregator
from leaguepy.src.reader_writer import ParquetRW
from leaguepy.src.train_model import TrainerEvaluator


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

    model = TeamAggregator()
    data_dir = Path("data") / "raw"
    for j in range(3):
        with open(data_dir / f"faker_norms_data_{j}.json") as json_file:
            data = json.load(json_file)
        model.format_json(data) # Add the data to the model
        del data

    storage_path = Path(__file__).parent / "data" / "processed" / "aggregated_games"
    ParquetRW.write_data(model.df, storage_path)

    X, y = model.prepare_train()
    dnn_model = TrainerEvaluator(X.astype(float), y, size=10, layers=2)
    dnn_model.train()
    accuracy = dnn_model.evaluate()
