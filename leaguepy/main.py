import json
import logging
from pathlib import Path

from leaguepy.src.format_data import TeamAggregator
from leaguepy.src.reader_writer import ParquetRW
from leaguepy.src.train_model import TrainerEvaluator
from leaguepy.src.constants import DEFAULT_LOGGER_CONFIG

logging.config.dictConfig(DEFAULT_LOGGER_CONFIG)
logger = logging.getLogger(__name__)


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
    data_dir = Path(__file__).parent / "data" / "raw"
    logger.info(f"Reading data from {data_dir}")
    for j in range(3):
        logger.info(f"Processing file {j+1} of 3")
        with open(data_dir / f"faker_norms_data_{j}.json", encoding="utf8") as json_file:
            data = json.load(json_file)
        model.format_json(data)  # Add the data to the model
        del data

    storage_path = Path(__file__).parent / "data" / "processed" / "aggregated_games"
    ParquetRW.write_data(model.df, storage_path)

    logger.info("Beginning to Train model")
    X, y = model.prepare_train()
    dnn_model = TrainerEvaluator(X.astype(float), y, size=10, layers=2)
    dnn_model.train()
    logger.info("Logger Trained")
    accuracy = dnn_model.evaluate()
    logger.info(f"Model Accuracy of {accuracy}")
