import numpy as np
import pandas as pd
from abc import ABC 

from pathlib import Path

if __name__ == "__main__":
    # Preprocess/Gather data
    def gather_data(path: Path|str)->None:
        """
        Gather a bunch of the raw data and put it in the path dir (could make path a default relative path)

        :param path: _description_
        """
        pass

# TODO: This file should be renamed to something more descriptive

# Gather data (Pull in raw data)
#   Should just ping Riot for json responses
    
    class data_aggregator(ABC):
        
        def format_json(raw_data: dict)->pd.DataFrame:
            pass
    
    class dummy_data(data_aggregator):
        
        col_names = ["something", "other"]
        
        @staticmethod
        def make_dummy_data():
            return pd.DataFrame()
        
        @staticmethod
        def format_json(raw_data):
            return self.make_dummy_data()
    
    class team_aggregation(data_aggregator):
        pass
    
    class role_aggregation(data_aggregator):
        pass
    
    class player_aggregation(data_aggregator):
        pass
        
# Format data, should take an algorithm and a file output type
#   Types: (These should be a class with a process_data method)
#       - Team Aggregation
#       - Role Aggregation
#       - Player/Champion Aggregation

    class rw_data(ABC):
        def __init__(self) -> None:
            file_end = ".csv"
            super().__init__()
        def read_data(path: Path):
            pass
        def write_data(path: Path):
            pass
        
    class parquet_rw(rw_data):
        def __init__(self) -> None:
            file_end = ".parquet"
            super().__init__()
        def read_data(path: Path):
            pass
        def write_data(path: Path):
            pass
    
    class db_rw(rw_data):
        pass
        
#   Ouptut Types: 
#       - Csv
#       - parquet
#       - sql

class TrainerEvaluator:
    def __init__(self, data: np.array, split: float = 0.2, layers: int = 2, size: int = 10) -> None:
        self.data = data
        self.split = split
        self.layers = layers
        self.size = size
    
    def train():
        pass
    
    def evaluate():
        pass

# Train on data 
#   Inputs - 
#       - Data format class
#       - Data to train on
#       - Size to make the network

# Evaluate the training
#   Inputs 
#       - new data 
#   Outputs
#       - what fraction of the snapshots were correct
#       - What snapshot the game was not wrong after

# Optimize Training
#   Inputs - 
#       - Data format class
#       - Data to train on
#   Explores
#       - Size and depth of network
#       - Could extend this to take a path to raw data and explore how different aggregations fair at predicting at different points in the game




