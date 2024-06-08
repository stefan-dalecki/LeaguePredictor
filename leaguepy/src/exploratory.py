
import pandas as pd
import json
import os

from pathlib import Path

from src.format_data import TeamAggregator

if __name__ == "__main__":

    model = TeamAggregator()
    data_dir = Path("data") / "raw"
    for file in data_dir.glob('*.json'):
        print(f"working on {file.stem}")
        with open(file) as json_file:
            data = json.load(json_file)
        model.format_json(data) # Add the data to the model
    a = 2+2
    
    print("stop here")
