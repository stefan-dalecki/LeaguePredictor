import pandas as pd

from abc import ABC
from pathlib import Path


class DataRW(ABC):
    suffix = ".any"

    @staticmethod
    def read_data(path: Path) -> None:
        pass

    @staticmethod
    def write_data(path: Path) -> None:
        pass


class CSVRW(DataRW):
    suffix = ".csv"

    @staticmethod
    def read_data(path: Path) -> None:
        pass

    @staticmethod
    def write_data(path: Path) -> None:
        pass


class ParquetRW(DataRW):
    suffix = ".parquet"

    @staticmethod
    def read_data(path: Path):
        return pd.read_parquet(path.with_suffix(ParquetRW.suffix))

    @staticmethod
    def write_data(data: pd.DataFrame, path: Path, **kwargs):
        data.to_parquet(path.with_suffix(ParquetRW.suffix), **kwargs)


class DBRW(DataRW):
    pass
