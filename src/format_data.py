import json
import pandas as pd
import numpy as np
import os

import pandas as pd
from abc import ABC

from pathlib import Path
from typing import List, Tuple

import retrieve_data

# We'll want to consider several things eventually, cs, gold/gold per min, objectives, KDA (by member), etc. . However, these data aren't logged cumulatively, so we'll need to pull these out

_TEAMS = np.array([100, 200])

_ID_TEAM_MAP = dict(zip(range(1, 11), np.repeat(_TEAMS, 5)))

_EVENT_MAP = {
    "BUILDING_KILL": {
        "TOWER_BUILDING": "tower",
        "INHIBITOR_BUILDING": "inhibitor",
    },
    "ELITE_MONSTER_KILL": {
        "DRAGON": "dragon",
        "RIFTHERALD": "riftherald",
        "BARON_NASHOR": "baron",
        "ELDER_DRAGON": "elder",
    },
    "CHAMPION_KILL": {
        "killerId": "kills",
        "assistingParticipantIds": "assists",
        "victimId": "deaths",
    },
}

DECISION_COLS = [
    "cs",
    "gold",
    "tower",
    "inhibitor",
    "dragon",
    "riftherald",
    "baron",
    "elder",
    "kills",
    "assists",
    "deaths",
]


def get_dict_vals(dic, vals=[], rec=True):
    for val in dic.values():
        if type(val) == dict:
            get_dict_vals(val, vals)
        else:
            vals.append(val)
    return vals


_PLAYER_COLS = ["cs", "gold"]
_TEAM_COLS = get_dict_vals(_EVENT_MAP)


def make_ts_df(data: dict):
    try:
        frames = data["info"]["frames"]
    except KeyError:
        return pd.DataFrame()

    # Make DF with index for frame and team

    ind = pd.MultiIndex.from_product(
        [range(len(frames)), _TEAMS], names=["frame", "team"]
    )
    ts_data = pd.DataFrame(0, index=ind, columns=_PLAYER_COLS + _TEAM_COLS)
    for i, frame in enumerate(frames):
        events = frame["events"]
        players = frame["participantFrames"]

        # Gather and aggregate player level data
        # XP could be added here, but likely won't be a driving factor until there is a sizeable xp diff
        for player in players:
            ts_data.loc[i].loc[
                _ID_TEAM_MAP[players[player]["participantId"]], _PLAYER_COLS
            ] += [
                players[player]["jungleMinionsKilled"]
                + players[player]["minionsKilled"],
                players[player]["totalGold"],
            ]

        # Gather team/event level data
        for event in events:
            event_type = event["type"]
            if event_type in _EVENT_MAP:
                try:
                    team = (
                        event["teamId"]
                        if "teamId" in event
                        else _ID_TEAM_MAP[event["killerId"]]
                    )
                except KeyError:
                    # TODO: make this a logger
                    # print("No matching team found. Skipping")
                    continue
                if event_type == "CHAMPION_KILL":
                    killed_team = _ID_TEAM_MAP[event["victimId"]]
                    if killed_team == team:
                        print("what the")

                    # assert killed_team != team

                    pos_metrics = ["killerId", "assistingParticipantIds"]
                    neg_metrics = ["victimId"]
                    ts_data.loc[i].loc[
                        team, [_EVENT_MAP[event_type][metric] for metric in pos_metrics]
                    ] += [
                        1,
                        len(event.get(pos_metrics[1], [])),
                    ]
                    ts_data.loc[i].loc[
                        killed_team,
                        [_EVENT_MAP[event_type][metric] for metric in neg_metrics],
                    ] += [1]
                elif event_type == "ELITE_MONSTER_KILL":
                    ts_data.loc[i].loc[
                        team, _EVENT_MAP[event_type][event["monsterType"]]
                    ] += 1
                elif event_type == "BUILDING_KILL":
                    ts_data.loc[i].loc[
                        team, _EVENT_MAP[event_type][event["buildingType"]]
                    ] += 1

                # TODO: assert that team kills lte opposite teams deaths
    return ts_data


def get_winner(data: dict):
    try:
        frames = data["info"]["frames"]
        events = frames[-1]["events"]
        for event in events:
            if "winningTeam" in event:
                return event.get("winningTeam")
    except KeyError:
        return 0


def make_cumulative(df: pd.DataFrame, group_cols: List[str], val_cols: List[str]):
    try:
        output = df.copy()
        holder = df.groupby(group_cols)
        for col in val_cols:
            if col in df.columns:
                output[col] = holder[col].cumsum()
        return output
    except KeyError:
        return pd.DataFrame()


# TODO: Add format option (default pqt) for the output files
def write_games(data: dict, out_dir: str) -> None:
    """Wite out the cumulative timeseries data by game into out_dir

    Args:
        data (dict): A dictionary of timeseries data gathered from RIOT's Timeseries data API
        out_dir (str): Path to the location to write the outputs (outputs written by game id)
    """
    winner_data = dict()

    for i, round in enumerate(data):
        matchId = data[round]["metadata"]["matchId"]
        try:
            df = make_ts_df(data[round])
            test = make_cumulative(df, group_cols="team", val_cols=_TEAM_COLS)
            test.reset_index().to_parquet(Path(out_dir) / f"{matchId}.parquet")

            winner_data[matchId] = get_winner(data[round])
        # TODO: Make print into logger
        # TODO make except a non broad except and return it
        except:
            print("file failed")
        if i % 5 == 0:
            print(f"On {i}/{len(data)}")
    pd.DataFrame.from_dict(winner_data, orient="index").rename(
        columns={0: "winner"}
    ).to_parquet(Path(out_dir) / "win_log.parquet")


# TODO: Make function that reads in each game data, makes an array of 1xMetrics (one for each team) for each frame of the differences between teams on each metric
# and makes a corresponding array of 1xMetrics for which team won
# TODO: figure out what parameters would allow the model to have harder classifications at the begining and end (eg more sure 50/50 at start and more sure 100/0 at end)


# TODO: make a function that calls the above for all matches in matches history folder, and appends it to have an array
# of arrays. length of matches*frames of arrays of length Metrics. Should output an X and y dataset to train on


def format_snapshots(data: pd.DataFrame) -> np.ndarray:
    rows = data.shape[0]
    cols = data.shape[1]

    assert rows % 2 == 0

    frames = data.values.reshape((rows // 2, cols * 2))

    return frames


def make_outcome_data(
    matches: List[str], dir: Path = Path(".")
) -> Tuple[np.ndarray, np.ndarray]:
    win_log = pd.read_parquet(dir / "win_log.pqt")
    outcomes = np.zeros(len(matches))
    final_snapshots = np.zeros((len(matches), len(DECISION_COLS) * 2))

    for i, match in enumerate(matches):
        game_data = pd.read_parquet(match, columns=DECISION_COLS)
        frames = format_snapshots(game_data)
        final_snapshots[i] = frames[-1]
        outcomes[i] = win_log.loc[match.stem] == 100

    return final_snapshots, outcomes


class data_aggregator(ABC):
    def format_json(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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

if __name__ == "__main__":
    DIR = Path(
        r"C:\Users\jonhuster\Desktop\General\Personal\Projects\Python\LeaguePredictor\data\raw"
    )
    for j in range(3):
        with open(DIR / f"faker_norms_data_{j}.json") as json_file:
            data = json.load(json_file)
        write_games(data, out_dir=DIR.parent / "matches")

    print("all done")
