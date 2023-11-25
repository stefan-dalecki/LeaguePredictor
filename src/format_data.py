import json
import pandas as pd
import numpy as np

import pandas as pd
from abc import ABC

from pathlib import Path

# We'll want to consider several things eventually, cs, gold/gold per min, objectives, KDA (by member), etc. . However, these data aren't logged cumulatively, so we'll need to pull these out

_TEAMS = np.array([100, 200])

_ID_TEAM_MAP = dict(zip(range(1, 11), np.repeat(_TEAMS, 5)))

# _EVENT_MAP = {
#     "BUILDING_KILL": {
#         "TOWER_BUILDING": "tower",
#         "INHIBITOR_BUILDING": "inhibitor",
#     },
#     "ELITE_MONSTER_KILL": {
#         "DRAGON": "dragon",
#         "RIFTHERALD": "riftherald",
#         "BARON_NASHOR": "baron",
#         "ELDER_DRAGON": "elder",
#     },
#     "CHAMPION_KILL": {
#         "killerId": "kills",
#         "assistingParticipantIds": "assists",
#         "victimId": "deaths",
#     },
# }

# DECISION_COLS = [
#     "cs",
#     "gold",
#     "tower",
#     "inhibitor",
#     "dragon",
#     "riftherald",
#     "baron",
#     "elder",
#     "kills",
#     "assists",
#     "deaths",
# ]


# def get_dict_vals(dic, vals=[], rec=True):
#     for val in dic.values():
#         if type(val) == dict:
#             get_dict_vals(val, vals)
#         else:
#             vals.append(val)
#     return vals


# _PLAYER_COLS = ["cs", "gold"]
# _TEAM_COLS = get_dict_vals(_EVENT_MAP)


# def make_ts_df(data: dict):
#     try:
#         frames = data["info"]["frames"]
#     except KeyError:
#         return pd.DataFrame()

#     # Make DF with index for frame and team

#     ind = pd.MultiIndex.from_product(
#         [range(len(frames)), _TEAMS], names=["frame", "team"]
#     )
#     ts_data = pd.DataFrame(0, index=ind, columns=_PLAYER_COLS + _TEAM_COLS)
#     for i, frame in enumerate(frames):
#         events = frame["events"]
#         players = frame["participantFrames"]

#         # Gather and aggregate player level data
#         # XP could be added here, but likely won't be a driving factor until there is a sizeable xp diff
#         for player in players:
#             ts_data.loc[i].loc[
#                 _ID_TEAM_MAP[players[player]["participantId"]], _PLAYER_COLS
#             ] += [
#                 players[player]["jungleMinionsKilled"]
#                 + players[player]["minionsKilled"],
#                 players[player]["totalGold"]/(3000*5),#Roughly the cost of one completed item
#             ]

#         # Gather team/event level data
#         for event in events:
#             event_type = event["type"]
#             if event_type in TeamAggregator.event_map:
#                 try:
#                     team = (
#                         event["teamId"]
#                         if "teamId" in event
#                         else _ID_TEAM_MAP[event["killerId"]]
#                     )
#                 except KeyError:
#                     # TODO: make this a logger
#                     # print("No matching team found. Skipping")
#                     continue
#                 if event_type == "CHAMPION_KILL":
#                     killed_team = _ID_TEAM_MAP[event["victimId"]]
#                     if killed_team == team:
#                         print("what the")

#                     # assert killed_team != team

#                     pos_metrics = ["killerId", "assistingParticipantIds"]
#                     neg_metrics = ["victimId"]
#                     ts_data.loc[i].loc[
#                         team, [_EVENT_MAP[event_type][metric] for metric in pos_metrics]
#                     ] += [
#                         1,
#                         len(event.get(pos_metrics[1], [])),
#                     ]
#                     ts_data.loc[i].loc[
#                         killed_team,
#                         [_EVENT_MAP[event_type][metric] for metric in neg_metrics],
#                     ] += [1]
#                 elif event_type == "ELITE_MONSTER_KILL":
#                     ts_data.loc[i].loc[
#                         team, _EVENT_MAP[event_type][event["monsterType"]]
#                     ] += 1
#                 elif event_type == "BUILDING_KILL":
#                     ts_data.loc[i].loc[
#                         team,
#                         TeamAggregator.event_map[event_type][event["buildingType"]],
#                     ] += 1

#                 # TODO: assert that team kills lte opposite teams deaths
#     return ts_data


# def get_winner(data: dict):
#     try:
#         frames = data["info"]["frames"]
#         events = frames[-1]["events"]
#         for event in events:
#             if "winningTeam" in event:
#                 return event.get("winningTeam")
#     except KeyError:
#         return 0


def get_winner(game: dict[str, dict]) -> int:
    """
    return the winning team of a game from a dict of the game

    :param game: a dictionary of metadata and events in a game
    :return: either 100 or 200 to indicate the winning team
    """
    return game["info"]["frames"][-1]["events"][-1]["winningTeam"]


# def make_cumulative(df: pd.DataFrame, group_cols: List[str], val_cols: List[str]):
#     try:
#         output = df.copy()
#         holder = df.groupby(group_cols)
#         for col in val_cols:
#             if col in df.columns:
#                 output[col] = holder[col].cumsum()
#         return output
#     except KeyError:
#         return pd.DataFrame()


# # TODO: Add format option (default pqt) for the output files
# def write_games(data: dict, out_dir: str) -> None:
#     """Wite out the cumulative timeseries data by game into out_dir

#     Args:
#         data (dict): A dictionary of timeseries data gathered from RIOT's Timeseries data API
#         out_dir (str): Path to the location to write the outputs (outputs written by game id)
#     """
#     winner_data = dict()

#     for i, round in enumerate(data):
#         matchId = data[round]["metadata"]["matchId"]
#         try:
#             df = make_ts_df(data[round])
#             test = make_cumulative(df, group_cols="team", val_cols=_TEAM_COLS)
#             test.reset_index().to_parquet(Path(out_dir) / f"{matchId}.parquet")

#             winner_data[matchId] = get_winner(data[round])
#         # TODO: Make print into logger
#         # TODO make except a non broad except and return it
#         except:
#             print("file failed")
#         if i % 5 == 0:
#             print(f"On {i}/{len(data)}")
#     pd.DataFrame.from_dict(winner_data, orient="index").rename(
#         columns={0: "winner"}
#     ).to_parquet(Path(out_dir) / "win_log.parquet")


# # TODO: Make function that reads in each game data, makes an array of 1xMetrics (one for each team) for each frame of the differences between teams on each metric
# # and makes a corresponding array of 1xMetrics for which team won
# # TODO: figure out what parameters would allow the model to have harder classifications at the begining and end (eg more sure 50/50 at start and more sure 100/0 at end)


# # TODO: make a function that calls the above for all matches in matches history folder, and appends it to have an array
# # of arrays. length of matches*frames of arrays of length Metrics. Should output an X and y dataset to train on


# def format_snapshots(data: pd.DataFrame) -> np.ndarray:
#     """
#     Return a numpy array where the first n/2 columns come from team one's data the second half comes from
#     team 2's data and each row is a snapshot of the game at a particular time

#     :param data: _description_
#     :return: _description_
#     """
#     rows = data.shape[0]
#     cols = data.shape[1]

#     assert rows % 2 == 0

#     frames = data.values.reshape((rows // 2, cols * 2))

#     return frames


# def make_outcome_data(
#     matches: List[str], dir: Path = Path(".")
# ) -> Tuple[np.ndarray, np.ndarray]:
#     win_log = pd.read_parquet(dir / "win_log.pqt")
#     outcomes = np.zeros(len(matches))
#     final_snapshots = np.zeros((len(matches), len(DECISION_COLS) * 2))

#     for i, match in enumerate(matches):
#         game_data = pd.read_parquet(match, columns=DECISION_COLS)
#         frames = format_snapshots(game_data)
#         final_snapshots[i] = frames[-1]
#         outcomes[i] = win_log.loc[match.stem] == 100

#     return final_snapshots, outcomes


class DataAggregator(ABC):
    def format_json(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass


class DummyData(DataAggregator):

    col_names = ["something", "other"]

    @staticmethod
    def make_dummy_data(n: int = 10_000) -> pd.DataFrame:
        data = np.random.randint(0, 100, (n, len(DummyData.col_names)))
        df = pd.DataFrame(columns=DummyData.col_names, data=data)
        return df

    @staticmethod
    def format_json(raw_data):
        data = DummyData.make_dummy_data()
        wins = np.random.choice(2, len(data))
        return data, wins


class TeamAggregator(DataAggregator):
    # TODO: Make these into a single dict that maps all these attributes to avoid 
    # duplicate rows for naming
    index = ["game_id", "timestamp"]
    event_map = {
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
    
    standard_values = {
        "cs":{
            'norm_val': 10*30*5,# Could instead normalize to CS per minute
            'pre_agg': True,
            'event_map':None, 
            }, 
        "gold":{
            'norm_val': 3000*5, # Average cost per item
            'pre_agg': True,
            'event_map':None, 
            },
        "tower": {
            'norm_val': 8, 
            'pre_agg': False,
            'event_map':{
                'event_type':'BUILDING_KILL',
                'event_name':'TOWER_BUILDING',
                }, 
            }, 
        "inhibitor": {
            'norm_val': 3, 
            'pre_agg': False,
            'event_map':{
                'event_type':'BUILDING_KILL',
                'event_name':'INHIBITOR_BUILDING',
                }, 
            }, 
        "dragon": {
            'norm_val': 4, 
            'pre_agg': False,
            'event_map':{
                'event_type':'ELITE_MONSTER_KILL',
                'event_name':'DRAGON',
                }, 
            }, 
        "riftherald": {
            'norm_val': 1, 
            'pre_agg': False,
            'event_map':{
                'event_type':'ELITE_MONSTER_KILL',
                'event_name':'RIFTHERALD',
                }, 
            }, 
        "baron": {
            'norm_val': 1, 
            'pre_agg': False,
            'event_map':{
                'event_type':'ELITE_MONSTER_KILL',
                'event_name':'BARON_NASHOR',
                }, 
            }, 
        "elder": {
            'norm_val': 1, 
            'pre_agg': False,
            'event_map':{
                'event_type':'ELITE_MONSTER_KILL',
                'event_name':'ELDER_DRAGON',
                }, 
            },
        "kills": {
            'norm_val': 15, 
            'pre_agg': False,
            'event_map':{
                'event_type':'CHAMPION_KILL',
                'event_name':'killerId',
                }, 
            },
        "assists": {
            'norm_val': 39, 
            'pre_agg': False,
            'event_map':{
                'event_type':'CHAMPION_KILL',
                'event_name':'assistingParticipantIds',
                }, 
            },
        "deaths": {
            'norm_val': 15, 
            'pre_agg': False,
            'event_map':{
                'event_type':'CHAMPION_KILL',
                'event_name':'victimId',
                }, 
            },
    }

    cum_team_cols = [f"{col}_{team}" for col in cum_cols for team in _TEAMS]
    multi_index = pd.MultiIndex.from_tuples([], names=index)

    def __init__(self) -> None:
        self.df = pd.DataFrame(
            columns=TeamAggregator.team_columns, index=TeamAggregator.multi_index
        )
        self.outcomes = pd.DataFrame(
            columns=["outcome"], index=TeamAggregator.multi_index
        )
        super().__init__()
        
    @property
    @classmethod
    def event_map(cls):
        events = dict()
        for key, value in cls.standard_values.values:
            if not value.get('event_map'):
                continue
            event_type = value['event_map']['event_type']
            event_name = value['event_map']['event_name']
            if events.get(event_type):
                events[event_type][event_name] = key
            else:
                events[event_type] = {event_name: key}
        return events
    
    @property
    @classmethod
    def normal_df(cls):
        pass
    
    @property
    @classmethod
    def team_columns(cls):
        return [f"{col}_{team}" for col in cls.standard_values for team in _TEAMS]
    
    @property
    @classmethod
    def cum_columns(cls):
        return [f"{col}_{team}" for col in cls.standard_values for team in _TEAMS if not cls.standard_values['col']['pre_agg']]

    @staticmethod
    def add_frame(
        df: pd.DataFrame, idx: tuple[str, str], frame: list[dict]
    ) -> pd.DataFrame:
        # TODO: This doesn't actually need to know about the dataframe. It should just return an ordered series that we add to the df in the add_game function
        events = frame["events"] or []
        player_frames = frame["participantFrames"] or dict()

        df.loc[idx, :] = 0

        # Add data from player snapshots
        for player in player_frames.values():
            team = _ID_TEAM_MAP[player["participantId"]]
            df.loc[idx, f"cs_{team}"] += (
                player["jungleMinionsKilled"] + player["minionsKilled"]
            )
            df.loc[idx, f"gold_{team}"] += player["totalGold"] 
            # TODO: XP could be added here, but likely won't be a driving factor until there is a sizeable xp diff
            # Would maybe be easier to do with just lv difference, but at a minute scale it would be hard
            # To catch the difference

        # Add data from event snapshots
        for event in events:
            event_type = event["type"]
            if event_type not in TeamAggregator.event_map:
                continue
            try:
                team = (
                    event["teamId"]
                    if "teamId" in event
                    else _ID_TEAM_MAP[event["killerId"]]
                )
            except KeyError:
                # Event not affiliated with a team. Skip to next
                continue

            # CONTINUE HERE
            if event_type == "ELITE_MONSTER_KILL":
                monster = TeamAggregator.event_map[event_type][event["monsterType"]]
                df.loc[idx, f"{monster}_{team}"] += 1
            if event_type == "BUILDING_KILL":
                monster = TeamAggregator.event_map[event_type][event["buildingType"]]
                df.loc[idx, f"{monster}_{team}"] += 1
            if event_type == "CHAMPION_KILL":
                killed_team = _ID_TEAM_MAP[event["victimId"]]
                if killed_team == team:
                    print("what the")
                # assert killed_team != team

                df.loc[idx, f"kills_{team}"] += 1
                df.loc[idx, f"assists_{team}"] += (
                    len(event.get("assistingParticipantIds", []))
                )
                df.loc[idx, f"deaths_{killed_team}"] += 1

    @staticmethod
    def format_game(game: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:

        game_df = pd.DataFrame(
            columns=TeamAggregator.team_columns, index=TeamAggregator.multi_index
        )
        outcomes = pd.DataFrame(columns=["outcome"], index=TeamAggregator.multi_index)

        game_id = game["info"]["gameId"]
        frames = game["info"]["frames"]

        winner = get_winner(game)
        second_half_frames = frames[int(len(frames) / 2) :]
        for frame in second_half_frames:
            idx = (game_id, frame["timestamp"] / (60 * 1000 * 30))
            TeamAggregator.add_frame(game_df, idx, frame)
            outcomes.loc[idx, "outcome"] = winner / 100 - 1
        game_df.loc[:, TeamAggregator.cum_team_cols] = game_df.loc[
            :, TeamAggregator.cum_team_cols
        ].agg(np.cumsum)
        return game_df, outcomes

    def format_json(self, raw_data: dict[str, dict]) -> pd.DataFrame:
        for game in raw_data.values():
            try:
                game_df, outcome_df = self.format_game(game)
            except KeyError:
                # The format of the game was malformed (eg we got a message in our json instead of a game)
                continue
            self.df = pd.concat([self.df, game_df])
            self.outcomes = pd.concat([self.outcomes, outcome_df])
        # TODO: Explore how aggregation of the results changes the outcomes

    def normalize(self):
        normalizer = pd.DataFrame(TeamAggregator.standard_values.values, index= self.df.index, columns=TeamAggregator.columns)
        
        # Assume we can get the data into a pandas dataframe with the same columsn
        
        self.df = self.df.div(normalizer)

    def prepare_train(self) -> tuple[np.ndarray, np.ndarray]:
        X = self.df.reset_index().drop("game_id", axis=1).values
        y = self.outcomes.values.reshape((len(self.outcomes),)).astype(int)
        return X, y


class RoleAggregator(DataAggregator):
    pass


class PlayerAggregator(DataAggregator):
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
    model = TeamAggregator()
    for j in range(3):
        with open(DIR / f"faker_norms_data_{j}.json") as json_file:
            data = json.load(json_file)
        model.format_json(data)
        # write_games(data, out_dir=DIR.parent / "matches")

    print("all done")
