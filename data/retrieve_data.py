import requests
import json
import os

from dotenv import load_dotenv
from enum import Enum
from pathlib import Path

#TODO: There should be a wrapper function for exponential wait (with cutoff) for these API calls

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
# To generate a key go to this web address, make an account, and regenerate a developement api key
# https://developer.riotgames.com/

RIOT_PARAMS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": f"{API_KEY}",
    "api_key": f"{API_KEY}",
}


class Region(Enum):
    americas = "americas"
    asia = "asia"
    europe = "europe"
    sea = "sea"


class Country(Enum):
    korea = "kr"
    europe_north = "EUN1"
    europe_west = "EUW1"
    japan = "JP1"
    north_america = "NA1"


class Type(Enum):
    normal = "normal"
    ranked = "ranked"


URLS = {
    "puuid": "https://{country}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{username}",
    "match_hist": "https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type={type}&start={start}&count={count}",
    "timeline": "https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline",
}


def write_if_not_none(data, filename: str = None) -> None:
    if filename is not None:
        with open(filename, "w") as outfile:
            json.dump(data, outfile)


def get_puuid(username: str, country: Country = Country.korea, filename: str = None):
    URL = URLS["puuid"]
    response = requests.get(
        URL.format(country=country.value, username=username), params=RIOT_PARAMS
    )
    # TODO: this should be a log not a print
    print(response.status_code)
    write_if_not_none(response.json(), filename)
    return response.json()["puuid"]


def get_match_history(
    puuid: str,
    region: Region = Region.asia,
    type: Type = Type.normal,
    start: int = 0,
    count: int = 20,
    filename: str = None,
) -> list:
    URL = URLS["match_hist"]
    response = requests.get(
        URL.format(
            region=region.value, puuid=puuid, type=type.value, start=start, count=count
        ),
        params=RIOT_PARAMS,
    )
    # TODO: this should be a log not a print
    print(response.status_code)
    write_if_not_none(response.json(), filename)
    return response.json()


def get_timeline(
    match_id: str, region: Region = Region.asia, filename: str = None
) -> dict:
    URL = URLS["timeline"]
    response = requests.get(
        URL.format(region=region.value, match_id=match_id), params=RIOT_PARAMS
    )
    # TODO: this should be a log not a print
    print(f"timeline: {response.status_code}")
    write_if_not_none(response.json(), filename)
    return response.json()


def get_game_history(
    username: str,
    country: Country = Country.korea,
    region: Region = Region.asia,
    filename: str = None,
    **args,
) -> json:
    """This function takes a username and pulls the time series for their last 20 games and puts it into a json

    Args:
        username (str): The username of the person you would like to search
        country (Country, optional): The country of the player in question. Defaults to Country.korea.
        region (Region, optional): The region of the player in question. Defaults to Region.asia.
        filename (str, optional): The output path if desired. Defaults to None.

    Returns:
        json: a dictionary of timeseries of (up to) the last 20 games
    """
    puuid = get_puuid(username, country)
    match_history = get_match_history(puuid, region=region, **args)
    timeseries = dict()
    for i, match in enumerate(match_history):
        # TODO: Have the functions return codes as well to indicate if we should wait
        timeseries[i] = get_timeline(match, region=region)

    write_if_not_none(timeseries, filename)
    return timeseries


# TODO: This code doesn't have a way of systematically retrieving different data. right now if we ran get_game_history it would grab 20 games and forever grab the same games. We need a way to systematically grab other data for the same players
if __name__ == "__main__":
    import time
    DIR = Path(r"C:\Users\jonhuster\Desktop\General\Personal\Projects\Python\LeaguePredictor\data\raw")
    for i in range(20):
        get_game_history(
            "is this a ward",
            country=Country.north_america,
            region=Region.americas,
            filename=DIR/f"personal_norms_data_{2+i}.json",
            type=Type.normal,
            start=200+i*100,
            count=100,
        )
        import time
        time.sleep(10) 
