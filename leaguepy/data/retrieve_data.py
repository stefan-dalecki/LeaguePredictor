import logging
import logging.config
import requests
import json
from pathlib import Path

from leaguepy.src.constants import Region, Country, MatchType, URLS, RIOT_PARAMS, DEFAULT_LOGGER_CONFIG

# TODO: There should be a wrapper function for exponential wait (with cutoff) for these API calls

logging.config.dictConfig(DEFAULT_LOGGER_CONFIG)
logger = logging.getLogger(__name__)

def write_if_not_none(data, filename: Path | None = None) -> None:
    if data is not None:
        with open(filename, "w", encoding='utf8') as outfile:
            json.dump(data, outfile)


def get_puuid(username: str, country: Country = Country.KOREA, filename: Path | None = None):
    URL = URLS["puuid"]
    response = requests.get(
        URL.format(country=country.value, username=username), params=RIOT_PARAMS
    )
    logger.info(f"PUUID Retrieval Status Code: {response.status_code}")
    write_if_not_none(response.json(), filename)
    return response.json()["puuid"]


def get_match_history(
    puuid: str,
    region: Region = Region.ASIA,
    type: MatchType = MatchType.NORMAL,
    start: int = 0,
    count: int = 20,
    filename: Path | None = None,
) -> dict:
    URL = URLS["match_hist"]
    response = requests.get(
        URL.format(region=region.value, puuid=puuid, type=type.value, start=start, count=count),
        params=RIOT_PARAMS,
    )
    logger.info(f"Match History Retrieval Status Code: {response.status_code}")
    write_if_not_none(response.json(), filename)
    return response.json()


def get_timeline(match_id: str, region: Region = Region.ASIA, filename: Path | None = None) -> dict:
    URL = URLS["timeline"]
    response = requests.get(URL.format(region=region.value, match_id=match_id), params=RIOT_PARAMS)
    logger.info(f"Timeline Retrieval Status Code: {response.status_code}")
    write_if_not_none(response.json(), filename)
    return response.json()


def get_game_history(
    username: str,
    country: Country = Country.KOREA,
    region: Region = Region.ASIA,
    filename: Path | None = None,
    **args,
) -> dict:
    """This function takes a username and pulls the time series for their last 20 games and puts it into a json

    Args:
        username (str): The username of the person you would like to search
        country (Country, optional): The country of the player in question. Defaults to Country.korea.
        region (Region, optional): The region of the player in question. Defaults to Region.asia.
        filename (str, optional): The output path if desired. Defaults to None.

    Returns:
        dict: a dictionary of timeseries of (up to) the last 20 games
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

    DIR = Path(
        r"C:\Users\jonhuster\Desktop\General\Personal\Projects\Python\LeaguePredictor\data\raw"
    )
    for i in range(20):
        get_game_history(
            "is this a ward",
            country=Country.NORTH_AMERICA,
            region=Region.AMERICAS,
            filename=DIR / f"personal_norms_data_{2+i}.json",
            type=MatchType.NORMAL,
            start=200 + i * 100,
            count=100,
        )
        import time

        time.sleep(10)
