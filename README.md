# League Predictor README

## Description

This project aims to predict the victor of a game of [Leage of Legends]() based on incomplete information. The core of the project to be able to train a learning/classification model on historical game data such that it is able to predict a winner from incomplete information. 


Currently the model pulls data from [Riot's API]() and aggregates it into a series of snapshots that are used, along with the final winner of each game, to train a neural network classification model that attempts to 'learn' what makes a winning situation vs a losing situation. The end goal of the model would be to use realtime data to predict the winner of an ongoing game. This could, hypothetically, take into account team/player history, champion picks, team postitioning, and current game status. 

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download Python from the official website: [Python.org](https://www.python.org/).

### Setting Up a Virtual Environment

It is a good practice to use a virtual environment to manage your project dependencies. This helps to keep your project isolated and avoids conflicts with other projects.

1. **Navigate to your project directory:**
    ```bash
    cd /path/to/your/project
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    This will create a directory named `venv` inside your project directory.

3. **Activate the virtual environment:**

    - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```

    After activation, you should see the name of the virtual environment in your terminal prompt, indicating that the virtual environment is active.

### Installing Requirements

This project uses a `pyproject.toml` file to specify its dependencies. You can install the required packages using `pip`.

1. **Ensure `pip` is up-to-date:**
    ```bash
    pip install --upgrade pip
    ```

2. **Install the requirements:**
    ```bash
    pip install -r requirements-dev.txt
    pip install -e .
    ```

    This command will install the dependencies specified in the `pyproject.toml` file located in the root of the repository.

### Running the Project

To run the model you will need access to RIOT's API. This is a free service that can be set up [here](). Obtain an API key used to request data from RIOT's servers and store the API key in a `.env` file in the root of this directory.

```bash
RIOT_API_KEY = "RIOT KEY HERE"
```

Once you have your API key stored in your `.env` file, call:

```bash
python main.py
```

This will run the main script which will gather, train, and evaluate the model. The package will gather 60 games of data from Faker (one of League of Legend's premier players), store the data in a local directory named `data`, process and aggregate that raw `json` data into parquet files, train a model on this historical data, then finally evaluate how the model performed on an unused portion of the downloaded data. 


## Contributing to the project

This project is open source and contributions are welcome to any areas of the code. There are ongoing discussions around the priority of different tasks located in the LEP.md. Feel free to contriubute with suggestions or commentary on future work within the project. 

### Following formatting

This project uses linting and formatting software including [ruff](https://docs.astral.sh/ruff/) and [black](https://black.readthedocs.io/en/stable/) to standardize the codebase. To ensure your code follows the same formatting used throughout the codebase you can simply call: 

```bash
pip install pre-commit
pre-commit install
```

This will install and setup pre-commit to run automatically on commit. 

