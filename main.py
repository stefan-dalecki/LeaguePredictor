


# TODO: This file should be renamed to something more descriptive

# Gather data (Pull in raw data)
#   Should just ping Riot for json responses

# Format data, should take an algorithm and a file output type
#   Types: (These should be a class with a process_data method)
#       - Team Aggregation
#       - Role Aggregation
#       - Player/Champion Aggregation
#   Ouptut Types: 
#       - Csv
#       - parquet
#       - sql

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




