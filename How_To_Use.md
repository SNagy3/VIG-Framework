# How to Use

## Important Setup Requirement

Before running the framework, **all required CSV files and `vig_master_driver.py` must be placed in the same destination directory**.

The Master Driver reads the CSV inputs from that shared location and generates the `Master_Results` output from there.

## Usage

1. Collect all required CSV files.
2. Copy all CSV files into a single working directory.
3. Copy `vig_master_driver.py` into that same directory.
4. Run:


python vig_master_driver.py


## Result

The script will compile the inputs and generate Master_Results in the working directory.

## Note

Do not separate the CSV files from the Master Driver before execution.
They must be kept together in the same directory for the code to run correctly.
