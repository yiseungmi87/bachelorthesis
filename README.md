README

Description

This repository contains query relaxation prototype.

    load_data.py: Script to load datasets, apply filters, augment them, and save a list of selected dataset IDs.
    Relaxation.py: Script that reads dataset IDs from the generated text file and loads the corresponding datasets for query relaxation.

Prerequisites

    Python 3
    Libraries: numpy, sklearn, kneed

Setup

    Ensure Python 3 is installed.
    Install required packages:

Execution

    Run load_data.py to filter and augment datasets and generate the list of selected dataset IDs.

    After obtaining the selected_dataset_ids.txt file from the first script, run Relaxation.py.

Output

    load_data.py: Modified datasets saved in the collection directory and a list of selected dataset IDs saved as selected_dataset_ids.txt.
    Relaxation.py: Query relaxation prototype based on the datasets specified in selected_dataset_ids.txt.

Recommendations

    Ensure both scripts (load_data.py and Relaxation.py) are in the same directory.
