import os
import json
import random
import numpy as np

# Load a single dataset
def load_all_datasets(directory):
    datasets = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            datasets[filename] = data
    return datasets

#all_datasets = load_all_datasets('/Users/seungmi/Downloads/collection')

# Load all datasets by using relative paths
if not os.path.exists('./collection'):
    raise Exception("The 'collection' directory does not exist in the current location. Please make sure you have downloaded the datasets.")
all_datasets = load_all_datasets('./collection')

# Select datasets that have all required qualities
def select_datasets(all_datasets):
    selected_datasets = []
    required_qualities = ['NumberOfInstances', 'NumberOfFeatures', 'NumberOfSymbolicFeatures', 'NumberOfNumericFeatures',
                          'PercentageOfMissingValues', 'NumberOfMissingValues', 'NumberOfInstancesWithMissingValues', 'PercentageOfInstancesWithMissingValues',
                          'PercentageOfNumericFeatures', 'PercentageOfSymbolicFeatures',
                          'MaxAttributeEntropy', 'MaxMutualInformation', 'MaxNominalAttDistinctValues',
                          'MeanAttributeEntropy', 'MeanMutualInformation', 'MeanNominalAttDistinctValues',
                          'MinAttributeEntropy', 'MinMutualInformation', 'MinNominalAttDistinctValues', 
                          ]

    for dataset_name, dataset_info in all_datasets.items():
        # Get the 'qualities' and 'features' attributes
        qualities = dataset_info.get('qualities')
        features = dataset_info.get('features')

        # If qualities is None or features is None, skip this dataset
        if qualities is None or features is None:
            continue

        # Check if all required qualities are in the dataset and they have non-zero values
        if all(quality in qualities and not np.isnan(qualities[quality]) and qualities[quality] != 0 for quality in required_qualities):
            selected_datasets.append(dataset_info)
    
    return selected_datasets

selected_datasets = select_datasets(all_datasets)

# Select 30 random datasets
random_datasets = random.sample(selected_datasets, 30)

# Add additional attributes to the datasets
def add_additional_attributes(datasets):
    countries = ["Poland", "India", "United States", "Indonesia", "Pakistan", "Brazil", "Nigeria", 
                 "Bangladesh", "Russia", "Mexico", "Japan", "Ethiopia", "Philippines", "Egypt", "Vietnam", 
                 "DR Congo", "Germany", "Turkey", "Iran", "Thailand", "United Kingdom", "France", "Italy", 
                 "South Africa", "Tanzania", "Myanmar", "South Korea", "Colombia", "Kenya", "Spain", 
                 "Argentina", "Ukraine", "Sudan", "Algeria", "Uganda", "Poland", "Iraq", "Canada", "Morocco", 
                 "Saudi Arabia", "Uzbekistan", "Peru", "Malaysia", "Angola", "Ghana", "Mozambique", "Yemen", 
                 "Nepal", "Venezuela", "Madagascar"]


    for dataset_info in datasets:
        # Modify the 'features' attribute
        for feature in dataset_info['features']:
            # Calculate 'percentage_missing_values'
            feature['percentage_missing_values'] = feature['number_missing_values'] / dataset_info['qualities']['NumberOfInstances']

            # Count the number of unique values for nominal features
            if feature['data_type'] == 'nominal':
                feature['number_unique_values'] = len(feature['nominal_values'])
        
        # Add new feature 'country'
        num_countries = random.randint(1, 50)  # Randomly select the number of countries for the nominal values
        selected_countries = random.sample(countries, num_countries)  # Randomly select 'num_countries' countries
        number_missing_values = random.randint(0, num_countries)  # Randomly generate the number of missing values
        percentage_missing_values = number_missing_values / dataset_info['qualities']['NumberOfInstances']  # Calculate the percentage of missing values

        # Create a new feature
        new_feature = {
            'index': len(dataset_info['features']),  # The index is the current number of features
            'name': 'country',
            'data_type': 'nominal',
            'nominal_values': selected_countries,
            'number_missing_values': number_missing_values,
            'percentage_missing_values': percentage_missing_values,
            'number_unique_values': len(selected_countries),
        }
        dataset_info['features'].append(new_feature)
        
        # Increase the count of features and nominal features in 'qualities'
        dataset_info['qualities']['NumberOfFeatures'] += 1
        dataset_info['qualities']['NumberOfSymbolicFeatures'] += 1

        # Save the changes back to the JSON file
        with open(f"{dataset_info['dataset_id']}.json", 'w') as file:
            json.dump(dataset_info, file)

# Apply the function to both numeric and nominal datasets
add_additional_attributes(random_datasets)

# Get the information of a random dataset
random_dataset_info = random.choice(random_datasets)

# Save the list of dataset ids to a file
with open('selected_dataset_ids.txt', 'w') as file:
    for dataset in random_datasets:
        file.write(f"{dataset['dataset_id']}\n")