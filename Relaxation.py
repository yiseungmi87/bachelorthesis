from abc import ABC, abstractmethod
import statistics
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import random

# Read the dataset ids from the file
with open('selected_dataset_ids.txt', 'r') as file:
    dataset_ids = file.read().splitlines()

# Load the datasets from the json files
selected_datasets = []
for dataset_id in dataset_ids:
    with open(f"{dataset_id}.json", 'r') as file:
        dataset_info = json.load(file)
    selected_datasets.append(dataset_info)

for dataset_info in selected_datasets:
    print(f"Dataset with ID: {dataset_info['dataset_id']}")

class Predicate:
    def __init__(self, key, operator, value, nested_predicate=None):
        self.key = key
        self.operator = operator
        self.value = value
        self.nested_predicate = nested_predicate  # This holds another Predicate object for Rule 16, 17

    def __repr__(self):
        if self.nested_predicate:
            nested = self.nested_predicate
            return f"{self.key} {self.operator} {self.value} (Nested_Predicate: {nested.key} {nested.operator} {nested.value})"
        else:
            return f"{self.key} {self.operator} {self.value}"
    
class Query:
    def __init__(self):
        self.predicates = []
        self.conjunctions = []

    def __eq__(self, other):
        if isinstance(other, Query):
            return self.predicates == other.predicates and self.conjunctions == other.conjunctions
        return False

    def add_predicate(self, predicate, conjunction=None):
        self.predicates.append(predicate)
        if conjunction is not None:
            self.conjunctions.append(conjunction)

    def __repr__(self):
        if not self.predicates:
            return ""
        result = str(self.predicates[0])
        for i in range(1, len(self.predicates)):
            conjunction = self.conjunctions[i - 1] if i <= len(self.conjunctions) else None
            predicate = self.predicates[i]
            if conjunction:
                result += f" {conjunction}"
            result += f" {predicate}"
        return result

def query_relaxation(query, strategy, predicate_indices=None):
    """
    Arguments:
    query: The original strict query
    strategy: An instance of a subclass of RelaxationStrategy, to use for query relaxation
    predicate_indices: The indices of the predicates to relax, or None to relax all predicates
    
    Returns:
    relaxed_query: The relaxed version of the original query
    """
    relaxed_query = strategy.relax_query(query, predicate_indices)
    return relaxed_query

class RelaxationRule(ABC):    
    @abstractmethod
    def relax_query(self, query):
        self._validate_predicates(query.predicates)


    def _should_relax(self, index, predicate_indices):
        if predicate_indices is None:
            return True
        if isinstance(predicate_indices, int):
            return index == predicate_indices
        if isinstance(predicate_indices, list):
            return index in predicate_indices
        return False
    
    # Helper method to get the values of a predicate from a list of datasets
    def _extract_values(self, predicate, datasets):
        if predicate.key.startswith('features.'):
            # Split the key into components
            key_parts = predicate.key.split('.')
            # The second part is the feature name, the last part is the attribute
            feature_name = key_parts[1]
            attribute = key_parts[-1]
            # Get the attribute value for all features with this name in all datasets
            values = [feature[attribute] for dataset in datasets for feature in dataset['features'] if feature['name'] == feature_name]
        else:
            # The predicate refers to a quality of the dataset, get the quality values from all datasets
            values = [dataset['qualities'].get(predicate.key) for dataset in datasets]

        # Zero None values
        values = [0 if value is None else value for value in values]
        return values
    
    def _validate_predicates(self, predicates):
        for predicate in predicates:
            # Validate key
            if not hasattr(predicate, 'key') or not hasattr(predicate, 'operator') or not hasattr(predicate, 'value'):
                raise ValueError("Each predicate must contain 'key', 'operator', and 'value' properties")
                
            # Validate operator
            valid_operators = ['=', '<', '>', '<=', '>=', 'range']
            if predicate.operator not in valid_operators:
                raise ValueError(f"Invalid operator '{predicate.operator}'. Operator must be one of {valid_operators}")

class ScalarToRange(RelaxationRule): # only scalar
    def __init__(self, datasets, k):
        self.datasets = datasets
        self.k = k

    def relax_query(self, query, predicate_indices=None):
        
        relaxed_query = Query()        
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)    
            
                if predicate.operator == '=':
                # If the predicate is an equality, change it to a range
                # Calculate the mean and standard deviation
                    values = self._extract_values(predicate, self.datasets)
                    stddev_value = statistics.stdev(values)
                # Define a range around the mean
                    lower_bound = round(predicate.value - self.k * stddev_value)
                    upper_bound = round(predicate.value + self.k * stddev_value)

                    # Ensure non-negative bounds for the specified keys
                    negative_keys = [
                        'MaxKurtosisOfNumericAtts',
                        'MaxSkewnessOfNumericAtts',
                        'MeanKurtosisOfNumericAtts',
                        'MeanSkewnessOfNumericAtts',
                        'MinKurtosisOfNumericAtts',
                        'MinSkewnessOfNumericAtts',
                        'NaiveBayesKappa'
                    ]
                    if predicate.key not in negative_keys:
                        lower_bound = max(0, lower_bound)

                # Define a new predicate with the range
                    predicate = Predicate(predicate.key, 'range', (lower_bound, upper_bound))
                    relaxed_query.add_predicate(predicate, conjunction)
                else:
                # If the operator is not '=', copy the predicate unchanged
                    relaxed_query.add_predicate(predicate, conjunction)       
            else:
                relaxed_query.add_predicate(predicate, conjunction)     
        return relaxed_query
    
class ScalarToQuartile(RelaxationRule): #Only scalar value or a range with an open end (no interval)
    def __init__(self, datasets):
        self.datasets = datasets

    def relax_query(self, query, predicate_indices=None):
        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)

            # For an interval predicate, keep it as it is
                if predicate.operator == 'range' and isinstance(predicate.value, tuple):
                    relaxed_query.add_predicate(predicate, conjunction)
                    continue # immediately go to the beginning of the loop, skipping any code below
            
                values = self._extract_values(predicate, self.datasets)
                values = np.array(values)
                values[np.where(values==None)] = 0
                q1, q2, q3 = np.percentile(values, [25, 50, 75])
                if predicate.value <= q1:
                    quartile = 'lowest'
                elif predicate.value <= q2:
                    quartile = 'second'
                elif predicate.value <= q3:
                    quartile = 'third'
                else:
                    quartile = 'top'
                # Define a new predicate with the quartile
                relaxed_predicate = Predicate(predicate.key, 'quartile', quartile)
                # Add the new predicate to the relaxed query
                relaxed_query.add_predicate(relaxed_predicate, conjunction)
            else:
                relaxed_query.add_predicate(predicate, conjunction)            
        return relaxed_query
    
class AttributesToCluster(RelaxationRule):
    def __init__(self, datasets):
        self.datasets = datasets

    def relax_query(self, query, predicate_indices=None):

        relaxed_query = Query()
        predicates_to_cluster = []
        if predicate_indices is None:
            predicates_to_cluster = query.predicates
        else:
            predicates_to_cluster = [query.predicates[i] for i in predicate_indices]

        # Collect the attribute values for each predicate
        attribute_values_by_predicate = {}
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices) and predicate.operator == '=':
                self._validate_predicates(query.predicates)
                attribute_values_by_predicate[predicate] = [
                    self._extract_values(predicate, [dataset])[0]
                    for dataset in self.datasets
                ]
            else:
                relaxed_query.add_predicate(predicate, conjunction)

        # Return the original query if there are no predicates to relax
        if not attribute_values_by_predicate:
            return query

        # Combine the attribute values into a single 2D array
        combined_values = np.column_stack([
            attribute_values
            for attribute_values in attribute_values_by_predicate.values()
        ])

        # Normalize attribute data
        scaler = MinMaxScaler()
        combined_values = scaler.fit_transform(combined_values)

        # Determine optimal number of clusters
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(combined_values)
            wcss.append(kmeans.inertia_)

        # Use KneeLocator to find the elbow point
        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        n_clusters = kn.elbow

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_values)
        clusters = kmeans.labels_

        # Find cluster that original query belongs to
        query_values = [predicate.value for predicate in predicates_to_cluster if predicate.operator == '=']
        query_cluster = kmeans.predict(scaler.transform([query_values]))

        # Add a single cluster predicate to the relaxed query
        relaxed_predicate = Predicate('cluster', '=', query_cluster[0])
        relaxed_query.add_predicate(relaxed_predicate, 'and')

        return relaxed_query

class MeanToMinOrMax(RelaxationRule):
    def __init__(self, datasets):
        self.datasets = datasets

    def relax_query(self, query, predicate_indices=None):
        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)
                # Check if the predicate key starts with 'Mean'
                if predicate.key.startswith('Mean'):
                    # Compute the median differences
                    median_difference_max = self._compute_median_difference(predicate, 'Max')
                    median_difference_min = self._compute_median_difference(predicate, 'Min')

                    # Handling interval predicates and scalar predicates with '=' operator
                    if isinstance(predicate.value, tuple) or predicate.operator == '=':
                        if not isinstance(predicate.value, tuple):
                            value_min, value_max = predicate.value, predicate.value
                        else:
                            value_min, value_max = predicate.value

                        new_key_min = predicate.key.replace('Mean', 'Min')
                        new_value_min = value_min + median_difference_min
                        new_key_max = predicate.key.replace('Mean', 'Max')
                        new_value_max = value_max + median_difference_max

                        # Create the new relaxed predicates
                        relaxed_predicate_min = Predicate(new_key_min, '>', new_value_min)
                        relaxed_query.add_predicate(relaxed_predicate_min, conjunction)
                        relaxed_predicate_max = Predicate(new_key_max, '<', new_value_max)
                        relaxed_query.add_predicate(relaxed_predicate_max, conjunction)
                        
                    else:
                        # Determine the new key and value based on the operator for scalar predicates
                        if predicate.operator in ['<', '<=']:
                            new_key = predicate.key.replace('Mean', 'Max')
                            new_value = predicate.value + median_difference_max
                            operator = predicate.operator
                        elif predicate.operator in ['>', '>=']:
                            new_key = predicate.key.replace('Mean', 'Min')
                            new_value = predicate.value + median_difference_min
                            operator = predicate.operator
                        else:
                            # If the operator is not <, <=, >, >=, =, copy the predicate
                            relaxed_query.add_predicate(predicate, conjunction)
                            continue
                        # Create the new relaxed predicate
                        relaxed_predicate = Predicate(new_key, operator, new_value)
                        relaxed_query.add_predicate(relaxed_predicate, conjunction)
                else:
                    # If the key does not start with 'Mean', copy the predicate and continue
                    relaxed_query.add_predicate(predicate, conjunction)
            else:
                # If an index is specified and this isn't it, copy the predicate and continue
                relaxed_query.add_predicate(predicate, conjunction)

        return relaxed_query

    def _compute_median_difference(self, predicate, suffix):
        """
        Computes the median difference between the max/min and mean value of the attribute defined by the predicate across all datasets.
        """
        attribute = predicate.key.replace('Mean', '',1)
        differences = []
        for dataset in self.datasets:
            mean_value = dataset['qualities']['Mean' + attribute]
            target_value = dataset['qualities'][suffix + attribute]
            if np.isnan(mean_value) or np.isnan(target_value):
                continue
            differences.append(target_value - mean_value)
        if len(differences) == 0:
            raise ValueError(f"No non-NaN values found")
        return statistics.median(differences)

class AbsoluteCountToProportional(RelaxationRule):
    def __init__(self, datasets, range_percentage=0.1):
        self.datasets = datasets
        self.RANGE_PERCENTAGE = range_percentage

    def _calculate_ratio(self, value, total):
        if total is None:
            raise ValueError("The total should not be None")
        elif total == 0:
            raise ValueError("The total should not be zero")
        else:
            if isinstance(value, tuple):
                return tuple(v / total for v in value)
            else:
                return value / total
    
    def _create_range(self, value):
        """Create a range around the value."""
        if isinstance(value, tuple):
            delta_low = value[0] * self.RANGE_PERCENTAGE
            delta_high = value[1] * self.RANGE_PERCENTAGE
            return (value[0] - delta_low, value[1] + delta_high)
        else:
            delta = value * self.RANGE_PERCENTAGE
            return (value - delta, value + delta)
        
    def _compute_average(self, attribute):
        values = []
        for dataset in self.datasets:
            value = None
            if attribute in dataset['qualities']:
                value = dataset['qualities'][attribute]
            elif 'features' in dataset and attribute in dataset['features']:
                value = dataset['features'][attribute]
            elif attribute == 'NumberOfValues':
                value = dataset['qualities']['NumberOfFeatures'] * dataset['qualities']['NumberOfInstances']
            if value is not None and not np.isnan(value):
                values.append(value)
        if len(values) == 0:
            raise ValueError(f"No non-NaN values found for attribute {attribute}")
        return sum(values) / len(values)
    
    def relax_query(self, query, predicate_indices):
        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)

                total = None
                if predicate.key in ['NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues']:
                    total = self._compute_average('NumberOfFeatures' if 'Features' in predicate.key else 'NumberOfInstances' if 'Instances' in predicate.key else 'NumberOfValues')
                    new_key = predicate.key + "_ratio"
                elif predicate.key.startswith('features.') and predicate.key != 'features.country.percentage_missing_values':
                    feature_name = predicate.key.split('.')[1]
                    feature_feature = predicate.key.split('.')[2]
                    if feature_feature in ['number_unique_values', 'number_missing_values']:
                        total = self._compute_average('NumberOfInstances')
                        new_key = 'features.' + feature_name + '.' + feature_feature + '_to_ratio'
                else:
                    relaxed_query.add_predicate(predicate, conjunction)
                    continue
                
                percentage = self._calculate_ratio(predicate.value, total) * 100
                
                if predicate.operator in ['=', 'range']:
                    percentage_range = self._create_range(percentage)  # Create the range
                    relaxed_predicate = Predicate(new_key, 'range', percentage_range)  # Use 'range' as the operator
                else:
                    relaxed_predicate = Predicate(new_key, predicate.operator, percentage)
                #percentage_range = self._create_range(percentage)  # Create the range
                #relaxed_predicate = Predicate(new_key, 'range', percentage_range)  # Use 'range' as the operator
                relaxed_query.add_predicate(relaxed_predicate, conjunction)
            else:
                relaxed_query.add_predicate(predicate, conjunction)

        return relaxed_query
        
class TotalToAttribute(RelaxationRule):
    def __init__(self, proportion):
        self.proportion = proportion

    def relax_query(self, query, predicate_indices=None):
        if not query.predicates:
            raise ValueError("Input query must have at least one predicate")

        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)
                
                if predicate.key == 'PercentageOfMissingValues':
                    nested_predicate = Predicate(predicate.key, predicate.operator, predicate.value)
                    relaxed_predicate = Predicate('attr_proportion', '>=', self.proportion, nested_predicate)
                    relaxed_query.add_predicate(relaxed_predicate, conjunction)
                elif predicate.key == 'MeanNominalAttDistinctValues':
                    nested_predicate = Predicate('NumberOfUniqueValues', predicate.operator, predicate.value)
                    relaxed_predicate = Predicate('attr_proportion', '>=', self.proportion, nested_predicate)
                    relaxed_query.add_predicate(relaxed_predicate, conjunction)
                else:
                    # If the predicate's key does not match any conditions, copy the predicate and continue
                    relaxed_query.add_predicate(predicate, conjunction)
            else:
                # If an index is specified and this isn't it, copy the predicate and continue
                relaxed_query.add_predicate(predicate, conjunction)

        return relaxed_query


def test_relaxation_strategies():
    # Given a loaded dataset, selected_dataset

    # Defining various predicates
    predicates = [
        Predicate("NumberOfInstances", "=", 100000),
        Predicate('NumberOfNumericFeatures', '>', 10),
        Predicate("PercentageOfMissingValues", "<", 5),
    ]

    # Creating queries
    query = Query() 
    query.add_predicate(predicates[0], "AND")
    query.add_predicate(predicates[1], "AND")
    query.add_predicate(predicates[2])

    # Defining relaxation strategies
    relaxation_strategies = [
        ScalarToRange(selected_datasets, k=1),
        ScalarToRange(selected_datasets, k=2),
        AbsoluteCountToProportional(selected_datasets, 0.01),
        AbsoluteCountToProportional(selected_datasets, 0.05),
        TotalToAttribute(0.75),
        TotalToAttribute(0.5),
    ]

    relaxed_1 = query_relaxation(query, relaxation_strategies[0], 0)
    print(relaxed_1.predicates[0])
    relaxed_2 = query_relaxation(query, relaxation_strategies[1], 0)
    print(relaxed_2.predicates[0])
    relaxed_3 = query_relaxation(query, relaxation_strategies[2], 1)
    print(relaxed_3.predicates[1])
    relaxed_4 = query_relaxation(query, relaxation_strategies[3], 1)
    print(relaxed_4.predicates[1])
    relaxed_5 = query_relaxation(query, relaxation_strategies[4], 2)
    print(relaxed_5.predicates[2])
    relaxed_6 = query_relaxation(query, relaxation_strategies[5], 2)
    print(relaxed_6.predicates[2])

    # Defining the indices to be used with each strategy
    indices = [0, 0, 1, 1, 2, 2]

# Loop over both strategies and indices
    for i, (strategy, index) in enumerate(zip(relaxation_strategies, indices)):
        relaxed_query = query_relaxation(query, strategy, index)
        print(relaxed_query.predicates[index])


if __name__ == "__main__":
    test_relaxation_strategies()


'''
import random
import time

# Define the keys and operators.
predicate_keys = ['features.country.number_missing_values', 
'features.country.percentage_missing_values', 
'features.country.number_unique_values',
'MaxAttributeEntropy', 'MaxKurtosisOfNumericAtts', 
'MaxMeansOfNumericAtts', 'MaxMutualInformation', 
'MaxNominalAttDistinctValues', 'MaxSkewnessOfNumericAtts', 
'MaxStdDevOfNumericAtts', 'MeanAttributeEntropy', 
'MeanKurtosisOfNumericAtts', 'MeanMeansOfNumericAtts', 
'MeanMutualInformation', 'MeanNominalAttDistinctValues', 'MeanSkewnessOfNumericAtts', 
'MeanStdDevOfNumericAtts', 'MinAttributeEntropy', 
'MinKurtosisOfNumericAtts', 'MinMeansOfNumericAtts', 
'MinMutualInformation', 'MinNominalAttDistinctValues', 
'MinSkewnessOfNumericAtts', 'MinStdDevOfNumericAtts', 
'MinorityClassPercentage', 'MinorityClassSize', 
'NaiveBayesAUC', 'NaiveBayesErrRate', 'NaiveBayesKappa', 
'NumberOfBinaryFeatures', 'NumberOfClasses', 
'NumberOfFeatures', 'NumberOfInstances', 
'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues', 
'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 
'PercentageOfBinaryFeatures', 'PercentageOfInstancesWithMissingValues', 
'PercentageOfMissingValues', 'PercentageOfNumericFeatures', 
'PercentageOfSymbolicFeatures']

operators = ['>', '<', '=', '>=', '<=', 'range']
conjunctions = ['AND']

# Define initial strategy usage
strategy_usage = {
    'ScalarToRange': 0,
    'ScalarToQuartile': 0,
    'AttributesToCluster': 0,
    'AbsoluteCountToProportional': 0,
    'TotalToAttribute': 0,
    'MeanToMinOrMax': 0 
}

# Define relaxation strategies
relaxation_strategies = [
    ScalarToRange(selected_datasets, k=1),
    ScalarToQuartile(selected_datasets),
    AttributesToCluster(selected_datasets),
    AbsoluteCountToProportional(selected_datasets),
    TotalToAttribute(0.75),
    MeanToMinOrMax(selected_datasets)
]

# Number of times to run the whole procedure
iterations = 10

for iteration in range(iterations):
    print(f"\nIteration {iteration+1} of {iterations}")

    # Reset usage counts for this iteration
    iteration_usage = strategy_usage.copy()

    random_queries = []
    for i in range(20):
        query = Query()
        for j in range(random.randint(1, 5)):  # Randomly choose between 1 to 5 predicates.
            key = random.choice(predicate_keys)
            operator = random.choice(operators)
            value = (random.randint(0, 50), random.randint(51, 100)) if operator == 'range' else random.randint(0, 100)
            conjunction = random.choice(conjunctions) if j != 0 else None
            predicate = Predicate(key, operator, value)
            query.add_predicate(predicate, conjunction)
        random_queries.append(query)

    # Print the generated queries.
    for i, query in enumerate(random_queries):
        print(f"Query {i+1}: {query}")


class BaselineRelaxation:
    def relax_query(self, query, predicate_indices=None):
        #Removes the last predicate from a given query and returns a new query.
        new_query = Query() 
        new_query.predicates = query.predicates[:-1]  # Copy all predicates except the last
        new_query.conjunctions = query.conjunctions[:-1] if query.conjunctions else []  # Remove the last conjunction if there is any
        return new_query

    # Testing each strategy
    for strategy in relaxation_strategies:
        print(f"\nTesting strategy: -------------- {strategy.__class__.__name__}------------------")
        for query in random_queries:
            print(f"\nOriginal query: {query}")               
            relaxed_query = query_relaxation(query, strategy)
            if str(relaxed_query) != str(query):
                print(f"Relaxed query: {relaxed_query}")
                strategy_usage[strategy.__class__.__name__] += 1
            else:
                print("The query was not relaxed.")

# Calculate and print average usage for each strategy
for strategy, total_usage in strategy_usage.items():
    avg_usage = total_usage / iterations
    print(f"{strategy}: {avg_usage} times on average")

# Identify the strategy with the highest average coverage
max_strategy = max(strategy_usage, key=strategy_usage.get)
print(f"The strategy with the highest average coverage is {max_strategy}.")

# Identify strategies that were never used on average
unused_strategies = [strategy for strategy, avg_usage in strategy_usage.items() if avg_usage == 0]
print(f"Strategies that were never used on average: {', '.join(unused_strategies)}")

import time
import statistics

def test_relaxation_strategies():
    # Given a loaded dataset, selected_dataset

    # Defining various predicates
    predicates = [
        
        Predicate("MeanNominalAttDistinctValues", "=", 10),
        Predicate('features.country.number_missing_values', '=', 3),

        #Predicate("PercentageOfMissingValues", "=", 3),
        
        Predicate("MeanNominalAttDistinctValues", ">", 10),
        Predicate('features.country.number_missing_values', "range", (10, 20))      
    ]

    # Creating queries
    query1 = Query() 
    query1.add_predicate(predicates[0])

    query2 = Query() 
    query2.add_predicate(predicates[1])

    query3 = Query() 
    query3.add_predicate(predicates[0], "AND")
    query3.add_predicate(predicates[1])

    query4 = Query()
    query4.add_predicate(predicates[2], "AND")
    query4.add_predicate(predicates[3], "AND")

    # Queries to be used
    queries = [query1, query2, query3, query4]

    # Defining relaxation strategies
    relaxation_strategies = [
        BaselineRelaxation(),
        ScalarToRange(selected_datasets, k=1),
        ScalarToQuartile(selected_datasets),
        AttributesToCluster(selected_datasets),
        AbsoluteCountToProportional(selected_datasets),
        TotalToAttribute(0.75),
        MeanToMinOrMax(selected_datasets)
    ]

    # Number of iterations for each experiment
    num_iterations = 10   

    # Testing each strategy
    for strategy in relaxation_strategies:
        print(f"\nTesting strategy: {strategy.__class__.__name__}")
        for query in queries:
            print(f"\nOriginal query: {query}")
            
            execution_times = []
            for _ in range(num_iterations):
                # Start time
                start_time = time.time()
                
                relaxed_query = query_relaxation(query, strategy)
                
                # End time
                end_time = time.time()
                execution_time = end_time - start_time
                
                execution_times.append(execution_time)
                
            avg_execution_time = statistics.mean(execution_times)
            
            print(f"Relaxed query: {relaxed_query}")
            print(f"Average execution time over {num_iterations} iterations: {avg_execution_time} seconds")


if __name__ == "__main__":
    test_relaxation_strategies()

# Queries to be used
queries_fixed = random_queries
print(queries_fixed)

def test_relaxation_strategies_30():
    # Given a loaded dataset, selected_dataset   

    # Defining relaxation strategies
    relaxation_strategies = [
        BaselineRelaxation(),
        ScalarToRange(selected_datasets, k=1),
        ScalarToQuartile(selected_datasets),
        AttributesToCluster(selected_datasets),
        AbsoluteCountToProportional(selected_datasets, 0.01),
        TotalToAttribute(0.75),
        MeanToMinOrMax(selected_datasets)
    ]

    # Number of iterations for each experiment
    num_iterations = 10   

    for query in queries_fixed:
            print(f"\nOriginal query: {query}")
    # Testing each strategy
    for strategy in relaxation_strategies:
        print(f"\nTesting strategy: {strategy.__class__.__name__}")

        for query in queries_fixed:
            #print(f"\nOriginal query: {query}")
            
            execution_times = []
            for _ in range(num_iterations):
                # Start time
                start_time = time.time()
                
                relaxed_query = query_relaxation(query, strategy)
                
                # End time
                end_time = time.time()
                execution_time = end_time - start_time
                
                execution_times.append(execution_time)
                
            avg_execution_time = statistics.mean(execution_times)
            
            #print(f"Relaxed query: {relaxed_query}")
            if str(relaxed_query) != str(query):
                print(f"Average execution time over {num_iterations} iterations: {avg_execution_time} seconds")
                
            else:
                print(f"Average execution time over {num_iterations} iterations: 0 seconds")
            


if __name__ == "__main__":
    test_relaxation_strategies_30()

print(queries_fixed)
'''