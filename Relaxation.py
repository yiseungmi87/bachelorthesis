from abc import ABC, abstractmethod
import statistics
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import random
import time
import statistics
import warnings
from sklearn.exceptions import ConvergenceWarning


# Read the dataset ids from the file
with open('selected_dataset_ids.txt', 'r') as file:
    dataset_ids = file.read().splitlines()

# Load the datasets from the json files
selected_datasets = []
for dataset_id in dataset_ids:
    with open(f"{dataset_id}.json", 'r') as file:
        dataset_info = json.load(file)
    selected_datasets.append(dataset_info)

#for dataset_info in selected_datasets:
    #print(f"Dataset with ID: {dataset_info['dataset_id']}")

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

class RelaxationStrategy(ABC):    
    @abstractmethod
    def relax_query(self, query):
        self._validate_predicates(query.predicates)

    # Helper method to determine whether a predicate should be relaxed
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
            key_parts = predicate.key.split('.')
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
    
    # Helper method to validate the predicates
    def _validate_predicates(self, predicates):
        for predicate in predicates:
            # Validate key
            if not hasattr(predicate, 'key') or not hasattr(predicate, 'operator') or not hasattr(predicate, 'value'):
                raise ValueError("Each predicate must contain 'key', 'operator', and 'value' properties")
            
            # Validate operator
            valid_operators = ['=', '<', '>', '<=', '>=', 'range']
            if predicate.operator not in valid_operators:
                raise ValueError(f"Invalid operator '{predicate.operator}'. Operator must be one of {valid_operators}")

class ScalarToRange(RelaxationStrategy): 
    def __init__(self, datasets, k):
        self.datasets = datasets
        self.k = k

    def relax_query(self, query, predicate_indices=None):
        relaxed_query = Query()        
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)    

                # If the predicate is an equality, change it to a range
                if predicate.operator == '=':                
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
                # If the predicate is not relaxed, copy it unchanged
                relaxed_query.add_predicate(predicate, conjunction) 

        return relaxed_query
    

class ScalarToQuartile(RelaxationStrategy):
    def __init__(self, datasets):
        self.datasets = datasets

    def relax_query(self, query, predicate_indices=None):
        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)

                # For an interval predicate, no change is needed
                if predicate.operator == 'range' and isinstance(predicate.value, tuple):
                    relaxed_query.add_predicate(predicate, conjunction)
                    continue 
            
                # For a non-interval predicate, determine the quartile of the predicate value
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
                # If the predicate is not relaxed, copy it unchanged
                relaxed_query.add_predicate(predicate, conjunction)      

        return relaxed_query
    

class ScalarToCluster(RelaxationStrategy):
    def __init__(self, datasets):
        self.datasets = datasets

    def relax_query(self, query, predicate_indices=None):
        # Ignore convergence warnings
        warnings.simplefilter("ignore", ConvergenceWarning)

        relaxed_query = Query()
        predicates_to_cluster = []
        # If no predicate indices are specified, cluster all predicates
        if predicate_indices is None:
            predicates_to_cluster = query.predicates
        # Otherwise, cluster only the specified predicates
        else:
            # Collect the predicates to be clustered
            predicates_to_cluster = [query.predicates[i] for i in predicate_indices]

        # Collect the attribute values for each predicate
        attribute_values_by_predicate = {}
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            # If the predicate is to be clustered, collect its attribute values
            if self._should_relax(i, predicate_indices) and predicate.operator == '=':
                self._validate_predicates(query.predicates)
                # Collect the attribute values for each dataset
                attribute_values_by_predicate[predicate] = [
                    self._extract_values(predicate, [dataset])[0]
                    for dataset in self.datasets
                ]
            else:
                # If the predicate is not to be clustered, copy it unchanged
                relaxed_query.add_predicate(predicate, conjunction)

        # If no predicates are to be clustered, return the original query
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
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(combined_values)
            wcss.append(kmeans.inertia_)

        # Use KneeLocator to find the elbow point
        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        n_clusters = kn.elbow

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(combined_values)
        clusters = kmeans.labels_

        # Find cluster that original query belongs to
        query_values = [predicate.value for predicate in predicates_to_cluster if predicate.operator == '=']
        query_cluster = kmeans.predict(scaler.transform([query_values]))

        # Add a single cluster predicate to the relaxed query
        relaxed_predicate = Predicate('cluster', '=', query_cluster[0])
        relaxed_query.add_predicate(relaxed_predicate, 'and')

        return relaxed_query


class MeanToMinMax(RelaxationStrategy):
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
                        
                        # Determine the new key and value based on the operator for interval predicates
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
        # Determine the attribute name
        attribute = predicate.key.replace('Mean', '',1)

        differences = []
        for dataset in self.datasets:
            # Get the mean and target values
            mean_value = dataset['qualities']['Mean' + attribute]
            target_value = dataset['qualities'][suffix + attribute]

            # Skip NaN values
            if np.isnan(mean_value) or np.isnan(target_value):
                continue

            # Compute the difference
            differences.append(target_value - mean_value)

        # If no non-NaN values are found, raise an error
        if len(differences) == 0:
            raise ValueError(f"No non-NaN values found")
        
        return statistics.median(differences)

class CountToRatio(RelaxationStrategy):
    def __init__(self, datasets, range=0.1):
        self.datasets = datasets
        self.range = range

    def relax_query(self, query, predicate_indices):
        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)

                total = None
                
                # Check if the predicate key starts with 'NumberOf'
                if predicate.key in ['NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues']:
                    # Compute the total
                    total = self._compute_average('NumberOfFeatures' if 'Features' in predicate.key else 'NumberOfInstances' if 'Instances' in predicate.key else 'NumberOfValues')
                    new_key = predicate.key + "_ratio"

                # Check if the predicate key starts with 'features.'
                elif predicate.key.startswith('features.') and predicate.key != 'features.country.percentage_missing_values':
                    feature_name = predicate.key.split('.')[1]
                    feature_feature = predicate.key.split('.')[2]
                    
                    if feature_feature in ['number_unique_values', 'number_missing_values']:
                        total = self._compute_average('NumberOfInstances')
                        new_key = 'features.' + feature_name + '.' + feature_feature + '_ratio'
                else:
                    # If the key does not start with 'NumberOf' or 'features.', copy the predicate and continue
                    relaxed_query.add_predicate(predicate, conjunction)
                    continue
                
                # Compute the percentage
                percentage = self._calculate_percentage(predicate.value, total)

                # Check if the predicate is a scalar predicate with = operator
                if predicate.operator in ['=']:
                    percentage_range = self._create_range(percentage)  # Create the range
                    relaxed_predicate = Predicate(new_key, 'range', percentage_range)  # Use 'range' as the operator

                else:
                    # Create the new relaxed predicate
                    relaxed_predicate = Predicate(new_key, predicate.operator, percentage)

                relaxed_query.add_predicate(relaxed_predicate, conjunction)

            else:
                # If an index is specified and this isn't it, copy the predicate and continue
                relaxed_query.add_predicate(predicate, conjunction)

        return relaxed_query
    
    def _calculate_percentage(self, value, total):
        # Check if the total is valid
        if total is None:
            raise ValueError("The total should not be None")
        elif total == 0:
            raise ValueError("The total should not be zero")
        else:
            # Check if the value is a tuple
            if isinstance(value, tuple):
                return tuple((v / total) * 100 for v in value)
            else:
                return (value / total) * 100
    
    def _create_range(self, value):
        width = value * self.range
        return (value - width, value + width)
        
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
    
        
class TotalToFeature(RelaxationStrategy):
    def __init__(self, proportion=0.75):
        self.proportion = proportion

    def relax_query(self, query, predicate_indices=None):
        if not query.predicates:
            raise ValueError("Input query must have at least one predicate")

        relaxed_query = Query()
        for i, (predicate, conjunction) in enumerate(zip(query.predicates, query.conjunctions + [None])):
            if self._should_relax(i, predicate_indices):
                self._validate_predicates(query.predicates)
                
                # Check if the predicate key is applicable
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
                # If the predicate should not be relaxed, copy the predicate and continue
                relaxed_query.add_predicate(predicate, conjunction)

        return relaxed_query
    
class BaselineRelaxation:
    def relax_query(self, query, predicate_indices=None):
        #Removes the last predicate from a given query and returns a new query.
        new_query = Query() 
        new_query.predicates = query.predicates[:-1]  # Copy all predicates except the last
        new_query.conjunctions = query.conjunctions[:-1] if query.conjunctions else []  # Remove the last conjunction if there is any
        
        return new_query
    
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

def generate_random_queries(num_queries, seed=None):

    if seed is not None:
        random.seed(seed)

    random_queries = []
    
    for _ in range(num_queries):
        query = Query()
        num_predicates = random.randint(1, 5)  # Choose between 1 to 5 predicates.

        # Randomly select predicates from the list of keys (no repetition).
        selected_keys = random.sample(predicate_keys, num_predicates)
        
        for j, key in enumerate(selected_keys):
            operator = random.choice(operators)  # Randomly choose an operator.
            value = (random.randint(0, 50), random.randint(51, 100)) if operator == 'range' else random.randint(0, 100)  # Randomly choose a value.
            conjunction = 'AND' if j != 0 else None  # Use 'AND' as conjunction if it is not the first predicate.
            
            predicate = Predicate(key, operator, value) 
            query.add_predicate(predicate, conjunction)
        random_queries.append(query)
    
    return random_queries

# Generate 20 random queries. (reusable)
benchmark_queries = generate_random_queries(20, seed=1)
#print(benchmark_queries)

'''
# Experiment1: Demonstration of relaxation strategies
def demonstrate_relaxation_strategies():
    # Defining various predicates
    predicates = [
        Predicate("MeanNominalAttDistinctValues", "=", 10),
        Predicate('features.country.number_missing_values', '=', 3),
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
        MeanToMinMax(selected_datasets),
        ScalarToCluster(selected_datasets),
        CountToRatio(selected_datasets),
        TotalToFeature(0.75),
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

                # Relaxing the query
                relaxed_query = query_relaxation(query, strategy)

                # End time
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)

            avg_execution_time = statistics.mean(execution_times)
            if str(relaxed_query) != str(query):
                print(f"Relaxed query: {relaxed_query}")
            else:
                print("The query was not relaxed.")
            print(f"Average execution time over {num_iterations} iterations: {avg_execution_time} seconds")

if __name__ == "__main__":
    demonstrate_relaxation_strategies()
'''
'''
# Experiment2: Application of Different Strategies for Each Predicate
def apply_each_predicate():

    # Defining various predicates
    predicates = [
        Predicate("NumberOfInstances", "=", 100000),
        Predicate('NumberOfNumericFeatures', '=', 10),
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
        CountToRatio(selected_datasets, 0.01),
        CountToRatio(selected_datasets, 0.05),
        TotalToFeature(0.75),
        TotalToFeature(0.5),
    ]
    # Defining the indices to be used with each strategy
    indices = [0, 0, 1, 1, 2, 2]

# Loop over both strategies and indices
    for i, (strategy, index) in enumerate(zip(relaxation_strategies, indices)):
        relaxed_query = query_relaxation(query, strategy, index)
        print(relaxed_query.predicates[index])


if __name__ == "__main__":
    apply_each_predicate()
'''


# Experiment3: Measure Execution Time for 20 benchmark queries
def measure_execution_time():
    
    for i, query in enumerate(benchmark_queries):
        print(f"Query {i+1}: {benchmark_queries}")

    # Defining relaxation strategies
    relaxation_strategies = [
        BaselineRelaxation(),
        ScalarToRange(selected_datasets, k=1),
        ScalarToQuartile(selected_datasets),
        MeanToMinMax(selected_datasets),
        ScalarToCluster(selected_datasets),
        CountToRatio(selected_datasets),
        TotalToFeature(0.75),
    ]

    # Number of iterations for each experiment
    num_iterations = 10   

    # Testing each strategy
    for strategy in relaxation_strategies:
        print(f"\nTesting strategy: {strategy.__class__.__name__}")
        for query in benchmark_queries:
            #print(f"\nOriginal query: {query}")
            execution_times = []
            for _ in range(num_iterations):
                # Start time
                start_time = time.time()

                # Relaxing the query
                relaxed_query = query_relaxation(query, strategy)

                # End time
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)

            avg_execution_time = statistics.mean(execution_times)
            if str(relaxed_query) != str(query):
                print(f"({benchmark_queries.index(query)+1}, {avg_execution_time}) ")
            else:
                print(f"({benchmark_queries.index(query)+1}, 0) ")

            #print(f"Average execution time over {num_iterations} iterations: {avg_execution_time} seconds")

if __name__ == "__main__":
    measure_execution_time()

'''
# Experiment4: Counting the Usage of Each Strategy
def count_usage():

    # Define initial strategy usage
    strategy_usage = {
    'ScalarToRange': 0,
    'ScalarToQuartile': 0,
    'ScalarToCluster': 0,
    'MeanToMinMax': 0,
    'CountToRatio': 0,
    'TotalToFeature': 0,
}

    # Define relaxation strategies
    relaxation_strategies = [
    ScalarToRange(selected_datasets, k=1),
    ScalarToQuartile(selected_datasets),
    ScalarToCluster(selected_datasets),
    MeanToMinMax(selected_datasets),
    CountToRatio(selected_datasets),
    TotalToFeature(proportion=0.75)
]
    iterations = 10

    for iteration in range(iterations):

        random_queries = generate_random_queries(30)

    # Testing each strategy
        for strategy in relaxation_strategies:
        #print(f"\nTesting strategy: {strategy.__class__.__name__}")
            for query in random_queries:
            #print(f"\nOriginal query: {query}")               
                relaxed_query = query_relaxation(query, strategy)
                if str(relaxed_query) != str(query):
                #print(f"Relaxed query: {relaxed_query}")
                    strategy_usage[strategy.__class__.__name__] += 1

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
            
if __name__ == "__main__":
    count_usage()
'''