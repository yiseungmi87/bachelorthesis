import re
import numpy as np
from scipy.stats import chi2

def relax_query(original_query, degree=0.5, conf_level=0.95):
    
    # Extract mean and standard deviation values from original query
    mean = float(re.findall(r'\d+\.?\d*', original_query)[0])
    std = float(re.findall(r'\d+\.?\d*', original_query)[1])
    # re.findall() return value list
    
    # Set the minimum and maximum values of the mean range
    min_value = mean - degree * std
    max_value = mean + degree * std
    
    # Set the minimum value of the standard deviation range
    std_min, std_max = compute_std_range(std, conf_level)
    
    # Construct relaxed query string
    relaxed_query = f"Find datasets with a mean value between {min_value:.2f} and {max_value:.2f} and a standard deviation between {std_min:.2f} and {std_max:.2f}"
    
    return relaxed_query

def compute_std_range(std, alpha=0.05):
    n = 20 # Random value... i need an academic basis...
    dof = n - 1
    chi_lower = chi2.ppf(alpha/2, dof)
    chi_upper = chi2.ppf(1-alpha/2, dof)
    std_lower = np.sqrt((dof * std ** 2) / chi_upper)
    std_upper = np.sqrt((dof * std ** 2) / chi_lower)
    return std_lower, std_upper


# Original query
original_query = "Find datasets with a mean 50 and a standard deviation 10"

# Relax the query
relaxed_query = relax_query(original_query)

# Print the original and relaxed queries
print("Original query:", original_query)
print("Relaxed query:", relaxed_query)
