def relax_query(query):
    # Extract the mean and standard deviation values from the original query
    # But does it work always? For example, what is user types mean value instead of mean?
    # Any improvement?
    mean = None
    std = None
    for word in query.split():
        if word == 'mean':
            mean = float(query.split()[query.split().index(word)+2])
        elif word == 'standard' and query.split()[query.split().index(word)+1] == 'deviation':
            std = float(query.split()[query.split().index(word)+3])
    
    # Calculate the new minimum and maximum values
    min_val = max(mean - std * 2, 0)
    max_val = mean + std * 2 + 50
    
    # Generate the relaxed query
    relaxed_query = f"Find datasets with a minimum value between {min_val:.2f} and {mean:.2f} and a maximum value between {mean:.2f} and {max_val:.2f}"
    
    return relaxed_query

# Original query
original_query = "Find datasets with a mean between 50 and 100 and a standard deviation between 10 and 20"

# Relax the query
relaxed_query = relax_query(original_query)

# Print the original and relaxed queries
print("Original query:", original_query)
print("Relaxed query:", relaxed_query)

# Test the function with different input queries
queries = [
    "Find datasets with a mean between 10 and 20 and a standard deviation between 2 and 5",
    "Find datasets with a mean between 0 and 50 and a standard deviation between 5 and 10",
    "Find datasets with a mean between 100 and 200 and a standard deviation between 20 and 30",
]

for query in queries:
    print("Original query:", query)
    relaxed_query = relax_query(query)
    print("Relaxed query:", relaxed_query)
    print()