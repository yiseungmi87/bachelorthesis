import pandas as pd
df = pd.read_csv('employee_promotion.csv')
from pandasql import sqldf
query = """
SELECT *
FROM df
WHERE department = 'Operations'
"""
operations_df = sqldf(query)
print(operations_df)
