import pandas as pd

# Load the CSV file
file_path = "processedResults.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=['ID'])

# Define the columns to group by
group_cols = [
    'BoxInclusion',
    'MaskInclusion',
    'NumberOfPOIs',
    'POIPlacementAlgorithm',
    'PerimeterBuffer',
    'BoundingBoxDistortion',
    'MaskDistortion'
]

# Group the DataFrame by the specified columns and calculate the mean of the metric
grouped_df = df.groupby(group_cols, dropna=False)['avg_sam_mask_AP_50_95'].mean().reset_index()

# Sort the grouped DataFrame by the average metric in descending order
sorted_df = grouped_df.sort_values('avg_sam_mask_AP_50_95', ascending=False)

# Print the top 5 rows
print(sorted_df.head(5))
