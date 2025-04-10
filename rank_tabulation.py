import pandas as pd
import numpy as np

# Load the assignment data
assignment_data = pd.read_csv("data/inputs/2025/student_assignment_summary.csv")

# Print basic information about the dataset
print(f"Total number of students: {len(assignment_data)}")
print(f"Number of assigned students: {assignment_data['any_assigned'].sum()}")
print(f"Number of unassigned students: {len(assignment_data) - assignment_data['any_assigned'].sum()}")
print(f"Average unmatched probability: {assignment_data['unmatched'].mean():.4f}")

# Check for missing values in final_rank
print(f"\nMissing values in final_rank: {assignment_data['final_rank'].isna().sum()}")

# Replace NaN values with "Unassigned" instead of 8
assignment_data['final_rank'] = assignment_data['final_rank'].fillna('Unassigned')

# Count the frequency of each rank
rank_counts = assignment_data['final_rank'].value_counts().sort_index()
total_students = len(assignment_data)

# Calculate percentages and cumulative percentages
rank_percent = (rank_counts / total_students * 100).round(2)
rank_cumsum = rank_percent.cumsum().round(2)

# Create a summary dataframe
rank_summary = pd.DataFrame({
    'Freq.': rank_counts,
    'Percent': rank_percent,
    'Cum.': rank_cumsum
})

# Print the rank distribution summary
print("\nRank Distribution Summary:")
print(rank_summary)

# Create a more readable assignment summary
assignment_df = pd.DataFrame({
    'Assignment': [f'Rank {rank}' if rank != 'Unassigned' else 'Unassigned' for rank in rank_summary.index],
    'Fraction': rank_percent.values
})

print("\nAssignment Summary:")
print(assignment_df) 