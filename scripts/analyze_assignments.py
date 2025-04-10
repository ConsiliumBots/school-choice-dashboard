import pandas as pd
import matplotlib.pyplot as plt

# Read the student assignment summary
df = pd.read_csv('data/inputs/2025/student_assignment_summary.csv')

# Create a frequency table of final_rank including NaN values
rank_counts = df['final_rank'].value_counts(dropna=False).sort_index()

# Replace NaN with "Unassigned" in the index for better readability
rank_counts.index = rank_counts.index.fillna('Unassigned')

# Print the frequency table
print("\nFinal Rank Distribution (including unassigned students):")
print(rank_counts)

# Calculate percentages
rank_percentages = (rank_counts / len(df) * 100).round(2)
print("\nFinal Rank Percentages (including unassigned students):")
print(rank_percentages)

# Create a bar plot
plt.figure(figsize=(12, 6))
rank_counts.plot(kind='bar')
plt.title('Distribution of Final Ranks (including unassigned students)')
plt.xlabel('Final Rank')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/final_rank_distribution.png')
plt.close() 