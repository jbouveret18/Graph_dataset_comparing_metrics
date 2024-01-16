import json
import matplotlib.pyplot as plt


# Load the JSON file
with open('results.json') as json_file:
    data = json.load(json_file)

# Access the data from the loaded JSON
cosine_similarity = data["CosineSimilarity"]
jaccard_similarity = data["JaccardSimilarity"]
overlapping_neighborhood = data["OverlappingNeighborhood"]
graph_edit_distance = data["GraphEditDifference"]
common_center = data["CommonCenters"]
aggregate_resemblance_score = data["AggregateResemblanceScore"]

cosine_similarity_time = data["CosineSimilarityTime"]
jaccard_similarity_time = data["JaccardSimilarityTime"]
overlapping_neighborhood_time = data["OverlappingNeighborhoodTime"]
graph_edit_distance_time = data["GraphEditDifferenceTime"]
common_center_time = data["CommonCentersTime"]
# Add more variables as needed

# Plot the data
metrics = ["Cosine Similarity",
           "Jaccard Similarity",
           "Overlapping Neighborhood",
           "Graph Edit Distance",
           "Common Centers",
           "Aggregate Resemblance score"]
values = [cosine_similarity,
          jaccard_similarity,
          overlapping_neighborhood,
          graph_edit_distance,
          common_center,
          aggregate_resemblance_score]

time = [cosine_similarity_time,
        jaccard_similarity_time,
        overlapping_neighborhood_time,
        graph_edit_distance_time,
        common_center_time]

running_times = ["Cosine Similarity",
           "Jaccard Similarity",
           "Overlapping Neighborhood",
           "Graph Edit Distance",
           "Common Centers"]


# Plot the data for percentages
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
plt.bar(metrics, values)
plt.ylabel('Percentage')
plt.title('Graph Comparison Metrics')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.legend(['Values'])  # Add legend for values

# Plot the data for running time
plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
plt.plot(running_times, time, marker='o', linestyle='-', color='b')
plt.ylabel('Time (microseconds)')
plt.title('Running Time for Graph Comparison Metrics')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.legend(['Time'])  # Add legend for time


plt.tight_layout()  # Adjust layout for better spacing
plt.show()
