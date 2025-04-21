import json

import pandas as pd

# Read JSON data from a file
reviewer_clusters = pd.read_json('gmm_predictions.json')
print(reviewer_clusters.keys())
with open('reviews_sorted', 'r') as file:
    user_reviews = json.load(file)
cluster_data = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}}
for cluster_num in range(0, 10):
    cluster_set = {}
    cluster_df = pd.DataFrame(columns=['product_id', 'num_reviews', 'average_score'])
    included_reviewers = reviewer_clusters[reviewer_clusters['label'] == cluster_num]
    print(included_reviewers.head())
    for reviewer in included_reviewers['id'].tolist():
        for review in user_reviews[reviewer]:
            if review[0] not in cluster_set:
                cluster_set[review[0]] = {'num_Of_Reviews': 0, 'total_Reviews_Score': 0}
            cluster_set[review[0]]['num_Of_Reviews'] += 1
            cluster_set[review[0]]['total_Reviews_Score'] += review[1]
    print(len(cluster_set.keys()))
    count_dat = 0
    for product in cluster_set.keys():
        count_dat += cluster_set[product]['num_Of_Reviews']
        curr_prod = cluster_set[product]
        if curr_prod['num_Of_Reviews'] > 50:
            cluster_df.loc[len(cluster_df)] = [product, cluster_set[product]['num_Of_Reviews'], cluster_set[product]['total_Reviews_Score']/cluster_set[product]['num_Of_Reviews']]
    sorted_cluster_df = cluster_df.sort_values(by='average_score', ascending=False).reset_index(drop=True)
    print(sorted_cluster_df.head(20))
    cluster_data[cluster_num] = sorted_cluster_df.head(20).to_dict(orient='records')

with open("cluster_recommendations.json", "w") as file:
  json.dump(cluster_data, file, indent=4)
