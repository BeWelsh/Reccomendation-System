import json

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

#Prepare
n_conponent = np.arange(1,10)

with open('review_data', 'r') as file:
    review_data = json.load(file)
rd_df = pd.DataFrame.from_dict(review_data)
formatted_data = rd_df.T
print(formatted_data.head())

#Create GGM Model
model = GaussianMixture(n_components = 6,random_state=42).fit(formatted_data)

cluster = pd.Series(model.predict(formatted_data), name = 'cluster')
print(cluster.head())
review_ids = pd.Series(formatted_data.index, name = 'review_id')
print(review_ids.head())

reviewer_clusters = pd.concat([review_ids, cluster], axis=1)
print(reviewer_clusters.head())

json_string = reviewer_clusters.to_json(orient='records')

# Optionally, write JSON string to a file
with open('reviewers_clusters.json', 'w') as f:
    json.dump(json.loads(json_string), f, indent=4)