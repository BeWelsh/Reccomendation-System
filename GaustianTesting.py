# Collect the 5 test ppl
# get their cluster
# get their top 10 recs that are not in their read books
# check each recs occurrences per recs list
import gzip
import json

import pandas as pd

reviewer_clusters = pd.read_json('reviewers_clusters.json')
with open('cluster_recommendations', 'r') as file:
    cluster_recs = json.load(file)
with open('reviews_sorted', 'r') as file:
    review_data = json.load(file)
test_sample = reviewer_clusters.head().to_dict(orient='records')
test_recs = {}
for test_user in test_sample:
    user_recs = []
    count = 0
    while len(user_recs) < 10:
        if cluster_recs[str(test_user['cluster'])][count] not in review_data[test_user['review_id']]:
            user_recs.append(cluster_recs[str(test_user['cluster'])][count]['product_id'])
        count += 1
    test_recs[test_user['review_id']] = user_recs
print(test_recs)
with open("test_recommendations", "w") as file:
  json.dump(test_recs, file, indent=4)
# def parse(path):
#   g = gzip.open(path, 'r')
#   for l in g:
#     yield eval(l)
# test_score_base = [{},{},{},{},{},{},{},{},{},{}]
# products = parse('meta_Kindle_Store.json.gz')
# for product in products:
#     for sample in test_sample:
#         if product['asin'] in review_data[sample['review_id']]:
#             if

# test_sample = {"A1F6404F1VG29J": reviewer_clusters[0], "AN0N05A9LIJEQ", "A795DMNCJILA6", "A1FV0SX13TWVXQ", "A3SPTOKDG7WBLN"}
