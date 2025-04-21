import json

with open('cluster_recommendations.json', 'r') as file:
    cluster_recs = json.load(file)

with open('gmm_predictions.json', 'r') as file:
    cluster_data = json.load(file)

with open('test_reviews_sorted.json', 'r') as file:
    test_data = json.load(file)

correct_count = 0
incorrect_count = 0

count = 0
for reviewer in cluster_data:
    count += 1
    # print(reviewer["id"] in test_data)
    # print(test_data[reviewer["id"]])
    # print(test_data[reviewer["id"]]['asin'])
    if reviewer["id"] in test_data:
        book_suggestions = [book['product_id'] for book in cluster_recs[str(reviewer['label'])]]
        # print(book_suggestions)
        if test_data[reviewer["id"]]['asin'] in book_suggestions:
            correct_count += 1
        else:
            incorrect_count += 1
print(count)
print(correct_count, incorrect_count)
print(str((correct_count/(correct_count + incorrect_count))*100) + "%")