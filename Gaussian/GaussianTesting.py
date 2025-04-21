import json

with open('cluster_recommendations.json', 'r') as file:
    cluster_recs = json.load(file)

with open('gmm_predictions.json', 'r') as file:
    cluster_data = json.load(file)

with open('test_reviews_sorted.json', 'r') as file:
    test_data = json.load(file)

# Compares each user's recommendations with their test book and checks the number of occurrences
correct_count = 0
incorrect_count = 0
count = 0
for reviewer in cluster_data:
    count += 1
    if reviewer["id"] in test_data:
        book_suggestions = [book['product_id'] for book in cluster_recs[str(reviewer['label'])]]
        if test_data[reviewer["id"]]['asin'] in book_suggestions:
            correct_count += 1
        else:
            incorrect_count += 1

# Displays the accuracy
print(str((correct_count/(correct_count + incorrect_count))*100) + "%")