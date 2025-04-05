import gzip
import json

import pandas as pd


with open('test_recommendations_original', 'r') as file:
    test_recs = json.load(file)

with open('reviews_sorted', 'r') as file:
    review_data = json.load(file)

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)
products = parse('meta_Kindle_Store.json.gz')
products2 = parse('meta_Kindle_Store.json.gz')
test_suggestions = {}
test_suggestions2 = {}
reviewed_books = {}
for rec in test_recs:
    test_suggestions[rec] = []
    test_suggestions2[rec] = []
    for review in review_data[rec]:
        if review[0] not in reviewed_books:
            reviewed_books[review[0]] = [rec]
        else:
            reviewed_books[review[0]].append(rec)

extended_books = {}
for product in products:
    if product['asin'] in reviewed_books:
        for reviewer in reviewed_books[product['asin']]:
            if 'related' in product:
                if 'also_bought' in product['related']:
                    test_suggestions[reviewer].extend(product['related']['also_bought'])
                    if product['asin'] not in extended_books:
                        extended_books[product['asin']] = [reviewer]
                    else:
                        extended_books[product['asin']].append(reviewer)
                if 'also_viewed' in product['related']:
                    test_suggestions[reviewer].extend(product['related']['also_viewed'])
                    if product['asin'] not in extended_books:
                        extended_books[product['asin']] = [reviewer]
                    else:
                        extended_books[product['asin']].append(reviewer)
                if 'buy_after_viewing' in product['related']:
                    test_suggestions[reviewer].extend(product['related']['buy_after_viewing'])
                    if product['asin'] not in extended_books:
                        extended_books[product['asin']] = [reviewer]
                    else:
                        extended_books[product['asin']].append(reviewer)
print("Size" + str(len(extended_books)))
for product in products2:
    if product['asin'] in extended_books:
        print("Check " + product['asin'])
        for reviewer in extended_books[product['asin']]:
            if 'related' in product:
                if 'also_bought' in product['related']:
                    test_suggestions2[reviewer].extend(product['related']['also_bought'])
                if 'also_viewed' in product['related']:
                    test_suggestions2[reviewer].extend(product['related']['also_viewed'])
                if 'buy_after_viewing' in product['related']:
                    test_suggestions2[reviewer].extend(product['related']['buy_after_viewing'])

for rec in test_recs:
    count = 0
    print(len(test_suggestions[rec]))
    print(len(test_suggestions2[rec]))
    for rec2 in test_recs[rec]:
        if rec2 in test_suggestions[rec] or rec2 in reviewed_books or rec2 in test_suggestions2[rec]:
            count += 1
    print("User " + rec + ": " + str(count) + "/10")

