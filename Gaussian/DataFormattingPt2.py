import gzip
import json


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)


products = parse('meta_Kindle_Store.json.gz')
with open('reviews_sorted.json', 'r') as file:
    user_reviews = json.load(file)

# Makes a list of books reviewed in the user reviews
reviewed_books = []
count = 0
for reviewer in user_reviews:
    for review in user_reviews[reviewer]:
        if review['asin'] not in reviewed_books:
            reviewed_books.append(review['asin'])
        count += 1
        print(count)

# Accumulates book data for each reviewed book
reviewed_products = {}
count2 = 0
for product in products:
    count2 += 1
    print(count2)
    if product['asin'] in reviewed_books and product['asin'] not in reviewed_products:
        reviewed_products[product['asin']] = {'Categories': product['categories']}
        if 'price' in product:
            reviewed_products[product['asin']]['Price'] = product['price']

# Creates the user statistics table based off the book data for all the books each user reviewed and then stores the data in a json
results = {}
count3 = 0
for reviewer in user_reviews:
    price = 0
    categories = []
    reviewWeights = 0
    for review in user_reviews[reviewer]:
        product = reviewed_products[review['asin']]
        reviewWeights += review['overall']
        if 'Price' in product:
            price += product['Price']
        for cats in product['Categories']:
            for cat in cats:
                if cat not in categories:
                    categories.append(cat)
    count3 += 1
    print(count3)
    results[reviewer] = {"Average Price": price/len(user_reviews[reviewer]), "Average Review": reviewWeights/len(user_reviews[reviewer]), "Categories Per Book": len(categories)/len(user_reviews[reviewer])}

with open("review_data.json", "w") as file:
  json.dump(results, file, indent=4)