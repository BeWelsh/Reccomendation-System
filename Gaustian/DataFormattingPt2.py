import gzip
import json


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

products = parse('meta_Kindle_Store.json.gz')
with open('reviews_sorted', 'r') as file:
    user_reviews = json.load(file)
# reviewed_books = []
# count = 0
# for reviewer in user_reviews:
#     for review in user_reviews[reviewer]:
#         if review[0] not in reviewed_books:
#             reviewed_books.append(review[0])
#         count += 1
#         print(count)
# print(len(reviewed_books))
#
# with open("reviewed_books", 'w') as file:
#     for item in reviewed_books:
#         file.write(str(item) + '\n')

reviewed_books = []
with open('reviewed_books', 'r') as file:
    for line in file:
        # Remove newline characters and any extra whitespace
        item = line.strip()
        # Attempt to convert to int, float, or keep as string
        try:
            item = int(item)
        except ValueError:
            try:
                item = float(item)
            except ValueError:
                pass  # Keep as string
        reviewed_books.append(item)


reviewed_products = {}
count2 = 0
for product in products:
    count2 += 1
    print(count2)
    if product['asin'] in reviewed_books and product['asin'] not in reviewed_products:
        reviewed_products[product['asin']] = {'Categories': product['categories']}
        if 'price' in product:
            reviewed_products[product['asin']]['Price'] = product['price']


results = {}
count3 = 0
for reviewer in user_reviews:
    price = 0
    categories = []
    reviewWeights = 0
    for review in user_reviews[reviewer]:
        product = reviewed_products[review[0]]
        reviewWeights += review[1]
        if 'Price' in product:
            price += product['Price']
        for cats in product['Categories']:
            for cat in cats:
                if cat not in categories:
                    categories.append(cat)
    count3 += 1
    print(count3)
    results[reviewer] = {"Average Price": price/len(user_reviews[reviewer]), "Average Review": reviewWeights/len(user_reviews[reviewer]), "Categories Per Book": len(categories)/len(user_reviews[reviewer])}

with open("review_data", "w") as file:
  json.dump(results, file, indent=4)