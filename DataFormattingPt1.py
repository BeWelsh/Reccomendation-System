import gzip
import json


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

reviews = parse('reviews_Kindle_Store_5.json.gz')
products = parse('meta_Kindle_Store.json.gz')
users_reviews = {}
print(type(reviews))
count = 0
for r in reviews:
  if r['reviewerID'] not in users_reviews:
    users_reviews[r['reviewerID']] = {[r['asin'],r['overall']]}
  else:
    users_reviews[r['reviewerID']].append([r['asin'], r['overall']])
  count += 1
  print(count)

with open("reviews_sorted", "w") as file:
  json.dump(users_reviews, file, indent=4)