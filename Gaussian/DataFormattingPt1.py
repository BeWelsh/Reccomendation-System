import gzip
import json


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

reviews = parse('reviews_Kindle_Store_5.json.gz')
products = parse('meta_Kindle_Store.json.gz')
users_reviews = {}
test_reviews = {}
print(type(test_reviews))
count = 0
for r in reviews:
  if r['reviewerID'] not in users_reviews:
    if r['overall'] >= 3:
      test_reviews[r['reviewerID']] = {'asin': r['asin'], 'overall': r['overall']}
      users_reviews[r['reviewerID']] = []

    else:
      users_reviews[r['reviewerID']] = [{'asin':r['asin'], 'overall': r['overall']}]

  else:
    if r['reviewerID'] not in test_reviews and r['overall'] >= 3:
      test_reviews[r['reviewerID']] = {'asin': r['asin'], 'overall': r['overall']}
    else:
      users_reviews[r['reviewerID']].append({'asin':r['asin'], 'overall': r['overall']})
  count += 1
  print(count)

with open("reviews_sorted.json", "w") as file:
  json.dump(users_reviews, file, indent=4)

with open("test_reviews_sorted.json", "w") as file:
  json.dump(test_reviews, file, indent=4)