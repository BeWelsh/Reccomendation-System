import gzip
import json

# parse function: Helps read a gzip json file
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

# parses the json files
reviews = parse('reviews_Kindle_Store_5.json.gz')
users_reviews = {}
test_reviews = {}
count = 0
# Iterates through each review
for r in reviews:
  # Adds the reviewerID if not yet seen
  if r['reviewerID'] not in users_reviews:
    if r['overall'] >= 3:
      test_reviews[r['reviewerID']] = {'asin': r['asin'], 'overall': r['overall']}
      users_reviews[r['reviewerID']] = []

    else:
      users_reviews[r['reviewerID']] = [{'asin':r['asin'], 'overall': r['overall']}]
  # Adds to reviewerID's list if already seen
  else:
    if r['reviewerID'] not in test_reviews and r['overall'] >= 3:
      test_reviews[r['reviewerID']] = {'asin': r['asin'], 'overall': r['overall']}
    else:
      users_reviews[r['reviewerID']].append({'asin':r['asin'], 'overall': r['overall']})

# Converts the sorted data to json files
with open("reviews_sorted.json", "w") as file:
  json.dump(users_reviews, file, indent=4)

with open("test_reviews_sorted.json", "w") as file:
  json.dump(test_reviews, file, indent=4)