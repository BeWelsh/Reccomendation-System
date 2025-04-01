# Reccomendation-System
Ben Welsh, Darius Saadat, Kaung Mo
Amazon Kindle Store Recommendation System

Problem Description:

Amazon Kindle is the premier online marketplace for eBooks hosting millions of books for users to buy, read and review. This project aims to build a reliable, efficient and flexible recommendation system for the Amazon Kindle Store by ranking Kindle books that a user is likely to be interested in based on their previous reviews. The goal of this project is to gain experience and practice creating functioning recommendation systems. 

Two datasets are being sourced from Julian McAuley, a Computer Science professor at UCSD. For the following datasets, variables colored red will be unnecessary variables that we won’t be using as input for our algorithms. 

The first dataset reviews contains 982,619 reviews of Kindle eBooks. The dataset was cleaned to be 5-core, such that each reviewer in the dataset must have at least 5 reviews. Therefore, we can reliably build user profiles as every user will have at least 5 reviews for our algorithms to base their preferences on. Each review in reviews contains the following data:

reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
asin - ID of the product, e.g. 0000013714
reviewerName - name of the reviewer
helpful - helpfulness rating of the review, e.g. 2/3
reviewText - text of the review
overall - rating of the product
summary - summary of the review
unixReviewTime - time of the review (unix time)
reviewTime - time of the review (raw)

The second dataset products contains the metadata of 9.4 million products across all Amazon sectors. We plan to clean this dataset such that only products included in the first dataset will remain. Therefore we remove products that aren’t Kindle eBooks, or eBooks that do not have reviews in reviews. Each product metadata in products contains the following data:


asin - ID of the product, e.g. 0000031852
title - name of the product (to be used as a readable identifier for products)
price - price in US dollars (at time of crawl - 2014)
imUrl - url of the product image
related - related products (also bought, also viewed, bought together, buy after viewing)
salesRank - sales rank information
brand - brand name
categories - list of categories the product belongs to

The use of specified variables of the two datasets will be further discussed below.

Our algorithms will be utilizing the following data to generate recommendations:
U = { u₁, u₂, …, uₘ }: the set of m users (identified by reviewerID)
Where u in U is defined as:
u = {reviewerID, asin, overall}
B = { b₁, b₂, …, bₙ }: the set of n Kindle books (identified by asin).
Where b in B is defined as:
b = {asin, title, price, related, salesRank, brand, categories}
General Algorithm Input and Output:
F:(u, k) → R 
Where the second input is a positive integer k representing the number of recommended books to return. For a given user u, the function outputs a ranked list of k recommendations:
R = {b1, b2, …, bk}: the set of the top ten recommended eBooks
Where each b in R is defined as:
b = {asin, title, price, related, salesRank, brand, categories}

Algorithms:
Matrix Factorization using Singular Value Decomposition (Kaung Mo):
	k-nearest Neighbors would be a good fit for creating a recommendation system, considering that our input is 5-core. This gives us the opportunity to create user profiles, which will allow us to calculate a user’s nearest neighbors (other users) based on their review history. This algorithm also requires minimal training as it will calculate similarity scores between users and recommend eBooks based on this information. 
	
Process:
For a given target user u, compute the similarity sim(u,v) with every other user in review dataset U based on their review features.
Compile the list of books that all k-nearest neighbors have reviewed. Sort this list of books based on a weight average of the neighbor’s ratings. 
Append this list to only include the top k recommended books and return as R.
Clustering-Based Recommendations (Darius Saadat):
Clustering-Based Recommendations would work well because we could use the large sample of different users and their reviews to group our user with like-minded users and then look for books that are popular within that cluster that the user hasn’t recommended and then suggest them based on their popularity and rating.
Process:
Apply a K-means clustering algorithm to group similar users in U based on their reviews of the same objects, prioritizing better reviews.
For a target user u, identify the cluster C ⊆ U that u belongs to.
Return the top k popular books (many purchases with high reviews) in C that u has not purchased
Deep Learning Methods (Ben Welsh):
	Deep Learning would work well for this problem, since many user review ratings are not linear and have underlying relationships. For example, if somebody buys two products at the same price point, does that mean they like all items at that price point? No! We would use neural collaborative filtering, which is frequently used for recommendation systems. It will also be computationally safe, since we are storing embeddings rather than a matrix of users/reviews. The algorithm would accept a user ID,  a number of returned items N, and candidate items (all not reviewed items). It will return the top N most compatible items for the user. 
Input:


user_ID: The id of the user we want to predict candidate items for (String)
N: The number of items we want to be ranked in the return (int)
Candidate_items: A list of items to be ranked by the model, the returned list (L) will be a subset of this list
Training data consisting of pairs (u, p) with observed interactions R(u, p).
Output:


A scoring function ŝ: U × P → ℝ learned via the neural network.
For a target user u, output a ranked list of products L = arg_top_k({ ŝ(u, p) : p ∈ P }).

Results:
For each of our algorithms, we expect to input a user and get an ordered set of 10 books that the user has not purchased that they would recommend based on their past purchases.  To determine which algorithm works best, we will run the algorithms on the same five users and compare the returned sets.  To determine this comparison, we will take in each book each user reviewed and see how many of the books we suggested appear on the ‘users also bought’ list associated with that book and add that to a score multiplied by the review.  The algorithm with the highest score should be the most accurate in determining a list of the books that the user would be interested in buying.
Input Users: {U1,U2,U3,U4,U5}
Book Recommendations:
Algo1_Rd = [U1{asin1 .. .asin10},U2{asin1 .. .asin10},U3{asin1 .. .asin10},U4{asin1 .. .asin10},U5{asin1 .. .asin10}]
Algo2_Rd = [U1{asin1 .. .asin10},U2{asin1 .. .asin10},U3{asin1 .. .asin10},U4{asin1 .. .asin10},U5{asin1 .. .asin10}]
Algo3_Rd = [U1{asin1 .. .asin10},U2{asin1 .. .asin10},U3{asin1 .. .asin10},U4{asin1 .. .asin10},U5{asin1 .. .asin10}]

For algo in Algorithms: 
	algo.score = 0
For user in length of Input Users:
	For book in user.reviews:
		Calculate num =  number of book recommendations for that user with algo that appear in book.’user also bought’
		Add num * book.rating to algo.score
Compare algo.score in algorithms to rank algorithms based on scores

Bibliography

Amazon 2014 Dataset

He, Ruining, and Julian McAuley. “Ups and Downs.” Proceedings of the 25th International Conference on World Wide Web, April 11, 2016. https://doi.org/10.1145/2872427.2883037.
 
McAuley, Julian. “Amazon Product Data.” Amazon review data. Accessed March 20, 2025. https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html.
 
McAuley, Julian, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. “Image-Based Recommendations on Styles and Substitutes.” Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, August 9, 2015. https://doi.org/10.1145/2766462.2767755.

K-Nearest Neighbor

“K-Nearest Neighbor(KNN) Algorithm.” GeeksforGeeks, January 29, 2025. https://www.geeksforgeeks.org/k-nearest-neighbours/.

Ibm. “What Is the K-Nearest Neighbors Algorithm?” IBM, February 12, 2025. https://www.ibm.com/think/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20(KNN)%20algorithm%20is%20a%20non,of%20an%20individual%20data%20point.

Hssina, Badr, Abdelkader Grota, and Mohammed Erritali. “Recommendation System Using the K-Nearest Neighbors and Singular Value Decomposition Algorithms.” International Journal of Electrical and Computer Engineering (IJECE) 11, no. 6 (December 1, 2021): 5541. https://doi.org/10.11591/ijece.v11i6.pp5541-5548. 

Clustering Algorithm

“K Means Clustering - Introduction.” GeeksforGeeks, January 15, 2025. https://www.geeksforgeeks.org/k-means-clustering-introduction/.

Ikotun, Abiodun M., Absalom E. Ezugwu, Laith Abualigah, Belal Abuhaija, and Jia Heming. “K-Means Clustering Algorithms: A Comprehensive Review, Variants Analysis, and Advances in the Era of Big Data.” Information Sciences 622 (April 2023): 178–210. https://doi.org/10.1016/j.ins.2022.11.139.

Deep Learning/Neural Collaborative Filtering

He, Xiangnan, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. "Neural collaborative filtering." In Proceedings of the 26th international conference on world wide web, pp. 173-182. 2017.
