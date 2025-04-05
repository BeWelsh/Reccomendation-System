import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#Prepare
n_conponent = np.arange(1,10)

with open('review_data', 'r') as file:
    review_data = json.load(file)
rd_df = pd.DataFrame.from_dict(review_data)
formatted_data = rd_df.T
print(formatted_data.head())

#Create GGM Model
models = [GaussianMixture(n_components = n,
                          random_state=42).fit(formatted_data) for n in n_conponent]
#Plot
plt.plot(n_conponent,
         [m.bic(formatted_data) for m in models],
         label = 'BIC')

plt.legend()
plt.xlabel("Number Of Component")
plt.show()