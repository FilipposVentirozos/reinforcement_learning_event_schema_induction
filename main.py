import numpy as np

# Data representing the NEs data. Where 0 are the Non NEs, and 1,2,3 are the different type of NEs.
# We consider 50 different recipes where each of one has 30 tokens.
y_train = np.random.randint(4, size=(50, 30))
# We hypothesize a vector of 100 length
X_train = np.random.uniform(low=0, high=1, size=(50, 30, 100))
print(y_train[0, ])
print(X_train[0, ])



