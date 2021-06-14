import numpy as np


class ToyData:

    def __init__(self):
        # Toy Data
        # Data representing the NEs data. Where 0 are the Non NEs, and 1,2,3 are the different type of NEs.
        # We consider 50 different recipes where each of one has 30 tokens.
        self.y_train = np.random.randint(4, size=(50, 30))
        # We hypothesize a vector of 100 length
        self.X_train = np.random.uniform(low=0, high=1, size=(50, 30, 100))
        # print(self.y_train[0, ])
        # print(self.y_train[0, ].shape)
        # print(self.X_train[0, ])
        print(self.X_train[0, ].shape)
        # for i in np.nditer(self.X_train):
        # for i in self.X_train:
        #     print(i)
        #     print(i.shape)

    def recipe_iter(self):
        for X, y in zip(self.X_train, self.y_train):
            yield X, y

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


if __name__ == '__main__':
    for i in ToyData().recipe_iter():
        print(i)
