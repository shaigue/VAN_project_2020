"""This is a class that represents the observation Models."""

"""Things that we can use:
- Gaussian Process:
    https://scikit-learn.org/stable/modules/gaussian_process.html

    https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-py

    https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py

- Naive Bayes:
    https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

- Covariance estimation:
    https://scikit-learn.org/stable/modules/covariance.html

- density estimation:
    https://scikit-learn.org/stable/modules/density.html
"""
from abc import ABC, abstractmethod
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt 

from sklearn.metrics import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier as GPC 
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from scipy.stats import multivariate_normal as Gaussian

from Field import Field

def parse_history(history: list) -> dict:
        """ history is a list of tuples:
        each tuple contains:
        - (x,y),Z where
        - (x,y) is the location in the board where the image was taken from, 
        - Z is a dict with `(i,j): s` 
        - (i,j) as keys(the grid cell observed), 
        - `s` the score 
        
        returns: a dictionary what maps grid cell to its np.array of observations:
        `(i,j): [[x0,y0,s0],[x1,y1,s1],...]`
        not as a tensor because the number of observations par coordinates may very.
        """
        parsed_history = dict()

        for entry in history:
            (x, y), Z = entry
            for (i, j), s in Z.items():
                if (i, j) not in parsed_history:
                    parsed_history[(i, j)] = np.array([[x, y, s]])
                else:
                    parsed_history[(i, j)] = \
                        np.concatenate((parsed_history[(i, j)], [[x,y,s]]))

        return parsed_history



class Model(ABC):
    """A model that predicts class scores of each grid cell, 
    given the history.
    
    Attributes:
        trained: have the model been trained?
        M: width of the board
        N: hight of the board

    Methods:
    """
    def __init__(self):
        super().__init__()
        self.trained = False
        self.M = 0
        self.N = 0
    
    @abstractmethod
    def train(self, field: Field) -> None:
        self.trained = True
        self.M = field.M
        self.N = field.N

    @abstractmethod
    def predict(self, history: list) -> np.array:
        if not self.trained:
            raise RuntimeError("cannot predict before the model is trained")
        return np.zeros((self.M, self.N))

    @abstractmethod
    def predict_proba(self, history: list) -> np.array:
        if not self.trained:
            raise RuntimeError("cannot predict before the model is trained")
        return np.zeros((self.M, self.N))



class GNB_Fixed(Model):
    """Gaussian Naive Bayes with fixed parameters, according to the 
        properties of the field.(size, mean = 0,1) and independent noise
    """
    def __init__(self):
        super().__init__()
        self.GNB_model = None

    @staticmethod
    def parse_history(history: list) -> dict:
        """for a dict: (i, j): [[x0, y0 , s0],...]
        to: (i, j): [[s0, s1, ...]]
        """
        p1_hist = parse_history(history)
        for (i, j), T in p1_hist.items():
            T = T[:, 2].reshape(1,-1)
            p1_hist[(i, j)] = T
        
        return p1_hist

    def train(self, field: Field) -> None:
        super().train(field)
        # NOTE: this is not the correct prior because it makes 
        # the classifier pretty bad if true prior is used.
        p1 = 0.5 # field.Q / (field.M * field.N)
        p0 = 0.5 # 1 - p1
        self.GNB_model = GaussianNB()
        self.GNB_model.class_prior_ = np.array([p0, p1])
        self.GNB_model.classes_ = np.array([0, 1])
        self.GNB_model.sigma_ = np.array([[field.noise_ind],[field.noise_ind]])
        self.GNB_model.theta_ = np.array([[0],[1]])


    def predict(self, history: list) -> np.array:
        super().predict(history)
        # parse history to get the
        pred = np.zeros((self.M, self.N))

        Data = self.parse_history(history)
        for (i,j), data in Data.items():
            pred[i,j] = int(self.GNB_model.predict(data))
        
        return pred

    def predict_proba(self, history: list) -> np.array:
        super().predict_proba(history)
        # parse history
        pred = np.full((self.M, self.N), 0.5)

        Data = self.parse_history(history)
        for (i,j), data in Data.items():
            p = self.GNB_model.predict_proba(data)
            pred[i,j] = p[0,1]
        
        return pred



class GP_corr(Model):
    """Gaussian Naive Bayes with fixed parameters, according to the 
        properties of the field.(size, mean = 0,1) and independent noise
    """
    def __init__(self, mu_0: float = 0, mu_1: float = 1, scale_x: float = 2,
     scale_y: float = 2, noise: float = 0.2, sigma: float = 0.75):
        super().__init__()
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.noise = noise
        self.sigma = sigma
        self.L = np.array([[1/scale_x, 0],[0, 1/scale_y]])

    def train(self, field: Field) -> None:
        super().train(field)
        # TODO: we might want to modifiy here the parameters, if have time

    def kernel_cov(self, Z1, Z2):
        """Z1 and Z2 are [x1,y1,s1], [x2,y2,s2], claculates the Squared Exponenetial kernal"""
        dist = Z1[:2] - Z2[:2]
        k = self.sigma * np.exp(-1/2 * np.dot(dist,np.dot(self.L,dist)))
        return k

    def predict(self, history: list) -> np.array:
        super().predict(history)

        # TODO: implement
        pred = self.predict_proba(history)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        return pred

    def predict_proba(self, history: list) -> np.array:
        super().predict_proba(history)

        pred = np.full((self.M, self.N), 0.5)

        Data = parse_history(history)

        for (i,j), data in Data.items():
            # calculate the covariance and mean matrices
            n = data.shape[0] # number of samples
            mean1 = np.full(n, self.mu_1)
            mean0 = np.full(n, self.mu_0)
            cov = np.zeros((n,n))
            for i1 in range(n):
                for j1 in range(n):
                    cov[i1,j1] = self.kernel_cov(data[i1],data[j1])
                    if i1 == j1:
                        cov[i1,i1] += self.noise
            S = data[:,2]
            # calculate the probability for each class
            S0 = Gaussian(mean0, cov)
            S1 = Gaussian(mean1, cov)
            p0 = S0.pdf(S)
            p1 = S1.pdf(S)
            # normalize to 1
            p_total = p0 + p1
            p1 /= p_total
            # p0 /= p_total
            pred[i,j] = p1
        
        return pred



if __name__ == "__main__":
    print("***Models.py main***")

    # M = 10
    # N = 10
    # F = 4
    # Q = 10
    # noise_ind = 0.2
    # p1 = Q / (M * N)
    # p0 = 1 - p1

    # field = Field(M,N,F,Q,noise_ind)
    # model = GP_corr()
    # model.train(field)
    
    # Z1 = {(1,1): 1, (2,2): 0.1, (4,4): 1.1, (3,4): 1.1}
    # Z2 = {(1,1): 0.9, (2,2): 0.1, (4,4): 1, (4,3): 0.1}
    # Z3 = {(2,2): 0.7, (1,1): 1, (4,4): 1, (3,4): 1.2}
    # history = [((1, 1), Z1), ((2, 2), Z2), ((3, 3), Z3)]
    
    # H = parse_history(history)
    # pprint(H)
    # pred = model.predict_proba(history)
    # for (x,y) in H.keys(): 
    #     print(f"{x},{y}: {pred[x,y]}")
    

# def generate_nb_train_data(field: Field, train_size: int):
#     """Generates `train_size` scores and true labels.

#         X: np.array of measured scores
#         y: np.array of actual labels
#     """ 
#     X_list = list()
#     y_list = list()
#     while len(X_list) < train_size:
#         field.randomize_objects()
#         inc = max(round(field.F / 2), 1)
        
#         for x in range(0, field.M, inc):
#             for y in range(0, field.N, inc):
#                 Z = field.generate_observation(x,y)
#                 for idx,s in Z.items():
#                     X_list.append(s)
#                     if idx in field.objects:
#                         y_list.append(1)
#                     else:
#                         y_list.append(0)


#     X = np.array(X_list)
#     X = X[:, np.newaxis]
#     y = np.array(y_list)

#     return X[:train_size],y[:train_size]

# n_train = 5000
# X_train, y_train = generate_nb_train_data(field, n_train)

# n_test = 2000
# X_test, y_test = generate_nb_train_data(field, n_test)


# y_pred = gauss_nb.predict(X_test)
# print(f"accuracy: {accuracy_score(y_test, y_pred)}")

# X0 = np.array([[0.1,0.1,0.1]])
# print(gauss_nb.predict_proba(X0))