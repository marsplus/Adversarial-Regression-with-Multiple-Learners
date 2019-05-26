import numpy as np
import cvxpy as cvx
import pandas as pd
import numpy.linalg as lin
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import train_test_split

np.random.seed(0)

## class definition for the Environment
class Environment(object):
    def __init__(self, path, N, preprocessor=None):
        super().__init__()
        self.dataPath = path
        self.numLearner = N
        self.learners = []

        data = pd.read_csv(self.dataPath, header=None)
        self.X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        self.y = y[:, np.newaxis]

        # pre-process the data
        if callable(preprocessor):
            self.X = preprocessor().fit_transform(self.X)

        # concatenate data and lable
        self.data = np.concatenate([self.X, np.ones((self.X.shape[0], 1)), self.y], axis=1)

    # split the whole dataset into training and testing subsets
    def generate_train_test_data(self, ratio=0.5):
        train_data, test_data = train_test_split(self.data, train_size=ratio)
        self.train_X, self.train_y = train_data[:, :-1], train_data[:, -1][:, np.newaxis]
        self.test_X, self.test_y = test_data[:, :-1], test_data[:, -1][:, np.newaxis]
        print("Finished generating train/test data\n")

    # part of the test data will be attacked by the attacker
    def generate_to_be_attacked_data(self, numAdv):
        to_be_attacked_data = self.test_X[:numAdv, :]
        return to_be_attacked_data



## class definition for the attacker
class Attacker(object):
    def __init__(self, target, Lambda, N):
        super().__init__()
        self.target = target
        self.Lambda = Lambda
        self.numLearner = N

    def generate_adversarial_data(self, theta, X):
        Theta = np.tile(theta, self.numLearner)

        assert(X.shape[0] == self.target.shape[0])
        assert(X.shape[1] == Theta.shape[0])
        numData = X.shape[0]
        numFeature = X.shape[1]

        c = np.ones((numData, 1)) * Theta[:, 0][-1]
        t = np.zeros((numFeature-1, numFeature-1))
        for i in range(self.numLearner):
            theta_i = Theta[:, i][:-1]
            t += np.dot(theta_i, theta_i.T)
        X_ast = np.dot( self.Lambda * X[:, :-1] + (self.target - c) * np.sum(Theta[:-1, :], axis=1)[:, np.newaxis].T, \
                        lin.inv(self.Lambda * np.identity(numFeature-1) + t) )
        # add additional feature that accounts for intercept
        return np.concatenate([X_ast, np.ones((X_ast.shape[0], 1))], axis=1)


## abstract class definition for the learner
class Learner(ABC):
    def __init__(self, Env):
        self.Environment = Env
        self.numLearner = Env.numLearner
        self.theta = None

    @abstractmethod
    def learn(self):
        pass

    def get_theta(self):
        assert(self.theta is not None)
        return self.theta

    def get_RMSE(self, attacked_X, numAdv):
        test_X = self.Environment.test_X
        test_y = self.Environment.test_y

        assert(self.theta is not None)
        Theta_opt = np.tile(self.theta, self.numLearner)  
        updated_test_X = np.concatenate([attacked_X, test_X[numAdv:, :]])
        R = np.dot(updated_test_X, self.theta) - test_y
        RMSE = np.sqrt(lin.norm(R, 2) / len(R))
        return RMSE


## Instantiation
class OLS_learner(Learner):
    def __init__(self, Env):
        super(OLS_learner, self).__init__(Env)

    def learn(self):
        train_X, train_y = self.Environment.train_X, self.Environment.train_y
        self.theta = lin.inv(train_X.T.dot(train_X)).dot(train_X.T).dot(train_y)


class Ridge_learner(Learner):
    def __init__(self, Env, cv_fold=5):
        super(Ridge_learner, self).__init__(Env)
        self.learner = RidgeCV(cv=cv_fold)

    def learn(self):
        train_X, train_y = self.Environment.train_X, self.Environment.train_y
        self.learner.fit(train_X, train_y.squeeze())
        self.theta = self.learner.coef_[:, np.newaxis]
        self.theta[-1] = self.learner.intercept_  


class Lasso_learner(Learner):
    def __init__(self, Env, cv_fold=5):
        super(Lasso_learner, self).__init__(Env)
        self.learner = LassoCV(cv=cv_fold)

    def learn(self):
        train_X, train_y = self.Environment.train_X, self.Environment.train_y
        self.learner.fit(train_X, train_y.squeeze())
        self.theta = self.learner.coef_[:, np.newaxis]
        self.theta[-1] = self.learner.intercept_  


class MLSG_learner(Learner):
    def __init__(self, Env, train_target, defender_Beta, defender_Lambda):
        super(MLSG_learner, self).__init__(Env)
        self.train_target = train_target
        self.defender_Beta = defender_Beta
        self.defender_Lambda = defender_Lambda

    def learn(self):
        N = self.numLearner
        Lambda = self.defender_Lambda
        Beta = self.defender_Beta
        target = self.train_target

        train_X, train_y = self.Environment.train_X, self.Environment.train_y
        m = train_X.shape[1]

        # MLSG 
        theta_opt = cvx.Variable((m, 1))
        obj = cvx.norm(train_X * theta_opt - train_y, 2) ** 2 + \
                                (Beta * (N+1)) / (2 * Lambda**2 ) * cvx.norm(target - train_y, 2)**2 * (cvx.norm(theta_opt[:m-1], 2)**4)
        prob = cvx.Problem(cvx.Minimize(obj))
        try:
            prob.solve(solver=cvx.CVXOPT, kktsolver='robust')
        except:
            prob.solve(solver=cvx.CVXOPT, kktsolver='robust', feastol=1e-6)
        if prob.status is "optimal":
            self.theta = theta_opt.value
        else:
            print("MLSG Learning Error!\n")





