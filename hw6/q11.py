import math as math
import numpy as np
import pandas as pd

def compute_error(u_i, X, y, s, i, theta, N):
    upper_sum = 0.0
    for n in range(N): 
        if (y[n] != decision(X[n], s, i, theta)):
            upper_sum += u_i[n]
    return upper_sum / sum(u_i)

def diamond_t(E):
    return math.sqrt((1 - E)/E)

def compute_alpha(E):
    return np.log(diamond_t(E))

def update_u(u_i, X, y, s, i, theta, N):
    E = compute_error(u_i, X, y, s, i, theta, N)
    d = diamond_t(E)
    new_u_i = np.zeros(N)
    for n in range(N):
        if (y[n] != decision(X[n], s, i, theta)):
            new_u_i[n] = u_i[n] * d
        else:
            new_u_i[n] = u_i[n] / d
    return new_u_i

def decision(x, s, i, theta):
    if (x[i] - theta >= 0):
        return s
    else:
        return -s

def getEuin(u, X, y, s, i, theta, N):
    E_sum = 0.0

    for n in range(N):
        if (y[n] != decision(X[n], s, i, theta)):
            E_sum += u[n]

    return E_sum / N

def stump(u_t, X, y, N):
    #initialize comparison variables for answer
    min_theta = np.inf
    min_s = 0
    min_i = 0
    min_Euin = np.inf

    for i in range(10): #go through the 10 features
        #sort based on i feature
        new_Y = np.array([y])
        new_X = np.append(X, new_Y.transpose(), axis = 1)
        sorted_X = np.array(new_X[np.argsort(new_X[:,i])])
        sorted_Y = np.array(sorted_X[:, -1])
        np.delete(sorted_X, -1, axis = 1)
        
        #initialize comparison variables for i
        theta = -np.inf
        min_theta_i = theta
        E_min_i = np.inf
        min_s_i = -1

        for n in range(N):
            #update theta
            if n > 0:
                theta = (sorted_X[n][i] + sorted_X[n - 1][i]) / 2
            
            #get error for s = {-1, 1}
            E_positive = getEuin(u_t, sorted_X, sorted_Y, 1, i, theta, N)
            E_negative = getEuin(u_t, sorted_X, sorted_Y, -1, i, theta, N)

            if (E_positive < E_negative):
                E_comp = E_positive
                s = 1
            else:
                E_comp = E_negative
                s = -1

            #update if current E_u_in is the smallest one
            if (E_comp < E_min_i):
                E_min_i = E_comp
                min_theta_i = theta
                min_s_i = s

        #update if current E_u_in is the smallest one
        if (E_min_i < min_Euin):
            min_Euin = E_min_i
            min_theta = min_theta_i
            min_s = min_s_i
            min_i = i

    h = [min_s, min_i, min_theta]
    return h

class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.g_T = []
        self.u_T = []
        self.T = None
        self.training_errors = []
        self.prediction_errors = []
        self.s = None

    def fit(self, X, y, T = 500):
        self.alphas = []
        self.training_errors = []
        self.T = T
        Gx = []
        alpha_t = 0.0
        N = len(y)
        u_i = np.ones(N) * (1 / N)
        self.u_T.append(u_i)

        for t in range(0, self.T):
            g_t = stump(u_i, X, y, N)
            self.g_T.append(g_t)

            u_i = update_u(u_i, X, y, g_t[0], g_t[1], g_t[2], N)
            self.u_T.append(u_i)

            #u_i, X, y, s, i, theta, N
            error_t = compute_error(u_i, X, y, g_t[0], g_t[1], g_t[2], N)
            self.training_errors.append(error_t)

            alpha_t = compute_alpha(error_t)
            self.alphas.append(alpha_t)

    def binary_error(self, X, y, t):
        N = len(X)
        E_sum = 0
        for n in range(N):
            d = decision(X[n], self.g_T[t][0], self.g_T[t][1], self.g_T[t][2])
            if (d != y[n]):
                E_sum += 1
        return E_sum / N

if __name__ == "__main__":
    training_file = open("./data/hw6_train.dat", "r")
    training_lines = training_file.readlines()

    xTrainingVectors = []
    yTraining = []

    for line in training_lines: #get x and y values
        temp = line.split() #temp holds each line broken into a list, with n elements for x values and a final value y
        temp_x = []
        for i in range(10):
            temp_x.append(float(temp[i]))
        xTrainingVectors.append(temp_x)
        yTraining.append(float(temp[10]))

    xTrainingNP = np.array(xTrainingVectors)
    yTrainingNP = np.array(yTraining)

    aboost = AdaBoost()
    aboost.fit(xTrainingNP, yTrainingNP, T = 1)
    E_in_g1 = aboost.binary_error(xTrainingNP, yTrainingNP, t = 0)
    print(E_in_g1)