import math as math
import numpy as np
import pandas as pd

def compute_error(u_i, X, y, s, i, theta, N, decided):
    upper_sum = 0.0
    for n in range(N): 
        if (y[n] != decided[n]):
            upper_sum += u_i[n]
    return upper_sum / sum(u_i)

def diamond_t(E):
    return math.sqrt((1 - E)/E)

def compute_alpha(E):
    return np.log(diamond_t(E))

def update_u(u_i, X, y, s, i, theta, N, decided):
    E = compute_error(u_i, X, y, s, i, theta, N, decided)
    d = diamond_t(E)
    new_u_i = np.zeros(N)
    result_g = []
    for n in range(N):
        if (y[n] != decided[n]):
            new_u_i[n] = u_i[n] * d
        else:
            new_u_i[n] = u_i[n] / d
        result_g.append(decided[n])
    return result_g, new_u_i

def decision(x, s, i, theta):
    if (x[i] - theta >= 0):
        return s
    else:
        return -s

def decision_array(X, s, i, theta):
    D = []
    for n in range(len(X)):
        d = decision(X[n], s, i, theta)
        D.append(d)

    return D

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
        #initialize comparison variables for i
        theta = -np.inf
        min_theta_i = theta
        E_min_i = np.inf
        min_s_i = -1

        for n in range(N):
            #update theta
            if n > 0:
                theta = (X[i][n][i] + X[i][n - 1][i]) / 2
            
            #get error for s = {-1, 1}
            E_positive = getEuin(u_t, X[i], y[i], 1, i, theta, N)
            E_negative = getEuin(u_t, X[i], y[i], -1, i, theta, N)

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

def sort_arrays(X, y):
    #sort based on i feature
    sorted_X = []
    sorted_Y = []
    for i in range(10):
        new_Y = np.array([y])
        new_X = np.append(X, new_Y.transpose(), axis = 1)
        sorted_X_i = np.array(new_X[np.argsort(new_X[:,i])])
        sorted_Y_i = np.array(sorted_X_i[:,-1])
        np.delete(sorted_X_i, -1, axis = 1)
        sorted_X.append(sorted_X_i)
        sorted_Y.append(sorted_Y_i)

    finalX = np.array(sorted_X)
    finalY = np.array(sorted_Y)
    return finalX, finalY

class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.g_T = []
        self.G = []
        self.u_T = []
        self.T = None
        self.training_errors = []
        self.prediction_errors = []
        self.s = None

    def fit(self, X, y, T = 500):
        self.alphas = []
        self.training_errors = []
        self.T = T
        G_sum = 0
        alpha_t = 0.0
        N = len(y)
        u_i = np.ones(N) * (1 / N)
        self.u_T.append(u_i)
        E_in_max = 0

        sorted_X, sorted_Y = sort_arrays(X, y)

        for t in range(0, self.T):
            g_t = stump(u_i, sorted_X, sorted_Y, N)
            self.g_T.append(g_t)

            decided = decision_array(X, g_t[0], g_t[1], g_t[2])

            result_g, u_i = update_u(u_i, X, y, g_t[0], g_t[1], g_t[2], N, decided)
            self.u_T.append(u_i)

            #u_i, X, y, s, i, theta, N
            error_t = compute_error(u_i, X, y, g_t[0], g_t[1], g_t[2], N, decided)
            self.training_errors.append(error_t)

            alpha_t = compute_alpha(error_t)
            self.alphas.append(alpha_t)

            G_sum += alpha_t * np.array(result_g)

            E_G = np.sum(np.sign(G_sum) != y) / N

            print(E_G)
            print(str(t) + " done ")

            if (E_G < 0.05):
                break

        return self.alphas, self.g_T


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
    alphas, G = aboost.fit(xTrainingNP, yTrainingNP, T = 500)
    