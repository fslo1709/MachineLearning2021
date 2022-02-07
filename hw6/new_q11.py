import math as math
import numpy as np
import pandas as pd

def sort_arrays(X, y):
    #sort based on i feature
    sorted_X = []
    sorted_Y = []
    for i in range(10):
        new_Y = np.array([y])
        new_X = np.append(X, new_Y.transpose(), axis = 1)
        sorted_X_i = np.array(new_X[np.argsort(new_X[:,i])])
        sorted_Y_i = np.array(sorted_X_i[:,-1])
        sorted_X.append(np.delete(sorted_X_i, -1, axis = 1))
        sorted_Y.append(sorted_Y_i)

    finalX = np.array(sorted_X)
    finalY = np.array(sorted_Y)
    return finalX, finalY

def get_diamond_t(E):
    return math.sqrt((1 - E)/E)

def get_alpha_t(diamond_t):
    return np.log(diamond_t)

def get_epsilon_t(u_t, X, y, d_Array, N):
    upper_sum = 0.0
    for n in range(N):
        if (y[n] != d_Array[n]):
            upper_sum += u_t[n]
    return upper_sum / sum(u_t)

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

def update_u(X, y, d_Array, u_t, N):
    e_t = get_epsilon_t(u_t, X, y, d_Array, N)
    diamond_t = get_diamond_t(e_t)
    new_u = []
    for n in range(N):
        if (y[n] != d_Array[n]):
            new_u.append(u_t[n] * diamond_t)
        else:
            new_u.append(u_t[n] / diamond_t)

    return new_u, diamond_t

def Euin(u_t, y, s, N, u_indexes):
    min_theta = -1
    sigma = 0.0
    #Get initial summation
    for n in range(N):
        if (s != y[n]):
            sigma += u_t[int(u_indexes[n][10])]
    min_Eu = sigma

    #update summation per change of theta
    for n in range(N-1):
        if (s == y[n]):
            sigma += u_t[int(u_indexes[n][10])]
        else:
            sigma -= u_t[int(u_indexes[n][10])]
        if (sigma < min_Eu):
            min_Eu = sigma
            min_theta = n

    return min_theta, min_Eu

def stump(u_t, X, y, N):
    min_theta = -np.inf
    min_s = 1
    min_i = 0
    min_Eu = np.inf
    p = 0

    for i in range(10):
        #Get best theta for s = -1 and 1
        temp_theta_1, temp_Eu_1 = Euin(u_t, y[i], 1, N, X[i])
        temp_theta_m1, temp_Eu_m1 = Euin(u_t, y[i], -1, N, X[i])

        #update accordingly
        if (temp_Eu_1 < min_Eu):
            if (min_theta == -1):
                min_theta = -np.inf
            else:
                min_theta = (X[i][temp_theta_1][i] + X[i][temp_theta_1 + 1][i]) / 2
            min_s = 1
            min_i = i
            min_Eu = temp_Eu_1

        if (temp_Eu_m1 < min_Eu):
            if (min_theta == -1):
                min_theta = -np.inf
            else:
                min_theta = (X[i][temp_theta_m1][i] + X[i][temp_theta_m1 + 1][i]) / 2
            min_s = -1
            min_i = i
            min_Eu = temp_Eu_m1

    return [min_s, min_i, min_theta, min_Eu]

class AdaBoost:
    def __init__(self):
        self.u_T = []
        self.g_T = []
        self.alphas = []
        self.T = None

    def train(self, X, y, T = 500):
        N = len(y)
        u_t = [1/N] * N
        self.u_T.append(u_t)
        self.T = T

        sortedX, sortedY = sort_arrays(X, y)
        max_E_in_g_t = 0.0
        GT_values = [0] * N
        GT_decision = [0] * N
        flag = True 
        
        for t in range(self.T):
            g_t = stump(u_t, sortedX, sortedY, N)
            self.g_T.append(g_t)

            d_Array = decision_array(X, g_t[0], g_t[1], g_t[2])

            Ein_t = self.Error_function(y, d_Array, N)

            if (Ein_t > max_E_in_g_t):
                max_E_in_g_t = Ein_t

            if (t == 0):
                print("Q11: E_in for g_1 is: " + str(Ein_t))

            u_t, diamond_t = update_u(X, y, d_Array, u_t, N)
            self.u_T.append(u_t)

            alpha_t = get_alpha_t(diamond_t)
            self.alphas.append(alpha_t)

            sum_EGT = 0
            for n in range(N):
                GT_values[n] += d_Array[n] * alpha_t
                if (np.sign(GT_values[n]) != np.sign(y[n])):
                    sum_EGT += 1

            EGT = sum_EGT / N
            if (EGT <= 0.05 and flag):
                print("Q13: Smallest t for EGT <= 0.05 is: " + str(t+1))
                flag = False

        print("Q12: Max E_in error is " + str(max_E_in_g_t))

    def Eout_g1(self, X, y):
        N = len(y)
        g1 = self.g_T[0]
        d_Array = decision_array(X, g1[0], g1[1], g1[2])
        print("Q14: Eout for g_1 is: " + str(self.Error_function(y, d_Array, N)))

    def Eout_uniform(self, X, y):
        N = len(y)
        G_Uniform = [0] * N
        for t in range(self.T):
            gt = self.g_T[t]
            d_Array = decision_array(X, gt[0], gt[1], gt[2])
            for n in range(N):
                G_Uniform[n] += d_Array[n]
        Eout = 0
        for n in range(N):
            if (np.sign(G_Uniform[n]) != np.sign(y[n])):
                Eout += 1
        Eout = Eout / N
        print("Q15: Eout for G uniform is: " + str(Eout))

    def Eout_500(self, X, y):
        N = len(y)
        G = [0] * N
        for t in range(self.T):
            gt = self.g_T[t]
            d_Array = decision_array(X, gt[0], gt[1], gt[2])
            for n in range(N):
                G[n] += d_Array[n] * self.alphas[t]
        Eout = 0
        for n in range(N):
            if (np.sign(G[n]) != np.sign(y[n])):
                Eout += 1
        Eout = Eout / N
        print("Q16: Eout for G 500 is: " + str(Eout))

    def Error_function(self, y, d_Array, N):
        count = 0
        for n in range(N):
            if (y[n] != d_Array[n]):
                count += 1
        return count / N

if __name__ == "__main__":
    training_file = open("./data/hw6_train.dat", "r")
    training_lines = training_file.readlines()

    xTrainingVectors = []
    yTraining = []
    count = 0

    for line in training_lines:
        temp = line.split() 
        temp_x = []
        for i in range(10):
            temp_x.append(float(temp[i]))
        temp_x.append(int(count))
        xTrainingVectors.append(temp_x)
        yTraining.append(float(temp[10]))
        count += 1

    xTrainingNP = np.array(xTrainingVectors)
    yTrainingNP = np.array(yTraining)

    AB = AdaBoost()
    AB.train(xTrainingNP, yTrainingNP, T = 500)

    testing_file = open("./data/hw6_test.dat", "r")
    testing_lines = testing_file.readlines()

    xTestingVectors = []
    yTesting = []

    for line in testing_lines:
        temp = line.split()
        temp_x = []
        for i in range(10):
            temp_x.append(float(temp[i]))
        xTestingVectors.append(temp_x)
        yTesting.append(float(temp[10]))

    xTestingNP = np.array(xTestingVectors)
    yTestingNP = np.array(yTesting)

    AB.Eout_g1(xTestingNP, yTestingNP)
    AB.Eout_uniform(xTestingNP, yTestingNP)
    AB.Eout_500(xTestingNP, yTestingNP)