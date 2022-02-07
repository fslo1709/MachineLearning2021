import numpy as np
import random
import math

eta = 0.1

def generate(N_elements): #function to generate data sets according to the rules
    temp_x_set = []
    temp_y_set = []
    for i in range(N_elements):
        x = random.randint(0, 1) #coin simulation, 0 is y = -1, 1 is y = 1
        y = 2*x - 1
        if (x == 0):#if y = -1
            mean = [0, 4]
            covariance = [[0.4, 0], [0, 0.4]]
        else: #if y = 1
            mean = [2, 3]
            covariance = [[0.6, 0], [0, 0.6]]
        x1, x2 = np.random.multivariate_normal(mean, covariance) #get multivariate distribution
        temp_x_set.append([1, x1, x2])
        temp_y_set.append(y)
    for i in range(20): #20 outlier elements
        y = 1
        mean = [6, 0]
        covariance = [[0.3, 0], [0, 0.1]]
        x1, x2 = np.random.multivariate_normal(mean, covariance)
        temp_x_set.append([1, x1, x2])
        temp_y_set.append(y)
    res_x_set = np.array(temp_x_set) #numpy array for x
    res_y_set = np.array(temp_y_set) #numpy array for y
    return res_x_set, res_y_set

def pseudo_inverse(x_mat): #function to obtain the pseudo inverse
    x_mat_transpose = np.transpose(x_mat) #first operate x^T
    x_mat_transpose_times_x_mat = np.matmul(x_mat_transpose, x_mat) #get inner product
    x_mat_inv = np.linalg.inv(x_mat_transpose_times_x_mat) #get the inverse
    x_mat_result = np.matmul(x_mat_inv, x_mat_transpose) #multiply inverse by transpose
    return x_mat_result

def my_sign(a):
    if a > 0.0:
        return 1
    else:
        return -1

def theta(yn, w, xn):
    temp = yn * np.matmul(w, xn)
    return 1/(1 + math.exp(-temp))

total_E_linear_out = 0.0
total_E_logistical_out = 0.0

for i in range(100): #100 iterations
    Training_Data_Set, y_training_set = generate(200)
    Test_Data_Set, y_test_set = generate(5000)
    #Linear Regression:
    x_sword = pseudo_inverse(Training_Data_Set) #getting the pseudo inverse  
    W_LIN = np.matmul(x_sword, y_training_set) #getting w_lin
    
    #Logistical Regression
    logistical_weight = np.array([0.0]*3)
    for T in range(500):
        sum_E_in = np.array([0.0]*3)
        for n in range(220):
            sum_E_in = np.add(sum_E_in, theta(-y_training_set[n], logistical_weight, Training_Data_Set[n]) * (-y_training_set[n]*Training_Data_Set[n]))
        E_gradient_in = sum_E_in * (1/220)
        logistical_weight = np.add(logistical_weight, -eta*E_gradient_in)

    #Calculate E_linear_out 0/1 and E_logistical_out 0/1
    linear_error_sum_testing = 0 #start counters
    logistical_error_sum_testing = 0
    for i in range(5000): 
        y_linear_pred = np.matmul(W_LIN, Test_Data_Set[ i ])
        y_logistical_pred = np.matmul(logistical_weight, Test_Data_Set[ i ])
        if my_sign(y_linear_pred) != y_test_set[ i ]:
            linear_error_sum_testing += 1
        if my_sign(y_logistical_pred) != y_test_set[ i ]:
            logistical_error_sum_testing += 1
    E_linear_out = linear_error_sum_testing / 5000 #average E_linear_out
    E_logistical_out = logistical_error_sum_testing / 5000 #average E_logistical_out
    total_E_linear_out += E_linear_out
    total_E_logistical_out += E_logistical_out

avg_E_linear_out = total_E_linear_out / 100
avg_E_logistical_out = total_E_logistical_out / 100
print("[",avg_E_linear_out, ", ", avg_E_logistical_out, "]")      
