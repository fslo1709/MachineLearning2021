import numpy as np
import random
import math

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

total_E_in_E_out = 0.0

for i in range(100): #100 iterations
    Training_Data_Set, y_training_set = generate(200)
    Test_Data_Set, y_test_set = generate(5000)

    x_sword = pseudo_inverse(Training_Data_Set) #getting the pseudo inverse
    
    W_LIN = np.matmul(x_sword, y_training_set) #getting w_lin

    error_sum_training = 0
    for i in range(200): #Calculating E_in 0/1
        y_pred = np.matmul(W_LIN, Training_Data_Set[ i ]) #predicting y
        if my_sign(y_pred) != y_training_set[ i ]:
            error_sum_training += 1
    E_in = error_sum_training / 200 #average E_in

    error_sum_testing = 0
    for i in range(5000): #Calculating E_out 0/1
        y_pred = np.matmul(W_LIN, Test_Data_Set[ i ])
        if my_sign(y_pred) != y_test_set[ i ]:
            error_sum_testing += 1
    E_out = error_sum_testing / 5000 #average E_out

    total_E_in_E_out += abs(E_in - E_out) #to get the average of the 100 iterations


avg_total_E = total_E_in_E_out / 100
print(avg_total_E)      
