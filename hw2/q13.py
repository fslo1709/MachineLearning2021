import numpy as np
import random
import math

def generate(N_elements, training = False): #function to generate data sets according to the rules
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
    if training:
        return res_x_set, res_y_set
    else:
        return res_x_set

def pseudo_inverse(x_mat): #function to obtain the pseudo inverse
    x_mat_transpose = np.transpose(x_mat) #first operate x^T
    x_mat_transpose_times_x_mat = np.matmul(x_mat_transpose, x_mat) #get inner product
    x_mat_inv = np.linalg.inv(x_mat_transpose_times_x_mat) #get the inverse
    x_mat_result = np.matmul(x_mat_inv, x_mat_transpose) #multiply inverse by transpose
    return x_mat_result

total_E_in = 0.0

for i in range(100): #100 iterations
    Training_Data_Set, y_set = generate(200, training = True)
    Test_Data_Set = generate(5000)

    x_sword = pseudo_inverse(Training_Data_Set) #getting the pseudo inverse
    
    W_LIN = np.matmul(x_sword, y_set) #getting w_lin

    error_sum = 0.0
    for i in range(200): #Calculating avg sqr error
        wTx_n = np.matmul(W_LIN, Training_Data_Set[ i ])
        error_sum += (wTx_n - y_set[ i ]) ** 2
        
    E_in = error_sum / 200
    total_E_in += E_in #to get the average of the 100 iterations

avg_total_E_in = total_E_in / 100
print(avg_total_E_in)

            
