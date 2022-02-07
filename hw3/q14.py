import math
import numpy as np

training_file = open("./data/hw3_train.dat", "r")
training_lines = training_file.readlines()
testing_file = open("./data/hw3_test.dat", "r")
testing_lines = testing_file.readlines()

xTrainingVectors = []
yTraining = []
xTestingVectors = []
yTesting = []

for line in training_lines: #get x and y values
    temp = line.split() #temp holds each line broken into a list, with n elements for x values and a final value y
    temp_x = []
    temp_x.append(1.0)
    for i in range(10):
        temp_x.append(float(temp[i]))
    xTrainingVectors.append(temp_x)
    yTraining.append(float(temp[10]))

for line in testing_lines: #get x and y values
    temp = line.split() #temp holds each line broken into a list, with n elements for x values and a final value y
    temp_x = []
    temp_x.append(1.0)
    for i in range(10):
        temp_x.append(float(temp[i]))
    xTestingVectors.append(temp_x)
    yTesting.append(float(temp[10]))

xTrainingNP = np.array(xTrainingVectors)
yTrainingNP = np.array(yTraining)
xTestingNP = np.array(xTestingVectors)
yTestingNP = np.array(yTesting)

def my_sign(a):
    if a > 0.0:
        return 1
    else:
        return -1

#returns the z vector
def q_2_transform(line):
    temp_z = [1]
    for i in range(10):
        temp_z.append(line[i+1])
    for i in range(9):
        for j in range(i+1, 10):
            temp_z.append(line[i+1] * line[j+1])
    for i in range(10):
        temp_z.append(math.pow(line[i+1], 2))
    return temp_z

def pseudo_inverse(x_mat): #function to obtain the pseudo inverse
    x_mat_transpose = np.transpose(x_mat) #first operate x^T
    x_mat_transpose_times_x_mat = np.matmul(x_mat_transpose, x_mat) #get inner product
    x_mat_inv = np.linalg.inv(x_mat_transpose_times_x_mat) #get the inverse
    x_mat_result = np.matmul(x_mat_inv, x_mat_transpose) #multiply inverse by transpose
    return x_mat_result

zTrainingVectors = []
zTestingVectors = []


for row in xTrainingNP:
    zTrainingVectors.append(q_2_transform(row))

for row in xTestingNP:
    zTestingVectors.append(q_2_transform(row))    

zTrainingNP = np.array(zTrainingVectors)
zTestingNP = np.array(zTestingVectors)
trainingRows, trainingCols = zTrainingNP.shape
testingRows, testingCols = zTestingNP.shape

zSword = pseudo_inverse(zTrainingNP)
wLIN = np.matmul(zSword, yTraining)

error_sum = 0
i = 0

for element in zTrainingNP: #Calculating 0/1 error
    yPrediction = np.matmul(wLIN, element)
    if my_sign(yPrediction) != yTrainingNP[i]:
        error_sum += 1
    i += 1

E_in = error_sum / trainingRows

error_sum = 0
i = 0

for element in zTestingNP:
    yPrediction = np.matmul(wLIN, element)
    if my_sign(yPrediction) != yTestingNP[i]:
        error_sum += 1
    i += 1

E_out = error_sum / testingRows
print(abs(E_in - E_out))
