import random

def sign_(a):
    if a>0:
        return 1
    else:
        return 0

file = open("hw1_q8.dat", "r")
lines = file.readlines()
vectors = []
y_values = []
count_elements = 0
n = 0 #elements per vector
N = 0 #number of vectors

for line in lines: #get x and y values
    temp = line.split() #temp holds each line broken into a list, with n elements for x values and a final value y
    x_array = []
    x_array.append(1.0)
    n = len(temp)-1 #how many elements per vector
    for i in range(n):
        x_array.append(float(temp[i]))
    vectors.append(x_array)
    y_values.append(float(temp[n]))
    count_elements+=1
    N += 1

#triple loop to get all combinations of D
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            indexes = [i, j, k]
            count = 0
            w = [0.0] * 3
            bol = True
            while bol:
                for index in indexes:
                    sigma = 0.0 #vector product result
                    for a in range(n+1):
                        sigma += w[a] * vectors[index][a] #w^T_(t)*x_n(t)
                    if sign_(sigma) != sign_(y_values[index]):
                        for a in range(n+1):
                            w[a] += y_values[index]*vectors[index][a] #w_(t+1) = w_(t) + y_n(t) * x_n(t)
                        break
                else:
                    bol = False
            diff_count = 0
            for t in range(N):
                if (t != i and t != j and t != k): #try the options we haven't tried yet
                    sigma = 0.0
                    for a in range(n+1):
                        sigma += w[a] * vectors[t][a] #w^T_(t)*x_n(t)
                    if sign_(sigma) != sign_(y_values[t]): #iff 
                        diff_count += 1
            print(diff_count)
                

