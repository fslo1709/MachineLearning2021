import random

def sign_(a):
    if a>0:
        return 1
    else:
        return 0

file = open("hw1_train.dat", "r")
lines = file.readlines()
vectors = []
y_values = []
count_elements = 0
n = 0 #elements per vector
N = 0 #number of vectors
sum_squared_lengths = 0 #sum of all squared lengths of w_pla

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

for repetitions in range(1000): #run the experiment 1000 times

    w = [0.0] * (n+1)
    count = 0 #number of random pickings before stabilizing
    
    while count < 5*N:
        pick = random.randint(0, N-1) #randint picks a seed from memory time, so it's different each loop step
        sigma = 0.0 #sum of all products
        for i in range(n+1):
            sigma += w[i]*vectors[pick][i]
        if sign_(sigma) != sign_(y_values[pick]):
            count = 0
            for i in range(n+1):
                w[i] = w[i] + y_values[pick]*vectors[pick][i]
        else:
            count += 1 #signs are different, increase the counter and try another value
            
    squared_lengths = 0
    for weight in w:
        squared_lengths += weight*weight
    sum_squared_lengths += squared_lengths #add this to the sum to form the averagesquared_length = 0

average_squared_length = sum_squared_lengths / 1000

print(average_squared_length)
