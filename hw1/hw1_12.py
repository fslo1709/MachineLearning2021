import random

dice_a = ["o", "g", "o", "g", "o", "g"]
dice_b = ["g", "g", "o", "o", "o", "g"]
dice_c = ["o", "o", "o", "o", "o", "g"]
dice_d = ["g", "g", "g", "o", "g", "o"]
dices = [dice_a, dice_b, dice_c, dice_d]

all_green = 0
total = 0
for i in range(1000000):
    green = 0
    for j in range(5):
        temp = random.choice(dices)
        res = random.choice(temp)
        if res == "g":
            green += 1
    if green == 5:
        all_green += 1
    total += 1

print((all_green*1.0)/(total*1.0))
    
