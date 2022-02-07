points = [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]]

possible = [6.5, 7.5, 8.5, 9.5, 10.5]

for p in possible:
    ans = 0.0
    points[2][0] = p

    # second possibility
    slope = (points[0][1] - points[2][1])/(points[0][0] - points[2][0])
    intercept = points[0][1] - slope * points[0][0]
    diff = abs(points[1][1] - (slope*points[1][0] + intercept))
    ans += diff*diff

    # third possibility 
    slope = (points[1][1] - points[2][1])/(points[1][0] - points[2][0])
    intercept = points[1][1] - slope * points[1][0]
    diff = abs(points[0][1] - (slope*points[0][0] + intercept))
    ans += diff*diff

    print((ans+4.0)/3)