import numpy as np
import matplotlib.pyplot as plt


def readFile(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    points = []
    num = -1
    thePoint = []

    count = 0

    for line in Lines:
        # print(count)
        line = line.strip()
        l = line.split(" ")
        res = [eval(i) for i in l]
        # print(res)
        if count == 0:
            num = res[0]
        elif count == num + 1:
            thePoint = res
        else:
            res.append(count - 1)
            points.append(res)
        count = count + 1

    return num, points, thePoint


def calcOrient(p1, p2, p3):
    mat = np.ones((3, 3))
    mat[1][0] = p1[0]
    mat[2][0] = p1[1]
    mat[1][1] = p2[0]
    mat[2][1] = p2[1]
    mat[1][2] = p3[0]
    mat[2][2] = p3[1]
    # print(mat)

    det = np.linalg.det(mat)
    # print(det)

    ans = 0

    if det > 0:
        ans = 'L'
    if det < 0:
        ans = 'R'
    return ans


# def bruteforce():
#     ans = []
#     for i in range(input[0]):
#         if (calcOrient(input[2],input[1][i],input[1][(i+1)%input[0]]) == 'L' and calcOrient(input[2],input[1][i],input[1][(i-1)%input[0]]) == 'L'):
#             ans.append(i)
#     return ans


def findRightSide1(start, end):

    left = start
    right = end
    middle = left + (end - start) // 2

    # xs = []
    # ys = []
    # n = []
    # for j in range(start, end + 1):
    #     xs.append(input[1][j][0])
    #     ys.append(input[1][j][1])
    #     n.append(input[1][j][2])
    # xs.append(input[2][0])
    # ys.append(input[2][1])
    # n.append("thePoint")
    #
    # fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.scatter(xs, ys)
    #
    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (xs[i], ys[i]))
    #
    # plt.show()

    if right - left > 1:

        orientMR = calcOrient(input[2], input[1][middle], input[1][middle + 1])
        orientML = calcOrient(input[2], input[1][middle], input[1][middle - 1])

        if orientMR == 'L' and orientML == 'L':
            return input[1][middle][2]

        if orientMR == 'R':
            return findRightSide1(middle+1, right)
        if orientML == 'R':
            return findRightSide1(left, middle-1)

        return -1
    else:
        if middle== left and middle == right:
            return middle

        if middle == left:
            orientMR = calcOrient(input[2], input[1][middle], input[1][middle + 1])
            orientML = calcOrient(input[2], input[1][middle], input[1][(middle - 1) % input[0]])
            if orientMR == 'L' and orientML == 'L':
                return middle
            else:
                return right
        else:
            orientML = calcOrient(input[2], input[1][middle], input[1][middle - 1])
            orientMR = calcOrient(input[2], input[1][middle], input[1][(middle + 1) % input[0]])
            if orientMR == 'L' and orientML == 'L':
                return middle
            else:
                return left
        return -1


input = readFile('input3.txt')
# print(input)
# print("****************")
# print("the correct answer is ", bruteforce())
print("****************")
print("the answer is ",findRightSide1(0,len(input[1])-1))
print("****************")