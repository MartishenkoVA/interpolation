__author__ = 'Admin'

from numpy import *
import matplotlib.pyplot as plt

FILE_NAME = "inputgr1_1.txt"
X = []
Y = []
Step = [] # N-1

PV = array([ [ 12.00, -12.00], [-12.00, 12.00] ])
PF = array([ [  6.00,   6.00], [ -6.00, -6.00] ])
PQ = array([ [ -0.35,  -0.15], [ -0.15, -0.35] ])
MV = PF.transpose()
MF = array([ [  4.00,   2.00], [  2.00,  4.00] ])
MQ = array([ [ -0.05,  -1/30], [  1/30,  0.05] ])

# print(PV, PF, PQ, MV, MF, MQ, sep="\n\n")

def read_data(filename):
    x_array = []
    y_array = []
    step_array = []
    with open(filename) as file:
        data = file.readlines()
        i = 0
        for line in data:
            data_line = line.strip()
            if len(data_line) != 0:
                array = data_line.split(",")
                print(str(data_line))
                if i == 0:
                    for item in array:
                        print(str(i) + " " + item)
                        x_array.append(float(item))
                elif i == 1:
                    for item in array:
                        print(str(i) + " " + item)
                        y_array.append(float(item))
                elif i == 2:
                    for item in array:
                        print(str(i) + " " + item)
                        step_array.append(float(item))
            i += 1
    return x_array, y_array, step_array

def get_dx(array):
    return [x-array[array.index(x)-1] for x in array][1:]

def get_matrix(delta_X, koef_matrix, formula):
    array = [ koef_matrix*formula(x) for x in delta_X]
    n = len(array)

    matrix = zeros( (n+1,n+1) )
    i1, j1 = 0, 0
    for k in range(n):
        i1 = k
        for i in range(len(koef_matrix)):
            j1 = k
            for j in range(len(koef_matrix[0])):
                matrix[i1][j1] += array[k][i][j]
                j1 += 1
            i1 += 1
    return matrix

def check_result(koef, x):
    time = len(koef) - 1
    result = 0.0
    for i in koef:
        st = 1
        for j in range(time):
            st *= x
        result += i * st
        time -= 1
    #if result < 0.000000001: # 1*10^-09
        #print("Correct result. {0} = 0\n".format(result))

def solve(koef):
    des = roots(koef)
    for i in des:
        #print(i)
        check_result(koef, i)
    return des

def solve_poly_4(k4, k3, k2, k1, k0 = 1):
    return solve([k4,k3,k2,k1,k0])

X,Y,Step = read_data(FILE_NAME)
N = len(X)
print("X = " + str(X))
print("Y = " + str(Y))
print("Step = " + str(Step) + "\n")

if len(X) != len(Y) or (len(X) - 1) != len(Step):
    print("Wrong input data.\nLength of X is " + str(len(X)) + "\nLength of Y is " + str(len(Y)) + "\nLength of Step is " + str(len(Step)))
    exit(-1)

dx = get_dx(X)
#print("dx = " + str(dx))

C = zeros( (2*N,2*N) )
D = zeros( (2*N, N) )

A1 = get_matrix(dx, PV, lambda x: 1/pow(x,3))
A2 = get_matrix(dx, PF, lambda x: 1/pow(x,2))
A3 = get_matrix(dx, PQ, lambda x: x)
B1 = get_matrix(dx, MV, lambda x: 1/pow(x,2))
B2 = get_matrix(dx, MF, lambda x: 1/x)
B3 = get_matrix(dx, MQ, lambda x: pow(x,2))

C = vstack( (hstack((A2,A3)), hstack((B2,B3))) )
D = vstack( (A1,B1) )

inv_C = matrix(C.tolist()).I
Y = matrix(Y).T
Solution = -1*inv_C*D*Y

F = Solution[0:N,:]
Q = Solution[N:2*N,:]
#print("F = ", F, sep="\n")
#print("Q = ", Q, sep="\n")

# Step = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01] # N-1
# Step = [0.01, 0.01, 0.01, 0.01, 0.01] # N-1
# Step = [0.1, 0.1, 0.1, 0.1, 0.1] # N-1
# Step = [1, 1, 1, 1, 1, 1] # N-1
# Step = [0.1, 0.1]
length = 1
for i in range(len(Step)):
    sub_length = int(dx[i]/Step[i])
    length += sub_length
    if (dx[i] - sub_length * Step[i]) >= Step[i]/100:
        length += 1
#print("Length = " + str(length))

X_t = zeros( (length,1) )
Y_t = zeros( (length,1) )
A_t = zeros( (length,4) )

i = 0
j = 0
count = 0
for x in X[:len(X)-1]:
    while True :
        if i != 0 and x == X[i]:
            x += Step[i]
            continue


        A_t[j][3] = Q[i] + (Q[i+1] - Q[i]) / dx[i] * (x - X[i])
        A_t[j][2] = Q[i] * (x - X[i]) + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**2)/2 + A_t[count][2]
        A_t[j][1] = Q[i] * ((x - X[i])**2)/2 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**3)/6 + A_t[count][2] * (x - X[i]) + A_t[count][1]
        A_t[j][0] = F[i] + Q[i] * ((x - X[i])**3)/6 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**4)/24 + A_t[count][2] * ((x - X[i])**2)/2 + A_t[count][1] * (x - X[i])

        X_t[j] = x
        Y_t[j] = Y[i] + F[i] * (x - X[i]) + Q[i] * ((x - X[i])**4)/24 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**5)/120 + A_t[count][2] * ((x - X[i])**3)/6 + A_t[count][1] * ((x - X[i])**2/2)

        j += 1

        if (x + Step[i]) <= (X[i+1]):
            x += Step[i]
        else:
            break

    if (X[i+1] - x) > Step[i]/100:
        x = X[i+1]

        A_t[j][3] = Q[i] + (Q[i+1] - Q[i]) / dx[i] * (x - X[i])
        A_t[j][2] = Q[i] * (x - X[i]) + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**2)/2 + A_t[count][2]
        A_t[j][1] = Q[i] * ((x - X[i])**2)/2 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**3)/6 + A_t[count][2] * (x - X[i]) + A_t[count][1]
        A_t[j][0] = F[i] + Q[i] * ((x - X[i])**3)/6 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**4)/24 + A_t[count][2] * ((x - X[i])**2)/2 + A_t[count][1] * (x - X[i])

        X_t[j] = x
        Y_t[j] = Y[i] + F[i] * (x - X[i]) + Q[i] * ((x - X[i])**4)/24 + (Q[i+1] - Q[i]) / dx[i] * ((x - X[i])**5)/120 + A_t[count][2] * ((x - X[i])**3)/6 + A_t[count][1] * ((x - X[i])**2)/2

        j += 1
    count = j-1
    i += 1

# print("Y_t = \n" + str(Y_t) + "\n")
# print("A_t = \n" + str(A_t) + "\n")

A_transpose = matrix(A_t).T
AA = A_transpose * matrix(A_t)
Koef = -1 * matrix(AA).I * matrix(A_transpose) * matrix(Y_t)
Koef = array(Koef.T)[0][:]
print("Koef = \n" + str(Koef) + "\n")

k1,k2,k3,k4 = Koef
roots_of_equation = solve_poly_4(k1,k2,k3,k4)
print("Roots = \n" + str(roots_of_equation) + "\n")

num_of_columns = 8

while True:
    column_formula = []
    column = 0
    for comp in roots_of_equation:
        if iscomplex(comp):
            if comp.imag > 0:
                column_formula.append(lambda x, b=comp: exp(b.real * x) * sin(b.imag * x))
            else:
                column_formula.append(lambda x, b=comp: exp(b.real * x) * cos(b.imag * x))
        else:
            column_formula.append(lambda x, b=comp: exp(b.real * x))
        column += 1

    while column < num_of_columns:
        index = column - len(roots_of_equation)
        column_formula.append(lambda x, b=index: x**(b))
        column += 1

    Private_Decision = zeros( (length, num_of_columns) )

    i,j = 0,0
    for i in range(length):
        for j in range(num_of_columns):
            # print("x = " + str(X_t[i]))
            # print(" from formula = " + str(column_formula[j](X_t[i])))
            Private_Decision[i][j] = column_formula[j](X_t[i])

    #print("Private_decision = \n" + str(Private_Decision) + "\n")

    Eq_Matrix = Private_Decision.T * matrix(Private_Decision)
    # print("Eq_Matrix = \n" + str(Eq_Matrix) + "\n")
    Eq_Matrix_INV = Eq_Matrix.I
    # print("Eq_Matrix_INV = \n" + str(Eq_Matrix_INV) + "\n")
    # print("Id = \n" + str(Eq_Matrix * Eq_Matrix_INV) + "\n")
    det = linalg.det(Eq_Matrix) * linalg.det(Eq_Matrix_INV)
    #print("DET = " + str(det) + "\n")
    if abs(1 - det) > 0.000001: # 1*10^-06
        num_of_columns -= 1
        if num_of_columns < len(roots_of_equation):
            print("Wrong structure of differential quation!")
            break
    else:
        print("Wright structure! " + str(num_of_columns) + "\n")
        break

Private_Decision_VECTOR = Private_Decision.T * matrix(Y_t)
#print("Private_Decision_VECTOR = \n" + str(Private_Decision_VECTOR) + "\n")

Integrate_Constants = linalg.solve(Eq_Matrix, Private_Decision_VECTOR)
print("Integrate_Constants =\n" + str(Integrate_Constants) + "\n")

Y_model = Private_Decision * Integrate_Constants

squared_Delta = 0
sum_Y = 0
for i in range(length):
    squared_Delta += (Y_t[i] - Y_model[i])**2
    sum_Y += Y_t[i]

math_prediction = sum_Y / length
print("math_prediction = " + str(math_prediction))

cko = sqrt(squared_Delta/length)
print("CKO = " + str(cko))

variation_koef = cko / math_prediction
print("variation_koef = " + str(variation_koef))

plt.plot(X_t,Y_t, color = 'blue', label = u'Y_t')
plt.plot(X_t,Y_model, color = 'red', label = u'Y_model')
plt.grid()
plt.savefig('chart.png')
plt.show()

with open("report.txt", "w") as file:
    file.write("X = \n" + str(X) + "\n")
    file.write("Y = \n" + str(Y) + "\n")
    file.write("Koef = \n" + str(Koef) + "\n")
    file.write("Roots = \n" + str(roots_of_equation) + "\n")
    file.write("Integrate_Constants = \n" + str(Integrate_Constants) + "\n")
    file.write("X_t = \n" + str(X_t) + "\n")
    file.write("Y_t = \n" + str(Y_t) + "\n")
    file.write("Y_model = \n" + str(Y_model) + "\n")
    file.write("math_prediction = " + str(math_prediction) + "\n")
    file.write("CKO = " + str(cko) + "\n")
    file.write("variation_koef = " + str(variation_koef) + "\n")
    file.flush()
    file.close()