import numpy as np
import math
import sys

def matMulti(A,B):

	matSize = A.shape[0]
	res = np.zeros((matSize, matSize))

	for i in range(matSize):
		for j in range(matSize):
			for k in range(matSize):
				res[i,j] += A[i,k] * B[k,j]

	return res


def vecMatMulti(M, V):

	matSize = M.shape[0]
	res = np.zeros(matSize)

	for i in range(matSize):

		for j in range(matSize):
			res[i] += M[i, j] * V[j]

	return res


def switchRows(A, a, b):

 	matSize = A.shape[0]
	sw = np.identity(matSize)

	sw[a,a] = 0
	sw[a,b] = 1

	sw[b,b] = 0
	sw[b,a] = 1

	return matMulti(sw, A)



def getMatrixAfterPermu(L,U,P,A,step):
	largest = 0
	index = step

	matSize = A.shape[0]

	if step == 0:
		for i in range(matSize):
			if math.fabs(A[i,0]) > math.fabs(largest):
				largest = A[i,0]
				index = i

		if index != step:
			A = switchRows(A, step, index)
			P = switchRows(P, step, index)

	else:
		for i in range(step, matSize):
			temp = 0
			for j in range(step):

				temp += L[i,j]*A[j,step] 

			if math.fabs(A[i, step] - temp) > math.fabs(largest):
				largest = A[i, step] - temp
				index = i

		if index != step:
			A = switchRows(A, step, index)
			L = switchRows(L, step, index)
			P = switchRows(P, step, index)

	return A, L, P


def LUdcomp(A):
	matSize = A.shape[0]
	u_res = np.zeros((matSize, matSize))
	l_res = np.zeros((matSize, matSize))
	p_res = np.identity(matSize)

	for j in range(0,matSize):

		A, l_res, p_res = getMatrixAfterPermu(l_res, u_res, p_res, A, j)

		for i in range(0,j+1):
			temp = 0
			for k in range(0, i):
				temp += l_res[i][k]*u_res[k][j]

			u_res[i,j] = A[i,j] - temp	

		# print("U")	
		# print(u_res)

		for i in range(j, matSize):	
			temp = 0
			for k in range(0,j):
				temp += l_res[i,k]*u_res[k,j]

			l_res[i,j] = (A[i,j] - temp)/u_res[j][j]

		# print("L")
		# print(l_res)

	return u_res, l_res, p_res



def LUbksub(L, U, b):
	matSize = L.shape[0]

	y_res = np.zeros(matSize)

	res = np.zeros(matSize)

	for i in range(matSize):
		temp = 0
		for k in range(i):
			temp += y_res[k]*L[i,k]

		y_res[i] =  b[i] - temp 

	for i in range(matSize):
		temp = 0
		for k in range(i):
			temp += res[matSize - k - 1]*U[matSize - i - 1, matSize - k - 1]

		res[matSize - i - 1] = (y_res[matSize - i - 1] - temp)/U[matSize - i - 1, matSize - i - 1]	

	return res


# A = np.matrix([[2, -7, 6, 5], [4,8,-10,3], [9,-6,-4,2], [5,1,3,3]])

A = np.matrix([[11, 2, -5, 6, 48], [1,0,17,29,-21], [-3, 4, 55, -61, 0], [41, 97, -32, 47, 23], [-6, 9, -4, -8, 50]])


u, l, p = LUdcomp(A)


print("L")
print(l)
print("\n")
print("U")
print(u)
print("\n")
print("P")
print(p)
print("\n")

# print(matMulti(l,u))

b1 = np.array([4, 0, -7, -2, -11])

b2 = np.array([2, 77, -1003, -7, 10])

print("b(1)")
b = vecMatMulti(p, b1)
print(LUbksub(l, u, b))
print("\n")
print("b(2)")
b = vecMatMulti(p, b2)
print(LUbksub(l, u, b))
