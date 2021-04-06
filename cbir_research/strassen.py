import numpy as np
import time

# Method used to return each quadrant of the matrix entered as a parameter
def split(m):

    n = len(m)
    half = n // 2
    a = m[:half, :half]
    b = m[:half, half:]
    c = m[half:, :half]
    d = m[half:, half:]
    return a, b, c, d


# Recursive strassen function
def strassen(A, B):

    # Base case for the recursion, if length of the matrix is 1, return the simple multiplication of the two
    if len(A) == 1:
        return A * B

    # Using the split function from above, split each matrices into 4 quadrants
    a11, a12, a21, a22 = split(A)
    b11, b12, b21, b22 = split(B)

    # Recursive call to strassen function with the appropriate parameters as per the original algorithm
    p1 = strassen(a11, b12 - b22)
    p2 = strassen(a11 + a12, b22)
    p3 = strassen(a21 + a22, b11)
    p4 = strassen(a22, b21 - b11)
    p5 = strassen(a11 + a22, b11 + b22)
    p6 = strassen(a12 - a22, b21 + b22)
    p7 = strassen(a11 - a21, b11 + b12)

    # Combine the results as per the original algorithm
    C11 = p5 + p4 - p2 + p6
    C12 = p1 + p2
    C21 = p3 + p4
    C22 = p1 + p5 - p3 - p7

    # Using numpy's vstack to combine the 4 quadrants found together
    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return C

def strassen_time():
    n = [2 ** i for i in range (1,11)]
    strassen_time = []
    for i in n:
        matrix = np.random.randint(i * i, size=(i,i))
        matrix2 = np.random.randint(i * i, size=(i,i))
        start = time.time()
        strassen(matrix, matrix2)
        end = time.time() - start
        strassen_time.append(end)
        print(end)


strassen_time()


'''
# def strassen(A, B):
#     size = len(A)
#
#     if size == 1:
#         C = [[0]]
#         C[0][0] = A[0][0] * B[0][0]
#         return C
#     else:
#         a, b, c, d = split(A)
#         e, f, g, h = split(B)
#
#         p1 = strassen(a, matrix_subtraction(f, h))
#         p2 = strassen(matrix_subtraction(a, b), h)
#         p3 = strassen(matrix_addition(c, d), e)
#         p4 = strassen(d, matrix_subtraction(g, e))
#         p5 = strassen(matrix_addition(a, d), matrix_addition(e, h))
#         p6 = strassen(matrix_subtraction(b, d), matrix_addition(g, h))
#         p7 = strassen(matrix_subtraction(a, c), matrix_addition(e, f))
#         C11 = matrix_addition(matrix_subtraction(matrix_addition(p5, p4), p2), p6)
#         C12 = matrix_addition(p1, p2)
#         C21 = matrix_addition(p3, p4)
#         C22 = matrix_addition(matrix_subtraction(matrix_subtraction(p5, p3), p7), p1)
#         C = [[0 for j in range(size)] for i in range(size)]
#         for i in range(size):
#             for j in range(size):
#                 C[i][j] = C11[i][j]
#                 C[i][j + size] = C12[i][j]
#                 C[i + size][j] = C21[i][j]
#                 C[i + size][j + size] = C22[i][j]
#         return C
#     # n = len(A)
#     # if n < 1:
#     #     return matrix_multiplication(A, B)
#     # else:
#     #     size = n//2
#     #     print(size)
#     #     a11 = [[0 for j in range(size)] for i in range(size)]
#     #     a12 = [[0 for j in range(size)] for i in range(size)]
#     #     a21 = [[0 for j in range(size)] for i in range(size)]
#     #     a22 = [[0 for j in range(size)] for i in range(size)]
#     #     b11 = [[0 for j in range(size)] for i in range(size)]
#     #     b12 = [[0 for j in range(size)] for i in range(size)]
#     #     b21 = [[0 for j in range(size)] for i in range(size)]
#     #     b22 = [[0 for j in range(size)] for i in range(size)]
#     #
#     #     for i in range(size):
#     #         for j in range(size):
#     #             a11[i][j] = A[i][j]
#     #             a12[i][j] = A[i][j + size]
#     #             a21[i][j] = A[i + size][j]
#     #             a22[i][j] = A[i + size][j + size]
#     #             b11[i][j] = B[i][j]
#     #             b12[i][j] = B[i][j + size]
#     #             b21[i][j] = B[i + size][j]
#     #             b22[i][j] = B[i + size][j + size]
#     #
#     #     p1 = strassen(matrix_addition(a11, a22), matrix_addition(b11, b22))
#     #     p2 = strassen(matrix_addition(a21, a22), b11)
#     #     p3 = strassen(a11, matrix_subtraction(b12, b22))
#     #     p4 = strassen(a22, matrix_subtraction(b21, b11))
#     #     p5 = strassen(matrix_addition(a11, a12), b22)
#     #     p6 = strassen(matrix_subtraction(a21, a11), matrix_addition(b11, b12))
#     #     p7 = strassen(matrix_addition(a12, a22), matrix_addition(b21, b22))
#     #
#     #     temp = matrix_addition(p1, p4)
#     #     temp = matrix_addition(temp, p7)
#     #     c11 = matrix_subtraction(temp, p5)
#     #     c12 = matrix_addition(p3, p5)
#     #     c21 = matrix_addition(p2, p4)
#     #     temp = matrix_addition(p1, p3)
#     #     temp = matrix_addition(temp, p6)
#     #     c22 = matrix_subtraction(temp, p2)
#     #
#     #     C = [[0 for j in range(size)] for i in range(size)]
#     #
#     #     for i in range(size):
#     #         for j in range(size):
#     #             C[i][j] = c11[i][j]
#     #             C[i][j + size] = c12[i][j]
#     #             C[i + size][j] = c21[i][j]
#     #             C[i + size][j + size] = c22[i][j]
#     #     return C
# def matrix_addition(A, B):
#     n = len(A)
#     for i in range(n):
#         for j in range(n):
#             A[i][j] += B[i][j]
#     return A
# def matrix_subtraction(A, B):
#     n = len(A)
#     for i in range(n):
#         for j in range(n):
#             A[i][j] -= B[i][j]
#     return A
#
# def matrix_multiplication(A, B):
#     n = len(A)
#     C = [[0 for j in range(n)] for i in range(n)]
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 C[i][j] += A[i][k] * B[k][j]
#     return C
#
# def split(m):
#     a = b = c = d = m
#
#     while len(a) > len(m)/2:
#         a = a[:len(a)//2]
#         b = b[:len(b)//2]
#         c = c[len(c)//2:]
#         d = d[len(d)//2:]
#
#     while len(a[0]) > len(m[0]) // 2:
#         for i in range(len(a[0]) // 2):
#             a[i] = a[i][:len(a[i]) // 2]
#             b[i] = b[i][len(b[i]) // 2:]
#             c[i] = c[i][:len(a[i]) // 2]
#             d[i] = d[i][len(d[i]) // 2:]
#     return a, b, c, d
# list1= [[1,2], [3,4]]
# list2 = [[2, 3], [4, 5]]
#
# print(strassen(list1, list2))
'''