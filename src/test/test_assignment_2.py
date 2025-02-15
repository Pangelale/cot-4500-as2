import numpy as np
import math 

# Neville's Method
def neville(x, val, interpolation):
    n = len(x)
    neville = [[0] * n for _ in range(n)]
    
    for i in range(n):
        neville[i][0] = val[i]
    
    for i in range(1, n):
        for j in range(1, i + 1):
            term1 = (interpolation - x[i - j]) * neville[i][j - 1]
            term2 = (interpolation - x[i]) * neville[i - 1][j - 1]
            neville[i][j] = (term1 - term2) / (x[i] - x[i - j])
    
    print(f"\n{neville[i][j]}\n")
    
x = [3.6, 3.8, 3.9]
val = [1.675, 1.436, 1.318]
p = 3.7

# Testing Neville's Method
neville(x, val, p)

# Newton's Forward Difference
def newton_forward(x, val):
    n = len(x)
    diffs = [[0] * n for _ in range(n)]
    
    for i in range(n):
        diffs[i][0] = val[i]
        
    for i in range(1, n):
        for j in range(1, i + 1):
            diffs[i][j] = (diffs[i][j - 1] - diffs[i - 1][j - 1]) / (x[i] - x[i - j])
        print(diffs[i][j])
    
    # Print/create table
    #for i in range(n):
    #   for j in range(i + 1):
    #       print(f"{diffs[i][j]}", end="  ")
    #   print()

nfx = [7.2, 7.4, 7.5, 7.6]
val_n = [23.5492, 25.3913, 26.8224, 27.4589]

# Testing Newton's Forward Difference
newton_forward(nfx, val_n)

# Approximate f(7.30)
neville(nfx, val_n, 7.3)

# Divided Difference w/ Hermite Polynomial Approx.
def hermite(x, val, val_prime):
    n = len(x) 
    cols = n + 2
    rows = n * 2
    
    table = np.zeros((rows, cols))
    np.set_printoptions(precision=8, suppress=False, linewidth=200)
    
    # Put known vals into table
    for i in range(n):
        table[2 * i][0] = x[i]
        table[2 * i + 1][0] = x[i]
        table[2 * i][1] = val[i]
        table[2 * i + 1][1] = val[i]
        table[2 * i + 1][2] = val_prime[i]
    
    for i in range(n):
        if 2 * i + 2 < rows:  
            table[2 * i + 2][2] = (table[2 * i + 2][1] - table[2 * i + 1][1]) / (table[2 * i + 2][0] - table[2 * i + 1][0])
    
    for j in range(n, cols):  # Start at column 3
        for i in range(j - 1, rows):  # Iterate through each row
            denominator = table[i][0] - table[i - (j - 1)][0]
            
            calc = (table[i, j - 1] - table[i - 1, j - 1]) / denominator

            if(math.isinf(calc) or math.isnan(calc)):
                table[i][j] = 0

            else:
                table[i, j] = calc
    
    print(table, "\n")

hfx = [3.6, 3.8, 3.9]
val_h = [1.675, 1.436, 1.318]
val_prime = [-1.195, -1.188, -1.182]

# Testing Hermite Polynomial Approx.
hermite(hfx, val_h, val_prime)

# Cubic Spline Interpolation
def cubic_spline(x, fx):
    n = len(x)  
    h = np.diff(x) # Differences in x values

    # Initialize vectors
    l = np.zeros(n)
    m = np.zeros(n)
    z = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    alpha = np.zeros(n)
    
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (fx[i + 1] - fx[i]) - (3 / h[i - 1]) * (fx[i] - fx[i - 1])

    l[0] = 1
    m[0] = 0
    z[0] = 0

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * m[i - 1]
        m[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1
    z[n - 1] = 0
    c[n - 1] = 0

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - m[j] * c[j+1]
        b[j] = (fx[j + 1] - fx[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    A = np.zeros((n, n))
    
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    print(A)     # Matrix A
    print("["," ".join(f"{val:.8f}".rstrip('0') for val in alpha), "]") # Vector B

    print("["," ".join(f"{val:.8f}".rstrip('0') for val in c), "]")
    
# Given dataset
cx = [2, 5, 8, 10]  
fx = [3, 5, 7, 9]

# Testing Cubic Spline
cubic_spline(cx, fx)