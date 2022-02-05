""" 
Author: Tyler Reiser
Date Created: 2/4/22
Date Modified: 2/4/22 @ 15:00 MST

Summary: Implementation of Newton's method to find the root of Chen's Equations.


- The nonlinear equations used are Chen's Equation
- The partial derivatives were calculated by hand and stored in the Jacobian matrix
- Both equations are stored in the vector function as an array.
- x0, x1, x2 are used to store the users input of the initial condition to be called on at the end in the printed result. 
- x is the initial condition used by the algorithm which stores x0, x1, and x2 in an array.
- The two variables in the nonlinear equations are stored in x after each iteration.
- Count stores the number of iterations. 
- The max counter is put in for the case that the initial guess does not result in a solution. 
- Epsilon is defined as machine epsilon. 
- fval is the value of y or f(x). 
- Fnorm is the 2 norm calculated by NumPy

Note: since x,y values are stored in an array called x, x[0]=x, x[1]=y, x[2]=z values
"""


import numpy as np


x0 = input('choose a starting value for x: ')  # choose initial condition, x
x1 = input('choose a starting value for y: ')  # choose initial condition, y
x2 = input('choose a starting value for z: ')  # choose initial condition, z
x = np.array([int(x0), int(x1), int(x2)])


# given parameters for Chaotic behavior
a = 35
b = 3
c = 28


def Vect_Function(x):  # the vector function stores f and g
    return np.array([a*(x[1]-x[0]),
                     (c-a)*x[0] + c*x[1] - x[0]*x[2],
                     -b*x[2]+x[0]*x[1]])


def Jacobian(x):  # partial derivatives calculated/added by hand
    return np.array([[-a, a, 0],
                     [c - a - x[2], c, x[0]],
                     [x[1], x[0], -b]])


def Newton_SystEq(Vect_Function, Jacobian, x, epsilon):

    fval = Vect_Function(x)
    Fnorm = np.linalg.norm(fval, ord=2)  # L2 norm by numpy
    count = 0

    # loop until epsilon or max iterations is reached
    while abs(Fnorm) > epsilon and count < 100:
        term = np.linalg.solve(Jacobian(x), -fval)
        x = x + term
        fval = Vect_Function(x)
        Fnorm = np.linalg.norm(fval, ord=2)
        count += 1  # add one iteration to the counter

        # not necessary to print but fun to look at for small approx.
        print('We get ', fval, 'after ', count, 'iterations')

    if abs(Fnorm) > epsilon:
        count = -1  # go back one iteration once epsilon is passed

    return x, count  # x output is an 1x2 array, count is an int


x, count = Newton_SystEq(Vect_Function, Jacobian, x, epsilon=1.0e-6)


# results
print('\nThe root value is ', x)
print('\nThe number of iterations to reach a solution is ', count,
      ' when initial conditions x and y are ', x0, ' and ', x1)
