#!usr/bin/python
#A function that uses the Runga-Ketta method to approximate the solution for an IVP.
#The program takes as input y(t_0), t_0, target_t, y'(y, t), h, and butcher_table.
#y(t_0) is the function y at the point t_0.
#t_0 is the starting value for t.
#target_t is the value for t that we want to approximate y at.
#y'(y, t) is the function that gives the derivative of y.
#butcher_table is a 2D array used to define which runga-ketta method to use.
#EXAMPLE
#
#the butcher table for RK4 is 
# 
# [[0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [1, 0, 0, 1, 0], [x, 1 / 6, 1 / 3, 1 / 3, 1]]
#
#The bottom-left element will not be read, so what you put in there doesn't matter.

import numpy as np
import sys

def estimate(h, t_n, y_n, derivative, butcher_table):
    #Estimates the value of the function at (t_n, y_n) using the butcher table
    #We will iterate this function in the main function
    #
    #ALGORITHM
    #
    #1. k_0 = derivative(t_n, y_n)
    #2. k_n = derivative(t_n + butcher_table[n][0] * h, y_n + k_(n-1) * h * butcher_table[n][n-1] + ... + k_1 * h * butcher_table[n][1])
    #3. Where k is a vector containing all of our k values in order, return  np.dot(butcher_table[max_n][:1], k.T)

    rows = butcher_table.shape[0]
    k_0 = derivative(t_n, y_n)
    k = np.array([k_0])
    for i in range(1, rows):
        y_term = 0
        for j in range(1, i):
            y_term += k[j - 1] * butcher_table[i][j]

        k.append(k, derivative(t_n + h * butcher_table[i][0], y_n + h * y_term))

    R = np.dot(butcher_table[rows][:1], k.T)
    return R

def main(h, t_0, y_0, t_n, derivative, butcher_table):
    #h is the step-size
    #t_0 is the initial t-value
    #y_0 is the value of the function at t_0
    #t_n is the value we want to evaluate the function at
    #derivative is the a function that models the derivative of the function we wish to evaluate
    #butcher_table is a 2D numpy array
    y_array = np.array([y_0])
    t = t_0
    y = y_0
    while t < t_n:
        y += estimate(h, t, y, derivative, butcher_table)
        t += h
        y_array = np.append(y_array, y)

    return y_array

if __name__  = '__main__':
    #The first 4 inputs should be numbers
    #The fifth input should be a python script
    #The sixth imput should be a npy file containing a 2D array
    butcher_table = load(sys.argv[6])
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], butcher_table)

