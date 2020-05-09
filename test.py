#!/usr/bin/python
import sys
sys.path.append('~/runge-kutta')
import runge_kutta as rk
import numpy as np
#Examples come from https://www.public.asu.edu/~hhuang38/example_Runge-Kutta.pdf
def example1():
    target = 1.348
    butcher_tableau = np.array([[0, 0, 0, 0], [0.5, 0.5, 0, 0], [1, -1, 2, 0], [0, 1 / 6, 2 / 3, 1]])
    derivative = lambda u, x: -2 * u + x + 4
    u_0 = 1
    t_0 = 0
    t_n = 0.2
    h = 0.2
    y = rk.main(h, t_0, u_0, t_n, derivative, butcher_tableau)
    print(abs(target - y[-1]))

def example2():
    target = 1.3472
    butcher_tableau = np.array([[0,0,0,0,0], [0.5, 0.5, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [1, 0, 0, 1, 0], [0, 1 / 6, 1 / 3, 1 / 3, 1]])
    derivative = lambda u, x: -2 * u + x + 4
    u_0 = 1
    t_0 = 0
    t_n = 0.2
    h = 0.2
    y = rk.main(h, t_0, u_0, t_n, derivative, butcher_tableau)
    print(abs(target - y[-1]))

def main():
    example1()
    example2()

if __name__ == '__main__':
    main()
