#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:58:12 2020

@author: ejalaa
"""
import numpy as np
from models.model1D import Model
from control.pid import PID
import matplotlib.pyplot as plt


# %% External forces

def external_force(t):
    wind = np.sin(t * 0.07) + 0.5 * np.cos(t * 0.012)

    total = wind
    return total


# %% Consigne

def w(t):
    if t < 150:
        return 0
    elif 150 < t < 200:
        return 10
    else:
        return 20 * np.cos(t * 0.01)


# %% SIMULATION

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m = Model(1, 0, k=0.2)
    pid = PID(10, 1.0, 80, 0)

    dt = 0.1
    TIME = np.arange(0, 400, dt)
    W = []

    for t in TIME:
        fe = external_force(t)
        fpid = pid.control(m.x, w(t), t)
        W.append(w(t))
        f = fe + fpid
        #        print fe, fpid
        m.evolve(f, dt)

    ax1, ax2 = m.plot(TIME)
    ax1.plot(TIME, W)
    plt.show()

    plt.show()
