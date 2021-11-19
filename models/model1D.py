#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:42:34 2020

@author: ejalaa
"""

import numpy as np
import matplotlib.pyplot as plt


class Model:
    """
    Simple model that evolve in 1D with position and speed
    It has a mass
    """
    def __init__(self, x, v, m=10, k=3):
        self.X = np.array([x, v]).reshape(-1, 1)
        self.m = m
        self.k = k
        
        # History
        self.hist = None
        
    def evolve(self, f, dt):
        A = np.array([[1, dt],
                      [0, 1 - self.k*dt/self.m]])
        B = np.array([[0], [dt/self.m]])
        
        self.X = A.dot(self.X) + B.dot(f)
        
        self.save_hist()
        
    def save_hist(self):
        if self.hist is None:
            self.hist = self.X
            return
        self.hist = np.hstack((self.hist, self.X))
        
    def plot(self, time=None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        x, v = self.hist
        if time is None:
            ax1.plot(x, label='x')
            ax2.plot(v, label='v')
        else:
            ax1.plot(time, x, label='x')
            ax2.plot(time, v, label='v')
        ax1.legend()
        ax2.legend()
        ax1.set_title('position')
        ax2.set_title('speed')
        return ax1, ax2
        
    @property
    def x(self):
        return self.X.item(0)
    @property
    def v(self):
        return self.X.item(1)
        
