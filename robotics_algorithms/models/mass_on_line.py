#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:30:03 2020

@author: ejalaa
"""

import numpy as np

from model import Model


class MassOnLineModel(Model):
    def __init__(self, mass, friction_coeff, x=0, v=0):
        self.mass = mass
        self.friction_coeff = friction_coeff
        self.X = np.array([x, v])
        self.update_history(self.X, 0)

    def step_update(self, u, dt=0.1):
        m, k = self.mass, self.friction_coeff
        A = np.array([[0, 1], [0, -k / m]])
        B = np.array([0, 1 / m])
        self.X = (np.eye(2) + dt * A).dot(self.X) + dt * B * u
        self.update_history(self.X, self.command)


# %% SIMULATION

SIM_TIME = 100
DT = 0.1
N = SIM_TIME / DT
