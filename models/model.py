#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:32:10 2020

@author: ejalaa
"""
import numpy as np
from abc import abstractmethod


class Model(object):
    def __init__(self):
        self._state = None
        self._state_history = []
        self._command_history = []

    def update_history(self, state, command):
        self._state_history.append(state.flatten())
        self._command_history.append(command.flatten())

    @property
    def state_history(self):
        return np.array(self._state_history)

    @property
    def command_history(self):
        return np.array(self._command_history)

    @abstractmethod
    def forward_step(self, command, dt):
        pass


    def model(self, t, x, params=None):
        if params is None:
            params = {}
        self._model(t, x, params)

    @abstractmethod
    def _model(self, t, x, params):
        pass
