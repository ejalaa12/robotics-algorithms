# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a simple PID implementation
"""


class PID:
    def __init__(self, kp, ki, kd, t0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.error = 0
        self.previous_error = 0
        self.integral = 0
        self.time = t0
        self.last_time = t0

    def reset(self, t0):
        self.error = 0
        self.previous_error = 0
        self.integral = 0
        self.time = t0
        self.last_time = t0

    def control(self, x, w, t):
        e = w - x
        self.error, self.previous_error = e, self.error
        self.time, self.last_time = t, self.time
        if self.time == self.last_time:
            return 0
        self.integral += e
        de = (e - self.previous_error) / (self.time - self.last_time)

        return self.kp * self.error + self.ki * self.integral + self.kd * de
