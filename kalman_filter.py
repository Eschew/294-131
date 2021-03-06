import numpy as np
from config import P_0, Q, R
"""
FULL KALMAN FILTER
Predict:
x = Fx + Bu
P = FPF^T + Q

Update:
observation(z, R)
K = PH^T(HPH^T + R)^-1
x = x + K(z - Hx)
P = P - KHP

OUR KALMAN FILTER: F = I, B = 0, H = I
Predict:
x' = x
P' = P + Q

Update:
observation(z, R)
K = P'(P' + R)^-1
x = x' + K(z - x')
P = P' - KP'
"""

class KalmanFilter:
    def __init__(self, xhat_0):
        self.xhats = [xhat_0]
        self.Ps = [P_0]
        self.Q = Q
        self.R = R

    def update_time(self):
        xhatprime = self.xhats[-1]
        Pprime = self.Ps[-1] + self.Q

        self.xhats.append(xhatprime)
        self.Ps.append(Pprime)
        return xhatprime, Pprime

    def update_measurement(self, z):
        xhatprime, Pprime = self.xhats[-1], self.Ps[-1]

        K = Pprime.dot(np.linalg.inv(Pprime + self.R))
        xhat = xhatprime + K.dot(z - xhatprime)
        P = (1 - K).dot(Pprime)

        self.xhats[-1] = xhat
        self.Ps[-1] = P
        return xhat, P
