import numpy as np


# poly order dep -
def kalman_c(x, P, measurement, R,
              motion = np.matrix('0. 0. 0.').T,
              Q = np.matrix(np.eye(3))):
    """
    Parameters:
    x: initial state of coefficients (c0, c1)
    P: initial uncertainty convariance matrix
    measurement: line fit coefficients
    R: line fit errors
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix(np.matrix(np.eye(3))),
                  H = np.matrix(np.matrix(np.eye(3))))

def predict(x, P, motion = np.matrix('0. 0. 0.').T, Q = np.matrix(np.eye(3)), F = np.matrix(np.matrix(np.eye(3)))):
    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q
    return x,P


def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P, y
