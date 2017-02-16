import numpy as np
from kalman import kalman

class Line():
    def __init__(self, historical_count = 3, order = 3):
        '''
        Parameters:
        historical_count: Number of epochs to look back in order to aid the lane search (3-5)
        order: Number of curve fit parameters, poly = 3, cubic = 4
        '''
        # KF Parameters
        self.x = np.matrix(np.zeros(order)).T
        self.P = np.matrix(np.eye(order))*0.5 # initial uncertainty
        self.Q = np.matrix(np.eye(order))*0.5 # delta changes

        self.order = order

        # Current curve fit
        self.current_fit = np.array([False])

        # Current fit error based on MSE line fit
        self.current_fit_error = np.array([False])

        self.historical_count = historical_count
        # Array of points to aid search
        self.historical_x = []
        #y values for detected line pixels
        self.historical_y = []

    def add_historical_points(self, x, y):
        '''
        Adds current point cloud into history to be included in the next processing interval
        '''
        self.historical_x.append(x)
        self.historical_y.append(y)
        if(len(self.historical_x) > self.historical_count):
            self.historical_x.pop(0)

        if(len(self.historical_y) > self.historical_count):
            self.historical_y.pop(0)

    def get_as_poly(self):
        return np.poly1d(self.current_fit)

    def curvature(self):
        '''
        Calculates the line curvature in meters. Assumes image size is 720px wide
        '''
        fit_cr = self.get_as_poly()

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        y = np.array(np.linspace(0, 719, num=10))
        x = np.array([fit_cr(x) for x in y])
        y_eval = np.max(y)

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curverad

    def predict(self, line, line_error):
        '''
        Applies KF smoothing to the curve fit parameters. This only produces
        the predicted values for the current epoch. apply_update must be called to save the
        state
        '''
        if(self.x.all() == False):
            self.x = np.asmatrix(line).T

        # This is a non-standard approach, but with the data format (adding historical) to cov mat gets
        # optimistically small
        self.P = np.matrix(np.eye(self.order))*0.5

        # Using fit error only determines how well the solution converged, not how accurate it is
        line_error = np.matrix(np.eye(self.order))*5;

        x,P,r = self.__kalman_c(self.x,self.P,line,line_error,self.Q)
        return x,P,r

    def apply_update(self,x,P,r):
        '''
        Applies the current measurements into the filter. Only required for smoothing
        '''
        self.x = x
        self.P = P
        self.current_fit = (np.asmatrix(x).T).tolist()[0]
        #self.current_fit_error = (np.asmatrix(r).T).tolist()[0]

    def reset(self):
        '''
        In the event that the filter diverges. this will reset the filter
        '''
        self.x = np.matrix(np.zeros(self.order)).T
        self.P = np.matrix(np.eye(self.order))*0.5 # initial uncertainty
        self.Q = np.matrix(np.eye(self.order))*0.5 # delta changes.
        #self.historical_x = []
        #self.historical_y = []


    def __kalman_c(self, x, P, measurement, R,Q):
        """
        Parameters:
        x: initial state of coefficients (c0, c1)
        P: initial uncertainty convariance matrix
        measurement: line fit coefficients
        R: line fit errors
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """
        motion = np.matrix(np.zeros(self.order)).T

        return kalman(x, P, measurement, R, motion, Q,
                      F = np.matrix(np.matrix(np.eye(self.order))),
                      H = np.matrix(np.matrix(np.eye(self.order))))
