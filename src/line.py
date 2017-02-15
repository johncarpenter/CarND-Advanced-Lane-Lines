import numpy as np
import kalman

class Line():
    def __init__(self):
        # KF Parameters
        self.x = np.matrix('0. 0. 0.').T
        self.xp = np.matrix('0. 0. 0.').T
        self.P = np.matrix(np.eye(3))*5 # initial uncertainty
        self.Q = np.matrix(np.eye(3))*0.5 # delta changes


        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        def is_valid(self):
            return x.all()

        def get_as_poly(self):
            return if x np.poly1d(self.x.flatten().tolist()[0]) else None

        def curvature(self):
            fit_cr = get_as_poly()
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension

            y = np.array(np.linspace(0, 719, num=10))
            x = np.array([fit_cr(x) for x in y])
            y_eval = np.max(y)

            fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
            curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

            return curverad


        def update(self, line, line_error):
            x,P,r = kalman_c(self.x,self.P,line,line_error)
            return x,P,r

        def apply_update(self,x,P):
            self.x = x
            self.P = P

        def reset(self):
            self.x = np.matrix('0. 0. 0.').T
            self.xp = np.matrix('0. 0. 0.').T
            self.P = np.matrix(np.eye(3))*5 # initial uncertainty
            self.Q = np.matrix(np.eye(3))*0.5 # delta changes.


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
