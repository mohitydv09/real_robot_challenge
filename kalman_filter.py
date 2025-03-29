import numpy as np

class KalmanFilter:
    def __init__(self):
        self.x = np.array([[0], 
                           [0], 
                           [0]])  ## State matrix.

        self.P = np.eye(3) * 0.1            ## Covariance Matrix
        self.Q = np.eye(3) * 0.01           ## Process Noise
        self.R = np.eye(3) * 0.5            ## Measurement Noise

    @staticmethod
    def normalize_angle(angle):
        '''Helper function to Normalize the angle between -pi to pi.'''
        return np.arctan2(np.sin(angle), np.cos(angle))

    def predict(self, v, w, dt):
        theta = self.x[2, 0]
        ## Motion Model Matrix
        F = np.eye(3)
        B = dt * np.array([ [np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [0,             1]])
        u = np.array([[v],
                      [w]])

        ## Predict the state.
        self.x = F @ self.x + B @ u
        self.x[2, 0] = self.normalize_angle(self.x[2, 0])

        ## Predict the Covariance Matrix
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z):
        '''Update the state and covariance matrix based on z. Where z is measurment.'''

        ## Measurement Model Matrix
        H = np.eye(3)

        ## Compute the Kalman Gain.
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        ## Update the state.
        innovation = z - H @ self.x
        innovation[2, 0] = self.normalize_angle(innovation[2, 0])

        self.x = self.x + K @ innovation

        self.x[2, 0] = self.normalize_angle(self.x[2, 0])

        ## Update the Covariance Matrix
        self.P = (np.eye(3) - K @ H) @ self.P

if __name__ == '__main__':
    kf = KalmanFilter()
    ## Write code to test the Kalman Filter.
    pass