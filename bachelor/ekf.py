import numpy as np

class plane_ekf:
    def __init__(self, initial_state, initial_covariance):
        self.state = initial_state.copy()
        self.P = initial_covariance.copy()

#Testumgebung
if __name__ == '__main__':
    initial_state = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0
    ])

    #Standardabweichung
    standard_deviations = np.array([
        0.1,                #x
        0.1,                #y
        np.deg2rad(5.0),    #yaw
        0.05,               #v
        0.05                #omega
    ])

    initial_covariance = np.diag(standard_deviations ** 2)

    ekf = plane_ekf(initial_state, initial_covariance)

    print(ekf.state)
    print(ekf.P)