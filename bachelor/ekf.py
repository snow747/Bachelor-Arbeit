import numpy as np
from motion_model import predict_state, state_jacobian

class plane_ekf:
    def __init__(self, initial_state, initial_covariance):
        self.state = initial_state.copy()
        self.P = initial_covariance.copy()

    def predict(self, dt, Q):
            
        #Berechnung aktueller Jacobi
        F = state_jacobian(self.state, dt)
        #Vorhersage des Zustands
        self.state = predict_state(self.state, dt)
        #Unsicherheitsvorhersage
        self.P = F @ self.P @ F.T + Q
        
    def update_wheel(self, v_measured, omega_measured, R):
        
        #Messung
        z = np.array([v_measured, omega_measured])

        #Messmatrix
        H = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        #Diff Messung und Vorhersage
        diff = z - H @ self.state
        
        #Unsicherheit Diff
        S = H @ self.P @ H.T + R
        
        #Gewchtung
        PHt = self.P @ H.T
        K = np.linalg.solve(S, PHt.T).T
        
        #Korrektur
        self.state = self.state + K @ diff
        
        #Korrektur Kovarianz
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
    
    def update_imu(self, omega_measured, R):
        
        #Messung
        z = np.array([omega_measured])
        
        #Messmatrix
        H = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        #Diff Messung und Vorhersage
        diff = z - H @ self.state
        
        #Unsicherheit Diff
        S = H @ self.P @ H.T + R
        
        #Gewichtung
        PHt = self.P @ H.T
        K = np.linalg.solve(S, PHt.T).T
        
        #Korrektur
        self.state = self.state + K @ diff
        
        #Korrektur Kovarianz
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

#Testumgebung
if __name__ == '__main__':
    initial_state = np.array([
        0.0, 0.0, 0.0, 1.0, 0.0
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

    dt = 0.1        #Sekunden
    Q = np.zeros((5, 5))  #Für testzwecke
    
    ekf.predict(dt, Q)
    print("Vor Messung:", ekf.state)
    
    measurement_std = np.array([
        0.1,
        0.1
    ])
    
    R = np.diag(measurement_std ** 2)
    
    ekf.update_wheel(
        v_measured = 0.8,
        omega_measured = 0.1,
        R = R
    )
    
    print("Nach Messung:", ekf.state)
    print("Kovarianz:\n", ekf.P)
    
    R_imu = np.array([[0.02 ** 2]])
    
    ekf.update_imu(
        omega_measured = 0.05,
        R = R_imu
    )
    
    print("Nach IMU Messung:", ekf.state)
    print("Kovarianz nach IMU Messung:\n", ekf.P)