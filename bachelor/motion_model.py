import numpy as np

#Berechnet Drehgeschwindigkeit des Roboters und Geschwindigkeit aus der Radometrie
def wheel_to_twist(omega_left, omega_right, wheel_radius, wheel_separation):

	v = wheel_radius/2*(omega_right+omega_left)
	omega = wheel_radius/wheel_separation*(omega_right-omega_left)

	return v, omega

#Funktion für den Integartionsschritt
def integrate_pose(x, y, yaw, v, omega,dt):

	x_new = x+v*np.cos(yaw)*dt
	y_new = y+v*np.sin(yaw)*dt
	yaw_new = yaw+omega*dt

	return x_new, y_new, yaw_new

def predict_state(state, dt):

	#Initialisieren
	x, y, yaw, v, omega = state
	#Vorhersage aus Model
	x_new, y_new, yaw_new = integrate_pose(
		x, y, yaw, v, omega, dt
		)
	
	return np.array([x_new, y_new, yaw_new, v, omega])

def state_jacobian(state, dt):

	#Initialisieren
	x, y, yaw, v, omega = state
	F = np.eye(5)

	#Einträge der jacobi füllen
	F[0, 2] = -v*np.sin(yaw)*dt
	F[1, 2] = v*np.cos(yaw)*dt
	F[0, 3] = np.cos(yaw)*dt
	F[1, 3] = np.sin(yaw)*dt
	F[2, 4] = dt

	return F

def process_noise_covariance(state, dt, std_acceleration, std_angular_acceleration):
    
    yaw = state[2]
    half_dt_sq = 0.5*dt**2
    
    #Einfluss von Längs und Winkelbeschleunigung auf die Zustände
    G = np.array([
		[half_dt_sq*np.cos(yaw), 0.0],
		[half_dt_sq*np.sin(yaw), 0.0],
		[0.0, half_dt_sq],
		[dt, 0.0],
		[0.0, dt]
	])
    
    #Kovarianzmatrix der unbekannten Beschleunigungen
    W = np.diag([])