import math


#Berechnet Drehgeschwindigkeit des Roboters und Geschwindigkeit aus der Radometrie
def wheel_to_twist(omega_left, omega_right, wheel_radius, wheel_separation):

	v = wheel_radius/2*(omega_right+omega_left)
	omega = wheel_radius/wheel_separation*(omega_right-omega_left)
	return v, omega

#Funktion für den Integartionsschritt
def integrate_pose(x, y, yaw, v, omega,dt):

	x_new = x+v*math.cos(yaw)*dt
	y_new = y+v*math.sin(yaw)*dt
	yaw_new = yaw+omega*dt
	return x_new, y_new, yaw_new
