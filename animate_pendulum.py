 # load numpy file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# take in arg from command line
print(f"Arguments count: {len(sys.argv)}")
print(f"Arguments: {str(sys.argv)}")

psi_x_t = np.load("data/psi_phi_t_progress.npy")

# Create a figure for the animation
fig, ax = plt.subplots()
d_phi = 0.02
phi = np.arange(-np.pi, np.pi, d_phi)

# Initialize two line objects (real and imaginary parts)
line1, = ax.plot(phi, np.real(psi_x_t[0]), 'b-')  # blue line for real part
line2, = ax.plot(phi, np.imag(psi_x_t[0]), 'r-')  # red line for imaginary part



def update(i):
    # Update the line data to reflect the new psi_x
    line1.set_ydata(np.real(psi_x_t[i]))
    line2.set_ydata(np.imag(psi_x_t[i]))

    return line1, line2,

steps = 1000000


def update_2(i):
    global psi_x
    line1.set_ydata(np.abs(psi_x_t[i])**2)
    return line1, 

def pendulum_potential_periodic(phi, m, g, l):
    return m * g * l * (1 - np.cos(phi))

m = 100
g = 0.01
l = 1

if sys.argv[1] == 'wavefunction':
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-10, 10)
    ani = FuncAnimation(fig, update, frames=range(int(steps/1000)),interval=int(sys.argv[2]), blit=True)

    # plot the potential
    ax.plot(phi, pendulum_potential_periodic(phi, m, g, l), 'g-')

    plt.show()

else:
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 10)
    ani = FuncAnimation(fig, update_2, frames=range(int(steps/1000)), interval=int(sys.argv[2]), blit=True)

    ax.plot(phi, pendulum_potential_periodic(phi, m, g, l), 'g-')
    plt.show()
     