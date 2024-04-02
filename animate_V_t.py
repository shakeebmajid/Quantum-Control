# load numpy file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# take in arg from command line
print(f"Arguments count: {len(sys.argv)}")
print(f"Arguments: {str(sys.argv)}")

psi_x_t = np.load("data/qhm_t/psi_x_t_progress.npy")
V_x_t = np.load("data/qhm_t/V_x_t_progress.npy")

# Create a figure for the animation
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)

# Initialize two line objects (real and imaginary parts)
line1, = ax.plot(x, np.real(psi_x_t[0]), 'b-')  # blue line for real part
line2, = ax.plot(x, np.imag(psi_x_t[0]), 'r-')  # red line for imaginary part
line3, = ax.plot(x, V_x_t[0], 'g-')  # green line for potential

def update(i):
    # Update the line data to reflect the new psi_x
    line1.set_ydata(np.real(psi_x_t[i]))
    line2.set_ydata(np.imag(psi_x_t[i]))
    line3.set_ydata(V_x_t[i])

    return line1, line2, line3, 

steps = 1000000


def update_2(i):
    global psi_x
    line1.set_ydata(np.abs(psi_x_t[i])**2)
    line3.set_ydata(V_x_t[i])
    return line1, line3,

if sys.argv[1] == 'wavefunction':
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 3)
    ani = FuncAnimation(fig, update, frames=range(int(steps/1000)),interval=int(sys.argv[2]), blit=True)

    plt.show()

else:
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ani = FuncAnimation(fig, update_2, frames=range(int(steps/1000)), interval=int(sys.argv[2]), blit=True)
    plt.show()

# plot expectation of position over time
x = np.linspace(-10, 10, 1000)
dx = 0.02
position_expectations = []
for psi_x in psi_x_t:
    position_expectations.append(np.sum(x * np.abs(psi_x)**2) * dx)
plt.plot(position_expectations, 'r-', label='Position Expectation')
plt.show()
 