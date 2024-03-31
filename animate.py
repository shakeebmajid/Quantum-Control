# load numpy file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

psi_x_t = np.load("psi_x_t_1.npy")

# Create a figure for the animation
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 1000)

# Initialize two line objects (real and imaginary parts)
line1, = ax.plot(x, np.real(psi_x_t[0]), 'b-')  # blue line for real part
line2, = ax.plot(x, np.imag(psi_x_t[0]), 'r-')  # red line for imaginary part

ax.set_xlim(-6, 6)
ax.set_ylim(-10, 10)


def update(i):
    # Update the line data to reflect the new psi_x
    line1.set_ydata(np.real(psi_x_t[i]))
    line2.set_ydata(np.imag(psi_x_t[i]))

    return line1, line2,

steps = 1000000
# ani = FuncAnimation(fig, update, frames=range(int(steps/1000)),interval=50, blit=True)

# plt.show()

ax.set_xlim(-5, 5)
ax.set_ylim(0, 60)

def update_2(i):
    global psi_x
    line1.set_ydata(np.abs(psi_x_t[i])**2)
    return line1, 

ani = FuncAnimation(fig, update_2, frames=range(int(steps/1000)), interval=50, blit=True)

plt.show()