 # load numpy file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# take in arg from command line
print(f"Arguments count: {len(sys.argv)}")
print(f"Arguments: {str(sys.argv)}")

psi_x_t = np.load("data/quantum_pendulum/psi_phi_t_progress.npy")
pendulum_potential = np.load("data/quantum_pendulum/potential.npy")
energies = np.load("data/quantum_pendulum/energy_t.npy")

# Create a figure for the animation
fig, ax = plt.subplots()
d_phi = 0.02
phi = np.arange(-np.pi, np.pi, d_phi)

# Initialize two line objects (real and imaginary parts)
line1, = ax.plot(phi, np.real(psi_x_t[0]), 'b-')  # blue line for real part
line2, = ax.plot(phi, np.imag(psi_x_t[0]), 'r-')  # red line for imaginary part

# plot energy level as a line
line3, = ax.plot(phi, energies[0]*np.ones_like(phi), 'k--')



def update(i):
    # Update the line data to reflect the new psi_x
    line1.set_ydata(np.real(psi_x_t[i]))
    line2.set_ydata(np.imag(psi_x_t[i]))
    line3.set_ydata(energies[i]*np.ones_like(phi))

    return line1, line2, line3, 

steps = 1000000


def update_2(i):
    global psi_x
    line1.set_ydata(np.abs(psi_x_t[i])**2)
    line3.set_ydata(energies[i]*np.ones_like(phi))
    return line1, line3, 

if sys.argv[1] == 'wavefunction':
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-3, 3)
    ani = FuncAnimation(fig, update, frames=range(int(steps/1000)),interval=int(sys.argv[2]), blit=True)

    # plot the potential
    ax.plot(phi, pendulum_potential, 'g-')

    plt.show()

else:
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 3)
    ani = FuncAnimation(fig, update_2, frames=range(int(steps/1000)), interval=int(sys.argv[2]), blit=True)

    ax.plot(phi, pendulum_potential, 'g-')
    plt.show()
     
# plot expectation of position over time
d_phi = 0.02
phi = np.arange(-np.pi, np.pi, d_phi)
position_expectations = []
for psi_x in psi_x_t:
    position_expectations.append(np.sum(phi * np.abs(psi_x)**2) * d_phi)
plt.plot(position_expectations, 'r-', label='Position Expectation')
plt.legend()
plt.show()

# plot variance of position over time
position_variances = []
for i, psi_x in enumerate(psi_x_t):
    position_variances.append(np.sum((phi - position_expectations[i])**2 * np.abs(psi_x)**2) * d_phi)

plt.plot(position_variances, 'b-', label='Position Variance')
plt.legend()
plt.show()
# maximum variance - minimum variance (that isn't 0)
print(f"Max variance: {max(position_variances)}")
print(f"Min variance: {min([x for x in position_variances if x != 0])}")
print(max(position_variances) - min([x for x in position_variances if x != 0]))

