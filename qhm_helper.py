import numpy as np

def momentum_operator(psi, dx):
    # create an array to hold the derivative
    d_psi = np.zeros(psi.shape, dtype=complex)
    
    # calculate the derivative
    for i in range(1, len(psi)-1):
        d_psi[i] = (psi[i+1] - psi[i-1]) / (2*dx)
    
    # calculate the derivative at the end points
    d_psi[0] = (psi[1] - psi[0]) / dx
    d_psi[-1] = (psi[-1] - psi[-2]) / dx
    
    return -1j * d_psi

def kinetic_energy(psi, dx, m=1e5):
    return (1/(2 * m)) * np.sum(np.conj(psi) * momentum_operator(momentum_operator(psi, dx), dx)) * dx

def potential_energy(psi, x, dx, m=1e5, w=1e-3):
    return (1/2) * m * w**2 * np.sum(x**2 * np.abs(psi)**2) * dx

def energy(psi, x, dx, m=1e5, w=1e-3):
    H_psi = (1 / (2 * m)) * momentum_operator(momentum_operator(psi, dx), dx) + 0.5 * m * w**2 * x ** 2 * psi
    return np.sum(np.conj(psi) * H_psi) * dx 

def d_psi(psi_x, x, dx, m=1e5, w=1e-3):
    # Calculate H_psi
    H_psi = 0.5 * (1/m) * momentum_operator(momentum_operator(psi_x, dx), dx) + 0.5 * m * w**2 * x ** 2 * psi_x
    # The derivative of psi_x is 1j * H_psi
    return -1j * H_psi

def euler_step(psi_x, dx, dt=0.001, m=1e5, w=1e-3):
    psi_x = psi_x + dt * d_psi(psi_x, dx, m, w) 
    # divide by norm
    psi_x = psi_x / np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
    return psi_x

def rk_step(psi_x, x, dx, dt=0.001, m=1e5, w=1e-3):
    k1 = dt * d_psi(psi_x, x, dx, m, w)
    k2 = dt * d_psi(psi_x + 0.5 * k1, x, dx, m, w)
    k3 = dt * d_psi(psi_x + 0.5 * k2, x, dx, m, w)
    k4 = dt * d_psi(psi_x + k3, x, dx, m, w)
    psi_x = psi_x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # divide by norm
    psi_x = psi_x / np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
    return psi_x

def position_expectation(psi_x, x, dx):
    return np.sum(x * np.abs(psi_x)**2) * dx

def position_variance(psi_x, x, dx):
    return np.sum((x - position_expectation(psi_x, x, dx))**2 * np.abs(psi_x)**2) * dx

def pendulum_potential(x, g=9.8, l=1):
    return g * l * (1 - np.cos(x))