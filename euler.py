#%%
import numpy as np   
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# write the momentum operator
def momentum_operator(psi, dx):
    # create an array to hold the derivative
    d_psi = np.zeros(psi.shape, dtype=complex)
    
    # calculate the derivative
    for i in range(1, len(psi)-1):
        d_psi[i] = (psi[i+1] - psi[i-1]) / (2*dx)
    
    # calculate the derivative at the end points
    d_psi[0] = (psi[1] - psi[0]) / dx
    d_psi[-1] = (psi[-1] - psi[-2]) / dx
    
    return d_psi

dx = 1 / 1000
x_min = -10
x_max = 10
x = np.linspace(x_min, x_max, int(1 / dx))

psi_x = np.exp(0.5 * -((x-0.1)**2))
psi_x = psi_x / np.sqrt(np.sum(np.abs(psi_x)**2) * dx)

dt = 0.001
steps = 1000000
E_list = np.zeros(steps, dtype=complex)
KE_list = np.zeros(steps, dtype=complex)
PE_list = np.zeros(steps, dtype=complex)
x_list = np.zeros(steps, dtype=complex)
var_list = np.zeros(steps, dtype=complex)
KE_constant = 0.00001
PE_constant = 0.1
def d_psi(t, psi_x):
    # Calculate H_psi
    H_psi = -0.5 * KE_constant * momentum_operator(momentum_operator(psi_x, dx), dx) + 0.5 * PE_constant * x ** 2 * psi_x
    # The derivative of psi_x is 1j * H_psi
    return 1j * H_psi

for i in range(steps):
    # H_psi = -0.5 * KE_constant * momentum_operator(momentum_operator(psi_x, dx), dx) + 0.5 * PE_constant * x ** 2 * psi_x
    # psi_x = psi_x + 1j * dt * H_psi
    # divide by norm
    # psi_x = psi_x / np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
    t0 = 0
    psi_x = solve_ivp(d_psi, [t0, dt], psi_x, t_eval=[dt], method='RK45').y.flatten()
    # print(f"shape of psi_x: {psi_x.shape}")
    # print(psi_x)
    if i % int(steps/100) == 0:
        print(f"H_psi: {H_psi[:5]}")
        print(f"psi_x: {psi_x[:5]}")
        plt.plot(x, np.real(psi_x))
        plt.plot(x, np.imag(psi_x))
        # plot potential
        plt.show()

        plt.plot(x, np.abs(psi_x) ** 2)
        plt.plot(x, 0.5 * PE_constant * x ** 2)
        plt.show()
        # print norm of psi
        print(np.sum(np.abs(psi_x)**2) * dx)
        # print expectation value of x
        print(f"expectation of x: {np.sum(x * np.abs(psi_x)**2) * dx}")
        # print Energy of psi
        print(f"Energy of psi: {np.sum(np.conj(psi_x) * H_psi)}")
    
    E_list[i] = np.sum(np.conj(psi_x) * H_psi)
    # calculate kinetic energy
    KE = np.sum(np.conj(psi_x) * 0.5 * KE_constant * -momentum_operator(momentum_operator(psi_x, dx), dx))
    PE = np.sum(np.conj(psi_x) * 0.5 * PE_constant * x ** 2 * psi_x)
    KE_list[i] = KE
    PE_list[i] = PE
    x_list[i] = np.sum(x * np.abs(psi_x)**2) * dx
    var_list[i] = np.sum((x - x_list[i])**2 * np.abs(psi_x)**2) * dx

    # break if energy goes higher than 100
    if np.abs(E_list[i]) > 50:
        break
    
    
# %%
plt.plot(E_list, label="E")
plt.plot(KE_list + PE_list , label="E_sum")
plt.legend()
plt.show()
plt.plot(KE_list)
plt.show()
plt.plot(PE_list)
plt.show()
plt.plot(x_list)
plt.show()
plt.plot(var_list)
plt.show()
# print first few values of E_list
print(E_list[:5])
print(KE_list[:5])
print(PE_list[:5])
# %%
