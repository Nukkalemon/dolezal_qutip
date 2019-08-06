import matplotlib.pyplot as plt
import numpy as np
from qutip import *


"""
Jaynes-Cumming model which describes matter-light interaction -- described by the following Hamiltonian:

H = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger + a)(\sigma_- + \sigma_+)

To simplify, we use rotating wave approximation:

H_{\rm RWA} = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger\sigma_- + a\sigma_+)

"""

wc = 1.0  * 2 * np.pi  # cavity frequency
wa = 1.0  * 2 * np.pi  # atom frequency
g  = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005       # cavity dissipation rate
gamma = 0.05        # atom dissipation rate
N = 15              # number of cavity fock states
n_th_a = 0.0        # avg number of thermal bath excitation
use_rwa = True

tlist = np.linspace(0,25,101)

"""
Now we provide an intial state and express the Hamiltonian using QuTiP
"""

psi0 = tensor(basis(N,0), basis(2,1)) # an excited atom

"""
Next we define our create/destroy operator and Sigma Z
"""

a = tensor(destroy(N), qeye(2))
sz = tensor(qeye(N), destroy(2))

"""
Now construct the Hamiltonian using right wave approximation
"""

if use_rwa:
	H = wc * a.dag() * a + wa * sz.dag() * sz + g * (a.dag() * sz + a * sz.dag()) 
else:
	H = wc * a.dag() * a + wa * sz.dag() * sz + g * (a.dag() + a) * (sz + sz.dag())

"""
Next we put together our list of collapse operators -- these will include the cavity relaxation, the cavity excitation (only present when T > 0), and the qubit relaxation
"""

c_ops = []

# cavity relaxation
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sz)


"""
Lastly, using the master equation solver, we evolve our state over time
"""

solution = mesolve(H, psi0, tlist, c_ops, [a.dag()*a, sz.dag()*sz])

n_c = solution.expect[0]
n_a = solution.expect[1]


fig, axes = plt.subplots(1, 1, figsize=(10,6))

axes.plot(tlist, n_c, label="Cavity")
axes.plot(tlist, n_a, label="Atom excited state")
axes.legend(loc=0)
axes.set_xlabel('Time')
axes.set_ylabel('Occupation probability')
axes.set_title('Vacuum Rabi oscillations')

plt.show()



