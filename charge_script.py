import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from charge_funcs import hamiltonian, visualize_dynamics, plot_energies

def energies(Ec, Ej, N):
	
	ng_vec = np.linspace(-4, 4, 200)

	energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
	
	return(plot_energies(ng_vec, energies, 'Transmon Regime', ymax = [75,3]))



"""
Charge qubit regime (Ej ~ Ec)
"""

#qb_charge = energies(1, 1, 10)

"""
Intermediate regime (Ej > Ec)
"""

#qb_intermediate = energies(1, 5, 10)

"""
Transmon regime (Ej = 50Ec)
"""

#qb_transmon = energies(1, 50, 10)

H = hamiltonian(1,1,10,0.5)

evals, ekets = H.eigenstates() # stores eigenvalues (evals) and eigenvectors (ekets)

#print(evals) 

'''
output = [  0.47065435   1.46676684   9.01371984   9.01760693  25.00520901
  25.00520943  49.00260427  49.00260427  81.00156252  81.00156252
 121.00104167 121.00104167 169.00074405 169.00074405 225.00055804
 225.00055804 289.00043403 289.00043411 361.00034728 361.00347214
 441.00312494]
'''
#print(evals[1]-evals[0]) # output = 0.9961124875822172

#print(abs(ekets[0].full() > 0.1))

'''
Of the states in my first eigenstate, which carry the most weight?
The output of this tells us that only two states meet this inequality
'''

#print(abs(ekets[1].full()) > 0.1)

'''
Similar to the above expression except now I am checking my second 
eigenstate. Once again, only two states hold any weight.
'''

psi_g = ekets[0] # ground state
psi_e = ekets[1] # first excited state

sx = psi_g * psi_e.dag() + psi_e * psi_g.dag() # sigma_x operator
sz = psi_g * psi_g.dag() - psi_e * psi_e.dag() # sigma_z operator

H0 = 0.5 * (evals[1]-evals[0]) * sz # Qubit Hamiltonian
A = 0.25 # driving amplitude
Hd = 0.5 * A * sx # a driving Hamiltonian which comes from driving ng(t)

Heff = [H0, [Hd, 'sin(wd*t)']] # an external field
args = {'wd': (evals[1]-evals[0])}

tlist = np.linspace(0.0, 100.0, 500) # time values

result = mesolve(Heff, psi_g, tlist, [], [ket2dm(psi_e)], args=args)
# evolving our system using the master equation solver

visualize_dynamics(result, r'$\rho_{ee}$')
plt.show()




