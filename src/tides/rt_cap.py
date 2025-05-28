import numpy as np
from scipy.linalg import inv

'''
Molecular Orbital Complex Absorbing Potential (CAP)
'''

class MOCAP:
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		self.expconst = expconst
		self.emin = emin
		self.prefac = prefac
		self.maxval = maxval

	def calculate_cap(self, rt_scf, fock, coeff_matrix=None, mo_energy=None):

		fock_orth = np.dot(rt_scf.orth.T, np.dot(fock,rt_scf.orth))
		if mo_energy is None or coeff_matrix is None:
			mo_energy, coeff_matrix = np.linalg.eigh(fock_orth)

		damping_diagonal = []

		for energy in mo_energy:
			energy_corrected = energy - self.emin

			if energy_corrected > 0:
				damping_term = self.prefac * (1 - np.exp(self.expconst* energy_corrected))
				if damping_term < (-1 * self.maxval):
					damping_term = -1 * self.maxval
				damping_diagonal.append(damping_term)
			else:
				damping_diagonal.append(0)

		damping_diagonal = np.array(damping_diagonal).astype(np.complex128)

		damping_matrix = np.diag(damping_diagonal)
		damping_matrix = np.dot(coeff_matrix, np.dot(damping_matrix, np.conj(coeff_matrix.T)))

		transform = inv(rt_scf.orth.T)
		damping_matrix_ao = np.dot(transform, np.dot(damping_matrix, transform.T))
		return 1j * damping_matrix_ao

	def calculate_potential(self, rt_scf, coeff_matrix=None, mo_energy=None):
		if rt_scf.nmat == 1:
			return self.calculate_cap(rt_scf, rt_scf.fock_ao, coeff_matrix, mo_energy)
		else:
			results = []
			for spin in range(rt_scf.nmat):
				coeff_matrix = None if coeff_matrix is None else coeff_matrix[spin]
				mo_energy = None if mo_energy is None else mo_energy[spin]
				results.append(self.calculate_cap(rt_scf, rt_scf.fock_ao[spin], coeff_matrix, mo_energy))
			return np.stack(results)


class DIMER(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100, dimer=None):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer

	def calculate_potential(self, rt_scf, dimer=None):
		dimer = dimer if dimer is not None else self.dimer
		X_inv = inv(rt_scf.orth)
		mo_coeff = self.dimer.mo_coeff
		scf_energy = self.dimer.mo_energy
		if mo_coeff.ndim == 2:
			dimer_coeff = np.dot(X_inv, mo_coeff)
		else:
			dimer_coeff = np.array([np.dot(X_inv, mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		return super().calculate_potential(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)


class NOSCF(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100, dimer=None, noscf_orbitals=None):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer
		self.noscf_orbitals = noscf_orbitals

	def calculate_potential(self, rt_scf):
		X_inv = inv(rt_scf.orth)
		mo_coeff = self.noscf_orbitals
		noscf_energy = self.noscf_energy
		if mo_coeff.ndim == 2:
			dimer_coeff = np.dot(X_inv, mo_coeff)
		else:
			dimer_coeff = np.array([np.dot(X_inv, mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		return super().calculate_potential(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)
