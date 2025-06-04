import numpy as np
from scipy.linalg import inv
from abc import ABC, abstractmethod
from tides.rt_utils import get_scf_orbitals, get_noscf_orbitals

'''
Molecular Orbital Complex Absorbing Potential (CAP)
'''

class MOCAP(ABC):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		self.expconst = expconst
		self.emin = emin
		self.prefac = prefac
		self.maxval = maxval

	def calculate_cap(self, rt_scf, coeff_matrix=None, mo_energy=None):

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

	def calculate_potential_spin(self, rt_scf, coeff_matrix=None, mo_energy=None):
		if rt_scf.nmat == 1:
			return self.calculate_cap(rt_scf, coeff_matrix, mo_energy)
		else:
			results = []
			for spin in range(rt_scf.nmat):
				cm = coeff_matrix[spin]  if coeff_matrix is not None else None
				me = mo_energy[spin] if mo_energy is not None else None
				results.append(self.calculate_cap(rt_scf, cm, me))
			return np.stack(results)

	@abstractmethod
	def calculate_potential(self, rt_scf, coeff_matrix=None, mo_energy=None):
		pass


class FORTHO(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.fock = None
	def calculate_potential(self, rt_scf):
		fock=rt_scf.fock_ao

		if fock.ndim == 2:
			fock_orth = np.dot(rt_scf.orth.T, np.dot(fock,rt_scf.orth))
			mo_energy, fock_eigvecs = np.linalg.eigh(fock_orth)
		else:
			mo_energy = []
			fock_eigvecs = []
			for spin in range(fock.shape[0]):
				fock_orth = np.dot(rt_scf.orth.T, np.dot(fock[spin],rt_scf.orth))
				eigval, eigvec = np.linalg.eigh(fock_orth)
				mo_energy.append(eigval)
				fock_eigvecs.append(eigvec)
		return super().calculate_potential_spin(rt_scf, coeff_matrix=fock_eigvecs, mo_energy=mo_energy)

class DIMER(MOCAP):
	def __init__(self, expconst, emin, dimer, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer

	def calculate_potential(self, rt_scf):
		X_inv = inv(rt_scf.orth)

	#here we are adding Ehrenfest compatibility, if we are ehrenfest then the energies and coefficient update whenever we update.
	#maybe add a modulo statement here so that we only update whenever we update on the ehrenfest timestep, talk to kretchmer about nailing this down.
		if rt_scf.istype('RT_Ehrenfest'):
			get_scf_orbitals(rt_scf)
			mo_coeff = rt_scf.mo_coeff_print
			scf_energy = rt_scf.mo_energy_print
		else:
			mo_coeff = self.dimer.mo_coeff
			scf_energy = self.dimer.mo_energy

		if mo_coeff.ndim == 2:
			dimer_coeff = np.dot(X_inv, mo_coeff)
		else:
			dimer_coeff = np.array([np.dot(X_inv, mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		return super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)


class NOSCF(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.noscf_orbitals = None
		self.noscf_energy = None

	def calculate_potential(self, rt_scf):
		X_inv = inv(rt_scf.orth)

		if rt_scf.istype('RT_Ehrenfest'):
			get_noscf_orbitals(rt_scf)
			mo_coeff = rt_scf.mo_coeff_print
			noscf_energy = rt_scf.mo_energy_print
		else:
			mo_coeff = self.noscf_orbitals
			scf_energy = self.noscf_energy

		if mo_coeff.ndim == 2:
			dimer_coeff = np.dot(X_inv, mo_coeff)
		else:
			dimer_coeff = np.array([np.dot(X_inv, mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		return super().calculate_potential_spin(rt_scf, coeff_matrix=noscf_coeff, mo_energy=noscf_energy)
