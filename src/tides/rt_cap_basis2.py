import numpy as np
from scipy.linalg import inv

class MOCAP:
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		self.expconst = expconst
		self.emin = emin
		self.prefac = prefac
		self.maxval = maxval

	def calculate_potential(self, rt_scf, fock=None):
		if fock is None:
			fock=rt_scf.fock_ao

		if rt_scf.nmat == 1:
			return self._calculate_cap_single(rt_scf, fock)
		else:
			cap_alpha = self._calculate_cap_single(rt_scf, fock[0])
			cap_beta = self._calculate_cap_single(rt_scf, fock[1])
			return np.stack([cap_alpha, cap_beta])


	def _calculate_cap_single(self, rt_scf, fock):
		fock_trans = self.trans_fock(rt_scf, fock)
		mo_energy, _ = np.linalg.eigh(fock_trans)

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

		damping_matrix = np.diag(np.array(damping_diagonal, dtype=np.complex128))
		C_OAO = self.get_OAO_coeff(fock, rt_scf)
	#revert into the basis upon which we propogate in rt_scf
		transform = np.linalg.inv(rt_scf.orth.T)
		damping_OAO = np.dot(C_OAO, np.dot(damping_matrix, C_OAO.T.conj()))
		damping_AO = np.dot(transform, np.dot(damping_OAO, transform.T))
		return 1j * damping_AO

	def get_OAO_coeff(self, fock, rt_scf):
		raise NotImplementedError("Must choose choice of basis")


class DIMER(MOCAP):
	def __init__(self, dimer, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer

	def get_OAO_coeff(self, fock, rt_scf):
		C_AO = self.dimer.mo_coeff
		overlap = self.dimer.get_ovlp()
		eigvals, eigvecs = np.linalg.eigh(overlap)
		X = np.dot(eigvecs, np.dot(np.diag(1.0/np.sqrt(eigvals)), eigvecs.T.conj()))

		if C_AO.ndim == 3:
			return np.stack([np.dot(X, C_AO[0]), np.dot(X, C_AO[1])])
		else:
			return np.dot(X, C_AO)

	def trans_fock(self, rt_scf, fock):
		overlap = self.dimer.get_ovlp()
		eigvals, eigvecs = np.linalg.eigh(overlap)
		X = np.dot(eigvecs, np.dot(np.diag(1.0/np.sqrt(eigvals)), eigvecs.T.conj()))

		if fock.ndim == 3:
			return np.stack([np.dot(X.T, np.dot(fock[0], X)), np.dot(X.T, np.dot(fock[1], X))])
		else:
			return np.dot(X.T, np.dot(fock, X))

class NOSCF(MOCAP):
	def __init__(self, dimer, noscf_orbitals, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer
		self.noscf_orbitals = noscf_orbitals

	def get_OAO_coeff(self, fock, rt_scf):
		C_AO = self.noscf_orbitals
		overlap = self.dimer.get_ovlp()
		overlap_NOSCF = np.dot(C_AO.T.conj(), np.dot(overlap, C_AO))
		overlap_NOSCF = 0.5 * (overlap_NOSCF + overlap_NOSCF.T.conj())
		eigvals, eigvecs = np.linalg.eigh(overlap_NOSCF)
		X = np.dot(eigvecs, np.dot(np.diag(1.0/np.sqrt(eigvals)), eigvecs.T.conj()))

		if C_AO.ndim == 3:
			return np.stack([np.dot(X, C_AO[0]), np.dot(X, C_AO[1])])
		else:
			return np.dot(X, C_AO)

	def trans_fock(self, rt_scf, fock):
		overlap = self.dimer.get_ovlp()
		overlap_NOSCF = np.dot(C_AO.T.conj(), np.dot(overlap, C_AO))
		overlap_NOSCF = 0.5 * (overlap_NOSCF + overlap_NOSCF.T.conj())
		eigvals, eigvecs = np.linalg.eigh(overlap_NOSCF)
		X = np.dot(eigvecs, np.dot(np.diag(1.0/np.sqrt(eigvals)), eigvecs.T.conj()))

		if fock.ndim == 3:
			return np.stack([np.dot(X.T, np.dot(fock[0], X)), np.dot(X.T, np.dot(fock[1], X))])
		else:
			return np.dot(X.T, np.dot(fock, X))


class FORTHO(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
	def get_OAO_coeff(self, fock, rt_scf):
		fock_orth = self.trans_fock(rt_scf, fock)
		_, mo_coeff = np.linalg.eigh(fock_orth)
		return mo_coeff
	def trans_fock(self, rt_scf, fock):
		trans_fock = np.dot(rt_scf.orth.T, np.dot(fock, rt_scf.orth))
		return trans_fock
