import numpy as np
from scipy.linalg import inv



'''
Molecular Orbital Complex Absorbing Potential
'''
# gotta change something so that the dimensions match up like the calculate potential function, in UKS the C_AO basis is 3D and is 2D for RKS

# we first define our parent class, in which we define our damping diagonal and we also form the OAO  CAP

class MOCAP:
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		self.expconst = expconst
		self.emin = emin
		self.prefac = prefac
		self.maxval = maxval

	def calculate_cap(self, rt_scf, fock):
	#this here orthogonalizes the fock matrix (rt_scf,orth = X)
		fock_orth = np.dot(rt_scf.orth.T, np.dot(fock, rt_scf.orth))
		mo_energy, mo_orth = np.linalg.eigh(fock_orth)

	#now we work to construct the damping diagonal (D), with the CAP in OAO basis equal to -1j*C'@D@C'.T
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

		C_OAO = self.get_OAO_coeff()

		damping_matrix_OAO = np.dot(C_OAO, np.dot(damping_matrix, C_OAO.T.conj()))
		return 1j * damping_matrix_OAO

	def get_OAO_coeff(self):
		raise NotImplementedError("Must choose a choice of basis for the CAP.")

	def calculate_potential(self, rt_scf):
		if rt_scf.nmat == 1:
			return self.calculate_cap(rt_scf, rt_scf.fock_ao)
		else:
			return np.stack((self.calculate_cap(rt_scf, rt_scf.fock_ao[0]), self.calculate_cap(rt_scf, rt_scf.fock_ao[1])))


	#in the original cap, we give the cap in the AO basis instead of the OAO basis for some reason.

#Create the OAO CAP using the dimer basis.
class DIMER(MOCAP):
	def __init__(self, dimer, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer

	def get_OAO_coeff(self):
		C_AO = self.dimer.mo_coeff
		overlap_DIMER  = self.dimer.get_ovlp()
		eigvals, eigvecs = np.linalg.eigh(overlap_DIMER)
		s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
		X = np.dot(eigvecs, np.dot(s_inv_sqrt, eigvecs.T.conj()))
	#Now we can use the lowdin orthogonalization  or canonical orthogonalization to get the OAO representation, in lowdin the transformation is C'=U@s^-0.5@U.T@C
		if C_AO.ndim == 3: #if we have UKS
			return np.stack([np.dot(X, C_AO[0]), np.dot(X, C_AO[1])])
		return np.dot(X, C_AO)

#Create the OAO CAP using the NOSCF basis.
class NOSCF(MOCAP):
	def __init__(self, dimer, noscf_orbitals, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer
		self.noscf_orbitals = noscf_orbitals

	def get_OAO_coeff(self):
		C_AO = self.noscf_orbitals
		overlap_dimer = self.dimer.get_overlap
		overlap_NOSCF = np.dot(C_AO.T.conj(), np.dot(overlap_dimer, C_AO))
		eigvals, eigvecs = np.linalg.eigh(overlap_NOSCF)
		s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
	#Now we can use the lowdin orthogonalization or canonical orthogonalization to get the OAO representation, in lowdin the transformation is C'=U@s^-0.5@U.T@C
		OAO_coeff = np.dot(eigvecs, np.dot(s_inv_sqrt, np.dot(eigvecs.T, C_AO)))
		return OAO_coeff

#Create the OAO CAP using the basis that diagonalizes the time-dependent fock matrix.
class FORTHO(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
	def get_OAO_coeff(self):
#		fock_orth = np.dot(self.rt_scf.orth.T, np.dot(self.fock, self.rt_scf.orth))
#		_, mo_orth = np.linalg.eigh(fock_orth)
		return mo_orth


