import numpy as np
from scipy.linalg import inv

'''
Molecular Orbital Complex Absorbing Potential (CAP)
'''

class DIMER_CAP:
	def __init__(self, dimer, expconst, emin, prefac=1, maxval=100):
		self.expconst = expconst
		self.emin = emin
		self.prefac = prefac
		self.maxval = maxval
		self.dimer = dimer

	def calculate_cap(self, rt_scf, fock):
		fock_orth = np.dot(rt_scf.orth.T, np.dot(fock,rt_scf.orth))
		mo_energy, mo_orth = np.linalg.eigh(fock_orth)

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
		damping_matrix = np.dot(mo_orth, np.dot(damping_matrix, np.conj(mo_orth.T)))

		transform = inv(rt_scf.orth.T)
		damping_matrix_ao = np.dot(transform, np.dot(damping_matrix, transform.T))
		S_AO = np.dot(damping_matrix_ao.T, damping_matrix_ao)
		dimer_basis = self.dimer.mo_coeff
		damping_matrix_dimer = np.dot(self.dimer.mo_coeff.T, np.dot(S_AO, np.dot(damping_matrix_ao, np.dot(S_AO, self.dimer.mo_coeff))))
		return 1j * damping_matrix_dimer

	def calculate_potential(self, rt_scf):
		if rt_scf.nmat == 1:
			return self.calculate_cap(rt_scf, rt_scf.fock_ao)
		else:
			return np.stack((self.calculate_cap(rt_scf, rt_scf.fock_ao[0]), self.calculate_cap(rt_scf, rt_scf.fock_ao[1])))

class NOSCF_CAP:
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

        #everything can be the same for the CAP, we can just rotate it into the NOSCF basis at the end
    def calculate_cap(self, rt_scf, noscf_orbitals, fock):
        # Construct fock_orth without CAP, this gives us the energies and the MO coefficients
        fock_orth = np.dot(rt_scf.orth.T, np.dot(fock,rt_scf.orth))
        mo_energy, mo_orth = np.linalg.eigh(fock_orth)

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
	eigvals_noscf, eigvecs_noscf = np.linalg.eigh(noscf_orbitals)
	s_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_noscf))
	X_noscf = np.dot(eigvecs_noscf, s_inv_sqrt)
	s_noscf = np.dot(eigvecs_noscf)

#        noscf_orth = np.dot(rt_scf.orth, self.noscf_orbitals)
        damping_matrix = np.diag(damping_diagonal)
	damping_matrix_ao_noscf = np.dot(noscf_orbitals, np.dot(damping_matrix, np.conj(noscf_orbitals.T))
	damping_matrix_oao_noscf = np.dot(s_noscf, np.dot(X_noscf.T, np.dot(damping_matrix_ao_noscf, np.dot(X_noscf, s_noscf)))) 
        return 1j * damping_matrix_oao_noscf

    def calculate_potential(self, rt_scf):
        if rt_scf.nmat == 1:
            return self.calculate_cap(rt_scf, self.noscf_orbitals, rt_scf.fock_ao)
        else:
            return np.stack((self.calculate_cap(rt_scf, self.noscf_orbitals, rt_scf.fock_ao[0]), self.calculate_cap(rt_scf, self.noscf_orbitals, rt_scf.fock_ao[1])))

class MOCAP:
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

    def calculate_cap(self, rt_scf, fock):
        # Construct fock_orth without CAP
        fock_orth = np.dot(rt_scf.orth.T, np.dot(fock,rt_scf.orth))

        # Calculate MO energies
        mo_energy, mo_orth = np.linalg.eigh(fock_orth)

        # Construct damping terms
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

        # Construct damping matrix
        damping_matrix = np.diag(damping_diagonal)
        damping_matrix = np.dot(mo_orth, np.dot(damping_matrix, np.conj(mo_orth.T)))

        # Rotate back to ao basis
        transform = inv(rt_scf.orth.T)
        damping_matrix_ao = np.dot(transform, np.dot(damping_matrix, transform.T))
        return 1j * damping_matrix_ao

    def calculate_potential(self, rt_scf):
        if rt_scf.nmat == 1:
            return self.calculate_cap(rt_scf, rt_scf.fock_ao)
        else:
            return np.stack((self.calculate_cap(rt_scf, rt_scf.fock_ao[0]), self.calculate_cap(rt_scf, rt_scf.fock_ao[1])))
