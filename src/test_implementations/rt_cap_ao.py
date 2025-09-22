import numpy as np
from scipy.linalg import inv, eigh

'''
Molecular Orbital Complex Absorbing Potential (CAP)
'''


class MOCAP:
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

    def calculate_cap(self, rt_scf, fock):
        # Construct fock_orth without CAP
        # Calculate MO energies
        mo_energy, mo_orth = np.linalg.eigh(fock)#this apparently gives canonical molecular orbitals.

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
        damping_matrix_mo = np.diag(damping_diagonal)
        overlap = rt_scf.ovlp
        damping_matrix_ao = np.dot(overlap, np.dot(mo_orth, np.dot(damping_matrix_mo, np.dot(np.conj(mo_orth.T), overlap))))

        return 1j * damping_matrix_ao

    def calculate_potential(self, rt_scf):
        if rt_scf.nmat == 1:
            return self.calculate_cap(rt_scf, rt_scf.fock_ao)
        else:
            return np.stack((self.calculate_cap(rt_scf, rt_scf.fock_ao[0]), self.calculate_cap(rt_scf, rt_scf.fock_ao[1])))