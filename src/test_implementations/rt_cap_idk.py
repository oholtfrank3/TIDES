import numpy as np
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
                damping_term = self.prefac * (1 - np.exp(self.expconst * energy_corrected))
                if damping_term < (-1 * self.maxval):
                    damping_term = -1 * self.maxval
                damping_diagonal.append(damping_term)
            else:
                damping_diagonal.append(0)
        damping_diagonal = np.array(damping_diagonal).astype(np.complex128)
        damping_matrix = np.diag(damping_diagonal)
        # Direct MO → AO transformation, no OAO step
        damping_matrix = np.dot(coeff_matrix, np.dot(damping_matrix, np.conj(coeff_matrix.T)))
        return 1j * damping_matrix

    def calculate_potential_spin(self, rt_scf, coeff_matrix=None, mo_energy=None):
        if rt_scf.nmat == 1:
            return self.calculate_cap(rt_scf, coeff_matrix, mo_energy)
        else:
            results = []
            for spin in range(rt_scf.nmat):
                cm = coeff_matrix[spin] if coeff_matrix is not None else None
                me = mo_energy[spin] if mo_energy is not None else None
                results.append(self.calculate_cap(rt_scf, cm, me))
            return np.stack(results)

    @abstractmethod
    def calculate_potential(self, rt_scf, coeff_matrix=None, mo_energy=None):
        pass

# Choose the basis that diagonalizes the Fock matrix
class FORTHO(MOCAP):
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        super().__init__(expconst, emin, prefac, maxval)
        self.fock = None

    def calculate_potential(self, rt_scf):
        fock = rt_scf.fock_ao
        if fock.ndim == 2:
            mo_energy, fock_eigvecs = np.linalg.eigh(fock)
        else:
            mo_energy = []
            fock_eigvecs = []
            for spin in range(fock.shape[0]):
                eigval, eigvec = np.linalg.eigh(fock[spin])
                mo_energy.append(eigval)
                fock_eigvecs.append(eigvec)
        return super().calculate_potential_spin(rt_scf, coeff_matrix=fock_eigvecs, mo_energy=mo_energy)

# Choose the basis of the SCF orbitals
class DIMER(MOCAP):
    def __init__(self, expconst, emin, dimer, prefac=1, maxval=100):
        super().__init__(expconst, emin, prefac, maxval)
        self.dimer = dimer
        self.dimer_coeff = None
        self.dimer_energy = None
        self._initialized = False

    def calculate_potential(self, rt_scf):
        # Ehrenfest compatibility: update on each call if Ehrenfest
        if rt_scf.istype('RT_Ehrenfest'):
            get_scf_orbitals(rt_scf)
            mo_coeff = rt_scf.mo_coeff_print
            scf_energy = rt_scf.mo_energy_print
        else:
            if not self._initialized:
                get_scf_orbitals(rt_scf)
                self.dimer_coeff = rt_scf.mo_coeff_print
                self.dimer_energy = rt_scf.mo_energy_print
                self._initialized = True
            mo_coeff = self.dimer_coeff
            scf_energy = self.dimer_energy
            
        print("SCF energies used in CAP:", scf_energy)  # <-- Add this line

        return super().calculate_potential_spin(rt_scf, coeff_matrix=mo_coeff, mo_energy=scf_energy)

# Choose the basis of the NOSCF orbitals
class NOSCF(MOCAP):
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        super().__init__(expconst, emin, prefac, maxval)
        self.noscf_orbitals = None
        self.noscf_energy = None

    def calculate_potential(self, rt_scf):
        if rt_scf.istype('RT_Ehrenfest'):
            get_noscf_orbitals(rt_scf)
            noscf_coeff = rt_scf.mo_coeff_print
            noscf_energy = rt_scf.mo_energy_print
        else:
            noscf_coeff = self.noscf_orbitals
            noscf_energy = self.noscf_energy
        return super().calculate_potential_spin(rt_scf, coeff_matrix=noscf_coeff, mo_energy=noscf_energy)