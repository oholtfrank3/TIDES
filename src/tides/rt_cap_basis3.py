
import numpy as np
from scipy.linalg import inv
from tides.rt_utils import _sym_orth

class MOCAP:
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

    def calculate_potential(self, rt_scf, fock=None, C_OAO=None):
        if fock is None:
            fock = rt_scf.fock_ao

        if rt_scf.nmat == 1:
            coao = C_OAO if (C_OAO is None or C_OAO.ndim == 2) else C_OAO[0]
            return self._calculate_cap_single(rt_scf, fock, coao)
        else:
            results = []
            for spin in range(2):
                spin_fock = fock[spin]
                if C_OAO is not None:
                    if C_OAO.ndim == 3:
                        spin_coao = C_OAO[spin]
                    elif C_OAO.ndim == 2:
                        spin_coao = C_OAO
                    else:
                        spin_coao = None
                else:
                    spin_coao = None
                results.append(self._calculate_cap_single(rt_scf, spin_fock, spin_coao))
            return np.stack(results)

    def _calculate_cap_single(self, rt_scf, fock, C_OAO=None):
        mo_energy = self.get_mo_ener(fock, rt_scf)
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

        damping_matrix = np.diag(np.array(damping_diagonal, dtype=np.complex128))
        if C_OAO is None:
            C_OAO = self.get_OAO_coeff(fock, rt_scf)
        assert C_OAO.ndim == 2, "C_OAO should be (n, n) for a single spin"

        # revert into the basis upon which we propagate in rt_scf
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
        # Always slice C_AO before returning: select the spin component matching fock shape if needed
        if C_AO.ndim == 3:
            # Try to match the shape (n, n) of fock with the shape of each spin in C_AO
            for i in range(C_AO.shape[0]):
                if C_AO[i].shape == fock.shape:
                    C_AO = C_AO[i]
                    break
            else:
                raise ValueError("Could not match spin component of C_AO to fock shape in get_OAO_coeff.")
        X = rt_scf.orth
        return np.dot(X, C_AO)

    def get_mo_ener(self, fock, rt_scf):
        scf_energies = dimer.mo_energy
        return scf_energies

class NOSCF(MOCAP):
    def __init__(self, dimer, noscf_orbitals, expconst, emin, prefac=1, maxval=100):
        super().__init__(expconst, emin, prefac, maxval)
        self.dimer = dimer
        self.noscf_orbitals = noscf_orbitals

    def get_OAO_coeff(self, fock, rt_scf):
        C_AO = self.dimer.mo_coeff
        # Always slice C_AO before returning: select the spin component matching fock shape if needed
        if C_AO.ndim == 3:
            # Try to match the shape (n, n) of fock with the shape of each spin in C_AO
            for i in range(C_AO.shape[0]):
                if C_AO[i].shape == fock.shape:
                    C_AO = C_AO[i]
                    break
            else:
                raise ValueError("Could not match spin component of C_AO to fock shape in get_OAO_coeff.")
        X = rt_scf.orth
        return np.dot(X, C_AO)

    def get_mo_ener(self, fock, rt_scf):
        noscf_fock = np.dot(self.noscf_orbitals.T, np.dot(fock, self.noscf_orbitals))
        noscf_energies, _ = np.linalg.eigh(noscf_fock)
        return noscf_energies

class FORTHO(MOCAP):
    def __init__(self, expconst, emin, prefac=1, maxval=100):
        super().__init__(expconst, emin, prefac, maxval)

    def get_OAO_coeff(self, fock, rt_scf):
        fock_orth = self.trans_fock(rt_scf, fock)
        _, mo_coeff = np.linalg.eigh(fock_orth)
        return mo_coeff

    def get_mo_ener(self, fock, rt_scf):
        scf_energies = dimer.mo_energy
        return scf_energies



