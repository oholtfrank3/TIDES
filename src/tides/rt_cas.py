import numpy as np
import os
from pyscf import gto, mcscf, ao2mo
from pyscf.lib import logger
from tides.rt_prop import propagate
from tides import rt_observables
from tides.rt_utils import restart_from_chkfile

class RT_CASSCF:
    #casscf is a pyscf mcscf CASSCF object
    #h1e is 1-electron Hamiltonian in AO basis. Ex: mol.intor('int1e_kin')+mol.intor('int1e_nuc')
    #h2e is 2-electron Hamiltonian in AO basis. Ex: mol.intor('int2e')
    #mo_coeff_canon is the transformation matrix from AO to MO
        
    def __init__(self, casscf, timestep, max_time, h1e, h2e, filename=None, prop=None, frequency=1, mo_coeff_canon=None, chkfile=None, verbose=3):
        
        self.timestep = timestep
        self.frequency = frequency
        self.maxtime = max_time
        self._casscf = casscf
        # Will discuss how to assign occ for get_spin_square
        self._h1e_AO = h1e
        self._h2e_AO = h2e

        self.verbose = verbose
        self._potential = []
        self.fragments = []

        self.labels = [self._casscf.mol._atom[idx][0] for idx, _ in enumerate(self._casscf.mol._atom)]
        if prop is None: prop = 'rk4'
        if mo_coeff_canon in None: mo_coeff_canon = self._casscf.mo_coeff
        self.prop = prop
        self.mo_coeff_canon = mo_coeff_canon

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.den_ao = mcscf.make_rdm1(self._casscf)
        if len(np.shape(self.den_ao)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        # Restart from chkfile, or create a chkfile
        # If restarting from chkfile, self.den_ao will be rewritten
        self.chkfile = chkfile
        if chkfile is not None:
            if os.path.exists(self.chkfile):
                restart_from_chkfile(self)
                self.den_ao = mcscf.make_rdm1(self._casscf)
            else:
                self.current_time = 0
        else:
            self.current_time = 0
        self._t0 = self.current_time

        rt_observables._init_observables(self) # Needs Discussion

    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)
    
    def update_time(self):
        self.current_time += self.timestep

    def get_h1e_mo(self):
        if self._potential: self.apply_potential()
        return np.matmul(self.mo_coeff_canon.conj().T,np.matmul(self._h1e_AO,self.mo_coeff_canon))
    
    def get_h2e_mo(self):
        nao = len(self._h1e_AO)
        toReturn=ao2mo.incore.full(self._h2e_AO,self.mo_coeff_canon,compact=False)
        return toReturn.reshape((nao,nao,nao,nao))
    
    def add_potential(self, *args):
        for v_ext in args:
            self._potential.append(v_ext)
    
    def apply_potential(self):
        for v_ext in self._potential:
            self._h1e_AO += v_ext.calculate_potential(self)

    def kernel(self, mo_coeff_print=None):
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            if np.isclose(self.current_time,self.maxtime):  # So calculation terminates once max_time is reached after restarts
                self._log.note('Done')
            else:
                self._log.note('Propogation Stopped Early')
            if hasattr(self,'fh'):
                self.fh.close()
            if hasattr(self,'_xyz_fh'):
                # This is only important for unfrozen nuclei, printing .xyz files
                # Putting this here anyways for RT_Ehrenfest and other future derived classes
                self._xyz_fh.close()

        return self
