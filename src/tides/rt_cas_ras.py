import numpy as np
import os
from scipy.linalg import inv
from pyscf import gto, mcscf, ao2mo
from pyscf.scf import addons
from pyscf.lib import logger
from tides.rt_casprop import propagate
from tides import rt_observables
from tides.rt_utils import restart_from_chkfile
import math

class RT_CAS_RAS:
    '''
    ras: pyscf mcscf CASSCF/CASCI object, solved for its initial state
    timestep: as defined for the given propogation method
    max_time: total time to run the dynamics for
    outputName: name of output file used to check for numerical stability
    corrDenName: name of output file that shows each AO occupation at each time step
    filename: I don't use this, left to keep consistent with rt_scf class
    h1e: 1 e Hamiltonian in AO basis at initial time
    h2e: 2 e Hamiltonian in AO basis at initial time
    prop: Propogation method in rt_integrators to use. Currently only rk4cr works.
    frequency: How many time steps you want between prints to output files
    mo_to_ao: MO to AO transformation matrix (MO coefficient matrix). Sort columns (core,active). Do not include columns for virtual orbitals.
    orth: Orthogonal AO coefficient matrix
    chkfile: I don't use this, left to keep consistent with rt_scf class
    verbose: I don't use this, left to keep consistent with rt_scf class
    ovlp: AO overlap matrix
    vext: Function that takes time in AU as input and outputs time-dependent portion of 1 e Hamiltonian in AO basis. Two inputs, first one is current time, second one is ras object
    '''
    def __init__(self, ras, timestep, max_time, outputName, corrDenName, filename=None, h1e=None, h2e=None, prop=None, frequency=1, mo_to_ao=None, orth=None, chkfile=None, verbose=3, ovlp=None, vext=None):
        
        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time
        self._scf = ras
        if ovlp is None: ovlp = self._scf.mol.intor('int1e_ovlp')
        self.ovlp = ovlp
        self.outName = outputName
        self.corName = corrDenName
        
        self._castype = ras.__class__.__name__
        self.numP = self._scf.ncore + self._scf.ncas
        self.neleca = self._scf.nelecas[0] + self._scf.ncore
        self.nelecb = self._scf.nelecas[1] + self._scf.ncore
        #self.fullNElecA = math.factorial(self.numP)/(math.factorial(self.neleca)*math.factorial(self._scf.ncas-self._scf.nelecas[0]))
        #self.fullNElecA = math.factorial(self.numP)/(math.factorial(self.nelecb)*math.factorial(self._scf.ncas-self._scf.nelecas[1]))

        self.verbose = verbose
        self._potential = []
        self.fragments = []

        self.labels = [self._scf.mol._atom[idx][0] for idx, _ in enumerate(self._scf.mol._atom)]
        if h1e is None: h1e=self._scf.get_hcore()
        if h2e is None: h2e=self._scf.mol.intor('int2e')
        if prop is None: prop = 'rk4cr'
        if mo_to_ao is None:
            mo_to_ao = self._scf.mo_coeff[:,:self.numP]
            #print(mo_to_ao)
        if orth is None: orth = addons.canonical_orth_(self.ovlp)
        if vext is not None: self._potential.append(vext)
        self._h1e_AO_0 = h1e
        self._h2e_AO_0 = h2e
        self._h1e_AO = h1e
        self._h2e_AO = h2e
        self.no = len(self._h1e_AO)
        self.prop = prop
        self.mo_to_ao = mo_to_ao
        self.ao_to_mo = self.get_ao_to_mo()
        self.orth = orth

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.den_ao = self._scf.make_rdm1()
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
                self.den_ao = mcscf.make_rdm1(self._scf)
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

    def get_ao_to_mo(self):
        return np.matmul(inv(np.matmul(self.mo_to_ao.T,self.mo_to_ao)),self.mo_to_ao.T)
    
    def get_mo_to_ao(self):
        return np.matmul(self.ao_to_mo.T,inv(np.matmul(self.ao_to_mo,self.ao_to_mo.T)))

    def get_h1e_mo(self):
        rawOut = np.matmul(self.mo_to_ao.conj().T,np.matmul(self._h1e_AO,self.mo_to_ao)).astype(np.complex128)
        #return np.matmul(self.mo_to_ao.conj().T,np.matmul(self._h1e_AO,self.mo_to_ao)).astype(np.complex128)
        #return (rawOut+rawOut.conj().T)/2
        return rawOut

    def get_h1e_orth(self):
        #if self._potential: self.apply_potential(self.current_time)
        #print(self._h1e_AO)
        #print(self.orth)
        return np.matmul(self.orth.conj().T,np.matmul(self._h1e_AO,self.orth)).astype(np.complex128)
    
    #def get_h2e_mo(self):
    #    toReturn=ao2mo.incore.full(self._h2e_AO,self.mo_coeff_canon,compact=False)
    #    return toReturn.reshape((self.numP,self.numP,self.numP,self.numP)).transpose(0,2,1,3)

    def get_h2e_mo(self):
        mat1 = np.einsum('ap,pqrs,qb',self.mo_to_ao.conj().T,self._h2e_AO,self.mo_to_ao)
        return np.einsum('cr,abrs,sd',self.mo_to_ao.conj().T,mat1,self.mo_to_ao).astype(np.complex128)
        '''
        if self.realAOs == False:
            mat1 = np.einsum('pqrs,pa',self._h2e_AO,self.mo_coeff_canon)
            mat2 = np.einsum('aqrs,qb',mat1,self.mo_coeff_canon)
            mat3 = np.einsum('abrs,rc',mat2,self.mo_coeff_canon)
            return np.einsum('abcs,sd',mat3,self.mo_coeff_canon)
        else:
            toReturn=ao2mo.incore.full(self._h2e_AO,self.mo_coeff_canon,compact=False)
            return toReturn.reshape((self.numP,self.numP,self.numP,self.numP))
        '''
    
    def get_h2e_orth(self):
        mat1 = np.einsum('ap,pqrs,qb',self.orth.conj().T,self._h2e_AO,self.orth)
        return np.einsum('cr,abrs,sd',self.orth.conj().T,mat1,self.orth).astype(np.complex128)
        '''
        if self.realAOs == False:
            mat1 = np.einsum('pqrs,pa',self._h2e_AO,self.orth)
            mat2 = np.einsum('aqrs,qb',mat1,self.orth)
            mat3 = np.einsum('abrs,rc',mat2,self.orth)
            return np.einsum('abcs,sd',mat3,self.orth)
        else:
            toReturn=ao2mo.incore.full(self._h2e_AO,self.orth,compact=False)
            return toReturn.reshape((self.no,self.no,self.no,self.no))
        '''
        
    def get_den_mo(self,ci=None):
        if ci is None: ci = self._scf.ci
        return self._scf.fcisolver.make_rdm12(ci,self._scf.ncas,self._scf.nelecas)
    '''
    def get_mo_to_orth(self):
        return np.matmul(inv(self.orth),self.mo_to_ao)
    
    def get_orth_to_mo(self):
        return np.matmul(self.ao_to_mo,self.orth)
    '''
    def rotate_mo_to_ao(self,coeff_mo): # I don't use this, left to keep consistent with rt_scf class
        return np.matmul(self.mo_coeff_canon,coeff_mo)
    
    def rotate_ao_to_orth(self, coeff_ao): # I don't use this, left to keep consistent with rt_scf class
        return np.matmul(inv(self.orth), coeff_ao)
    
    def mo_unitvec_to_orth(self,mo):
        return self.mo_to_orth[:][mo]
    
    def add_potential(self, *args): # I don't use this, left to keep consistent with rt_scf class
        for v_ext in args:
            self._potential.append(v_ext)
    
    def apply_potential(self,t,h1,h2): # Modify 1 e Hamiltonian to reflect the given time t
        for vext in self._potential:
            td1e, td2e = vext(t,h1,h2)
            self._h1e_AO = self._h1e_AO_0 + td1e
            self._h2e_AO = self._h2e_AO_0 + td2e

    def get_q_orth(self): # For get_x
        toReturn = np.eye(self.no)
        for p in range(self.numP):
            #p_orth = self.mo_to_orth @ self.getMoVec(p)
            p_orth = self.mo_to_orth[:][p]
            toReturn = toReturn - np.matmul(p_orth,p_orth.conj().T)
        return toReturn
    
    def get_w_orth(self,a,b): # For get_x
        toReturn = np.zeros((self.numP,self.numP))
        for i in range(self.numP):
            for j in range(self.numP):
                toReturn[i][j] = self._h2e_mo[i][j][a][b]
        return np.matmul(self.orth_to_mo.conj().T,np.matmul(toReturn,self.orth_to_mo))
    
    def getQU(self,qn,un,dBarInv): # For get_x
        for k in range(self._scf.ncas):
            kMO = k+self._scf.ncore
            prefac = dBarInv[qn,kMO]
            toAdd = 0
            for v in range(self._scf.ncore):
                toAdd = toAdd + 2*((2*self._h2e_mo[v][v][kMO][un])-self._h2e_mo[v][un][kMO][v])
                for l in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    toAdd = toAdd - (self.casrdm1[k][l]*((2*self._h2e_mo[v][v][lMO][un])-self._h2e_mo[lMO][v][v][un]))
            for l in range(self._scf.ncas):
                for m in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    mMO = m+self._scf.ncore
                    toAdd = toAdd + (self.casrdm1[m][l]*((2*self._h2e_mo[lMO][mMO][kMO][un])-self._h2e_mo[lMO][un][kMO][mMO]))
                    for j in range(self._scf.ncas):
                        jMO = j+self._scf.ncore
                        toAdd = toAdd - (self._h2e_mo[jMO][lMO][mMO][un]*self.casrdm2[j][l][m][k])
            toReturn = toReturn + (prefac*toAdd)
        return toReturn

    def get_x(self): # Get X-matrix for cas/ras-SCF. Acts as time evolution of MO coefficient matrix.
        toReturn = np.zeros((self.numP,self.numP))
        if self._castype == 'CASCI':
            return toReturn
        else:
            qMat = self.get_q_orth()
            dInv = inv(self.casrdm1)
            dBar = 2*np.eye(self._scf.ncas)
            for k in range(self._scf.ncas):
                for l in range(self._scf.ncas):
                    dBar[k][l] = dBar[k][l]-self.casrdm1[k][l]
            dBarInv = inv(dBar)
            for u in range(self._scf.ncore):
                uOrth = self.mo_unitvec_to_orth(u)
                virt = np.matmul(self._h1e_orth,uOrth)
                for v in range(self._scf.ncore):
                    virt = virt + (2*np.matmul(self.get_w_orth(v,v),uOrth)) - (np.matmul(self.get_w_orth(v,u),self.mo_unitvec_to_orth(v)))
                for l in range(self._scf.ncas):
                    for k in range(self._scf.ncas):
                        lMO = l+self._scf.ncore
                        kMO = k+self._scf.ncore
                        virt = virt + (self.casrdm1[k,l]*(np.matmul(self.get_w_orth(lMO,kMO),uOrth)-(np.matmul(self.get_w_orth(lMO,u),self.mo_unitvec_to_orth(kMO))/2)))
                aoCol = np.matmul(qMat,virt)
                for q in range(self._scf.ncas):
                    qMO = q+self._scf.ncore
                    aoCol = aoCol + (self.getQU(qMO,u,dBarInv)*self.mo_unitvec_to_orth(qMO))
                moCol = np.matmul(self.orth_to_mo,aoCol)
                for index in range(self.numP):
                    toReturn[index,u] = moCol[index,0]
            for q in range(self._scf.ncas):
                qMO = q + self._scf.ncore
                qOrth = self.mo_unitvec_to_orth(qMO)
                virt = np.matmul(self._h1e_orth,qOrth)
                for j in range(self._scf.ncas):
                    jMO = j + self._scf.ncore
                    for k in range(self._scf.ncas):
                        kMO = k+self._scf.ncore
                        for l in range(self._scf.ncas):
                            for m in range(self._scf.ncas):
                                mMO = m + self._scf.ncore
                                virt = virt + (self.casrdm2[l][m][j][k]*dInv[l][q]*np.matmul(self.get_w_orth(jMO,kMO),self.mo_unitvec_to_orth(mMO)))
                for u in range(self._scf.ncore):
                    virt = virt + (2*np.matmul(self.get_w_orth(u,u),qOrth)) - np.matmul(self.get_w_orth(u,qMO),self.mo_unitvec_to_orth(u))
                aoCol = np.matmul(qMat,virt)
                for u in range(self._scf.ncore):
                    aoCol = aoCol + (self.getQU(qMO,u,dBarInv).conjugate()*self.mo_unitvec_to_orth(u))
                moCol = np.matmul(self.orth_to_mo,aoCol)
                for index in range(self.numP):
                    toReturn[index,qMO] = moCol[index,0]
            return toReturn
        
    def get_embH(self,x):
        #Returns active space 1e and 2e Hamiltonians
        h1 = self._h1e_mo - x
        Econst = 0
        for e in range(self._scf.ncore):
            Econst = Econst + (2*h1[e,e])
            for f in range(self._scf.ncore):
                Econst = Econst + (2*self._h2e_mo[e][e][f][f]) - self._h2e_mo[e][f][f][e]
        h1Mat = np.zeros((self._scf.ncas,self. _scf.ncas),dtype=np.complex128)
        for a in range(self._scf.ncas):
            for b in range(self._scf.ncas):
                aMO = a+self._scf.ncore
                bMO = b+self._scf.ncore
                h1Mat[a,b] = h1[aMO,bMO]
                for e in range(self._scf.ncore):
                    h1Mat[a,b] = h1Mat[a,b] + (2*self._h2e_mo[bMO][aMO][e][e]) - self._h2e_mo[bMO][e][e][aMO]
        h2Mat = np.zeros((self._scf.ncas,self. _scf.ncas,self._scf.ncas,self. _scf.ncas),dtype=np.complex128)
        for a in range(self._scf.ncas):
            aMO = a+self._scf.ncore
            for b in range(self._scf.ncas):
                bMO = b+self._scf.ncore
                for c in range(self._scf.ncas):
                    cMO = c+self._scf.ncore
                    for d in range(self._scf.ncas):
                        dMO = d+self._scf.ncore
                        h2Mat[a][b][c][d] = self._h2e_mo[aMO][bMO][cMO][dMO]
        #h1Mat = (h1Mat + h1Mat.conj().T)/2
        #h2Mat = (h2Mat + h2Mat.conj().T)/2
        return Econst,h1Mat,h2Mat

    def kernel(self, mo_coeff_print=None): # Begin time propogation. I don't use mo_coeff_print, left to keep consistent with rt_scf class
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            if np.isclose(self.current_time,self.max_time):  # So calculation terminates once max_time is reached after restarts
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
