import numpy as np
from scipy.linalg import expm, ishermitian
from pyscf.fci.cistring import make_strings
from pyscf import mcscf
from tides import applyham_pyscf as applyham_pyscf
from tides import fci_mod as fci_mod
import sys

'''
Real-time Integrator Functions
'''

def magnus_step(rt_scf):
    '''
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')
    '''

    fock_orth = rt_scf._fock_orth

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    u = expm(-1j*2*rt_scf.timestep*fock_orth)

    mo_coeff_orth_new = np.matmul(u, rt_scf.mo_coeff_orth_old)
    
    rt_scf.mo_coeff_orth_old = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    rt_scf._scf.mo_coeff = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

def magnus_interpol(rt_scf):
    '''
    C'(t+dt) = U(t+0.5dt)C'(t)
    U(t+0.5dt) = exp(-i*dt*F')

    1. Extrapolate F'(t+0.5dt)
    2. Propagate
    3. Build new F'(t+dt), interpolate new F'(t+0.5dt)
    4. Repeat propagation and interpolation until convergence
    '''

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    fock_orth_p12dt = 2 * rt_scf._fock_orth - rt_scf._fock_orth_n12dt
    
    # Update time, mol is updated here if rt_scf is an Ehrenfest obj
    rt_scf.update_time()

    for iteration in range(rt_scf.magnus_maxiter):
        u = expm(-1j*rt_scf.timestep*fock_orth_p12dt)

        mo_coeff_orth_pdt = np.matmul(u, mo_coeff_orth)
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                          mo_occ=rt_scf.occ)
        rt_scf.current_time += rt_scf.timestep
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)
        rt_scf.current_time -= rt_scf.timestep

        if (iteration > 0 and
        abs(np.linalg.norm(mo_coeff_ao_pdt)
        - np.linalg.norm(mo_coeff_ao_pdt_old)) < rt_scf.magnus_tolerance):

            rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
            rt_scf.den_ao = den_ao_pdt
            rt_scf.fock_orth = fock_orth_pdt
            rt_scf.fock_orth_n12dt = fock_orth_p12dt
            break
        fock_orth_p12dt = 0.5 * (rt_scf._fock_orth + fock_orth_pdt)

        mo_coeff_ao_pdt_old = mo_coeff_ao_pdt

        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt
    rt_scf._fock_orth = fock_orth_pdt
    rt_scf._fock_orth_n12dt = fock_orth_p12dt


def rk4(rt_scf):
    '''
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    '''

    fock_orth = rt_scf._fock_orth
    
    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)

    # k1
    k1 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth))
    mo_coeff_orth_1 = mo_coeff_orth + 1/2 * k1

    # k2
    k2 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_1))
    mo_coeff_orth_2 = mo_coeff_orth + 1/2 * k2

    # k3
    k3 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_2))
    mo_coeff_orth_3 = mo_coeff_orth + k3

    # k4
    k4 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_3))

    mo_coeff_orth_new = mo_coeff_orth + (k1/6 + k2/3 + k3/3 + k4/6)
    mo_coeff_ao_new = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)

    rt_scf._scf.mo_coeff = mo_coeff_ao_new
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

def rk4cr(rt_cr,fo,fs,fc):
    print(ishermitian(rt_cr._h1e_mo))
    '''
    i d/dt|r> = sum(s) X(sr)|s>
    i d/dt C(I) = sum(J) H(JI)C(J)-X(JI)C(J)
    '''

    def updateMO(moNew):
        rt_cr.ao_to_mo = np.copy(moNew)
        rt_cr.mo_to_ao = rt_cr.get_mo_to_ao()
        rt_cr._scf.mo_coeff[:,:rt_cr.numP] = np.copy(rt_cr.mo_to_ao)
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()

    def updateHam(t,h1,h2,c):
        rt_cr.apply_potential(t,h1,h2,c)
        #print(ishermitian(rt_cr._h1e_AO))
        rt_cr._h1e_orth = rt_cr.get_h1e_orth()
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_orth = rt_cr.get_h2e_orth()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()
        #print(ishermitian(rt_cr._h1e_mo))
        #print(t)

    x_mat = rt_cr.get_x()

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_cr.update_time()

    e0, h1Act, h2Act = rt_cr.get_embH(x_mat)
    print(ishermitian(h1Act))
    #if not ishermitian(h1Act):
    #    print(h2Act)
    #    rawOut = np.copy(h1Act)
    #    h1Act = (rawOut+rawOut.conj().T)/2
    reci0 = np.copy(rt_cr._scf.ci.real)
    imci0 = np.copy(rt_cr._scf.ci.imag)
    ck1 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0))+(applyham_pyscf.apply_ham_pyscf_check(imci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0))
    rk1 = -1j*np.matmul(x_mat,rt_cr.ao_to_mo)

    c0 = np.copy(rt_cr._scf.ci)
    #print('c0')
    #print(c0)
    mo0 = np.copy(rt_cr.ao_to_mo)

    c1 = rt_cr._scf.ci + (rt_cr.timestep*ck1/2)
    print('c1')
    print(c1)
    mo1 = rt_cr.ao_to_mo +(rt_cr.timestep*rk1/2)
    if rt_cr._castype == 'CASSCF':
        updateMO(mo1)
    rt_cr._scf.ci = np.copy(c1)
    rt_cr.den_ao = rt_cr._scf.make_rdm1()
    if len(rt_cr._potential) > 0:
        updateHam(rt_cr.current_time - (rt_cr.timestep/2),h1Act,h2Act,rt_cr)

    x2 = rt_cr.get_x()
    e2, h1a2, h2a2 = rt_cr.get_embH(x2)
    print(ishermitian(h1a2))
    reci1 = np.copy(c1.real)
    imci1 = np.copy(c1.imag)
    ck2 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2))+(applyham_pyscf.apply_ham_pyscf_check(imci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2))
    rk2 = -1j*np.matmul(x2,rt_cr.ao_to_mo)

    c2 = c0 + (rt_cr.timestep*ck2/2)
    print('c2')
    print(c2)
    mo2 = mo0 + (rt_cr.timestep*rk2/2)
    if rt_cr._castype == 'CASSCF':
        updateMO(mo2)
    rt_cr._scf.ci = np.copy(c2)
    rt_cr.den_ao = rt_cr._scf.make_rdm1()

    x3 = rt_cr.get_x()
    e3, h1a3, h2a3 = rt_cr.get_embH(x3)
    print(ishermitian(h1a3))
    reci2 = np.copy(c2.real)
    imci2 = np.copy(c2.imag)
    ck3 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3))+(applyham_pyscf.apply_ham_pyscf_check(imci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3))
    rk3 = -1j*np.matmul(x3,rt_cr.ao_to_mo)

    c3 = c0 + (rt_cr.timestep*ck3)
    print('c3')
    print(c3)
    mo3 = mo0 + (rt_cr.timestep*rk3)
    if rt_cr._castype == 'CASSCF':
        updateMO(mo3)
    rt_cr._scf.ci = np.copy(c3)
    rt_cr.den_ao = rt_cr._scf.make_rdm1()
    #if len(rt_cr._potential) > 0:
    #    updateHam(rt_cr.current_time)

    x4 = rt_cr.get_x()
    e4, h1a4, h2a4 = rt_cr.get_embH(x4)
    print(ishermitian(h1a4))
    reci3 = np.copy(c3.real)
    imci3 = np.copy(c3.imag)
    ck4 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4))+(applyham_pyscf.apply_ham_pyscf_check(imci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4))
    rk4 = -1j*np.matmul(x4,rt_cr.ao_to_mo)

    cf = c0 + ((rt_cr.timestep/6)*(ck1+(2*ck2)+(2*ck3)+ck4))
    print('cf')
    print(cf)
    mof = mo0 + ((rt_cr.timestep/6)*(rk1+(2*rk2)+(2*rk3)+rk4))
    if rt_cr._castype == 'CASSCF':
        updateMO(mof)
    rt_cr._scf.ci = np.copy(cf)
    rt_cr.den_ao = rt_cr._scf.make_rdm1()
    ef, h1f, h2f = rt_cr.get_embH(np.zeros((rt_cr.numP,rt_cr.numP)))
    print(ishermitian(h1f))
    output = np.zeros(3)
    output[0] = rt_cr.current_time
    output[1] = fci_mod.get_FCI_E(
                h1f,
                h2f,
                ef,
                cf,
                rt_cr._scf.ncas,
                rt_cr._scf.nelecas[0],
                rt_cr._scf.nelecas[1],
                gen=False,
            ) # Not actually sure what this does
    corr1RDMcas = fci_mod.get_corr1RDM(cf, rt_cr._scf.ncas, rt_cr._scf.nelecas)
    corr1RDMmo = np.zeros((rt_cr.numP,rt_cr.numP)).astype(np.complex128)
    for a in range(rt_cr._scf.ncore):
        corr1RDMmo[a][a] = 2
    for a in range(rt_cr._scf.ncas):
        for b in range(rt_cr._scf.ncas):
            corr1RDMmo[a+rt_cr._scf.ncore][b+rt_cr._scf.ncore] = corr1RDMcas[a][b]
    corr1RDM = np.matmul(mof.conj().T,np.matmul(corr1RDMmo,mof))
    #corr1RDM = corr1RDMmo
    #diagcorr1RDM = np.real(np.diag(corr1RDM))
    #corrdens = np.copy(diagcorr1RDM)
    #corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    output[2] = np.real(np.sum(np.diag(corr1RDM))) # Gives number of electrons. Shouldn't ever change.
    #print(cf)
    p0 = cf[0][0]
    p1a = cf[0][1]
    p1b = cf[1][0]
    corrdens = np.array([np.abs(p0),np.abs(p1a),np.abs(p1b)])
    corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    if len(rt_cr._potential) > 0:
        updateHam(rt_cr.current_time)
    
    np.savetxt(fo, output.reshape(1, output.shape[0]), fs)
    fo.flush()
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()
    

INTEGRATORS = {
    'magnus_step' : magnus_step,
    'magnus_interpol' : magnus_interpol,
    'rk4' : rk4,
    'rk4cr': rk4cr
}

def get_integrator(rt_scf):
    return INTEGRATORS[rt_scf.prop]
