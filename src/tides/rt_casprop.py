import numpy as np
from tides import rt_observables, rt_integrators
from tides.rt_utils import update_chkfile, print_info
from scipy.linalg import inv

# Real-Time Propagation for CAS/RAS

def propagate(rt_cr,mo_coeff_print):
    print_info(rt_cr,mo_coeff_print) # I don't mess with this
    rt_observables._check_observables(rt_cr) # I don't mess with this

    rt_cr._integrate_function = rt_integrators.get_integrator(rt_cr)
    rt_cr._h1e_orth = rt_cr.get_h1e_orth()
    rt_cr._h2e_orth = rt_cr.get_h2e_orth()
    rt_cr._h1e_mo = rt_cr.get_h1e_mo()
    rt_cr._h2e_mo = rt_cr.get_h2e_mo()
    #print(rt_cr._h1e_mo)
    #rt_cr.mo_to_orth = rt_cr.get_mo_to_orth()
    #rt_cr.orth_to_mo = rt_cr.get_orth_to_mo()
    file_output = open(rt_cr.outName, "wb")
    file_corrdens = open(rt_cr.corName, "wb")
    fmt_str = "%20.8e"
    #print('init')
    #print(rt_cr._scf.mo_coeff)
    for i in range(0, int((rt_cr.max_time - rt_cr._t0) / rt_cr.timestep)): # So calculation terminates once max_time is reached after restarts
        if np.mod(i,rt_cr.frequency) == 0:
            rt_observables.get_observables(rt_cr)
            if rt_cr.chkfile is not None:
                update_chkfile(rt_cr)

        rt_cr._integrate_function(rt_cr,file_output,fmt_str,file_corrdens)

    rt_observables.get_observables(rt_cr)  # Collect observables at final time
    #print('mo coeff')
    #print(rt_cr._scf.mo_coeff)
    if rt_cr.chkfile is not None:
        update_chkfile(rt_cr)