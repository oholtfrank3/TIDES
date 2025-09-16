import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

'''
Real-time Output Functions
'''

def update_output(rt_scf):
    rt_scf._log.note(f'{"="*25} \n')
    rt_scf._log.note(f'Current Time (AU): {rt_scf.current_time:.8f} \n')
    for key, function in rt_scf._observables_functions.items():
        function[1](rt_scf)

    rt_scf._log.note(f'{"="*25} \n')

def _print_energy(rt_scf):
    energy = rt_scf._energy
    rt_scf._log.note(f'Total Energy (AU): {energy[0]} \n')
    if len(energy) > 1:
        for index, fragment in enumerate(energy[1:]):
            rt_scf._log.note(f'Fragment {index + 1} Energy (AU): {fragment} \n')
    if rt_scf.istype('RT_Ehrenfest'):
        kinetic_energy = rt_scf._kinetic_energy
        rt_scf._log.note(f'Total Kinetic Energy (AU): {np.sum(kinetic_energy)} \n')
        rt_scf._log.info(f'Atom Kinetic Energies (AU):')
        for atom in zip(rt_scf.nuc.labels, kinetic_energy):
            rt_scf._log.info(f' {atom[0]} {atom[1]}')
        rt_scf._log.info(' ')
        for index, frag in enumerate(rt_scf.fragments):
            rt_scf._log.note(f'Fragment {index + 1} Kinetic Energy (AU): {np.sum(kinetic_energy[frag.match_indices])} \n')

def _print_mo_occ(rt_scf):
    mo_occ = rt_scf._mo_occ
    rt_scf._log.note(f'Molecular Orbital Occupations: {" ".join(map(str,mo_occ))} \n')

def _print_charge(rt_scf):
    charge = rt_scf._charge
    rt_scf._log.note(f'Total Electronic Charge: {np.real(charge[0])} \n')
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            rt_scf._log.note(f'Fragment {index + 1} Electronic Charge: {np.real(fragment)} \n')

def _print_hirshfeld_charge(rt_scf):
    labels = rt_scf.labels
    atom_charges = rt_scf._hirshfeld_charges
    rt_scf._log.note('Hirshfeld Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_scf._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_scf._log.note(' ')

def _print_dipole(rt_scf):
    dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')

def _print_quadrupole(rt_scf):
    quadrupole = rt_scf._quadrupole
    rt_scf._log.note(f'Total Quadrupole Moment [[XX,XY,XZ], [YX,YY,YZ], [ZX,ZY,ZZ]] (AU): {" ".join(map(str,quadrupole))} \n')

def _print_mag(rt_scf):
    mag = rt_scf._mag
    rt_scf._log.note(f'Total Magnetization [X, Y, Z]: {" ".join(map(str,np.real(mag)))} \n')

def _print_hirshfeld_mag(rt_scf):
    labels = rt_scf.labels
    mx = rt_scf._hirshfeld_mx_atoms
    my = rt_scf._hirshfeld_my_atoms
    mz = rt_scf._hirshfeld_mz_atoms
    m = np.transpose([mx, my, mz])
    rt_scf._log.note(f'Hirshfeld Magnetization [X, Y, Z]:')
    for atom in zip(labels, m):
        rt_scf._log.note(f' {atom[0]}: {np.real(atom[1][0])} {np.real(atom[1][1])} {np.real(atom[1][2])}')
    rt_scf._log.note(' ')

def _print_mulliken_charge(rt_scf):
    labels = rt_scf.labels
    atom_charges = rt_scf._atom_charges
    rt_scf._log.note('Atomic Electronic Charges:')
    for atom in zip(labels, atom_charges):
        rt_scf._log.note(f' {atom[0]} \t {np.real(atom[1])}')
    rt_scf._log.note(' ')

def _print_nuclei(rt_scf):
    nuclei = rt_scf._nuclei
    rt_scf._log.note(f'Nuclear Coordinates (Angstrom):')
    for atom in zip(nuclei[0], nuclei[1]):
        coords_str = "\t".join([f"{x:.11f}" for x in atom[1]])
        rt_scf._log.note(f' {atom[0]} \t {coords_str}')
   #      rt_scf._log.note(f' {atom[0]} \t {"\t".join(map(lambda x: f"{x:.11f}",atom[1]))}')
    rt_scf._log.note(' ')
    rt_scf._log.info(f'Nuclear Velocities (Angstrom / AU):')
    for atom in zip(nuclei[0], nuclei[2]):
#        rt_scf._log.info(f' {atom[0]} \t {"\t".join(map(lambda x: f"{x:.11f}",atom[1]))}')
        vel_str = "\t".join([f"{x:.11f}" for x in atom[1]])
        rt_scf._log.info(f' {atom[0]} \t {vel_str}')
    rt_scf._log.info(' ')
    rt_scf._log.debug(f'Nuclear Forces (AU):')
    for atom in zip(nuclei[0], nuclei[3]):
#        rt_scf._log.debug(f' {atom[0]} \t {"\t".join(map(lambda x: f"{x:.11f}",atom[1]))}')
        force_str = "\t".join([f"{x:.11f}" for x in atom[1]])
        rt_scf._log.debug(f' {atom[0]} \t {force_str}')
    rt_scf._log.debug(' ')

def _print_mo_coeff(rt_scf):
    rt_scf._log.note(f'\n{"*"*25} Molecular Orbital Coefficients (AO Basis): {"*"*25}\n {rt_scf._scf.mo_coeff} \n{"*"*50}\n')

def _print_den_ao(rt_scf):
    rt_scf._log.note(f'\n{"@"*25} Density Matrix (AO Basis): {"@"*25}\n {rt_scf.den_ao} \n{"@"*50}\n')

def _print_fock_ao(rt_scf):
    rt_scf._log.note(f'\n{"+"*25} Fock Matrix (AO Basis): {"+"*25}\n {rt_scf.fock_ao} \n{"+"*50}\n')

def _print_plane_partition_charge(rt_scf):
    charge = rt_scf._plane_partition_charge
    rt_scf._log.note(f'Plane Partition Charges: Frag1={charge[0]:.6f}, Frag2={charge[1]:.6f}\n')