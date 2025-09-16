import numpy as np

def parse_output(filename):
    '''
    Simple script that parses output file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Get the mol_length
    for line in lines[:100]:
        if 'Mol Length: ' in line:
            mol_length = int(line.split()[2])
            break

    time = []
    energy = []
    kinetic_energy = []
    dipole = []
    quadrupole = []
    mulliken_charge = []
    mulliken_atom_charge = []
    hirsh_atom_charge = []
    mag = []
    hirsh_atom_mag = []
    mo_occ = []
    coords = []
    vels = []
    frag_charge = []
    alpha_energies = []
    beta_energies = []
    plane_partition_charge = []
    
    for idx, line in enumerate(lines):
        if 'Current Time' in line:
            time.append(get_time(line))
        if 'Total Dipole Moment' in line:
            dipole.append(get_dipole(line))
        if 'Total Quadrupole Moment' in line:
            quadrupole.append(get_quadrupole(line))
        if 'Total Energy' in line:
            energy.append(get_energy(line))
        if 'Total Kinetic Energy' in line:
            kinetic_energy.append(get_kinetic_energy(line))
        if 'Molecular Orbital Occupations' in line:
            mo_occ.append(get_mo_occ(line))
        if 'Total Electronic Charge' in line:
            mulliken_charge.append(get_charge(line))
        if 'Fragment' in line and 'Electronic Charge' in line:
            frag_charge.append(get_frag_charge(line))
        if 'Plane Partition Charges' in line:
            plane_partition_charge.append(get_plane_partition_charge(line))
        if 'Atomic Electronic Charges' in line and 'Hirshfeld' not in line:
            mulliken_atom_charge.append(get_atom_charge(lines[idx+1:idx+mol_length+1]))
        if 'Hirshfeld Atomic Electronic Charges' in line:
            hirsh_atom_charge.append(get_atom_charge(lines[idx+1:idx+mol_length+1]))
        if 'Total Magnetization' in line:
            mag.append(get_mag(line))
        if 'Hirshfeld Magnetization' in line:
            hirsh_atom_mag.append(get_atom_mag(lines[idx+1:idx+mol_length+1]))
        if 'Nuclear Coordinates' in line:
            coords.append(get_coords(lines[idx+1:idx+mol_length+1]))
        if 'Nuclear Velocities' in line:
            vels.append(get_vels(lines[idx+1:idx+mol_length+1]))
        if 'Molecular Orbital Energies (Alpha): ' in line:
            alpha_energies.append(get_mo_energy(line))
        if 'Molecular Orbital Energies (Beta): ' in line:
            beta_energies.append(get_mo_energy(line))

    time = np.array(time)
    energy = np.array(energy)
    kinetic_energy = np.array(kinetic_energy)
    dipole = np.array(dipole)
    quadrupole = np.array(quadrupole)
    mulliken_charge = np.array(mulliken_charge)
    mulliken_atom_charge = np.array(mulliken_atom_charge)
    hirsh_atom_charge = np.array(hirsh_atom_charge)
    mag = np.array(mag)
    hirsh_atom_mag = np.array(hirsh_atom_mag)
    mo_occ = np.array(mo_occ)
    coords = np.array(coords)
    vels = np.array(vels)
    frag_charge = np.array(frag_charge).reshape([len(time), int(np.size(frag_charge) / len(time))])
    alpha_energies = np.array(alpha_energies)
    beta_energies = np.array(beta_energies)
    result = {
    'time': time,
    'energy': energy,
    'kinetic_energy': kinetic_energy,
    'dipole': dipole,
    'quadrupole': quadrupole,
    'charge': mulliken_charge,
    'atom_charge': mulliken_atom_charge,
    'mulliken_charge': mulliken_atom_charge,
    'mulliken_atom_charge': mulliken_atom_charge,
    'hirsh_charge': hirsh_atom_charge,
    'hirsh_atom_charge': hirsh_atom_charge,
    'mag': mag,
    'hirsh_mag': hirsh_atom_mag,
    'hirsh_atom_mag': hirsh_atom_mag,
    'mo_occ': mo_occ,
    'coords': coords,
    'vels': vels,
    'frag_charge': frag_charge,
    'plane_partition_charge': plane_partition_charge,
    'alpha_energies': alpha_energies,
    'beta_energies': beta_energies,
    }
    return result

def get_time(line):
    return float(line.split()[3])

def get_energy(line):
    return float(line.split()[3])

def get_kinetic_energy(line):
    return float(line.split()[4])

def get_dipole(line):
    x = float(line.split()[-3])
    y = float(line.split()[-2])
    z = float(line.split()[-1])
    return [x, y, z]

def get_quadrupole(line):
    qline = line.split(':')[1].split('[')[1:4]
    qx = qline[0].split(']')[0].split()
    qy = qline[1].split(']')[0].split()
    qz = qline[2].split(']')[0].split()
    return np.concatenate((qx,qy,qz)).astype(np.float64)


def get_mag(line):
    x = float(line.split()[-3])
    y = float(line.split()[-2])
    z = float(line.split()[-1])
    return [x, y, z]

def get_atom_mag(lines):
    mag = []
    for line in lines:
        x = float(line.split()[-3])
        y = float(line.split()[-2])
        z = float(line.split()[-1])
        mag.append([x, y, z])
    return mag

def get_mo_occ(line):
    return np.array(line.split('Molecular Orbital Occupations: ')[1].split()).astype(np.float64)

def get_mo_energy(line):
    return np.array(line.split('a):')[1].split()).astype(np.float64)

def get_charge(line):
    return float(line.split()[3])

def get_frag_charge(line):
    return float(line.split()[4])

def get_coords(lines):
    coords = []
    for line in lines:
        atom_coords = []
        for i in range(1,4):
            atom_coords.append(float(line.split()[i]))
        coords.append(atom_coords)
    return coords

def get_atom_charge(lines):
    charges = []
    for line in lines:
        charges.append(float(line.split()[1]))
    return charges

def get_vels(lines):
    vels = []
    for line in lines:
        atom_vels = []
        for i in range(1,4):
            atom_vels.append(float(line.split()[i]))
        vels.append(atom_vels)
    return vels

def get_length(coords, atoms):
    # Gets length between 2 atoms
    atoms = [atoms[i] - 1 for i, _ in enumerate(atoms)]
    lens = []
    for coord in coords:
        dist_3d = coord[atoms[1]] - coord[atoms[0]]
        dx, dy, dz = dist_3d[0], dist_3d[1], dist_3d[2]
        lens.append(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    return lens

def get_plane_partition_charge(line):
    import re, ast
    payload = line.split(':', 1)[1].strip()
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?=\s*[+\-]\s*\d*\.?\d+(?:[eE][-+]?\d+)?j)', payload)
    return [float(m) for m in matches]
