import numpy as np
from scipy.linalg import inv
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
				damping_term = self.prefac * (1 - np.exp(self.expconst* energy_corrected))
				if damping_term < (-1 * self.maxval):
					damping_term = -1 * self.maxval
				damping_diagonal.append(damping_term)
			else:
				damping_diagonal.append(0)

		damping_diagonal = np.array(damping_diagonal).astype(np.complex128)

		damping_matrix = np.diag(damping_diagonal)
		damping_matrix = np.dot(coeff_matrix, np.dot(damping_matrix, np.conj(coeff_matrix.T)))

		transform = inv(rt_scf.orth.T)
		damping_matrix_ao = np.dot(transform, np.dot(damping_matrix, transform.T))
		return 1j * damping_matrix_ao

	def calculate_potential_spin(self, rt_scf, coeff_matrix=None, mo_energy=None):
		if rt_scf.nmat == 1:
			return self.calculate_cap(rt_scf, coeff_matrix, mo_energy)
		else:
			results = []
			for spin in range(rt_scf.nmat):
				cm = coeff_matrix[spin]  if coeff_matrix is not None else None
				me = mo_energy[spin] if mo_energy is not None else None
				results.append(self.calculate_cap(rt_scf, cm, me))
			return np.stack(results)

	@abstractmethod
	def calculate_potential(self, rt_scf, coeff_matrix=None, mo_energy=None):
		pass

#Choose the basis that diagonalizes the Fock matrix
class FOCK_PRINT_DIMER_IN_NOSCF(MOCAP):
	def __init__(self, expconst, emin, dimer, prefac=1, maxval=100, cap_filename='cap_dimer_in_noscf_with_fock_prop_print.txt'):
		super().__init__(expconst, emin, prefac, maxval)
		self.fock = None
		self.dimer = dimer
		self.cap_filename = cap_filename
	def calculate_potential(self, rt_scf):
		fock=rt_scf.fock_ao

		if fock.ndim == 2:
			fock_orth = rt_scf.rotate_fock_to_orth(fock)
			mo_energy, fock_eigvecs = np.linalg.eigh(fock_orth)
		else:
			mo_energy = []  #was having issues here with dimensionality, so we will just make a list and append to it.
			fock_eigvecs = []
			for spin in range(fock.shape[0]):
				fock_orth = rt_scf.rotate_fock_to_orth(fock[spin])
				eigval, eigvec = np.linalg.eigh(fock_orth)
				mo_energy.append(eigval)
				fock_eigvecs.append(eigvec)

		cap_matrix = super().calculate_potential_spin(rt_scf, coeff_matrix=fock_eigvecs, mo_energy=mo_energy)
		
		dimer_coeff = self.dimer.mo_coeff
		scf_energy = self.dimer.mo_energy
		if dimer_coeff.ndim == 2:
			dimer_coeff_orth = rt_scf.rotate_coeff_to_orth(dimer_coeff)
		else:
			dimer_coeff_orth = np.array([rt_scf.rotate_coeff_to_orth(dimer_coeff[spin]) for spin in range(dimer_coeff.shape[0])])
		dimer_cap_ao = super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff_orth, mo_energy=scf_energy)

		# 2. Transform DIMER CAP from AO to FOCK basis
		noscf_coeff = rt_scf.mo_coeff_print
		if noscf_coeff.ndim == 2:
			noscf_coeff = rt_scf.rotate_coeff_to_orth(noscf_coeff)
			cap_noscf = np.dot(np.conj(noscf_coeff.T), np.dot(dimer_cap_ao, noscf_coeff))
		else:
			noscf_coeff = np.array([rt_scf.rotate_coeff_to_orth(noscf_coeff[spin]) for spin in range(noscf_coeff.shape[0])])
			cap_noscf = np.array([np.dot(np.conj(noscf_coeff[spin].T), np.dot(dimer_cap_ao[spin], noscf_coeff[spin])) for spin in range(noscf_coeff.shape[0])])

		# 3. Print
		if not hasattr(rt_scf, '_cap_printed') or rt_scf._cap_printed != rt_scf.current_time:
			with open(self.cap_filename, 'a') as f:
				f.write(f"Time: {getattr(rt_scf, 'current_time', 'unknown')}\n")
				f.write(np.array2string(cap_noscf, precision=6, separator=', '))
				f.write("\n\n")
			rt_scf._cap_printed = rt_scf.current_time

		return cap_matrix

	
class FOCK_NOSCF(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100, cap_filename='density_noscf.txt'):
		super().__init__(expconst, emin, prefac, maxval)
		self.fock = None
		self.cap_filename = cap_filename
	def calculate_potential(self, rt_scf):
		fock=rt_scf.fock_ao

		if fock.ndim == 2:
			fock_orth = rt_scf.rotate_fock_to_orth(fock)
			mo_energy, fock_eigvecs = np.linalg.eigh(fock_orth)
		else:
			mo_energy = []  #was having issues here with dimensionality, so we will just make a list and append to it.
			fock_eigvecs = []
			for spin in range(fock.shape[0]):
				fock_orth = rt_scf.rotate_fock_to_orth(fock[spin])
				eigval, eigvec = np.linalg.eigh(fock_orth)
				mo_energy.append(eigval)
				fock_eigvecs.append(eigvec)


		cap_matrix = super().calculate_potential_spin(rt_scf, coeff_matrix=fock_eigvecs, mo_energy=mo_energy)

		if not hasattr(rt_scf, '_cap_printed') or rt_scf._cap_printed != rt_scf.current_time:
			noscf_coeff = rt_scf.mo_coeff_print
			if noscf_coeff.ndim == 2:
				noscf_coeff = rt_scf.rotate_coeff_to_orth(noscf_coeff)
				den_ao = rt_scf.den_ao
				den_noscf = np.dot(np.conj(noscf_coeff.T), np.dot(den_ao, noscf_coeff))
			else:
				noscf_coeff = np.array([rt_scf.rotate_coeff_to_orth(noscf_coeff[spin]) for spin in range(noscf_coeff.shape[0])])
				den_noscf = np.array([np.dot(np.conj(noscf_coeff[spin].T), np.dot(rt_scf.den_ao[spin], noscf_coeff[spin])) for spin in range(noscf_coeff.shape[0])])
				with open(self.cap_filename, 'a') as f:
					f.write(f"Time: {getattr(rt_scf, 'current_time', 'unknown')}\n")
					f.write("Density in NOSCF basis:\n")
					f.write(np.array2string(den_noscf, precision=6, separator=', '))
					f.write("\n\n")
			rt_scf._cap_printed = rt_scf.current_time
	
		return cap_matrix


class FOCK_STATIC(MOCAP):
	def __init__(self, expconst, emin, rt_scf, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		# Store the initial Fock matrix and orthogonalization
		fock = rt_scf._scf.get_fock(dm=rt_scf.den_ao).astype(np.complex128)
		if fock.ndim == 2:
			self.fock_orth = rt_scf.rotate_fock_to_orth(fock)
			self.mo_energy, self.fock_eigvecs = np.linalg.eigh(self.fock_orth)
		else:
			self.mo_energy = []
			self.fock_eigvecs = []
			for spin in range(fock.shape[0]):
				fock_orth = rt_scf.rotate_fock_to_orth(fock[spin])
				eigval, eigvec = np.linalg.eigh(fock_orth)
				self.mo_energy.append(eigval)
				self.fock_eigvecs.append(eigvec)

	def calculate_potential(self, rt_scf):
		# Always use the stored initial Fock eigensystem
		return super().calculate_potential_spin(rt_scf, coeff_matrix=self.fock_eigvecs, mo_energy=self.mo_energy)
	
#Choose the basis of the SCF orbitals
class DIMER_NOSCF(MOCAP):
	def __init__(self, expconst, emin, dimer, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer


	def calculate_potential(self, rt_scf):

	#here we are adding Ehrenfest compatibility, if we are ehrenfest then the energies and coefficient update whenever we update.
	#maybe add a modulo statement here so that we only update whenever we update on the ehrenfest timestep, talk to kretchmer about nailing this down.
		if rt_scf.istype('RT_Ehrenfest'):
			#get_scf_orbitals(rt_scf)
			mo_coeff = rt_scf.mo_coeff_print
			scf_energy = rt_scf.mo_energy_print
		else:
			mo_coeff = self.dimer.mo_coeff  #am i having it update over time here, as the dimer changes?
			scf_energy = self.dimer.mo_energy

		if mo_coeff.ndim == 2:
			dimer_coeff = rt_scf.rotate_coeff_to_orth(mo_coeff)
		else:
			dimer_coeff = np.array([rt_scf.rotate_coeff_to_orth(mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		
	#	print("SCF energies used in CAP:", scf_energy)  # <-- Add this line

#		return super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)

		cap_matrix = super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)

#		if not hasattr(rt_scf, '_cap_printed') or rt_scf._cap_printed != rt_scf.current_time:
#			noscf_coeff = rt_scf.mo_coeff_print
#			if noscf_coeff.ndim == 2:
#				noscf_coeff = rt_scf.rotate_coeff_to_orth(noscf_coeff)
#				cap_noscf = np.dot(np.conj(noscf_coeff.T), np.dot(cap_matrix, noscf_coeff))
#			else:
#				noscf_coeff = np.array([rt_scf.rotate_coeff_to_orth(noscf_coeff[spin]) for spin in range(noscf_coeff.shape[0])])
#				cap_noscf = np.array([np.dot(np.conj(noscf_coeff[spin].T), np.dot(cap_matrix[spin], noscf_coeff[spin])) for spin in range(noscf_coeff.shape[0])])
#			with open(self.cap_filename, 'a') as f:
#				f.write(f"Time: {getattr(rt_scf, 'current_time', 'unknown')}\n")
#				f.write(np.array2string(cap_noscf, precision=6, separator=', '))
#				f.write("\n\n")
#			rt_scf._cap_printed = rt_scf.current_time
	
		return cap_matrix
	
class DIMER_FOCK(MOCAP):
	def __init__(self, expconst, emin, dimer, prefac=1, maxval=100, cap_filename='cap_dimer_fock_basis_with_DIMER_prop.txt'):
		super().__init__(expconst, emin, prefac, maxval)
		self.dimer = dimer
		self.cap_filename = cap_filename


	def calculate_potential(self, rt_scf):

	#here we are adding Ehrenfest compatibility, if we are ehrenfest then the energies and coefficient update whenever we update.
	#maybe add a modulo statement here so that we only update whenever we update on the ehrenfest timestep, talk to kretchmer about nailing this down.
		if rt_scf.istype('RT_Ehrenfest'):
			#get_scf_orbitals(rt_scf)
			mo_coeff = rt_scf.mo_coeff_print
			scf_energy = rt_scf.mo_energy_print
		else:
			mo_coeff = self.dimer.mo_coeff  #am i having it update over time here, as the dimer changes?
			scf_energy = self.dimer.mo_energy

		if mo_coeff.ndim == 2:
			dimer_coeff = rt_scf.rotate_coeff_to_orth(mo_coeff)
		else:
			dimer_coeff = np.array([rt_scf.rotate_coeff_to_orth(mo_coeff[spin]) for spin in range(mo_coeff.shape[0])])
		
	#	print("SCF energies used in CAP:", scf_energy)  # <-- Add this line

#		return super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)

		cap_matrix = super().calculate_potential_spin(rt_scf, coeff_matrix=dimer_coeff, mo_energy=scf_energy)

		if not hasattr(rt_scf, '_cap_printed') or rt_scf._cap_printed != rt_scf.current_time:
			fock = rt_scf.fock_ao
			if fock.ndim == 2:
				fock_orth = rt_scf.rotate_fock_to_orth(fock)
				_, fock_coeff = np.linalg.eigh(fock_orth)
				cap_fock = np.dot(np.conj(fock_coeff.T), np.dot(cap_matrix, fock_coeff))
			else:
				cap_fock = []
				for spin in range(fock.shape[0]):
					fock_orth = rt_scf.rotate_fock_to_orth(fock[spin])
					_, fock_coeff = np.linalg.eigh(fock_orth)
					cap_fock.append(np.dot(np.conj(fock_coeff.T), np.dot(cap_matrix[spin], fock_coeff)))
				cap_fock = np.array(cap_fock)
			with open(self.cap_filename, 'a') as f:
				f.write(f"Time: {getattr(rt_scf, 'current_time', 'unknown')}\n")
				f.write(np.array2string(cap_fock, precision=6, separator=', '))
				f.write("\n\n")
			rt_scf._cap_printed = rt_scf.current_time
	
		return cap_matrix


# Choose the basis of the NOSCF orbitals
class NOSCF(MOCAP):
	def __init__(self, expconst, emin, prefac=1, maxval=100):
		super().__init__(expconst, emin, prefac, maxval)
		self.noscf_coeff = None
		self.noscf_energy = None

	def calculate_potential(self, rt_scf):
		if rt_scf.istype('RT_Ehrenfest'):
			get_noscf_orbitals(rt_scf)  #this is the line to change later, maybe just pull or only do a calculation when we havent in the ehrenfest.
			noscf_coeff = rt_scf.mo_coeff_print
			noscf_energy = rt_scf.mo_energy_print
		else:
			if self.noscf_coeff is None or self.noscf_energy is None:
				get_noscf_orbitals(rt_scf)
				self.noscf_coeff = rt_scf.mo_coeff_print
				self.noscf_energy = rt_scf.mo_energy_print
			noscf_coeff = self.noscf_coeff
			noscf_energy = self.noscf_energy
		if noscf_coeff.ndim == 2:
			noscf_coeff = rt_scf.rotate_coeff_to_orth(noscf_coeff)
		else:
			noscf_coeff = np.array([rt_scf.rotate_coeff_to_orth(noscf_coeff[spin]) for spin in range(noscf_coeff.shape[0])])
		return super().calculate_potential_spin(rt_scf, coeff_matrix=noscf_coeff, mo_energy=noscf_energy)