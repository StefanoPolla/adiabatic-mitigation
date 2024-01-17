from typing import Callable

import numpy as np
import openfermion as of


class bump_dephaser():
    def __init__(self, n_pts: int):
        self.n_pts = n_pts
        self.x = np.linspace(0, 1, n_pts)
        self.y = np.zeros_like(self.x)
        self.y[1:-1] = np.exp((4 * (self.x[1:-1] - 1) * self.x[1:-1])**-1)
        self.nyquist_freq = n_pts / 2  

    def fourier_bump(self, frequencies):
        if np.any(np.abs(frequencies) > self.nyquist_freq):
            raise NotImplementedError('a point above the implemented Nyquist frequency was requested')
        waves = np.exp(-1.j * np.outer(self.x, frequencies))
        return self.y @ waves / self.n_pts

    def dephasing_matrix(self, dephasing_time, eigvals):
        energy_differences = np.ravel(eigvals[np.newaxis, :] - eigvals[:, np.newaxis])
        fourier_coeffs = self.fourier_bump(dephasing_time * energy_differences)
        return np.reshape(fourier_coeffs, [len(eigvals), len(eigvals)])


def pauli_field(nspins, P):
    return np.sum([of.QubitOperator(((i, P),),) for i in range(nspins)])

def ising(nspins):
    return np.sum([of.QubitOperator(((i, 'Z'), (i+1, 'Z')),)
                   for i in range(nspins-1)])

def ground_state(ham_dense):
    eigvals, eigvecs = np.linalg.eigh(ham_dense)
    return eigvecs[:, 0]

def fidelity(a, b):
    if len(a.shape) == 1 and len(b.shape) == 1:
        return np.abs(a.conj() @ b)**2
    elif len(a.shape) == 2 and len(b.shape) == 1:
        return np.abs(b.conj() @ a @ b)
    elif len(a.shape) == 1 and len(b.shape) == 2:
        return np.abs(a.conj() @ b @ a)
    raise NotImplementedError()

def state_to_dm(state):
    return np.outer(state.conj(), state)

class XZchain():
    def __init__(self, nspins, hx, hz = 1, hzz = 1,
                 schedule: Callable = None):
        '''
        h0 = - \sum_j X_j
        h1 = - hx \sum_j X_j - hz \sum_j Z_j + hzz \sum_j Z_j Z_{j+1}
        
        a positive hzz implies ferromagnetism
        '''
        self.nspins = nspins
        self.h0_qop = - pauli_field(nspins, 'X')
        self.h1_qop = (- hx * pauli_field(nspins, 'X')
                       - hz * pauli_field(nspins, 'Z')
                       + hzz * ising(nspins))
        self.h0_dense = of.get_sparse_operator(self.h0_qop, n_qubits=nspins).A
        self.h1_dense = of.get_sparse_operator(self.h1_qop, n_qubits=nspins).A
        if schedule:
            self.schedule = schedule
        else:
            self.schedule = lambda x: np.array([1-x, x])
        self.init_state = ground_state(self.h0_dense)
        
        self.final_eigvals, self.final_eigvecs = np.linalg.eigh(self.h1_dense)
        
        self.final_ground_state = self.final_eigvecs[:, 0]
        self.final_gs_reflection = (np.eye(2**self.nspins) 
                                    - 2 * np.outer(self.final_ground_state, 
                                                   self.final_ground_state.conj()))
        
        self.dephaser = bump_dephaser(10000)

    def ham_qop(self, x):
        sx = self.schedule(x)
        return sx[0] * self.h0_qop + sx[1] * self.h1_qop

    def ham_dense(self, x):
        sx = self.schedule(x)
        return sx[0] * self.h0_dense + sx[1] * self.h1_dense
    
    def eig(self, x):
        if x == 1:
            return self.final_eigvals, self.final_eigvecs
        else:
            return np.linalg.eigh(self.ham_dense(x))

    def evolution(self, x, dt):
        eigvals, eigvecs = self.eig(x)
        phases = np.exp(1.j * eigvals * dt)
        return np.einsum('ij, j, jk', 
                         eigvecs, phases, eigvecs.T.conj())
    
    def spectrum(self, x):
        eigvals, eigvecs = self.eig(x)
        return eigvals
    
    def adiab_evo(self, T, nsteps):
        dt = T / nsteps
        U = np.eye(2**self.nspins, dtype=complex)
        for step in range(nsteps):
            x = (step + 0.5) / nsteps
            U = self.evolution(x, dt) @ U
        return U
    
    def adiab_state(self, T, nsteps):
        return self.adiab_evo(T, nsteps) @ self.init_state
    
    def adiabatic_expval(self, T, nsteps, observable):
        state = self.adiab_state(T, nsteps)
        return state.conj().T @ observable @ state
    
    def reversed_adiab_evo(self, T, nsteps):
        dt = T/nsteps
        U = np.eye(2**self.nspins, dtype=complex)
        for step in range(nsteps):
            x = 1 - ((step + 0.5) / nsteps)
            U = self.evolution(x, dt) @ U
        return U
    
    def adiabatic_echo(self, T, nsteps):
        return self.reversed_adiab_evo(T, nsteps) @ self.adiab_evo(T, nsteps)
    
    def echoed_state(self, T, nsteps):
        return self.adiabatic_echo(T, nsteps) @ self.init_state

    def echoed_state_dm(self, T, nsteps):
        dm = state_to_dm(self.init_state)
        evo = self.adiab_evo(T, nsteps)
        dm = evo @ dm @ evo.conj().T
        rev_evo = self.reversed_adiab_evo(T, nsteps)
        dm = rev_evo @ dm @ rev_evo.conj().T
        return dm
    
    def dephasing(self, dm, dephasing_time, *, x=1):
        eigvals, eigvecs = self.eig(x)
        if dephasing_time is not None:
            deph_matr = self.dephaser.dephasing_matrix(dephasing_time, eigvals)
        else:
            deph_matr = np.eye(2**self.nspins)
        
        dm = eigvecs.T.conj() @ dm @ eigvecs
        dm = dm * deph_matr
        dm = eigvecs @ dm @ eigvecs.T.conj()
        return dm
    
    def dephased_state_dm(self, T, nsteps, dephasing_time=None):
        dm = state_to_dm(self.adiab_state(T, nsteps))
        dm = self.dephasing(dm, dephasing_time)
        return dm
    
    def dephased_echoed_state_dm(self, T, nsteps, dephasing_time=None):
        dm = self.dephased_state_dm(T, nsteps, dephasing_time)
        rev_evo = self.reversed_adiab_evo(T, nsteps)
        dm = rev_evo @ dm @ rev_evo.conj().T
        return dm
    
    def dephased_adiabatic_echo(self, T, nsteps, dephasing_time=None):
        dm = self.dephased_echoed_state_dm(T, nsteps, dephasing_time)
        return self.init_state.conj() @ dm @ self.init_state
    
    def aev_circuit_state(self, T, nsteps, dephasing_time, observable):
        dm = self.dephased_state_dm(T, nsteps, dephasing_time)
        dm = observable @ dm
        dm = self.dephasing(dm, dephasing_time)
        rev_evo = self.reversed_adiab_evo(T, nsteps)
        dm = rev_evo @ dm @ rev_evo.conj().T
        return dm

    def aev_nodeph_signal_and_echo(self, T, nsteps, observable):
        dm = state_to_dm(self.init_state)
        evo = self.adiab_evo(T, nsteps)
        rev_evo = self.reversed_adiab_evo(T, nsteps)
        signal = np.trace(rev_evo @ evo @ dm @ evo.conj().T @ observable @ rev_evo.conj().T @ dm)
        echo = np.trace(rev_evo @ evo @ dm @ evo.conj().T @ rev_evo.conj().T @ dm)
        return signal, echo

    def aev_nodeph_expval(self, T, nsteps, observable):
        signal, echo = self.aev_nodeph_signal_and_echo(T, nsteps, observable)
        return signal / echo
    
    def aev_signal_and_echo(self, T, nsteps, dephasing_time, observable):
        dm = state_to_dm(self.init_state)
        evo = self.adiab_evo(T, nsteps)
        rev_evo = self.reversed_adiab_evo(T, nsteps)
        
        in_dm = self.dephasing(evo @ dm @ evo.conj().T, dephasing_time)
        signal_op = self.dephasing(in_dm @ observable, dephasing_time)
        echo_op = self.dephasing(in_dm, dephasing_time)
        
        signal = np.trace(rev_evo @ signal_op @ rev_evo.conj().T @ dm)
        echo = np.trace(rev_evo @ echo_op @ rev_evo.conj().T @ dm)
        
        return signal, echo
    
    def aev_expval(self, T, nsteps, dephasing_time, observable):
        signal, echo = self.aev_signal_and_echo(T, nsteps, dephasing_time, observable)
        return signal / echo