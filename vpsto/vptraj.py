import numpy as np
from .obf import OBF

np.seterr(all="ignore") # ignore sqrt warning as it is handled inside the code

# Trajectory representation for the VPSTO algorithm
class VPTraj:
    def __init__(self, ndof, N_eval, N_via, vel_lim, acc_lim):
        self.ndof = ndof       # number of degrees of freedom
        self.N_eval = N_eval   # number of evaluation points along trajectories in cost function
        self.N_via = N_via     # number of via-points
        self.vel_lim = vel_lim # -vel_lim < dq < vel_lim (element-wise)
        self.acc_lim = acc_lim # -acc_lim < ddq < acc_lim (element-wise)
        self.__setup_basis()     # setup the basis functions for fast online retrieval

    def sample_trajectories(self, N_traj, q0, dq0=None, qT=None, dqT=None, Q=None, R=None, mu_prior=None, P_prior=None, T=None):
        # Sample trajectories from the posterior distribution.
        # N_traj: number of trajectories to sample
        # q0: initial position
        # dq0: initial velocity, will be set to zero if not given
        # qT: final position, assumed to be fixed if Q is None, will be assumed to be contained in p if not given
        # dqT: final velocity, will be assumed to be contained in p if not given
        # Q: precision matrix for qT, ignored if qT is None
        # R: penalization matrix for accelerations over time, assumed to be 1 if not given
        # p_prior: prior mean for via-point parameters
        # P_prior: prior precision for via-point parameters
        # T: duration of the trajectory, will be computed if not given

        if dqT is None and T is None:
            print('Either T or dqT must be given. Setting dqT to zero.')
            dqT = np.zeros(self.ndof)

        # Check the input
        if R is None:
            R = 1
        # Check if Q is a float and turn it into a matrix
        if Q is not None and isinstance(Q, (int, float)):
            Q = Q * np.eye(self.ndof)
        if dq0 is None:
            dq0 = np.zeros(self.ndof)
        if qT is None or Q is None:
            # qT is not fixed, so it is assumed to be contained in p
            mu_qT = np.zeros(self.ndof)
            Q = np.zeros((self.ndof, self.ndof))
        else:
            mu_qT = qT

        if dqT is None:
            # dqT is not fixed, so it is assumed to be contained in p
            dim_p = self.ndof * (self.N_via + 1)
            ddPhi_p = np.concatenate((self.ddPhi[:, self.ndof:-2*self.ndof],
                                      self.ddPhi[:, -self.ndof:]), axis=1)
            ddPhi_b = np.concatenate((self.ddPhi[:, :self.ndof],
                                      self.ddPhi[:, -2*self.ndof:-self.ndof]), axis=1)
            P_smooth = ddPhi_p.T @ ddPhi_p * R / self.N_eval
            mu_smooth = - self.ddPhi_p_qdq @ ddPhi_b @ np.concatenate((q0, dq0))

            Phi_T = np.concatenate((self.Phi[-self.ndof:, self.ndof:-2*self.ndof],
                                    self.Phi[-self.ndof:, -self.ndof:]), axis=1)
            P_bias = Phi_T.T @ Q @ Phi_T
            mu_bias = np.tile(mu_qT, self.N_via+1)
        else:
            # dqT is fixed, so it is not contained in p
            dim_p = self.ndof * self.N_via
            ddPhi_p = self.ddPhi[:, self.ndof:-2*self.ndof]
            ddPhi_b = np.concatenate((self.ddPhi[:, :self.ndof],
                                      self.ddPhi[:, -2*self.ndof:]), axis=1)
            P_smooth = ddPhi_p.T @ ddPhi_p * R / self.N_eval
            mu_smooth = - self.ddPhi_p_q @ ddPhi_b @ np.concatenate((q0, dq0, dqT))

            Phi_T = self.Phi[-self.ndof:, self.ndof:-2*self.ndof]
            P_bias = Phi_T.T @ Q @ Phi_T
            mu_bias = np.tile(mu_qT, self.N_via)

        if mu_prior is None or len(mu_prior) != dim_p:
            mu_prior = np.zeros(dim_p)
        if P_prior is None:
            P_prior = np.zeros((dim_p, dim_p))
        elif isinstance(P_prior, (int, float)):
            P_prior = P_prior * np.eye(dim_p)

        # Compute the posterior precision and its Cholesky decomposition for fast sampling
        P_post = P_smooth + P_prior + P_bias
        L_P = np.linalg.cholesky(P_post)
        # Compute the posterior mean
        f_post = np.linalg.solve(L_P, P_smooth @ mu_smooth + P_prior @ mu_prior + P_bias @ mu_bias)

        # Sample from the posterior
        self.white_noise = np.random.normal(size=(N_traj, dim_p))
        p = np.linalg.solve(L_P.T, (f_post + self.white_noise).T).T # np.random.multivariate_normal(mu_post, sigma_post, N_traj)

        if T is None:
            T = self.get_min_duration(p, q0, dq0, None, dqT)

        # Compute the trajectories
        q, dq, ddq = self.get_trajectory(p, q0, dq0, None, dqT, T)

        return q, dq, ddq, p, T

    def get_min_duration(self, p, q0, dq0=None, qT=None, dqT=None):
        # Compute the minimum duration the trajectory can have such that the
        # velocity and acceleration limits are not violated.
        #
        # p: via-point parameters. Can be in batch.
        # q0: initial position
        # dq0: initial velocity, will be set to zero if not given
        # qT: final position, will be assumed to be contained in p if not given
        # dqT: final velocity, will be assumed to be contained in p if not given

        # Check if p is single or batch
        if len(p.shape) == 1:
            p = p.reshape(1,-1)
        batch_size = len(p)

        q0 = np.tile(q0, (batch_size, 1))
        # Check the input
        if dq0 is None:
            dq0 = np.zeros((batch_size, self.ndof))
        else:
            dq0 = np.tile(dq0, (batch_size, 1))

        if qT is None and dqT is None: # qT and dqT are contained in p
            q_via_list = np.concatenate((q0, p[:,:-self.ndof]), axis=1)
            dq_list = np.concatenate((dq0, p[:,-self.ndof:]), axis=1)
        elif qT is None: # qT is contained in p
            dqT = np.tile(dqT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p), axis=1)
            dq_list = np.concatenate((dq0, dqT), axis=1)
        elif dqT is None: # dqT is contained in p
            qT = np.tile(qT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p[:,:-self.ndof], qT), axis=1)
            dq_list = np.concatenate((dq0, p[:,-self.ndof:]), axis=1)
        else: # qT and dqT are given
            qT = np.tile(qT, (batch_size, 1))
            dqT = np.tile(dqT, (batch_size, 1))
            q_via_list = np.concatenate((q0, p, qT), axis=1)
            dq_list = np.concatenate((dq0, dqT), axis=1)

        dq_q = (q_via_list @ self.dPhi[self.ndof:,:-2*self.ndof].T).reshape(batch_size, -1, self.ndof)
        dq_dq = (dq_list @ self.dPhi[self.ndof:,-2*self.ndof:].T).reshape(batch_size, -1, self.ndof)
        T_dq = np.maximum(np.max(dq_q / (self.vel_lim - dq_dq), axis=(1, 2)),
                          np.max(- dq_q / (self.vel_lim + dq_dq), axis=(1, 2)))
        
        ddq_q = (q_via_list @ self.ddPhi[self.ndof:,:-2*self.ndof].T).reshape(batch_size, -1, self.ndof)
        ddq_dq = (dq_list @ self.ddPhi[self.ndof:,-2*self.ndof:].T).reshape(batch_size, -1, self.ndof)
        T_p = ddq_dq / (2. * self.acc_lim)
        T_ddq = np.maximum(np.max(T_p + np.nan_to_num(np.sqrt(T_p**2 + ddq_q / self.acc_lim), nan=-np.inf), axis=(1, 2)),
                           np.max(-T_p + np.nan_to_num(np.sqrt(T_p**2 - ddq_q / self.acc_lim), nan=-np.inf), axis=(1, 2)))
        return np.maximum(T_dq, T_ddq)

    def get_trajectory(self, p, q0, dq0=None, qT=None, dqT=None, T=None):
        # Compute the trajectory from the given parameters
        # p: via-point parameters. Can be in batch.
        # q0: initial position
        # dq0: initial velocity, will be set to zero if not given
        # qT: final position, will be assumed to be contained in p if not given
        # dqT: final velocity, will be assumed to be contained in p if not given
        # T: duration of the trajectory, will be assumed to be T=1 if not given

        # Check if p is single or batch
        if len(p.shape) == 1:
            p = p.reshape(1,-1)
        batch_size = len(p)

        q0 = np.tile(q0, (batch_size, 1))
        # Check the input
        if dq0 is None:
            dq0 = np.zeros((batch_size, self.ndof))
        else:
            dq0 = np.tile(dq0, (batch_size, 1))

        if qT is None and dqT is None: # qT and dqT are contained in p
            w = np.concatenate((q0, p[:,:-self.ndof], dq0, p[:,-self.ndof:]), axis=1)
        elif qT is None: # qT is contained in p
            dqT = np.tile(dqT, (batch_size, 1))
            w = np.concatenate((q0, p, dq0, dqT), axis=1)
        elif dqT is None: # dqT is contained in p
            qT = np.tile(qT, (batch_size, 1))
            w = np.concatenate((q0, p[:,:-self.ndof], qT, dq0, p[:,-self.ndof:]), axis=1)
        else: # qT and dqT are given
            qT = np.tile(qT, (batch_size, 1))
            dqT = np.tile(dqT, (batch_size, 1))
            w = np.concatenate((q0, p, qT, dq0, dqT), axis=1)
        if T is None:
            T = 1.0

        # Adapt w to the duration of the trajectory
        w[:,-2*self.ndof:] = (T * w[:,-2*self.ndof:].T).T

        # Compute the trajectory
        q = (self.Phi @ w.T).T.reshape(batch_size, -1, self.ndof)
        dq = (self.dPhi @ (w.T/T)).T.reshape(batch_size, -1, self.ndof)
        ddq = (self.ddPhi @ (w.T/T**2)).T.reshape(batch_size, -1, self.ndof)

        return q, dq, ddq

    def get_trajectory_at_time(self, t, p, q0, dq0=None, qT=None, dqT=None, T=None):
        # Compute the trajectory from the given parameters at the given time
        # t: time at which the trajectory should be evaluated, 0 <= t <= T
        # p: via-point parameters
        # q0: initial position
        # dq0: initial velocity, will be set to zero if not given
        # qT: final position, will be assumed to be contained in p if not given
        # dqT: final velocity, will be assumed to be contained in p if not given
        # T: duration of the trajectory, will be assumed to be T=1 if not given

        if dq0 is None:
            dq0 = np.zeros(self.ndof)

        if qT is None and dqT is None: # qT and dqT are contained in p
            w = np.concatenate((q0, p[:-self.ndof], dq0, p[-self.ndof:]))
        elif qT is None: # qT is contained in p
            w = np.concatenate((q0, p, dq0, dqT))
        elif dqT is None: # dqT is contained in p
            w = np.concatenate((q0, p[:-self.ndof], qT, dq0, p[-self.ndof:]))
        else: # qT and dqT are given
            w = np.concatenate((q0, p, qT, dq0, dqT))
        if T is None:
            T = 1.0
            
        # Recompute the basis functions for the duration and at the specified time
        obf = OBF(self.ndof)
        obf.setup_task(T * np.ones(self.N_via) / self.N_via)
        q = (obf.get_Phi(t) @ w).reshape(-1,self.ndof)
        dq = (obf.get_dPhi(t) @ w).reshape(-1,self.ndof)
        ddq = (obf.get_ddPhi(t) @ w).reshape(-1,self.ndof)

        return q, dq, ddq
    
    def __setup_basis(self):
        s_eval = np.linspace(0., 1., self.N_eval)
        obf = OBF(self.ndof)
        obf.setup_task(np.ones(self.N_via)/self.N_via)
        
        # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
        self.Phi = obf.get_Phi(s_eval)
        self.dPhi = obf.get_dPhi(s_eval)
        self.ddPhi = obf.get_ddPhi(s_eval)
        
        # Compute the smoothing matrix for the case: qT fixed and dqT fixed
        ddPhi_p = self.ddPhi[:, self.ndof:-3*self.ndof]
        self.S = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) # Covariance matrix correlating the via-points to minimize acceleration
        self.ddPhi_p = self.S @ ddPhi_p.T / self.N_eval           # pseudo-inverse of ddPhi_p
        self.S_chol = np.linalg.cholesky(self.S)                  # Cholesky decomposition of S
        self.S_chol_inv = np.linalg.inv(self.S_chol)              # inverse of S_chol

        # Compute the smoothing matrix for the case: qT free and dqT fixed
        ddPhi_p = self.ddPhi[:, self.ndof:-2*self.ndof]
        self.S_q = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) # Covariance matrix correlating the via-points to minimize acceleration
        self.ddPhi_p_q = self.S_q @ ddPhi_p.T / self.N_eval         # pseudo-inverse of ddPhi_p
        self.S_q_chol = np.linalg.cholesky(self.S_q)                # Cholesky decomposition of S
        self.S_q_chol_inv = np.linalg.inv(self.S_q_chol)            # inverse of S_chol

        # Compute the smoothing matrix for the case: qT fixed and dqT free
        ddPhi_p = np.concatenate((self.ddPhi[:, self.ndof:-3*self.ndof],
                                  self.ddPhi[:, -self.ndof:]), axis=1)
        self.S_dq = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) # Covariance matrix correlating the via-points to minimize acceleration
        self.ddPhi_p_dq = self.S_dq @ ddPhi_p.T / self.N_eval        # pseudo-inverse of ddPhi_p
        self.S_dq_chol = np.linalg.cholesky(self.S_dq)               # Cholesky decomposition of S
        self.S_dq_chol_inv = np.linalg.inv(self.S_dq_chol)           # inverse of S_chol

        # Compute the smoothing matrix for the case: qT free and dqT free
        ddPhi_p = np.concatenate((self.ddPhi[:, self.ndof:-2*self.ndof],
                                  self.ddPhi[:, -self.ndof:]), axis=1)
        self.S_qdq = np.linalg.inv(ddPhi_p.T @ ddPhi_p / self.N_eval) # Covariance matrix correlating the via-points to minimize acceleration
        self.ddPhi_p_qdq = self.S_qdq @ ddPhi_p.T / self.N_eval       # pseudo-inverse of ddPhi_p
        self.S_qdq_chol = np.linalg.cholesky(self.S_qdq)              # Cholesky decomposition of S
        self.S_qdq_chol_inv = np.linalg.inv(self.S_qdq_chol)          # inverse of S_chol