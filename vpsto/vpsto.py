import numpy as np
from obf import OBF
import cma

class VPSTO():
    def __init__(self, dq_max, ddq_max, N_eval, h=None):
        """
        Args:
            ndof (int): number of degrees of freedom
            T (float): duration of motion
            N_eval (int): number of points to evaluate along trajectory
            h (np array): delta t between two consecutive via points (should sum up to T)
        """
        self.dq_max = dq_max
        self.ddq_max = ddq_max
        self.ndof = len(dq_max)
        self.s_eval = np.linspace(0, 1., N_eval)
        self.obf = OBF(self.ndof)
        self.N = 0
        self.setup_basis(1, h)
        self.w_best = None
        self.T_best = None
        self.x_init = None
        
    def setup_basis(self, N, h=None):
        """
        Args:
            N (int): number of via points to construct the trajectory, including the final point, excluding the initial point
            h (np array): delta t between two consecutive via points (should sum up to T)
        """
        if N == self.N:
            return
        self.N = N
        self.num_basis = N + 3
        self.nw = self.num_basis * self.ndof
        if h is None:
            self.h = np.ones(N) / N
        else:
            self.h = h
        self.obf.setup_task(self.h)
        
        self.Phi = self.obf.get_Phi(self.s_eval)
        self.dPhi = self.obf.get_dPhi(self.s_eval)
        self.ddPhi = self.obf.get_ddPhi(self.s_eval)
        
        self.dPhi_p = self.dPhi[:, self.ndof:-2*self.ndof]
        self.dPhi_b = np.concatenate((self.dPhi[:, :self.ndof],
                                      self.dPhi[:, -2*self.ndof:]), axis=1)
        self.ddPhi_p = self.ddPhi[:, self.ndof:-2*self.ndof]
        self.ddPhi_b = np.concatenate((self.ddPhi[:, :self.ndof],
                                       self.ddPhi[:, -2*self.ndof:]), axis=1)
        
        sigma_p = np.linalg.inv(self.ddPhi_p.T @ self.ddPhi_p)
        self.ddPhi_p_pinv = sigma_p @ self.ddPhi_p.T
        eigv, V = np.linalg.eigh(sigma_p)
        D = np.diag(eigv / np.max(eigv))
        sigma_via = V @ D @ V.T
        self.sigma_p_chol = np.linalg.cholesky(sigma_via)
        self.sigma_p_chol_inv = np.linalg.inv(self.sigma_p_chol)
    
    def sample_trajectory(self, t):
        obf = OBF(self.ndof)
        h = self.T_best * np.ones(self.N) / self.N
        obf.setup_task(h)
        Phi = obf.get_Phi(t)
        dPhi = obf.get_dPhi(t)
        ddPhi = obf.get_ddPhi(t)
        return (Phi@self.w_best).reshape(-1,self.ndof), (dPhi@self.w_best).reshape(-1,self.ndof), (ddPhi@self.w_best).reshape(-1,self.ndof)
    
    def shift_solution_forward(self, dt, q_0, dq_bound, N_next=0):
        if self.w_best is None:
            return
        if N_next == 0:
            N_next = self.N
        obf_via = OBF(self.ndof)
        h = self.T_best * np.ones(self.N) / self.N
        obf_via.setup_task(h)
        T_next = self.T_best - dt
        t_via = np.linspace(0, T_next, N_next+1) + dt
        p_next = (obf_via.get_Phi(t_via) @ self.w_best).reshape(-1, self.ndof)[1:]
        self.setup_basis(N_next)
        mu_p = - self.ddPhi_p_pinv @ self.ddPhi_b @ np.concatenate((q_0, dq_bound))
        self.x_init = self.sigma_p_chol_inv @ (p_next.flatten() - mu_p)

    def get_duration(self, p_list, q_0, dq_bound):
        dq_q = (p_list @ self.dPhi_p.T + self.dPhi_b[:,:self.ndof] @ q_0).reshape(len(p_list), -1, self.ndof)
        dq_dq = (self.dPhi_b[:,self.ndof:] @ dq_bound).reshape(-1, self.ndof)
        T_dq = np.maximum(np.max(dq_q / (self.dq_max - dq_dq), axis=(1, 2)),
                          np.max(- dq_q / (self.dq_max + dq_dq), axis=(1, 2)))
        
        ddq_q = (p_list @ self.ddPhi_p.T + self.ddPhi_b[:,:self.ndof] @ q_0).reshape(len(p_list), -1, self.ndof)
        ddq_dq = (self.ddPhi_b[:,self.ndof:] @ dq_bound).reshape(-1, self.ndof)
        T_p = ddq_dq / (2. * self.ddq_max)
        T_ddq = np.maximum(np.max(T_p + np.nan_to_num(np.sqrt(T_p**2 + ddq_q / self.ddq_max), nan=-np.inf), axis=(1, 2)),
                           np.max(-T_p + np.nan_to_num(np.sqrt(T_p**2 - ddq_q / self.ddq_max), nan=-np.inf), axis=(1, 2)))
        return np.maximum(T_dq, T_ddq)
    
    def get_phenotype(self, p_list, q_0, dq_bound):
        # Compute duration of each movement and roll out
        T_list = self.get_duration(p_list, q_0, dq_bound)
        w_list = np.concatenate((np.tile(q_0, (len(p_list), 1)), p_list, 
                                 np.diag(T_list) @ np.tile(dq_bound, (len(p_list), 1))), axis=1)
        Population = dict()
        Population['Q'] = (w_list @ self.Phi.T).reshape(len(p_list), -1, self.ndof)
        Population['dQ'] = (np.diag(1/T_list) @ (w_list @ self.dPhi.T)).reshape(len(p_list), -1, self.ndof)
        Population['ddQ'] = (np.diag(1/T_list**2) @ (w_list @ self.ddPhi.T)).reshape(len(p_list), -1, self.ndof)
        Population['T'] = T_list
        return Population
        
    def minimize(self, loss, q_0, dq_bound, sigma_init=1.0, max_iter=1000, popsize=10, CMA_diagonal=False):
        dim_x = self.N * self.ndof
        
        if self.x_init is None:
            x_init = np.zeros(dim_x)
        else:
            x_init = self.x_init.copy()
        self.x_init = None
            
        mu_p = - self.ddPhi_p_pinv @ self.ddPhi_b @ np.concatenate((q_0, dq_bound))
        
        cmaes = cma.CMAEvolutionStrategy(x_init, sigma_init, 
                                         {'CMA_diagonal': CMA_diagonal, 
                                          'verbose': -1,
                                          'CMA_active': True,
                                          'popsize': popsize,
                                          'tolfun': 1e-9})
        loss_list = []
        i = 0
        while not cmaes.stop() and i < max_iter:
            x_samples = np.array(cmaes.ask())
            p_samples = mu_p+(self.sigma_p_chol@x_samples.T).T
            pop = self.get_phenotype(p_samples, q_0, dq_bound)
            cmaes.tell(x_samples, loss(pop))
            #Res_pop = self.get_phenotype(np.array([mu_p+self.sigma_p_chol@cmaes.result.xfavorite]), q_0, dq_bound)
            #loss_list.append(loss(Res_pop)[0])
            print(i, end='\r')
            i += 1
        
        x_best = cmaes.result.xfavorite
        p_best = mu_p+self.sigma_p_chol@x_best
        Res_pop = self.get_phenotype(np.array([p_best]), q_0, dq_bound)
        loss_list.append(loss(Res_pop)[0])
        self.w_best = np.concatenate((q_0, p_best, dq_bound))
        self.T_best = Res_pop['T'][0]
        
        return Res_pop['Q'][0], Res_pop['dQ'][0], Res_pop['ddQ'][0], Res_pop['T'][0], loss_list