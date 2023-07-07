import numpy as np

# OBF class
# This class is used to calculate the OBF coefficients in an efficient manner.
# The OBF coefficients are calculated in the setup_task method.
# The OBF coefficients are time-dependent and can be calculated using get_Phi(t), get_dPhi(t), and get_ddPhi(t).
# The OBF coefficients map a weight vector w to a function y(t) = Phi(t) @ w.
# What makes OBF special is how the weight vector w is composed.
# The weight vector w is composed of a sequence of node positions, including the initial and final position and positions in between.
# The weight vector w is also composed of the initial and final velocity.
# The number of node positions that Phi is computed for is specified by the h argument in the setup_task method.
class OBF():
    def __init__(self, ndof):
        # ndof: number of degrees of freedom
        self.ndof = ndof
        self.h = np.empty(0)

    def setup_task(self, h):
        # h: sequence of time intervals (e.g. [0.5, 0.5] for a 1 second task with two 0.5 second intervals)
        if np.array_equal(h, self.h):
            return
        self.h = h
        self.t_nodes = np.concatenate(([0], np.cumsum(h)))
        self.N = len(h)
        self.T = np.sum(h)
        self.nw_scalar = self.N + 1 + 2

        self.__M = np.zeros((self.N, 2, 2))
        for n in range(self.N):
            h_ = self.h[n]
            M_inv = np.array([[h_**3/3, -h_**2/2],[-h_**2/2, h_]])
            self.__M[n] = np.linalg.inv(M_inv)

        self.__P = self.__get_P()

        Omegas = []
        for n in range(self.N):
            Omegas.append(self.__get_Omega(n))
        self.Omegas = np.array(Omegas)

    def get_basis(self, t):
        # Compute Phi, dPhi, ddPhi
        # t: time, scalar or np.array
        t_len = np.size(t)

        t = np.maximum(0.0, np.minimum(self.T, t))
        lower_node_indeces = np.argmax(t[:,None] <= self.t_nodes[1:], axis=1)
        t__ = t - self.t_nodes[lower_node_indeces]

        c_q = np.zeros((t_len, self.nw_scalar))
        c_dq = np.zeros((t_len, self.N + 1))
        # Set the correct column according to lower_node_indeces to 1
        c_q[np.arange(t_len), lower_node_indeces] = 1.
        c_dq[np.arange(t_len), lower_node_indeces] = 1.
        Omega = self.Omegas[lower_node_indeces] # (t_len, 2, 5)

        t_feature = np.concatenate((-t__[:, None]**3/6, t__[:, None]**2/2), axis=1) # (t_len, 2)
        dt_feature = np.concatenate((-t__[:, None]**2/2, t__[:, None]), axis=1) # (t_len, 2)
        ddt_feature = np.concatenate((-t__[:, None], np.ones_like(t__).reshape(-1, 1)), axis=1) # (t_len, 2)

        Phi_ = c_q + t__[:,None] * c_dq @ self.__P + np.sum(t_feature[:, :, None] * Omega, axis=1)
        Phi = np.kron(Phi_, np.eye(self.ndof))
        dPhi_ = c_dq @ self.__P + np.sum(dt_feature[:, :, None] * Omega, axis=1)
        dPhi = np.kron(dPhi_, np.eye(self.ndof))
        ddPhi_ = np.sum(ddt_feature[:, :, None] * Omega, axis=1)
        ddPhi = np.kron(ddPhi_, np.eye(self.ndof))

        return Phi, dPhi, ddPhi

    def get_Phi(self, t):
        # t: time, scalar or np.array
        t_len = np.size(t)

        t = np.maximum(0.0, np.minimum(self.T, t))
        lower_node_indeces = np.argmax(t[:,None] <= self.t_nodes[1:], axis=1)
        t__ = t - self.t_nodes[lower_node_indeces]

        c_q = np.zeros((t_len, self.nw_scalar))
        c_dq = np.zeros((t_len, self.N + 1))
        # Set the correct column according to lower_node_indeces to 1
        c_q[np.arange(t_len), lower_node_indeces] = 1.
        c_dq[np.arange(t_len), lower_node_indeces] = 1.
        Omega = self.Omegas[lower_node_indeces] # (t_len, 2, 5)

        t_feature = np.concatenate((-t__[:, None]**3/6, t__[:, None]**2/2), axis=1) # (t_len, 2)

        base_ = c_q + t__[:,None] * c_dq @ self.__P + np.sum(t_feature[:, :, None] * Omega, axis=1)
        base = np.kron(base_, np.eye(self.ndof))

        return base

    def get_dPhi(self, t):
        # Compute dPhi
        # t: time, scalar or np.array
        t_len = np.size(t)

        t = np.maximum(0.0, np.minimum(self.T, t))
        lower_node_indeces = np.argmax(t[:,None] <= self.t_nodes[1:], axis=1)
        t__ = t - self.t_nodes[lower_node_indeces]

        c_q = np.zeros((t_len, self.nw_scalar))
        c_dq = np.zeros((t_len, self.N + 1))
        # Set the correct column according to lower_node_indeces to 1
        c_q[np.arange(t_len), lower_node_indeces] = 1.
        c_dq[np.arange(t_len), lower_node_indeces] = 1.
        Omega = self.Omegas[lower_node_indeces] # (t_len, 2, 5)

        dt_feature = np.concatenate((-t__[:, None]**2/2, t__[:, None]), axis=1) # (t_len, 2)

        dPhi_ = c_dq @ self.__P + np.sum(dt_feature[:, :, None] * Omega, axis=1)
        dPhi = np.kron(dPhi_, np.eye(self.ndof))

        return dPhi

    def get_ddPhi(self, t):
        # Compute ddPhi
        # t: time, scalar or np.array
        t_len = np.size(t)

        t = np.maximum(0.0, np.minimum(self.T, t))
        lower_node_indeces = np.argmax(t[:,None] <= self.t_nodes[1:], axis=1)
        t__ = t - self.t_nodes[lower_node_indeces]

        c_q = np.zeros((t_len, self.nw_scalar))
        c_dq = np.zeros((t_len, self.N + 1))
        # Set the correct column according to lower_node_indeces to 1
        c_q[np.arange(t_len), lower_node_indeces] = 1.
        c_dq[np.arange(t_len), lower_node_indeces] = 1.
        Omega = self.Omegas[lower_node_indeces] # (t_len, 2, 5)

        ddt_feature = np.concatenate((-t__[:, None], np.ones_like(t__).reshape(-1, 1)), axis=1) # (t_len, 2)

        ddPhi_ = np.sum(ddt_feature[:, :, None] * Omega, axis=1)
        ddPhi = np.kron(ddPhi_, np.eye(self.ndof))

        return ddPhi
        
    def get_y(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        Phi = self.get_Phi(t)
        y = Phi @ w
        if np.size(t) == 1:
            return y.reshape(self.ndof)
        return y.reshape(np.size(t), self.ndof)

    def get_dy(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        dPhi = self.get_dPhi(t)
        dy = dPhi @ w
        if np.size(t) == 1:
            return dy.reshape(self.ndof)
        return dy.reshape(np.size(t), self.ndof)

    def get_ddy(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        ddPhi = self.get_ddPhi(t)
        ddy = ddPhi @ w
        if np.size(t) == 1:
            return ddy.reshape(self.ndof)
        return ddy.reshape(np.size(t), self.ndof)

    def __get_Omega(self, n):
        return self.__M[n] @ (self.__get_L_w(n) + self.__get_L_dq(n) @ self.__P)

    # Outdated: This function is not used anymore. It was too slow.
    def __get_base(self, t, der):
        base = np.zeros((self.ndof * np.size(t), self.ndof * self.nw_scalar))
        if np.size(t) == 1:
            t_array = np.array([t])
        else:
            t_array = t

        for i in range(np.size(t)):
            t_ = np.max([0.0, np.min([self.T, t_array[i]])])
            t_start = 0.
            for n in range(self.N):
                if t_ <= t_start + self.h[n] + 1e-6:
                    t__ = t_ - t_start
                    c_q = np.zeros((1, self.nw_scalar))
                    c_q[0, n] = 1.
                    c_dq = np.zeros((1, self.N + 1))
                    c_dq[0, n] = 1.
                    Omega = self.__get_Omega(n)
                    if der == 0:
                        base_ = c_q + t__ * c_dq @ self.__P + np.array([[-t__**3/6, t__**2/2]]) @ Omega
                    elif der == 1:
                        base_ = c_dq @ self.__P + np.array([[-t__**2/2, t__]]) @ Omega
                    elif der == 2:
                        base_ = np.array([[-t__, 1.]]) @ Omega
                    else:
                        print('Invalid argument.')
                    base[i*self.ndof:(i+1)*self.ndof] = np.kron(base_, np.eye(self.ndof))
                    break
                t_start += self.h[n]

        return base

    def __get_P(self):
        P_dq = np.zeros((self.N+1, self.N+1))
        P_w  = np.zeros((self.N+1, self.nw_scalar))
        for n in range(self.N-1):
            a_n = np.array([[0., 1.]]) @ self.__M[n+1]
            b_n = np.array([[-self.h[n], 1.]]) @ self.__M[n]
            P_dq[n] = b_n @ self.__get_L_dq(n) - a_n @ self.__get_L_dq(n+1)
            P_w[n]  = a_n @ self.__get_L_w(n+1) - b_n @ self.__get_L_w(n)
        P_dq[self.N-1, 0] = 1
        P_w[self.N-1, self.N+1] = 1
        P_dq[self.N, self.N] = 1
        P_w[self.N, self.N+2] = 1

        return np.linalg.inv(P_dq) @ P_w

    def __get_L_w(self, n):
        if n < 0 or n >= self.N:
            print('Invalid argument.')
            return []
        L_w_n = np.zeros((2, self.nw_scalar))
        L_w_n[0, n] = -1
        L_w_n[0, n+1] = 1
        return L_w_n

    def __get_L_dq(self, n):
        if n < 0 or n >= self.N:
            print('Invalid argument.')
            return []
        L_dq_n = np.zeros((2, self.N+1))
        L_dq_n[0, n+1] = -self.h[n]
        L_dq_n[1, n] = -1
        L_dq_n[1, n+1] = 1
        return L_dq_n
