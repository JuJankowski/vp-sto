from .obf import OBF
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, random
from functools import partial

np.seterr(all="ignore") # ignore sqrt warning as it is handled inside the code

# CONFIG = {
#     0: qT subject to optimization
#     1: qT is fixed

@partial(jit, static_argnums=(0,1,), inline=True)
def sample_via_points(batch_size, N_via, q0, dq0, R, mu_qT, Q, mu_prior, P_prior, key):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    N_eval = 100 # just used for numerical integration
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    Phi = obf.get_Phi(s_eval)
    dPhi = obf.get_dPhi(s_eval)
    ddPhi = obf.get_ddPhi(s_eval)

    ddPhi_p = ddPhi[:, ndof:-2*ndof]
    ddPhi_b = np.concatenate((ddPhi[:, :ndof],
                              ddPhi[:, -2*ndof:]), axis=1)

    dqT = np.zeros(dq0.shape)
    dim_p = ndof * N_via

    P_smooth = ddPhi_p.T @ ddPhi_p * R / N_eval
    ddPhi_p_pinv = np.linalg.pinv(ddPhi_p)
    M = - ddPhi_p_pinv @ ddPhi_b

    PhiT_p = Phi[-ndof:, ndof:-2*ndof]
    PhiT_p_sq = PhiT_p.T @ PhiT_p

    ### Traced operations (optimized in run-time) ###
    mu_smooth = M @ jnp.concatenate((q0, dq0, dqT))
    
    P_bias = Q * PhiT_p_sq
    mu_bias = jnp.tile(mu_qT, N_via)

    # Compute the posterior precision and its Cholesky decomposition for fast sampling
    P_post = P_smooth + P_bias + P_prior
    L_P = jnp.linalg.cholesky(P_post)
    # Compute the posterior mean
    f_post = lax.linalg.triangular_solve(L_P, P_smooth @ mu_smooth + P_bias @ mu_bias + P_prior @ mu_prior, left_side=True, lower=True)

    # Sample from the posterior
    white_noise = random.normal(key, shape=(batch_size, dim_p))
    return lax.linalg.triangular_solve(L_P, (f_post + white_noise).T, left_side=True, lower=True, transpose_a=True).T

@jit
def get_T_vel(p, q0, dq0, dqT, vel_lim):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    dim_p = p.shape[0]
    N_via = dim_p // ndof
    N_eval = 100 # just used for sampling the trajectory
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    dPhi = obf.get_dPhi(s_eval)
    ddPhi = obf.get_ddPhi(s_eval)

    dPhi_q = dPhi[ndof:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof), don't include the first pos
    ddPhi_q = ddPhi[ndof:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof), don't include the first pos
    dPhi_dq = dPhi[ndof:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof), don't include the first pos
    ddPhi_dq = ddPhi[ndof:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof), don't include the first pos

    ### Traced operations (optimized in run-time) ###
    q_via_list = jnp.concatenate((q0, p))
    dq_list = jnp.concatenate((dq0, dqT))

    dq_q = (dPhi_q @ q_via_list).reshape(N_eval-1, ndof)
    dq_dq = (dPhi_dq @ dq_list).reshape(N_eval-1, ndof)
    T_dq = jnp.maximum(jnp.max(dq_q / (vel_lim - dq_dq)),
                       jnp.max(- dq_q / (vel_lim + dq_dq)))
    
    return T_dq

@jit
def get_T_velacc(p, q0, dq0, dqT, vel_lim, acc_lim):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    dim_p = p.shape[0]
    N_via = dim_p // ndof
    N_eval = 100 # just used for sampling the trajectory
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    dPhi = obf.get_dPhi(s_eval)
    ddPhi = obf.get_ddPhi(s_eval)

    dPhi_q = dPhi[ndof:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof), don't include the first pos
    ddPhi_q = ddPhi[ndof:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof), don't include the first pos
    dPhi_dq = dPhi[ndof:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof), don't include the first pos
    ddPhi_dq = ddPhi[ndof:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof), don't include the first pos

    ### Traced operations (optimized in run-time) ###
    q_via_list = jnp.concatenate((q0, p))
    dq_list = jnp.concatenate((dq0, dqT))

    dq_q = (dPhi_q @ q_via_list).reshape(N_eval-1, ndof)
    dq_dq = (dPhi_dq @ dq_list).reshape(N_eval-1, ndof)
    T_dq = jnp.maximum(jnp.max(dq_q / (vel_lim - dq_dq)),
                       jnp.max(- dq_q / (vel_lim + dq_dq)))
    
    ddq_q = (ddPhi_q @ q_via_list).reshape(N_eval-1, ndof)
    ddq_dq = (ddPhi_dq @ dq_list).reshape(N_eval-1, ndof)
    T_p = ddq_dq / (2. * acc_lim)
    T_ddq = jnp.maximum(jnp.max(T_p + jnp.nan_to_num(jnp.sqrt(T_p**2 + ddq_q / acc_lim), nan=-jnp.inf)),
                        jnp.max(-T_p + jnp.nan_to_num(jnp.sqrt(T_p**2 - ddq_q / acc_lim), nan=-jnp.inf)))
    return jnp.maximum(T_dq, T_ddq)

@jit
def batched_get_T_vel(p, q0, dq0, dqT, vel_lim):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    batch_size, dim_p = p.shape
    N_via = dim_p // ndof
    N_eval = 50 # just used for sampling the trajectory
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    dPhi = obf.get_dPhi(s_eval[1:]) # don't include the first pos
    ddPhi = obf.get_ddPhi(s_eval[1:]) # don't include the first pos

    dPhi_q = dPhi[:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof)
    ddPhi_q = ddPhi[:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof)
    dPhi_dq = dPhi[:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof)
    ddPhi_dq = ddPhi[:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof)

    ### Traced operations (optimized in run-time) ###
    q0_ = jnp.repeat(q0[None,:], batch_size, axis=0)
    dq0_ = jnp.repeat(dq0[None,:], batch_size, axis=0)
    dqT_ = jnp.repeat(dqT[None,:], batch_size, axis=0)

    q_via_list = jnp.concatenate((q0_, p), axis=1).T
    dq_list = jnp.concatenate((dq0_, dqT_), axis=1).T

    dq_q = (dPhi_q @ q_via_list).T.reshape(batch_size, N_eval-1, ndof)
    dq_dq = (dPhi_dq @ dq_list).T.reshape(batch_size, N_eval-1, ndof)
    T_dq = jnp.maximum(jnp.max(dq_q / (vel_lim - dq_dq), axis=(1, 2)),
                       jnp.max(- dq_q / (vel_lim + dq_dq), axis=(1, 2)))
    
    return T_dq

#@jit
def batched_get_T_velacc(p, q0, dq0, dqT, vel_lim, acc_lim):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    batch_size, dim_p = p.shape
    N_via = dim_p // ndof
    N_eval = 50 # just used for sampling the trajectory
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    dPhi = obf.get_dPhi(s_eval[1:]) # don't include the first pos
    ddPhi = obf.get_ddPhi(s_eval[1:]) # don't include the first pos

    dPhi_q = dPhi[:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof)
    ddPhi_q = ddPhi[:,:-2*ndof] # ((N_eval-1)*ndof, (N_via+1)*ndof)
    dPhi_dq = dPhi[:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof)
    ddPhi_dq = ddPhi[:,-2*ndof:] # ((N_eval-1)*ndof, 2*ndof)

    ### Traced operations (optimized in run-time) ###
    q0_ = jnp.repeat(q0[None,:], batch_size, axis=0)
    dq0_ = jnp.repeat(dq0[None,:], batch_size, axis=0)
    dqT_ = jnp.repeat(dqT[None,:], batch_size, axis=0)

    q_via_list = jnp.concatenate((q0_, p), axis=1).T
    dq_list = jnp.concatenate((dq0_, dqT_), axis=1).T

    dq_q = (dPhi_q @ q_via_list).T.reshape(batch_size, N_eval-1, ndof)
    dq_dq = (dPhi_dq @ dq_list).T.reshape(batch_size, N_eval-1, ndof)
    T_dq = jnp.maximum(jnp.max(dq_q / (vel_lim - dq_dq), axis=(1, 2)),
                       jnp.max(- dq_q / (vel_lim + dq_dq), axis=(1, 2)))
    
    ddq_q = (ddPhi_q @ q_via_list).T.reshape(batch_size, N_eval-1, ndof)
    ddq_dq = (ddPhi_dq @ dq_list).T.reshape(batch_size, N_eval-1, ndof)
    T_p = ddq_dq / (2. * acc_lim)
    T_ddq = jnp.maximum(jnp.max(T_p + jnp.nan_to_num(jnp.sqrt(T_p**2 + ddq_q / acc_lim), nan=-jnp.inf), axis=(1, 2)),
                        jnp.max(-T_p + jnp.nan_to_num(jnp.sqrt(T_p**2 - ddq_q / acc_lim), nan=-jnp.inf), axis=(1, 2)))
    return jnp.maximum(T_dq, T_ddq)

#@partial(jit, static_argnums=(0,))
def get_trajectory(N_eval, p, T, q0, dq0, dqT):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    dim_p = p.shape[0]
    N_via = dim_p // ndof
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    Phi = obf.get_Phi(s_eval)
    dPhi = obf.get_dPhi(s_eval)
    ddPhi = obf.get_ddPhi(s_eval)

    ### Traced operations (optimized in run-time) ###
    w = jnp.concatenate((q0, p, T*dq0, T*dqT))
    # Compute the trajectory
    q = (Phi @ w).reshape(N_eval, ndof)
    dq = (dPhi @ (w/T)).reshape(N_eval, ndof)
    ddq = (ddPhi @ (w/T**2)).reshape(N_eval, ndof)

    return q, dq, ddq

@partial(jit, static_argnums=(0,))
def batched_get_trajectory(N_eval, p, T, q0, dq0, dqT):
    ### Static operations (only executed at compile time) ###
    ndof = q0.shape[0]
    batch_size, dim_p = p.shape
    N_via = dim_p // ndof
    s_eval = np.linspace(0., 1., N_eval)
    obf = OBF(ndof)
    obf.setup_task(np.ones(N_via)/N_via)

    # Compute the basis functions: Phi, dPhi, ddPhi. (N_eval*ndof, (N_via+3)*ndof)
    Phi = obf.get_Phi(s_eval)
    dPhi = obf.get_dPhi(s_eval)
    ddPhi = obf.get_ddPhi(s_eval)

    ### Traced operations (optimized in run-time) ###
    q0_ = jnp.repeat(q0[None,:], batch_size, axis=0)
    dq0_ = jnp.outer(T, dq0)
    dqT_ = jnp.outer(T, dqT)
    w = jnp.concatenate((q0_, p, dq0_, dqT_), axis=1)
    # Compute the trajectory
    q = (Phi @ w.T).T.reshape(batch_size, N_eval, ndof)
    dq = (dPhi @ (w.T/T)).T.reshape(batch_size, N_eval, ndof)
    ddq = (ddPhi @ (w.T/T**2)).T.reshape(batch_size, N_eval, ndof)

    return q, dq, ddq