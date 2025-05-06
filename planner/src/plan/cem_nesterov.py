#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
#import matplotlib.pyplot as plt
import jax

from plan import cem
from plan import gradient_descent


class cem_nesterov:
    def __init__(self,
                 maxiter = 200,
                 num_controls = 20,
                 cem_num_samples = 50,
                 cem_percentage_elite = 0.1,
                 cem_stomp_like=True,
                 ):

        self.cem_optimizer = cem.CEM(maxiter=maxiter,
                                         num_controls=num_controls,
                                         num_samples=cem_num_samples,
                                         percentage_elite=cem_percentage_elite,
                                         stomp_like=cem_stomp_like)
        self.gradient_descent_optimizer = gradient_descent.gradient_descent(maxiter=maxiter,
                                                                           num_controls=num_controls)

        print("!!PLANNER INITIALIZED!!")

    @partial(jit, static_argnums=(0,))
    def compute_controls(self, x_init, y_init, theta_init, 
                v_init, omega_init,
                x_goal, y_goal,
                x_obs, y_obs,
                mean_init):

        # CEM
        v_optimal, omega_optimal, traj_optimal, new_mean = self.cem_optimizer.compute_controls(x_init, y_init, theta_init,
                                                       v_init, omega_init,
                                                       x_goal, y_goal,
                                                       x_obs, y_obs,
                                                       mean_init)

        cem_controls = jnp.concatenate((v_optimal, omega_optimal))

        # Gradient Descent
        v_optimal, omega_optimal, traj_optimal = self.gradient_descent_optimizer.compute_controls(x_init, y_init, theta_init,
                                                                   v_init, omega_init,
                                                                   x_goal, y_goal,
                                                                   x_obs, y_obs,
                                                                   cem_controls)

        new_mean = jnp.concatenate((v_optimal, omega_optimal))

        return v_optimal, omega_optimal, traj_optimal, new_mean
