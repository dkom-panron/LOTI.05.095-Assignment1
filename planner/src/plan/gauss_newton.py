#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
import matplotlib.pyplot as plt
import jax

class gauss_newton:
    def __init__(self, maxiter = 200, num_controls = 20):

        self.delta_t = 0.1
        self.n = num_controls
        self.maxiter = maxiter

        self.eta = 0.1 # nb, inverse of gradient descent eta

        self.w_goal = 2.0
        self.w_velocity = 1.0
        self.w_omega = 1.0
        self.w_obstacle = 0.1

        self.beta = 0.3

        self.jac_func = jit(jacfwd(self.compute_error_func, argnums = (0)))
        print("!!PLANNER INITIALIZED!!")
        
    @partial(jit, static_argnums=(0, ))
    def compute_rollout(self, controls, x_init, y_init, theta_init,  v_init, omega_init):

        v = controls[0:self.n]
        omega = controls[self.n : 2*self.n]

        v = v.at[0].set(v_init)
        omega = omega.at[0].set(omega_init)

        v = jnp.maximum( jnp.zeros(self.n), v  )

        theta = theta_init + jnp.cumsum(omega*self.delta_t)
        x = x_init + jnp.cumsum(v*jnp.cos(theta)*self.delta_t)
        y = y_init + jnp.cumsum(v*jnp.sin(theta)*self.delta_t)

        return x, y, v, omega

    @partial(jit, static_argnums=(0,))
    def compute_error_func(self, controls, 
                  x_init, y_init, theta_init, 
                  v_init, omega_init,
                  x_goal, y_goal,
                  x_obs, y_obs
                ): 
        
        x, y, v, omega = self.compute_rollout(controls, x_init, y_init, theta_init,  v_init, omega_init)

        error_goal_x = x[-1] - x_goal
        error_goal_y = y[-1] - y_goal

        error_velocity = jnp.diff(v)
        error_omega = jnp.diff(omega)

        # obstacle avoidance error calculation
        x_traj = x[:, jnp.newaxis]
        y_traj = y[:, jnp.newaxis]
        x_obst = x_obs[jnp.newaxis, :]
        y_obst = y_obs[jnp.newaxis, :]
        diffs_x = x_traj - x_obst
        diffs_y = y_traj - y_obst

        # sqrt because error function is everything inside the squares
        # this leads to nans in the trajectory when using the softplus from gradient_descent.py however....
        distances = jnp.sqrt(diffs_x**2 + diffs_y**2 - (0.5)**2)

        # softplus (from to gradient_descent) **is** smooth but sometimes produces nans in the trajectory
        # especially when beta is high. too low of a beta leads to crashing into the obstacles though.
        error_obstacle = (1/self.beta) * jnp.log(1 + jnp.exp(-self.beta * distances))

        # maximum is non smooth and makes the trajectory not return from a "bend" after avoiding an obstacle
        #error_obstacle = jnp.maximum(0.5 - distances, 0.0)
        
        error = jnp.hstack((
            self.w_goal * error_goal_x,
            self.w_goal * error_goal_y,
            self.w_velocity * error_velocity,
            self.w_omega * error_omega,
            self.w_obstacle * error_obstacle.ravel()
        ))

        return error

    @partial(jit, static_argnums=(0,))
    def compute_controls(self, x_init, y_init, theta_init, 
                v_init, omega_init,
                x_goal, y_goal,
                x_obs, y_obs,
                controls_init):
        
        def lax_gauss_newton(carry, idx):
            # from NLS notebook
            X_K, state = carry

            A = self.jac_func(X_K, *state)
            b = jnp.dot(A, X_K) - self.compute_error_func(X_K, *state)

            Q = jnp.dot(A.T, A) + (1/self.eta) * jnp.identity(2*self.n)
            q = -jnp.dot(A.T, b) - (1/self.eta) * X_K

            X_K_new = jnp.linalg.solve(Q, -q)
            
            error_value = self.compute_error_func(X_K_new, *state)
            cost_current = jnp.linalg.norm(error_value)

            return (X_K_new, state), cost_current
        
        X_init = controls_init
        
        state = (x_init, y_init, theta_init, 
                v_init, omega_init,
                x_goal, y_goal,
                x_obs, y_obs)
        
        carry_init = (X_init, state)
        carry_final, _ = jax.lax.scan(lax_gauss_newton, carry_init, jnp.arange(self.maxiter))
        
        controls_optimal = carry_final[0]
        v_optimal = controls_optimal[0:self.n]
        omega_optimal = controls_optimal[self.n:2*self.n]
        
        x_optimal, y_optimal, _, _ = self.compute_rollout(controls_optimal, x_init, y_init, theta_init, v_init, omega_init)
        traj_optimal = jnp.vstack((x_optimal, y_optimal)).T
        
        return v_optimal, omega_optimal, traj_optimal