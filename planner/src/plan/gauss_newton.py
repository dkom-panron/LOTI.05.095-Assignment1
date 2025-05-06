#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
#import matplotlib.pyplot as plt
import jax

class gauss_newton:
    def __init__(self, maxiter = 200, num_controls = 20):

        self.delta_t = 0.1
        self.n = num_controls
        self.maxiter = maxiter

        self.eta = 0.1 # nb, inverse of gradient descent eta

        self.w_goal = 0.25
        self.w_velocity = 0.95
        self.w_omega = 0.8
        self.w_smoothness_velocity = 0.5
        self.w_smoothness_omega = 0.5
        self.w_obstacle = 5.0

        self.w_trajectory = 0.2

        self.v_max = 1.5
        self.v_min = -0.1
        self.omega_max = 1.
        self.omega_min = - self.omega_max

        self.beta = 5.0

        self.jac_func = jit(jacfwd(self.compute_error_func, argnums = (0)))
        self.compute_obs_cost_batch = jit(vmap(self.compute_obs_cost, in_axes = (0,0,None,None)))
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
    def compute_obs_cost(self,x_obs,y_obs,x,y):

        obstacle = (x - x_obs)**2+(y-y_obs)**2-(.5)**2

        #cost_obstacle = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*obstacle))
        # cost_obstacle = jnp.maximum(0,-obstacle)

        return obstacle

    @partial(jit, static_argnums=(0,))
    def compute_vel_cost(self,v):
        g_1 = self.v_max -  v  
        cost_v_max = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_1))
        # cost_v_max = jnp.maximum(0,-g_1)

        g_2 = + v - self.v_min
        cost_v_min = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_2))
        # cost_v_min = jnp.maximum(0,-g_2)

        return cost_v_max + cost_v_min

    @partial(jit, static_argnums=(0,))
    def compute_omega_cost(self,omega):
        
        g_1 = self.omega_max - omega  
        cost_omega_max = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_1))
        # cost_omega_max = jnp.maximum(0,-g_1)
        
        g_2 = + omega - self.omega_min
        cost_omega_min = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_2))
        # cost_omega_min = jnp.maximum(0,-g_2)
        return cost_omega_max + cost_omega_min


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

        error_smoothness_velocity = jnp.diff(v)
        error_smoothness_omega = jnp.diff(omega)

        """
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
        """
    
        cost_obstacle_b = self.compute_obs_cost_batch(x_obs,y_obs,x,y)

        cost_obstacle_b = cost_obstacle_b.flatten()
        cost_obstacle = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*cost_obstacle_b))
        #print(f"{cost_obstacle_b.shape=}")
        #jax.debug.print("shape: {x}", x = cost_obstacle_b.shape)
        #cost_obstacle = jnp.sum(jnp.maximum(jnp.zeros(self.n), cost_obstacle_b))

        error_velocity = self.compute_vel_cost(v)
        error_omega = self.compute_omega_cost(omega)        

        c_goal_traj = ((x-x_goal)**2+(y-y_goal)**2)
        error_goal_traj = c_goal_traj

        error = jnp.hstack((
            self.w_goal * error_goal_x,
            self.w_goal * error_goal_y,
            self.w_smoothness_velocity * error_smoothness_velocity,
            self.w_smoothness_omega * error_smoothness_omega,
            self.w_obstacle * cost_obstacle,
            self.w_velocity * error_velocity,
            self.w_omega * error_omega,
            self.w_trajectory * error_goal_traj
        ))

        #jax.debug.print("error shape: {x}", x = error.shape)

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

"""
# # # ## Testing
planner = gauss_newton(maxiter = 50)

x_init = 0. 
y_init = 0. 
theta_init = 0.
v_init = 0. 
omega_init = 0. 
x_goal = 10. 
y_goal = 5.
obs_loc = 0.75
pcd = obs_loc*jnp.ones((200,2))
x_obs = pcd[:,0] 
y_obs = pcd[:,1] 

v_optimal, omega_optimal, traj_optimal = planner.compute_controls(
                    x_init, y_init, theta_init,
                    v_init, omega_init, 
                    x_goal, y_goal,
                    x_obs, y_obs, 0.01*jnp.ones(2*20))
a = jnp.concatenate((v_optimal,omega_optimal))

# print(traj_optimal.shape)

th = jnp.linspace(0, jnp.pi*2)
r_obs = 0.5
x_circ = obs_loc +r_obs*jnp.cos(th)
y_circ = obs_loc +r_obs*jnp.sin(th)

#plt.figure(1)
#plt.plot(traj_optimal[:,0], traj_optimal[:,1], '-om')
#plt.plot(x_circ, y_circ, '-k')
#plt.axis('equal')
#
#plt.show()
"""