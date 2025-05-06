#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
#import matplotlib.pyplot as plt
import jax

class CEM:
    def __init__(self, maxiter = 200, num_controls = 20, num_samples = 50, percentage_elite = 0.1, stomp_like=False):

        self.delta_t = 0.1
        self.n = num_controls
        self.maxiter = maxiter

        self.num_samples = num_samples
        self.percentage_elite = percentage_elite

        ## For cost
        self.goal_cost = 1.0
        self.smoothness_cost = 0.1
        self.obstacle_cost = 8.0
        self.obstacle_radius = 1.0
        self.velocity_cost = 0.2
        self.omega_cost = 0.2

        self.v_max = 1.5
        self.v_min = -0.1
        self.omega_max = 1.0
        self.omega_min = -self.omega_max

        ## CEM initialization

        self.key = random.PRNGKey(0)
        self.key, subkey = random.split(self.key)
        #self.init_mean = jnp.zeros(2*self.n)
        # normal CEM
        if not stomp_like:
            self.init_cov_v = 2*jnp.identity(self.n)
            self.init_cov_omega = jnp.identity(self.n)*0.5

            self.init_cov = jax.scipy.linalg.block_diag(self.init_cov_v, self.init_cov_omega)
        else:
            # "stomp-like" for smoothness
            A = np.diff(np.diff(np.identity(self.n), axis = 0), axis = 0)

            temp_1 = np.zeros(self.n)
            temp_2 = np.zeros(self.n)
            temp_3 = np.zeros(self.n)
            temp_4 = np.zeros(self.n)

            temp_1[0] = 1.0
            temp_2[0] = -2
            temp_2[1] = 1
            temp_3[-1] = -2
            temp_3[-2] = 1

            temp_4[-1] = 1.0

            A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))

            R = np.dot(A_mat.T, A_mat)
            mu = np.zeros(self.n) # not needed
            cov = np.linalg.pinv(R)
            self.init_cov = jax.scipy.linalg.block_diag(0.001*cov, 0.0003*cov)



        self.compute_cost_batch = jit(vmap(self.compute_cost,
                                     in_axes=(0,None,None,None,None,None,
                                                None,None,None,None)))

        print("!!PLANNER INITIALIZED!!")
        
    @partial(jit, static_argnums=(0, ))
    def compute_rollout(self, controls, x_init, y_init, theta_init, v_init, omega_init):

        v = controls[0:self.n]
        omega = controls[self.n : 2*self.n]

        v = v.at[0].set(v_init)
        omega = omega.at[0].set(omega_init)

        v = v.at[-1].set(0)
        omega = omega.at[-1].set(0)

        #print(f"compute_rollout {v.shape=}")
        v = jnp.maximum( jnp.zeros(self.n), v  )

        theta = theta_init + jnp.cumsum(omega*self.delta_t)
        x = x_init + jnp.cumsum(v*jnp.cos(theta)*self.delta_t)
        y = y_init + jnp.cumsum(v*jnp.sin(theta)*self.delta_t)

        return x, y, v, omega

    @partial(jit, static_argnums=(0,))
    def compute_vel_cost(self,v):
        g_1 = self.v_max -  v  
        #cost_v_max = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_1))
        cost_v_max = jnp.maximum(0,-g_1)

        g_2 = + v - self.v_min
        #cost_v_min = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_2))
        cost_v_min = jnp.maximum(0,-g_2)

        return cost_v_max + cost_v_min

    @partial(jit, static_argnums=(0,))
    def compute_omega_cost(self,omega):
        
        g_1 = self.omega_max - omega  
        #cost_omega_max = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_1))
        cost_omega_max = jnp.maximum(0,-g_1)
        
        g_2 = + omega - self.omega_min
        #cost_omega_min = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*g_2))
        cost_omega_min = jnp.maximum(0,-g_2)
        return cost_omega_max + cost_omega_min


    @partial(jit, static_argnums=(0,))
    def compute_obs_cost_batch(self, x_obs, y_obs, x, y):
        # reshape for broadcasting: (num_obstacles, 1) vs (1, n)
        x_diff = x[None, :] - x_obs[:, None] # shape (num_obstacles, n)
        y_diff = y[None, :] - y_obs[:, None] # shape (num_obstacles, n)
        
        f_obs = -(x_diff**2 + y_diff**2) + (self.obstacle_radius)**2 # shape (num_obstacles, n)
        
        c_obs = jnp.sum(jnp.maximum(0, f_obs), axis=1) # shape (num_obstacles,)

        return c_obs

    @partial(jit, static_argnums=(0,))
    def compute_cost(self, controls, 
                  x_init, y_init, theta_init, 
                  v_init, omega_init,
                  x_goal, y_goal,
                  x_obs, y_obs
                ): 
        
        x, y, v, omega = self.compute_rollout(
            controls,
            x_init,
            y_init,
            theta_init,
            v_init, omega_init
        )

        cost_goal_wpt = (x[-1]-x_goal)**2+(y[-1]-y_goal)**2

        #c_goal_traj = ((x-x_goal)**2+(y-y_goal)**2)
        #cost_goal_traj = jnp.sum(c_goal_traj)

        cost_obstacle_per_obs = self.compute_obs_cost_batch(x_obs, y_obs, x, y)
        cost_obstacle = jnp.sum(cost_obstacle_per_obs)
        
        cost_smoothness_b_v = jnp.diff(v)**2 
        cost_smoothness_b_o = jnp.diff(omega)**2
        cost_smoothness = jnp.sum(cost_smoothness_b_v) + jnp.sum(cost_smoothness_b_o)

        cost_velocity = jnp.sum(self.compute_vel_cost(v))
        cost_omega = jnp.sum(self.compute_omega_cost(omega))

        cost = \
            self.goal_cost*cost_goal_wpt \
            + self.smoothness_cost*cost_smoothness \
            + self.obstacle_cost*cost_obstacle \
            + self.velocity_cost*cost_velocity \
            + self.omega_cost*cost_omega
        
        return cost

    @partial(jit, static_argnums=(0,))
    def compute_controls(self, x_init, y_init, theta_init, 
                v_init, omega_init,
                x_goal, y_goal,
                x_obs, y_obs,
                mean_init):
        
        # inner function for lax scan
        def lax_cem(carry, _):
            mean, cov, key = carry
            key, subkey = random.split(key)
            
            controls = jax.random.multivariate_normal(subkey, mean, cov, (self.num_samples,))
            
            cost_samples = self.compute_cost_batch(
                controls,
                x_init, y_init, theta_init, 
                v_init, omega_init,
                x_goal, y_goal,
                x_obs, y_obs
            )
            
            idx = jnp.argsort(cost_samples)
            elite_num = int(self.percentage_elite * self.num_samples)
            controls_elite = controls[idx[:elite_num]]
            
            new_mean = jnp.mean(controls_elite, axis=0)
            new_cov = jnp.cov(controls_elite.T) + 1e-3*jnp.identity(2*self.n)
            
            return (new_mean, new_cov, key), None
        
        init_carry = (mean_init, self.init_cov, self.key)
        
        (final_mean, final_cov, final_key), _ = jax.lax.scan(
            lax_cem, 
            init_carry, 
            jnp.arange(self.maxiter)
        )
        
        key, subkey = random.split(final_key)
        controls = jax.random.multivariate_normal(subkey, final_mean, final_cov, (self.num_samples,))
        cost_samples = self.compute_cost_batch(
            controls,
            x_init, y_init, theta_init, 
            v_init, omega_init,
            x_goal, y_goal,
            x_obs, y_obs
        )
        
        controls_best = controls[jnp.argmin(cost_samples)]
        x, y, v, omega = self.compute_rollout(controls_best, x_init, y_init, theta_init, v_init, omega_init)
        trajectory = jnp.vstack((x,y)).T
        
        return v, omega, trajectory, final_mean
