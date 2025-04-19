#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
import matplotlib.pyplot as plt
import jax

class CEM:
	def __init__(self, maxiter = 200, num_controls = 20, num_samples = 50, percentage_elite = 0.1):

		self.delta_t = 0.1
		self.n = num_controls
		self.maxiter = maxiter

		self.num_samples = num_samples
		self.percentage_elite = percentage_elite

		## For cost
		self.goal_cost = 1.0
		self.smoothness_cost = 0.01
		self.obstacle_cost = 10.0

		## CEM initialization

		# normal CEM
		self.key = random.PRNGKey(0)
		self.key, subkey = random.split(self.key)

		self.init_mean = jnp.zeros(2*self.n)
		self.init_cov_v = 2*jnp.identity(self.n)
		self.init_cov_omega = jnp.identity(self.n)*0.5

		self.init_cov = jax.scipy.linalg.block_diag(self.init_cov_v, self.init_cov_omega)

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
	def compute_obs_cost_batch(self, x_obs, y_obs, x, y):
		# reshape for broadcasting: (num_obstacles, 1) vs (1, n)
		x_diff = x[None, :] - x_obs[:, None] # shape (num_obstacles, n)
		y_diff = y[None, :] - y_obs[:, None] # shape (num_obstacles, n)
		
		# assuming the obstacle is a circle with radius 0.5
		f_obs = -(x_diff**2 + y_diff**2) + (0.5)**2 # shape (num_obstacles, n)
		
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

		cost = \
			self.goal_cost*cost_goal_wpt \
			+ self.smoothness_cost*cost_smoothness \
			+ self.obstacle_cost*cost_obstacle
		
		return cost

	@partial(jit, static_argnums=(0,))
	def compute_controls(self, x_init, y_init, theta_init, 
				v_init, omega_init,
				x_goal, y_goal,
				x_obs, y_obs,
				controls_init):
		
		# inner function for lax scan
		def body_fun(carry, _):
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
		
		init_carry = (self.init_mean, self.init_cov, self.key)
		
		(final_mean, final_cov, final_key), _ = jax.lax.scan(
			body_fun, 
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
		
		return v, omega, trajectory
