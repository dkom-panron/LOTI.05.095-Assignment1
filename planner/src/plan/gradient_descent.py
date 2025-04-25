#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import time
import matplotlib.pyplot as plt
import jax

class gradient_descent:
	def __init__(self, maxiter = 200, num_controls = 20):

		self.delta_t = 0.1
		self.n = num_controls
		self.maxiter = maxiter

		self.gamma = 0.95
		self.eta = 0.002
		self.beta = 3.
		self.v_max = 1.5
		self.v_min = -0.1
		self.omega_max = 1.
		self.omega_min = - self.omega_max
		#------------------------------------------

		## For cost
		self.w_1 = .175	# Goal Cost - final waypt.
		self.w_2 = .175	# Goal Cost - entire rollout
		self.w_3 = 1.25	# Obstacle cost
		self.w_4 = .075	# Smoothness cost
		self.w_5 = .75	# Velocity cost
		self.w_6 = .5	# Omega cost
		#------------------------------------------

		self.compute_obs_cost_batch = jit(vmap(self.compute_obs_cost, in_axes = (0,0,None,None)))
		self.grad_func = jit(grad(self.compute_cost, argnums = (0)))
		print("!!PLANNER INITIALIZED!!")
		
	@partial(jit, static_argnums=(0, ))
	def compute_rollout(self, controls, x_init, y_init, theta_init,  v_init, omega_init):

		v = controls[0:self.n]
		omega = controls[self.n : 2*self.n]

		v = v.at[0].set(v_init)
		omega = omega.at[0].set(omega_init)

		# v = v.at[-1].set(0)
		# omega = omega.at[-1].set(0)

		v = jnp.maximum( jnp.zeros(self.n), v  )

		theta = theta_init + jnp.cumsum(omega*self.delta_t)
		x = x_init + jnp.cumsum(v*jnp.cos(theta)*self.delta_t)
		y = y_init +jnp.cumsum(v*jnp.sin(theta)*self.delta_t)

		return x, y, v, omega

	@partial(jit, static_argnums=(0,))
	def compute_obs_cost(self,x_obs,y_obs,x,y):

		obstacle = (x - x_obs)**2+(y-y_obs)**2-(.5)**2

		cost_obstacle = (1/self.beta)*jnp.log(1 + jnp.exp(-self.beta*obstacle))
		# cost_obstacle = jnp.maximum(0,-obstacle)

		return cost_obstacle

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
	def compute_cost(self, controls, 
				  x_init, y_init, theta_init, 
				  v_init, omega_init,
				  x_goal, y_goal,
				  x_obs, y_obs
				): 
		
		x, y, v, omega = self.compute_rollout(controls, x_init, y_init, theta_init,  v_init, omega_init)

		cost_goal_wpt = (x[-1]-x_goal)**2+(y[-1]-y_goal)**2

		c_goal_traj = ((x-x_goal)**2+(y-y_goal)**2)
		cost_goal_traj = jnp.sum(c_goal_traj)
		

		# for GD the obstacle cost function
		# is sum(max(0, f)), where f is
		# -(x-x0)^2 - (y-y0)^2 + d^2
		# for gauss newton, we simply remove the sum
		cost_obstacle_b = self.compute_obs_cost_batch(x_obs,y_obs,x,y)
		cost_obstacle = jnp.sum(jnp.maximum(jnp.zeros(self.n), cost_obstacle_b))
		
		cost_smoothness_b_v = jnp.diff(v)**2 
		cost_smoothness_b_o = jnp.diff(omega)**2
		cost_smoothness = jnp.sum(cost_smoothness_b_v) + jnp.sum(cost_smoothness_b_o)

		cost_velocity = jnp.sum(self.compute_vel_cost(v))
		cost_omega = jnp.sum(self.compute_omega_cost(omega))

		cost = self.w_1*cost_goal_wpt+self.w_2*cost_goal_traj+self.w_3*cost_obstacle+self.w_4*cost_smoothness + self.w_5*cost_velocity + self.w_6*cost_omega
		
		## Debugging
		# jax.debug.print("Cost Goal Wpt: {x}", x = cost_goal_wpt*self.w_1)
		# jax.debug.print("Cost Goal Traj: {x}", x = cost_goal_traj*self.w_2)
		# jax.debug.print("Cost Obstacle: {x}", x = cost_obstacle*self.w_3)
		# jax.debug.print("Cost Smoothness: {x}", x = cost_smoothness*self.w_4)

		return cost

	@partial(jit, static_argnums=(0,))
	def compute_controls(self, x_init, y_init, theta_init, 
				  v_init, omega_init,
				  x_goal, y_goal,
				  x_obs, y_obs,
				  controls_init):

		def lax_nesterov_grad_descent(carry, idx):
			
			X_K, V_K, state = carry

			# -------------------------------------
			### Nesterov accelerated gradient


			grad_vector = self.grad_func(X_K - self.gamma*V_K, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8])
			V_K_1 = self.gamma*V_K + grad_vector*self.eta
			X_K_1 = X_K - V_K_1
			# -------------------------------------

			cost_current = self.compute_cost(X_K_1, state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]) 
			
			return (X_K_1, V_K_1, state), (cost_current)
		
		# X_init = 0.01*jnp.ones(2*self.n)
		X_init = controls_init		# warm-starting
		V_init = jnp.zeros(2*self.n)

		state = (x_init, y_init, theta_init, 
				v_init, omega_init,
				x_goal, y_goal,
				x_obs, y_obs)
		
		carry_init = X_init, V_init, state
		carry_final, result = jax.lax.scan(lax_nesterov_grad_descent, carry_init, jnp.arange(self.maxiter))

		v_optimal = carry_final[0][0:self.n]
		omega_optimal = carry_final[0][self.n:2*self.n]
		x_optimal, y_optimal, _, _ = self.compute_rollout(carry_final[0], x_init, y_init, theta_init,  v_init, omega_init)
		traj_optimal = jnp.vstack((x_optimal,y_optimal)).T

		return v_optimal, omega_optimal, traj_optimal
		
# # # ## Testing
# planner = gradient_descent(maxiter = 50)

# x_init = 0. 
# y_init = 0. 
# theta_init = 0.
# v_init = 0. 
# omega_init = 0. 
# x_goal = 10. 
# y_goal = 5.
# obs_loc = 0.75
# pcd = obs_loc*jnp.ones((200,2))
# x_obs = pcd[:,0] 
# y_obs = pcd[:,1] 

# v_optimal, omega_optimal, traj_optimal = planner.compute_controls_nesterov(
#                     x_init, y_init, theta_init,
# 					v_init, omega_init, 
# 					x_goal, y_goal,
# 					x_obs, y_obs, 0.01*jnp.ones(2*20))
# a = jnp.concatenate((v_optimal,omega_optimal))

# # print(traj_optimal.shape)

# th = jnp.linspace(0, jnp.pi*2)
# r_obs = 0.5
# x_circ = obs_loc +r_obs*jnp.cos(th)
# y_circ = obs_loc +r_obs*jnp.sin(th)

# plt.figure(1)
# plt.plot(traj_optimal[:,0], traj_optimal[:,1], '-om')
# plt.plot(x_circ, y_circ, '-k')
# plt.axis('equal')

# plt.show()