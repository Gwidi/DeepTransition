# MIT License
# 
# Copyright (c) 2024 EPFL Biorobotics Laboratory (BioRob). 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from time import time
from warnings import WarningMessage
import numpy as np
import os
import torch
from isaacgym.torch_utils import *

class CPG_RL():
    """ CPG-RL Implementation. 
    IsaacGym order is FL, FR, RL, RR (alphabetical)
    """
    LEG_INDICES = np.array([1,0,3,2])
    def __init__(self,
          omega_swing=8*2*np.pi,
          omega_stance=2*2*np.pi, 
          gait="TROT", # TROT or GALLOP
          couple=True,
          coupling_strength=10,
          time_step=0.001,
          robot_height=0.3, 
          des_step_len=0.03,
          ground_clearance=0.07,
          ground_penetration=0.01,
          num_envs=1,
          device=None,
          rl_task_string=None,
          mu_low = 1.0,
          mu_up = 2.0,
          max_step_len = 0.03,
          num_CPGs = 4,
        ):
        self._rl_task_string = rl_task_string
        #global device
        self._device = device 
        self.num_CPGs = num_CPGs 
        self.X = torch.zeros(num_envs,2,self.num_CPGs,dtype=torch.float, device=device, requires_grad=False)
        self.X_dot = torch.zeros(num_envs,2,self.num_CPGs,dtype=torch.float, device=device, requires_grad=False)
        self.d2X = torch.zeros(num_envs,1,self.num_CPGs,dtype=torch.float, device=device, requires_grad=False)
        self.num_envs = num_envs
        self._mu = torch.zeros(num_envs,self.num_CPGs,dtype=torch.float, device=device, requires_grad=False)
        self._robot_height = torch.ones(num_envs,dtype=torch.float, device=device, requires_grad=False) * robot_height
        self._ground_clearance = torch.ones(num_envs,dtype=torch.float, device=device, requires_grad=False) * ground_clearance
        self._ground_penetration = torch.ones(num_envs,dtype=torch.float, device=device, requires_grad=False) * ground_penetration
        if "OFFSETX" in rl_task_string:
            self._offset_x= torch.zeros(num_envs,4,dtype=torch.float, device=device, requires_grad=False)
            self._offset_z = torch.zeros(num_envs,4,dtype=torch.float, device=device, requires_grad=False)
        self.y = torch.zeros(num_envs,4,dtype=torch.float, device=device, requires_grad=False)
        self.mu_low= mu_low,
        self.mu_up = mu_up,
        self.max_step_len = max_step_len,
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance  
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)
        
        self.X[:,0,:] = torch.rand(num_envs,self.num_CPGs,device=self._device) * .1
        self.X[:,1,:4] = self.PHI[0,:] * 0.0
        if "SPINE" in rl_task_string:
            self.X[:,1,4] = 0.0 # spine initial phase

        self._des_step_len = des_step_len

        self.robot_height_range = [0.18, 0.35] # min and max height of the CPG oscillation center
        self.ground_clearance_range = [0.02, 0.12]
        self.ground_penetration_range = [0.00, 0.015]
        self.offset_x_range = [-0.08, 0.03]

    def reset(self,env_ids):
        self._mu[env_ids,:] = 0
        self.X[env_ids,0,:] = torch.rand(len(env_ids),self.num_CPGs,device=self._device) * .1
        self.X[env_ids,1,:4] = self.PHI[0,:] #*0.0
        if "SPINE" in self._rl_task_string:
            self.X[env_ids,1,4] = 0.0 # spine initial phase
        self.X_dot[env_ids,:,:] = 0.
        self._resample_parameters(env_ids)

    def _resample_parameters(self, env_ids):
        """ Resample parameters h, xoff, gc, and gp  used by the CPG controller for some environments
        Args:
            env_ids (List[int]): Environments ids for which new parameters are needed
        """
        self._robot_height[env_ids] = torch_rand_float(self.robot_height_range[0], self.robot_height_range[1], (len(env_ids), 1), device=self._device).squeeze(1)
        self._ground_clearance[env_ids] = torch_rand_float(self.ground_clearance_range[0], self.ground_clearance_range[1], (len(env_ids), 1), device=self._device).squeeze(1)
        self._ground_penetration[env_ids] = torch_rand_float(self.ground_penetration_range[0], self.ground_penetration_range[1], (len(env_ids), 1), device=self._device).squeeze(1)
        self._offset_x[env_ids] = torch_rand_float(self.offset_x_range[0], self.offset_x_range[1], (len(env_ids), 4), device=self._device)

 

    def _set_gait(self,gait):
        device = self._device 

        trot = torch.tensor([[ 0, 0.5, 0.5, 0 ], # FL, FR, RL, RR
                            [-0.5, 0, 0,-0.5 ],
                            [-0.5, 0, 0,-0.5 ],
                            [ 0, 0.5, 0.5, 0 ]],dtype=torch.float, device=device, requires_grad=False)

        gallop = torch.tensor([[ 1, 1, 0, 0 ],
                                [0, 0, -1,-1 ],
                                [0, 0, -1,-1 ],
                                [ 1, 1, 0, 0 ]],dtype=torch.float, device=device, requires_grad=False)
        gallop = np.pi * gallop
        trot = np.pi * trot
        self.PHI_trot = trot
        self.PHI_gallop = gallop

        if gait == "TROT":
            print('TROT')
            self.PHI = self.PHI_trot
        elif gait == "GALLOP":
            print('GALLOP')
            self.PHI = self.PHI_gallop
        else:
            raise ValueError( gait + 'not implemented.')


    def update(self):
        """ Update oscillator states. """
        device = self._device 

        # update parameters, integrate
        self._integrate_hopf_equations()

        # map CPG variables to Cartesian foot xz positions
        x = self.X[:,0,:4] * torch.cos(self.X[:,1,:4]) 
        z = torch.where(torch.sin(self.X[:,1,:4]) > 0, 
                        -self._robot_height.unsqueeze(1) + self._ground_clearance.unsqueeze(1)*torch.sin(self.X[:,1,:4]),# swing)
                        -self._robot_height.unsqueeze(1) + self._ground_penetration.unsqueeze(1)*torch.sin(self.X[:,1,:4]))

        return -self._des_step_len * x, z

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits. 
        This needs to be made general in case action range is not [-1,1]
        """
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        # verify clip
        new_a = torch.clip(new_a, lower_lim, upper_lim)
        return new_a

    def get_CPG_RL_actions(self, actions, frequency_high, frequency_low, normal_forces):
        """ Map RL actions to CPG signals """
        MU_LOW = self.mu_low[0]
        MU_UPP = self.mu_up[0]
        MAX_STEP_LEN = self.max_step_len[0]
        MAX_SPINE_ANGLE = torch.tensor(5.0 * np.pi / 180) # 15° in radians
  
        device = self._device 
        a = torch.clip(actions, -1, 1)
        if "SPINE" in self._rl_task_string:
            self._mu = self._scale_helper(a[:,:5],MU_LOW**2,MU_UPP**2) 
            self._omega_residuals = self._scale_helper(a[:,5:10],frequency_low,frequency_high)
        else:
            self._mu = self._scale_helper(a[:,:4],MU_LOW**2,MU_UPP**2) 
            self._omega_residuals = self._scale_helper(a[:,4:8],frequency_low,frequency_high)
        

        self.integrate_oscillator_equations()
        
        x = torch.clip(self.X[:,0,:4],MU_LOW,MU_UPP)
        if "SPINE" in self._rl_task_string: 
            sp = torch.clip(self.X[:,0,4], MU_LOW, MU_UPP) 
        x = MAX_STEP_LEN * (x - MU_LOW) / (MU_UPP - MU_LOW)
        if "SPINE" in self._rl_task_string:
            sp = MAX_SPINE_ANGLE * (sp - MU_LOW) / (MU_UPP - MU_LOW)
 
        if "OFFSETX" in self._rl_task_string:
            x = -x * torch.cos(self.X[:,1,:4]) - self._offset_x
            y = self.y
        else:
            x = -x * torch.cos(self.X[:,1,:4]) 
            y = self.y  
        z = torch.where(torch.sin(self.X[:,1,:4]) > 0, 
                        -self._robot_height.unsqueeze(1) + self._ground_clearance.unsqueeze(1)   * torch.sin(self.X[:,1,:4]),
                        -self._robot_height.unsqueeze(1) + self._ground_penetration.unsqueeze(1) * torch.sin(self.X[:,1,:4]))
        if "SPINE" in self._rl_task_string:
            sp = sp * torch.sin(self.X[:,1,4])
        else: 
            sp = 0.0
    
        return x, y, z

    def integrate_oscillator_equations(self):
        device = self._device 
        X_dot = self.X_dot.clone() 
        d2X = self.d2X.clone()
        _a = 150
        dt = 0.001 
        for _ in range(int(self._dt/dt)):
            d2X_prev = self.d2X.clone()
            X_dot_prev = self.X_dot.clone()
            X = self.X.clone()

            d2X = (_a * ( _a/4 * (torch.sqrt(self._mu) - X[:,0,:]) - X_dot_prev[:,0,:] )).unsqueeze(1)
            if self._couple:
                for i in range(4):
                    self._omega_residuals[:,i] += torch.sum(   X[:,0,:] * self._coupling_strength * torch.sin(X[:,1,:] - torch.remainder(self.X[:,1,i], (2*np.pi)).unsqueeze(1) - self.PHI[i,:].unsqueeze(0) ) , 1)
            X_dot[:,1,:] = self._omega_residuals
            X_dot[:,0,:] = X_dot_prev[:,0,:] + (d2X_prev[:,0,:] + d2X[:,0,:]) * dt / 2
            self.X = X + (X_dot_prev + X_dot) * dt / 2 
            self.X_dot = X_dot
            self.d2X = d2X 
            self.X[:,1,:] = torch.remainder(self.X[:,1,:], (2*np.pi))

    def compute_inverse_kinematics(self,robot,legID, x, y, z):
        l1 = robot.hip_link_length
        l2 = robot.thigh_link_length
        l3 = robot.calf_link_length

        D = (y**2 + (-z)**2 - l1**2 +
        (-x)**2 - l2**2 - l3**2) / (
                 2 * l3 * l2)
        D = torch.clip(D, -1.0, 1.0)

        # check Right vs Left leg for hip angle
        sideSign = 1
        if legID == 0 or legID == 2:
            sideSign = -1

        knee_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
        # if legID == 1 or legID == 3:
        #     knee_angle *= -1 # reversed tf 
        sqrt_component = y**2 + (-z)**2 - l1**2
        hip_roll_angle = -1*(-torch.atan2(z, y) - torch.atan2(
            torch.sqrt(sqrt_component), sideSign*l1*torch.ones_like(x)))
        hip_thigh_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) -1* torch.atan2(
            l3 * torch.sin(knee_angle),
            l2 + l3 * torch.cos(knee_angle)) 
        output= torch.stack([hip_roll_angle, hip_thigh_angle, knee_angle], dim=-1)  
        return output
