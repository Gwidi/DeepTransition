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
    """CPG-RL implementation.

    Leg oscillators and gait matrices use the physical order FL, FR, RL, RR.
    The IK equations use a separate historical ID convention.
    """
    LEG_NAMES = ("fl", "fr", "rl", "rr")
    IK_LEG_ID = {"fr": 0, "fl": 1, "rr": 2, "rl": 3}
    LEG_LATERAL_SIGN = {"fl": 1.0, "fr": -1.0, "rl": 1.0, "rr": -1.0}
    def __init__(self,
          omega_swing=8*2*np.pi,
          omega_stance=2*2*np.pi, 
          gait="TROT",
          couple=True,
          coupling_strength=10,
          time_step=0.001,
          robot_height=0.28, 
          des_step_len=0.05,
          ground_clearance=0.05,
          ground_penetration=0.015,
          num_envs=1,
          device=None,
          rl_task_string=None,
          mu_low = 1.0,
          mu_up = 2.0,
          max_step_len = 0.16,
          num_CPGs = 4,
          spine_phase_mode = "phase_locked",
          spine_phase_source = "rr",
          spine_phase_offset = 0.0,
          max_spine_angle = 15.0 * np.pi / 180.0,
          spine_amplitude_mode = "policy",
          spine_fixed_amplitude = 10.0 * np.pi / 180.0,
          spine_control_mode = "cpg",
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
        self._spine_phase_mode = spine_phase_mode
        self._spine_phase_source = spine_phase_source
        self._spine_phase_offset = spine_phase_offset
        self._max_spine_angle = max_spine_angle
        self._spine_amplitude_mode = spine_amplitude_mode
        self._spine_fixed_amplitude = spine_fixed_amplitude
        self._spine_control_mode = spine_control_mode
        self._set_gait(gait)
        self.PHI_batch = self.PHI.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._coupling_mask = torch.ones(
            self.num_CPGs,
            self.num_CPGs,
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        if "SPINE" in self._rl_task_string and self.num_CPGs >= 5:
            self._coupling_mask[4, :] = 0.0
            self._coupling_mask[:, 4] = 0.0

        self.gait_names = ["TROT", "WALK", "PACE", "BOUND", "PRONK", "CANTER", "TRANSVERSE_GALLOP", "ROTARY_GALLOP", "AMBLE"]
        initial_gait_idx = self.gait_names.index(gait) if gait in self.gait_names else 0
        self.current_gait_indices = torch.full((self.num_envs,), initial_gait_idx, dtype=torch.long, device=device)
        self.previous_gait_indices = self.current_gait_indices.clone()
        
        self.X[:,0,:] = torch.rand(num_envs,self.num_CPGs,device=self._device) * .1
        self.X[:,1,:] = self.PHI[0,:] #* 0.0

        self._des_step_len = des_step_len

        self.robot_height_range = [0.26, 0.285] # min and max height of the CPG oscillation center
        self.ground_clearance_range = [0.035, 0.055]
        self.ground_penetration_range = [0.002, 0.006]
        self.offset_x_range = [-0.035, -0.005]

    def reset(self,env_ids):
        self._mu[env_ids,:] = 0
        self.X[env_ids,0,:] = torch.rand(len(env_ids),self.num_CPGs,device=self._device) * .1
        self.X[env_ids,1,:] = self.PHI[0,:] *0.0
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
        
        walk = torch.tensor([[ 0, -0.5, -0.75, -0.25 ], # FL, FR, RL, RR
                            [0.5, 0, -0.25, 0.25 ],
                            [0.75, 0.25, 0, 0.5 ],
                            [ 0.25, -0.25, -0.5, 0 ]],dtype=torch.float, device=device, requires_grad=False)

        trot = torch.tensor([[ 0, -0.5, -0.5, 0 ], # FL, FR, RL, RR
                            [0.5, 0, 0, 0.5 ],
                            [0.5, 0, 0, 0.5 ],
                            [ 0, -0.5, -0.5, 0 ]],dtype=torch.float, device=device, requires_grad=False)

        amble = torch.tensor([[ 0.0, -0.5, -0.8, -0.3 ],  # FL, FR, RL, RR
                            [ 0.5,  0.0, -0.3,  0.2 ],
                            [ 0.8,  0.3,  0.0,  0.5 ],
                            [ 0.3, -0.2, -0.5,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        pace = torch.tensor([[ 0.0, -0.5,  0.0, -0.5 ],  # FL, FR, RL, RR
                            [ 0.5,  0.0,  0.5,  0.0 ],
                            [ 0.0, -0.5,  0.0, -0.5 ],
                            [ 0.5,  0.0,  0.5,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        # bound: A=0, B=0.5, C=0.5
        bound = torch.tensor([[ 0.0,  0.0, -0.5, -0.5 ],  # FL, FR, RL, RR
                            [ 0.0,  0.0,  0.0, -0.5 ],
                            [ 0.5,  0.0,  0.0,  0.5 ],
                            [ 0.5,  0.5, -0.5,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        # pronk: A=0, B=0, C=0
        pronk = torch.tensor([[ 0.0,  0.0,  0.0,  0.0 ],  # FL, FR, RL, RR
                            [ 0.0,  0.0,  0.0,  0.0 ],
                            [ 0.0,  0.0,  0.0,  0.0 ],
                            [ 0.0,  0.0,  0.0,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        # canter: A=0.7, B=0.3, C=0
        canter = torch.tensor([[ 0.0, -0.7, -0.3,  0.0 ],  # FL, FR, RL, RR
                            [ 0.7,  0.0,  0.4, -0.7 ],
                            [ 0.3, -0.4,  0.0, -0.3 ],
                            [ 0.0,  0.7,  0.3,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        # transverse_gallop: A=-0.1, B=-0.5, C=-0.6
        transverse_gallop = torch.tensor([[ 0.0,  0.1,  0.5,  0.6 ],   # FL, FR, RL, RR
                            [-0.1,  0.0, -0.4, -0.5 ],
                            [-0.5,  0.4,  0.0, -0.1 ],
                            [-0.6,  0.5,  0.1,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)

        # rotary_gallop: A=0.1, B=-0.4, C=-0.5
        rotary_gallop = torch.tensor([[ 0.0, -0.1,  0.4,  0.5 ],   # FL, FR, RL, RR
                            [ 0.1,  0.0,  0.5,  0.6 ],
                            [-0.4, -0.5,  0.0,  0.1 ],
                            [-0.5, -0.6, -0.1,  0.0 ]],dtype=torch.float, device=device, requires_grad=False)
        trot = 2 * np.pi * trot
        walk = 2 * np.pi * walk
        amble = 2 * np.pi * amble
        pace = 2 * np.pi * pace
        bound = 2 * np.pi * bound
        pronk = 2 * np.pi * pronk
        canter = 2 * np.pi * canter
        transverse_gallop = 2 *np.pi * transverse_gallop
        rotary_gallop = 2 *np.pi * rotary_gallop

        # If SPINE mode, expand all gait matrices to 5x5 with spine coupling
        if "SPINE" in self._rl_task_string:
            gaits_4x4 = [trot, walk, pace, bound, pronk, canter, transverse_gallop, rotary_gallop, amble]
            expanded_gaits = []
            for g in gaits_4x4:
                expanded_gaits.append(self._expand_gait_with_spine(g))
            trot, walk, pace, bound, pronk, canter, transverse_gallop, rotary_gallop, amble = expanded_gaits

        self.available_gaits = [
            trot,
            walk,
            pace,
            bound,
            pronk,
            canter,
            transverse_gallop,
            rotary_gallop,
            amble
        ]


        self.PHI_trot = trot
        self.PHI_walk = walk
        self.PHI_pace = pace
        self.PHI_bound = bound
        self.PHI_pronk = pronk
        self.PHI_canter = canter
        self.PHI_rotary_gallop = rotary_gallop
        self.PHI_transverse_gallop = transverse_gallop
        self.PHI_amble = amble

        if gait == "TROT":
            print('TROT')
            self.PHI = self.PHI_trot
        elif gait == "ROTARY_GALLOP":
            print('ROTARY_GALLOP')
            self.PHI = self.PHI_rotary_gallop
        elif gait == "TRANSVERSE_GALLOP":
            print('TRANSVERSE_GALLOP')
            self.PHI = self.PHI_transverse_gallop
        elif gait == "WALK":
            print('WALK')
            self.PHI = self.PHI_walk
        elif gait == "AMBLE":
            self.PHI = self.PHI_amble
            print('AMBLE')
        elif gait == "PACE":
            self.PHI = self.PHI_pace
            print('PACE')
        elif gait == "BOUND":
            self.PHI = self.PHI_bound
            print('BOUND')
        elif gait == "PRONK":
            self.PHI = self.PHI_pronk
            print('PRONK')
        elif gait == "CANTER":
            self.PHI = self.PHI_canter
            print('CANTER')
        else:
            raise ValueError( gait + 'not implemented.')
        
    def _expand_gait_with_spine(self, gait_4x4):
        """Expand a 4x4 leg coupling matrix to 5x5 by adding spine (5th oscillator).

        The coupling mask disables all spine-leg terms. Zero entries alone are
        insufficient because they represent a desired zero phase difference.
        """
        device = self._device
        gait_5x5 = torch.zeros(5, 5, dtype=torch.float, device=device, requires_grad=False)
        
        # Copy the original 4x4 leg coupling
        gait_5x5[:4, :4] = gait_4x4
        
        return gait_5x5

    def _spine_phase_source_indices(self):
        source_to_indices = {
            "fl": [0],
            "fr": [1],
            "rl": [2],
            "rr": [3],
            "front_mean": [0, 1],
            "rear_mean": [2, 3],
            "left_diagonal": [0, 3],   # FL + RR
            "right_diagonal": [1, 2],  # FR + RL
            "all_mean": [0, 1, 2, 3],
        }
        if self._spine_phase_source not in source_to_indices:
            raise ValueError(f"Unsupported spine_phase_source: {self._spine_phase_source}")
        return source_to_indices[self._spine_phase_source]

    def _circular_mean_phase(self, phases):
        return torch.atan2(torch.mean(torch.sin(phases), dim=1), torch.mean(torch.cos(phases), dim=1))

    def _apply_spine_phase_controller(self, X_dot):
        if not ("SPINE" in self._rl_task_string and self.num_CPGs >= 5):
            return
        if self._spine_phase_mode == "uncoupled":
            return
        if self._spine_phase_mode != "phase_locked":
            raise ValueError(f"Unsupported spine_phase_mode: {self._spine_phase_mode}")

        source_indices = self._spine_phase_source_indices()
        source_phases = self.X[:, 1, source_indices]
        source_phase_vels = X_dot[:, 1, source_indices]
        self.X[:, 1, 4] = torch.remainder(
            self._circular_mean_phase(source_phases) + self._spine_phase_offset,
            2 * np.pi,
        )
        X_dot[:, 1, 4] = torch.mean(source_phase_vels, dim=1)

    def _phase_locked_spine_frequency(self, leg_frequencies):
        source_indices = self._spine_phase_source_indices()
        return torch.mean(leg_frequencies[:, source_indices], dim=1, keepdim=True)


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
        device = self._device 
        MAX_SPINE_ANGLE = torch.tensor(self._max_spine_angle, device=device) # radians
        a = torch.clip(actions, -1, 1)
        if "SPINE" in self._rl_task_string:
            if self._spine_control_mode == "direct":
                if a.shape[1] != 9:
                    raise ValueError(
                        f"Direct spine expects 9 actions "
                        f"(4 leg amplitudes + direct spine angle + 4 leg frequencies), got {a.shape[1]}"
                    )
                leg_mu = self._scale_helper(a[:, :4], MU_LOW**2, MU_UPP**2)
                spine_mu = torch.ones(a.shape[0], 1, dtype=torch.float, device=device) * MU_LOW**2
                self._mu = torch.cat((leg_mu, spine_mu), dim=1)
                leg_frequencies = self._scale_helper(
                    a[:, 5:9],
                    frequency_low[:, :4],
                    frequency_high[:, :4],
                ) * (2 * np.pi)
                spine_frequency = torch.mean(leg_frequencies, dim=1, keepdim=True)
                self._omega_residuals = torch.cat((leg_frequencies, spine_frequency), dim=1)
            else:
                self._mu = self._scale_helper(a[:,:5],MU_LOW**2,MU_UPP**2)
            if self._spine_control_mode != "direct" and self._spine_phase_mode == "phase_locked":
                if a.shape[1] != 9:
                    raise ValueError(
                        f"Phase-locked spine expects 9 actions "
                        f"(5 amplitudes + 4 leg frequencies), got {a.shape[1]}"
                    )
                leg_frequencies = self._scale_helper(
                    a[:, 5:9],
                    frequency_low[:, :4],
                    frequency_high[:, :4],
                ) * (2 * np.pi)
                spine_frequency = self._phase_locked_spine_frequency(leg_frequencies)
                self._omega_residuals = torch.cat((leg_frequencies, spine_frequency), dim=1)
            elif self._spine_control_mode != "direct":
                if a.shape[1] != 10:
                    raise ValueError(
                        f"Uncoupled spine expects 10 actions "
                        f"(5 amplitudes + 5 frequencies), got {a.shape[1]}"
                    )
                self._omega_residuals = self._scale_helper(
                    a[:, 5:10],
                    frequency_low,
                    frequency_high,
                ) * (2 * np.pi)
        else:
            self._mu = self._scale_helper(a[:,:4],MU_LOW**2, MU_UPP**2)
            self._omega_residuals = self._scale_helper(a[:,4:8],frequency_low,frequency_high) * (2*np.pi)
        

        self.integrate_oscillator_equations()
        
        x = torch.clip(self.X[:,0,:4],MU_LOW,MU_UPP)
        x = MAX_STEP_LEN * (x - MU_LOW) / (MU_UPP - MU_LOW)
        
        if "SPINE" in self._rl_task_string: 
            if self._spine_control_mode == "direct":
                sp = torch.clip(a[:, 4], -1.0, 1.0) * MAX_SPINE_ANGLE
            elif self._spine_amplitude_mode == "fixed":
                sp = torch.ones(self.num_envs, dtype=torch.float, device=device) * self._spine_fixed_amplitude
            elif self._spine_amplitude_mode == "policy":
                sp = torch.clip(self.X[:,0,4], MU_LOW, MU_UPP)
                sp = MAX_SPINE_ANGLE * (sp - MU_LOW) / (MU_UPP - MU_LOW)
            else:
                raise ValueError(f"Unsupported spine_amplitude_mode: {self._spine_amplitude_mode}")
        
            
 
        if "OFFSETX" in self._rl_task_string:
            x = -x * torch.cos(self.X[:,1,:4]) - self._offset_x
            y = self.y
        else:
            x = -x * torch.cos(self.X[:,1,:4]) 
            y = self.y  
        z = torch.where(torch.sin(self.X[:,1,:4]) > 0, 
                        -self._robot_height.unsqueeze(1) + self._ground_clearance.unsqueeze(1)   * torch.sin(self.X[:,1,:4]),
                        -self._robot_height.unsqueeze(1) + self._ground_penetration.unsqueeze(1) * torch.sin(self.X[:,1,:4]))
        if "SPINE" not in self._rl_task_string:
            sp = 0.0
        elif self._spine_control_mode != "direct":
            sp = sp * torch.sin(self.X[:,1,4])
    
        return x, y, z, sp

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

            coupling_effect = torch.zeros_like(self._omega_residuals)
            if self._couple:
                for i in range(self.num_CPGs):
                    coupling_effect[:,i] += torch.sum(
                        X[:,0,:]
                        * self._coupling_strength
                        * self._coupling_mask[i, :]
                        * torch.sin(
                            X[:,1,:]
                            - torch.remainder(self.X[:,1,i], (2*np.pi)).unsqueeze(1)
                            - self.PHI_batch[:,i,:]
                        ),
                        1,
                    )
            X_dot[:,1,:] = self._omega_residuals + coupling_effect
            X_dot[:,0,:] = X_dot_prev[:,0,:] + (d2X_prev[:,0,:] + d2X[:,0,:]) * dt / 2
            self.X = X + (X_dot_prev + X_dot) * dt / 2 
            self.X_dot = X_dot
            self.d2X = d2X 
            self.X[:,1,:] = torch.remainder(self.X[:,1,:], (2*np.pi))
            self._apply_spine_phase_controller(X_dot)

    # def compute_inverse_kinematics(self,robot,legID, x, y, z):
    #     z = -0.35
    #     z = torch.as_tensor(z, device=self._device, dtype=torch.float)
    #     x = 0.0
    #     x = torch.as_tensor(x, device=self._device, dtype=torch.float)

    #     l1 = robot.hip_link_length
    #     l2 = robot.thigh_link_length
    #     l3 = robot.calf_link_length

    #     D = (y**2 + (-z)**2 - l1**2 +
    #     (-x)**2 - l2**2 - l3**2) / (
    #              2 * l3 * l2)
    #     D = torch.clip(D, -1.0, 1.0)

    #     # check Right vs Left leg for hip angle
    #     sideSign = 1
    #     if legID == 0 or legID == 2:
    #         sideSign = -1

    #     knee_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
    #     if legID == 1 or legID == 3:
    #         knee_angle *= -1 # reversed tf 
    #     sqrt_component = y**2 + (-z)**2 - l1**2
    #     hip_roll_angle = -1*(-torch.atan2(z, y) - torch.atan2(
    #         torch.sqrt(sqrt_component), sideSign*l1*torch.ones_like(x)))
    #     hip_thigh_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) -1* torch.atan2(
    #         l3 * torch.sin(knee_angle),
    #         l2 + l3 * torch.cos(knee_angle))


    #     output= torch.stack([hip_roll_angle, hip_thigh_angle, knee_angle], dim=-1)

    #     return output
    
    def compute_inverse_kinematics(self,robot,legID, x, y, z):
        l1 = 0.0125
        l2 = robot.thigh_link_length
        l3 = robot.calf_link_length

        l1_x= 0.0555
        x_for_ik = x - l1_x

        dist_sq_yz = y**2 + z**2
        sqrt_component = dist_sq_yz - l1**2
        sqrt_component = torch.clip(sqrt_component, min=0.0)
        dist_yz = torch.sqrt(sqrt_component)

        D = (dist_yz**2 +
        x_for_ik**2 - l2**2 - l3**2) / (
                 2 * l3 * l2)
        D = torch.clip(D, -1.0, 1.0)

        # check Right vs Left leg for hip angle
        sideSign = 1
        if legID == 0 or legID == 2:
            sideSign = -1

        knee_angle = torch.acos(D) - 3.14159


        if legID == 1 or legID == 3:
            knee_angle *= -1 # reversed tf 
        
        hip_roll_angle = -1*(-torch.atan2(z, y) - torch.atan2(
            torch.sqrt(sqrt_component), sideSign*l1*torch.ones_like(x)))
        if legID >= 2:
            hip_roll_angle = -hip_roll_angle

        
        # 5. HIP THIGH (Pitch)
        # Alpha angle: inclination resulting from x and vertical distance
        alpha = torch.atan2(-x_for_ik, dist_yz)
        # Beta angle: correction resulting from thigh and shin triangle
        # We use acos because standard atan2 with your knee definition fails
        cos_beta = (l2**2 + (x_for_ik**2 + dist_yz**2) - l3**2) / (2 * l2 * torch.sqrt(x_for_ik**2 + dist_yz**2))
        cos_beta = torch.clip(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)
        
        if legID == 1 or legID == 3:
            hip_thigh_angle = alpha - beta
        else: 
            hip_thigh_angle = -alpha + beta


        output= torch.stack([hip_roll_angle, hip_thigh_angle, knee_angle], dim=-1)

        return output
    
    
    def random_resample_gaits(self, env_ids):
        """ Randomly resample gaits for the specified environments."""
        if len(env_ids) == 0:
            return
            
        # Available gaits 
        available = torch.stack(self.available_gaits) 
        
        # Randomly select new gait indices
        random_indices = torch.randint(0, len(self.available_gaits), (len(env_ids),), device=self._device)
        
        # Store previous gaits for transitions
        self.previous_gait_indices = self.current_gait_indices.clone()
        self.current_gait_indices[env_ids] = random_indices
        
        # Assign new matrices to the batch
        self.PHI_batch[env_ids] = available[random_indices]
