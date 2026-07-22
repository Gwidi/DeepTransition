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


from legged_gym.envs.base.base_config import BaseConfig

class SBActiveSpineRobotCfg(BaseConfig):
    class env:
        num_envs = 2048
        num_observations = 66
        num_privileged_obs = 73
        # Five CPG amplitudes (four legs + spine), followed by four leg frequencies
        # and four per-leg x-offset actions. The phase-locked spine inherits the
        # selected leg frequency.
        num_actions = 13
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        play = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.1 # [m]
        border_size = 0 # [m]
        curriculum = False
        curriculum_gap = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False

        measured_points_x = [  0.13, 0.18, 0.23, 0.28, 0.32, 0.37, 0.43, 0.48,0.52,0.57,0.62,0.67 ] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.23, -0.15, 0.,0.15, 0.23]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 6.
        terrain_width = 3.
        num_rows= 24 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class commands:
        curriculum = False
        gait_resampling_time = 3.0
        resample_gait_style = True
        save_data= False
        num_commands = 1 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5  # time before command are changed[s]
        heading_command = False #True # if true: compute ang vel command from heading error
        max_vel_x = 1.2
        freq_max= 8
        freq_low= 0
        class ranges:
            lin_vel_x = [ 0.2 , 1.2] # min max [m/s]
            lin_vel_y = [-0.0 , 0.0]   # min max [m/s]
            ang_vel_yaw = [-1e-7, 1e-7]    # min max [rad/s]
            heading = [-1e-7, 1e-7]


    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'sp_j0': 0.0,    # [rad]

            'fl_j0': 0.1,    # [rad]
            'rl_j0': -0.1,   # [rad]
            'fr_j0': -0.1 ,  # [rad]
            'rr_j0': 0.1,    # [rad]

            'fl_j1': -0.8,   # [rad]
            'rl_j1': -1.0,     # [rad]
            'fr_j1': 0.8,    # [rad]
            'rr_j1': 1.,     # [rad]

            'fl_j2': 1.5,    # [rad]
            'rl_j2': 1.5,   # [rad]
            'fr_j2': -1.5,   # [rad]
            'rr_j2': -1.5,    # [rad]
        }

    class control:
        control_type = 'CPG_OFFSETX_ACTION_SPINE'
        action_scale = 1.0

        decimation = 10
        stiffness = {'j': 71.23}  # [N*m/rad]
        damping = {'j': 1.86}     # [N*m*s/rad]

        # Spine CPG/controller options:
        # - "uncoupled": previous behavior; the policy controls an independent spine oscillator.
        # - "phase_locked": spine phase is anchored to a leg phase reference while amplitude
        #   can still come from the policy. Useful for testing whether coordination helps.
        spine_phase_mode = "phase_locked"  # uncoupled or phase_locked
        spine_phase_source = "rr"  # fl, fr, rl, rr, front_mean, rear_mean, left_diagonal, right_diagonal, all_mean
        spine_phase_offset = 0.0
        max_spine_angle = 0.2617993877991494  # 15 deg [rad]
        spine_amplitude_mode = "policy"  # policy or fixed
        spine_fixed_amplitude = 0.17453292519943295  # 10 deg [rad]
        spine_control_mode = "cpg"  # cpg or direct
        max_step_len = 0.16



    class asset:
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up. Check your model's orientation.

        #density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.013122
        thickness = 0.002

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/silver_badger/urdf/intention.urdf'
        # Accordingly to the URDF file:
        foot_name = "foot" 
        penalize_contacts_on = ["l1", "l2"]
        terminate_after_contacts_on = ["body", "l1", "rear"]

        self_collisions = 1 #
        hip_link_length = 0.0555
        thigh_link_length = 0.2083
        calf_link_length = 0.2
        spine_locked = False
        num_CPGs = 5 # 4 for legs only, 5 for legs + spine

    class domain_rand:
        latency =False
        alpha_latency =0.43
        randomize_PD = False
        stiffness_range = [30., 100.] # [N*m/rad]
        damping_range = [0.5, 2.0]     # [N*m*s/r
        randomize_friction = True
        friction_range = [0.3,1.0]
        randomize_base_mass = True 
        added_mass_range = [0., 5.] 
        push_robots = True 
        push_interval_s = 15 
        max_push_vel_xy = 0.05 
        lag_timesteps = 6
        randomize_lag_timesteps = False

    class rewards:
        class scales:
            tracking_lin_vel = 3.0 
            linear_velocity = -2.0
            angular_velocity = -0.1
            energy = -0.001
            feet_contact_forces = -0.01
            torque_saturation = 0.0
            dof_velocity_limits = 0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 180. # forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32 #25

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100 #4 #100.

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03 
            dof_vel = 1.5 
            lin_vel = 0.1
            ang_vel = 0.4
            gravity = 0.1 
            height_measurements = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [8, -6, 6]  # [m]
        lookat = [8., 2, 1.]  # [m]

    class sim:
        dt = 0.001 #0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.002  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.2 #0.5 [m/s]
            max_depenetration_velocity = 5.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class SBActiveSpineRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        normalize_obs = False
        normalize_clip = 10.0

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24#64 # per iteration
        max_iterations = 1500 # number of policy updates
        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'silver_badger_active_spine_phase_locked_9act'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        wandb_project ='silver_badger_active_spine'
        wandb_entity = 'cpg_put'
        wandb_group = ''
