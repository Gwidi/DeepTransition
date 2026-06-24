# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

from legged_gym.envs.silver_badger.sb_active_spine_config import (
    SBActiveSpineRobotCfg,
    SBActiveSpineRobotCfgPPO,
)
from legged_gym.envs.silver_badger.sb_rigid_spine_config import (
    SBRigidSpineRobotCfg,
    SBRigidSpineRobotCfgPPO,
)


class SBSpineRigidCfg(SBRigidSpineRobotCfg):
    pass


class SBSpineRigidCfgPPO(SBRigidSpineRobotCfgPPO):
    class runner(SBRigidSpineRobotCfgPPO.runner):
        experiment_name = "silver_badger_spine_ablation"
        run_name = "rigid"


class SBSpinePhaseLockedBaseCfg(SBActiveSpineRobotCfg):
    class env(SBActiveSpineRobotCfg.env):
        num_observations = 75
        num_privileged_obs = 82
        num_actions = 9

    class control(SBActiveSpineRobotCfg.control):
        spine_control_mode = "cpg"
        spine_phase_mode = "phase_locked"
        spine_phase_source = "rr"
        spine_amplitude_mode = "policy"


class SBSpinePhaseLockedBaseCfgPPO(SBActiveSpineRobotCfgPPO):
    class runner(SBActiveSpineRobotCfgPPO.runner):
        experiment_name = "silver_badger_spine_ablation"
        run_name = "phase_rr_0"


class SBSpinePhaseLockedNegHalfPiCfg(SBSpinePhaseLockedBaseCfg):
    class control(SBSpinePhaseLockedBaseCfg.control):
        spine_phase_offset = -0.5 * math.pi


class SBSpinePhaseLockedNegHalfPiCfgPPO(SBSpinePhaseLockedBaseCfgPPO):
    class runner(SBSpinePhaseLockedBaseCfgPPO.runner):
        run_name = "phase_rr_neg_half_pi"


class SBSpinePhaseLockedHalfPiCfg(SBSpinePhaseLockedBaseCfg):
    class control(SBSpinePhaseLockedBaseCfg.control):
        spine_phase_offset = 0.5 * math.pi


class SBSpinePhaseLockedHalfPiCfgPPO(SBSpinePhaseLockedBaseCfgPPO):
    class runner(SBSpinePhaseLockedBaseCfgPPO.runner):
        run_name = "phase_rr_half_pi"


class SBSpinePhaseLockedPiCfg(SBSpinePhaseLockedBaseCfg):
    class control(SBSpinePhaseLockedBaseCfg.control):
        spine_phase_offset = math.pi


class SBSpinePhaseLockedPiCfgPPO(SBSpinePhaseLockedBaseCfgPPO):
    class runner(SBSpinePhaseLockedBaseCfgPPO.runner):
        run_name = "phase_rr_pi"


class SBSpineUncoupledCfg(SBActiveSpineRobotCfg):
    class env(SBActiveSpineRobotCfg.env):
        num_observations = 76
        num_privileged_obs = 83
        num_actions = 10

    class control(SBActiveSpineRobotCfg.control):
        spine_control_mode = "cpg"
        spine_phase_mode = "uncoupled"
        spine_amplitude_mode = "policy"


class SBSpineUncoupledCfgPPO(SBActiveSpineRobotCfgPPO):
    class runner(SBActiveSpineRobotCfgPPO.runner):
        experiment_name = "silver_badger_spine_ablation"
        run_name = "uncoupled"


class SBSpineDirectCfg(SBActiveSpineRobotCfg):
    class env(SBActiveSpineRobotCfg.env):
        num_observations = 75
        num_privileged_obs = 82
        num_actions = 9

    class control(SBActiveSpineRobotCfg.control):
        spine_control_mode = "direct"
        spine_phase_mode = "uncoupled"
        spine_amplitude_mode = "policy"


class SBSpineDirectCfgPPO(SBActiveSpineRobotCfgPPO):
    class runner(SBActiveSpineRobotCfgPPO.runner):
        experiment_name = "silver_badger_spine_ablation"
        run_name = "direct_angle"
