# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.quadruped import Quadruped
from .base.quadruped_with_spine import QuadrupedWithSpine
from .silver_badger.sb_active_spine_config import SBActiveSpineRobotCfg, SBActiveSpineRobotCfgPPO
from .silver_badger.sb_rigid_spine_config import SBRigidSpineRobotCfg, SBRigidSpineRobotCfgPPO
from .silver_badger.sb_spine_ablation_configs import (
    SBSpineDirectCfg,
    SBSpineDirectCfgPPO,
    SBSpinePhaseLockedBaseCfg,
    SBSpinePhaseLockedBaseCfgPPO,
    SBSpinePhaseLockedHalfPiCfg,
    SBSpinePhaseLockedHalfPiCfgPPO,
    SBSpinePhaseLockedNegHalfPiCfg,
    SBSpinePhaseLockedNegHalfPiCfgPPO,
    SBSpinePhaseLockedPiCfg,
    SBSpinePhaseLockedPiCfgPPO,
    SBSpineRigidCfg,
    SBSpineRigidCfgPPO,
    SBSpineUncoupledCfg,
    SBSpineUncoupledCfgPPO,
)
from .base.quadruped_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.utils.task_registry import task_registry
task_registry.register( "silver_badger_rigid_spine", QuadrupedWithSpine, SBRigidSpineRobotCfg(), SBRigidSpineRobotCfgPPO())
task_registry.register( "silver_badger_active_spine", QuadrupedWithSpine, SBActiveSpineRobotCfg(), SBActiveSpineRobotCfgPPO())
task_registry.register( "silver_badger_spine_ablation_rigid", QuadrupedWithSpine, SBSpineRigidCfg(), SBSpineRigidCfgPPO())
task_registry.register( "silver_badger_spine_phase_rr_0", QuadrupedWithSpine, SBSpinePhaseLockedBaseCfg(), SBSpinePhaseLockedBaseCfgPPO())
task_registry.register( "silver_badger_spine_phase_rr_neg_half_pi", QuadrupedWithSpine, SBSpinePhaseLockedNegHalfPiCfg(), SBSpinePhaseLockedNegHalfPiCfgPPO())
task_registry.register( "silver_badger_spine_phase_rr_half_pi", QuadrupedWithSpine, SBSpinePhaseLockedHalfPiCfg(), SBSpinePhaseLockedHalfPiCfgPPO())
task_registry.register( "silver_badger_spine_phase_rr_pi", QuadrupedWithSpine, SBSpinePhaseLockedPiCfg(), SBSpinePhaseLockedPiCfgPPO())
task_registry.register( "silver_badger_spine_uncoupled", QuadrupedWithSpine, SBSpineUncoupledCfg(), SBSpineUncoupledCfgPPO())
task_registry.register( "silver_badger_spine_direct", QuadrupedWithSpine, SBSpineDirectCfg(), SBSpineDirectCfgPPO())
task_registry.register( "quadruped", Quadruped, LeggedRobotCfg(), LeggedRobotCfgPPO())
