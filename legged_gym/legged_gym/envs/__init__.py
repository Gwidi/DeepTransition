# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.quadruped import Quadruped
from .base.quadruped_with_spine import QuadrupedWithSpine
from .silver_badger.sb_active_spine_config import SBActiveSpineRobotCfg, SBActiveSpineRobotCfgPPO
from .silver_badger.sb_rigid_spine_config import SBRigidSpineRobotCfg, SBRigidSpineRobotCfgPPO
from legged_gym.utils.task_registry import task_registry
task_registry.register( "silver_badger_rigid_spine", QuadrupedWithSpine, SBRigidSpineRobotCfg(), SBRigidSpineRobotCfgPPO())
task_registry.register( "silver_badger_active_spine", QuadrupedWithSpine, SBActiveSpineRobotCfg(), SBActiveSpineRobotCfgPPO())

