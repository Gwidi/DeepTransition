# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.quadruped import Quadruped
from .base.quadruped_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .honey_badger.hb_config import HBRobotCfg, HBRobotCfgPPO
from .silver_badger.sb_config import SBRobotCfg, SBRobotCfgPPO
from legged_gym.utils.task_registry import task_registry
task_registry.register( "quadruped", Quadruped, LeggedRobotCfg(), LeggedRobotCfgPPO())
task_registry.register( "honey_badger", Quadruped, HBRobotCfg(), HBRobotCfgPPO())
task_registry.register( "silver_badger", Quadruped, SBRobotCfg(), SBRobotCfgPPO())

