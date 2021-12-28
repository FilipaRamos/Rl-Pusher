from envs.mainEnv import MainEnv
from envs.deepPusherEnv import DeepPusherEnv

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='deepPusher-v0',
    entry_point='envs:DeepPusherEnv',
    # More arguments here
)