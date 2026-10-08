# Base environment
from rl4co.envs.common.base import RL4COEnvBase

# Routing
from rl4co.envs.routing import (
    EVRPEnv,
    EVRPTWEnv,
    UnifiedVRPEnv,
)

# Register environments
ENV_REGISTRY = {
    "evrp": EVRPEnv,
    "evrptw": EVRPTWEnv,
    "unified_vrp": UnifiedVRPEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
