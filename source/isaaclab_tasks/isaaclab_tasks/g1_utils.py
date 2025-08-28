# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consolidated utilities for G1 walking task.

This file contains all utility functions needed for the G1 humanoid robot
walking task, including package importing, configuration parsing, and hydra utilities.
"""

from __future__ import annotations

import collections
import functools
import gymnasium as gym
import importlib
import inspect
import os
import pkgutil
import re
import sys
import yaml
from collections.abc import Callable

# Import IsaacLab dependencies
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

# Try to import Hydra dependencies (optional)
try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False


# =============================================================================
# PACKAGE IMPORTING UTILITIES
# =============================================================================

def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the `blacklist_pkgs` argument to skip blacklisted packages.
    """
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)


# =============================================================================
# CONFIGURATION PARSING UTILITIES
# =============================================================================

def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports both YAML and Python configuration files.

    Args:
        task_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
        If the entry point is a Python class, it is instantiated and returned.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.
    """
    # obtain the configuration entry point
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        # get existing agents and algorithms
        agents = collections.defaultdict(list)
        for k in gym.spec(task_name.split(":")[-1]).kwargs:
            if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
                spec = (
                    k.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
        # raise error
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        if callable(cfg_cls):
            cfg = cfg_cls()
        else:
            cfg = cfg_cls
    return cfg


def parse_env_cfg(
    task_name: str, device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        device: The device to run the simulation on. Defaults to "cuda:0".
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object.

    Raises:
        RuntimeError: If the configuration for the task is not a class.
    """
    # load the default configuration
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")

    # check that it is not a dict
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    # simulation device
    cfg.sim.device = device
    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return cfg


def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run.
        other_dirs: The intermediate directories between the run directory and the checkpoint file.
        checkpoint: The regex expression for the model checkpoint file.
        sort_alpha: Whether to sort the runs by alphabetical order.

    Returns:
        The path to the model checkpoint.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.
    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)


# =============================================================================
# HYDRA CONFIGURATION UTILITIES (Optional)
# =============================================================================

def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store.

    This function requires Hydra to be installed.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.

    Raises:
        ImportError: If Hydra is not installed.
    """
    if not HAS_HYDRA:
        raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")
    
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    # store the configuration to Hydra
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator requires Hydra to be installed.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the environment's and agent's configurations updated from command line arguments.

    Raises:
        ImportError: If Hydra is not installed.
    """
    if not HAS_HYDRA:
        raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            # define the new Hydra main function
            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator
