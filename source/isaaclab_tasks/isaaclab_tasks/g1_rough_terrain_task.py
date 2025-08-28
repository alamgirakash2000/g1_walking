# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Humanoid Robot Rough Terrain Walking Task
============================================

This module contains the complete configuration for training and testing 
the G1 humanoid robot to walk on rough terrain using RSL-RL PPO algorithm.

This single file replaces the entire complex directory structure and provides:
- Environment configuration (scene, observations, actions, rewards, etc.)
- Training algorithm configuration  
- Both training and play modes

Usage:
  Training: --task=Isaac-Velocity-Rough-G1-v0
  Testing:  --task=Isaac-Velocity-Rough-G1-v0 (with trained model)
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import MDP functions and configs
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_assets import G1_MINIMAL_CFG

# Import RSL-RL configurations
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# Scene Configuration
##

@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 rough terrain walking."""

    # Rough terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    # G1 robot
    robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Height scanner for terrain awareness
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    # Contact force sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )
    
    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP Configuration
##

@configclass
class G1CommandsCfg:
    """Velocity commands for G1 robot."""
    
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),     # Forward velocity
            lin_vel_y=(-0.0, 0.0),    # Side velocity (disabled)
            ang_vel_z=(-1.0, 1.0),    # Turning velocity
            heading=(-math.pi, math.pi)
        ),
    )


@configclass
class G1ActionsCfg:
    """Joint position actions for G1 robot."""
    
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True
    )


@configclass
class G1ObservationsCfg:
    """Observations for G1 robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for G1 walking."""

        # Robot state observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        # Command observations
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # Joint observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        # Terrain observations
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1RewardsCfg:
    """Reward configuration optimized for G1 humanoid walking."""

    # Primary rewards - velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=2.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # Stability rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    
    # Penalty terms
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)  # Vertical velocity penalty
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # Roll/pitch angular velocity
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7)  # Joint torque penalty
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)  # Joint acceleration penalty
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)  # Action smoothness
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # Stay upright
    
    # Joint limit and deviation penalties
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    
    # G1-specific joint deviation penalties
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint", ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
                    ".*_zero_joint", ".*_one_joint", ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
    
    # Contact penalties
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    # Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1TerminationsCfg:
    """Termination conditions for G1 walking."""
    
    # Time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Robot fell (torso contact)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class G1EventsCfg:
    """Event configuration for domain randomization."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Reset events
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1CurriculumCfg:
    """Curriculum for progressive difficulty."""
    
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment Configuration
##

@configclass 
class G1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """
    Complete environment configuration for G1 humanoid rough terrain walking.
    
    This replaces the entire complex directory structure with a single,
    well-organized configuration class.
    """
    
    # Scene configuration
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    
    # MDP configuration  
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventsCfg = G1EventsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()

    def __post_init__(self):
        """Post initialization configuration."""
        # Simulation settings
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # Update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Enable curriculum for terrain generation
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    """Environment configuration for testing/playing with trained models."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for testing
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # Spawn robots randomly instead of by terrain level
        self.scene.terrain.max_init_terrain_level = None
        
        # Reduce terrain complexity for testing
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Fixed forward velocity for testing
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        # Disable noise and randomization for testing
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


##
# Training Configuration
##

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    RSL-RL PPO training configuration optimized for G1 rough terrain walking.
    
    This replaces the separate training configuration files.
    """
    
    # Training parameters
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    empirical_normalization = False
    
    # Neural network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # Actor network: 310 -> 512 -> 256 -> 128 -> 37
        critic_hidden_dims=[512, 256, 128], # Critic network: 310 -> 512 -> 256 -> 128 -> 1  
        activation="elu",
    )
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Configuration registration for Isaac Lab task system
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

# Register the task configurations
gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": G1RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1RoughPPORunnerCfg,
    },
)
