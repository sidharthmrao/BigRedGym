"""Current deployment utilities, independent of the Unitree SDK/DDS stack.

The CPU parity check covers sensor, command, and phase observations. Applied
residual-action parity needs separate work: deployment currently clips residuals
against absolute joint limits before exposing them as observations.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility
from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import select_backend, task_registry

SDK_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
SDK_IN_QGYM_ORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def _controller(cfg=None):
    if cfg is None:
        cfg = DeployConfig()
    return SimpleNamespace(
        cfg=cfg,
        obs_vec_size=sum(cfg.obs_sizes[name] for name in cfg.obs_vector),
        last_command=torch.tensor([0.5, -0.25, 0.75, 0.4]),
        last_action=torch.zeros(2, 12),
        phase=0.4,
        _gait_reference=torch.linspace(-0.3, 0.3, 12),
    )


def _message():
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(i), dq=-2.0 * i, ddq=3.0 * i) for i in range(12)
        ],
        imu_state=SimpleNamespace(
            quaternion=[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            gyroscope=[0.1, -0.2, 0.3],
        ),
    )


def test_joint_mapping_for_positions_velocities_and_accelerations():
    controller, message = _controller(), _message()
    expected = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_pos_obs(controller, message), expected
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_vel(controller, message), -2 * expected
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_dof_accel(controller, message), 3 * expected
    )
    canonical = torch.arange(12)
    sdk = canonical[deploy_utility.QGYM_TO_UNITREE_JOINT_IDX]
    assert sdk.tolist() == SDK_IN_QGYM_ORDER
    assert torch.equal(sdk[deploy_utility.UNITREE_TO_QGYM_JOINT_IDX], canonical)


def test_wxyz_quaternion_projects_gravity_but_gyro_is_already_body_local():
    controller, message = _controller(), _message()
    # A positive quarter-turn about X maps world-down to body negative Y.
    torch.testing.assert_close(
        deploy_utility._get_obs_projected_gravity(controller, message),
        torch.tensor([0.0, -1.0, 0.0]),
        atol=2e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        deploy_utility._get_obs_base_ang_vel(controller, message),
        torch.tensor([0.1, -0.2, 0.3]),
    )


def test_observation_dispatch_preserves_configured_order_sizes_and_scaling():
    cfg = DeployConfig()
    cfg.obs_vector = [
        "phase_frequency",
        "dof_vel",
        "base_ang_vel",
        "dof_pos_target",
        "commands",
        "projected_gravity",
        "dof_pos_obs",
        "dof_accel",
        "phase_obs",
    ]
    cfg.DeployScaling = SimpleNamespace(
        phase_frequency=4.0,
        dof_vel=2.0,
        base_ang_vel=0.5,
        dof_pos_target=0.25,
        commands=[2.0, 4.0, 8.0, 0.5],
        projected_gravity=2.0,
        dof_pos_obs=list(range(1, 13)),
        dof_accel=3.0,
        phase_obs=2.0,
    )
    controller = _controller(cfg)
    controller.last_action[0] = 0.5 * (cfg.lower_joint_limit + cfg.upper_joint_limit)
    ordered = torch.tensor(SDK_IN_QGYM_ORDER, dtype=torch.float)
    expected = torch.cat(
        (
            torch.tensor([DeployConfig.phase_frequency / 4]),
            -ordered,
            torch.tensor([0.2, -0.4, 0.6]),
            controller.last_action[0] / 0.25,
            torch.tensor([0.25, -0.0625, 0.09375, 0.8]),
            torch.tensor([0.0, -0.5, 0.0]),
            ordered / torch.arange(1, 13),
            ordered,
            torch.tensor([math.sin(0.4), math.cos(0.4)]) / 2,
        )
    )

    actual = deploy_utility.lowstate_to_obs(controller, _message())

    assert actual.shape == (controller.obs_vec_size,)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


def test_trot_action_target_roundtrip_includes_default_pose_and_gait():
    controller = _controller()
    controller.cfg.default_dof_pos = torch.linspace(0.1, 1.2, 12)
    action = torch.linspace(-0.2, 0.2, 12)

    target = deploy_utility.action_to_target_pos(controller, action)

    torch.testing.assert_close(
        target,
        action + controller.cfg.default_dof_pos + controller._gait_reference,
    )
    torch.testing.assert_close(
        deploy_utility.target_pos_to_action(controller, target), action
    )


def test_existing_lowcmd_receives_permuted_bounded_targets_and_scaled_gains():
    controller = _controller()
    controller.default_lowcmd = SimpleNamespace(
        motor_cmd=[SimpleNamespace(q=0.0, kp=0.0, kd=0.0) for _ in range(20)]
    )
    canonical_target = torch.arange(12, dtype=torch.float) - 6
    action = (
        canonical_target - controller._gait_reference - controller.cfg.default_dof_pos
    )

    command = deploy_utility.action_to_lowcmd(
        controller, action, kp_mult=0.5, kd_mult=0.25
    )

    expected = canonical_target[SDK_IN_QGYM_ORDER].clamp(
        controller.cfg.lower_joint_limit, controller.cfg.upper_joint_limit
    )
    assert command is controller.default_lowcmd
    torch.testing.assert_close(
        torch.tensor([motor.q for motor in command.motor_cmd[:12]]), expected
    )
    assert all(motor.kp == controller.cfg.kp * 0.5 for motor in command.motor_cmd[:12])
    assert all(motor.kd == controller.cfg.kd * 0.25 for motor in command.motor_cmd[:12])
    assert all(motor.q == 0.0 for motor in command.motor_cmd[12:])


@pytest.mark.parametrize(
    "include_applied_action",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Deployment clips the residual action against absolute joint "
                    "limits; the task observes the applied residual after full "
                    "gait/default position projection"
                ),
            ),
            id="full-observation-parity",
        ),
    ],
)
def test_deploy_observations_match_cpu_task_at_100hz(include_applied_action):
    cfg = Go2TrotCfg()
    cfg.seed = 7
    cfg.env.num_envs = 1
    cfg.control.ctrl_frequency = 100
    cfg.control.desired_sim_frequency = 100
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    cfg.init_state.pos = [0.0, 0.0, 3.0]
    cfg.init_state.reset_mode = "reset_to_basic"
    task_registry.convert_frequencies_to_params(cfg, Go2TrotRunnerCfg())
    backend = select_backend(cfg, "cpu", "mujoco")
    try:
        env = Go2Trot(cfg, "cpu", True, backend)
        assert env.dt == 0.01 and cfg.sim_dt == 0.01
        assert torch.count_nonzero(env.default_dof_pos) == 0
        env.root_states[:, 3:7] = torch.tensor([0.2, -0.3, 0.4, math.sqrt(0.71)])
        env.root_states[:, 10:13] = torch.tensor([0.7, -0.4, 0.2])
        backend.reset_state(torch.ones(1, dtype=torch.bool))
        env.phase_frequency[:] = DeployConfig.phase_frequency
        deploy_cfg = DeployConfig()
        deploy_cfg.obs_vector = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "phase_obs",
            "phase_frequency",
        ]
        if include_applied_action:
            deploy_cfg.obs_vector.insert(5, "dof_pos_target")
        deploy_cfg.DeployScaling = SimpleNamespace(**class_to_dict(cfg.scaling))
        controller = _controller(deploy_cfg)
        # Resolve this independent inverse mapping by joint names, so a wrong
        # deployment permutation cannot cancel out in message synthesis.
        sdk_indices = [list(env.dof_names).index(name) for name in SDK_JOINT_NAMES]

        for _ in range(3):
            env.step()
            controller.last_command = env.commands[0].clone()
            controller.phase = env.phase[0].item()
            controller.last_action[0] = env.dof_pos_target[0]
            message = SimpleNamespace(
                motor_state=[
                    SimpleNamespace(
                        q=env.dof_pos[0, index].item(),
                        dq=env.dof_vel[0, index].item(),
                        ddq=0.0,
                    )
                    for index in sdk_indices
                ],
                imu_state=SimpleNamespace(
                    quaternion=env.base_quat[0, [3, 0, 1, 2]].tolist(),
                    # Native free-joint angular qvel is body-local, as is an IMU
                    # gyro. Public root angular velocity is world-frame.
                    gyroscope=backend._datas[0].qvel[3:6].tolist(),
                ),
            )

            actual = deploy_utility.lowstate_to_obs(controller, message)
            expected = env.get_states(deploy_cfg.obs_vector)[0]

            expected_size = 49 if include_applied_action else 37
            assert actual.shape == expected.shape == (expected_size,)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    finally:
        backend.close()
