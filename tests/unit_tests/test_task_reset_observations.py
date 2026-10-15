"""Episode resets must publish current actor/critic inputs before the next step."""

import math

import mujoco
import numpy as np
import pytest
import torch

from tests.unit_tests.conftest import vsim_guard

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.task_registry import select_backend, task_registry


@pytest.fixture(
    params=(
        pytest.param(("cpu", "mujoco"), id="cpu"),
        pytest.param(("cuda:0", "mujoco"), marks=pytest.mark.warp, id="warp"),
        pytest.param(("cuda:0", "vsim"), marks=pytest.mark.vsim, id="vsim"),
    )
)
def task(request):
    device, engine = request.param
    if engine == "vsim":
        vsim_guard()
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("GPU tests requested but CUDA is not available", pytrace=False)
    torch.manual_seed(7)
    cfg = Go2TrotCfg()
    runner_cfg = Go2TrotRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = 4
    cfg.control.ctrl_frequency = 100
    cfg.control.desired_sim_frequency = 100
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    # Allocate the range-reset buffers as well as the basic-reset state.
    cfg.init_state.reset_mode = "reset_to_range"
    cfg.init_state.root_pos_range = [
        list(bounds) for bounds in cfg.init_state.root_pos_range
    ]
    cfg.init_state.root_pos_range[2] = [3.0, 3.0]
    cfg.init_state.pos = [0.0, 0.0, 3.0]
    cfg.init_state.rot = [0.2, -0.3, 0.4, math.sqrt(0.71)]
    cfg.init_state.lin_vel = [0.3, -0.2, 0.1]
    cfg.init_state.ang_vel = [0.6, 0.3, -0.4]
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, device, engine)
    # VSim refreshes every native root pose after a masked command. Its float32
    # quaternion readback can move by about 1e-7 even in unselected worlds.
    # Only native root views receive this tolerance; derived caches stay exact.
    root_refresh_atol = 2e-7 if engine == "vsim" else 0.0
    try:
        yield Go2Trot(cfg, device, True, backend), runner_cfg, root_refresh_atol
    finally:
        backend.close()


def _expected_base_state(env):
    """Independent rotation-matrix oracle, without task quaternion helpers."""
    root = env.root_states.cpu().numpy()
    rotations = np.empty((env.num_envs, 3, 3))
    for index, state in enumerate(root):
        quat = state[[6, 3, 4, 5]].astype(np.float64)
        mujoco.mju_quat2Mat(rotations[index].reshape(9), quat)
    local = np.einsum("nji,nj->ni", rotations, root[:, 7:10])
    angular = np.einsum("nji,nj->ni", rotations, root[:, 10:13])
    gravity = np.einsum("nji,nj->ni", rotations, env.gravity_vec.cpu().numpy())
    return {
        "base_quat": env.root_states[:, 3:7].clone(),
        "base_lin_vel": torch.tensor(local, dtype=torch.float, device=env.device),
        "base_ang_vel": torch.tensor(angular, dtype=torch.float, device=env.device),
        "projected_gravity": torch.tensor(
            gravity, dtype=torch.float, device=env.device
        ),
        "base_height": env.root_states[:, 2:3].clone(),
    }


def test_standing_policy_observations_ignore_gait_phase(task):
    env, runner_cfg, _ = task
    for names, size in (
        (runner_cfg.actor.obs, 46),
        (runner_cfg.critic.obs, 50),
    ):
        assert "phase_obs" not in names
        assert "phase_frequency" not in names
        assert env.get_states(names).shape == (env.num_envs, size)

    before_actor = env.get_states(runner_cfg.actor.obs).clone()
    before_critic = env.get_states(runner_cfg.critic.obs).clone()
    env.phase[:] = torch.tensor(
        [[0.0], [math.pi / 2], [math.pi], [3 * math.pi / 2]],
        device=env.device,
    )
    env.phase_frequency[:] = torch.tensor(
        [[1.0], [1.5], [2.0], [3.0]], device=env.device
    )
    env._update_phase_observation()
    torch.testing.assert_close(env.get_states(runner_cfg.actor.obs), before_actor)
    torch.testing.assert_close(env.get_states(runner_cfg.critic.obs), before_critic)


@pytest.mark.parametrize("reset_mode", ("reset_to_basic", "reset_to_range"))
def test_selected_reset_refreshes_actor_and_critic_observations(
    task, reset_mode, monkeypatch
):
    env, runner_cfg, root_refresh_atol = task
    env._reset_state = getattr(env, reset_mode)
    # Both reset paths replace a moving pose. The range reset uses nontrivial
    # roll/pitch/yaw so a stale orientation cannot hide behind an upright pose.
    if reset_mode == "reset_to_range":
        env.root_pos_range[:] = torch.tensor(
            [
                [0.1, 0.1],
                [-0.2, -0.2],
                [3.2, 3.2],
                [0.3, 0.3],
                [-0.4, -0.4],
                [0.7, 0.7],
            ],
            device=env.device,
        )
        env.root_vel_range[:] = torch.tensor(
            [
                [0.2, 0.2],
                [-0.4, -0.4],
                [0.1, 0.1],
                [0.6, 0.6],
                [-0.5, -0.5],
                [0.3, 0.3],
            ],
            device=env.device,
        )
    cached_height = env.base_height
    for _ in range(3):
        env.step()
        assert env.base_height is cached_height
        torch.testing.assert_close(cached_height, env.root_states[:, 2:3])
    names = tuple(_expected_base_state(env))
    cached = {name: getattr(env, name) for name in names}
    before = {name: value.clone() for name, value in cached.items()}
    before_actor = env.get_states(runner_cfg.actor.obs).clone()
    before_critic = env.get_states(runner_cfg.critic.obs).clone()
    before_root = env.root_states.clone()
    before_episode = env.episode_length_buf.clone()
    reset_mask = torch.tensor([True, False, True, False], device=env.device)

    def unexpected_step(*args, **kwargs):
        pytest.fail("Reset must refresh observations without advancing physics")

    monkeypatch.setattr(env._backend, "step", unexpected_step)
    env._reset_idx(reset_mask)
    expected = _expected_base_state(env)
    for name in names:
        assert getattr(env, name) is cached[name]
        torch.testing.assert_close(cached[name], expected[name], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(
            cached[name][~reset_mask],
            before[name][~reset_mask],
            atol=root_refresh_atol if name in ("base_quat", "base_height") else 0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        env.root_states[~reset_mask],
        before_root[~reset_mask],
        atol=root_refresh_atol,
        rtol=0.0,
    )
    assert torch.equal(env.episode_length_buf[~reset_mask], before_episode[~reset_mask])
    assert not env.episode_length_buf[reset_mask].any()

    # The runner reads these concatenated, scaled observations immediately
    # after reset. Verify those actual actor/critic inputs, not only attributes.
    for obs_names, before_obs in (
        (runner_cfg.actor.obs, before_actor),
        (runner_cfg.critic.obs, before_critic),
    ):
        actual = env.get_states(obs_names)
        pieces = []
        for name in obs_names:
            if name in expected:
                value = expected[name]
                if name in env.scales:
                    value = value / env.scales[name]
            else:
                value = env.get_state(name)
            pieces.append(value)
        torch.testing.assert_close(
            actual, torch.cat(pieces, dim=-1), atol=1e-5, rtol=2e-5
        )
        assert torch.equal(actual[~reset_mask], before_obs[~reset_mask])


def test_empty_reset_preserves_derived_observation_caches(task):
    env, runner_cfg, root_refresh_atol = task
    env.step()
    cached = {name: getattr(env, name) for name in _expected_base_state(env)}
    before = {name: value.clone() for name, value in cached.items()}
    actor = env.get_states(runner_cfg.actor.obs).clone()
    critic = env.get_states(runner_cfg.critic.obs).clone()

    env._reset_idx(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    for name, value in cached.items():
        assert getattr(env, name) is value
        torch.testing.assert_close(
            value,
            before[name],
            atol=root_refresh_atol if name in ("base_quat", "base_height") else 0.0,
            rtol=0.0,
        )
    assert torch.equal(env.get_states(runner_cfg.actor.obs), actor)
    assert torch.equal(env.get_states(runner_cfg.critic.obs), critic)
