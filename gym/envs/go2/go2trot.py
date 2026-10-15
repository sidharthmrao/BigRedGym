import torch

from gym.utils.sampling import torch_rand_float
from gym.utils.sampling import masked_update

from gym.envs.base.legged_robot import LeggedRobot


class Go2Trot(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _init_buffers(self):
        super()._init_buffers()

        self._actuated_dof_pos_limits = self.dof_pos_limits.index_select(
            0, self.actuated_dof_indices
        )
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.phase_obs = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.phase_frequency = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.gait_reference = torch.zeros_like(self.dof_pos_target)
        foot_names = self.robot_layout.body_groups["feet"]
        phase_offsets = self.cfg.control.gait_phase_offsets
        self._gait_phase_offsets = (
            2
            * torch.pi
            * torch.tensor(
                [phase_offsets[name] for name in foot_names],
                dtype=torch.float,
                device=self.device,
            )
        )
        self._gait_dof_phase_offsets = self._gait_phase_offsets.repeat_interleave(3)
        self._gait_joint_offsets = torch.tensor(
            self.cfg.control.gait_joint_offsets,
            dtype=torch.float,
            device=self.device,
        )
        self._gait_joint_amplitudes = torch.tensor(
            self.cfg.control.gait_joint_amplitudes,
            dtype=torch.float,
            device=self.device,
        )
        self._update_phase_observation()
        self._update_gait_reference()

    def _update_phase_observation(self):
        self.phase_obs[:, 0:1] = torch.sin(self.phase)
        self.phase_obs[:, 1:2] = torch.cos(self.phase)

    def _update_gait_reference(self):
        joint_phase = self.phase + self._gait_dof_phase_offsets.unsqueeze(0)
        self.gait_reference[:] = (
            self._gait_joint_offsets
            + self._gait_joint_amplitudes * torch.sin(joint_phase)
        )

    def _leg_phases(self):
        return torch.remainder(
            self.phase + self._gait_phase_offsets.unsqueeze(0), 2 * torch.pi
        )

    def _expected_stance(self):
        return torch.sin(self._leg_phases()) > 0

    def _pre_decimation_step(self):
        self._update_gait_reference()
        reference = self.gait_reference + self.default_dof_pos.index_select(
            1, self.actuated_dof_indices
        )
        # Bound the full PD position command. Keep the applied residual in the
        # task buffer so observations and action-history rewards describe it;
        # the runner retains its separate raw samples for PPO likelihoods.
        self.dof_pos_target.clamp_(
            min=self._actuated_dof_pos_limits[:, 0] - reference,
            max=self._actuated_dof_pos_limits[:, 1] - reference,
        )

    def _compute_torques(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        torques = (
            self.p_gains
            * (self.gait_reference + self.dof_pos_target + default_pos - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        return torch.clip(
            torques, -self.actuated_torque_limits, self.actuated_torque_limits
        ).view(self.torques.shape)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        phase = torch_rand_float(
            0,
            2 * torch.pi,
            shape=self.phase.shape,
            device=self.device,
        )
        phase_frequency = torch_rand_float(
            self.cfg.control.gait_freq[0],
            self.cfg.control.gait_freq[1],
            shape=self.phase_frequency.shape,
            device=self.device,
        )
        masked_update(self.phase, phase, reset_mask)
        masked_update(self.phase_frequency, phase_frequency, reset_mask)

    def _resample_commands(self, command_mask):
        height_min, height_max = self.command_ranges["base_height"]
        height = torch_rand_float(
            height_min,
            height_max,
            (self.num_envs,),
            device=self.device,
        )
        self.commands[command_mask, :3] = 0.0
        masked_update(self.commands[:, 3], height, command_mask)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._update_gait_reference()
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )
        self._update_phase_observation()

    def _post_physics_step(self):
        super()._post_physics_step()
        self._advance_phase()

    def _advance_phase(self):
        # phase_frequency is in cycles/s; _post_physics_step runs once per
        # physics substep, so convert cycles to radians and use the simulation dt.
        self.phase.add_(
            2 * torch.pi * self.dt * self.phase_frequency / self.cfg.control.decimation
        ).remainder_(2 * torch.pi)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._update_phase_observation()

    def _foot_contact_strength(self):
        """Map upward foot load smoothly from zero to nominal body-weight."""
        load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2]
            / (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)

    def _reward_trot_support(self):
        """Require both feet in the scheduled trot diagonal to support the body."""
        phase = torch.sin(self._leg_phases())
        contact_strength = self._foot_contact_strength()
        stance_strength = torch.where(
            phase > 0.0,
            contact_strength,
            torch.ones_like(contact_strength),
        )
        paired_support = torch.clamp(
            2.0 * torch.sqrt(torch.prod(stance_strength, dim=1)),
            max=1.0,
        )
        return torch.relu(phase).amax(dim=1) * paired_support

    def _reward_swing_contact(self):
        """Penalize load carried by feet during their scheduled swing phase."""
        phase = torch.sin(self._leg_phases())
        return -torch.mean(torch.relu(-phase) * self._foot_contact_strength(), dim=1)

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_tracking_base_height(self):
        """Reward matching the requested base height from either direction."""
        error = (self.base_height[:, 0] - self.commands[:, 3]) / self.scales[
            "base_height"
        ]
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
        )
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
