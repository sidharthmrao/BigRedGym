from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2TrotCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 5

    class init_state(Go2Cfg.init_state):
        # The gait reference supplies the nominal posture; residual targets
        # therefore use zero joint offsets here.
        default_joint_angles = {
            "hip_joint": 0.0,
            "calf_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RR_thigh_joint": 0.0,
        }
        reset_mode = "reset_to_basic"

        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.450, 0.50],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]
        root_vel_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.0, 0.0],  # z
            [0.0, 0.0],  # roll
            [0.0, 0.0],  # pitch
            [0.0, 0.0],  # yaw
        ]

    class control(Go2Cfg.control):
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 100
        gait_freq = [0.0, 0.0]  # no gait cycle for standing height control
        # Cycle offsets define a trot: front-left/rear-right move together,
        # half a cycle away from front-right/rear-left.
        gait_phase_offsets = {
            "FL_foot": 0.0,
            "FR_foot": 0.5,
            "RL_foot": 0.5,
            "RR_foot": 0.0,
        }
        # Canonical order is FL, FR, RL, RR; hip, thigh, calf within each leg.
        # q_ref = offset + amplitude * sin(phase + leg_phase).
        # These are relative PD targets; LeggedRobot adds default_dof_pos.
        # Keep a fixed standing posture while the policy adjusts base height.
        gait_joint_offsets = 4 * [0.0, 0.96, -1.36]
        gait_joint_amplitudes = 12 * [0.0]

    class commands(Go2Cfg.commands):
        num_commands = 4  # x velocity, y velocity, yaw velocity, desired base height
        height_only = True
        default_height = Go2Cfg.reward_settings.base_height_target
        height_step = 0.02

        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = 0.0
            yaw_vel = 0.0
            base_height = [0.05, 0.7]  # metres, sampled during training

    class push_robots(Go2Cfg.push_robots):
        toggle = False
        interval_s = 5

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            link_mass_scale_range = [0.9, 1.2]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.9, 1.1],
                "d_gains": [0.8, 1.2],
            }

    class asset(Go2Cfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/" + "go2/urdf/go2.urdf"
        # Use the SDK's OBJ visuals; keep our URDF's complete physical model.
        vsim_visual_mesh_dir = "{GYM_ROOT_DIR}/thirdparty/vlearn/assets/go2/assets"
        foot_name = "foot"
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base"]
        end_effector_names = ["foot"]
        fix_base_link = False
        disable_gravity = False
        disable_motors = False

    class reward_settings(Go2Cfg.reward_settings):
        base_height_target = Go2Cfg.reward_settings.base_height_target

    class scaling(Go2Cfg.scaling):
        # Canonical RobotLayout order is FL, FR, RL, RR, with
        # hip, thigh, calf inside each leg. Backends map native order to it.
        base_height = 0.3
        commands = [3, 1, 3, 0.4]
        dof_pos = 4 * [1.0472, 2.53075, 0.94247]
        dof_pos_obs = dof_pos
        dof_pos_target = [0.5 * x for x in dof_pos]
        tau_ff = 4 * [23.7, 23.7, 45.43]


class Go2TrotRunnerCfg(Go2RunnerCfg):
    class actor:
        hidden_dims = [256, 256, 128]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False
        smooth_exploration = False
        actions = ["dof_pos_target"]
        add_noise = False
        disable_actions = False

        class noise:
            scale = 1.0
            dof_pos_obs = 0.01
            base_ang_vel = 0.01
            dof_pos = 0.005
            dof_vel = 0.005
            lin_vel = 0.05
            ang_vel = [0.3, 0.15, 0.4]
            gravity_vec = 0.1

    class critic:
        hidden_dims = [128, 64]
        # * can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        activation = "elu"
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]
        normalize_obs = False

        class reward:
            class weights:
                tracking_lin_vel = 2.0  # zero-command station keeping
                tracking_ang_vel = 1.0  # keep yaw stationary
                lin_vel_z = 0.0
                ang_vel_xy = 0.01
                orientation = 1.0
                torques = 5.0e-6
                dof_vel = 0.0
                stand_still = 0.0
                dof_pos_limits = 0.0
                feet_contact_forces = 0.0
                dof_near_home = 0.0
                tracking_base_height = 4.0
                action_rate = 0.25
                action_rate2 = 0.025
                trot_support = 0.0
                swing_contact = 0.0

            class termination_weight:
                termination = 0.01

    class algorithm:
        # both
        gamma = 0.99
        lam = 0.95
        # shared
        batch_size = 2**15
        max_gradient_steps = 24
        # new
        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        rollout_size = 2**16
        # Critic
        use_clipped_value_loss = True
        # Actor
        entropy_coef = 0.01
        schedule = "adaptive"  # could be adaptive, fixed
        desired_kl = 0.01
        lr_range = [2e-5, 1e-2]
        lr_ratio = 1.5

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2trot"
        max_iterations = 550
