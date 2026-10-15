import time
import threading
import math

from go2_deploy.state import State
from go2_deploy.rl_controller import RLController
from go2_deploy.unitree_remote_controller import UnitreeRemoteController, RCHandler
from go2_deploy.deploy_config import DeployConfig
from go2_deploy.utility import deploy_utility
from go2_deploy.utility.csv_logger import CSVLogger
from go2_deploy.utility.thread import RecurrentThread

import torch
from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    SportModeState_,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.utils.crc import CRC


class MainController:
    def __init__(self):
        # Flags are monitored by watchdog_thread to switch between states
        self._state = State.EMERGENCY_STOP
        self._estop_flag = False
        self._recovery_flag = False
        self._intermediate_flag = False
        self._custom_ctrl_flag = False

        self.rl_controller = RLController()
        self.remote_controller = UnitreeRemoteController()
        self.rc_handler = RCHandler(self)
        self.cfg = DeployConfig()
        self.csv_logger = CSVLogger(self)

        self.obs_vec_size = 0
        for obs in self.cfg.obs_vector:
            self.obs_vec_size += self.cfg.obs_sizes[obs]

        self._init_buffers()
        if self.cfg.task_name == "go2trot":
            self._init_go2trot_buffers()

        self._init_unitree_clients()

        self.default_lowcmd = deploy_utility.default_lowcmd()
        self.emergency_lowcmd = deploy_utility.emergency_lowcmd()

        # Change kp, kd using keyboard/RC input
        self.kp_mult = 1.0
        self.kd_mult = 1.0

        # Threads
        self._create_lowcmd_thread()
        self.emergency_lowcmd_thread = None
        self.watchdog_thread = RecurrentThread(
            interval=0.01, target=self._watchdog_loop
        )

        self.switch_to_recovery()
        self.watchdog_thread.Start()
        # At this point, buffers initialized, self._state = RECOVERY

    # Init --------------------------------------------------

    def _init_buffers(self):
        # SportModeState_: last recieved sportmode msg
        self.last_sportmodestate_msg = None
        self.last_sportmodestate_msg_lock = threading.Lock()

        # LowState_: last recieved lowstate msg
        self.last_lowstate_msg = None
        self.last_lowstate_msg_lock = threading.Lock()

        # torch tensor: obs vector computed from last_lowstate_msg
        self.last_obs = torch.zeros(self.obs_vec_size)
        self.last_obs_lock = threading.Lock()

        # torch.tensor((2, 12)): Last 2 actor outputs (in radians),
        # already averaged using exp moving avg (if applicable)
        self.last_action = torch.zeros((2, 12))
        self.last_action_lock = threading.Lock()

        # [x_vel, y_vel, yaw_vel, desired base height]
        self.last_command = torch.tensor(
            [0.0, 0.0, 0.0, self.cfg.default_height_command]
        )
        self.last_command_lock = threading.Lock()

        # Log obs freq / control freq
        self.last_terminal_output_time = time.monotonic()
        self.lowstate_obs_count = 0
        self.sportmodestate_obs_count = 0
        self.action_count = 0

    def _init_go2trot_buffers(self):
        self.phase_frequency = self.cfg.phase_frequency
        self.phase = 0.0
        self._gait_reference = torch.zeros(12)
        self._gait_phase_offsets = torch.tensor([0, math.pi, math.pi, 0])
        self._gait_dof_phase_offsets = self._gait_phase_offsets.repeat_interleave(3)
        self._gait_joint_offsets = torch.tensor(4 * [0.0, 0.96, -1.36])
        self._gait_joint_amplitudes = torch.tensor(4 * [0.0, -0.15, 0.30])

    # Unitree SDK pub/sub clients
    def _init_unitree_clients(self):
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate_msg, 10)
        self.sportmodestate_subscriber = ChannelSubscriber(
            "rt/sportmodestate", SportModeState_
        )
        self.sportmodestate_subscriber.Init(self._on_sportmodestate_msg, 10)
        self.motion_switcher_client = MotionSwitcherClient()
        self.motion_switcher_client.SetTimeout(5.0)
        self.motion_switcher_client.Init()
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()
        self.crc = CRC()

    # Read and write messages ---------------------------------------------

    # torch.tensor(12) -> LowCmd_
    def action_to_lowcmd(self, action):
        return deploy_utility.action_to_lowcmd(
            self, action, kp_mult=self.kp_mult, kd_mult=self.kd_mult
        )

    # LowState_ -> torch tensor
    def msg_to_obs(self, lowstate_msg):
        return deploy_utility.lowstate_to_obs(self, lowstate_msg)

    # Save the last lowstate msg, update velocity command
    def _on_lowstate_msg(self, msg):
        t = time.monotonic()
        with self.last_lowstate_msg_lock:
            self.last_lowstate_msg = msg
            self.remote_controller.parse(msg.wireless_remote)
        self.rc_handler._process_input()

        self.csv_logger.log_lowstate(t, msg, self.remote_controller)
        self.lowstate_obs_count += 1

    # Save the last sportmodestate_msg
    def _on_sportmodestate_msg(self, msg):
        t = time.monotonic()
        with self.last_sportmodestate_msg_lock:
            self.last_sportmodestate_msg = msg
        self.csv_logger.log_sportmodestate(t, msg)
        self.sportmodestate_obs_count += 1

    # Main control loop ------------------------------------------

    # Update self.phase, self._gait_reference
    def _process_go2trot_buffers(self, t):
        self.phase = 2 * math.pi * self.phase_frequency * t % (2 * math.pi)
        joint_phase = self.phase + self._gait_dof_phase_offsets
        self._gait_reference = (
            self._gait_joint_offsets
            + self._gait_joint_amplitudes * torch.sin(joint_phase)
        )

    # Read last lowstate msg, publish LowCmd_
    def _control_loop(self):
        t = time.monotonic()
        if self.cfg.task_name == "go2trot":
            self._process_go2trot_buffers(t)

        with self.last_command_lock:
            self.last_command = torch.tensor(
                [
                    self.remote_controller.lin_vel_x,
                    self.remote_controller.lin_vel_y,
                    self.remote_controller.yaw_vel,
                    self.cfg.default_height_command,
                ]
            )

        with (
            self.last_lowstate_msg_lock,
            self.last_sportmodestate_msg_lock,
            self.last_obs_lock,
        ):
            self.last_obs = self.msg_to_obs(self.last_lowstate_msg)
            last_obs = self.last_obs.clone().detach()
        action = self._act(last_obs)

        with self.last_action_lock:
            self.last_action[1] = self.last_action[0]
            self.last_action[0] = action
            if self.cfg.exp_moving_avg:
                alpha = self.cfg.ema_smoothing_factor
                self.last_action[0] = (alpha * self.last_action[0]) + (
                    1 - alpha
                ) * self.last_action[1]

            lowcmd = self.action_to_lowcmd(self.last_action[0])
            smoothed_action = self.last_action[0].clone().detach()

        lowcmd.crc = self.crc.Crc(lowcmd)
        self.lowcmd_publisher.Write(lowcmd)

        self.csv_logger.log_control(t, self.last_obs, action, smoothed_action, lowcmd)
        self.action_count += 1

    # Return torch tensor: target joint position (minus default pos, gait traj)
    def _act(self, last_obs):
        if self._state == State.CUSTOM_CTRL:
            # return torch.zeros(12) (replay gait trajectory only)
            return self.rl_controller.act(last_obs)
        elif self._state == State.INTERMEDIATE:
            return deploy_utility.target_pos_to_action(self, self.intermediate_pos)
        else:
            return self.last_action[0]

    # Emergency stop ----------------------------------

    def emergency_stop(self):
        self._estop_flag = False

        if self._state == State.EMERGENCY_STOP:
            self._estop_flag = False
            return

        self._state = State.EMERGENCY_STOP

        if self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Wait()

        self.emergency_lowcmd_thread = RecurrentThread(
            interval=0.01, target=self._emergency_control_loop
        )
        self.motion_switcher_client.ReleaseMode()
        self.emergency_lowcmd_thread.Start()
        print("Emergency stop activated!")
        time.sleep(5)
        self.emergency_lowcmd_thread.Wait()
        self.emergency_lowcmd_thread = None
        self._create_lowcmd_thread()

        self.switch_to_recovery()
        self._estop_flag = False

    # Can implement checking for unsafe conditions automatically later
    # This should be the only thread that calls the functions to
    # change states
    def _watchdog_loop(self):
        if self._estop_flag:
            self.emergency_stop()
        elif self._recovery_flag:
            self.switch_to_recovery()
        elif self._intermediate_flag:
            self.switch_to_intermediate()
        elif self._custom_ctrl_flag:
            self.switch_to_custom_controller()

        t = time.monotonic()
        if t > self.last_terminal_output_time + self.cfg.terminal_log_period:
            self._output_terminal_info(t)

    # publish LowCmd_ damping messages
    def _emergency_control_loop(self):
        self.lowcmd_publisher.Write(self.emergency_lowcmd)

    # Switch between states -------------------------------------

    def request_emergency_stop(self):
        print("Estop requested")
        self._estop_flag = True

    def request_recovery(self):
        print("recovery requested")
        self._recovery_flag = True

    def request_intermediate(self):
        print("intermediate requested")
        self._intermediate_flag = True

    def request_custom_ctrl(self):
        print("custom ctrl requested")
        self._custom_ctrl_flag = True

    def switch_to_recovery(self):
        self._recovery_flag = False

        if self._state == State.RECOVERY:
            print("Already in recovery state!")
            return

        self._state = State.RECOVERY
        print("Switching to recovery state")

        if self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Wait()
            self._create_lowcmd_thread()

        # Necessary to allow MSC to start up after LowCmd_ stream stops
        time.sleep(5)

        self.motion_switcher_client.SelectMode("mcf")
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "mcf":
            print("Failed to switch to sport mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.SelectMode("mcf")
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        for _ in range(10):
            error_code = self.sport_client.RecoveryStand()
            if error_code == 0:
                print("call to RecoveryStand() succeeded")
                break
            time.sleep(1)
        else:
            print("RecoveryStand() did not succeed")
        time.sleep(10)
        print("Recovered")

    def switch_to_intermediate(self):
        self._intermediate_flag = False

        if self._state != State.RECOVERY:
            print("Must be in recovery state to switch to intermediate")
            return

        self._state = State.INTERMEDIATE
        print("Switching to intermediate")

        self.intermediate_pos = (
            torch.tensor(
                deploy_utility._get_obs_dof_pos_obs(self, self.last_lowstate_msg)
            )
            .detach()
            .clone()
        )

        self.motion_switcher_client.ReleaseMode()
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "":
            print("Failed to switch to low state mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.ReleaseMode()
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        self.lowcmd_thread.Start()

    def switch_to_custom_controller(self):
        self._custom_ctrl_flag = False

        if self._state != State.INTERMEDIATE and self._state != State.RECOVERY:
            print(
                "Must be in recovery or intermediate state"
                + "to switch to custom controller!"
            )
            return

        self._state = State.CUSTOM_CTRL
        print("Switching to custom controller")

        self.motion_switcher_client.ReleaseMode()
        mode = self.motion_switcher_client.CheckMode()[1]["name"]
        while mode != "":
            print("Failed to switch to low state mode, trying again in 5s")
            time.sleep(5)
            self.motion_switcher_client.ReleaseMode()
            mode = self.motion_switcher_client.CheckMode()[1]["name"]

        if not self.lowcmd_thread.IsAlive():
            self.lowcmd_thread.Start()

    # Utility -----------------------------------------------------

    def _create_lowcmd_thread(self):
        self.lowcmd_thread = RecurrentThread(
            interval=1 / self.cfg.ctrl_freq, target=self._control_loop
        )

    # maybe use later. claude's suggestion
    def _check_msc_mode(self, retries=5):
        for _ in range(retries):
            code, data = self.motion_switcher_client.CheckMode()
            if code == 0 and data is not None:
                return data.get("name")
        time.sleep(1)
        return None

    def _output_terminal_info(self, current_time):
        print(
            f"Logging to {self.csv_logger.run_dir} "
            + f"({self.csv_logger.dropped_rows} rows dropped)"
        )
        print(
            "\nlowstate obs freq = "
            + f"{self.lowstate_obs_count / (current_time - self.last_terminal_output_time):.5} Hz, "  # noqa: E501
            + "sportmodestate obs freq = "
            + f"{self.sportmodestate_obs_count / (current_time - self.last_terminal_output_time):.5} Hz, "  # noqa: E501
            + "custom policy control freq = "
            + f"{self.action_count / (current_time - self.last_terminal_output_time):.5} Hz"  # noqa: E501
        )
        print("Current mode: " + self._state.name)
        print("Keyboard command [RC command]: meaning")
        print(
            "<enter> [X]: emergency stop | q [B]: intermediate | c [A]: custom controller"  # noqa: E501
        )
        print(
            "i [Up]: increase kp (by 10%) | k [Down]: decrease kp | l [Right]: increase kd | j [Left]: decrease kd\n"  # noqa: E501
        )

        self.action_count = 0
        self.lowstate_obs_count = 0
        self.sportmodestate_obs_count = 0
        self.last_terminal_output_time = current_time
