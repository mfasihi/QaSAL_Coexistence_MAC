import numpy as np
import simpy
from transmitter import Transmitter
from channel import Channel
from utils import jain_fairness_index_between_networks


class NetworkEnvironment:
    def __init__(self,
                 seed,
                 gnb_config,
                 ap_config,
                 num_gnbs,
                 num_aps,
                 step_duration,
                 gnb_protocol,
                 gnb_priorities,
                 ap_priorities,
                 numerology,
                 debug,
                 mode,
                 multi_objective=False,
                 alpha=0.5,
                 primal_dual=False,
                 augmented_state=False,
                 constraints={}):
        """
        Initialize the network environment.
        :param gnb_config: Configuration for gNBs.
        :param ap_config: Configuration for APs.
        :param num_gnbs: Number of gNBs in the environment.
        :param num_aps: Number of APs in the environment.
        :param step_duration: Duration of a single step for observation.
        :param gnb_protocol: LBT protocol used by gNBs.
        :param: gnb_priorities: List of gnb priority IDs.
        :param: ap_priorities: List of ap priority IDs.
        :param numerology: Numerology of licensed spectrum.
        :param debug: Enables debug by printing the outputs.
        :param mode: Training, Execution, Fixed.
        """
        self.gnb_config = gnb_config
        self.ap_config = ap_config
        self.num_gnbs = num_gnbs
        self.num_aps = num_aps
        self.step_duration = step_duration
        self.gnb_protocol = gnb_protocol
        self.numerology = numerology
        self.channel = Channel()
        self.sim_env = simpy.Environment()
        self.mode = mode
        self.multi_objective = multi_objective
        self.alpha = alpha
        self.primal_dual = primal_dual
        self.augmented_state = augmented_state
        self.constraints = constraints
        self.violations = {key: [] for key in constraints.keys()}
        self.lambda_values = {key: 0 for key in constraints.keys()}  # Initial λ
        self.eta_lambda = 0.1
        self.beta_lambda = 0.1  # Sensitivity factor
        self.lambda_max = 5.0

        self.step_metrics = None
        self.cumulative_network_metrics = {
            ("gNB", "PC1"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("gNB", "PC2"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("gNB", "PC3"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("gNB", "PC4"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("AP", "PC1"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("AP", "PC2"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("AP", "PC3"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
            ("AP", "PC4"): {"delay": [0], "collisions": 0, "transmissions": 0, "succ_airtime": []},
        }

        # Initialize gNBs with the specified protocol
        self.gnbs = [
            Transmitter(
                id=i,
                seed=seed,
                environment=self,
                priority_class_id=gnb_priorities[i],
                config=gnb_config,
                protocol=gnb_protocol,
                network="gNB",
                numerology=numerology,
                debug=debug
            )
            for i in range(num_gnbs)
        ]

        # Initialize APs with CSMA/CA protocol
        self.aps = [
            Transmitter(
                id=j,
                seed=seed,
                environment=self,
                priority_class_id=ap_priorities[j],
                config=ap_config,
                protocol="CSMA/CA",
                network="AP",
                numerology=None,
                debug=debug,
            )
            for j in range(num_aps)
        ]

        # Start transmitter processes
        for gnb in self.gnbs:
            self.sim_env.process(gnb.run())
        for ap in self.aps:
            self.sim_env.process(ap.run())

    def reset(self, randomDual=False):
        """Reset the environment for a new episode."""
        self.channel = Channel()
        self.reset_transmitters(reset_indicators=True)
        self.reset_cumulative_network_metrics()
        state = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]
        self.violations = {key: [] for key in self.constraints.keys()}
        if self.augmented_state:
            self.reset_lambda_values(randomDual)
            state += list(self.lambda_values.values())
        if self.primal_dual:
            self.reset_lambda_values(randomDual=False)

        return state

    def reset_transmitters(self, reset_indicators=False):
        """Reset transmitter performance metrics."""
        self.channel.busy_time = 0
        self.channel.total_time = 0
        for transmitter in self.gnbs + self.aps:
            transmitter.successful_transmissions = 0
            transmitter.total_transmissions = 0
            transmitter.successful_airtime = 0
            transmitter.total_airtime = 0
            transmitter.collisions = 0
            transmitter.delay_list = list()
            if reset_indicators:
                transmitter.failed_transmissions_in_a_row = 0
                transmitter.last_successful_end_time = None

    def reset_lambda_values(self, randomDual):
        if randomDual:
            self.lambda_values = {key: round(np.random.uniform(0, self.lambda_max), 2) for key in self.constraints.keys()}
        else:
            self.lambda_values = {key: 0 for key in self.constraints.keys()}

    def reset_cumulative_network_metrics(self):
        for key, metrics in self.cumulative_network_metrics.items():
            metrics["collisions"] = 0  # Reset collisions to 0
            metrics["transmissions"] = 0  # Reset transmissions to 0
            metrics["succ_airtime"] = []  # Empty the succ_airtime list
            metrics["delay"] = [0]  # Keep only the last element in the delay list

    def update_network_metrics(self):
        step_start_time = self.sim_env.now - self.step_duration
        step_end_time = self.sim_env.now

        step_delays = {key: None for key in self.cumulative_network_metrics.keys()}
        step_succ_airtime = {key: 0 for key in self.cumulative_network_metrics.keys()}
        step_collisions = {key: 0 for key in self.cumulative_network_metrics.keys()}
        channel_busy_time = 0

        active_transmissions = [t for t in self.channel.active_transmissions if t.tx_type == 'packet']
        for t in active_transmissions:
            key = (t.network, f"PC{t.priority_class}")
            delay, succ = self._calculate_delay_contribution(t, step_start_time, step_end_time)
            if not self.channel.is_collision(t):
                if step_delays[key] is None:
                    step_delays[key] = delay
                else:
                    # previous successful transmission of the same network was ended and this one has started.
                    prev_tx_delay = step_delays[key]
                    new_tx_delay = delay
                    step_delays[key] = (prev_tx_delay + new_tx_delay) - self.step_duration
                    self.cumulative_network_metrics[key]["delay"][-1] = 0

                channel_busy_time = self.step_duration - step_delays[key]
                step_succ_airtime[key] = channel_busy_time
                if succ:
                    self.cumulative_network_metrics[key]["delay"].append(0)
                self.cumulative_network_metrics[key]["delay"][-1] += step_delays[key]
            else:
                step_collisions[key] += 1
                self.cumulative_network_metrics[key]["collisions"] += 1 if not t.collided else 0
                t.collided = True

            if succ:
                self.cumulative_network_metrics[key]["transmissions"] += 1

        for key, delay in step_delays.items():
            if delay is None:
                step_delays[key] = self.step_duration
                self.cumulative_network_metrics[key]["delay"][-1] += step_delays[key]

        for key, airtime in step_succ_airtime.items():
            self.cumulative_network_metrics[key]["succ_airtime"].append(airtime)

        self.channel.clean_old_transmissions(self.sim_env.now)
        self.channel.busy_time += channel_busy_time
        self.channel.total_time += self.step_duration

        return {
            "step_collisions": step_collisions,
            "step_delays": step_delays,
            "step_channel_busy_time_ratio": channel_busy_time / self.step_duration,
        }

    def _calculate_delay_contribution(self, transmission, step_start_time, step_end_time):
        """
        Helper function to calculate delay contribution of a single transmission within the step duration.
        :param transmission: The transmission object.
        :param step_start_time: The start time of the step.
        :param step_end_time: The end time of the step.
        :return: The delay contribution for this transmission.
        """
        # Case 1: Transmission spans the entire step
        if transmission.start_time <= step_start_time and transmission.end_time >= step_end_time:
            return 0, False
        # Case 2: Transmission entirely outside the step
        elif transmission.end_time <= step_start_time or transmission.start_time >= step_end_time:
            return self.step_duration, False
        # Case 3: Transmission starts within the step and extends beyond
        elif transmission.start_time > step_start_time and transmission.end_time >= step_end_time:
            return transmission.start_time - step_start_time, False
        # Case 4: Transmission starts before the step and ends within
        elif transmission.start_time <= step_start_time and transmission.end_time < step_end_time:
            return step_end_time - transmission.end_time, True
        # Case 5: Transmission completely within the step
        else:
            return self.step_duration - (transmission.end_time - transmission.start_time), True

    def step(self, policy, actions=None):
        """
        Perform a step in the environment.
        :param policy: The policy to use for the simulation.
        :param actions: Tuple of contention window sizes for gNB PC1, gNB PC3, and AP PC3.
        :return: Reward and state.
        """
        if self.mode in ["Training", "Execution"] and policy in ["DQN"]:
            gnb_pc1_action, gnb_pc3_action, ap_pc3_action = actions
            for gnb in self.gnbs:
                if gnb.priority_class.id == 1:  # PC1
                    gnb.agent_action = gnb_pc1_action
                elif gnb.priority_class.id == 3:  # PC3
                    gnb.agent_action = gnb_pc3_action
            for ap in self.aps:
                if ap.priority_class.id == 3:  # PC3
                    ap.agent_action = ap_pc3_action

        # Run the simulation for the step duration
        self.sim_env.run(until=self.sim_env.now + self.step_duration)
        self.step_metrics = self.update_network_metrics()
        violation = 0

        if policy in ["DQN"]:
            if self.primal_dual or self.augmented_state:
                delay_th = self.constraints["smoothed_delay_pc1"]
                delay = self._get_constraint_metric("smoothed_delay_pc1")
                violation = (delay - delay_th) / delay_th

                self.violations["smoothed_delay_pc1"].append(violation)

            next_state = self.build_state(self.step_metrics)
            reward = self.compute_reward() if self.mode == "Training" else None

            return next_state, reward, violation

    def compute_reward(self):
        """Calculate the reward for the current step based on metrics."""
        step_reward = 0
        if self.mode == "Training":
            delay = self._get_constraint_metric("smoothed_delay_pc1")
            jfi = self._get_constraint_metric("jfi")

            if self.multi_objective:
                alpha = self.alpha
                delay_th = 2.0
                delay_term = max(0, min(1, 1 - delay / delay_th))
                jfi_term = (jfi - 0.5) / 0.5
                step_reward += (1 - alpha) * jfi_term + alpha * delay_term

            if self.augmented_state or self.primal_dual:
                beta = 2.0
                delay_th = self.constraints["smoothed_delay_pc1"]
                reward = beta * (jfi - 0.5) / 0.5
                if delay < delay_th:
                    reward -= beta * (delay_th - delay) / delay_th
                step_reward += reward

        return step_reward

    def build_state(self, step_metrics):
        """Build the state representation for the current step."""
        avg_delay_pc1 = np.mean(self.cumulative_network_metrics[("gNB", "PC1")]["delay"]) * 1e-3
        last_delay_pc1 = self.cumulative_network_metrics[("gNB", "PC1")]["delay"][-1] * 1e-3
        smoothed_delay_pc1 = self._get_constraint_metric("smoothed_delay_pc1")
        collisions_pc1 = self.cumulative_network_metrics[("gNB", "PC1")]["collisions"]
        transmissions_pc1 = self.cumulative_network_metrics[("gNB", "PC1")]["transmissions"] + 1
        collision_percent_pc1 = collisions_pc1 / transmissions_pc1 if transmissions_pc1 != 0 else 0
        step_collisions_pc1 = step_metrics["step_collisions"][("gNB", "PC1")]
        jfi_cumulative_total = self._get_constraint_metric("jfi")
        channel_utilization_ratio = self.channel.get_busy_time_ratio()
        step_channel_busy_time_ratio = step_metrics["step_channel_busy_time_ratio"]
        last_delays = np.array(self.cumulative_network_metrics[("gNB", "PC1")]["delay"][-5:]) * 1e-3
        delays = np.array(self.cumulative_network_metrics[("gNB", "PC1")]["delay"]) * 1e-3
        state = []

        if self.augmented_state or self.primal_dual:
            step_violation_count = np.sum(last_delays > self.constraints["smoothed_delay_pc1"])
            violation_count = np.sum(delays > self.constraints["smoothed_delay_pc1"])
            step_violation_ratio = step_violation_count / len(last_delays) if len(last_delays) != 0 else 0
            violation_ratio = violation_count / len(delays) if len(delays) != 0 else 0

            state = [
                avg_delay_pc1,
                collision_percent_pc1,
                channel_utilization_ratio,
                step_channel_busy_time_ratio,
                step_violation_ratio,
                violation_ratio,
                smoothed_delay_pc1,
                jfi_cumulative_total
            ]
            state += list(self.lambda_values.values())

        if self.multi_objective:
            state = [
                avg_delay_pc1,
                collision_percent_pc1,
                channel_utilization_ratio,
                step_channel_busy_time_ratio,
                last_delay_pc1,
                step_collisions_pc1,
                smoothed_delay_pc1,
                jfi_cumulative_total
            ]

        state_round = [round(s, 3) for s in state]
        return state_round

    def update_dual_variables(self):
        lambda_value = np.clip(
            np.maximum(0,
                       self.lambda_values["smoothed_delay_pc1"] +
                       self.eta_lambda * np.mean(self.violations["smoothed_delay_pc1"])
                       ), 0, self.lambda_max)

        self.update_eta_lambda()
        self.lambda_values["smoothed_delay_pc1"] = np.round(lambda_value, 2)
        self.violations = {key: [] for key in self.constraints.keys()}

    def update_eta_lambda(self):
        eta_min = 0.01  # minimum learning rate when violations are rare
        eta_max = 0.2  # maximum learning rate when violations are frequent

        violations = np.array(self.violations["smoothed_delay_pc1"])
        num_violations = sum(1 for v in violations if v > 0)
        violation_ratio = num_violations / len(violations)

        self.eta_lambda = eta_min + (eta_max - eta_min) * violation_ratio

    def _get_constraint_metric(self, name):
        if "smoothed_delay_pc1" in name:
            return np.mean(self.cumulative_network_metrics[("gNB", "PC1")]["delay"][-5:]) * 1e-3
        elif "jfi" in name:
            return jain_fairness_index_between_networks(self.gnbs, self.aps, priority_class=1)
