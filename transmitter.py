import random
import simpy
from utils import log


class Transmission:
    def __init__(self, start_time, airtime_duration, rs_duration, network, id, priority_class, tx_type):
        self.id = id
        self.start_time = start_time
        self.rs_duration = rs_duration
        self.airtime_duration = airtime_duration
        self.end_time = start_time + airtime_duration + rs_duration
        self.network = network
        self.priority_class = priority_class
        self.tx_type = tx_type
        self.collided = False


class Transmitter:
    def __init__(self, id, seed, environment, priority_class_id, config, protocol, network, numerology, debug):
        """
        :param id: Transmitter ID.
        :param priority_class_id: Assigned priority class ID.
        :param config: Network-specific configuration object.
        :param protocol: LBT protocol variant (e.g., "GAP_LBT", "RS_LBT", "DB_LBT", "CR_LBT").
        :param network: Network to which the transmitter belongs ("gNB" or "AP").
        :param numerology: Numerology of licensed spectrum.
        :param debug: Enables debugging.
        """
        if seed:
            random.seed(seed)

        self.id = id
        self.environment = environment
        self.debug = debug
        self.priority_class = config.priority_classes[priority_class_id]
        self.protocol = protocol
        self.network = network
        self.sync_slot_duration = config.numerologies[numerology].sync_slot_duration if self.network == "gNB" else None
        self.deter_period = config.deter_period
        self.prioritization_slot_duration = config.prioritization_slot_duration
        self.observation_slot_duration = config.observation_slot_duration

        self.cr_slot_duration = getattr(config, "cr_slot_duration", None)
        self.cr_rs_duration = getattr(config, "cr_rs_duration", None)
        self.cr_prob_first_rs = getattr(config, "cr_prob_first_rs", None)
        self.cr_prob_next_rs = getattr(config, "cr_prob_next_rs", None)
        self.cr_max_num_slots = getattr(config, "cr_max_num_slots", None)

        self.last_successful_end_time = None
        self.failed_transmissions_in_a_row = 0
        self.backoff_value = 0
        self.backoff_log = list()
        self.current_transmission = None

        # Learning-related
        self.agent_action = self.priority_class.min_contention_window  # Agent-determined CW min

        self.successful_transmissions = 0
        self.total_transmissions = 0
        self.successful_airtime = 0
        self.total_airtime = 0
        self.collisions = 0
        self.delay_list = list()

    def generate_backoff_value(self):
        """Generate backoff time based on the assigned priority class."""
        # Use agent-selected cw_min for learning mode; otherwise, priority class cw_min
        cw_min = self.priority_class.min_contention_window
        # cw_min = self.agent_action if self.environment.learning_mode else self.priority_class.min_contention_window
        cw_max = self.agent_action \
            if self.environment.mode in ["Training", "Execution"] else self.priority_class.max_contention_window
        upper_limit = min((2 ** self.failed_transmissions_in_a_row) * cw_min, cw_max)
        self.backoff_value = random.randint(0, upper_limit - 1)

        # Log details
        self.backoff_log.append((self.environment.sim_env.now, self.backoff_value))
        self._log(f"CW Min: {cw_min}, CW Max: {cw_max - 1}, Upper Limit: {upper_limit - 1}, Backoff Value: {self.backoff_value}")

    def sense_channel(self, slots, slot_duration):
        try:
            while slots > 0:
                yield self.environment.sim_env.timeout(slot_duration)
                slots -= 1
        except simpy.Interrupt:
            self._log("Sensing the channel was interrupted.", "fail")
        return slots

    def wait_for_idle_channel(self):
        """Wait until the channel is sensed idle."""
        while self.environment.channel.time_until_idle(self.environment.sim_env.now) > 0:
            yield self.environment.sim_env.timeout(self.environment.channel.time_until_idle(self.environment.sim_env.now))

    def wait_for_prioritization_period(self):
        """Perform prioritization based on assigned priority class."""
        slots = self.priority_class.number_of_prioritization_slots
        prioritization_slot_duration = self.prioritization_slot_duration

        while slots > 0:
            self._log("Waiting for channel to be idle ...")
            yield self.environment.sim_env.process(self.wait_for_idle_channel())
            self._log("Waiting for prioritization period ...")
            yield self.environment.sim_env.timeout(self.deter_period)

            if self.environment.channel.time_until_idle(self.environment.sim_env.now) > 0:
                continue

            proc = self.environment.sim_env.process(self.sense_channel(slots, prioritization_slot_duration))
            self.environment.channel.active_senses.append(proc)
            slots = yield proc
            try:
                self.environment.channel.active_senses.remove(proc)
            except ValueError:
                pass

            if slots > 0:
                self._log("channel BUSY - prioritization period failed.", "fail")

    def wait_for_backoff_period(self):
        """Perform backoff based on the calculated backoff value."""
        self._log("Waiting for backoff period with {} slots".format(self.backoff_value))
        if self.environment.channel.time_until_idle(self.environment.sim_env.now) > 0:
            return

        observation_slot_duration = self.observation_slot_duration
        proc = self.environment.sim_env.process(self.sense_channel(self.backoff_value, observation_slot_duration))
        self.environment.channel.active_senses.append(proc)
        self.backoff_value = yield proc
        try:
            self.environment.channel.active_senses.remove(proc)
        except ValueError:
            pass

    def generate_transmission(self, tx_type):
        """Attempt transmission."""
        rs_duration = 0
        airtime_duration = self.priority_class.max_channel_occupancy_time
        current_time = self.environment.sim_env.now
        if self.network == "gNB" and self.protocol in ["RS_LBT", "CR_LBT"]:
            next_boundary = ((current_time // self.sync_slot_duration) + 1) * self.sync_slot_duration
            rs_duration = (next_boundary - current_time) % 500
            airtime_duration -= rs_duration

        return Transmission(
            current_time, airtime_duration, rs_duration, self.network, self.id, self.priority_class.id, tx_type)

    def transmit(self, transmission):
        """Perform the transmission."""
        # self.environment.channel.clean_old_transmissions(self.environment.sim_env.now)
        for p in self.environment.channel.active_senses:
            try:
                if p.is_alive:
                    p.interrupt()
            except RuntimeError:
                self._log("Failed to interrupt a sensing process.", "fail")

        self._log("Starts RS transmission for {} us and packet transmission for {} us."
                  .format(transmission.rs_duration, transmission.airtime_duration))

        self.environment.channel.active_transmissions.append(transmission)
        yield self.environment.sim_env.timeout(transmission.rs_duration + transmission.airtime_duration)
        self._log(f"Transmission of {transmission.tx_type} completed. Channel freed.")

    def record_collision(self):
        self.collisions += 1
        self.failed_transmissions_in_a_row += 1
        self._log(f"Failed transmission for {self.failed_transmissions_in_a_row} times in a row", "fail", force=False)

    def record_successful_transmission(self):
        self.successful_transmissions += 1
        self.successful_airtime += \
            (self.current_transmission.airtime_duration + self.current_transmission.rs_duration)
        self.failed_transmissions_in_a_row = 0
        if self.last_successful_end_time:
            delay = self.current_transmission.end_time - self.last_successful_end_time
            self.delay_list.append(delay)
        self.last_successful_end_time = self.current_transmission.end_time
        self._log("Successful transmission", "success", force=False)

    def wait_for_slot_boundary(self):
        """Wait until the next slot boundary."""
        pass

    def perform_gap_lbt(self):
        """Perform GAP_LBT: Wait until the next slot boundary after backoff."""
        pass

    def perform_cr_lbt(self):  # check this !!!
        """Perform CR_LBT: Divide the gap into cr_slots and randomly act."""

        # self.cr_max_num_slots

        current_time = self.environment.sim_env.now
        next_boundary = ((current_time // self.sync_slot_duration) + 1) * self.sync_slot_duration
        num_cr_slots = (next_boundary - current_time) // self.cr_slot_duration

        for slot in range(num_cr_slots):
            self._log(f"Starting CR-slot {slot + 1}/{num_cr_slots}")

            # Step 1: Transmit the reservation signal
            try:
                rs_duration = self.cr_rs_duration
                cr_small_reserve_tx = Transmission(
                    current_time, 0, rs_duration, self.network, self.id, self.priority_class.id, "cr_small_reserve")
                yield self.environment.sim_env.process(self.transmit(cr_small_reserve_tx))
                self.environment.channel.active_transmissions.remove(cr_small_reserve_tx)
            except simpy.Interrupt:
                self._log("Reservation signal transmission interrupted.", "fail")
                return False  # Abort current slot and restart backoff

            # Step 2: Randomly decide to sense or continue transmitting
            t_sense_duration = self.cr_slot_duration - rs_duration
            decision_prob = self.cr_prob_first_rs if slot == 0 else self.cr_prob_next_rs

            if random.random() < decision_prob:  # True: Sense the channel, False: Continue RS
                self._log(f"Sensing the channel for {t_sense_duration} μs.")
                try:
                    idle_time = self.environment.channel.time_until_idle(self.environment.sim_env.now)
                    if idle_time > 0:
                        self._log(f"Channel is busy. Idle in {idle_time} μs. Restarting backoff.", "fail")
                        return False  # Restart backoff

                    proc = self.environment.sim_env.process(self.sense_channel(1, t_sense_duration))
                    self.environment.channel.active_senses.append(proc)
                    slots = yield proc
                    self.environment.channel.active_senses.remove(proc)
                    if slots != 0:
                        self._log("Sensing the channel interrupted.", "fail")
                        return False
                except simpy.Interrupt:
                    self._log("Sensing the channel interrupted.", "fail")
                    return False
            else:
                try:
                    current_time = self.environment.sim_env.now
                    cr_reserve_tx = Transmission(
                        current_time, 0, t_sense_duration, self.network, self.id, self.priority_class.id, "cr_reserve")
                    yield self.environment.sim_env.process(self.transmit(cr_reserve_tx))
                    self.environment.channel.active_transmissions.remove(cr_reserve_tx)
                except simpy.Interrupt:
                    self._log("Reservation signal interrupted.", "fail")
                    return False

            # # Ensure slot does not overrun
            # if self.environment.sim_env.now > start_slot_time + self.cr_slot_duration:
            #     self._log(f"CR-slot {slot + 1} exceeded duration.", "fail")
            #     break

        self._log("Completed all CR-slots. Proceeding to the next step.")
        return True

    def post_backoff_lbt_behavior(self):
        result = True
        if self.network == "gNB" and self.protocol in ["GAP_LBT", "RS_LBT", "DB_LBT", "CR_LBT"]:
            if self.protocol == "GAP_LBT":
                yield self.environment.sim_env.process(self.perform_gap_lbt())
            elif self.protocol == "CR_LBT":
                result = yield self.environment.sim_env.process(self.perform_cr_lbt())
        return result

    def _log(self, output, output_type=None, force=False):
        log("{}|{}-{}\t: {}".format(self.environment.sim_env.now, self.network.rjust(3), self.id, output), output_type, self.debug, force)

    def run(self):
        """Run method for continuous operation."""
        while True:
            self._log("Attempting a new transmission")
            self.generate_backoff_value()

            if self.backoff_value == 0:
                yield self.environment.sim_env.process(self.wait_for_prioritization_period())

            while self.backoff_value > 0:
                yield self.environment.sim_env.process(self.wait_for_prioritization_period())
                yield self.environment.sim_env.process(self.wait_for_backoff_period())

            self._log("Finished backoff process")
            result = yield self.environment.sim_env.process(self.post_backoff_lbt_behavior())
            if not result:
                continue

            self.current_transmission = self.generate_transmission(tx_type="packet")
            yield self.environment.sim_env.process(self.transmit(self.current_transmission))
            success = not self.environment.channel.is_collision(self.current_transmission)
            self.total_transmissions += 1
            self.total_airtime += self.current_transmission.airtime_duration + self.current_transmission.rs_duration

            if success:
                self.record_successful_transmission()
            else:
                self.record_collision()
                if self.failed_transmissions_in_a_row > \
                        self.priority_class.max_backoff_stages + self.priority_class.max_retransmissions:
                    if self.last_successful_end_time:
                        self.last_successful_end_time = self.current_transmission.end_time
                    self.failed_transmissions_in_a_row = 0

            self.current_transmission = None
