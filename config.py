class PriorityClass:
    def __init__(self,
                 id,
                 number_of_prioritization_slots,
                 min_contention_window,
                 max_contention_window,
                 max_backoff_stages,
                 max_channel_occupancy_time,
                 max_retransmissions
                 ):
        """
        Defines a priority class with specific channel access parameters.
        :param id: Priority class ID.
        :param number_of_prioritization_slots: Number of prioritization slots.
        :param min_contention_window: Minimum contention window size.
        :param max_contention_window: Maximum contention window size.
        :param max_backoff_stages: Maximum backoff stages.
        :param max_channel_occupancy_time: Maximum channel occupancy time.
        :param max_retransmissions: Maximum number of retransmissions allowed.
        """
        self.id = id
        self.number_of_prioritization_slots = number_of_prioritization_slots
        self.min_contention_window = min_contention_window
        self.max_contention_window = max_contention_window
        self.max_backoff_stages = max_backoff_stages
        self.max_channel_occupancy_time = max_channel_occupancy_time
        self.max_retransmissions = max_retransmissions


class Numerology:
    def __init__(self, id, sync_slot_duration, mini_slot_duration, max_sync_slot_desync, min_sync_slot_desync):
        """
        :param sync_slot_duration:
        :param mini_slot_duration:
        :param max_sync_slot_desync:
        :param min_sync_slot_desync:
        """
        self.id = id
        self.sync_slot_duration = sync_slot_duration
        self.mini_slot_duration = mini_slot_duration
        self.max_sync_slot_desync = max_sync_slot_desync
        self.min_sync_slot_desync = min_sync_slot_desync


class GNBConfig:
    def __init__(self,
                 priority_classes,
                 numerologies,
                 mini_slot=False,
                 deter_period=16,
                 prioritization_slot_duration=9,
                 observation_slot_duration=9,
                 cr_slot_duration=30,
                 cr_rs_duration=8,
                 cr_prob_first_rs=0.5,
                 cr_prob_next_rs=0.5,
                 cr_max_num_slots=None):
        """
        Configuration for 5G NR-U gNBs.
        :param priority_classes: List of priority classes with parameters.
        :param numerologies:
        :param mini_slot:
        :param deter_period:
        :param prioritization_slot_duration:
        :param cr_slot_duration: Duration of CR slots (in time units) for CR_LBT.
        """
        self.priority_classes = {pc.id: pc for pc in priority_classes}
        self.numerologies = {n.id: n for n in numerologies}
        self.mini_slot = mini_slot
        self.deter_period = deter_period
        self.prioritization_slot_duration = prioritization_slot_duration
        self.observation_slot_duration = observation_slot_duration
        self.cr_slot_duration = cr_slot_duration
        self.cr_rs_duration = cr_rs_duration
        self.cr_prob_first_rs = cr_prob_first_rs
        self.cr_prob_next_rs = cr_prob_next_rs
        self.cr_max_num_slots = cr_max_num_slots


class APConfig:
    def __init__(self,
                 priority_classes,
                 deter_period=16,
                 prioritization_slot_duration=9,
                 observation_slot_duration=9,):
        """
        Configuration for Wi-Fi APs.
        :param priority_classes: List of priority classes with parameters.
        """
        self.priority_classes = {pc.id: pc for pc in priority_classes}
        self.deter_period = deter_period
        self.prioritization_slot_duration = prioritization_slot_duration
        self.observation_slot_duration = observation_slot_duration

