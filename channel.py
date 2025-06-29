class Channel:
    def __init__(self):
        self.active_transmissions = []  # Tracks ongoing transmissions
        self.active_senses = []  # Track ongoing senses
        self.collisions = {
            "inter": 0,  # Inter-network collisions (gNB vs. AP)
            "intra_gnb": 0,  # Intra-network collisions within gNBs
            "intra_ap": 0,  # Intra-network collisions within APs
        }
        self.total_time = 0  # Total simulation time
        self.busy_time = 0  # Time the channel was busy

    def get_busy_time_ratio(self):
        """
        Calculate the ratio of time the channel was busy.
        :return: Ratio of busy time to total time.
        """
        return self.busy_time / self.total_time if self.total_time > 0 else 0

    def check_collision(self, transmission, current_time):
        """
        Add a transmission to the channel and check for inter/intra-network collisions.
        :param transmission: Transmission object
        :param current_time: Current time
        :return: True if successful, False if a collision occurred
        """

        overlapping_transmissions = [
            t for t in self.active_transmissions
            if (t is not transmission) and (transmission.end_time > t.start_time and transmission.start_time < t.end_time)
        ]

        if overlapping_transmissions:
            # Determine type of collision
            same_network = all(t.network == transmission.network for t in overlapping_transmissions)
            if same_network:
                if transmission.network == "gNB":
                    self.collisions["intra_gnb"] += 1
                elif transmission.network == "AP":
                    self.collisions["intra_ap"] += 1
            else:
                self.collisions["inter"] += 1
            return False  # Indicate collision

        return True  # Indicate success

    def is_collision(self, transmission):
        """
        Determine if a given transmission is part of a collision.
        :param transmission: The transmission to check.
        :return: True if the transmission is part of a collision, False otherwise.
        """
        for other_transmission in self.active_transmissions:
            # Skip checking the transmission against itself
            if other_transmission == transmission:
                continue

            # Check if the transmission overlaps with another active transmission
            if not (transmission.end_time <= other_transmission.start_time or
                    transmission.start_time >= other_transmission.end_time):
                return True  # Overlap detected, collision occurs

        return False  # No overlap, no collision

    def clean_old_transmissions(self, current_time):
        """
        Remove transmissions that have ended by the given current time.
        :param current_time: The current simulation time.
        """
        self.active_transmissions = [
            t for t in self.active_transmissions if t.tx_type == 'packet' and t.end_time > current_time
        ]

    def time_until_idle(self, current_time):
        """
        Calculate the remaining time until the channel is idle (no active transmissions).
        :param current_time: Current simulation time.
        :return: Time until the channel becomes idle.
        """
        max_time = 0
        for t in self.active_transmissions:
            time_left = t.end_time - current_time
            if time_left > max_time:
                max_time = time_left
        return max_time
