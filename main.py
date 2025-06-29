import numpy as np
import pandas as pd
import torch
import time
import random
from matplotlib import pyplot as plt

from config import GNBConfig, APConfig, PriorityClass, Numerology
from environment import NetworkEnvironment
from utils import network_performance_observation, moving_average, summarize_backoff_logs
from model import QNetwork, MultiHeadQNetwork
from dqn_agent import DQNAgent

# Define priority classes
gnb_priority_classes = [
    PriorityClass(id=1,
                  number_of_prioritization_slots=1,
                  min_contention_window=4,
                  max_contention_window=8,
                  max_backoff_stages=1,
                  max_channel_occupancy_time=2000,
                  max_retransmissions=0),
    PriorityClass(id=2,
                  number_of_prioritization_slots=1,
                  min_contention_window=8,
                  max_contention_window=16,
                  max_backoff_stages=1,
                  max_channel_occupancy_time=3000,
                  max_retransmissions=0),
    PriorityClass(id=3,
                  number_of_prioritization_slots=3,
                  min_contention_window=16,
                  max_contention_window=64,
                  max_backoff_stages=2,
                  max_channel_occupancy_time=8000,
                  max_retransmissions=0),
    PriorityClass(id=4,
                  number_of_prioritization_slots=7,
                  min_contention_window=16,
                  max_contention_window=1024,
                  max_backoff_stages=6,
                  max_channel_occupancy_time=8000,
                  max_retransmissions=0),
]
ap_priority_classes = [
    PriorityClass(id=1,
                  number_of_prioritization_slots=2,
                  min_contention_window=4,
                  max_contention_window=8,
                  max_backoff_stages=1,
                  max_channel_occupancy_time=2000,
                  max_retransmissions=0),
    PriorityClass(id=2,
                  number_of_prioritization_slots=2,
                  min_contention_window=8,
                  max_contention_window=16,
                  max_backoff_stages=1,
                  max_channel_occupancy_time=4000,
                  max_retransmissions=0),
    PriorityClass(id=3,
                  number_of_prioritization_slots=3,
                  min_contention_window=16,
                  max_contention_window=1024,
                  max_backoff_stages=6,
                  max_channel_occupancy_time=8000,
                  max_retransmissions=0),
    PriorityClass(id=4,
                  number_of_prioritization_slots=7,
                  min_contention_window=16,
                  max_contention_window=1024,
                  max_backoff_stages=6,
                  max_channel_occupancy_time=8000,
                  max_retransmissions=0),
]
numerologies = [
    Numerology(id=0, sync_slot_duration=1000, mini_slot_duration=72, max_sync_slot_desync=1000, min_sync_slot_desync=0),
    Numerology(id=1, sync_slot_duration=500, mini_slot_duration=36, max_sync_slot_desync=500, min_sync_slot_desync=0),
    Numerology(id=2, sync_slot_duration=250, mini_slot_duration=18, max_sync_slot_desync=250, min_sync_slot_desync=0),
    Numerology(id=3, sync_slot_duration=125, mini_slot_duration=9, max_sync_slot_desync=125, min_sync_slot_desync=0)
]

# Define configurations for gNBs and APs
gnb_config = GNBConfig(
    priority_classes=gnb_priority_classes,
    numerologies=numerologies
)
ap_config = APConfig(priority_classes=ap_priority_classes)

# Hyperparameters
SIM_TIME = 20 * 1e6
SEED_COUNT = 100
DEBUG = False
VALIDATION = False
MAX_TRANSMITTERS = 50
FIXED_TRANSMITTERS = False
NUM_TRANSMITTERS = 25
EPSILON_MIN = 0.01
NUMEROLOGY = 1
GNB_PROTOCOL = "RS_LBT"
ENV_STEP_DURATION = numerologies[NUMEROLOGY].sync_slot_duration * 5
NUM_STATES = 8

NUM_EPISODES_TRAIN = 1000
MAX_STEPS_PER_EPISODE = 500
EPSILON_DECAY = 0.99
NUM_EPISODES_EVAL = 200
T0 = 5  # Dual update interval
TAU = 50  # Target network update interval

PRIMAL_DUAL = False
AUGMENTED_STATE = False
MULTI_OBJECTIVE = True
ALPHA = 0.3
if AUGMENTED_STATE or PRIMAL_DUAL:
    constraints = {
        "smoothed_delay_pc1": 2.0,
    }
else:
    constraints = {}


def train_dqn_agent(seed=None, model_save_path='results/dqn_trained_model.pth'):
    if FIXED_TRANSMITTERS:
        train_dqn_agent_fixed_transmitters(seed=seed, model_save_path=model_save_path)
    else:
        train_dqn_agent_variable_transmitters(seed=seed, model_save_path=model_save_path)


def train_dqn_agent_fixed_transmitters(seed=None, model_save_path='results/dqn_trained_model.pth'):
    if seed:
        random.seed(seed)

    num_states = NUM_STATES
    if AUGMENTED_STATE:
        num_states += len(constraints)
    num_actions_pc1 = 7  # Actions for gNB PC1
    num_actions_pc3 = 7  # Actions for gNB PC3
    num_actions_ap = 7  # Actions for AP PC3

    agent = DQNAgent(num_states,
                     num_actions_pc1, num_actions_pc3, num_actions_ap,
                     augmented_reward=(AUGMENTED_STATE or PRIMAL_DUAL),
                     multi_objective=MULTI_OBJECTIVE,
                     alpha=ALPHA,
                     constraints=constraints)

    env = NetworkEnvironment(
        None,
        gnb_config,
        ap_config,
        num_gnbs=NUM_TRANSMITTERS + 1,
        num_aps=NUM_TRANSMITTERS,
        step_duration=ENV_STEP_DURATION,
        gnb_protocol=GNB_PROTOCOL,
        gnb_priorities=[1 if i == 0 else 3 for i in range(NUM_TRANSMITTERS + 1)],
        ap_priorities=[3 for _ in range(NUM_TRANSMITTERS)],
        numerology=NUMEROLOGY,
        debug=DEBUG,
        mode="Training",
        multi_objective=MULTI_OBJECTIVE,
        alpha=ALPHA,
        primal_dual=PRIMAL_DUAL,
        augmented_state=AUGMENTED_STATE,
        constraints=constraints,
    )
    training_rewards = []
    training_losses = []
    validation_rewards = []
    validation_losses = []
    pc1_smoothed_delay = []
    pc1_avg_delay = []
    pc1_last_delay = []
    jfi = []
    lambda_values = {key: [] for key in constraints.keys()}

    for episode in range(1, NUM_EPISODES_TRAIN + 1):
        state = env.reset(randomDual=True if AUGMENTED_STATE else False)
        start_time = time.time()
        episode_pc1_smoothed_delay = 0
        episode_pc1_avg_delay = 0
        episode_pc1_last_delay = 0
        episode_jfi = 0
        episode_reward = 0
        episode_loss = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            pc1_action, pc3_action, ap_action = agent.act(state)
            pc1_cw = 2 ** pc1_action
            pc3_cw = 2 ** (pc3_action + 4)
            ap_cw = 2 ** (ap_action + 4)

            next_state, reward, violation = env.step("DQN", (pc1_cw, pc3_cw, ap_cw))
            episode_reward += reward
            if AUGMENTED_STATE or PRIMAL_DUAL:
                episode_reward -= env.lambda_values["smoothed_delay_pc1"] * violation

            agent.remember(state, (pc1_action, pc3_action, ap_action), reward, violation, next_state, False)

            if not PRIMAL_DUAL:
                agent.replay(env.lambda_values)

            if PRIMAL_DUAL and (step + 1) % T0 == 0:
                agent.replay(env.lambda_values)
                env.update_dual_variables()

            state = next_state

            episode_pc1_smoothed_delay += next_state[6]
            episode_pc1_avg_delay += next_state[0]
            episode_pc1_last_delay += next_state[1]
            episode_jfi += next_state[7]

            if agent.loss_history:
                episode_loss += agent.loss_history[-1]

            if (step + 1) % TAU == 0:
                agent.update_target_network()

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        avg_reward = episode_reward / MAX_STEPS_PER_EPISODE
        avg_loss = episode_loss / MAX_STEPS_PER_EPISODE
        avg_pc1_smoothed_delay = episode_pc1_smoothed_delay / MAX_STEPS_PER_EPISODE
        avg_pc1_avg_delay = episode_pc1_avg_delay / MAX_STEPS_PER_EPISODE
        avg_pc1_last_delay = episode_pc1_last_delay / MAX_STEPS_PER_EPISODE
        avg_jfi = episode_jfi / MAX_STEPS_PER_EPISODE
        training_losses.append(avg_loss)
        training_rewards.append(avg_reward)
        pc1_smoothed_delay.append(avg_pc1_smoothed_delay)
        pc1_avg_delay.append(avg_pc1_avg_delay)
        pc1_last_delay.append(avg_pc1_last_delay)
        jfi.append(avg_jfi)

        for key in lambda_values.keys():
            lambda_values[key].append(env.lambda_values[key])

        end_time = time.time()
        episode_duration = end_time - start_time  # Calculate duration

        avg_val_reward = None
        avg_val_loss = None

        validation_rewards.append(avg_val_reward)
        validation_losses.append(avg_val_loss)

        print(f"Episode {episode}/{NUM_EPISODES_TRAIN} | "
              f"# PC3 APs: {NUM_TRANSMITTERS:02d} | "
              f"Reward: {avg_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"PC1 Delay: {avg_pc1_smoothed_delay:.4f} | "
              f"Lambda: {env.lambda_values} | "
              f"JFI: {avg_jfi:.2f} | "
              f"Epsilon: {agent.epsilon:.2f} | "
              f"Duration: {episode_duration:.2f} seconds")

    metrics_df = {
        "episode": list(range(1, NUM_EPISODES_TRAIN + 1)),
        "training_rewards": training_rewards,
        "training_losses": training_losses,
        "validation_rewards": validation_rewards,
        "validation_losses": validation_losses,
        "smoothed_delay_pc1": pc1_smoothed_delay,
        "avg_delay_pc1": pc1_avg_delay,
        "last_delay_pc1": pc1_last_delay,
        "jfi": jfi,
        "lambda_smoothed_delay_pc1": lambda_values["smoothed_delay_pc1"]
    }
    pd.DataFrame(metrics_df).to_csv("results/dqn_episode_metrics.csv", index=False)

    agent.save("results/dqn_trained_model.pth")
    print(f"Training complete. Model saved to '{model_save_path}'. Metrics saved to 'dqn_episode_metrics.csv'.")


def train_dqn_agent_variable_transmitters(seed=None, model_save_path='results/dqn_trained_model.pth'):
    if seed:
        random.seed(seed)

    num_states = NUM_STATES
    if AUGMENTED_STATE:
        num_states += len(constraints)
    num_actions_pc1 = 7  # Actions for gNB PC1
    num_actions_pc3 = 7  # Actions for gNB PC1
    num_actions_ap = 7  # Actions for AP PC3

    agent = DQNAgent(num_states,
                     num_actions_pc1, num_actions_pc3, num_actions_ap,
                     augmented_reward=(AUGMENTED_STATE or PRIMAL_DUAL),
                     multi_objective=MULTI_OBJECTIVE,
                     alpha=ALPHA,
                     constraints=constraints)

    training_rewards = []
    training_losses = []
    validation_rewards = []
    validation_losses = []
    pc1_smoothed_delay = []
    pc1_avg_delay = []
    pc1_last_delay = []
    jfi = []

    for episode in range(1, NUM_EPISODES_TRAIN + 1):
        start_time = time.time()

        num_transmitters = random.randint(0, MAX_TRANSMITTERS)

        env = NetworkEnvironment(
            None,
            gnb_config,
            ap_config,
            num_gnbs=num_transmitters + 1,
            num_aps=num_transmitters,
            step_duration=ENV_STEP_DURATION,
            gnb_protocol=GNB_PROTOCOL,
            gnb_priorities=[1 if i == 0 else 3 for i in range(num_transmitters + 1)],
            ap_priorities=[3 for _ in range(num_transmitters)],
            numerology=NUMEROLOGY,
            debug=DEBUG,
            mode="Training",
            multi_objective=MULTI_OBJECTIVE,
            alpha=ALPHA,
            primal_dual=PRIMAL_DUAL,
            augmented_state=AUGMENTED_STATE,
            constraints=constraints,
        )
        state = env.reset(randomDual=True)
        episode_pc1_smoothed_delay = 0
        episode_pc1_avg_delay = 0
        episode_pc1_last_delay = 0
        episode_pc1_collision = 0
        episode_channel_utilization = 0
        episode_jfi = 0
        episode_reward = 0
        episode_loss = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            pc1_action, pc3_action, ap_action = agent.act(state)
            pc1_cw = 2 ** pc1_action
            pc3_cw = 2 ** (pc3_action + 4)
            ap_cw = 2 ** (ap_action + 4)

            next_state, reward, violation = env.step("DQN", (pc1_cw, pc3_cw, ap_cw))
            episode_reward += reward
            if AUGMENTED_STATE or PRIMAL_DUAL:
                episode_reward -= env.lambda_values["smoothed_delay_pc1"] * violation

            agent.remember(state, (pc1_action, pc3_action, ap_action), reward, violation, next_state, False)

            if not PRIMAL_DUAL:
                agent.replay(env.lambda_values)

            if PRIMAL_DUAL and (step + 1) % T0 == 0:
                agent.replay(env.lambda_values)
                env.update_dual_variables()

            state = next_state

            episode_pc1_avg_delay += next_state[0]
            episode_pc1_collision += next_state[1]
            episode_channel_utilization += next_state[2]
            episode_pc1_smoothed_delay += next_state[6]
            episode_jfi += next_state[7]

            if agent.loss_history:
                episode_loss += agent.loss_history[-1]

            if (step + 1) % TAU == 0:
                agent.update_target_network()

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        avg_reward = episode_reward / MAX_STEPS_PER_EPISODE
        avg_loss = episode_loss / MAX_STEPS_PER_EPISODE
        avg_pc1_smoothed_delay = episode_pc1_smoothed_delay / MAX_STEPS_PER_EPISODE
        avg_pc1_avg_delay = episode_pc1_avg_delay / MAX_STEPS_PER_EPISODE
        avg_pc1_last_delay = episode_pc1_last_delay / MAX_STEPS_PER_EPISODE
        avg_pc1_collision = episode_pc1_collision / MAX_STEPS_PER_EPISODE
        avg_channel_utilization = episode_channel_utilization / MAX_STEPS_PER_EPISODE
        avg_jfi = episode_jfi / MAX_STEPS_PER_EPISODE
        training_losses.append(avg_loss)
        training_rewards.append(avg_reward)
        pc1_smoothed_delay.append(avg_pc1_smoothed_delay)
        pc1_avg_delay.append(avg_pc1_avg_delay)
        pc1_last_delay.append(avg_pc1_last_delay)
        jfi.append(avg_jfi)

        end_time = time.time()
        episode_duration = end_time - start_time  # Calculate duration

        avg_val_reward = None
        avg_val_loss = None

        validation_rewards.append(avg_val_reward)
        validation_losses.append(avg_val_loss)

        print(f"Episode {episode}/{NUM_EPISODES_TRAIN} | "
              f"# PC3 APs: {num_transmitters:02d} | "
              f"Reward: {avg_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"PC1 Delay: {avg_pc1_smoothed_delay:.4f} | "
              f"JFI: {avg_jfi:.2f} | "
              f"PC1 Collision: {avg_pc1_collision:.2f} | "
              f"Ch. Utilization: {avg_channel_utilization:.2f} | "
              f"Duration: {episode_duration:.2f} seconds")

    metrics_df = {
        "episode": list(range(1, NUM_EPISODES_TRAIN + 1)),
        "training_rewards": training_rewards,
        "training_losses": training_losses,
        "validation_rewards": validation_rewards,
        "validation_losses": validation_losses,
        "smoothed_delay_pc1": pc1_smoothed_delay,
        "avg_delay_pc1": pc1_avg_delay,
        "last_delay_pc1": pc1_last_delay,
        "jfi": jfi
    }
    pd.DataFrame(metrics_df).to_csv("results/dqn_episode_metrics.csv", index=False)

    agent.save("results/dqn_trained_model.pth")
    print(f"Training complete. Model saved to '{model_save_path}'. Metrics saved to 'dqn_episode_metrics.csv'.")


def execute_dqn_policy(model_path='results/dqn_trained_model.pth'):
    """
    Evaluate the policy by running a learned-based or fixed policy simulation.
    :param model_path: Path to the saved DQN model file (only used for learned-based policy).
    """
    num_states = NUM_STATES
    if AUGMENTED_STATE:
        num_states += len(constraints)
    num_actions_pc1 = 7  # Actions for gNB PC1
    num_actions_pc3 = 7  # Actions for gNB PC3
    num_actions_ap = 7  # Actions for AP PC3
    num_actions = num_actions_pc1 * num_actions_pc3 * num_actions_ap

    if MULTI_OBJECTIVE:
        q_network = MultiHeadQNetwork(input_dim=num_states, output_dim=num_actions)
    else:
        q_network = QNetwork(input_dim=num_states, output_dim=num_actions)

    q_network.load_state_dict(torch.load(model_path, weights_only=True))
    q_network.eval()

    seeds = [random.randint(10, 1000) for _ in range(SEED_COUNT)]
    transmitters_range = [NUM_TRANSMITTERS] if FIXED_TRANSMITTERS else range(0, 51)

    for num_transmitters in transmitters_range:
        for i, seed in enumerate(seeds):
            env = NetworkEnvironment(
                seed=None,
                gnb_config=gnb_config,
                ap_config=ap_config,
                num_gnbs=num_transmitters + 1,
                num_aps=num_transmitters,
                step_duration=ENV_STEP_DURATION,
                gnb_protocol=GNB_PROTOCOL,
                gnb_priorities=[1 if i == 0 else 3 for i in range(num_transmitters + 1)],
                ap_priorities=[3 for _ in range(num_transmitters)],
                numerology=NUMEROLOGY,
                debug=DEBUG,
                mode="Execution",
                multi_objective=MULTI_OBJECTIVE,
                alpha=ALPHA,
                primal_dual=PRIMAL_DUAL,
                augmented_state=AUGMENTED_STATE,
                constraints=constraints
            )
            state = env.reset(randomDual=False)
            pc1_smoothed_delay = []
            pc1_avg_delay = []
            pc1_last_delay = []
            jfi = []
            lambda_values = {key: [] for key in constraints.keys()}

            for episode in range(1, NUM_EPISODES_EVAL + 1):
                with torch.no_grad():
                    if MULTI_OBJECTIVE:
                        q_delay, q_fair = q_network(torch.FloatTensor(state))
                        combined_q = ALPHA * q_delay + (1 - ALPHA) * q_fair
                        action_idx = combined_q.argmax().item()

                    else:
                        q_values = q_network(torch.FloatTensor(state))
                        action_idx = q_values.argmax().item()

                pc1_action = action_idx // (num_actions_pc3 * num_actions_ap)
                pc3_action = (action_idx % (num_actions_pc3 * num_actions_ap)) // num_actions_ap
                ap_action = action_idx % num_actions_ap

                pc1_cw = 2 ** pc1_action
                pc3_cw = 2 ** (pc3_action + 4)
                ap_cw = 2 ** (ap_action + 4)

                next_state, _, _ = env.step("DQN", (pc1_cw, pc3_cw, ap_cw))

                if (AUGMENTED_STATE or PRIMAL_DUAL) and ((episode + 1) % T0 == 0):
                    env.update_dual_variables()

                # if AUGMENTED_STATE and episode == 1000:
                #     env.reset_cumulative_network_metrics()
                #     env.reset_transmitters()

                state = next_state
                pc1_smoothed_delay.append(next_state[6])
                pc1_avg_delay.append(next_state[0])
                pc1_last_delay.append(next_state[1])
                jfi.append(next_state[7])

                for key in lambda_values.keys():
                    lambda_values[key].append(env.lambda_values[key])

            env.cumulative_network_metrics[("gNB", "PC1")]["smoothed_delay"] = pc1_smoothed_delay
            network_performance_observation(seed, env.gnbs, env.aps, env.cumulative_network_metrics, GNB_PROTOCOL, "DQN", constraints, dump_csv=True)
            print(f"{i:02d} | seed #{seed:03d} | "
                  f"#gnb/ap: {num_transmitters}/{num_transmitters} | "
                  f"PC1 Delay: {np.average(pc1_smoothed_delay):.4f} | "
                  f"JFI: {jfi[-1]:.2f}")

        print(f"--------------------------------------")


def execute_fixed_policy():
    total_steps = round(SIM_TIME / ENV_STEP_DURATION)
    seeds = [random.randint(10, 1000) for _ in range(SEED_COUNT)]
    # seeds = [15]

    for num_transmitters in range(0, MAX_TRANSMITTERS + 1):
        for i, seed in enumerate(seeds):
            env = NetworkEnvironment(
                seed=None,
                gnb_config=gnb_config,
                ap_config=ap_config,
                num_gnbs=num_transmitters + 1,
                num_aps=num_transmitters,
                step_duration=ENV_STEP_DURATION,
                gnb_protocol=GNB_PROTOCOL,
                gnb_priorities=[1 if i == 0 else 3 for i in range(num_transmitters + 1)],
                ap_priorities=[3 for _ in range(num_transmitters)],
                numerology=NUMEROLOGY,
                debug=DEBUG,
                mode="Fixed",
                augmented_state=AUGMENTED_STATE,
                primal_dual=PRIMAL_DUAL
            )

            for step in range(total_steps):
                env.step("FIXED")

            # Record performance
            network_performance_observation(seed, env.gnbs, env.aps, env.cumulative_network_metrics, GNB_PROTOCOL, "FIXED", constraints, dump_csv=True)

            print('{:02d} - seed #{:03d} - #gnb/ap: {}/{}'.format(i, seed, num_transmitters, num_transmitters))

            # summarize_backoff_logs([gnb for gnb in env.gnbs if gnb.priority_class.id == 1], "gNB PC1")
            # summarize_backoff_logs([gnb for gnb in env.gnbs if gnb.priority_class.id == 3], "gNB PC3")
            # summarize_backoff_logs([ap for ap in env.aps if ap.priority_class.id == 3], "AP PC3")

        print(f"--------------------------------------")


if __name__ == "__main__":

    train_dqn_agent()
    # execute_fixed_policy()
    # execute_dqn_policy()
