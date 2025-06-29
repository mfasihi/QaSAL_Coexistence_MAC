import csv
import math
import os
import time
import numpy as np
from collections import defaultdict
import torch


def log(output, output_type, debug, force):
    if debug or force:
        if output_type == "fail":
            output = "\033[91m" + output + "\033[0m"
        elif output_type == "success":
            output = "\033[92m" + output + "\033[0m"
        print(output)


def process_entity(entity, results):
    """Process a single gNB or AP entity."""
    res = {
        'id': entity.id,
        'PC': entity.priority_class.id,
        'network': entity.network,
        'successful_transmissions': entity.successful_transmissions,
        'total_transmissions': entity.total_transmissions,
        'successful_airtime': entity.successful_airtime,
        'total_airtime': entity.total_airtime,
        # 'collisions': entity.collisions / entity.total_transmissions if entity.total_transmissions > 0 else 0,
        # 'trans_delay': np.mean(entity.delay_list) if len(entity.delay_list) > 0 else 0,
        'efficiency': (entity.successful_airtime / entity.total_airtime) if entity.total_airtime > 0 else 0,
    }
    results.append(res)
    return results


def compute_aggregate_metrics(results):
    """Compute aggregated metrics for gNB or AP."""
    metrics = defaultdict(lambda: 0)
    num_transmitters = defaultdict(lambda: 0)

    for i in range(1, 5):
        for p in ["gNB", "AP"]:
            metrics[f'succ_trans_{p}_PC{i}'] = 0
            metrics[f'total_trans_{p}_PC{i}'] = 0
            metrics[f'succ_airtime_{p}_PC{i}'] = 0
            metrics[f'total_airtime_{p}_PC{i}'] = 0
            # metrics[f'collisions_{p}_PC{i}'] = 0
            # metrics[f'trans_delay_{p}_PC{i}'] = 0
            metrics[f'efficiency_{p}_PC{i}'] = 0
            num_transmitters[f'{p}_PC{i}'] = 0

    for res in results:
        PC = f"{'PC'}{res['PC']}"
        network = res['network']
        num_transmitters[f'{network}_{PC}'] += 1
        metrics[f'succ_trans_{network}_{PC}'] += res['successful_transmissions']
        metrics[f'total_trans_{network}_{PC}'] += res['total_transmissions']
        metrics[f'succ_airtime_{network}_{PC}'] += res['successful_airtime']
        metrics[f'total_airtime_{network}_{PC}'] += res['total_airtime']
        # metrics[f'collisions_{network}_{PC}'] += res['collisions']
        # metrics[f'trans_delay_{network}_{PC}'] += res['trans_delay']
        metrics[f'efficiency_{network}_{PC}'] += res['efficiency']

    num_transmitters['gNB_total'] = num_transmitters['AP_total'] = 0
    for i in range(1, 5):
        for p in ["gNB", "AP"]:
            num_transmitters[f'{p}_total'] += num_transmitters[f'{p}_PC{i}']
            n = num_transmitters[f'{p}_PC{i}'] if num_transmitters[f'{p}_PC{i}'] != 0 else 1
            # metrics[f'collisions_{p}_PC{i}'] /= n
            # metrics[f'trans_delay_{p}_PC{i}'] /= n
            metrics[f'efficiency_{p}_PC{i}'] /= n

    res = dict()
    for key in metrics:
        res[key] = round(metrics[key], 2)

    return res, num_transmitters


def dump_to_csv(results, filename='results.csv', output_dir='results/'):
    """
    Write results to a CSV file.
    :param results: Dictionary containing results to be written.
    :param filename: Name of the output CSV file.
    :param output_dir: Directory where the file will be saved.
    """
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    write_header = not os.path.isfile(filepath)  # Check if the file already exists
    with open(filepath, mode='a', newline='') as csv_file:  # Use newline='' for compatibility
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def performance_observation(seed, gnb_list, ap_list, protocol, policy, dump_csv=True):
    # Process gNBs and APs
    results = []
    for transmitter in gnb_list + ap_list:
        results = process_entity(transmitter, results)

    # Aggregate metrics for gNBs and APs
    metrics, num_transmitters = compute_aggregate_metrics(results)

    # Generate the result dictionary
    now = time.localtime()
    csv_return = {
        "seed": seed,
        "time": time.strftime("%H:%M:%S", now),
        "protocol": protocol,
        "policy": policy,
        **num_transmitters,
        **metrics,
    }
    if dump_csv:
        filename = f"network_performance_vs_num_nodes_{protocol}{'_' + policy}.csv"
        dump_to_csv(csv_return, filename)

    return csv_return


def network_performance_observation(
        seed, gnb_list, ap_list, cumulative_network_metrics, protocol, policy, constraints, dump_csv=True):
    """Calculates and exports the metric of whole network"""

    results = []
    for transmitter in gnb_list + ap_list:
        results = process_entity(transmitter, results)

    # Aggregate metrics for gNBs and APs
    metrics, num_transmitters = compute_aggregate_metrics(results)
    jfi_total = jain_fairness_index_between_networks(gnb_list, ap_list, priority_class=1)
    jfi_pc3 = jain_fairness_index_between_networks(gnb_list, ap_list, priority_class=3)

    network_metrics = defaultdict(lambda: 0)
    for i in range(1, 5):
        for p in ["gNB", "AP"]:
            num_tx = num_transmitters[f'{p}_PC{i}']
            network_metrics[f'avg_delay_{p}_PC{i}'] = 0
            network_metrics[f'total_delay_{p}_PC{i}'] = 0
            network_metrics[f'collision_{p}_PC{i}'] = 0
            if num_tx != 0:
                delay_list = np.array(cumulative_network_metrics[(p, f'PC{i}')]["delay"])
                avg_delay = np.mean(delay_list[1:-1]) if len(delay_list) > 2 else 0
                delay_total = avg_delay * num_tx
                network_metrics[f'avg_delay_{p}_PC{i}'] = avg_delay
                network_metrics[f'total_delay_{p}_PC{i}'] = delay_total

                if p == "gNB" and i == 1:
                    smoothed_delay_pc1 = np.array(cumulative_network_metrics[("gNB", "PC1")]["delay"][-5:])
                    avg_smoothed_delay_pc1 = np.mean(smoothed_delay_pc1)
                    network_metrics[f'avg_smoothed_delay_{p}_PC{i}'] = avg_smoothed_delay_pc1.item()
                    violation_ratio_delay = 0.0
                    violation_ratio_smoothed_delay_pc1 = 0.0

                    if constraints:
                        num_violations_delay = np.sum(
                            delay_list * 1e-3 > constraints["smoothed_delay_pc1"]) if constraints else 0
                        violation_ratio_delay = num_violations_delay / len(delay_list) if len(delay_list) != 0 else 0

                        num_violations_smoothed_delay_pc1 = np.sum(
                            smoothed_delay_pc1 * 1e-3 > constraints["smoothed_delay_pc1"]) if constraints else 0
                        violation_ratio_smoothed_delay_pc1 = num_violations_smoothed_delay_pc1 / len(smoothed_delay_pc1) if len(smoothed_delay_pc1) != 0 else 0

                    network_metrics[f'delay_violation_ratio_{p}_PC{i}'] = violation_ratio_delay
                    network_metrics[f'smoothed_delay_violation_ratio_{p}_PC{i}'] = violation_ratio_smoothed_delay_pc1

                collisions = cumulative_network_metrics[(p, f'PC{i}')]["collisions"]
                transmissions = cumulative_network_metrics[(p, f'PC{i}')]["transmissions"]
                collision_percent = collisions / transmissions if transmissions != 0 else 0
                network_metrics[f'collision_{p}_PC{i}'] = collision_percent

    now = time.localtime()
    csv_return = {
        "seed": seed,
        "time": time.strftime("%H:%M:%S", now),
        "protocol": protocol,
        "policy": policy,
        **num_transmitters,
        **network_metrics,
        **metrics,
        "jfi_total": jfi_total,
        "jfi_PC3": jfi_pc3
    }
    if dump_csv:
        filename = f"network_performance_vs_num_nodes_{protocol}{'_' + policy}.csv"
        dump_to_csv(csv_return, filename)

    return csv_return


def summarize_backoff_logs(transmitters, transmitter_type):
    """Summarize backoff values for a list of transmitters."""
    backoff_values = []
    for transmitter in transmitters:
        if hasattr(transmitter, "backoff_log"):
            backoff_values.extend([log[1] for log in transmitter.backoff_log])  # Extract backoff values

    if not backoff_values:
        print(f"No backoff values recorded for {transmitter_type}.")
        return

    avg_backoff = sum(backoff_values) / len(backoff_values)
    print(f"Average backoff value for {transmitter_type}: {avg_backoff:.2f}")


def moving_average(data, window_size=10):
    """Calculate the moving average of a list."""
    return [
        sum(data[max(i - window_size + 1, 0):i + 1]) / (i - max(i - window_size + 1, 0) + 1)
        for i in range(len(data))
    ]


def jain_fairness_index_between_networks(gnbs, aps, priority_class=None):
    """
    Calculate Jain's Fairness Index between the total successful airtime of gNBs (NR-U)
    and APs (Wi-Fi).
    :param gnbs: List of gNB transmitters.
    :param aps: List of AP transmitters.
    :param priority_class: Strict the transmitters to ones of a priority_class
    :return: Jain's Fairness Index between the two networks (float).
    """
    if priority_class:
        total_succ_airtime_1 = sum(gnb.successful_airtime if gnb.priority_class.id == priority_class else 0 for gnb in gnbs)
        total_succ_airtime_1 += sum(ap.successful_airtime if ap.priority_class.id == priority_class else 0 for ap in aps)
        total_succ_airtime_2 = sum(gnb.successful_airtime if gnb.priority_class.id != priority_class else 0 for gnb in gnbs)
        total_succ_airtime_2 += sum(ap.successful_airtime if ap.priority_class.id != priority_class else 0 for ap in aps)
    else:
        total_succ_airtime_1 = sum(gnb.successful_airtime for gnb in gnbs)
        total_succ_airtime_2 = sum(ap.successful_airtime for ap in aps)

    # Avoid division by zero
    if total_succ_airtime_1 == 0 and total_succ_airtime_2 == 0:
        return 1.0  # Perfect fairness when no airtime is used

    values = [total_succ_airtime_1, total_succ_airtime_2]
    numerator = sum(values) ** 2
    denominator = 2 * sum(v ** 2 for v in values)
    return numerator / denominator


def get_gamma_from_step_time(base_gamma, step_times, max_step_duration):
    ratio = step_times / max_step_duration
    return torch.exp(-1 * base_gamma * ratio)

