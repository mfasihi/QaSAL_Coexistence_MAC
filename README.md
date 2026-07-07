# 📡 5G NR-U and Wi-Fi Coexistence Simulation with Reinforcement Learning

This repository provides a Python-based simulation environment built on **SimPy** that models the **MAC-layer behavior** of 5G NR-U and Wi-Fi networks operating under saturated traffic conditions. The simulation captures the dynamic and competitive nature of the unlicensed spectrum where heterogeneous wireless technologies coexist.

5G NR-U employs scheduled transmissions governed by Listen-Before-Talk (LBT), while Wi-Fi operates using a distributed contention-based protocol, namely CSMA/CA. These fundamental differences can lead to unfair spectrum access and performance degradation when both technologies compete in congested environments. We propose **QaSAL**, a state-augmented reinforcement learning framework for intelligent coexistence parameter management between 5G NR-U and Wi-Fi. The goal is to maintain QoS for high-priority traffic while improving fairness and efficiency in shared spectrum environments.

## 🔬 Implemented Approaches

This codebase includes implementations of the following algorithms for **Coexistence Parameter Management (CPM)**:
* Multi-Objective Reinforcement Learning (MORL)
* Primal-Dual Constrained Reinforcement Learning (CRL)
* QaSAL: QoS-aware State-Augmented Learnable Algorithm

These learning-based approaches aim to tune MAC-layer coexistence parameters to improve fairness and meet Quality-of-Service (QoS) requirements in mixed-technology environments.

## 📄 Citation

If you use this code in your work, please cite our paper:

M. R. Fasihi, B. L. Mark and O. A. Alotaibi, "QoS-Aware State-Augmented Learning for Wireless Coexistence Parameter Management," in IEEE Transactions on Machine Learning in Communications and Networking, vol. 4, pp. 1120-1137, 2026, doi: 10.1109/TMLCN.2026.3707514.

## 🧱 Project Structure

Key classes and components:
* Channel – Models the shared coexistence channel.
* NetworkEnvironment – Simulates the interaction between 5G NR-U and Wi-Fi networks.
* Transmitter – Defines the behavior of transmitters (gNB or AP).
* QNetwork – Neural network model for single-head Q-learning.
* MultiHeadQNetwork – Neural network for multi-head Q-learning (multi-objective).
* DQNAgent – The reinforcement learning agent using Deep Q-Networks.

## 🚀 How to Run

* The entry point is main.py, which configures and runs the simulation.
* Common utilities and helper functions are included in utils.py.
