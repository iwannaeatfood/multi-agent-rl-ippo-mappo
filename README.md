# Multi-Agent RL: IPPO vs MAPPO on a Cooperative Cleaning Grid World

UNSW COMP9414 (Artificial Intelligence), Term 2 2026 — Assignment 2.

**中文简介**：在 12×12 协作清扫网格世界（3 个智能体、易碎物体、两类污渍）中，从零实现模块化 PPO 流水线，并对比 IPPO（独立 PPO）与 MAPPO（参数共享 Actor + 集中式 Critic）在固定/随机污渍两种场景下的表现。

## Task

- Environment: 12×12 grid, 3 cleaning robots, fragile objects, normal/large dirt; a fixed-position mode and a random-dirt mode. Colliding with a fragile object, the boundary or another robot ends the episode with a −20 penalty.
- Task 2 – modular PPO pipeline: actor/critic networks, rollout buffer, GAE estimator, PPO optimiser.
- Task 3 – **IPPO**: fully independent actor/critic per agent, no communication.
- Task 4 – **MAPPO**: one parameter-shared actor (local observation + one-hot agent id) and a centralised critic.
- Task 5 – evaluation of IPPO vs MAPPO, fixed vs random dirt placement.

## Results

Final selected checkpoints, evaluated over repeated episodes (`experiments/checkpoints_best/summary.json`). Selection rule: lower violation rate first, then higher success rate, then higher average team return.

| Run | Source | Success rate | Violation rate | Avg. team return |
|---|---|---|---|---|
| IPPO, fixed | 700-iter optimised | 0.25% | 99.75% | 153.2 |
| IPPO, random | 700-iter optimised | 12.25% | 0.0% | 220.7 |
| MAPPO, fixed | 400-iter submitted | 0.5% | 0.0% | 131.8 |
| MAPPO, random | 400-iter submitted | 1.2% | 0.0% | 170.3 |

Absolute success rates are low; the table is reported as measured. Notes from the experiments:

- IPPO with random dirt was the only configuration that completed the task with a meaningful success rate (12.25%).
- MAPPO policies avoided violations in both scenarios but rarely completed the task.
- For MAPPO-fixed, raising the critic learning rate (3e-4 → 1e-3) in the 700-iteration run destabilised the critic (loss oscillating between 40 and 217) and the policy collapsed to unsafe behaviour, so the 400-iteration models were kept.

## Repository layout

| Path | Contents |
|---|---|
| `Assignment2_Master.ipynb` | Full notebook: Tasks 1–5 with code, plots and discussion |
| `build_notebook.py` | Script used to generate the notebook |
| `checkpoints/` | Submitted 400-iteration models and training histories |
| `experiments/` | 700-iteration optimised runs (`checkpoints_v2`), selected models (`checkpoints_best`), training scripts and logs |
| `logs/` | Task 2 rollout logs |

Course specification and submission pages are not included.

## Related coursework

The same course also had a Webots (TurtleBot3) + TD3 continuous-control tutorial exercise; it is not part of this repository.
