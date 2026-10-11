# Multi-Agent RL: IPPO vs MAPPO on a Cooperative Cleaning Grid World

UNSW COMP9414 (Artificial Intelligence), Term 2 2026 — Assignment 2 (July–August 2026), individual coursework.

**中文简介**：在 12×12 协作清扫网格世界（3 个智能体、易碎物体、两类污渍）中，从零实现模块化 PPO 流水线，并对比 IPPO（独立 PPO）与 MAPPO（参数共享 Actor + 集中式 Critic）在固定/随机污渍两种场景下的表现。结果以实测为准，成功率总体偏低（见下表）。

## Overview

All code in this repository is my own work: the environment, actor/critic networks, rollout buffer, GAE and PPO update, the IPPO and MAPPO trainers, the notebook generator and the follow-up experiments. Environment rules, reward structure and algorithm outline come from the course specification (not included).

- **Environment:** 12×12 grid, 3 cleaning robots, 14 fragile objects, 8 normal and 8 large dirt cells; fixed-position and random-dirt modes; 50-step episodes. Hitting a fragile object, the boundary or another robot ends the episode with a −20 penalty. Each agent observes a 12×12×5 local tensor (fragile objects, own position, other agents, explored area, own start position).
- **PPO pipeline:** actor/critic MLPs (128–128), rollout buffer, GAE, clipped PPO update.
- **IPPO:** an independent actor and critic per agent. **MAPPO:** one parameter-shared actor (observation + one-hot agent id) and a centralised critic over the concatenated observations.
- **Submitted run:** 400 iterations, actor/critic lr 3e-4, γ = 0.95, GAE λ = 0.95, entropy coefficient 0.01. **Optimised run:** 700 iterations, critic lr 1e-3, γ = 0.97, entropy coefficient 0.02.

## Results

Selected checkpoints, each evaluated over 400 episodes (seed 123); selection rule: lower violation rate first, then higher success rate, then higher average team return (`experiments/checkpoints_best/summary.json`).

| Run | Source | Success rate | Violation rate | Avg. team return |
|---|---|---|---|---|
| IPPO, fixed | 700-iter optimised | 0.25% | 99.75% | 153.2 |
| IPPO, random | 700-iter optimised | 12.25% | 0.0% | 220.7 |
| MAPPO, fixed | 400-iter submitted | 0.5% | 0.0% | 131.8 |
| MAPPO, random | 400-iter submitted | 1.2% | 0.0% | 170.3 |

Absolute success rates are low and are reported as measured. IPPO with random dirt is the only configuration with a meaningful success rate; MAPPO policies avoid violations but rarely finish the task; IPPO-fixed ends in a violation in 99.75% of episodes (its 400-iteration alternative was 100%). Raising the MAPPO critic learning rate from 3e-4 to 1e-3 in the 700-iteration run destabilised the critic (loss oscillating between 40 and 217) and the policy collapsed to unsafe behaviour, so the 400-iteration MAPPO models were kept. Each configuration was trained once (single seed).

## How to reproduce

Python 3.10+, CPU is sufficient:

```bash
pip install torch numpy matplotlib nbformat jupyter
```

1. **Full notebook (Tasks 1–5):** open `Assignment2_Master.ipynb` and *Restart & Run All* — four training runs of 400 iterations (about 2.1 s per iteration, roughly 15 minutes per configuration) followed by 400-episode evaluations. `python build_notebook.py` regenerates the notebook.
2. **Optimised 700-iteration runs:** `cd experiments && python train_optimized.py` (about 1,450–1,500 s per configuration on my machine); `resume_remaining.py` finishes interrupted runs.
3. **Without training:** the `.pt` weights and `.npz` training histories in `checkpoints/`, `experiments/checkpoints_v2/` and `experiments/checkpoints_best/`.

The full pipeline has not been re-run on a clean machine; the tables above are the stored results of my runs.

## Repository layout

| Path | Contents |
|---|---|
| `Assignment2_Master.ipynb` | Full notebook: Tasks 1–5 with code, plots and discussion |
| `build_notebook.py` | Script used to generate the notebook |
| `checkpoints/` | Submitted 400-iteration models and training histories |
| `experiments/` | 700-iteration runs (`checkpoints_v2`), selected models (`checkpoints_best`), training scripts and logs |

The course specification and submission pages are not included. The same course also had a Webots (TurtleBot3) + TD3 tutorial exercise, which is not part of this repository.
