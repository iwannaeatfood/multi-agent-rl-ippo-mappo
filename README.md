# Multi-Agent RL: IPPO vs MAPPO on a Cooperative Cleaning Grid World

UNSW COMP9414 (Artificial Intelligence), Term 2 2026 — Assignment 2 (July–August 2026), individual coursework.

**中文简介**：在 12×12 协作清扫网格世界（3 个智能体、易碎物体、两类污渍）中，从零实现模块化 PPO 流水线，并对比 IPPO（独立 PPO）与 MAPPO（参数共享 Actor + 集中式 Critic）在固定/随机污渍两种场景下的表现。结果以实测为准，成功率总体偏低（见下表与"局限"）。

## My contribution

Everything in this repository is my own work: the grid-world environment, the actor/critic networks, rollout buffer, GAE and PPO update, the IPPO and MAPPO trainers, the notebook generator (`build_notebook.py`), and the follow-up experiments in `experiments/`. The environment rules, reward structure and algorithm outline come from the course specification (not included).

## Task

- Environment: 12×12 grid, 3 cleaning robots, 14 fragile objects, 8 normal and 8 large dirt cells; a fixed-position mode and a random-dirt mode. Colliding with a fragile object, the boundary or another robot ends the episode with a −20 penalty. Episode limit: 50 steps.
- Task 2 – modular PPO pipeline: actor/critic networks, rollout buffer, GAE estimator, PPO optimiser.
- Task 3 – **IPPO**: fully independent actor/critic per agent, no communication.
- Task 4 – **MAPPO**: one parameter-shared actor (local observation + one-hot agent id) and a centralised critic over the concatenated observations.
- Task 5 – evaluation of IPPO vs MAPPO, fixed vs random dirt placement.

## Results

Final selected checkpoints, each evaluated over 400 episodes (seed 123; `experiments/checkpoints_best/summary.json`). Selection rule: lower violation rate first, then higher success rate, then higher average team return.

| Run | Source | Success rate | Violation rate | Avg. team return |
|---|---|---|---|---|
| IPPO, fixed | 700-iter optimised | 0.25% | 99.75% | 153.2 |
| IPPO, random | 700-iter optimised | 12.25% | 0.0% | 220.7 |
| MAPPO, fixed | 400-iter submitted | 0.5% | 0.0% | 131.8 |
| MAPPO, random | 400-iter submitted | 1.2% | 0.0% | 170.3 |

Absolute success rates are low; the table is reported as measured, not tuned for presentation.

- IPPO with random dirt was the only configuration that completed the task with a meaningful success rate (12.25%).
- MAPPO policies avoided violations in both scenarios but rarely completed the task.
- IPPO-fixed is a failure case: 99.75% of its episodes end in a violation (the 400-iteration alternative was 100%), so it was kept only as the better of two unsafe options.
- For MAPPO, raising the critic learning rate (3e-4 → 1e-3) in the 700-iteration run destabilised the critic (loss oscillating between 40 and 217) and the policy collapsed to unsafe behaviour, so the 400-iteration models were kept.

## How to reproduce

Needs Python 3.10+ and a CPU is enough (no GPU code paths).

```bash
pip install torch numpy matplotlib nbformat jupyter
```

1. **Full notebook (Tasks 1–5):** open `Assignment2_Master.ipynb` and *Restart & Run All*. It trains the four runs (IPPO/MAPPO × fixed/random) with `N_ITER = 400` — about 2.1 s per iteration on CPU, roughly 15 minutes per configuration — then evaluates each for 400 episodes. `build_notebook.py` regenerates the notebook (`python build_notebook.py`, needs `nbformat`).
2. **Optimised 700-iteration runs:** `cd experiments && python train_optimized.py` writes to `checkpoints_v2/` (about 1,450–1,500 s per configuration on my machine, see `train_seconds` in `summary.json`). `resume_remaining.py` finishes configurations that were interrupted; `train_log.txt` / `resume_log*.txt` are the original logs.
3. **Inspect without training:** `checkpoints/` (400-iteration, submitted), `experiments/checkpoints_v2/` (700-iteration) and `experiments/checkpoints_best/` (the selected models plus `summary.json`) hold `.pt` weights and `.npz` training histories.

Seeds are fixed in the code, but exact numbers can still differ across PyTorch versions and hardware; I have not re-run the whole pipeline on a clean machine for this README, so treat the tables above as the stored results of my runs.

## Design notes and likely interview questions

**What is the observation, and how do agents find dirt?** Following the assignment specification, each agent gets a 12×12×5 local tensor: fragile objects, own position, other agents, explored area, and own start (dock) position. Dirt locations are *not* in the observation, so agents have to discover dirt by exploring; the reward gives +2 for a new cell, −1 for revisiting, +2 / +4 for normal / large dirt and +20 team bonus when all dirt is cleaned. This is the main reason random-dirt is hard.

**IPPO vs MAPPO — what actually differs in code?** IPPO builds three independent actor/critic pairs on the local observation. MAPPO uses one actor (input: observation + one-hot agent id, so parameters are shared) and one critic over the concatenation of all three observations (`OBS_DIM × 3`). Execution is decentralised in both; only the critic's input differs.

**Hyperparameters.** Both nets are 128–128 MLPs; PPO clip ε = 0.2, value coefficient 0.5, entropy coefficient 0.01 (0.02 in the optimised run), GAE λ = 0.95, γ = 0.95 (0.97 optimised), rollout length 300, minibatch 150, 8 PPO epochs, Adam with 1e-5 weight decay, actor lr 3e-4, critic lr 3e-4 (1e-3 in the optimised run).

**Why are the results weak, and what would I do next?** Plausible causes (not isolated by ablation): the dirt is not observed, the reward is sparse relative to the −20 episode-ending collision penalty, and training is short. The one controlled finding is the critic-lr instability above. Next steps I would try: add a dirt-remaining channel or count to the critic input (allowed for the centralised critic), reward normalisation, longer training with a smaller critic lr, and several seeds so differences between configurations can be compared with confidence intervals — here each configuration is a single training run.

**Why did you pick those checkpoints?** Safety first: the task penalises violations, so I rank by violation rate, then success, then return. Higher-return checkpoints were rejected when they were unsafe (see `rejected_alternative` in `summary.json`).

**Limitations.** Single seed per configuration; no hyperparameter search beyond the one tuned run; evaluation uses the training environment's fixed/random dirt generators, not held-out layouts.

## Repository layout

| Path | Contents |
|---|---|
| `Assignment2_Master.ipynb` | Full notebook: Tasks 1–5 with code, plots and discussion |
| `build_notebook.py` | Script used to generate the notebook |
| `checkpoints/` | Submitted 400-iteration models and training histories |
| `experiments/` | 700-iteration optimised runs (`checkpoints_v2`), selected models (`checkpoints_best`), training scripts and logs |

The course specification and submission pages are not included. The same course also had a Webots (TurtleBot3) + TD3 continuous-control tutorial exercise; it is not part of this repository.
