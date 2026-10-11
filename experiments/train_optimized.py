"""
Exploratory optimization pass, kept separate from the submitted notebook.
Same environment/algorithm, two changes:
1. Faster rollout collection: MAPPO batches all 3 agents through the shared
   actor/critic in one forward call instead of three, and the wasted
   entropy computation during rollout is dropped for both algorithms.
2. Tuned hyperparameters within Table 3's ranges (higher critic learning
   rate, higher discount factor, higher entropy bonus) and more training
   iterations, at the same per-iteration compute cost as the submitted run.

Run: python train_optimized.py
"""
import copy
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

CKPT_DIR = "checkpoints_v2"


# --------------------------------------------------------------------- #
# Environment (identical to the submitted notebook)
# --------------------------------------------------------------------- #
def to_zero_indexed(coords):
    return np.array([(r - 1, c - 1) for r, c in coords], dtype=int)


class CleaningGridWorld:
    N_AGENTS = 3
    GRID = 12
    MAX_STEPS = 50

    ACTIONS = {0: "IDLE", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    ACTION_DELTA = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    START_POS_RAW = [(1, 3), (12, 3), (7, 12)]
    FRAGILE_RAW = [(2, 2), (4, 4), (6, 6), (7, 2), (8, 4), (10, 6),
                   (1, 7), (4, 7), (8, 7), (12, 7),
                   (3, 11), (8, 9), (6, 10), (11, 11)]
    LARGE_DIRT_RAW = [(2, 5), (3, 9), (4, 2), (6, 7), (8, 8), (9, 4), (10, 11), (12, 2)]
    NORMAL_DIRT_RAW = [(1, 4), (1, 10), (3, 2), (4, 8), (5, 9), (7, 3), (9, 11), (12, 6)]

    R_NORMAL_DIRT = 2.0
    R_LARGE_DIRT = 4.0
    R_NEW_CELL = 2.0
    R_REVISIT = -1.0
    R_IDLE = -0.1
    R_TIMESTEP = -0.1
    R_COLLISION = -20.0
    R_COMPLETE = 20.0

    def __init__(self, random_dirt=False, seed=None):
        self.random_dirt = random_dirt
        self.rng = np.random.default_rng(seed)
        self._fragile = to_zero_indexed(self.FRAGILE_RAW)
        self._start = to_zero_indexed(self.START_POS_RAW)
        self._fragile_mask = np.zeros((self.GRID, self.GRID), dtype=bool)
        self._fragile_mask[self._fragile[:, 0], self._fragile[:, 1]] = True
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.random_dirt:
            self._large_dirt, self._normal_dirt = self._sample_dirt_positions()
        else:
            self._large_dirt = to_zero_indexed(self.LARGE_DIRT_RAW)
            self._normal_dirt = to_zero_indexed(self.NORMAL_DIRT_RAW)
        self.large_active = np.ones(len(self._large_dirt), dtype=bool)
        self.normal_active = np.ones(len(self._normal_dirt), dtype=bool)
        self.agent_pos = self._start.copy()
        self.explored = np.zeros((self.GRID, self.GRID), dtype=bool)
        for r, c in self.agent_pos:
            self.explored[r, c] = True
        self.t = 0
        obs = self._get_obs()
        info = {"large_dirt_active": self.large_active.copy(), "normal_dirt_active": self.normal_active.copy()}
        return obs, info

    def step(self, actions):
        actions = np.asarray(actions, dtype=int)
        self.t += 1
        rewards = np.full(self.N_AGENTS, self.R_TIMESTEP)
        proposed = self.agent_pos.copy()
        collided = np.zeros(self.N_AGENTS, dtype=bool)

        for i, a in enumerate(actions):
            if a == 0:
                rewards[i] += self.R_IDLE
                continue
            dr, dc = self.ACTION_DELTA[a]
            nr, nc = self.agent_pos[i, 0] + dr, self.agent_pos[i, 1] + dc
            if not (0 <= nr < self.GRID and 0 <= nc < self.GRID):
                rewards[i] += self.R_COLLISION
                collided[i] = True
                continue
            if self._fragile_mask[nr, nc]:
                rewards[i] += self.R_COLLISION
                collided[i] = True
                continue
            proposed[i] = (nr, nc)

        for i in range(self.N_AGENTS):
            for j in range(i + 1, self.N_AGENTS):
                same_target = tuple(proposed[i]) == tuple(proposed[j])
                swap = (tuple(proposed[i]) == tuple(self.agent_pos[j])
                        and tuple(proposed[j]) == tuple(self.agent_pos[i]))
                if same_target or swap:
                    for k in (i, j):
                        if not collided[k]:
                            rewards[k] += self.R_COLLISION
                            collided[k] = True
                    proposed[i], proposed[j] = self.agent_pos[i], self.agent_pos[j]

        self.agent_pos = proposed

        for i, (r, c) in enumerate(self.agent_pos):
            if collided[i]:
                continue
            if not self.explored[r, c]:
                self.explored[r, c] = True
                rewards[i] += self.R_NEW_CELL
            else:
                rewards[i] += self.R_REVISIT
            large_hit = np.where((self._large_dirt[:, 0] == r) & (self._large_dirt[:, 1] == c) & self.large_active)[0]
            if len(large_hit):
                self.large_active[large_hit[0]] = False
                rewards[i] += self.R_LARGE_DIRT
            normal_hit = np.where((self._normal_dirt[:, 0] == r) & (self._normal_dirt[:, 1] == c) & self.normal_active)[0]
            if len(normal_hit):
                self.normal_active[normal_hit[0]] = False
                rewards[i] += self.R_NORMAL_DIRT

        all_clean = (not self.large_active.any()) and (not self.normal_active.any())
        terminated = bool(all_clean)
        truncated = bool(collided.any()) or (self.t >= self.MAX_STEPS and not terminated)
        if terminated:
            rewards += self.R_COMPLETE

        obs = self._get_obs()
        info = {"collided": collided.copy(), "large_dirt_active": self.large_active.copy(),
                "normal_dirt_active": self.normal_active.copy(), "t": self.t}
        return obs, rewards, terminated, truncated, info

    def _get_obs(self):
        explored_map = self.explored.astype(np.float32)
        obs = []
        for i in range(self.N_AGENTS):
            self_map = np.zeros((self.GRID, self.GRID), dtype=np.float32)
            self_map[self.agent_pos[i, 0], self.agent_pos[i, 1]] = 1.0
            other_map = np.zeros((self.GRID, self.GRID), dtype=np.float32)
            for j in range(self.N_AGENTS):
                if j != i:
                    other_map[self.agent_pos[j, 0], self.agent_pos[j, 1]] = 1.0
            dock_map = np.zeros((self.GRID, self.GRID), dtype=np.float32)
            dock_map[self._start[i, 0], self._start[i, 1]] = 1.0
            o_i = np.stack([self._fragile_mask.astype(np.float32), self_map, other_map, explored_map, dock_map], axis=-1)
            obs.append(o_i)
        return obs

    def get_global_state(self):
        return np.concatenate(self._get_obs(), axis=-1)

    def _sample_dirt_positions(self):
        occupied = set(map(tuple, self._fragile)) | set(map(tuple, self._start))
        free_cells = [(r, c) for r in range(self.GRID) for c in range(self.GRID) if (r, c) not in occupied]
        n_large, n_normal = len(self.LARGE_DIRT_RAW), len(self.NORMAL_DIRT_RAW)
        idx = self.rng.choice(len(free_cells), size=n_large + n_normal, replace=False)
        chosen = [free_cells[k] for k in idx]
        return np.array(chosen[:n_large]), np.array(chosen[n_large:])


# --------------------------------------------------------------------- #
# Networks (identical)
# --------------------------------------------------------------------- #
OBS_DIM = 12 * 12 * 5
N_ACTIONS = 5


def build_mlp(in_dim, out_dim, hidden_sizes):
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_sizes=(128, 128)):
        super().__init__()
        self.body = build_mlp(obs_dim, n_actions, hidden_sizes)

    def forward(self, obs):
        return Categorical(logits=self.body(obs))

    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, action):
        dist = self.forward(obs)
        return dist.log_prob(action), dist.entropy()


class CriticNet(nn.Module):
    def __init__(self, in_dim=OBS_DIM, hidden_sizes=(128, 128)):
        super().__init__()
        self.body = build_mlp(in_dim, 1, hidden_sizes)

    def forward(self, x):
        return self.body(x).squeeze(-1)


def build_ippo_networks(n_agents=3, hidden_sizes=(128, 128)):
    actors = [ActorNet(OBS_DIM, N_ACTIONS, hidden_sizes) for _ in range(n_agents)]
    critics = [CriticNet(OBS_DIM, hidden_sizes) for _ in range(n_agents)]
    return actors, critics


def build_mappo_networks(n_agents=3, hidden_sizes=(128, 128)):
    actor = ActorNet(OBS_DIM + n_agents, N_ACTIONS, hidden_sizes)
    critic = CriticNet(OBS_DIM * n_agents, hidden_sizes)
    return actor, critic


# --------------------------------------------------------------------- #
# Rollout buffer (identical)
# --------------------------------------------------------------------- #
class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.value_inputs, self.actions = [], [], []
        self.log_probs, self.rewards, self.values, self.dones = [], [], [], []
        self.advantages, self.returns = None, None

    def add(self, obs, value_input, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.value_inputs.append(value_input)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma=0.95, lam=0.95):
        T = len(self.rewards)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:T]
        self.advantages, self.returns = advantages, returns
        return advantages, returns

    def get_arrays(self):
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "value_inputs": np.asarray(self.value_inputs, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "advantages": self.advantages,
            "returns": self.returns,
        }


# --------------------------------------------------------------------- #
# Optimized rollout collection: one no_grad context per step, and a single
# batched forward pass for MAPPO's shared actor/critic instead of 3 calls.
# --------------------------------------------------------------------- #
def collect_rollout_fast(env, actors, critics, n_steps, use_global_state=False):
    n_agents = env.N_AGENTS
    buffers = [RolloutBuffer() for _ in range(n_agents)]
    episode_records = []
    ep_return = np.zeros(n_agents)
    ep_len = 0
    shared = use_global_state and all(a is actors[0] for a in actors)
    eye = torch.eye(n_agents) if shared else None

    obs_list, _ = env.reset()
    for _ in range(n_steps):
        flat_obs = [o.reshape(-1) for o in obs_list]
        global_state = env.get_global_state().reshape(-1) if use_global_state else None

        with torch.no_grad():
            if shared:
                obs_t = torch.tensor(np.stack(flat_obs), dtype=torch.float32)
                actor_in = torch.cat([obs_t, eye], dim=-1)
                dist = actors[0].forward(actor_in)
                a_t = dist.sample()
                logp_t = dist.log_prob(a_t)
                v_in = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).expand(n_agents, -1)
                v_t = critics[0](v_in)
                actions = a_t.tolist()
                log_probs = logp_t.tolist()
                values = v_t.tolist()
            else:
                actions, log_probs, values = [], [], []
                for i in range(n_agents):
                    o_t = torch.tensor(flat_obs[i], dtype=torch.float32).unsqueeze(0)
                    dist = actors[i].forward(o_t)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                    v = critics[i](o_t)
                    actions.append(a.item())
                    log_probs.append(logp.item())
                    values.append(v.item())

        next_obs_list, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        ep_return += np.asarray(rewards)
        ep_len += 1

        for i in range(n_agents):
            value_input = global_state if use_global_state else flat_obs[i]
            buffers[i].add(flat_obs[i], value_input, actions[i], log_probs[i], float(rewards[i]), values[i], float(done))

        obs_list = next_obs_list
        if done:
            episode_records.append({"return": ep_return.copy(), "length": ep_len,
                                     "success": bool(terminated), "violation": bool(info["collided"].any())})
            ep_return[:] = 0.0
            ep_len = 0
            obs_list, _ = env.reset()

    return buffers, episode_records


# --------------------------------------------------------------------- #
# PPO core (identical)
# --------------------------------------------------------------------- #
def retry_on_memory_error(fn, retries=8, delay=5.0):
    """This machine runs other memory-hungry apps alongside training, so a
    small numpy allocation can transiently fail under system-wide memory
    pressure even though the array itself is a few MB. Wait for the spike
    to pass and retry rather than aborting the whole run."""
    for attempt in range(retries):
        try:
            return fn()
        except MemoryError:
            gc.collect()
            print(f"  MemoryError, retrying in {delay}s (attempt {attempt+1}/{retries})", flush=True)
            time.sleep(delay)
    return fn()


def probability_ratio(new_log_prob, old_log_prob):
    return torch.exp(new_log_prob - old_log_prob)


def clipped_actor_loss(ratio, advantages, epsilon=0.2):
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()


def critic_loss_fn(values_new, returns):
    return ((values_new - returns) ** 2).mean()


def entropy_bonus_fn(entropy):
    return entropy.mean()


def total_loss_fn(l_clip, l_value, s_entropy, c1=0.5, c2=0.01):
    return l_clip + c1 * l_value - c2 * s_entropy


def ppo_update(actor, critic, actor_opt, critic_opt, batch, epsilon=0.2, c1=0.5, c2=0.01):
    obs, value_in = batch["obs"], batch["value_in"]
    actions, old_log_probs = batch["actions"], batch["old_log_probs"]
    advantages, returns = batch["advantages"], batch["returns"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    new_log_probs, entropy = actor.evaluate(obs, actions)
    values_new = critic(value_in)

    ratio = probability_ratio(new_log_probs, old_log_probs)
    l_clip = clipped_actor_loss(ratio, advantages, epsilon)
    l_value = critic_loss_fn(values_new, returns)
    s_entropy = entropy_bonus_fn(entropy)
    loss = total_loss_fn(l_clip, l_value, s_entropy, c1, c2)

    actor_opt.zero_grad()
    critic_opt.zero_grad()
    loss.backward()
    actor_opt.step()
    critic_opt.step()

    return {"actor_loss": l_clip.item(), "critic_loss": l_value.item(),
            "entropy": s_entropy.item(), "total_loss": loss.item()}


# --------------------------------------------------------------------- #
# Trainers (same structure as the submitted notebook, using collect_rollout_fast)
# --------------------------------------------------------------------- #
class IPPOTrainer:
    def __init__(self, random_dirt=False, hidden_sizes=(128, 128), lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.97, lam=0.95, epsilon=0.2, c1=0.5, c2=0.02, rollout_len=300,
                 minibatch_size=150, k_epochs=8, weight_decay=1e-5, seed=0):
        self.env = CleaningGridWorld(random_dirt=random_dirt, seed=seed)
        self.n_agents = self.env.N_AGENTS
        self.gamma, self.lam = gamma, lam
        self.epsilon, self.c1, self.c2 = epsilon, c1, c2
        self.rollout_len, self.minibatch_size, self.k_epochs = rollout_len, minibatch_size, k_epochs

        torch.manual_seed(seed)
        self.actors, self.critics = build_ippo_networks(self.n_agents, hidden_sizes)
        self.actor_opts = [optim.Adam(a.parameters(), lr=lr_actor, weight_decay=weight_decay) for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=lr_critic, weight_decay=weight_decay) for c in self.critics]

        self.history = {i: {"total_loss": [], "actor_loss": [], "critic_loss": [], "entropy": []} for i in range(self.n_agents)}
        self.episode_metrics = {"return": [], "length": [], "success": [], "violation": []}
        self.best_return, self.best_state = -1e18, None

    def _state_dict(self):
        d = {f"actor_{i}": self.actors[i].state_dict() for i in range(self.n_agents)}
        d.update({f"critic_{i}": self.critics[i].state_dict() for i in range(self.n_agents)})
        return d

    def _update_agent(self, i, arrs):
        obs = torch.tensor(arrs["obs"])
        value_in = torch.tensor(arrs["value_inputs"])
        actions = torch.tensor(arrs["actions"])
        old_log_probs = torch.tensor(arrs["log_probs"])
        advantages = torch.tensor(arrs["advantages"])
        returns = torch.tensor(arrs["returns"])
        n = len(actions)
        epoch_losses = {"actor_loss": [], "critic_loss": [], "entropy": [], "total_loss": []}
        for _ in range(self.k_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                if len(idx) < 2:
                    continue
                batch = {"obs": obs[idx], "value_in": value_in[idx], "actions": actions[idx],
                         "old_log_probs": old_log_probs[idx], "advantages": advantages[idx], "returns": returns[idx]}
                losses = ppo_update(self.actors[i], self.critics[i], self.actor_opts[i], self.critic_opts[i],
                                     batch, self.epsilon, self.c1, self.c2)
                for k in epoch_losses:
                    epoch_losses[k].append(losses[k])
        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    def train(self, n_iterations, verbose=True, log_every=50):
        for it in range(n_iterations):
            buffers, ep_records = retry_on_memory_error(
                lambda: collect_rollout_fast(self.env, self.actors, self.critics, self.rollout_len, use_global_state=False))
            for buf in buffers:
                retry_on_memory_error(lambda buf=buf: buf.compute_gae(last_value=0.0, gamma=self.gamma, lam=self.lam))
            for i in range(self.n_agents):
                losses = self._update_agent(i, retry_on_memory_error(lambda i=i: buffers[i].get_arrays()))
                for k, v in losses.items():
                    self.history[i][k].append(v)
            for rec in ep_records:
                for k in self.episode_metrics:
                    self.episode_metrics[k].append(rec[k])
            if ep_records:
                team_return = np.mean([rec["return"].sum() for rec in ep_records])
                if team_return > self.best_return:
                    self.best_return = team_return
                    self.best_state = copy.deepcopy(self._state_dict())
            if it % 100 == 0:
                gc.collect()
            if verbose and (it % log_every == 0 or it == n_iterations - 1):
                avg_total = np.mean([self.history[i]["total_loss"][-1] for i in range(self.n_agents)])
                print(f"[IPPO] iter {it+1}/{n_iterations} episodes_in_rollout={len(ep_records)} avg_total_loss={avg_total:.3f}", flush=True)

    def save_checkpoint(self, path_prefix):
        torch.save(self._state_dict(), path_prefix + ".pt")
        torch.save(self.best_state or self._state_dict(), path_prefix + "_best.pt")
        np.savez(path_prefix + "_history.npz",
                 **{f"agent{i}_{k}": np.array(v) for i in range(self.n_agents) for k, v in self.history[i].items()})


def load_ippo(path_prefix, n_agents=3, hidden_sizes=(128, 128)):
    actors, critics = build_ippo_networks(n_agents, hidden_sizes)
    state = torch.load(path_prefix + ".pt", map_location="cpu")
    for i in range(n_agents):
        actors[i].load_state_dict(state[f"actor_{i}"])
        critics[i].load_state_dict(state[f"critic_{i}"])
    return actors, critics


class MAPPOTrainer:
    def __init__(self, random_dirt=False, hidden_sizes=(128, 128), lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.97, lam=0.95, epsilon=0.2, c1=0.5, c2=0.02, rollout_len=300,
                 minibatch_size=150, k_epochs=8, weight_decay=1e-5, seed=0):
        self.env = CleaningGridWorld(random_dirt=random_dirt, seed=seed)
        self.n_agents = self.env.N_AGENTS
        self.gamma, self.lam = gamma, lam
        self.epsilon, self.c1, self.c2 = epsilon, c1, c2
        self.rollout_len, self.minibatch_size, self.k_epochs = rollout_len, minibatch_size, k_epochs

        torch.manual_seed(seed)
        self.actor, self.critic = build_mappo_networks(self.n_agents, hidden_sizes)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        self.history = {i: {"total_loss": [], "actor_loss": [], "critic_loss": [], "entropy": []} for i in range(self.n_agents)}
        self.episode_metrics = {"return": [], "length": [], "success": [], "violation": []}
        self.best_return, self.best_state = -1e18, None

    def train(self, n_iterations, verbose=True, log_every=50):
        actors_list = [self.actor] * self.n_agents
        critics_list = [self.critic] * self.n_agents
        for it in range(n_iterations):
            buffers, ep_records = retry_on_memory_error(
                lambda: collect_rollout_fast(self.env, actors_list, critics_list, self.rollout_len, use_global_state=True))
            for buf in buffers:
                retry_on_memory_error(lambda buf=buf: buf.compute_gae(last_value=0.0, gamma=self.gamma, lam=self.lam))

            all_obs, all_value_in, all_actions, all_logp, all_adv, all_ret, all_agent_idx = [], [], [], [], [], [], []
            for i, buf in enumerate(buffers):
                arrs = retry_on_memory_error(lambda buf=buf: buf.get_arrays())
                n = len(arrs["actions"])
                all_obs.append(arrs["obs"])
                all_value_in.append(arrs["value_inputs"])
                all_actions.append(arrs["actions"])
                all_logp.append(arrs["log_probs"])
                all_adv.append(arrs["advantages"])
                all_ret.append(arrs["returns"])
                all_agent_idx.append(np.full(n, i))

            def _concat_tensors():
                return (torch.tensor(np.concatenate(all_obs), dtype=torch.float32),
                        torch.tensor(np.concatenate(all_value_in), dtype=torch.float32),
                        torch.tensor(np.concatenate(all_actions), dtype=torch.int64),
                        torch.tensor(np.concatenate(all_logp), dtype=torch.float32),
                        torch.tensor(np.concatenate(all_adv), dtype=torch.float32),
                        torch.tensor(np.concatenate(all_ret), dtype=torch.float32),
                        np.concatenate(all_agent_idx))
            obs, value_in, actions, old_log_probs, advantages, returns, agent_idx = retry_on_memory_error(_concat_tensors)
            actor_in = torch.cat([obs, torch.eye(self.n_agents)[agent_idx]], dim=-1)

            per_agent_losses = {i: {"actor_loss": [], "critic_loss": [], "entropy": [], "total_loss": []} for i in range(self.n_agents)}
            n_total = len(actions)
            for _ in range(self.k_epochs):
                perm = np.random.permutation(n_total)
                for start in range(0, n_total, self.minibatch_size):
                    idx = perm[start:start + self.minibatch_size]
                    if len(idx) < 2:
                        continue
                    batch = {"obs": actor_in[idx], "value_in": value_in[idx], "actions": actions[idx],
                             "old_log_probs": old_log_probs[idx], "advantages": advantages[idx], "returns": returns[idx]}
                    losses = ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt,
                                         batch, self.epsilon, self.c1, self.c2)
                    for a_id in np.unique(agent_idx[idx]):
                        for k in losses:
                            per_agent_losses[a_id][k].append(losses[k])

            for i in range(self.n_agents):
                for k, v in per_agent_losses[i].items():
                    self.history[i][k].append(float(np.mean(v)) if v else np.nan)
            for rec in ep_records:
                for k in self.episode_metrics:
                    self.episode_metrics[k].append(rec[k])
            if ep_records:
                team_return = np.mean([rec["return"].sum() for rec in ep_records])
                if team_return > self.best_return:
                    self.best_return = team_return
                    self.best_state = copy.deepcopy({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()})
            if it % 100 == 0:
                gc.collect()
            if verbose and (it % log_every == 0 or it == n_iterations - 1):
                avg_total = np.nanmean([self.history[i]["total_loss"][-1] for i in range(self.n_agents)])
                print(f"[MAPPO] iter {it+1}/{n_iterations} episodes_in_rollout={len(ep_records)} avg_total_loss={avg_total:.3f}", flush=True)

    def save_checkpoint(self, path_prefix):
        current = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(current, path_prefix + ".pt")
        torch.save(self.best_state or current, path_prefix + "_best.pt")
        np.savez(path_prefix + "_history.npz",
                 **{f"agent{i}_{k}": np.array(v) for i in range(self.n_agents) for k, v in self.history[i].items()})


def load_mappo(path_prefix, n_agents=3, hidden_sizes=(128, 128)):
    actor, critic = build_mappo_networks(n_agents, hidden_sizes)
    state = torch.load(path_prefix + ".pt", map_location="cpu")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    return actor, critic


# --------------------------------------------------------------------- #
# Evaluation (identical logic to the submitted notebook)
# --------------------------------------------------------------------- #
def make_ippo_policy(actors):
    def act_fn(i, obs_flat):
        with torch.no_grad():
            o = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
            probs = actors[i].forward(o).probs.squeeze(0).numpy()
        return int(np.argmax(probs)), probs
    return act_fn


def make_mappo_policy(actor, n_agents):
    eye = torch.eye(n_agents)

    def act_fn(i, obs_flat):
        with torch.no_grad():
            o = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
            actor_in = torch.cat([o, eye[i].unsqueeze(0)], dim=-1)
            probs = actor.forward(actor_in).probs.squeeze(0).numpy()
        return int(np.argmax(probs)), probs
    return act_fn


def rollout_episode(env, act_fn, max_steps=50):
    obs_list, _ = env.reset()
    ep_return = np.zeros(env.N_AGENTS)
    violation, terminated, length = False, False, 0
    for t in range(max_steps):
        flat_obs = [o.reshape(-1) for o in obs_list]
        actions = [act_fn(i, flat_obs[i])[0] for i in range(env.N_AGENTS)]
        obs_list, rewards, terminated, truncated, info = env.step(actions)
        ep_return += rewards
        length += 1
        if info["collided"].any():
            violation = True
        if terminated or truncated:
            break
    return {"return": ep_return, "length": length, "success": bool(terminated), "violation": violation}


def evaluate_policy(env, act_fn, n_episodes=400, seed=123):
    returns, lengths, successes, violations = [], [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        rec = rollout_episode(env, act_fn, max_steps=env.MAX_STEPS)
        returns.append(rec["return"])
        lengths.append(rec["length"] if rec["success"] else np.nan)
        successes.append(rec["success"])
        violations.append(rec["violation"])
    return {"return": np.array(returns), "length": np.array(lengths),
            "success": np.array(successes), "violation": np.array(violations)}


HPARAMS = dict(hidden_sizes=(128, 128), lr_actor=3e-4, lr_critic=1e-3, gamma=0.97, lam=0.95,
               epsilon=0.2, c1=0.5, c2=0.02, rollout_len=300, minibatch_size=150,
               k_epochs=8, weight_decay=1e-5)
CONFIGS = {"ippo_fixed": (IPPOTrainer, False), "ippo_random": (IPPOTrainer, True),
           "mappo_fixed": (MAPPOTrainer, False), "mappo_random": (MAPPOTrainer, True)}


def run_one_config(name, n_iter):
    """Trains a single config in this process, saves its checkpoint and its own
    summary json. Meant to be invoked as an isolated subprocess so memory is
    fully released by the OS after each config, regardless of accumulation
    inside a long-lived process."""
    import os
    os.makedirs(CKPT_DIR, exist_ok=True)
    TrainerCls, random_dirt = CONFIGS[name]

    print(f"\n=== training {name} for {n_iter} iterations ===", flush=True)
    t0 = time.time()
    trainer = TrainerCls(random_dirt=random_dirt, seed=0, **HPARAMS)
    trainer.train(n_iterations=n_iter, verbose=True, log_every=max(1, n_iter // 20))
    trainer.save_checkpoint(f"{CKPT_DIR}/{name}")
    elapsed = time.time() - t0
    print(f"{name} training took {elapsed:.0f}s", flush=True)

    eval_env = CleaningGridWorld(random_dirt=True, seed=2026)
    if TrainerCls is IPPOTrainer:
        act_fn = make_ippo_policy(trainer.actors)
    else:
        act_fn = make_mappo_policy(trainer.actor, eval_env.N_AGENTS)
    res = evaluate_policy(eval_env, act_fn, n_episodes=400, seed=2026)
    result = {
        "train_seconds": elapsed,
        "success_rate": float(res["success"].mean()),
        "violation_rate": float(res["violation"].mean()),
        "avg_team_return": float(res["return"].sum(axis=1).mean()),
        "avg_ep_length": float(np.nanmean(res["length"])) if not np.all(np.isnan(res["length"])) else None,
    }
    print(f"{name} eval: {result}", flush=True)
    with open(f"{CKPT_DIR}/{name}_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_driver(n_iter):
    """Launches each config as its own subprocess sequentially, then merges
    the per-config summaries into one file."""
    import os
    import subprocess
    import sys

    os.makedirs(CKPT_DIR, exist_ok=True)
    summary = {}
    for name in CONFIGS:
        print(f"\n--- launching subprocess for {name} ---", flush=True)
        subprocess.run([sys.executable, __file__, "--config", name, "--n_iter", str(n_iter)], check=True)
        with open(f"{CKPT_DIR}/{name}_summary.json") as f:
            summary[name] = json.load(f)

    with open(f"{CKPT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default=None,
                         help="Train only this one config (used internally by --driver subprocesses).")
    parser.add_argument("--n_iter", type=int, default=int(os.environ.get("N_ITER", "700")))
    args = parser.parse_args()

    if args.config is None:
        run_driver(args.n_iter)
    else:
        run_one_config(args.config, args.n_iter)
