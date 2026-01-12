#!/usr/bin/env python3
import math
import random
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

# ----------------------------
# Model
# ----------------------------
class PVNet(nn.Module):
    """
    Simple CNN trunk with policy + value heads.
    Input: observation tensor shaped (C,H,W) = (38,8,8) for your crazyhouse.
    Policy head: logits for all actions in the game's action space.
    Value head: scalar in [-1,1] via tanh.
    """
    def __init__(self, in_channels: int, num_actions: int, channels: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Flatten
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
        )
        self.value_head = nn.Conv2d(channels, 1, kernel_size=1)
        #self.value_head = nn.Sequential(
        #    nn.Conv2d(channels, 1, kernel_size=1),
        #    nn.ReLU(),
        #)

        # 8x8 board
        self.policy_fc = nn.Linear(2 * 8 * 8, num_actions)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,C,H,W)
        # self.debug_value(x)
        h = self.trunk(x)

        # Policy
        p = self.policy_head(h).flatten(1)  # (B, 2*8*8)
        logits = self.policy_fc(p)          # (B, num_actions)

        # Value
        v = self.value_head(h).flatten(1)   # (B, 1*8*8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return logits, v

    def debug_value(self, x):
        with torch.no_grad():
            h = self.trunk(x)

            print(
                "TRUNK h mean/std/min/max:",
                h.mean().item(),
                h.std().item(),
                h.min().item(),
                h.max().item(),
            )

            vv = self.value_head(h)
            print(
                "value_head(h) mean/std/min/max:",
                vv.mean().item(),
                vv.std().item(),
                vv.min().item(),
                vv.max().item(),
            )

            vflat = vv.flatten(1)

            v1 = F.relu(self.value_fc1(vflat))
            print(
                "value_fc1 out mean/std/min/max:",
                v1.mean().item(),
                v1.std().item(),
                v1.min().item(),
                v1.max().item(),
            )

            v2 = torch.tanh(self.value_fc2(v1)).squeeze(-1)
            print("FINAL v:", v2.item())



# ----------------------------
# Helpers
# ----------------------------
def obs_tensor(state: pyspiel.State) -> np.ndarray:
    # OpenSpiel returns flat list; reshape using game shape.
    # We'll reshape to (C,H,W) for CNN.
    obs = np.array(state.observation_tensor(), dtype=np.float32)
    # For crazyhouse you saw shape [38,8,8]

    return obs.reshape((38, 8, 8))


def legal_mask(state: pyspiel.State, num_actions: int) -> np.ndarray:
    mask = np.zeros((num_actions,), dtype=np.float32)
    for a in state.legal_actions():
        mask[a] = 1.0
    return mask


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    logits: (B, A)
    mask:   (B, A) in {0,1}
    Returns probs with zero on illegal actions.
    """
    # Put -inf on illegal moves so softmax assigns 0 prob
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask > 0, logits, neg_inf)
    return F.softmax(masked_logits, dim=dim)

def run_mcts(state: pyspiel.State,
             net: PVNet,
             num_simulations: int = 800,
             c_puct: float = 1.5,
             temperature: float = 1.0):
    """
    Returns:
      pi: np.ndarray of shape (num_actions,)
      a: selected action (int)
    """
    num_actions = state.get_game().num_distinct_actions()
    root = {}
    legal = state.legal_actions()

    # --- Expand root ---
    obs = obs_tensor(state)
    x = torch.from_numpy(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = net(x)
        value = value.item()
        mask = torch.zeros((1, num_actions), device=device)
        mask[0, legal] = 1
        priors = masked_softmax(logits, mask)[0].cpu().numpy()
    # After priors = masked_softmax(...).cpu().numpy()
    # and after `legal = state.legal_actions()`

    epsilon = 0.25          # how much noise to mix in
    alpha = 0.3             # Dirichlet concentration (works fine as a starting point)

    noise = np.random.dirichlet([alpha] * len(legal)).astype(np.float32)

    # mix noise into priors ONLY on legal moves
    for i, a in enumerate(legal):
       priors[a] = (1 - epsilon) * priors[a] + epsilon * noise[i]

    for a in legal:
        root[a] = MCTSNode(prior=priors[a])

    # --- MCTS simulations ---
    for _ in range(num_simulations):
        sim_state = state.clone()
        path = []
        node = root
        #player = sim_state.current_player()
        player = 1 - sim_state.current_player()

        # Selection
        while True:
            total_visits = sum(n.visit_count for n in node.values())
            best_score = -1e9
            best_action = None

            for a, n in node.items():
                u = (
                    n.value
                    + c_puct * n.prior * math.sqrt(total_visits + 1) / (1 + n.visit_count)
                )
                if u > best_score:
                    best_score = u
                    best_action = a
            cur_player = sim_state.current_player()
            path.append((node, best_action, cur_player))
            sim_state.apply_action(best_action)

            if sim_state.is_terminal():
                returns = sim_state.returns()
                leaf_value = returns[0]
                break

            if sim_state.is_chance_node():
                outcomes = sim_state.chance_outcomes()
                acts, probs = zip(*outcomes)
                sim_state.apply_action(random.choices(acts, probs)[0])
                continue

            # Expand
            legal = sim_state.legal_actions()
            obs = obs_tensor(sim_state)
            x = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, v = net(x)
                leaf_value = v.item()
                mask = torch.zeros((1, num_actions), device=device)
                mask[0, legal] = 1
                priors = masked_softmax(logits, mask)[0].cpu().numpy()

            new_node = {}
            for a in legal:
                new_node[a] = MCTSNode(prior=priors[a])

            node = new_node
            break

        # Backprop
        for node, a, cur_player in reversed(path):
            n = node[a]
            n.visit_count += 1
            if cur_player == 0:
               n.value_sum += leaf_value      # player 0 wants to maximize
            else:
               n.value_sum -= leaf_value      # player 1 wants to minimize

    # --- Build policy from visit counts ---
    pi = np.zeros(num_actions, dtype=np.float32)
    visits = np.array([root[a].visit_count if a in root else 0 for a in range(num_actions)])

    if temperature == 0:
        a = visits.argmax()
        pi[a] = 1.0
    else:
        visits = visits ** (1 / temperature)
        pi = visits / visits.sum()
        a = np.random.choice(num_actions, p=pi)

    return pi, int(a)



@dataclass
class BatchStats:
    games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_game_length: int = 0

    def reset(self):
        self.games = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_game_length = 0

    def record_game(self, result, game_length):
        """
        result: +1 = white win
                -1 = black win
                 0 = draw
        """
        self.games += 1
        self.total_game_length += game_length

        if result > 0:
            self.white_wins += 1
        elif result < 0:
            self.black_wins += 1
        else:
            self.draws += 1

    def report(self):
        if self.games == 0:
            return "No games played"

        avg_len = self.total_game_length / self.games

        w = 100 * self.white_wins / self.games
        b = 100 * self.black_wins / self.games
        d = 100 * self.draws / self.games

        return (
            "Batch Stats: "
            f"Games: {self.games} | "
            f"Avg length: {avg_len:.1f} | "
            f"White: {w:.1f}% | "
            f"Black: {b:.1f}% | "
            f"Draws: {d:.1f}%"
        )

batch_stats = BatchStats()

@dataclass
class Transition:
    obs: np.ndarray           # (C,H,W)
    mask: np.ndarray          # (A,)
    action: int
    player: int               # player who took the action
    pi: np.ndarray   # (A,) target policy from MCTS visit counts


def play_selfplay_game(game: pyspiel.Game, net: PVNet, device: torch.device,
                       temperature: float = 1.0) -> Tuple[List[Transition], List[float]]:
    """
    Plays one self-play game using the current policy (stochastic).
    Returns:
      transitions: list of (obs, legal_mask, action, player)
      returns: terminal returns from OpenSpiel (length NumPlayers)
    """
    state = game.new_initial_state()
    transitions: List[Transition] = []
    num_actions = game.num_distinct_actions()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            a = random.choices(actions, probs)[0]
            state.apply_action(a)
            continue

        p = state.current_player()
        o = obs_tensor(state)
        m = legal_mask(state, num_actions)


        # NN forward
        x = torch.from_numpy(o).unsqueeze(0).to(device)   # (1,C,H,W)
        mm = torch.from_numpy(m).unsqueeze(0).to(device)  # (1,A)
        pi, a = run_mcts(state, net)

        transitions.append(Transition(obs=o, mask=m, action=a, player=p, pi=pi))
        state.apply_action(a)
    returns = state.returns()
    if returns[1] > 0:
       result = +1    # white win
    elif returns[1] < 0:
        result = -1    # black win
    else:
        result = 0     # draw
    game_length = state.move_number()
    batch_stats.record_game(result, game_length)
    return transitions, state.returns()


def train_step(net: PVNet, optimizer: torch.optim.Optimizer, batch: List[Transition],
               returns: List[float], device: torch.device,
               value_coef: float = 1.0, entropy_coef: float = 0.01) -> Tuple[float, float, float]:
    """
    One policy-gradient update from a single game (or concatenated games).
    Uses terminal return as Monte Carlo target.
    Advantage = (z - v(s)), where z is return for player 0 black
    """
    num_actions = len(batch[0].mask)

    # Build tensors
    obs_b = torch.from_numpy(np.stack([t.obs for t in batch], axis=0)).to(device)          # (T,C,H,W)
    mask_b = torch.from_numpy(np.stack([t.mask for t in batch], axis=0)).to(device)        # (T,A)
    act_b = torch.tensor([t.action for t in batch], device=device, dtype=torch.long)       # (T,)
    ply_b = torch.tensor([t.player for t in batch], device=device, dtype=torch.long)       # (T,)

    # Forward
    logits, v = net(obs_b)   # logits (T,A), v (T,)

    probs = masked_softmax(logits, mask_b)  # (T,A)
    logp = torch.log(torch.clamp(probs, min=1e-12))  # (T,A)
    chosen_logp = logp.gather(1, act_b.view(-1, 1)).squeeze(1)  # (T,)

    ent = -(probs * logp).sum(dim=1)


    # Terminal return for each transition, from the perspective of the acting player
    # returns is a Python list like [r0, r1]
    r = torch.tensor(returns, device=device, dtype=torch.float32)  # (P,)

    # sanity checks
    assert torch.isfinite(logits).all()
    assert torch.isfinite(v).all()
    assert torch.isfinite(mask_b).all()

    # print("v stats:", v.detach().min().item(), v.detach().mean().item(), v.detach().max().item())
    # print("v first 10:", v.detach()[:10].cpu().numpy())
    pi_b = torch.from_numpy(np.stack([t.pi for t in batch])).to(device)
    policy_loss = -(pi_b * logp).sum(dim=1).mean()
    #z = r[ply_b]
    z = torch.full_like(v, r[0])

    value_loss = F.mse_loss(v, z)
    #print("v min/max:", v.min().item(), v.max().item(), "target z:", z.unique().tolist())

    # Entropy bonus (encourage exploration)
    entropy = -(probs * logp).sum(dim=1).mean()  # mean over time
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
    #print("grad norm:", total_norm)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item()), float(entropy.item())

def get_model():

    game = pyspiel.load_game("crazyhouse")
    shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    # print("Obs shape:", shape, "Num actions:", num_actions)

    # sanity: expected shape [38,8,8]
    in_channels = shape[0]

    model = PVNet(in_channels=in_channels, num_actions=num_actions, channels=64).to(device)
    ckpt_path = "crazyhouse_pvnet.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    return model


# ----------------------------
# Main training loop
# ----------------------------
def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)

    game = pyspiel.load_game("crazyhouse")
    shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    save_every = 100
    print("Obs shape:", shape, "Num actions:", num_actions)

    # sanity: expected shape [38,8,8]
    in_channels = shape[0]

    net = PVNet(in_channels=in_channels, num_actions=num_actions, channels=64).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    g = 0
    ckpt_path = "crazyhouse_pvnet.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        g = ckpt["game"]
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("Starting from scratch")

    # Training params
    temperature = 1.0        # >1 more random, <1 more greedy
    value_coef = 10.0
    entropy_coef = 0.001

    while True:
        g += 1
        traj, rets = play_selfplay_game(game, net, device, temperature=temperature)

        pl, vl, ent = train_step(
            net, optimizer, traj, rets, device,
            value_coef=value_coef, entropy_coef=entropy_coef
        )

        if g % 10 == 0:
            print(f"Game {g+1:4d} | returns={rets} | policy_loss={pl:.6f} | value_loss={vl:.6f} | entropy={ent:.6f}")
        if  g  % save_every == 0:
            torch.save({
               "model": net.state_dict(),
               "optimizer": optimizer.state_dict(),
               "game": g,
            }, f"crazyhouse_pvnet_run_{g}.pt")
            print(batch_stats.report())

            batch_stats.reset()
            print("Saved model")

    # Save model


if __name__ == "__main__":
    main()
