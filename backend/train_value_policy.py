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
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
        )

        # 8x8 board
        self.policy_fc = nn.Linear(2 * 8 * 8, num_actions)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,C,H,W)
        h = self.trunk(x)

        # Policy
        p = self.policy_head(h).flatten(1)  # (B, 2*8*8)
        logits = self.policy_fc(p)          # (B, num_actions)

        # Value
        v = self.value_head(h).flatten(1)   # (B, 1*8*8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return logits, v


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
        with torch.no_grad():
            logits, _v = net(x)
            if temperature != 1.0:
                logits = logits / temperature
            probs = masked_softmax(logits, mm)[0]  # (A,)

        # Sample an action (stochastic policy)
        a = torch.multinomial(probs, num_samples=1).item()

        transitions.append(Transition(obs=o, mask=m, action=a, player=p))
        state.apply_action(a)
    return transitions, state.returns()


def train_step(net: PVNet, optimizer: torch.optim.Optimizer, batch: List[Transition],
               returns: List[float], device: torch.device,
               value_coef: float = 1.0, entropy_coef: float = 0.01) -> Tuple[float, float, float]:
    """
    One policy-gradient update from a single game (or concatenated games).
    Uses terminal return as Monte Carlo target.
    Advantage = (z - v(s)), where z is return for the acting player.
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

    # Terminal return for each transition, from the perspective of the acting player
    # returns is a Python list like [r0, r1]
    r = torch.tensor(returns, device=device, dtype=torch.float32)  # (P,)
    # z = r[ply_b]  # (T,)
    z = r[0].expand_as(v)

    # Advantage with value baseline
    adv = (z - v).detach()

    # Policy gradient loss: maximize E[logpi * adv] => minimize -logpi * adv
    policy_loss = -(chosen_logp * adv).mean()
    # temprorarily disable policy loss
    # policy_loss = 0

    # Value loss: regress to z
    with torch.no_grad():
        # bootstrap target: v(s_t) ≈ γ * v(s_{t+1})
        gamma = 0.99

        z = torch.empty_like(v)
        z[:-1] = gamma * v[1:]
        z[-1] = r[0]  # terminal value
    value_loss = F.mse_loss(v, z)

    # Entropy bonus (encourage exploration)
    entropy = -(probs * logp).sum(dim=1).mean()  # mean over time
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
    print("grad norm:", total_norm)
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
    num_games = 1000          # start small
    temperature = 1.0        # >1 more random, <1 more greedy
    value_coef = 1.0
    entropy_coef = 0.01

    while True:
        g += 1
    #for g in range(num_games):
        traj, rets = play_selfplay_game(game, net, device, temperature=temperature)

        pl, vl, ent = train_step(
            net, optimizer, traj, rets, device,
            value_coef=value_coef, entropy_coef=entropy_coef
        )

        if g % 10 == 0:
            print(f"Game {g+1:4d} | returns={rets} | policy_loss={pl:.4f} | value_loss={vl:.4f} | entropy={ent:.4f}")
        if  g  % 1000 == 0:
            torch.save({
               "model": net.state_dict(),
               "optimizer": optimizer.state_dict(),
               "game": g,
            }, f"crazyhouse_pvnet_{g}.pt")
            torch.save({
               "model": net.state_dict(),
               "optimizer": optimizer.state_dict(),
               "game": g,
            }, "crazyhouse_pvnet_latest.pt")
            print("Saved model")

    # Save model


if __name__ == "__main__":
    main()
