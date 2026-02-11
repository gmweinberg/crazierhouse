import math
import random
import numpy as np
import torch
import torch.nn.functional as F


class Node:
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


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask > 0, logits, neg_inf)
    return F.softmax(masked_logits, dim=-1)


class MCTS:
    def __init__(
        self,
        game,
        net,
        device,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.game = game
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.num_actions = game.num_distinct_actions()

    def _expand(self, state):
        """Evaluate state with NN and return (priors, value)."""
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.net(x)
            value = value.item()  # from black's perspective

            mask = torch.zeros((1, self.num_actions), device=self.device)
            mask[0, state.legal_actions()] = 1.0

            priors = masked_softmax(logits, mask)[0].cpu().numpy()

        return priors, value

    def run(self, root_state):
        root_state = root_state.clone()
        root = {}

        # --- Root expansion ---
        priors, _ = self._expand(root_state)
        legal = root_state.legal_actions()

        # Dirichlet noise (root only)
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(legal)
        ).astype(np.float32)

        for i, a in enumerate(legal):
            p = priors[a]
            p = (1 - self.dirichlet_eps) * p + self.dirichlet_eps * noise[i]
            root[a] = Node(prior=p)

        # --- Simulations ---
        for _ in range(self.num_simulations):
            state = root_state.clone()
            node = root
            path = []
            value_sign = 1.0  # flips every ply

            while True:
                # Selection
                total_visits = sum(n.visit_count for n in node.values())
                best_score = -1e9
                best_action = None

                for a, n in node.items():
                    u = (
                        n.value
                        + self.c_puct
                        * n.prior
                        * math.sqrt(total_visits + 1)
                        / (1 + n.visit_count)
                    )
                    if u > best_score:
                        best_score = u
                        best_action = a

                path.append((node, best_action, value_sign))
                state.apply_action(best_action)
                value_sign = -value_sign

                # Terminal
                if state.is_terminal():
                    leaf_value = state.returns()[0]  # black perspective
                    break

                # Chance nodes (not used in Gomoku, but safe)
                if state.is_chance_node():
                    actions, probs = zip(*state.chance_outcomes())
                    state.apply_action(random.choices(actions, probs)[0])
                    continue

                # Expansion
                priors, leaf_value = self._expand(state)
                new_node = {}
                for a in state.legal_actions():
                    new_node[a] = Node(prior=priors[a])
                node = new_node
                break

            # Backpropagation
            for node, action, sign in reversed(path):
                n = node[action]
                n.visit_count += 1
                n.value_sum += sign * leaf_value

        # --- Build policy ---
        pi = np.zeros(self.num_actions, dtype=np.float32)
        for a, n in root.items():
            pi[a] = n.visit_count

        pi /= pi.sum()
        return pi
