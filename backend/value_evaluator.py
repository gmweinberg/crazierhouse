# value_evaluator.py
import numpy as np
import torch
from open_spiel.python.algorithms.mcts import Evaluator



class ValueNetEvaluator(Evaluator):
    def __init__(self, game, model, device):
        self.game = game
        self.model = model
        self.device = device
        self.model.eval()
        self.obs_shape = tuple(game.observation_tensor_shape())  # e.g. (38,8,8)

    def evaluate(self, state):
        # MCTS expects a value per player.
        if state.is_terminal():
            return state.returns()  # length = num_players

        obs = np.asarray(state.observation_tensor(), dtype=np.float32).reshape(self.obs_shape)

        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1,C,H,W)
        with torch.no_grad():
            _, value = self.model(x)
            v0 = float(value.item())

        # 2-player zero-sum
        return [v0, -v0]

    def prior(self, state):
        # Uniform prior over legal moves (fine for value-only MCTS)
        legal = state.legal_actions()
        if not legal:
            return []
        p = 1.0 / len(legal)
        return [(a, p) for a in legal]
