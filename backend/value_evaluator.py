import torch
import numpy as np
from open_spiel.python.algorithms.mcts import Evaluator

class ValueNetEvaluator(Evaluator):
    def __init__(self, game, model, device):
        self.game = game
        self.model = model
        self.device = device
        #self.model.eval()
        self.model.train()

        # cache expected shape once
        self.obs_shape = tuple(game.observation_tensor_shape())  # (38,8,8)

    def _obs(self, state):
        obs = np.asarray(state.observation_tensor(), dtype=np.float32).reshape(self.obs_shape)
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1,C,H,W)
        return x

    def evaluate(self, state):
        # MCTS expects a value per player (2p zero-sum)
        #print("EVAL state id:", id(state), "ply:", state.move_number())
        if state.is_terminal():
            return state.returns()

        x = self._obs(state)
        with torch.no_grad():
            _logits, v = self.model(x)          # v shape (1,)
            v0 = float(v.item())                # player-0 perspective

        return [v0, -v0]


    def prior(self, state):
        legal = state.legal_actions()
        if not legal:
            return []

        x = self._obs(state)
        with torch.no_grad():
            logits, _ = self.model(x)

        logits = logits.squeeze(0).cpu().numpy()

        # Mask illegal moves
        mask = np.zeros_like(logits)
        mask[legal] = 1.0

        probs = np.exp(logits - np.max(logits))
        probs *= mask
        probs /= probs.sum()

        return [(a, float(probs[a])) for a in legal]

