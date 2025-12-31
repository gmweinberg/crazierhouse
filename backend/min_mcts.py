from open_spiel.python.algorithms import mcts
import numpy as np
import pyspiel
import torch

from value_evaluator import ValueNetEvaluator
#from value_net import ValueNet  # your model
from train_value_policy import get_model

game = pyspiel.load_game("crazyhouse")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()

def get_mcts_stuff():
    evaluator = ValueNetEvaluator(game, model, device)
    mcts_bot =  mcts.MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=400,
        evaluator=evaluator,
        solve=False,
        random_state=np.random.RandomState(0),
    )
    return {'model':model, 'evaluator':evaluator, 'mcts_bot':mcts_bot}
