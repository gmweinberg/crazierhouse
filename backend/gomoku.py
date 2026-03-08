import pyspiel
import torch
import numpy as np
from util import terminal_payload
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from gomoku9_net import GomokuNet as Gomoku9Net
from gomoku_mcts import MCTS

class MCTSBot:
    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.mcts = MCTS(game, net, DEVICE)

    def step(self, state):
        pi = self.mcts.run(state)
        return np.argmax(pi)

    def value(self, state):
        return self.mcts.value(state)

class Gomoku:
    def __init__(self):
        self.game_name = "gomoku"
        self.game = None
        self.game_params = None
        self.state = None
        self.dims = None
        self.size = None
        self.connect = None
        self.wrap = None

    def get_initial_state(self, data):
        params = {}
        params['size'] = data.get("size", 15)
        params['connect'] = data.get("connect", 5)
        params['dims'] = data.get("dims", 2)
        params['wrap'] =  data.get("wrap", False)
        self.game_params = params
        self.game = pyspiel.load_game(self.game_name, params)
        state = self.game.new_initial_state()
        self.state = state
        return state

    def get_state_data(self, last_action):
        state = self.state
        statestr = str(state)
        result = {}
        result['type'] = 'state'
        result['pom'] = 'black' if statestr[0] == 'B' else 'white'
        result['board'] = statestr[1:]

        if last_action:
            print("last_action", last_action)
            result['last_move'] = self.game.action_to_move(last_action)
        else:
            result['last_move'] = None
        if state.is_terminal():
            result.update(terminal_payload(state))
        return result

    def apply_player_move(self, data):
        print(data)
        coord = data['coord']
        action = self.game.move_to_action(coord=coord)
        legal = self.state.legal_actions()
        if action in legal:
            self.state.apply_action(action)
            return True, action

        return False, None

    def get_mcts_bot(self):
        pt_name = None
        if self.game_params['size'] == 9:
            net = Gomoku9Net().to(DEVICE)
            pt_name = 'gomoku_9.pt'
            checkpoint = torch.load(pt_name, map_location=DEVICE)
            net.load_state_dict(checkpoint['model_state_dict'])
            bot = MCTSBot(self.game, net)
            return bot
        raise ValueError("Unsupported mcts bot")

        



    def handle_command(self, data):
        cmd = data['cmd']
        return {}, False

