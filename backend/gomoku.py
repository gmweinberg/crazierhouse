import pyspiel
from util import terminal_payload

class Gomoku:
    def __init__(self):
        self.game_name = "gomoku"
        self.game = None
        self.state = None
        self.dims = None
        self.size = None
        self.connect = None
        self.wrap = None

    def get_game_string(self, data):
        self.size = data.get("size", 15)
        self.connect = data.get("connect", 5)
        self.dims = data.get("dims", 2)
        self.wrap =  data.get("wrap", False)
        swrap = str(self.wrap).lower()
        params = f"(size={self.size},dims={self.dims},connect={self.connect},wrap={swrap})"
        return self.game_name + params

    def get_initial_state(self, data):
        game_string = self.get_game_string(data)
        self.game = pyspiel.load_game(game_string)
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



    def handle_command(self, data):
        cmd = data['cmd']
        return {}, False

