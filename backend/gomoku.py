import pyspiel
from util import terminal_payload

class Gomoku:
    def __init__(self):
        self.game_name = "gomoku"
        self.game = None

    def get_game_string(self, data):
        game_params = ""
        size = data.get("size")
        if size and size != 15:
            game_params = f"size={size}"
        connect = data.get("connect")
        if connect and connect != 5:
            if game_params:
                game_params = game_params + ","
            game_params = game_params + f"connect={connect}"
        dims = data.get("dims")
        if dims and dims != 5:
            if game_params:
                game_params = game_params + ","
            game_params = game_params + f"dims={dims}"
        wrap =  data.get("wrap")
        if wrap:
            if game_params:
                game_params = game_params + ","
            if wrap:
                game_params = game_params + "wrap=true"
        if game_params:
            game_params = "(" + game_params + ")"
        print("game_params", game_params)
        return self.game_name + game_params

    def get_initial_state(self, data):
        game_string = self.get_game_string(data)
        self.game = pyspiel.load_game(game_string)
        state = self.game.new_initial_state()
        return state

    def get_state_data(self, state, last_move):
        statestr = str(state)
        result = {}
        result['type'] = 'state'
        result['pom'] = 'black' if statestr[0] == 'B' else 'white'
        result['board'] = statestr[1:]
        if last_move:
            result['last_move'] = last_move
        if state.is_terminal():
            result.extend(terminal_payload(state))
        return result

    def handle_command(self, data):
        cmd = data['cmd']
        return {}, False

