import pyspiel
from util import terminal_payload

def get_position_result(position):
    # Send updated board
    return {
        "type": "state",
        "board": position.board,
        "pockets": position.pockets
    }

class Position:
    def __init__(self, fen):
        self.board = fen_to_board(fen)
        self.pockets = parse_fen_pockets(fen)
        self.white_castle_king = True
        self.white_castle_queen = True
        self.black_castle_king = True
        self.black_castle_queen = True
        self.moves = 0
        self.to_move = "white"
        self.epsquare = None

    def set_square(self, x, y, piece):
        self.board[y][x] = piece

    def pocket(inc, piece, color):
        pass

    def castling(self):
        result = ''
        result += 'K' if  self.white_castle_king else ''
        result += 'Q' if  self.white_castle_queen else ''
        result += 'k' if self.black_castle_king else ''
        result += 'q' if self.black_castle_queen else ''
        return result

    def to_fen(self):
        board_part = board_to_fen(self.board)
        pocket_part = pockets_to_fen(self.pockets)

        side = "w" if self.to_move == "white" else "b"
        castling = self.castling()
        ep = self.epsquare if self.epsquare else '-'

        halfmove = "0"        # Crazyhouse engines usually ignore this
        fullmove = str(self.moves + 1)

        return f"{board_part}{pocket_part} {side} {castling} {ep} {halfmove} {fullmove}"
        #return f"{board_part} {side} {castling} {ep} {halfmove} {fullmove}"

class Crazyhouse:
    def __init__(self):
        self.game_name = "crazyhouse"
        self.game = None
        self.state = None
        self.position = None

    def get_game_string(self, data):
        startpos = data.get("startpos", "standard")
        insanity = int(data.get("insanity", 1))
        koth = bool(data.get("koth", False))
        sticky = bool(data.get("sticky", False))
        chance_node = False
        game_params = ""
        if startpos == "random":
            chance_node = True
            game_params = "chess960=true"
        if koth:
            if game_params:
                game_params += ","
            game_params += "king_of_hill=true"
        if sticky:
            if game_params:
                game_params += ","
            game_params += "sticky_promotions=true"
        if insanity != 1:
            if game_params:
                game_params += ","
            game_params += "insanity=" + str(insanity)
        if game_params:
            game_params = "(" + game_params + ")"
        print("game_params", game_params)
        game_string = self.game_name + game_params
        return game_string

    def get_initial_state(self, data):
        game_string = self.get_game_string(data)
        game = pyspiel.load_game(game_string)
        startpos = data.get("startpos", "standard")
        chance_node = False
        if startpos == "random":
            chance_node = True
        self.game = pyspiel.load_game(game_string)
        fen = data.get('fen')
        if fen:
            state = self.game.new_initial_state(fen)
        else:
            state = self.game.new_initial_state()
            if chance_node:
                apply_chess960_nature(state)
        self.state = state
        return state

    def get_state_data(self, last_move=None):
        state = self.state
        # Send updated board
        fen = str(state)
        print("fen", fen)
        board = fen_to_board(fen)
        pockets = parse_fen_pockets(fen)
        result = {"type": "state",
            "board": board,
            "pockets": pockets,
            "last_move": last_move}

        if state.is_terminal():
            result.update(terminal_payload(state))
        return result

    def apply_player_move(self, data):
        state = self.state
        uci = data.get("uci", None)
        last_move_uci = None
        if uci is None:
            print("Client did not send UCI move!")
            return False, None
        try:
            print("uci", uci)
            action = state.parse_move_to_action(uci)
        except Exception as e:
            print("parse_move_to_action failed for", uci, e)
            return False, None
        legal = state.legal_actions()

        # Apply human move
        if action is not None and action in legal:
            state.apply_action(action)
            last_move_uci = uci
            return True, uci
        else:
            print("Illegal move:", uci)
            return False, None




    def handle_command(self, data):
        cmd = data['cmd']

        if cmd == "reset_position":
            state = self.get_initial_state(data)
            self.state = state
            fen = str(state)
            self.position = Position(fen)
            result = get_position_result(self.position)
            return result, True

        if cmd == "set_square":
            print("data", data)
            x = data.get('x')
            y = data.get('y')
            piece = data.get('piece')
            print(self.position.board)
            if x and y and (piece  is not None):
                self.position.set_square(x, y, piece)
            print("board", self.position.board)
            result = get_position_result(self.position)
            return result, True

        if cmd == "get_fen":
            handled = True
            fen = self.position.to_fen()
            print("fen", fen)
            result = {"type":"fen", "fen":fen}
            return result, True


        if cmd == "round_trip_fen":
            fen = data.get('fen')
            new_pos = Position(fen)
            new_fen = new_pos.to_fen()
            print("old", fen)
            print("new", new_fen)
            return {}, True

        return {}, False

# ----------------------------------------------------
# Unicode mapping for FEN chars.
# We draw the promoted pieces looking like the regular ones
# ----------------------------------------------------
FEN_TO_UNICODE = {
    "P": "♙",
    "N": "♘",
    "H":  "♘",
    "B": "♗",
    "A": "♗",
    "R": "♖",
    "C": "♖",
    "Q": "♕",
    "E": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "h": "♞",
    "b": "♝",
    "a": "♝",
    "r": "♜",
    "c": "♜",
    "q": "♛",
    "e": "♛",
    "k": "♚",
}

def un_unicode(char):
    if char == "♙":
        return 'P'
    if char == "♘":
        return 'N'
    if char == "♗":
        return 'B'
    if char == "♖":
        return 'R'
    if char == "♕":
        return 'Q'
    if char == "♔":
        return 'K'
    if char == "♟":
        return 'p'
    if char == "♞":
        return 'n'
    if char == "♝":
        return 'b'
    if char == "♜":
        return 'r'
    if char == "♛":
        return 'q'
    if char == "♚":
        return 'k'
    return char

def fen_to_board(fen: str):
    """Convert FEN into 8×8 array of Unicode chess pieces."""
    piece_field = fen.split()[0]
    rows = piece_field.split("/")
    board = []

    for row in rows:
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend([""] * int(ch))
            else:
                cells.append(FEN_TO_UNICODE.get(ch, ""))
        # Ensure row is 8 wide
        board.append(cells[:8] + [""] * (8 - len(cells)))

    return board

def board_to_fen(board):
    rows = []
    for rank in board:
        empty = 0
        row = ""
        for sq in rank:
            if sq is None or sq == "." or sq == '':
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += un_unicode(sq)
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)

def parse_fen_pockets(fen: str):
    pockets = {"white": {}, "black": {}}

    if "[" not in fen:
        return pockets

    pocket_str = fen[fen.index("[")+1 : fen.index("]")]

    for ch in pocket_str:
        if ch.isupper():
            pockets["white"][ch] = pockets["white"].get(ch, 0) + 1
        else:
            pockets["black"][ch] = pockets["black"].get(ch, 0) + 1

    return pockets

def pockets_to_fen(pockets):
    s = ""

    # White pocket (uppercase)
    for piece in ["P", "N", "B", "R", "Q"]:
        s += piece * pockets["white"].get(piece, 0)

    # Black pocket (lowercase)
    for piece in ["p", "n", "b", "r", "q"]:
        s += piece * pockets["black"].get(piece, 0)

    return f"[{s}]"


def apply_chess960_nature(state):
    """Apply the Chess960 nature move (random starting setup)."""
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = random.choices(actions, probs)[0]
        state.apply_action(action)


