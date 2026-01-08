from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import math
import random
import time
import pyspiel

import torch
from train_value_policy import PVNet  # or wherever you defined it
import numpy as np
import torch.nn.functional as F
from min_mcts import get_mcts_stuff

app = FastAPI()

mcts_stuff = get_mcts_stuff()
mcts_bot = mcts_stuff['mcts_bot']
evaluator = mcts_stuff['evaluator']
model = mcts_stuff['model']


def apply_random_move(state):
    legal_actions = state.legal_actions()
    if legal_actions:
        random_action = random.choice(legal_actions)
        uci = state.action_to_string(random_action)
        state.apply_action(random_action)
        return uci

def apply_mcts_move(state, mcts_bot):
    #print("MCTS MOVE for player:", state.current_player())
    ai_action = mcts_bot.step(state)
    if ai_action is None:
        return
    if ai_action not in state.legal_actions():
        print("ILLEGAL ACTION:", ai_action)
        print("LEGAL:", state.legal_actions())
        raise RuntimeError("Illegal MCTS action")
    ai_uci = state.action_to_string(ai_action)
    v = evaluator.evaluate(state)
    print(state.current_player(), v)
    cp = state.current_player()
    print("value_for_side_to_move:", v[cp], "raw:", v)
    state.apply_action(ai_action)
    # diagnostic
    # obs2 = obs.reshape(self.obs_shape)
    #x = torch.from_numpy(obs2).unsqueeze(0).to(self.device)

    #with torch.no_grad():
    #    policy, value = model(x)

    #print("RAW value:", value.item())
    return ai_uci

def print_eval(state):
    v = evaluator.evaluate(state)
    print(v)

def terminal_payload(state):
    returns = state.returns()
    print(returns)
    if returns[0] > returns[1]:
        result = "black_win"
    elif returns[1] > returns[0]:
        result = "white_win"
    else:
        result = "draw"

    return {
        "type": "terminal",
        "result": result,
        "returns": returns,
    }


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

def whiteOnMove(state):
    if state.current_player() == 1:
        return True
    return False

def blackOnMove(state):
    return not whiteOnMove(state)

def maybe_bot_move(state, players):
    if players.human_on_move(state):
        return
    if state.is_terminal():
        return
    if (whiteOnMove(state) and players.white) == "bot" or (blackOnMove(state) and players.black == 'bot'):
        last_move_uci = apply_mcts_move(state, mcts_bot)
    else:
         last_move_uci = apply_random_move(state)
    return last_move_uci

def get_game_string(data):
    startpos = data.get("startpos", "standard")
    insanity = int(data.get("insanity", 1))
    koth = bool(data.get("koth", False))
    chance_node = False
    game_params = ""
    if startpos == "random":
        chance_node = True
        game_params = "chess960=true"
    if koth:
        if game_params:
            game_params += ","
        game_params += "king_of_hill=true"
    if insanity != 1:
        if game_params:
            game_params += ","
        game_params += "insanity=" + str(insanity)
    if game_params:
        game_params = "(" + game_params + ")"
    print("game_params", game_params)
    game_string = "crazyhouse" + game_params
    return game_string

# ----------------------------------------------------
# WebSocket endpointstate
# ----------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    game = None
    state = None
    whitePlayer = None
    blackPlayer = None
    position = None

    try:
        while True:
            handled = False
            data = await ws.receive_json()
            cmd = data.get("cmd")
            print("cmd", cmd)

            # ----------------------------------------
            #                NEW GAME
            # ----------------------------------------
            if cmd == "newgame":
                handled = True
                print('data', data)
                #side = data.get("side", "white")
                whitePlayer = data.get("whitePlayer", "human")
                blackPlayer = data.get("blackPlayer", "bot")
                players = Players(data)

                startpos = data.get("startpos", "standard")
                chance_node = False
                if startpos == "random":
                    chance_node = True
                game_string = get_game_string(data)
                game = pyspiel.load_game(game_string)
                fen = data.get('fen')
                if fen:
                    state = game.new_initial_state(fen)
                else:
                    state = game.new_initial_state()
                    if chance_node:
                        apply_chess960_nature(state)
                print("state", state)
                if players.all_bots():
                    await playBotGame(ws=ws, state=state, players=players)

                # Send initial board
                await send_state(ws, state, None)
                last_move_uci = maybe_bot_move(state, players)
                if last_move_uci:
                    await send_state(ws, state, last_move_uci)


            # ----------------------------------------
            #                PLAYER MOVE
            # ----------------------------------------
            if cmd == "move":
                handled = True
                if game is not None and state is not None:
                    if players.human_on_move(state):
                        uci = data.get("uci", None)
                        last_move_uci = None

                        if uci is None:
                            print("Client did not send UCI move!")
                        else:
                            # Parse human move
                            try:
                                print("uci", uci)
                                action = state.parse_move_to_action(uci)
                            except Exception as e:
                                print("parse_move_to_action failed for", uci, e)
                                action = None

                            legal = state.legal_actions()

                            # Apply human move
                            if action is not None and action in legal:
                                state.apply_action(action)
                                print_eval(state)
                                last_move_uci = uci

                            else:
                                print("Illegal move:", uci)

                        # Send updated board
                        await send_state(ws, state, last_move_uci)
                        moved = maybe_bot_move(state, players)
                        if moved:
                            await send_state(ws, state, last_move_uci)

            if cmd == "reset_position":
                handled = True
                game_string = get_game_string(data)
                game = pyspiel.load_game(game_string)
                state = game.new_initial_state()
                fen = str(state)
                position = Position(fen)
                await send_position(ws, position)

            if cmd == "set_square":
                handled = True
                print("data", data)
                x = data.get('x')
                y = data.get('y')
                piece = data.get('piece')
                print(position.board)
                if x and y and (piece  is not None):
                    position.set_square(x, y, piece)
                    print("board", position.board)
                    await send_position(ws, position)

            if cmd == "get_fen":
                handled = True
                fen = position.to_fen()
                print("fen", fen)
                await send_fen(ws, fen)


            if cmd == "round_trip_fen":
                handled = True
                fen = data.get('fen')
                new_pos = Position(fen)
                new_fen = new_pos.to_fen()
                print("old", fen)
                print("new", new_fen)


            if not handled:
                print("unknown command", data)
    except WebSocketDisconnect:
        return

class Position():
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

        #return f"{board_part}{pocket_part} {side} {castling} {ep} {halfmove} {fullmove}"
        return f"{board_part} {side} {castling} {ep} {halfmove} {fullmove}"

class Players:
    def __init__(self, data):
        self.white = data.get("whitePlayer", "human")
        self.black = data.get("blackPlayer", "bot")

    def all_bots(self):
        if self.white in ['bot', 'random'] and self.black in ['bot', 'random']:
            return True
        return False

    def human_on_move(self, state):
        if (self.white == 'human' and whiteOnMove(state)) or (self.black == 'human' and blackOnMove(state)):
            return True
        return False

async def send_state(ws, state, last_move):
    # Send updated board
    fen = str(state)
    print("fen", fen)
    board = fen_to_board(fen)
    pockets = parse_fen_pockets(fen)

    await ws.send_json({
        "type": "state",
        "board": board,
        "pockets": pockets,
        "last_move": last_move
    })
    if state.is_terminal():
        print("this is the end")
        await ws.send_json(terminal_payload(state))

async def send_position(ws, position):
    # Send updated board

    await ws.send_json({
        "type": "state",
        "board": position.board,
        "pockets": position.pockets
    })

async def send_fen(ws, fen):
    await ws.send_json({
        "type": "fen",
        "fen": fen
    })


#playBotGame(ws=ws, state=state, blackPlayer=blackPlayer, whitePlayer=whitePlayer)
async def playBotGame(ws, state, players):
    print('blackPlayer', players.black, 'whitePlayer', players.white)
    while True:
        last_move_uci = None
        if state.is_terminal():
            break

        if (whiteOnMove(state) and players.white == 'bot') or (blackOnMove(state) and players.black == 'bot'):
            last_move_uci = apply_mcts_move(state, mcts_bot)
        else:
             last_move_uci = apply_random_move(state)
        if last_move_uci:
            await send_state(ws, state, last_move_uci)
            await asyncio.sleep(1)
