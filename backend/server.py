from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import random
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


def apply_ai_move(state):
   ai_action = select_ai_move(state)
   if ai_action is not None:
       ai_uci = state.action_to_string(ai_action)
       state.apply_action(ai_action)
       return ai_uci

def apply_random_move(state):
    ai_action = select_ai_move(state)
    if ai_action is not None:
        ai_uci = state.action_to_string(ai_action)
        state.apply_action(ai_action)
        return ai_uci

def apply_mcts_move(state, mcts_bot):
    ai_action = mcts_bot.step(state)
    if ai_action is None:
        return
    if ai_action not in state.legal_actions():
        print("ILLEGAL ACTION:", ai_action)
        print("LEGAL:", state.legal_actions())
        raise RuntimeError("Illegal MCTS action")
    ai_uci = state.action_to_string(ai_action)
    state.apply_action(ai_action)
    v = evaluator.evaluate(state)
    print(state.current_player(), v)
    cp = state.current_player()
    print("value_for_side_to_move:", v[cp], "raw:", v)
    # diagnostic
    obs = np.asarray(state.observation_tensor(), dtype=np.float32)

    print("OBS min/max/sum:",
          obs.min(),
          obs.max(),
          obs.sum())

    # obs2 = obs.reshape(self.obs_shape)
    #x = torch.from_numpy(obs2).unsqueeze(0).to(self.device)

    #with torch.no_grad():
    #    policy, value = model(x)

    #print("RAW value:", value.item())
    return ai_uci



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


def apply_chess960_nature(state):
    """Apply the Chess960 nature move (random starting setup)."""
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = random.choices(actions, probs)[0]
        state.apply_action(action)


# ----------------------------------------------------
# WebSocket endpoint
# ----------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    game = None
    state = None

    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("cmd")
            print("cmd", cmd)

            # ----------------------------------------
            #                NEW GAME
            # ----------------------------------------
            if cmd == "newgame":
                side = data.get("side", "white")
                startpos = data.get("startpos", "standard")
                insanity = int(data.get("insanity", 1))  # unused for now

                # Standard chess
                if startpos == "standard":
                    game = pyspiel.load_game("crazyhouse")
                    state = game.new_initial_state()

                # Chess960
                elif startpos == "random":
                    game = pyspiel.load_game("crazyhouse(chess960=true)")
                    state = game.new_initial_state()
                    apply_chess960_nature(state)

                else:
                    game = pyspiel.load_game("crazyhouse")
                    state = game.new_initial_state()
                print("state", state)

                # If human plays black, AI moves first
                if side == "black" and not state.is_terminal():
                    last_move_uci = apply_mcts_move(state, mcts_bot)
                    #ai_uci = apply_ai_move(state)

                # Send initial board
                fen = str(state)
                board = fen_to_board(fen)
                pockets = parse_fen_pockets(fen)

                await ws.send_json({
                    "type": "state",
                    "board": board,
                    "pockets":pockets,
                    "last_move": None
                })

                continue

            # ----------------------------------------
            #                PLAYER MOVE
            # ----------------------------------------
            if cmd == "move" and game is not None and state is not None:
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
                        last_move_uci = uci

                        # AI reply
                        if not state.is_terminal():
                            #ai_action = select_ai_move(state)
                            last_move_uci = apply_mcts_move(state, mcts_bot)
                            # last_move_uci = apply_ai_move(state)
                    else:
                        print("Illegal move:", uci)

                # Send updated board
                fen = str(state)
                print("fen", fen)
                board = fen_to_board(fen)
                pockets = parse_fen_pockets(fen)

                await ws.send_json({
                    "type": "state",
                    "board": board,
                    "pockets": pockets,
                    "last_move": last_move_uci
                })
                if state.is_terminal():
                    print("this is the end")
                    await ws.send_json(terminal_payload(state))

    except WebSocketDisconnect:
        return

