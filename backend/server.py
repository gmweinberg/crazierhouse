from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import random
import pyspiel

app = FastAPI()

# ----------------------------------------------------
# Unicode mapping for FEN chars
# ----------------------------------------------------
FEN_TO_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
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
                    game = pyspiel.load_game("chess")
                    state = game.new_initial_state()

                # Chess960
                elif startpos == "random":
                    game = pyspiel.load_game("chess(chess960=true)")
                    state = game.new_initial_state()
                    apply_chess960_nature(state)

                else:
                    game = pyspiel.load_game("chess")
                    state = game.new_initial_state()

                # If human plays black, AI moves first
                if side == "black" and not state.is_terminal():
                    legal = state.legal_actions()
                    if legal:
                        ai_action = random.choice(legal)
                        state.apply_action(ai_action)

                # Send initial board
                fen = str(state)
                board = fen_to_board(fen)

                await ws.send_json({
                    "type": "state",
                    "board": board,
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
                            legal = state.legal_actions()
                            if legal:
                                ai_action = random.choice(legal)
                                ai_uci = state.action_to_string(ai_action)
                                state.apply_action(ai_action)
                                last_move_uci = ai_uci
                    else:
                        print("Illegal move:", uci)

                # Send updated board
                fen = str(state)
                board = fen_to_board(fen)

                await ws.send_json({
                    "type": "state",
                    "board": board,
                    "last_move": last_move_uci
                })

    except WebSocketDisconnect:
        return
