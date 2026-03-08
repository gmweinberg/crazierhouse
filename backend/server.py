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
#from min_mcts import get_mcts_stuff
from crazyhouse import Crazyhouse
from gomoku import Gomoku
from shogi import Shogi

app = FastAPI()

#mcts_stuff = get_mcts_stuff()
#mcts_bot = mcts_stuff['mcts_bot']
#model = mcts_stuff['model']
mcts_bot = None


def apply_random_move(state):
    legal_actions = state.legal_actions()
    if legal_actions:
        random_action = random.choice(legal_actions)
        state.apply_action(random_action)
        #uci = state.action_to_string(random_action)
        #return uci
        return random_action

def apply_mcts_move(state, mcts_bot):
    #print("MCTS MOVE for player:", state.current_player())
    ai_action = mcts_bot.step(state)
    if ai_action is None:
        return
    if ai_action not in state.legal_actions():
        print("ILLEGAL ACTION:", ai_action)
        print("LEGAL:", state.legal_actions())
        raise RuntimeError("Illegal MCTS action")
    if True:
        v = mcts_bot.value(state)
        print("value for black:", v)
    state.apply_action(ai_action)
    # diagnostic
    # obs2 = obs.reshape(self.obs_shape)
    #x = torch.from_numpy(obs2).unsqueeze(0).to(self.device)

    #with torch.no_grad():
    #    policy, value = model(x)

    #print("RAW value:", value.item())
    return ai_action


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
        last_action = apply_mcts_move(state, mcts_bot)
    else:
         last_action = apply_random_move(state)
    return last_action

def get_server_class(game_name):
    if game_name == "crazyhouse":
        return Crazyhouse()
    if game_name == "gomoku":
        return Gomoku()
    if game_name == "shogi":
        return Shogi()
    raise Exception("Invalid game name")


# ------------------ await ws.send_json(data)----------------------------------
# WebSocket endpointstate
# ----------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global mcts_bot
    await ws.accept()

    game = None
    state = None
    position = None
    server_class = None

    try:
        while True:
            handled = False
            data = await ws.receive_json()
            # data['game'] = 'crazyhouse' # not anymore :-)
            cmd = data.get("cmd")
            print("cmd", cmd)
            if server_class is None:
                server_class = get_server_class(data['game'])

            # ----------------------------------------
            #                NEW GAME
            # ----------------------------------------
            if cmd == "newgame":
                handled = True
                print('data', data)
                #side = data.get("side", "white")
                state = server_class.get_initial_state(data)
                print("state", state)
                players = Players(data)
                if players.any_mcts_bots():
                    mcts_bot = server_class.get_mcts_bot()
                if players.all_bots():
                    await playBotGame(ws=ws, state=state, players=players, server_class=server_class)

                # Send initial board
                initial_data = server_class.get_state_data(None)
                #print("initial data", initial_data)
                await ws.send_json(initial_data)

                # await ws.send_json(terminal_payload(state))

                last_action = maybe_bot_move(state, players)
                if last_action:
                    bot_move_data =  server_class.get_state_data(last_action)
                    await ws.send_json(bot_move_data)


            # ----------------------------------------
            #                PLAYER MOVE
            # ----------------------------------------
            if cmd == "move":
                handled = True
                if state is not None:
                    if players.human_on_move(state):
                        applied, last_action =  server_class.apply_player_move(data)
                    if applied:
                        move_data = server_class.get_state_data(last_action)
                        print("sending", move_data)
                        await ws.send_json(move_data)
                        
                        # Send updated board
                        last_action = maybe_bot_move(state, players)
                        if last_action:
                           move_data = server_class.get_state_data(last_action)
                           print("sending", move_data)
                           await ws.send_json(move_data)

            if not handled:
                result, handled = server_class.handle_command(data)
                if result:
                    await ws.send_json(result)

            if not handled:
                print("unknown command", data)
    except WebSocketDisconnect:
        return


class Players:
    def __init__(self, data):
        self.white = data.get("whitePlayer", "human")
        self.black = data.get("blackPlayer", "bot")

    def all_bots(self):
        if self.white in ['bot', 'random'] and self.black in ['bot', 'random']:
            return True
        return False

    def any_mcts_bots(self):
        if self.white == 'bot' or self.black == 'bot':
            return True
        return False

    def human_on_move(self, state):
        if (self.white == 'human' and whiteOnMove(state)) or (self.black == 'human' and blackOnMove(state)):
            return True
        return False



#playBotGame(ws=ws, state=state, blackPlayer=blackPlayer, whitePlayer=whitePlayer)
async def playBotGame(ws, state, players, server_class):
    print('blackPlayer', players.black, 'whitePlayer', players.white)
    while True:
        last_action = None
        if state.is_terminal():
            break

        if (whiteOnMove(state) and players.white == 'bot') or (blackOnMove(state) and players.black == 'bot'):
            last_action= apply_mcts_move(state, mcts_bot)
        else:
             last_action = apply_random_move(state)
        if last_action:
            move_data = server_class.get_state_data(last_action)
            await ws.send_json(move_data)
            await asyncio.sleep(1)
