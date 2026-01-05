#!/usr/bin/env python
import random
from open_spiel.python.algorithms import mcts
import numpy as np
import pyspiel
import torch
from value_evaluator import ValueNetEvaluator
from train_value_policy import PVNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
black = None
white = None
#TODO: support variants
def get_game():
    game = pyspiel.load_game("crazyhouse")
    return game

def get_model(ckpt_path):

    game = pyspiel.load_game("crazyhouse")
    shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    # print("Obs shape:", shape, "Num actions:", num_actions)

    # sanity: expected shape [38,8,8]
    in_channels = shape[0]

    model = PVNet(in_channels=in_channels, num_actions=num_actions, channels=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt['model'])
    model.eval()

    return model

def play_game(white_bot, black_bot, game):
    state = game.new_initial_state()

    while not state.is_terminal():
        cp = state.current_player()
        if cp == 0:
            action = white_bot.step(state)
        else:
            action = black_bot.step(state)
        state.apply_action(action)

    return state.returns()  # [r0, r1]

def make_bot(game, model, device, seed=None, max_sims=800):
    if seed is None:
        seed = random.random()
    evaluator = ValueNetEvaluator(game, model, device)
    return mcts.MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=max_sims,
        evaluator=evaluator,
        solve=False,
        random_state=np.random.RandomState(seed),
    )

def run_match(old_model, new_model, num_pairs):
    game = get_game()
    set_player_ids(game)

    stats = {
        "new": 0,
        "old": 0,
        "draw": 0,
    }

    for i in range(num_pairs):
        seed = 1000 + i

        # Game 1: new = white
        new_bot = make_bot(game, new_model, device, seed)
        old_bot = make_bot(game, old_model, device, seed)

        r = play_game(new_bot, old_bot, game)
        update_stats(stats, r, new_white=True)

        # Game 2: new = black
        new_bot = make_bot(game, new_model, device, seed)
        old_bot = make_bot(game, old_model, device, seed)

        r = play_game(old_bot, new_bot, game)
        update_stats(stats, r, new_white=False)

        if (i + 1) % 5 == 0:
            report(stats, games=2 * (i + 1))

    return stats

def update_stats(stats, returns, new_white):
    if returns[0] == 0:
        stats["draw"] += 1
    else:
        winner = 0 if returns[0] > 0 else 1

        if new_white:
            new_player = white
        else:
            new_player = black

        if winner == new_player:
            stats["new"] += 1
        else:
            stats["old"] += 1

def report(stats, games):
    winrate = stats["new"] / max(1, games - stats["draw"])
    print(
        f"Games: {games} | "
        f"New: {stats['new']} | "
        f"Old: {stats['old']} | "
        f"Draw: {stats['draw']} | "
        f"New winrate: {winrate:.3f}"
    )

def set_player_ids(game):
    global black
    global white
    for pid in range(game.num_players()):
        name = game.player_to_string(pid).lower()
        if "black" in name:
            black = pid
        elif "white" in name:
            white = pid
    assert black is not None and white is not None
    return black, white

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--old', help="old pt file")
    parser.add_argument('--new', help="new pt file")
    parser.add_argument('--rounds', help="rounds", type=int, default=100)
    args = parser.parse_args()
    old_model = get_model(args.old)
    new_model = get_model(args.new)
    run_match(old_model, new_model, args.rounds)


# ./self_play.py -- old ../training/miles/crazyhouse_pvnet_mile_11700.pt --new ../training/miles/crazyhouse_pvnet_mile_14500.pt

