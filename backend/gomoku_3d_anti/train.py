import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel

from net import ValuePolicyNet
from mcts import MCTS

# ----------------------------
# Config
# ----------------------------

GAME_PARAMS = {
    "size": 3,
    "connect": 3,
    "anti": True,
    "dims": 3,
}



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SIMULATIONS = 200
C_PUCT = 1.5

LEARNING_RATE = 1e-3
VALUE_LOSS_WEIGHT = 1.0

NUM_TRAIN_GAMES = 2000
PRINT_EVERY = 50


# ----------------------------
# Game + Net
# ----------------------------

game = pyspiel.load_game("gomoku", GAME_PARAMS)

INPUT_SIZE = len(game.new_initial_state().observation_tensor())
NUM_ACTIONS = game.num_distinct_actions()




net = ValuePolicyNet(
    input_size=INPUT_SIZE,
    num_actions=NUM_ACTIONS,
    hidden_size=128,
).to(DEVICE)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

pt_name = "misery.pt"
if os.path.exists(pt_name):
    print("Loading checkpoint...")
    checkpoint = torch.load(pt_name, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    game_count = checkpoint['step']
else:
    game_count = 0


sym_buffer = defaultdict(list)

mcts = MCTS(
    game=game,
    net=net,
    device=DEVICE,
    num_simulations=NUM_SIMULATIONS,
    c_puct=C_PUCT,
)

# ----------------------------
# Self-play
# ----------------------------

def play_selfplay_game():
    data = []
    oops = True
    while oops:
        oops = False
        state = game.new_initial_state()
        state.apply_action(13)
        for ii in range(7):
            legal = state.legal_actions()
            action = np.random.choice(legal)
            state.apply_action(action)
            if state.is_terminal():
                oops = True
                break
            state.apply_action(26 - action)

    while not state.is_terminal():
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        pi = mcts.run(state)
        sym = state.symmetric_hash()

        action = np.random.choice(NUM_ACTIONS, p=pi)
        #if state.move_number() < 8:
        #    action = np.random.choice(NUM_ACTIONS, p=pi)
        #else:
        #    action = np.argmax(pi)

        data.append((sym, obs, pi))
        state.apply_action(action)
    if False:
        print(str(state))
        print(state.returns())
        print(state.winning_line())
        print("last move", game.action_to_move(state.history()[-1]))
        state.undo_action(0, 0) # params required but ingnored
        print(state.pretty())
        print("all terminal?", all_terminal(state))
        raise Exception()
    if False:
        test = state.clone()
        print(test.returns())
        test.undo_action(0,0)
        print(mcts.value(test))

    z = state.returns()[0]  # black perspective
    return [(sym, obs, pi, z) for sym, obs, pi in data]

def all_terminal(state):
    for a in state.legal_actions():
        test = state.clone()
        test.apply_action(a)
        if test.is_terminal():
            pass
            print("action", a, "line", test.winning_line())
        else:
            print("action", a, "nope")
            return False
    return True

def aggregate_by_symmetry(samples):
    agg = {}

    for sym, obs, pi, z in samples:
        if sym not in agg:
            agg[sym] = {
                "obs": obs,
                "pi": np.copy(pi),
                "z": z,
                "count": 1,
            }
        else:
            agg[sym]["pi"] += pi
            agg[sym]["count"] += 1

    result = []
    for v in agg.values():
        pi_avg = v["pi"] / v["count"]
        result.append((v["obs"], pi_avg, v["z"]))

    return result

# ----------------------------
# Training step
# ----------------------------

def train_step(batch):
    obs, pi, z = zip(*batch)

    obs = torch.from_numpy(np.stack(obs)).to(DEVICE)
    pi = torch.tensor(np.stack(pi), device=DEVICE)
    z = torch.full( (len(obs),), z[0], device=DEVICE,
      dtype=torch.float32, )

    logits, values = net(obs)

    policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, z)

    loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
    )


if __name__ == '__main__':
    # ----------------------------
    # Main loop
    # ----------------------------

    first_move_counts = np.zeros(NUM_ACTIONS, dtype=np.int32)
    bw = 0
    print(f"Starting game count {game_count}")
    while True:
    #for g in range(1, NUM_TRAIN_GAMES + 1):
        game_count += 1
        raw = play_selfplay_game()
        last = raw[-1]
        br = last[3]
        if br == 1:
            bw += 1
        batch = aggregate_by_symmetry(raw)

        # record first move statistics
        first_obs, first_pi, _ = batch[0]
        first_action = np.argmax(first_pi)
        first_move_counts[first_action] += 1

        loss, pl, vl = train_step(batch)

        if game_count % PRINT_EVERY == 0:
            total = first_move_counts.sum()
            top_moves = np.argsort(first_move_counts)[-5:][::-1]

            print(
                f"Game {game_count:4d} | "
                f"loss={loss:.4f} "
                f"(policy={pl:.4f}, value={vl:.4f}) "
                f"bw={bw}"
            , flush=True)

            #print("  Top first moves:")
            #for a in top_moves:
            #    print(f"    action {a:2d}: {first_move_counts[a] / total:.2%}")

            #first_move_counts[:] = 0
            pt_name = f"misery_{game_count}.pt"
            if game_count % 500 == 0:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': game_count,
                }, pt_name)

