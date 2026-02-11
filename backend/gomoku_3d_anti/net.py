import torch
import torch.nn as nn
import torch.nn.functional as F


class ValuePolicyNet(nn.Module):
    def __init__(self, input_size: int, num_actions: int, hidden_size: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Policy head
        self.policy = nn.Linear(hidden_size, num_actions)

        # Value head
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: Tensor of shape (B, input_size)
        Returns:
          policy_logits: (B, num_actions)
          value: (B,) in [-1, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy(x)
        value = torch.tanh(self.value(x)).squeeze(-1)

        return policy_logits, value

if __name__ == '__main__':
    import pyspiel
    game = pyspiel.load_game( "gomoku",
     {"size": 3, "connect": 3, "anti": True, "dims": 3})
    net = ValuePolicyNet(
    input_size=3 * 27,
    num_actions=game.num_distinct_actions(),
    )
    state = game.new_initial_state()
    obs = torch.tensor(state.observation_tensor(), dtype=torch.float32).unsqueeze(0)

    logits, value = net(obs)

    print(logits.shape)  # (1, 27)
    print(value.shape)   # (1,)
    print(value.item())  # should be in [-1, 1]

