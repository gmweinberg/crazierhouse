
import json
import subprocess


class AlphaZeroEngine:
    def __init__(self, executable, game_string, model_path,
                 checkpoint=0, simulations=100):
        self.process = subprocess.Popen(
            [
                executable,
                f"--game={game_string}",
                f"--az_path={model_path}",
                f"--az_checkpoint={checkpoint}",
                f"--max_simulations={simulations}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def evaluate(self, state):
        if self.process.poll() is not None:
            raise RuntimeError("AlphaZero engine has exited")

        fen = str(state)

        self.process.stdin.write(fen + "\n")
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("No response from AlphaZero engine")

        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])

        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()

