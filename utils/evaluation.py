import numpy as np
import torch
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from utils.env import reset_environment

def eval_model(player, GameClass, env_range, num_tests, horizon, target_hitrate):
    if getattr(player, "model", None):
        player.model.eval()

    pcts = []

    for _ in range(num_tests):
        game = reset_environment(player, GameClass, env_range)

        start_balance = player.balance
        done = False
        hands = 0

        while not done and hands < horizon:
            game.start_round()
            if game.check_deal():
                game.play_game()

            hands += 1
            done = (
                player.balance >= player.max_balance
                or player.balance <= game.min_bet
                or not game.in_game(player.name)
            )

        roi = (player.balance - start_balance) / start_balance
        pcts.append(roi)

    pcts_tensor = torch.tensor(pcts, dtype=torch.float32)
    std, mean = torch.std_mean(pcts_tensor)
    winrate = torch.mean((pcts_tensor > target_hitrate).float())

    return std, mean, winrate


class EvaluationSession:
    """
    Incremental evaluation session

    - GUI or CLI can call .step() repeatedly.
    - ROI values live in self.pcts
    - Summary stats are available via .summary or .std / .mean / .winrate
    """
    def __init__(
        self,
        player,
        GameClass,
        env_range,
        num_tests,
        horizon,
        target_hitrate,
        update_every=10,
    ):
        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range
        self.num_tests = num_tests
        self.horizon = horizon
        self.target_hitrate = target_hitrate
        self.update_every = update_every

        if getattr(player, "model", None):
            player.model.eval()

        self.pcts = []  # list of ROI values
        self.i = 0      # number of tests completed

        # cached stats
        self._std = 0.0
        self._mean = 0.0
        self._winrate = 0.0

    def _run_one_test(self):
        player = self.player
        game = reset_environment(player, self.GameClass, self.env_range)

        start_balance = player.balance
        done = False
        hands = 0

        while not done and hands < self.horizon:
            game.start_round()
            if game.check_deal():
                game.play_game()

            hands += 1
            done = (
                player.balance >= player.max_balance
                or player.balance <= game.min_bet
                or not game.in_game(player.name)
            )

        roi = (player.balance - start_balance) / start_balance
        self.pcts.append(roi)

    def _update_stats(self):
        if not self.pcts:
            self._std = 0.0
            self._mean = 0.0
            self._winrate = 0.0
            return

        data = np.asarray(self.pcts)
        self._std = float(np.std(data))
        self._mean = float(np.mean(data))
        self._winrate = float(np.mean(data > self.target_hitrate))

    def step(self, tests_per_step: int = 1) -> bool:
        remaining = self.num_tests - self.i
        if remaining <= 0:
            return True

        to_run = min(tests_per_step, remaining)
        for _ in range(to_run):
            self._run_one_test()
            self.i += 1

        self._update_stats()
        return self.i >= self.num_tests

    @property
    def n(self) -> int:
        """Number of tests completed so far."""
        return self.i

    @property
    def std(self) -> float:
        return self._std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def winrate(self) -> float:
        return self._winrate

    @property
    def summary(self) -> dict:
        return {
            "n": self.n,
            "num_tests": self.num_tests,
            "std": self._std,
            "mean": self._mean,
            "winrate": self._winrate,
        }


def plot_eval_distribution(ax, pcts, num_tests, mode: str = "dist"):
    """
    Draw ROI histogram / CDF / survival on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    pcts : Sequence[float]
        List/array of ROI values.
    num_tests : int
        Total number of tests planned (for labeling).
    mode : {"dist", "cdf", "survival"}
        Which view to draw.
    """
    ax.clear()

    data = np.asarray(pcts)
    n = len(data)

    if n == 0:
        ax.set_title("No evaluation data yet")
        ax.set_xlabel("ROI")
        ax.set_ylabel("Density")
        return

    mean = data.mean()
    median = np.median(data)
    std = np.std(data)

    mode_val = np.nan

    if mode == "dist":
        ax.hist(data, bins=40, density=True, alpha=0.4, label="ROI Histogram")

        if n > 5 and np.std(data) > 1e-6:
            try:
                kde = gaussian_kde(data)
                xs = np.linspace(data.min(), data.max(), 500)
                ys = kde(xs)
                mode_val = xs[np.argmax(ys)]
                ax.plot(xs, ys, lw=2, label="KDE")
            except Exception:
                # KDE can fail due to numerical degeneracy; ignore gracefully.
                pass

        ax.set_ylabel("Density")
        ax.set_title(f"ROI Distribution ({n}/{num_tests})")

    else:
        sorted_data = np.sort(data)
        F = np.arange(1, n + 1) / n

        if mode == "cdf":
            ax.plot(sorted_data, F, lw=2, label="F(x)")
            ax.set_ylabel("F(x)")
            ax.set_title(f"Empirical CDF ({n}/{num_tests})")
        elif mode == "survival":
            ax.plot(sorted_data, 1 - F, lw=2, label="1 - F(x)")
            ax.set_ylabel("1 - F(x)")
            ax.set_title(f"Survival Function ({n}/{num_tests})")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Shared decorations
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean {mean:.3f}")
    ax.axvline(median, color="green", linestyle="--", label=f"Median {median:.3f}")

    if not np.isnan(mode_val):
        ax.axvline(mode_val, color="purple", linestyle="--", label=f"Mode {mode_val:.3f}")

    ax.set_xlabel("ROI")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="none", label=""))
    labels.append(f"STD={std:.3f}")
    ax.legend(handles, labels)
    ax.grid(alpha=0.3)