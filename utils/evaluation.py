import numpy as np
import torch
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from utils.env import reset_environment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from typing import Dict, Tuple

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

_STRATEGY_CACHE = {}
ActionGrid = np.ndarray  # alias for readability

def compute_strategy_tables(player, deck_count: int, true_count: float):
    """
    Compute or retrieve from cache the strategy tables for the given
    (deck_count, true_count), snapping true_count to 0.5 increments.
    """

    # --- snap true count to nearest 0.5 ---
    tc = round(true_count * 2) / 2.0     # e.g. 0.73 → 0.5, 3.1 → 3.0

    key = (deck_count, tc)
    if key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[key]

    tables, meta = _compute_strategy_tables_uncached(player, deck_count, tc)

    # Store
    _STRATEGY_CACHE[key] = (tables, meta)
    return tables, meta

def _compute_strategy_tables_uncached(
    player,
    deck_count: int,
    true_count: int,
) -> Tuple[Dict[str, ActionGrid], Dict[str, dict]]:
    """
    Compute three strategy tables for visualization:

      - hard: hard totals (no usable Ace)
      - soft: soft totals (hands with a usable Ace)
      - pairs: pair splitting decisions

    This is still a STUB; replace the internals with calls into your real model
    when you're ready.

    Returns
    -------
    tables : dict[str, ActionGrid]
        {
          "hard": 2D array [n_hard_totals, n_dealer_upcards],
          "soft": 2D array [n_soft_totals, n_dealer_upcards],
          "pairs": 2D array [n_pair_ranks, n_dealer_upcards],
        }

    meta : dict[str, dict]
        Per-table metadata, plus global info:

        {
          "hard": {
              "row_labels": [...],
              "col_labels": [...],
              "action_labels": [...],
              "action_chars": [...],
              "title": "Hard Totals",
          },
          "soft": {...},
          "pairs": {...},
          "deck_count": deck_count,
          "true_count": true_count,
        }
    """

    # ----- General settings -----
    # Actions: 0=Stand, 1=Hit, 2=Double, 3=Split
    action_labels = ["Stand", "Hit", "Double", "Split"]
    action_chars = ["S", "H", "D", "P"]  # single-character codes

    # Dealer upcards: 2..11 (11 = Ace)
    dealer_upcards = np.arange(2, 12)  # 2..11
    dealer_labels = [str(v) if v <= 10 else "A" for v in dealer_upcards]

    # ----- Hard totals -----
    # Example: 5..21
    hard_totals = np.arange(5, 22)  # 5..21
    hard_grid = np.zeros((len(hard_totals), len(dealer_upcards)), dtype=int)

    for i, pt in enumerate(hard_totals):
        for j, du in enumerate(dealer_upcards):
            # Stub strategy
            if pt < 12:
                hard_grid[i, j] = 1  # Hit
            elif pt >= 17:
                hard_grid[i, j] = 0  # Stand
            else:
                # Intermediate region – vary with true_count and dealer card
                if du >= 7:
                    hard_grid[i, j] = 1  # Hit vs high dealer card
                else:
                    if true_count >= 2:
                        hard_grid[i, j] = 2  # Double at high count
                    else:
                        hard_grid[i, j] = 0  # Stand
    hard_meta = {
        "row_labels": [str(v) for v in hard_totals],
        "col_labels": dealer_labels,
        "action_labels": action_labels,
        "action_chars": action_chars,
        "title": "Hard Totals",
    }

    # ----- Soft totals -----
    # Represent soft totals as "A2".."A9" (soft 13..20)
    soft_totals = np.arange(13, 21)  # 13..20
    soft_labels = [f"A{v - 11}" for v in soft_totals]  # 13->A2, 14->A3, ...
    soft_grid = np.zeros((len(soft_totals), len(dealer_upcards)), dtype=int)

    for i, total in enumerate(soft_totals):
        for j, du in enumerate(dealer_upcards):
            effective_total = total
            if effective_total <= 17:
                # Aggressive with soft hands
                if true_count >= 1 and du in (4, 5, 6):
                    soft_grid[i, j] = 2  # Double vs 4–6 at higher counts
                else:
                    soft_grid[i, j] = 1  # Hit
            else:
                soft_grid[i, j] = 0  # Stand on soft 18+ (stub)
    soft_meta = {
        "row_labels": soft_labels,
        "col_labels": dealer_labels,
        "action_labels": action_labels,
        "action_chars": action_chars,
        "title": "Soft Totals",
    }

    # ----- Pair splitting -----
    # Pair ranks: 2..A
    pair_ranks = list(range(2, 11)) + ["A"]
    pair_labels = [str(r) if r != "A" else "A" for r in pair_ranks]
    pairs_grid = np.zeros((len(pair_ranks), len(dealer_upcards)), dtype=int)

    for i, rank in enumerate(pair_ranks):
        for j, du in enumerate(dealer_upcards):
            # Stub pair strategy
            if rank in (8, "A"):
                pairs_grid[i, j] = 3  # Split
            elif rank in (2, 3, 7) and du in (2, 3, 4, 5, 6, 7):
                pairs_grid[i, j] = 3  # Split small pairs vs low upcards
            elif rank == 9 and du not in (7, 10, 11):
                pairs_grid[i, j] = 3
            else:
                if true_count >= 3 and du in (3, 4, 5, 6):
                    pairs_grid[i, j] = 2  # Double at high counts
                else:
                    pairs_grid[i, j] = 0  # Stand (stub)
    pairs_meta = {
        "row_labels": pair_labels,
        "col_labels": dealer_labels,
        "action_labels": action_labels,
        "action_chars": action_chars,
        "title": "Pair Splitting",
    }

    tables = {
        "hard": hard_grid,
        "soft": soft_grid,
        "pairs": pairs_grid,
    }

    meta = {
        "hard": hard_meta,
        "soft": soft_meta,
        "pairs": pairs_meta,
        "deck_count": deck_count,
        "true_count": true_count,
        # global action info so the extra panel can build the key
        "action_labels": action_labels,
        "action_chars": action_chars,
    }

    return tables, meta


def _plot_single_table(
    ax: Axes,
    grid: ActionGrid,
    row_labels,
    col_labels,
    action_labels,
    action_chars,
    title: str,
):
    """
    Plot a single strategy table in an Excel-like style:

    - Discrete background colors per action
    - Grid lines
    - A single character (S/H/D/P) inside each cell
    """
    n_actions = len(action_labels)
    n_rows, n_cols = grid.shape

    # Discrete colormap
    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i) for i in range(n_actions)]
    cmap = ListedColormap(colors[:n_actions])

    # Draw colored background
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=n_actions - 1,
        extent=[
            -0.5,
            n_cols - 0.5,
            -0.5,
            n_rows - 0.5,
        ],
    )

    # Axis ticks/labels like a table
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Dealer Upcard")
    ax.set_ylabel("Player State")
    ax.set_title(title)

    # Draw Excel-like grid lines (on minor ticks)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Put a single character in each cell
    for i in range(n_rows):
        for j in range(n_cols):
            action_idx = int(grid[i, j])
            char = action_chars[action_idx] if 0 <= action_idx < len(action_chars) else "?"
            ax.text(
                j,
                i,
                char,
                ha="center",
                va="center",
                fontsize=9,
                color="black",  # tweak if some backgrounds are dark
            )


def plot_strategy_tables(fig, tables: Dict[str, ActionGrid], meta: Dict[str, dict]):
    """
    Plot multiple strategy tables in one figure, stacked vertically:

      1. Hard totals
      2. Soft totals
      3. Pair splitting
      4. Key / summary

    The hard totals panel is given extra vertical space via height_ratios.
    """
    import matplotlib.gridspec as gridspec

    fig.clear()

    # Layout: 4 rows, 1 column.
    # Give hard totals more vertical real estate so it's not squished.
    gs = gridspec.GridSpec(
        4,
        1,
        height_ratios=[3, 2, 2, 1],  # hard, soft, pairs, key
        hspace=0.35,
        figure=fig,
    )

    ax_hard = fig.add_subplot(gs[0, 0])
    ax_soft = fig.add_subplot(gs[1, 0])
    ax_pairs = fig.add_subplot(gs[2, 0])
    ax_key = fig.add_subplot(gs[3, 0])

    # Pull global action info for key
    action_labels = meta["action_labels"]
    action_chars = meta["action_chars"]
    n_actions = len(action_labels)

    # Common colormap for tables (so colors match across panels)
    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i) for i in range(n_actions)]
    cmap = ListedColormap(colors[:n_actions])

    # Hard
    hard_grid = tables["hard"]
    hard_meta = meta["hard"]
    _plot_single_table(
        ax_hard,
        hard_grid,
        hard_meta["row_labels"],
        hard_meta["col_labels"],
        hard_meta["action_labels"],
        hard_meta["action_chars"],
        hard_meta["title"],
    )

    # Soft
    soft_grid = tables["soft"]
    soft_meta = meta["soft"]
    _plot_single_table(
        ax_soft,
        soft_grid,
        soft_meta["row_labels"],
        soft_meta["col_labels"],
        soft_meta["action_labels"],
        soft_meta["action_chars"],
        soft_meta["title"],
    )

    # Pairs
    pairs_grid = tables["pairs"]
    pairs_meta = meta["pairs"]
    _plot_single_table(
        ax_pairs,
        pairs_grid,
        pairs_meta["row_labels"],
        pairs_meta["col_labels"],
        pairs_meta["action_labels"],
        pairs_meta["action_chars"],
        pairs_meta["title"],
    )

    # Key / summary panel
    ax_key.set_axis_off()

    deck_count = meta.get("deck_count", None)
    tc = meta.get("true_count", None)

    title_lines = ["Key / Summary"]
    if deck_count is not None:
        title_lines.append(f"Decks: {deck_count}")
    if tc is not None:
        title_lines.append(f"True Count: {tc}")

    ax_key.text(
        0.5,
        0.9,
        "\n".join(title_lines),
        ha="center",
        va="top",
        fontsize=11,
        transform=ax_key.transAxes,
    )

    # Draw key as colored boxes w/ single character and label
    from matplotlib.patches import Rectangle

    y_start = 0.7
    y_step = 0.12
    box_width = 0.15
    box_height = 0.08

    for idx, (char, label) in enumerate(zip(action_chars, action_labels)):
        y = y_start - idx * y_step

        # Colored box
        box = Rectangle(
            (0.1, y - box_height / 2),
            box_width,
            box_height,
            transform=ax_key.transAxes,
            facecolor=cmap(idx),
            edgecolor="black",
        )
        ax_key.add_patch(box)

        # Character in the box
        ax_key.text(
            0.1 + box_width / 2,
            y,
            char,
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_key.transAxes,
        )

        # Label to the right
        ax_key.text(
            0.1 + box_width + 0.05,
            y,
            f"= {label}",
            ha="left",
            va="center",
            fontsize=10,
            transform=ax_key.transAxes,
        )

    fig.tight_layout()