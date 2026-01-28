import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from player.default import DefaultPlayer

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
            done = (player.balance >= player.max_balance 
                    or player.balance <= game.min_bet
                    or not game.in_game(player.name))
        pcts.append((player.balance - start_balance) / start_balance)
    
    winrate = torch.sum(torch.tensor(pcts) > target_hitrate) / num_tests
    return torch.std_mean(torch.tensor(pcts)) + (winrate,)

def visualize_stats_realtime(
    player,
    GameClass,
    env_range,
    num_tests,
    horizon,
    target_hitrate,
    update_every=10
):
    if getattr(player, "model", None):
        player.model.eval()

    pcts = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    current_view = {"mode": "dist"}  # dist | cdf | survival
    
    matplotlib.rcParams["keymap.save"] = []

    def on_key(event):
        if event.key == "d":
            current_view["mode"] = "dist"
            redraw()
        elif event.key == "c":
            current_view["mode"] = "cdf"
            redraw()
        elif event.key == "s":
            current_view["mode"] = "survival"
            redraw()


    fig.canvas.mpl_connect("key_press_event", on_key)

    def redraw():
        ax.clear()

        data = np.asarray(pcts)
        n = len(data)

        if n == 0:
            return

        mean = data.mean()
        median = np.median(data)
        std = np.std(data)

        if current_view["mode"] == "dist":
            ax.hist(data, bins=40, density=True, alpha=0.4, label="ROI Histogram")
            mode = np.nan

            if n > 5 and np.std(data) > 1e-6:
                try:
                    kde = gaussian_kde(data)
                    xs = np.linspace(data.min(), data.max(), 500)
                    ys = kde(xs)
                    mode = xs[np.argmax(ys)]
                    ax.plot(xs, ys, lw=2, label="KDE")
                except Exception:
                    # KDE failed due to numerical degeneracy
                    pass

            ax.set_ylabel("Density")
            ax.set_title(f"ROI Distribution ({n}/{num_tests})")

        else:
            sorted_data = np.sort(data)
            F = np.arange(1, n + 1) / n

            if current_view["mode"] == "cdf":
                ax.plot(sorted_data, F, lw=2, label="F(x)")
                ax.set_ylabel("F(x)")
                ax.set_title(f"Empirical CDF ({n}/{num_tests})")
            else:
                ax.plot(sorted_data, 1 - F, lw=2, label="1 − F(x)")
                ax.set_ylabel("1 − F(x)")
                ax.set_title(f"Survival Function ({n}/{num_tests})")

            mode = np.nan

        # Shared decorations
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean {mean:.3f}")
        ax.axvline(median, color="green", linestyle="--", label=f"Median {median:.3f}")

        if not np.isnan(mode):
            ax.axvline(mode, color="purple", linestyle="--", label=f"Mode {mode:.3f}")

        ax.set_xlabel("ROI")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='none', label=""))
        labels.append(f"STD={std:.3f}")
        ax.legend(handles, labels)
        ax.grid(alpha=0.3)

        plt.pause(0.001)

    for i in range(num_tests):
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

        if (i + 1) % update_every == 0:
            redraw()

    plt.ioff()
    redraw()
    plt.show()

    pcts = np.asarray(pcts)
    winrate = np.mean(pcts > target_hitrate)

    return torch.std(torch.tensor(pcts)), torch.mean(torch.tensor(pcts)), winrate
