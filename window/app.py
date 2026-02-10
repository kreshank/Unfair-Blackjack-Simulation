import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
)

from window.pages.training_page import TrainingPage
from window.pages.eval_page import EvalPage
from window.pages.analysis_page import AnalysisPage
from utils.evaluation import compute_strategy_tables


class MainWindow(QMainWindow):
    def __init__(self, player, GameClass, env_range):
        """
        Parameters
        ----------
        player : Player
            Your blackjack player object (with .name, .balance, .max_balance, etc.)
        GameClass : type
            Your blackjack game class, e.g. FairBlackjack.
        env_range : dict
            The env_range dict you already use in training/eval.
        """
        super().__init__()

        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range

        self.setWindowTitle("RL Strategy Lab")
        self.resize(1200, 800)

        # --- Central layout --- #
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Top navigation bar --- #
        nav_layout = QHBoxLayout()

        self.btn_training = QPushButton("Training")
        self.btn_eval = QPushButton("Evaluation")
        self.btn_analysis = QPushButton("Analysis")

        nav_layout.addWidget(self.btn_training)
        nav_layout.addWidget(self.btn_eval)
        nav_layout.addWidget(self.btn_analysis)
        nav_layout.addStretch()

        main_layout.addLayout(nav_layout)

        # --- Stacked pages --- #
        self.stacked = QStackedWidget()
        main_layout.addWidget(self.stacked)

        # Create pages, injecting dependencies
        self.training_page = TrainingPage(parent=self)
        self.eval_page = EvalPage(
            parent=self,
            player=self.player,
            GameClass=self.GameClass,
            env_range=self.env_range,
        )
        self.analysis_page = AnalysisPage(
            parent=self,
            player=self.player,
            GameClass=self.GameClass,
            env_range=self.env_range,
        )

        # Add to stacked widget
        self.stacked.addWidget(self.training_page)   # index 0
        self.stacked.addWidget(self.eval_page)       # index 1
        self.stacked.addWidget(self.analysis_page)   # index 2

        # Wire up navigation
        self.btn_training.clicked.connect(
            lambda: self.stacked.setCurrentWidget(self.training_page)
        )
        self.btn_eval.clicked.connect(
            lambda: self.stacked.setCurrentWidget(self.eval_page)
        )
        self.btn_analysis.clicked.connect(
            lambda: self.stacked.setCurrentWidget(self.analysis_page)
        )

def preload_strategy_cache(player, GameClass, env_range):
    deck_values = sorted(set(env_range["deck_count"])) if env_range else [6]
    true_counts = [tc / 2 for tc in range(-10, 11)]    # -5, -4.5, ..., 4.5, 5

    for deck in deck_values:
        for tc in true_counts:
            compute_strategy_tables(player, deck, tc)


def launch_window(player=None, GameClass=None, env_range=None):
    if player is None or GameClass is None or env_range is None:
        raise ValueError(
            "You must provide player, GameClass, and env_range to launch_window."
        )
    app = QApplication(sys.argv)
    win = MainWindow(player, GameClass, env_range)
    win.show()
    preload_strategy_cache(player, GameClass, env_range)
    return app.exec_()
if __name__ == "__main__":
    launch_window(player=DefaultPlayer("TestPlayer"), GameClass=None, env_range={})