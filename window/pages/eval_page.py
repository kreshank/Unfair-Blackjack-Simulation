from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QLineEdit,
)
from PyQt5.QtCore import QTimer

from window.widgets.mpl_widget import MatplotlibWidget
from utils.evaluation import EvaluationSession, plot_eval_distribution


class EvalPage(QWidget):
    """
    Evaluation page that runs the same logic as visualize_stats_realtime,
    but inside the Qt app using EvaluationSession + plot_eval_distribution.
    """

    def __init__(self, parent=None, player=None, GameClass=None, env_range=None):
        super().__init__(parent)

        # These should match the arguments you use when calling eval_model / visualize_stats_realtime
        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range

        self.session: EvaluationSession | None = None

        # --- Main layout --- #
        main_layout = QVBoxLayout(self)

        # ==========================
        # Controls panel
        # ==========================
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)

        # Row 1: information about what we're evaluating
        info_row = QHBoxLayout()
        self.player_label = QLabel(
            f"Player: {getattr(self.player, 'name', 'None')}"
        )
        self.game_label = QLabel(
            f"GameClass: {getattr(self.GameClass, '__name__', 'None')}"
        )

        info_row.addWidget(self.player_label)
        info_row.addWidget(self.game_label)
        info_row.addStretch()
        controls_layout.addLayout(info_row)

        # Row 2: evaluation parameters (num_tests, horizon, target ROI)
        params_row = QHBoxLayout()

        params_row.addWidget(QLabel("Num tests:"))
        self.num_tests_spin = QSpinBox()
        self.num_tests_spin.setRange(1, 1_000_000)
        self.num_tests_spin.setValue(500)
        params_row.addWidget(self.num_tests_spin)

        params_row.addWidget(QLabel("Horizon (hands cap):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 1_000_000)
        self.horizon_spin.setValue(50)
        params_row.addWidget(self.horizon_spin)

        params_row.addWidget(QLabel("Target ROI (hit-rate threshold):"))
        self.target_roi_edit = QLineEdit("0.0")  # e.g. hit-rate for ROI > 0
        params_row.addWidget(self.target_roi_edit)

        params_row.addStretch()
        controls_layout.addLayout(params_row)

        # Row 3: view mode (distribution / CDF / survival)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))

        self.view_mode = QComboBox()
        self.view_mode.addItem("Distribution", userData="dist")
        self.view_mode.addItem("CDF", userData="cdf")
        self.view_mode.addItem("Survival", userData="survival")

        view_row.addWidget(self.view_mode)
        view_row.addStretch()
        controls_layout.addLayout(view_row)

        # Row 4: buttons (Start / Pause / Reset)
        buttons_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Evaluation")
        self.btn_pause = QPushButton("Pause")
        self.btn_reset = QPushButton("Reset")

        buttons_row.addWidget(self.btn_start)
        buttons_row.addWidget(self.btn_pause)
        buttons_row.addWidget(self.btn_reset)
        buttons_row.addStretch()
        controls_layout.addLayout(buttons_row)

        # Row 5: summary label
        self.summary_label = QLabel("No evaluation run yet.")
        controls_layout.addWidget(self.summary_label)

        # ==========================
        # Plot area
        # ==========================
        self.mpl = MatplotlibWidget()
        main_layout.addWidget(self.mpl)

        # ==========================
        # Timer (drives EvaluationSession.step)
        # ==========================
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # ms; tune as you like
        self.timer.timeout.connect(self.on_eval_tick)

        # ==========================
        # Signal connections
        # ==========================
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)
        self.view_mode.currentIndexChanged.connect(self.redraw_plot)

    # ------------------------------------------------------------------
    # Helper: (optional) let app set the evaluation target later
    # ------------------------------------------------------------------
    def set_evaluation_target(self, player, GameClass, env_range):
        """
        Optional helper if you prefer to construct EvalPage() with no args
        and inject these later from MainWindow.
        """
        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range

        self.player_label.setText(f"Player: {getattr(self.player, 'name', 'None')}")
        self.game_label.setText(
            f"GameClass: {getattr(self.GameClass, '__name__', 'None')}"
        )

        # Reset existing session if any
        self.reset_session(clear_plot=True)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_start_clicked(self):
        """
        Start or resume evaluation.
        If no EvaluationSession exists, create one from the current params.
        """
        # Sanity checks
        if self.player is None or self.GameClass is None or self.env_range is None:
            self.summary_label.setText(
                "Eval target not set: player / GameClass / env_range is None."
            )
            return

        if self.session is None:
            num_tests = int(self.num_tests_spin.value())
            horizon = int(self.horizon_spin.value())
            try:
                target_hitrate = float(self.target_roi_edit.text())
            except ValueError:
                target_hitrate = 0.0

            self.session = EvaluationSession(
                player=self.player,
                GameClass=self.GameClass,
                env_range=self.env_range,
                num_tests=num_tests,
                horizon=horizon,
                target_hitrate=target_hitrate,
                update_every=10,  # not strictly needed in session, but kept for parity
            )

            # Clear plot for new run
            self.mpl.clear()
            self.summary_label.setText("Evaluation started...")

        if not self.timer.isActive():
            self.timer.start()

    def on_pause_clicked(self):
        if self.timer.isActive():
            self.timer.stop()
            self.summary_label.setText("Evaluation paused.")

    def on_reset_clicked(self):
        self.reset_session(clear_plot=True)
        self.summary_label.setText("Evaluation reset.")

    def reset_session(self, clear_plot: bool = False):
        self.timer.stop()
        self.session = None
        if clear_plot:
            self.mpl.clear()

    def on_eval_tick(self):
        """
        Called periodically by QTimer to advance evaluation and redraw plot.
        """
        if self.session is None:
            self.timer.stop()
            return

        # You can bump tests_per_step if each eval is cheap
        done = self.session.step(tests_per_step=1)

        # Plot using the same logic as visualize_stats_realtime's redraw()
        ax = self.mpl.get_axis()
        mode = self.view_mode.currentData() or "dist"
        plot_eval_distribution(
            ax,
            self.session.pcts,
            self.session.num_tests,
            mode=mode,
        )
        self.mpl.redraw()

        # Update summary label from session.summary
        s = self.session.summary
        self.summary_label.setText(
            f"Progress: {s['n']}/{s['num_tests']}  "
            f"Mean ROI: {s['mean']:.3f}  "
            f"STD: {s['std']:.3f}  "
            f"Hit-rate (ROI > target): {s['winrate']:.3f}"
        )

        if done:
            self.timer.stop()
            self.summary_label.setText(self.summary_label.text() + "  [DONE]")

    def redraw_plot(self):
        if self.session is None or not self.session.pcts:
            return

        ax = self.mpl.get_axis()
        mode = self.view_mode.currentData() or "dist"
        plot_eval_distribution(
            ax,
            self.session.pcts,
            self.session.num_tests,
            mode=mode,
        )
        self.mpl.redraw()