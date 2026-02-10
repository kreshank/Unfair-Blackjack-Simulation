# window/pages/analysis_page.py

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QScrollArea,
)
from PyQt5.QtCore import Qt

from window.widgets.mpl_widget import MatplotlibWidget
from utils.evaluation import compute_strategy_tables, plot_strategy_tables


class AnalysisPage(QWidget):
    """
    Analyze the model's strategy with multiple tables:

      - Hard totals strategy
      - Soft totals strategy
      - Pair splitting strategy
      - Extra summary panel

    Controls:
      - True count (slider)
      - Number of decks (combo box)
    """

    def __init__(self, parent=None, player=None, GameClass=None, env_range=None):
        super().__init__(parent)

        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range

        self.tables = None
        self.meta = None

        main_layout = QVBoxLayout(self)

        # ==========================
        # Controls
        # ==========================
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)

        # Row 1: info
        info_row = QHBoxLayout()
        self.player_label = QLabel(f"Player: {getattr(self.player, 'name', 'None')}")
        self.game_label = QLabel(
            f"GameClass: {getattr(self.GameClass, '__name__', 'None')}"
        )
        info_row.addWidget(self.player_label)
        info_row.addWidget(self.game_label)
        info_row.addStretch()
        controls_layout.addLayout(info_row)

        # Row 2: deck count
        deck_row = QHBoxLayout()
        deck_row.addWidget(QLabel("Decks:"))

        self.deck_select = QComboBox()
        deck_values = None
        if self.env_range and "deck_count" in self.env_range:
            deck_values = sorted(set(self.env_range["deck_count"]))
        else:
            deck_values = [1, 2, 4, 6, 8]

        for d in deck_values:
            self.deck_select.addItem(f"{d} decks", userData=int(d))

        deck_row.addWidget(self.deck_select)
        deck_row.addStretch()
        controls_layout.addLayout(deck_row)

        # Row 3: true count slider
        tc_row = QHBoxLayout()
        tc_row.addWidget(QLabel("True Count:"))

        self.tc_slider = QSlider(Qt.Horizontal)
        self.tc_slider.setMinimum(-10)
        self.tc_slider.setMaximum(10)
        self.tc_slider.setValue(0)
        self.tc_slider.setTickPosition(QSlider.TicksBelow)
        self.tc_slider.setTickInterval(1)

        self.tc_label = QLabel("0")

        tc_row.addWidget(self.tc_slider)
        tc_row.addWidget(self.tc_label)
        tc_row.addStretch()
        controls_layout.addLayout(tc_row)

        # Row 4: refresh button (optional; we also auto-update on change)
        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Update Strategy")
        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch()
        controls_layout.addLayout(btn_row)

        # ==========================
        # Plot area
        # ==========================
        self.mpl = MatplotlibWidget()

        # Make the matplotlib area tall enough to be scrollable
        self.mpl.setMinimumHeight(1400)  # tweak as you like

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.mpl)

        main_layout.addWidget(scroll)

        # ==========================
        # Connections
        # ==========================
        self.btn_refresh.clicked.connect(self.update_strategy_plot)
        self.tc_slider.valueChanged.connect(self.on_true_count_changed)
        self.deck_select.currentIndexChanged.connect(self.on_deck_changed)

        # Initial draw
        self.update_strategy_plot()

    # ------------------------------------------------------------
    # Optional helper to inject target later
    # ------------------------------------------------------------
    def set_analysis_target(self, player, GameClass, env_range):
        self.player = player
        self.GameClass = GameClass
        self.env_range = env_range

        self.player_label.setText(f"Player: {getattr(self.player, 'name', 'None')}")
        self.game_label.setText(
            f"GameClass: {getattr(self.GameClass, '__name__', 'None')}"
        )

        if self.env_range and "deck_count" in self.env_range:
            deck_values = sorted(set(self.env_range["deck_count"]))
            self.deck_select.clear()
            for d in deck_values:
                self.deck_select.addItem(f"{d} decks", userData=int(d))

        self.update_strategy_plot()

    # ------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------
    def on_true_count_changed(self, value: int):
        self.tc_label.setText(str(value))
        # Auto-update; if too slow, you can throttle or only update on release
        self.update_strategy_plot()

    def on_deck_changed(self, index: int):
        self.update_strategy_plot()

    # ------------------------------------------------------------
    # Core plotting logic
    # ------------------------------------------------------------
    def update_strategy_plot(self):
        """
        Compute and plot the model's strategy tables
        for current (deck_count, true_count).
        """
        if self.player is None:
            fig = self.mpl.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No player set for analysis.", ha="center", va="center")
            ax.set_axis_off()
            self.mpl.redraw()
            return

        deck_count = self.deck_select.currentData()
        if deck_count is None:
            deck_count = 6

        raw_tc = self.tc_slider.value()
        true_count = round(raw_tc * 2) / 2.0
        self.tc_label.setText(f"{true_count:.1f}")

        tables, meta = compute_strategy_tables(
            player=self.player,
            deck_count=deck_count,
            true_count=true_count,
        )
        self.tables = tables
        self.meta = meta

        # Plot all of them in one figure
        fig = self.mpl.figure
        plot_strategy_tables(fig, tables, meta)
        self.mpl.redraw()
