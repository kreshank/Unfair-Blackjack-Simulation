from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibWidget(QWidget):
    """
    Simple reusable Matplotlib widget.

    Methods:
    --------
    clear():
        Clear the current figure and redraw.

    get_axis():
        Clear the figure, create a single Axes (1x1), and return it.

    redraw():
        Redraw the canvas (use after you've finished plotting).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create a Matplotlib Figure and embed it in a Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)

    def clear(self):
        """Clear the entire figure."""
        self.figure.clear()
        self.canvas.draw()

    def get_axis(self):
        """
        Clear the figure and create a single Axes.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        return ax

    def redraw(self):
        """Redraw canvas"""
        self.canvas.draw()