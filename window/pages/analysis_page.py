from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class AnalysisPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Analysis page (coming soon)"))
        layout.addStretch()