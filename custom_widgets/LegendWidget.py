from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel


class LegendWidget(QWidget):
    """
    Widget to be put in the StatusBar of the DatabaseQueryViewer window. It shows a legend that maps a color to the
    policy assigned to that color in the form a small colored square and a label like "Red": "Retrieved by content".

    Args:
        color (tuple): Triplet of the form (R, G, B).
        text (str): Policy assigned to that color.
    """

    def __init__(self, color: tuple, text: str):
        super().__init__()

        # Horizontal Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Creating colored square
        color_label = QLabel()
        color_label.setFixedSize(15, 15)
        color_label.setStyleSheet(f"background-color: rgb{color};")

        # Policy assigned to that color
        text_label = QLabel(text)

        # Add to layout
        layout.addWidget(color_label)
        layout.addWidget(text_label)

        self.setLayout(layout)
