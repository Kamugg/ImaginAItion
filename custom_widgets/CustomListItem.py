from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel


class CustomListItem(QWidget):
    """
    Custom ListItem widget that holds an image and a string of text, useful for query result visualization.

    Args:
        image_path (Path): Path to the retrieved image
        text (str): Text to be shown in the widget
        retrieved_by (str): String that can be either 'SEMANTIC', 'CONTENT' or 'INTERSECTION' and describes what policy
        retrieved the image.
    """

    def __init__(self, image_path: Path, text: str, retrieved_by: str):
        super().__init__()
        self.size = 0
        # Horizontal layout
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)  # Imposta margini a 0
        self.layout.setSpacing(0)  # Imposta spaziatura a 0

        # Create image
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(400, 400)  # Imposta dimensione fissa per l'icona
        pixmap = QPixmap(image_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.size = pixmap.width()
        self.icon_label.setPixmap(pixmap)

        # Text
        self.text_label = QLabel(f'\t{text}')
        self.size += self.text_label.width()

        # Add image and text to the layout
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_label)

        # Set widget color relative to the policy that retrieved the image
        if retrieved_by == 'CONTENT':
            self.setStyleSheet("background-color: rgba(255, 0, 0, 25)")
        elif retrieved_by == 'SEMANTIC':
            self.setStyleSheet("background-color: rgba(0, 255, 0, 25)")
        else:
            self.setStyleSheet("background-color: rgba(0, 0, 255, 25)")

        # Set layout
        self.setLayout(self.layout)

    def tot_width(self):
        return self.sizeHint().width()
