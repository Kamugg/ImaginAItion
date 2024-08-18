from PyQt5.QtCore import QRect
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QListWidget, QListWidgetItem, QStatusBar, QLabel

from custom_widgets.CustomListItem import CustomListItem
from custom_widgets.LegendWidget import LegendWidget


class DataBaseQueryViewer(QMainWindow):
    """
    Window that shows the result of the query against th Multimedia DB.

    Args:
        query_result: A dict of the form idx: dict where the key is used to order the relevance of the retrieved images
        and dict is a dictionary that contains all the relevance of the retrieved image like its path, how it was
        retrieved, its content/semantic similarity with the queried image etc...
    """

    def __init__(self, query_result: dict):
        super().__init__()
        self.list_widget = QListWidget()
        self.setCentralWidget(self.list_widget)
        self.setWindowTitle('Query Viewer')
        self.setGeometry(QRect(300, 300, 0, 0))
        self.setWindowIcon(QIcon('res/assets/icons/query_icon.png'))
        max_size = 0
        retrieved = {'INTERSECTION': 0,
                     'SEMANTIC': 0,
                     'CONTENT': 0}
        for v in query_result.values():
            item_text = f'Semantic similarity: {round(v["semantic_similarity"], 2)}  |  Content similarity: {round(v["content_similarity"], 2)}  |  Path: {v["path"]}'
            item_widget = CustomListItem(v['path'], item_text, v['retrieved_by'])
            retrieved[v['retrieved_by']] += 1
            item = QListWidgetItem(self.list_widget)
            item.setSizeHint(item_widget.sizeHint())
            if item_widget.tot_width() > max_size:
                max_size = item_widget.tot_width()
            self.list_widget.setItemWidget(item, item_widget)
        self.setFixedWidth(max_size + 50)
        self.setFixedHeight(max_size)
        self.status = QStatusBar()
        total_images = QLabel(f'Retrieved images: {sum(retrieved.values())}')
        intersection_images = QLabel(f'Both by content and semantic: {retrieved["INTERSECTION"]}')
        semantic_images = QLabel(f'By semantic: {retrieved["SEMANTIC"]}')
        content_images = QLabel(f'By content: {retrieved["CONTENT"]}')
        inter_widget = LegendWidget((0, 0, 255), 'Content and Semantic')
        semantic_widget = LegendWidget((0, 255, 0), 'Semantic')
        content_widget = LegendWidget((255, 0, 0), 'Content')
        self.status.addWidget(total_images)
        self.status.addWidget(intersection_images)
        self.status.addWidget(semantic_images)
        self.status.addWidget(content_images)
        self.status.addWidget(inter_widget)
        self.status.addWidget(semantic_widget)
        self.status.addWidget(content_widget)
        self.setStatusBar(self.status)
