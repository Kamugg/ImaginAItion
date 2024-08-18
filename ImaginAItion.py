import pickle
import sys
import traceback
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import torchvision
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QImage, QPixmap, QCloseEvent, QIcon
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QListWidgetItem
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import Normalize, Resize
from torchvision.transforms import PILToTensor
from transformers import CLIPVisionModelWithProjection, AutoProcessor

from DataBaseQueryViewer import DataBaseQueryViewer
from MultimediaDB import MultimediaDB
from SettingsWindow import SettingsWindow
from models.AutoEncoder import Autoencoder

# ---------- HERE TO HANDLE CRASHES ----------

# This line are needed to show the traceback if a crash occurs

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook


# -------------------------------------------


class AppMainWindow(QMainWindow):

    def __init__(self):

        # --------------- UI ELEMENTS ---------------
        super().__init__()
        self.centralwidget = QtWidgets.QWidget(self)
        self.addDatabase = QtWidgets.QPushButton(self.centralwidget)
        self.imageContainer = QtWidgets.QLabel(self.centralwidget)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.classList = QtWidgets.QListWidget(self.groupBox)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.searchSimilar = QtWidgets.QPushButton(self.centralwidget)
        self.uploadImg = QtWidgets.QPushButton(self.centralwidget)
        self.resetBtn = QtWidgets.QPushButton(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menuOptions = QtWidgets.QMenu(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.actionPreferences = QtWidgets.QAction(self)
        self.setWindowIcon(QIcon('res/assets/icons/main_icon.png'))
        self.setWindowTitle('ImaginAItion')
        # -------------------------------------------

        self.setupUi(self)

        # Connect upload button to its function

        self.uploadImg.clicked.connect(self.upload_image)

        # Load deeplab

        self.deeplab = deeplabv3_resnet101(weights=None, progress=False, aux_loss=True)
        self.deeplab.load_state_dict(torch.load('models/deeplab.pth'))
        self.deeplab.cpu()

        # Load its preprocessing

        with open('models/deeplab_preprocessing.pkl', 'rb') as inp:
            self.deeplab_preprocessing = pickle.load(inp)

        # Load its id->name mapping for the classes and create dictionaries that map them

        with open('models/deeplab_classes.pkl', 'rb') as inp:
            self.deeplab_classes = pickle.load(inp)['categories']
        self.class_to_idx = {cls: idx for (idx, cls) in enumerate(self.deeplab_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load CLIP for semantic embedding

        self.CLIPEmbedder = CLIPVisionModelWithProjection.from_pretrained("models/clip_embedder")
        self.CLIPProcessor = AutoProcessor.from_pretrained("models/clip_processor")

        # Load my autoencoder for the content embeddings

        self.autoencoder = Autoencoder(3,
                                       1024,
                                       chain_conv=3,
                                       start_conv=32,
                                       channel_cap=128,
                                       dropout=.0,
                                       pool='MAX',
                                       use_batchnorm=True)
        self.autoencoder_preprocessing = torchvision.transforms.Compose(
            [Normalize(mean=torch.Tensor([0.4802, 0.4481, 0.3975]),
                       std=torch.Tensor([0.2296, 0.2263, 0.2255])),
             Resize((64, 64))
             ])
        self.autoencoder.load_state_dict(torch.load('models/autoencoder.pth'))

        # Set all models to evaluation mode

        self.deeplab.eval()
        self.CLIPEmbedder.eval()
        self.autoencoder.eval()

        # Loading database and the window to visualize the results of the query
        # Notice that the window is set to none because it's dinamically created when the database is called

        self.multimedia_db = MultimediaDB()
        self.query_viewer = None

        # This variables will store runtime variables for the loaded image such as the segmentation masks, the semantic
        # embedding and the content embedding

        self.detected_masks = {}
        self.semantic_embeds = None
        self.content_embeds = None

        # Loading the settings window, the settings and the way to access it

        self.settings = QSettings('Kamugg', 'ImaginAItion')
        self.setting_window = SettingsWindow(self.settings, self.multimedia_db, self.CLIPEmbedder, self.CLIPProcessor,
                                             self.autoencoder)
        self.actionPreferences.triggered.connect(lambda x: self.setting_window.show())

        self.show()

    def setupUi(self, window: QMainWindow):

        # ----------------- BOILERPLATE -----------------
        # Just sets up all the UI elements' properties

        window.setObjectName("MainWindow")
        window.resize(1336, 845)
        self.centralwidget.setObjectName("centralwidget")
        self.addDatabase.setGeometry(QtCore.QRect(48, 710, 360, 71))
        self.addDatabase.setObjectName("addDatabase")
        self.resetBtn.setGeometry(QtCore.QRect(1051, 646, 281, 48))
        self.resetBtn.setObjectName("resetBtn")
        self.imageContainer.setGeometry(QtCore.QRect(10, 10, 1031, 661))
        self.imageContainer.setAlignment(QtCore.Qt.AlignCenter)
        self.imageContainer.setObjectName("imageContainer")
        self.groupBox.setGeometry(QtCore.QRect(1051, 10, 281, 631))
        self.groupBox.setObjectName("groupBox")
        self.classList.setGeometry(QtCore.QRect(5, 21, 271, 600))
        self.classList.setObjectName("classList")
        self.line.setGeometry(QtCore.QRect(0, 690, 1331, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.searchSimilar.setGeometry(QtCore.QRect(928, 710, 360, 71))
        self.searchSimilar.setObjectName("searchSimilar")
        self.uploadImg.setGeometry(QtCore.QRect(488, 710, 360, 71))
        self.uploadImg.setObjectName("uploadImg")
        window.setCentralWidget(self.centralwidget)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1336, 26))
        self.menubar.setObjectName("menubar")
        self.menuOptions.setObjectName("menuOptions")
        window.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")
        window.setStatusBar(self.statusbar)
        self.actionPreferences.setObjectName("actionPreferences")
        self.menuOptions.addAction(self.actionPreferences)
        self.menubar.addAction(self.menuOptions.menuAction())

        self.retranslateUi(window)
        QtCore.QMetaObject.connectSlotsByName(window)

        # ----------------- BOILERPLATE -----------------

    # Function used by the "upload image button"

    def retranslateUi(self, window: QMainWindow):
        """
        Updates the text of UI elements to reflect the current language settings.
        This function was created automatically by the PyQt5 GUI editor, not by me!

        Args:
            window (QMainWindow): The main window instance whose UI elements are being updated.

        Behavior:
            - Sets the window title to "MainWindow".
            - Updates the text of various UI components, including buttons, labels, and menu items, to their translated values.
        """
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("MainWindow", "ImaginAItion"))
        self.addDatabase.setText(_translate("MainWindow", "Add to database"))
        self.resetBtn.setText(_translate("MainWindow", "Remove blur"))
        self.imageContainer.setText(_translate("MainWindow", "Waiting for image..."))
        self.groupBox.setTitle(_translate("MainWindow", "Proposed labels"))
        self.searchSimilar.setText(_translate("MainWindow", "Search similar images"))
        self.uploadImg.setText(_translate("MainWindow", "Upload image"))
        self.menuOptions.setTitle(_translate("MainWindow", "Options..."))
        self.actionPreferences.setText(_translate("MainWindow", "Preferences"))

    def upload_image(self):

        # This IF statements disconnects ALL the functions from their respective buttons This is because each button
        # "remembers" the function they were assigned to and its parameter So if the user uploads a second image
        # after the first they will still refer to all the data related to the first.
        # This caused quite a bit of headaches
        if self.detected_masks:
            self.classList.itemClicked.disconnect()
            self.resetBtn.clicked.disconnect()
            self.searchSimilar.clicked.disconnect()
            self.addDatabase.clicked.disconnect()

        # Reset everything
        self.classList.clear()
        self.detected_masks = {}
        self.semantic_embeds = None
        self.content_embeds = None

        # Open file
        file = Path(QFileDialog.getOpenFileName()[0])
        if file.is_dir(): return
        img = Image.open(file)

        # Prepare the pixmap, the UI feature that actually shows the image on the window. The image is loaded and
        # resized to fit the window, and a mask of 255 (full opacity) is attached to it. Notice that the image is
        # flipped along the color channels with np.flip(). This is because while PIL opens the image with the RGB
        # channel order, the pixmap expects them in the BGRA order.
        # This caused a funny amount of headaches
        array_img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)
        array_img = self.resize_to_fit(array_img)
        alpha_mask = np.ones((array_img.shape[0], array_img.shape[1], 1), dtype=np.uint8)
        array_img = np.concatenate([np.flip(array_img, axis=-1), 255 * alpha_mask], axis=2)
        qImg = QImage(array_img.data, array_img.shape[1], array_img.shape[0], 4 * array_img.shape[1],
                      QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qImg)
        self.imageContainer.setPixmap(pixmap)

        # The image is then converted to a torch tensor, and fed to the DeepLab preprocessor and DeepLab itself.
        pil_img = PILToTensor()(img)[None, ...]
        preproc_img = self.deeplab_preprocessing()(pil_img)
        outputs = self.deeplab(preproc_img)['out']

        # Given a (C x H x W) image, DeepLab outputs a (21 x H_i x W_i) tensor where:
        # 21 is the number of classes
        # H_i is very close to H (this is because DeepLab slightly downsamples the image)
        # Same for W_i
        # So in each of the 21 channels there is the segmentation mask for one of the specific classes. Softmax is
        # applied to each "pixel" of the output to find to whic class it was assigned to, and the result id used to find
        # out how many UNIQUE classes have been detected.
        normalized = outputs.softmax(dim=1).cpu()
        detected = torch.argmax(normalized, dim=1).unique()

        # The output is preprocessed and used to populate the segmentation_mask variable, and the unique classes are
        # used to populate the detected classes list in the window.
        self.populate_masks(detected, normalized, array_img.shape[:-1][::-1])
        for cl in [self.idx_to_class[d.item()] for d in detected if d != 0]:
            self.classList.addItem(str(cl))

        # Compute semantic embedding (vector of size 512) and normalize it
        inputs = self.CLIPProcessor(images=[img], return_tensors="pt")
        self.semantic_embeds = self.CLIPEmbedder(**inputs).image_embeds.detach().numpy().astype(np.float32)
        faiss.normalize_L2(self.semantic_embeds)

        # Compute content embedding (vector of size 1024) and normalize it
        autoencoder_input = pil_img / 255
        autoencoder_input = self.autoencoder_preprocessing(autoencoder_input)
        self.content_embeds = self.autoencoder.encode(autoencoder_input)
        self.content_embeds = torch.flatten(self.content_embeds, start_dim=1, end_dim=3).detach().numpy().astype(
            np.float32)
        norm_factor = np.linalg.norm(self.content_embeds, ord=1, axis=1)
        self.content_embeds /= norm_factor[:, None]

        # Finally all the buttons in the UI are connected to their respective function
        self.classList.itemClicked.connect(lambda clickedItem: self.clicked_class(clickedItem, array_img))
        self.resetBtn.clicked.connect(lambda: self.clicked_class('default', array_img))
        self.searchSimilar.clicked.connect(self.search_similar)
        self.addDatabase.clicked.connect(lambda: self.add_database(file))

    # Function to resize the image to fit in the window without changeing its aspect ratio.

    def resize_to_fit(self, img: np.ndarray) -> np.ndarray:
        """
        Resizes an image to fit within the dimensions of the image container while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Resized image as a NumPy array.

        Behavior:
            - Retrieves the dimensions of container.
            - Computes the scaling factor based on the container size and image size.
            - Resizes the image with the computed scaling factor.
            - Returns the resized image.
        """
        c_h, c_w = self.imageContainer.height(), self.imageContainer.width()
        i_h, i_w = img.shape[0], img.shape[1]
        res_factor = min(c_h / i_h, c_w / i_w)
        resized = cv2.resize(img, None, fx=res_factor, fy=res_factor)
        return resized

    def populate_masks(self, detected: torch.Tensor, output: torch.Tensor, shape: tuple):
        """
            Uses the DeepLab output to populate the detcted_masks variable

            Args:
                detected (torch.Tensor): A 1D tensor containing class indices of detected objects.
                                          Each value corresponds to a class index in `output`.
                output (torch.Tensor): A 3D tensor with shape `(num_classes, height, width)`
                                       containing the model softmaxed output.
                shape (tuple): A tuple `(height, width)` specifying the desired shape for the output masks.

            Behavior:
                - For each detected class, a mask is generated by thresholding the corresponding
                  slice of the `output` tensor.
                - The threshold is determined by the value stored in the settings and 30 is used if none is found,
                  which is divided by 100 to convert it into the [0, 1] range.
                - The mask is then processed to create a binary mask, which is resized to the specified
                  shape.
                - The processed mask is stored in the `self.detected_masks` dictionary, with the key
                  being the class name obtained from `self.idx_to_class`.
        """
        alpha = 100
        for d in detected:
            if d.item() != 0:
                mask = output[0, d].detach().numpy()
                mask = (mask > self.settings.value('thr', 30) / 100).astype(np.uint8)
                negative_mask = alpha * (1 - mask).astype(np.uint8)
                mask = negative_mask + 255 * mask
                mask = cv2.resize(mask, shape)
                self.detected_masks[self.idx_to_class[d.item()]] = mask
        self.detected_masks['default'] = cv2.resize(255 * np.ones(shape).astype(np.uint8), shape)

    def clicked_class(self, clicked_item: QListWidgetItem, arr_img: np.ndarray):
        """
        Updates the displayed image based on the selected class mask.

        Args:
            clickedItem: The selected item, either a string ('default') or a list item with a `text()` method.
            arr_img (np.ndarray): A 3D NumPy array representing the image, expected to have an alpha channel.

        Behavior:
            - Retrieves the corresponding mask from `self.detected_masks` based on the selected class.
            - Applies the mask to the alpha channel of `arr_img`.
            - Converts `arr_img` to a `QImage` and then to a `QPixmap`.
            - Sets the `QPixmap` for display.
        """
        if clicked_item == 'default':
            mask = self.detected_masks['default']
        else:
            mask = self.detected_masks[clicked_item.text()]
        arr_img[:, :, -1] = mask
        qImg = QImage(arr_img.data, arr_img.shape[1], arr_img.shape[0], 4 * arr_img.shape[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qImg)
        self.imageContainer.setPixmap(pixmap)

    def search_similar(self):
        """
        Searches for similar items in the database based on the provided image file.

        Behavior:.
            - Queries the database with the image embeddings.
            - Displays the query results using by calling the `DataBaseQueryViewer` window.
        """
        result = self.multimedia_db.query(self.semantic_embeds, self.content_embeds, self.settings)
        self.query_viewer = DataBaseQueryViewer(result)
        self.query_viewer.show()

    def add_database(self, file: Path):
        """
        Adds an image to the database.

        Args:
            file (Path): Path to the image file to be added.

        Behavior:
            - Checks if a file with the same name already exists in the database.
            - Displays an error message if the file exists.
            - Adds the image and its embeddings to the database.
            - Shows a success message after the image is added.
        """
        db_path = Path('dbs/db_images') / file.name
        if db_path.exists():
            error = QMessageBox()
            error.setIcon(QMessageBox.Critical)
            error.setWindowTitle('Error')
            error.setText('A file with this name already exists in the database')

            error.setWindowIcon(QIcon('res/assets/icons/error_icon.png'))
            error.exec_()
            return
        self.multimedia_db.add(file, self.semantic_embeds, self.content_embeds)
        success = QMessageBox()
        success.setIcon(QMessageBox.Information)
        success.setWindowTitle('Image added successfully!')
        success.setText('Image added to the database!')
        success.setWindowIcon(QIcon('res/assets/icons/success_icon.png'))
        success.exec_()

    def closeEvent(self, event: QCloseEvent):
        """
        Handles the window close event to save the state of the multimedia database. This method is ALWAYS called when
        the window tries to close.

        Args:
            event (QCloseEvent): The event object associated with the window close event.

        Behavior:
            - Saves the state of the multimedia database using `self.multimedia_db.save_state()`.
            - Accepts the close event to proceed with closing the window.
        """
        self.multimedia_db.save_state()
        event.accept()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    ui = AppMainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    sys.exit(app.exec_())
