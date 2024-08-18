import sys
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QCloseEvent, QShowEvent, QIcon
from PyQt5.QtWidgets import QMainWindow
from torch.nn import Module
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from MultimediaDB import MultimediaDB
from ResetDBWindow import ResetDBWindow

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook


class SettingsWindow(QMainWindow):
    """
    Class to create the settings window.
    """

    def __init__(self, settings: QSettings,
                 db: MultimediaDB,
                 clip: CLIPVisionModelWithProjection,
                 clip_processor: AutoProcessor,
                 autoencoder: Module) -> None:
        """
        Initializes the main application window and its UI components. Notice that the models are needed since from this
        window you can go to the "reset database" section that needs the models to recompute the various embeddings!

        Arguments:
        ----------
        settings : QSettings
            Configuration settings for the application.
        db : MultimediaDB
            Database object for multimedia content management.
        clip : CLIPVisionModelWithProjection
            The CLIP model used for semantic embedding computation.
        clip_processor : AutoProcessor
            Processor for preparing inputs for the CLIP model.
        autoencoder : torch.Module
            Autoencoder model used for content embedding computation.

        Behavior:
        ---------
        - Sets up the central widget and various UI components such as group boxes, sliders, and buttons.
        - Connects signals to slots for interactive UI elements, such as updating label values when sliders change.
        - Initializes a `ResetDBWindow` for handling database resets and configures the button to display this window.
        - Validates input in the k-value text field to ensure only integers are accepted.

        Returns:
        --------
        None
        """
        super().__init__()
        self.centralwidget = QtWidgets.QWidget(self)
        self.segmentation_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.segmentation_title = QtWidgets.QLabel(self.segmentation_gbox)
        self.segmentation_description = QtWidgets.QLabel(self.segmentation_gbox)
        self.thr_slider = QtWidgets.QSlider(self.segmentation_gbox)
        self.thr_val_label = QtWidgets.QLabel(self.segmentation_gbox)
        self.thr_slider_val = QtWidgets.QLabel(self.segmentation_gbox)
        self.dbretrieval_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.k_title = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.k_desc = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.k_val_label = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.k_val_insert = QtWidgets.QLineEdit(self.dbretrieval_gbox)
        self.policy_title = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.policy_desc = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.semantic_btn = QtWidgets.QRadioButton(self.dbretrieval_gbox)
        self.intersect_chbox = QtWidgets.QCheckBox(self.dbretrieval_gbox)
        self.content_btn = QtWidgets.QRadioButton(self.dbretrieval_gbox)
        self.ksplit_title = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.ksplit_desc = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.ksplit_val_label = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.ksplit_slider = QtWidgets.QSlider(self.dbretrieval_gbox)
        self.ksplit_slider_val = QtWidgets.QLabel(self.dbretrieval_gbox)
        self.dbrestart_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.dbfolder_title = QtWidgets.QLabel(self.dbrestart_gbox)
        self.dbfolder_desc = QtWidgets.QLabel(self.dbrestart_gbox)
        self.dbstatus_label = QtWidgets.QLabel(self.dbrestart_gbox)
        self.resetdb_button = QtWidgets.QPushButton(self.dbrestart_gbox)
        self.dbstatus_value = QtWidgets.QLabel(self.dbrestart_gbox)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setWindowIcon(QIcon('res/assets/icons/settings_icon.png'))
        self.setupUi()

        self.thr_slider_val.setText(str(self.thr_slider.value()))
        self.thr_slider.valueChanged.connect(lambda: self.thr_slider_val.setText(str(self.thr_slider.value())))

        self.ksplit_slider_val.setText(str(self.ksplit_slider.value()))
        self.ksplit_slider.valueChanged.connect(lambda: self.ksplit_slider_val.setText(str(self.ksplit_slider.value())))

        self.multimedia_db = db
        self.settings = settings
        self.reset_db_win = ResetDBWindow(db, clip, clip_processor, autoencoder, self.update_db_status)

        self.k_val_insert.setValidator(QtGui.QIntValidator())
        self.resetdb_button.clicked.connect(lambda: self.reset_db_win.show())

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Saves the settings when the window is closed.

        Arguments:
        ----------
        event : QCloseEvent
            The close event triggered when the user attempts to close the application window.

        Behavior:
        ---------
        - Saves the current state of UI elements to the application's settings before closing.
          Specifically, it stores:
            - The value of the threshold slider (`thr_slider`).
            - The value entered in the k-value input field (`k_val_insert`).
            - The state (checked or unchecked) of the intersection checkbox (`intersect_chbox`).
            - The selected policy, either 'semantic' or 'content', based on the radio button selection.
            - The value of the k-split slider (`ksplit_slider`).

        Returns:
        --------
        None
        """
        self.settings.setValue('thr', int(self.thr_slider.value()))
        self.settings.setValue('k', int(self.k_val_insert.text()))
        self.settings.setValue('intersection', self.intersect_chbox.isChecked())
        self.settings.setValue('policy', 'semantic' if self.semantic_btn.isChecked() else 'content')
        self.settings.setValue('k_split', int(self.ksplit_slider.value()))

    def showEvent(self, event: QShowEvent) -> None:
        """
        Handles the actions to perform when the application window is shown.

        Arguments:
        ----------
        event : QShowEvent
            The show event triggered when the application window is displayed.

        Behavior:
        ---------
        - Updates the database status by checking the file integrity and reflecting it in the UI.
        - Loads and applies the saved settings to the UI components, including:
            - Setting the threshold slider (`thr_slider`) to the saved value, with a default of 30.
            - Setting the k-value input field (`k_val_insert`) to the saved value, with a default of 50.
            - Checking or unchecking the intersection checkbox (`intersect_chbox`) based on the saved state.
            - Selecting the appropriate policy radio button (`semantic_btn` or `content_btn`) based on the saved setting.
            - Setting the k-split slider (`ksplit_slider`) to the saved value, with a default of 50.

        Returns:
        --------
        None
        """
        self.update_db_status(self.multimedia_db.check_file_integrity())
        self.thr_slider.setValue(self.settings.value('thr', 30))
        self.k_val_insert.setText(str(self.settings.value('k', 50)))
        self.intersect_chbox.setChecked(self.settings.value('intersection', True) == 'true')
        if self.settings.value('policy', 'semantic') == 'semantic':
            self.semantic_btn.setChecked(True)
        else:
            self.content_btn.setChecked(True)
        self.ksplit_slider.setValue(self.settings.value('k_split', 50))

    def update_db_status(self, code) -> None:
        """
        Updates the database status label in the UI based on the provided status code.

        Arguments:
        ----------
        code : int
            An integer representing the status of the database integrity check. The possible values are:
            - 0: Database is OK.
            - 1: Missing database file.
            - 2: Missing image folder.
            - 3: Index mismatch.
            - 4: Missing images.

        Behavior:
        ---------
        - Sets the text of the database status label (`dbstatus_value`) according to the status code.
        - Changes the color of the status text to reflect the status:
            - Green (RGB: 0, 255, 0) for OK status.
            - Red (RGB: 255, 0, 0) for any error condition.

        Returns:
        --------
        None
        """
        codes = {
            0: ('OK!', (0, 255, 0)),
            1: ('Missing database file', (255, 0, 0)),
            2: ('Missing image folder', (255, 0, 0)),
            3: ('Index mismatch', (255, 0, 0)),
            4: ('Missing images', (255, 0, 0))
        }
        self.dbstatus_value.setText(codes[code][0])
        self.dbstatus_value.setStyleSheet(f'color: rgb{codes[code][1]}')

    # Autogenerated by te GUI creation software of PyQt5. Sets various properties of the UI elements
    def setupUi(self):
        self.setObjectName("SettingsWindow")
        self.resize(800, 754)
        self.centralwidget.setObjectName("centralwidget")
        self.segmentation_gbox.setGeometry(QtCore.QRect(10, 0, 781, 131))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.segmentation_gbox.setFont(font)
        self.segmentation_gbox.setObjectName("segmentation_gbox")
        self.segmentation_title.setGeometry(QtCore.QRect(10, 20, 161, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.segmentation_title.setFont(font)
        self.segmentation_title.setObjectName("segmentation_title")
        self.segmentation_description.setGeometry(QtCore.QRect(10, 41, 771, 51))
        self.segmentation_description.setScaledContents(False)
        self.segmentation_description.setWordWrap(True)
        self.segmentation_description.setObjectName("segmentation_description")
        self.thr_slider.setGeometry(QtCore.QRect(131, 107, 160, 22))
        self.thr_slider.setMaximum(100)
        self.thr_slider.setSingleStep(10)
        self.thr_slider.setProperty("value", 30)
        self.thr_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.thr_slider.setObjectName("thr_slider")
        self.thr_val_label.setGeometry(QtCore.QRect(10, 107, 111, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.thr_val_label.setFont(font)
        self.thr_val_label.setObjectName("thr_val_label")
        self.thr_slider_val.setGeometry(QtCore.QRect(296, 107, 151, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.thr_slider_val.setFont(font)
        self.thr_slider_val.setObjectName("thr_slider_val")
        self.dbretrieval_gbox.setGeometry(QtCore.QRect(10, 140, 781, 391))
        self.dbretrieval_gbox.setObjectName("dbretrieval_gbox")
        self.k_title.setGeometry(QtCore.QRect(10, 20, 191, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.k_title.setFont(font)
        self.k_title.setObjectName("k_title")
        self.k_desc.setGeometry(QtCore.QRect(10, 41, 341, 20))
        self.k_desc.setScaledContents(False)
        self.k_desc.setWordWrap(True)
        self.k_desc.setObjectName("k_desc")
        self.k_val_label.setGeometry(QtCore.QRect(10, 76, 121, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.k_val_label.setFont(font)
        self.k_val_label.setObjectName("k_val_label")
        self.k_val_insert.setGeometry(QtCore.QRect(141, 76, 113, 22))
        self.k_val_insert.setObjectName("k_val_insert")
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.policy_title.setGeometry(QtCore.QRect(10, 112, 191, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.policy_title.setFont(font)
        self.policy_title.setObjectName("policy_title")
        self.policy_desc.setGeometry(QtCore.QRect(10, 133, 771, 71))
        self.policy_desc.setScaledContents(False)
        self.policy_desc.setWordWrap(True)
        self.policy_desc.setObjectName("policy_desc")
        self.intersect_chbox.setGeometry(QtCore.QRect(10, 219, 321, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.intersect_chbox.setFont(font)
        self.intersect_chbox.setChecked(True)
        self.intersect_chbox.setTristate(False)
        self.intersect_chbox.setObjectName("intersect_chbox")
        self.semantic_btn.setGeometry(QtCore.QRect(10, 249, 361, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.semantic_btn.setFont(font)
        self.semantic_btn.setChecked(True)
        self.semantic_btn.setObjectName("semantic_btn")
        self.content_btn.setGeometry(QtCore.QRect(381, 249, 361, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.content_btn.setFont(font)
        self.content_btn.setChecked(False)
        self.content_btn.setObjectName("content_btn")
        self.ksplit_title.setGeometry(QtCore.QRect(10, 289, 191, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ksplit_title.setFont(font)
        self.ksplit_title.setObjectName("ksplit_title")
        self.ksplit_desc.setGeometry(QtCore.QRect(10, 310, 771, 41))
        self.ksplit_desc.setScaledContents(False)
        self.ksplit_desc.setWordWrap(True)
        self.ksplit_desc.setObjectName("ksplit_desc")
        self.ksplit_val_label.setGeometry(QtCore.QRect(10, 366, 41, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ksplit_val_label.setFont(font)
        self.ksplit_val_label.setObjectName("ksplit_val_label")
        self.ksplit_slider.setGeometry(QtCore.QRect(61, 366, 160, 22))
        self.ksplit_slider.setMaximum(100)
        self.ksplit_slider.setSingleStep(10)
        self.ksplit_slider.setProperty("value", 50)
        self.ksplit_slider.setSliderPosition(50)
        self.ksplit_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ksplit_slider.setObjectName("ksplit_slider")
        self.ksplit_slider_val.setGeometry(QtCore.QRect(226, 366, 151, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ksplit_slider_val.setFont(font)
        self.ksplit_slider_val.setObjectName("ksplit_slider_val")
        self.dbrestart_gbox.setGeometry(QtCore.QRect(10, 540, 781, 151))
        self.dbrestart_gbox.setObjectName("dbrestart_gbox")
        self.dbfolder_title.setGeometry(QtCore.QRect(10, 20, 111, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dbfolder_title.setFont(font)
        self.dbfolder_title.setObjectName("dbfolder_title")
        self.dbfolder_desc.setGeometry(QtCore.QRect(10, 41, 761, 31))
        self.dbfolder_desc.setScaledContents(False)
        self.dbfolder_desc.setWordWrap(True)
        self.dbfolder_desc.setObjectName("dbfolder_desc")
        self.dbstatus_label.setGeometry(QtCore.QRect(10, 87, 121, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dbstatus_label.setFont(font)
        self.dbstatus_label.setObjectName("dbstatus_label")
        self.resetdb_button.setGeometry(QtCore.QRect(10, 113, 161, 28))
        self.resetdb_button.setObjectName("resetdb_button")
        self.dbstatus_value.setGeometry(QtCore.QRect(140, 87, 211, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.dbstatus_value.setFont(font)
        self.dbstatus_value.setObjectName("dbstatus_value")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    # Autogenerated by te GUI creation software of PyQt5. Translates the various strings to the system's language
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("SettingsWindow", "Settings"))
        self.segmentation_gbox.setTitle(_translate("SettingsWindow", "Segmentation settings"))
        self.segmentation_title.setText(_translate("SettingsWindow", "Segmentation threshold"))
        self.segmentation_description.setText(_translate("SettingsWindow",
                                                         "Changing the segmentation threshold will affect segmentation quality. A value too high (> 50) will result in precise segmentation with the risk of missing details and small objects. On the other hand a value too small (< 30) will result in the segmentation of all the relevants objects in the image with some errors."))
        self.thr_val_label.setText(_translate("SettingsWindow", "Threshold value:"))
        self.thr_slider_val.setText(_translate("SettingsWindow", "SliderValue"))
        self.dbretrieval_gbox.setTitle(_translate("SettingsWindow", "Database retrieval settings"))
        self.k_title.setText(_translate("SettingsWindow", "Number of images retrieved"))
        self.k_desc.setText(_translate("SettingsWindow", "Images retrieved from the database whenever it\'s queried."))
        self.k_val_label.setText(_translate("SettingsWindow", "Images retrieved:"))
        self.k_val_insert.setText(_translate("SettingsWindow", "50"))
        self.policy_title.setText(_translate("SettingsWindow", "Retrieval policies"))
        self.policy_desc.setText(_translate("SettingsWindow",
                                            "How the query results should be organized. The software collects the relevant images considering both the raw content of the images and their semantic content. By default the application shows the images retrieved from both the semantic and the query engine first regardless of their respective scores, then only the ones retrieved from the semantic engine and lastly the ones retrieved by the content engine."))
        self.intersect_chbox.setText(_translate("SettingsWindow", "Show images retrieved by both engines first"))
        self.semantic_btn.setText(_translate("SettingsWindow", "Prioritize images retrieved by the semantic engine"))
        self.content_btn.setText(_translate("SettingsWindow", "Prioritize images retrieved by the content engine"))
        self.ksplit_title.setText(_translate("SettingsWindow", "Semantic/Content split"))
        self.ksplit_desc.setText(_translate("SettingsWindow",
                                            "Select the percentage of image to be retrieved under the selected policy. By default the software returns an even split between images retrieved by semantic and by content"))
        self.ksplit_val_label.setText(_translate("SettingsWindow", "Split"))
        self.ksplit_slider_val.setText(_translate("SettingsWindow", "SliderValue"))
        self.dbrestart_gbox.setTitle(_translate("SettingsWindow", "Database initialization"))
        self.dbfolder_title.setText(_translate("SettingsWindow", "Database folder"))
        self.dbfolder_desc.setText(_translate("SettingsWindow",
                                              "Use this field to load the initial images in the database, or to restart it if the database files are corrupted. Mind that this operation may require several minutes."))
        self.dbstatus_label.setText(_translate("SettingsWindow", "Database Status:"))
        self.resetdb_button.setText(_translate("SettingsWindow", "Reset/Initialize Database"))
        self.dbstatus_value.setText(_translate("SettingsWindow", "0"))
