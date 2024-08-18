from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow
from torch.nn import Module
from transformers import CLIPVisionModelWithProjection, AutoProcessor

from MultimediaDB import MultimediaDB


class DBReinitializer(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, db: MultimediaDB,
                 folder: Path,
                 clip: CLIPVisionModelWithProjection,
                 clip_processor: AutoProcessor,
                 autoencoder: Module,
                 update_func: Callable,
                 calling_win: QMainWindow) -> None:
        """
        Initializes the DBReinitializer thread with the necessary components.

        Arguments:
        ----------
        db : MultimediaDB
            The multimedia database object to be reinitialized.
        folder : Path
            The directory path where the database files are stored.
        clip : CLIPVisionModelWithProjection
            The CLIP model used for semantic embeddings.
        clip_processor : AutoProcessor
            Processor for preparing inputs for the CLIP model.
        autoencoder : torch.Module
            Autoencoder model used for content embeddings.
        update_func : Callable
            A callable function to update the database status label in the settings window.
        calling_win : QMainWindow
            The window that initiated the database reinitialization, which will be closed upon completion.

        Returns:
        --------
        None
        """
        super().__init__()
        self.db = db
        self.folder = folder
        self.clip = clip
        self.clip_processor = clip_processor
        self.autoencoder = autoencoder
        self.update_func = update_func
        self.calling_win = calling_win

    def run(self) -> None:
        """
        Executes the database reinitialization process.

        Behavior:
        ---------
        - Reinitializes the database using the provided folder, models, and processor.
        - Emits progress updates and status messages during the process.
        - Resets the database object after reinitialization.
        - Calls the update function to synchronize the database status with the settings window.
        - Closes the calling window upon completion of the reinitialization.

        Returns:
        --------
        None
        """
        self.db.reinitialize_db(self.folder, self.clip, self.clip_processor, self.autoencoder, self.progress.emit,
                                self.status.emit)
        self.db = MultimediaDB()
        self.update_func(self.db.check_file_integrity())
        self.calling_win.close()
