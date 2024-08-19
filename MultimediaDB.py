import json
import os
import shutil
from pathlib import Path
from typing import Callable

import faiss
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import QSettings
from torch.nn import Module
from torchvision.transforms import PILToTensor, Resize, Normalize, Compose
from transformers import CLIPVisionModelWithProjection, AutoProcessor


class MultimediaDB(object):

    def __init__(self):

        # Load required files

        try:
            with open('dbs/path_index.json') as f:
                self.path_index = json.load(f)
            self.path_index = {int(k): self.path_index[k] for k in self.path_index.keys()}
        except FileNotFoundError:
            self.path_index = {}

        try:
            self.semantic_index = faiss.read_index('dbs/semantic_db.faiss')
        except RuntimeError:
            self.semantic_index = None

        try:
            self.content_index = faiss.read_index('dbs/content_db.faiss')
        except RuntimeError:
            self.content_index = None

        self.status_ok = self.check_file_integrity()

        # JSON store integers as strings when loaded, let's convert them back to ints
        self.path_index = {int(k): v for k, v in self.path_index.items()}

        # Sentry variable that checks if the files need to be updated
        self.need_to_save = False

    def query(self, semantic_repr: np.ndarray, content_repr: np.ndarray, settings: QSettings) -> dict:
        """
            Perform a search query using both semantic and content-based representations.

            Parameters:
                semantic_repr (np.ndarray): The semantic representation of the query.
                content_repr (np.ndarray): The content-based representation of the query.
                settings (QSettings): QSettings object from PyQt5 that stores the user's settings.

            Returns:
                dict: A dictionary containing the search results, combining semantic and content-based
                      matches, with keys indicating retrieval method and similarity scores.

            Behavior:
                - Retrieves `k` items based on the semantic and content representations from the FAISS indexes.
                - If `intersection` is enabled, items found in both representations are prioritized regardless of their
                score.
                - The retrieval policy ('SEMANTIC_FIRST' or 'CONTENT_FIRST') determines the order
                  and proportion (`k_split`) of results fetched from each representation.
                - Final results are aggregated, with metadata indicating whether each result
                  was retrieved by semantic match, content match, or intersection.
            """
        if self.status_ok == 0:

            # Get parameters from settings
            k = settings.value('k', 50)
            k_split = settings.value('k_split', 50) / 100
            policy = 'SEMANTIC_FIRST' if settings.value('policy', 'semantic') == 'semantic' else 'CONTENT_FIRST'
            use_intersection = settings.value('intersection', 'true') == 'true'

            # Retrieve semantic similarities

            semantic_sim, semantic_ids = self.semantic_index.search(semantic_repr, k)
            semantic_sim = 1. - .5 * semantic_sim
            semantic_sim, semantic_ids = semantic_sim[0].tolist(), semantic_ids[0].tolist()

            semantic_res = dict(zip(semantic_ids, semantic_sim))

            # Retrieve content similarities

            content_sim, content_ids = self.content_index.search(content_repr, k)
            content_sim = 1. - .5 * content_sim
            content_sim, content_ids = content_sim[0].tolist(), content_ids[0].tolist()

            content_res = dict(zip(content_ids, content_sim))

            # Start building the final result dictionary

            final_res = {}
            dict_index = 0
            intersect = []
            if use_intersection:
                intersect = np.intersect1d(list(semantic_res.keys()), list(content_res.keys())).tolist()
                intersect.sort(reverse=True,
                               key=lambda key: semantic_res[key] if policy == 'SEMANTIC_FIRST' else content_res[key])

                for key in intersect:
                    final_res[dict_index] = self.__build_dict_from_id(key, semantic_res, content_res, semantic_repr,
                                                                      content_repr)
                    final_res[dict_index]['retrieved_by'] = 'INTERSECTION'
                    dict_index += 1

                # Remove the intersection keys from the dictionaries

                for key in intersect:
                    semantic_res.pop(key, None)
                    content_res.pop(key, None)

            # Set how many image to retrieve semantically and how many to retrieve by content

            k -= len(intersect)
            if policy == 'SEMANTIC_FIRST':
                sem_k = round(k * k_split)
                cont_k = k - sem_k
            elif policy == 'CONTENT_FIRST':
                cont_k = round(k * k_split)
                sem_k = k - cont_k

            # Retrieve remaining images according to policy and split

            if policy == 'SEMANTIC_FIRST':
                ordered = [(semantic_res, sem_k, 'SEMANTIC'), (content_res, cont_k, 'CONTENT')]
            elif policy == 'CONTENT_FIRST':
                ordered = [(content_res, cont_k, 'CONTENT'), (semantic_res, sem_k, 'SEMANTIC')]

            for item in ordered:
                for key in list(item[0].keys())[:item[1]]:
                    final_res[dict_index] = self.__build_dict_from_id(key, semantic_res, content_res, semantic_repr,
                                                                      content_repr)
                    final_res[dict_index]['retrieved_by'] = item[2]
                    dict_index += 1

            return final_res

    def __build_dict_from_id(self, key: int, sem: dict, con: dict, s_repr: np.ndarray, c_repr: np.ndarray) -> dict:
        """
        Build a dictionary containing metadata and similarity scores for an image identified by `key`.

        Parameters:
            key (int): The unique identifier for the image.
            sem (dict): A dictionary of precomputed semantic similarities with image IDs as keys.
            con (dict): A dictionary of precomputed content similarities with image IDs as keys.
            s_repr (np.ndarray): The semantic representation of the query.
            c_repr (np.ndarray): The content-based representation of the query.

        Returns:
            dict: A dictionary containing the image's ID, file path, and similarity scores
                  (both semantic and content-based).

        Behavior:
            - Retrieves the file path associated with the image ID.
            - Calculates and includes the semantic similarity:
                - Uses the precomputed value if available, otherwise computes it using the semantic index.
            - Calculates and includes the content similarity:
                - Uses the precomputed value if available, otherwise computes it using the content index.
            - The computed similarities are normalized and adjusted to fit the expected similarity range.
        """
        image_dict = {}
        image_dict['id'] = key
        image_dict['path'] = self.path_index[key]
        s_repr = s_repr.squeeze()
        c_repr = c_repr.squeeze()
        if key in sem:
            image_dict['semantic_similarity'] = sem[key]
        else:
            sem_vector = self.semantic_index.reconstruct(key)
            dist = np.linalg.norm(s_repr - sem_vector)
            image_dict['semantic_similarity'] = 1 - .5 * dist
        if key in con:
            image_dict['content_similarity'] = con[key]
        else:
            con_vector = self.content_index.reconstruct(key)
            dist = np.linalg.norm(c_repr - con_vector, ord=1)
            image_dict['content_similarity'] = 1 - .5 * dist
        return image_dict

    def check_file_integrity(self):
        """
        Check the integrity of database files, directories, and indices.

        Returns:
            int: Status code indicating the result of the integrity check:
                 - 0: All files, directories, and indices are intact.
                 - 1: A required database file is missing.
                 - 2: A required directory (db or db_images) is missing.
                 - 3: Mismatch between the number of entries in semantic and content indices, or image files.
                 - 4: An image listed in the index is missing from the file system.

        Behavior:
            - Verifies the existence of critical database files (content_db.faiss, semantic_db.faiss, path_index.json).
            - Ensures that required directories (db and db_images) exist.
            - Checks that the total number of entries in the semantic and content indices match, and that they align with the number of image files.
            - Confirms that each file path listed in the path index exists in the file system.
        """
        paths = {
            'content_db_path': Path('dbs/content_db.faiss'),
            'semantic_db_path': Path('dbs/semantic_db.faiss'),
            'path_index': Path('dbs/path_index.json')
        }

        for path in paths.values():
            if not path.is_file():
                return 1  # Missing db file

        db_path = Path('dbs')
        db_img_path = db_path / Path('db_images')

        if not (db_img_path.is_dir() and db_path.is_dir()):
            return 2  # Missing one of the db folders

        if not ((self.semantic_index.ntotal == self.content_index.ntotal) and (
                self.semantic_index.ntotal == len(os.listdir(db_img_path)))):
            return 3  # Index mismatch

        for path in self.path_index.values():
            if not Path(path).is_file():
                return 4  # Image present in index but not in files

        return 0  # Everything's ok!

    def reinitialize_db(self, folder: Path,
                        clip: CLIPVisionModelWithProjection,
                        clip_processor: AutoProcessor,
                        autoencoder: Module,
                        progress_callback: Callable,
                        status_callback: Callable):
        """
        Reinitialize the database by copying images, generating embeddings, and creating new FAISS indices.

        Parameters:
            folder (Path): The path to the folder containing images to be indexed.
            clip (CLIPVisionModelWithProjection): The CLIP model used for generating semantic embeddings.
            clip_processor (AutoProcessor): The processor for preparing images for the CLIP model.
            autoencoder (torch.Module): The autoencoder model used for generating content embeddings.
            progress_callback (Callable): A function to update progress as a percentage.
            status_callback (Callable): A function to report the current status of the process.

        Behavior:
            - Sets the CLIP and autoencoder models to evaluation mode and moves them to the GPU.
            - Deletes existing images and index files from the database directory.
            - Copies images from the specified folder to the database, updating the path index.
            - Computes semantic embeddings using the CLIP model and content embeddings using the autoencoder:
                - Processes images in batches, updating the progress and status after each batch.
                - Semantic embeddings are normalized and stored in a FAISS index using L2 distance.
                - Content embeddings are normalized and stored in a FAISS index using L1 distance.
            - Saves the path index and the newly created FAISS indices to disk.
            - Calls the provided callbacks to report progress and status throughout the process.
        """
        BATCH_SIZE = 64
        necessary_steps = len(os.listdir(folder))
        current_step = 0

        # Setting all models to evaluation mode

        clip.eval()
        clip.to(device='cuda')
        autoencoder.eval()
        autoencoder.to(device='cuda')

        # Delete all db files

        db_path = Path('dbs')
        db_image_path = Path('dbs/db_images')
        if db_path.is_dir():

            # Delete all db stored images

            if db_image_path.is_dir():
                necessary_steps += len(os.listdir(db_image_path))
                for file in os.listdir(db_image_path):
                    status_callback(f'Deleting {file}')
                    file_path = db_image_path / file
                    os.unlink(file_path)
                    current_step += 1
                    progress_callback(round(100 * (current_step / necessary_steps)))
            else:
                os.mkdir(db_image_path)

            # Delete all indexes

            for file in os.listdir(db_path):
                file_path = db_path / file
                if file_path.is_file():
                    os.unlink(file_path)
        else:
            os.mkdir(db_path)
            os.mkdir(db_image_path)

        # Move all images and create new index

        self.path_index = {}

        for i, file in enumerate(os.listdir(folder)):
            status_callback(f'Copying {file}')
            t_path = Path('dbs/db_images') / file
            self.path_index[i] = str(t_path)
            src = Path(folder) / file
            shutil.copy(src, t_path)
            current_step += 1
            progress_callback(round(100 * (current_step / necessary_steps)))

        # Compute embeddings

        autoencoder_preprocessing = Compose(
            [Normalize(mean=torch.Tensor([0.4802, 0.4481, 0.3975]),
                       std=torch.Tensor([0.2296, 0.2263, 0.2255])),
             Resize((64, 64))
             ])

        with torch.no_grad():
            max_id = max(self.path_index.keys()) + 1
            semantic_embeds = np.zeros((max_id, 512))
            content_embeds = np.zeros((max_id, 1024))
            batches = [(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, max_id)) for b in range(max_id // BATCH_SIZE + 1)]

            current_step = 0
            necessary_steps = len(batches) + 2
            progress_callback(0)
            status_callback('Starting embedding computation...')
            computed_files = 0

            for k, batch in enumerate(batches):

                status_callback(f'Computing embeddings... (files {computed_files}/{max_id})')

                # Semantic embeds

                image_paths = [self.path_index[p] for p in range(batch[0], batch[1])]
                images = [Image.open(p) for p in image_paths]
                inputs = clip_processor(images=images, return_tensors="pt")
                inputs = inputs.to(device='cuda')
                outputs = clip(**inputs).image_embeds.cpu().detach().numpy()
                semantic_embeds[batch[0]:batch[1], :] = outputs

                # Content embeds

                autoencoder_images = [PILToTensor()(i) / 255 for i in images]
                batch_tensor = torch.zeros((len(autoencoder_images), 3, 64, 64))
                for idx, img in enumerate(autoencoder_images):
                    batch_tensor[idx, ...] = autoencoder_preprocessing(img)
                batch_tensor = batch_tensor.to(device='cuda')
                outputs = torch.flatten(autoencoder.encode(batch_tensor).cpu(), start_dim=1, end_dim=3).numpy()
                content_embeds[batch[0]:batch[1], :] = outputs

                current_step += 1
                computed_files += (batch[1]-batch[0])
                progress_callback(round(100 * (current_step / necessary_steps)))

        autoencoder.cpu()
        clip.cpu()

        # Normalization and creation of semantic faiss indexes

        status_callback('Creating semantic index...')
        semantic_embeds = semantic_embeds.astype(np.float32)
        self.semantic_index = faiss.IndexFlatL2(512)
        faiss.normalize_L2(semantic_embeds)
        self.semantic_index.add(semantic_embeds)
        faiss.write_index(self.semantic_index, 'dbs/semantic_db.faiss')
        current_step += 1
        progress_callback(round(100 * (current_step / necessary_steps)))

        # Normalization and creation of content faiss indexes

        status_callback('Creating content index...')
        content_embeds = content_embeds.astype(np.float32)
        norm_factor = np.linalg.norm(content_embeds, ord=1, axis=1)
        content_embeds = content_embeds / norm_factor[:, None]
        self.content_index = faiss.IndexFlat(1024, faiss.METRIC_L1)
        self.content_index.add(content_embeds)
        faiss.write_index(self.content_index, 'dbs/content_db.faiss')
        current_step += 1
        progress_callback(round(100 * (current_step / necessary_steps)))

        # Saving path mapping

        with open('dbs/path_index.json', 'w') as f:
            json.dump(self.path_index, f)

        # Update DB status

        self.status_ok = self.check_file_integrity()

    def add(self, path: Path, semantic_embeds: np.ndarray, content_embeds: np.ndarray):
        """
        Add a new image to the database along with its semantic and content embeddings.

        Parameters:
            path (Path): The file path to the image being added.
            semantic_embeds (np.ndarray): The semantic embedding of the image.
            content_embeds (np.ndarray): The content embedding of the image.

        Behavior:
            - Sets a flag indicating that the database needs to be saved upon application closure.
            - Generates a new ID for the image and updates both the semantic and content FAISS indices.
            - Copies the image to the database's image directory and updates the path index with the new image's location.
        """
        # Updates sentry variable to save the database when the app is closed
        self.need_to_save = True

        # Updating indexes
        new_id = max(self.path_index.keys()) + 1
        self.semantic_index.add(semantic_embeds)
        self.content_index.add(content_embeds)
        src_path = Path(path)
        db_path = Path('dbs/db_images') / path.name
        self.path_index[new_id] = str(db_path)
        shutil.copy(src_path, db_path)

    def save_state(self):
        """
        Save the current state of the database to disk if changes have been made.

        Behavior:
            - Checks if the database has been modified (indicated by `need_to_save`).
            - If so, saves the path index to a JSON file and writes the semantic and content FAISS indices to their respective files.
        """
        if self.need_to_save:
            with open('dbs/path_index.json', 'w') as f:
                json.dump(self.path_index, f)
            faiss.write_index(self.semantic_index, 'dbs/semantic_db.faiss')
            faiss.write_index(self.content_index, 'dbs/content_db.faiss')
