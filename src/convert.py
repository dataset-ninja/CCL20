# https://www.kaggle.com/datasets/downloader007/ccl20

import os
import shutil
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
    mkdir,
    remove_dir,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:


    # project_name = "Citrus Leaf Diseases"
    dataset_path = "/home/grokhi/rawdata/ccl20/CCL'20 dataset"
    images_ext = ".jpg"
    masks_ext = ".xml"
    batch_size = 30


    def create_ann(image_path):
        labels = []

        ann_path = os.path.join(curr_ds_path, get_file_name(image_path) + ".xml")

        tree = ET.parse(ann_path)
        root = tree.getroot()
        img_height = int(root.find(".//height").text)
        img_width = int(root.find(".//width").text)
        objects_content = root.findall(".//object")
        for obj_data in objects_content:
            name = obj_data.find(".//name").text
            bndbox = obj_data.find(".//bndbox")
            top = int(bndbox.find(".//ymin").text)
            left = int(bndbox.find(".//xmin").text)
            bottom = int(bndbox.find(".//ymax").text)
            right = int(bndbox.find(".//xmax").text)

            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            obj_class = name_to_class.get(name)
            label = sly.Label(rectangle, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_width), labels=labels)


    idx_to_obj_class = {}
    obj_class_anthracnose = sly.ObjClass("anthracnose", sly.Rectangle)
    obj_class_melanose = sly.ObjClass("melanose", sly.Rectangle)
    obj_class_brown_spot = sly.ObjClass("brown Spot", sly.Rectangle)

    name_to_class = {
        "Disease-A": obj_class_anthracnose,
        "Disease-B": obj_class_brown_spot,
        "Disease-C": obj_class_melanose,
    }

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class_anthracnose, obj_class_brown_spot, obj_class_melanose]
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):
        if ds_name in ["2020", "2021"]:
            continue
        curr_ds_path = os.path.join(dataset_path, ds_name)

        if dir_exists(curr_ds_path):
            dataset = api.dataset.create(project.id, ds_name.lower(), change_name_if_conflict=True)
            images_names = [
                im_name for im_name in os.listdir(curr_ds_path) if get_file_ext(im_name) == images_ext
            ]

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [os.path.join(curr_ds_path, im_name) for im_name in img_names_batch]

                img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                if ds_name in ["Test", "train", "val"]:
                    anns = [create_ann(image_path) for image_path in img_pathes_batch]
                    api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(img_names_batch))
    return project


