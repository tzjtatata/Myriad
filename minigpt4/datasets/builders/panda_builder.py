import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.panda_instructions import PandaInstructionDataset


@registry.register_builder("panda")
class PandaBuilder(BaseDatasetBuilder):
    train_dataset_cls = PandaInstructionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/panda/base.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")

        build_info = self.config.build_info
        storage_path = build_info.storage
        meta_path = build_info.get('meta_path', '/mnt/vdb1/datasets/anomalyGPT_dataset/pandagpt4_visual_instruction_data.json')
        is_preload = self.config.get("is_preload", False)

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            meta_path=meta_path,
            image_root_path=storage_path,
        )

        return datasets
