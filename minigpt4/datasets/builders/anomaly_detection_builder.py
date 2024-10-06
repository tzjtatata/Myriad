import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.anomaly_detection import AnomalyDetectionDataset
from minigpt4.datasets.datasets.twocls_ad import TwoClassAnomalyDetectionDataset


@registry.register_builder("anomaly_detection")
class AnomalyDetectionBuilder(BaseDatasetBuilder):
    train_dataset_cls = AnomalyDetectionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anomaly_detection/base.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        ve_root = build_info.ve_storage
        ann_paths = build_info.get('ann_paths', ['DC_VISA_train_normal.jsonl'])
        is_preload = self.config.get("is_preload", False)
        nsa_max_width = self.config.augment.get("nsa_max_width", 0.4)

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            img_size=self.config.get("img_size", 224), 
            crop_size=self.config.get("crop_size", 224), 
            vis_root=storage_path,
            ve_root=ve_root,
            version=self.config.get("version", 0), 
            with_mask=self.config.with_mask,
            with_pos=self.config.get("with_pos", False), 
            with_ref=self.config.with_ref,
            is_preload=is_preload, 
            nsa_max_width=nsa_max_width, 
        )

        return datasets


@registry.register_builder("two_class_anomaly_detection")
class TwoClassAnomalyDetectionBuilder(BaseDatasetBuilder):
    train_dataset_cls = TwoClassAnomalyDetectionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/anomaly_detection/2cls.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        ann_paths = build_info.get('ann_paths', ['DC_MVTEC_train_2cls.jsonl'])
        is_preload = self.config.get("is_preload", False)

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=storage_path,
            dynamic_instruction=self.config.get("dynamic_instruction", False), 
            version=self.config.get("version", '1'), 
            img_size=build_info.get("img_size", 224), 
            crop_size=build_info.get("crop_size", 224), 
            is_preload=is_preload, 
        )

        return datasets

