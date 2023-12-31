from typing import Dict, Any, List, Tuple
import json
import pickle
from pathlib import Path

from PIL import Image

from .config import Config


class ContainerData:
    """Static data class.
    Once loaded, the values of this class is never changed.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.conference_options = config.webapp.conference_options
        self.legend_orders = {"award_label": self.config.webapp.award_labels}
        print("legend_orders: ", self.legend_orders)

        self.default_options = [
            "essence_en",
            "essence_ja",
        ]
        self.info_opts = [
            {"label": "Abstract", "value": "abstract"},
            {"label": "Image", "value": "image"},
        ]
        self.info_order = ["abstract"]
        with open(config.summary.function_schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
            self.properties = schema["parameters"]["properties"]
            for key in self.properties.keys():  # this keys are not optional.
                if key in [
                    "task_en",
                    "topic_en",
                    "application_en",
                    "task_ja",
                    "topic_ja",
                    "application_ja",
                ]:
                    continue
                self.info_opts.append(
                    {"label": self.properties[key]["description"], "value": key}
                )
                self.info_order.append(key)
            del schema
        self.info_opt_vals = [d["value"] for d in self.info_opts]

        # Remove Japanese options for the English page.
        self.default_options_en = [
            "advantages_en",
            "essence_en",
        ]
        self.info_opts_en = list(
            filter(lambda d: d.get("value").endswith("ja") == False, self.info_opts)
        )
        self.info_opts_vals_en = list(
            filter(lambda d: d.endswith("ja") == False, self.info_opt_vals)
        )

        print("Loading reduced_features")
        with open(self.config.files.reduced_feature_file, "rb") as f:
            self.reduced_features = pickle.load(f)
            for val in self.reduced_features.values():
                val = val[: self.config.project.max_papers, :]

        print("Loading trees")
        with open(
            Path(self.config.files.reduced_feature_file).parent
            / f"trees-{self.config.project.max_papers}",
            "rb",
        ) as f:
            self.trees = pickle.load(f)

        print("Loading papers")
        with open(self.config.files.papers_file, "rb") as f:
            self.papers = pickle.load(f)
        if self.config.project.max_papers < len(self.papers):
            self.papers = self.papers[: self.config.project.max_papers]
        self.num_data = len(self.papers)

    def get_image(self, index: int) -> Image.Image | None:
        if index < self.num_data and self.papers[index].get("image"):
            return self.papers[index]["image"]
        else:
            return None


class WebappData:
    def __init__(self, config: Config, container_data: ContainerData) -> None:
        self.config = config
        self.container_data = container_data

        print("Loading data_frame")
        with open(self.config.files.data_frame_file, "rb") as f:
            self.df = pickle.load(f)
            if self.config.project.max_papers < len(self.df):
                self.df = self.df.head(self.config.project.max_papers)

    def update_center(
        self, index: int, distances: List[float], indices: List[int]
    ) -> None:
        print("k-nearest indices: ", indices)
        max_dist = max(distances)
        self.df[self.config.webapp.label_distance] = [
            max_dist * 1.1
        ] * self.container_data.num_data
        self.df["text"] = [""] * self.container_data.num_data

        for i in range(len(indices)):
            dist = distances[i]
            idx = indices[i]
            self.df.at[idx, self.config.webapp.label_distance] = dist

            if i == 0:
                self.df.at[idx, "text"] = "👇"
            elif i < self.config.webapp.num_text_nodes:
                self.df.at[idx, "text"] = str(i)

            if self.df.at[idx, "award"] is not None and 0 < len(
                self.df.at[idx, "award"]
            ):
                if i < self.config.webapp.num_text_nodes:
                    self.df.at[idx, "text"] = (
                        self.df.at[idx, "award"] + "<br>" + self.df.at[idx, "text"]
                    )
                else:
                    self.df.at[idx, "text"] = self.df.at[idx, "award"]
