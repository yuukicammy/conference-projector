from pathlib import Path
from typing import Dict, Any, List, Tuple
import modal
import numpy as np
from PIL import Image
from .config import Config
from .utils import insert_line_breaks

SHARED_ROOT = "/root/.cache"
CONFIG_FILE = "configs/debug.toml"


class ModalImageBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.reduced_feature_file = (
            Path(SHARED_ROOT) / self.config.files.reduced_feature_file
        )
        self.trees_file = (
            Path(SHARED_ROOT) / self.config.files.reduced_feature_file
        ).parent / f"trees-{self.config.project.max_papers}.pickle"
        self.papers_file = Path(SHARED_ROOT) / self.config.files.papers_file
        self.data_frame_file = Path(SHARED_ROOT) / config.files.data_frame_file
        self.reduced_features = None
        self.trees = None
        self.papers = None
        self.data_frame = None

    def setup_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up the paper dictionary by adding formatted titles, categories, and applications, as well as loading the image.

        Args:
            paper (Dict[str, Any]): The paper dictionary.

        Returns:
            Dict[str, Any]: The updated paper dictionary.
        """
        updated_paper = paper.copy()

        updated_paper["short_title"] = insert_line_breaks(
            text=updated_paper["title"],
            max_chars_per_line=self.config.webapp.max_chars_short,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )
        updated_paper["long_title"] = insert_line_breaks(
            text=updated_paper["title"],
            max_chars_per_line=self.config.webapp.max_chars_long,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )
        updated_paper["short_category"] = insert_line_breaks(
            text=updated_paper["category_en"],
            max_chars_per_line=self.config.webapp.max_chars_short,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )
        updated_paper["long_category"] = insert_line_breaks(
            text=updated_paper["category_en"],
            max_chars_per_line=self.config.webapp.max_chars_long,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )
        updated_paper["short_application"] = insert_line_breaks(
            text=updated_paper["application_en"],
            max_chars_per_line=self.config.webapp.max_chars_short,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )
        updated_paper["long_application"] = insert_line_breaks(
            text=updated_paper["application_en"],
            max_chars_per_line=self.config.webapp.max_chars_long,
            prefix="<br>",
            suffix="<br>",
            newline="<br>",
        )

        if 0 < len(paper["award"]):
            # For details description
            updated_paper["award_label"] = paper["award"]
            # For scatter plot
            if paper["award"] == "Highlight" or paper["award"] == "Award Candidate":
                updated_paper["award_text"] = ""
            else:
                updated_paper["award_text"] = paper["award"]
        else:
            updated_paper["award_label"] = "None"
            updated_paper["award_text"] = ""

        if "image_path" in paper.keys() and 0 < len(paper["image_path"]):
            updated_paper["image"] = self.load_image(image_path=paper["image_path"])

        return updated_paper

    def reduction(self, embeddings: np.ndarray, method: str, dim: int):
        """
        Reduce the dimensionality of embeddings using a specified method.

        Args:
            embeddings (np.ndarray): The input embeddings to be reduced in dimensionality.
            method (str): The dimensionality reduction method to be used. Options: "pca", "tsne", "umap".
            dim (int): The number of dimensions to reduce to.

        Returns:
            np.ndarray: The embeddings with reduced dimensionality.

        Raises:
            ValueError: If an unknown method is specified.

        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from umap import UMAP

        if method == "pca":
            reducer = PCA(n_components=dim)
        elif method == "tsne":
            reducer = TSNE(
                n_components=dim, perplexity=self.config.webapp.num_neighborhoods * 2
            )
        elif method == "umap":
            reducer = UMAP(
                n_components=dim,
                output_metric="manhattan",
                n_neighbors=self.config.webapp.num_neighborhoods * 2,
            )
        else:
            raise ValueError("Specified an unknown method to reduce dimensions.")
        return reducer.fit_transform(embeddings)

    def load_image(self, image_path: str) -> Image.Image | None:
        Image.MAX_IMAGE_PIXELS = None
        try:
            path = Path(SHARED_ROOT) / image_path
            if path.is_file():
                with Image.open(path) as img:
                    if 50 < min(img.size[0], img.size[1]):
                        img = img.resize(
                            (400, int((400.0 / img.size[0]) * img.size[1]))
                        )
                        return img
            return None
        except Exception as e:
            print(f"cannot load image from {image_path}. \n{e}")
            return None

    def load_pickle(self, path: Path):
        import pickle

        if path.is_file():
            with open(path, "rb") as f:
                obj = pickle.load(f)
                return obj
        return None

    def create_features_and_trees(self) -> None:
        from scipy.spatial import cKDTree

        trees = {}
        reduced_features = {}
        for key, path in self.config.files.embeddings_files.items():
            path = Path(SHARED_ROOT) / path
            if not path.exists():
                raise Exception(f"Embedding file does not exist. {path}")
            print(f"Loading a file from {path}...")
            embeddings = np.load(path)
            print(f"Done.")
            for method in self.config.webapp.dimension_reduction_options:
                for dim in self.config.webapp.dimension_options:
                    feature_name = f'{key}_{method["value"]}_{str(dim["value"])}'
                    print(f"Creating projected features: {feature_name}")
                    feature = self.reduction(
                        embeddings=embeddings[: self.config.project.max_papers, :],
                        method=method["value"],
                        dim=int(dim["value"]),
                    )
                    reduced_features[feature_name] = feature
                    trees.update({feature_name: cKDTree(feature)})
        return reduced_features, trees

    def setup_papers(self):
        import concurrent

        print("Loading papers from Azure Cosmos.")
        print(self.config.db)
        papers = modal.Function.lookup(
            self.config.project._stub_db, "get_all_papers"
        ).call(self.config.db, self.config.project.max_papers, force=True)
        print(f"Done. The num of papers: {len(papers)}.")

        print("Loading image files...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.project.num_workers
        ) as executor:
            futures = [executor.submit(self.setup_paper, paper) for paper in papers]
            for future in concurrent.futures.as_completed(futures):
                paper = future.result()
                papers[int(paper["id"])] = paper
        return papers

    def create_data_frame(self):
        import pickle
        import pandas as pd

        print("Makign data_frame...")
        assert 0 < len(self.papers)
        assert self.reduced_features is not None
        data_frame = {}
        for key in self.papers[0].keys():
            if key != "image" and key[0] != "_":
                data_frame[key] = []
        for key in self.papers[0].keys():
            if key != "image" and key[0] != "_":
                data_frame[key] = [paper[key] for paper in self.papers]
        print("Converted papers to data_frame.")

        for key in self.config.files.embeddings_files.keys():
            for method in self.config.webapp.dimension_reduction_options:
                for dim in self.config.webapp.dimension_options:
                    feature_name = f'{key}_{method["value"]}_{str(dim["value"])}'
                    data_frame[f"{feature_name}_x"] = self.reduced_features[
                        feature_name
                    ][:, 0]
                    data_frame[f"{feature_name}_y"] = self.reduced_features[
                        feature_name
                    ][:, 1]
                    if int(dim["value"]) == 3:
                        data_frame[f"{feature_name}_z"] = self.reduced_features[
                            feature_name
                        ][:, 2]

        for key, value in data_frame.items():
            print(f"{key} length: {len(value)}")
        data_frame = pd.DataFrame(data=data_frame)
        return data_frame

    def build(self):
        import pickle
        import pandas as pd

        self.reduced_features = (
            None
            if self.config.webapp.init_trees
            else self.load_pickle(self.reduced_feature_file)
        )
        self.trees = (
            None if self.config.webapp.init_trees else self.load_pickle(self.trees_file)
        )
        if self.reduced_features is None or self.trees is None:
            self.reduced_features, self.trees = self.create_features_and_trees()
            # Save in SharedVolume
            with open(self.reduced_feature_file, "wb") as f:
                pickle.dump(self.reduced_features, f)
            with open(self.trees_file, "wb") as f:
                pickle.dump(self.trees, f)
            print(
                "Done preparation for features and trees. Saved them into SharedVolume."
            )
        else:
            print(f"Done loading reduced_features and trees.")
            print(f"reduced_features: type {type(self.reduced_features)}.")
            print(f"trees: type {type(self.trees)}.")

        self.papers = (
            None
            if self.config.webapp.init_papers
            else self.load_pickle(self.papers_file)
        )
        if self.papers is None:
            self.papers = self.setup_papers()
            with open(self.papers_file, "wb") as f:
                pickle.dump(self.papers, f)
            print("Done preparation for papers. Saved it into SharedVolume.")
        else:
            print(
                f"Done loading papers with type {type(self.papers)} and size {len(self.papers)}."
            )
        if not (self.config.webapp.init_trees or self.config.webapp.init_papers):
            self.data_frame = self.load_pickle(self.data_frame_file)
        else:
            self.data_frame = None
        if self.data_frame is None:
            self.data_frame = self.create_data_frame()
            with open(
                self.data_frame_file,
                "wb",
            ) as f:
                pickle.dump(self.data_frame, f)
            print("Done preparation for data_frame. Saved it into SharedVolume.")
        else:
            print(
                f"Done loading data_frame with type {type(self.data_frame)} and size {len(self.data_frame)}."
            )
        # for col in self.data_frame.columns:
        #     print(f"{col}: {len(self.data_frame[col])}")


def build_modal_image():
    import pickle
    import dacite
    import toml
    import numpy as np
    import pandas as pd
    from PIL import Image

    print("Building Modal Image...")
    print(CONFIG_FILE)
    config = dacite.from_dict(data_class=Config, data=toml.load(CONFIG_FILE))

    builder = ModalImageBuilder(config=config)
    builder.build()

    # Make a directory into the Image.
    Path(config.files.reduced_feature_file).parent.mkdir(parents=True, exist_ok=True)

    # Store large size data in Image instead of SharedVolume.
    with open(config.files.reduced_feature_file, "wb") as f:
        pickle.dump(builder.reduced_features, f)
    with open(
        Path(config.files.reduced_feature_file).parent
        / f"trees-{config.project.max_papers}",
        "wb",
    ) as f:
        pickle.dump(builder.trees, f)
    with open(config.files.papers_file, "wb") as f:
        pickle.dump(builder.papers, f)
    with open(config.files.data_frame_file, "wb") as f:
        pickle.dump(builder.data_frame, f)

    print("Done building the image.")
