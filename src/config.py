from dataclasses import dataclass, field

from typing import List, Any, Dict
import json
from pathlib import Path


def stub_dict_factory(items: list[tuple[str, str]]) -> dict[str, str]:
    """dict_factory for stubs"""
    adict = {}
    for key, value in items:
        adict[key] = value

    return adict


@dataclass
class ProjectConfig:
    _name: str = "cp-stg"
    _shared_vol: str = f"{_name}-vol"
    _stub_embedding: str = f"{_name}-embedding"
    _stub_scraper: str = f"{_name}-scraper"
    _stub_paper_image: str = f"{_name}-extract-paper-image"
    _stub_summary: str = f"{_name}-summary"
    _stub_highlights: str = f"{_name}-highlights"
    _stub_pipeline: str = f"{_name}-pipeline"
    _stub_webapp: str = f"{_name}-webapp"
    _stub_db: str = f"{_name}-db"
    _stub_test: str = f"{_name}-test"
    stub_names: List[str] = field(default_factory=list)
    stub_files: List[str] = field(default_factory=list)
    num_workers: int = 0
    max_papers: int = None
    dataname: str = ""
    config_file: str = ""


@dataclass
class PipelineConfig:
    deploy_stubs: bool = True
    deploy_webapp: str = ""
    stop_stubs: bool = False
    download_data_locally: bool = True
    initialize_volume: bool = False
    run_embed: bool = True
    run_scrape: bool = True
    run_paper_image: bool = True
    run_summarize: bool = True


@dataclass
class MedatadaFileConfig:
    local_output_dir: str = "data"
    json_file: str = ""
    save_json: bool = False
    json_indent: int = 4
    embeddings_files: Dict[str, str] = field(default_factory=dict)

    image_name_width: int = 4
    image_max_size: int = 1000
    force_extract_image: bool = False

    # The following file paths are defined in __post_init__()
    reduced_feature_file: str = ""
    papers_file: str = ""
    data_frame_file: str = ""


@dataclass
class ScraperConfig:
    base_url: str = field(default_factory=str)
    path_papers: str = field(default_factory=str)

    # For award/highlight papers
    award_details_url: str = ""
    award: list = field(default_factory=list)
    img_ignore_paths: list = field(default_factory=list)
    img_base_url: str = ""


@dataclass
class EmbeddingConfig:
    batch_size: int = 20
    model: str = field(default_factory=str)
    retry: int = 0
    keys: List[str] = field(default_factory=list)


@dataclass
class SummaryConfig:
    model: str = field(default_factory=str)
    prompt_file: str = field(default_factory=str)
    retry: int = 0
    function_schema_file: str = field(default_factory=str)
    prompt: str = field(default=str)
    function_schema: Dict[str, Any] = field(default_factory=dict)
    sleep: float = 5


@dataclass
class WebAppConfig:
    init_trees: bool = False
    init_papers: bool = False

    # Trees
    num_neighborhoods: int = 0

    # Website View
    title: str = "Papers Projector: CVPR 2023"
    title_url: str = ""

    text_details_default: str = ""
    label_embeddings: str = ""
    label_dimension: str = ""
    label_projection_algorithm: str = ""
    text_recommendation_description: str = ""
    text_selection_description: str = ""
    label_options: str = ""
    label_distance: str = ""
    text_concern_description: str = ""
    text_top_description: str = ""
    text_figure_title_format: str = ""

    num_text_nodes: int = 0
    num_colors: int = 2000

    max_chars_long: int = 100
    max_chars_short: int = 100
    max_hight: str = "1000px"

    size_title: int = 20
    size_code: int = 12
    size_default: int = 12
    size_fig_title_large: int = 16
    size_fig_title_small: int = 12

    margin_title_bottom: str = "20px"
    margin_default: str = "10px"

    color_selected: str = ""
    color_not_selected: str = ""
    color_fig_title: str = ""

    node_size_default: int = 8
    node_symbol_clicked: str = "circle"
    node_symbol_default: str = "circle"
    node_symbol_selected: str = "circle"

    embedding_options: list = field(default_factory=list)
    dimension_options: list = field(default_factory=list)
    dimension_reduction_options: list = field(default_factory=list)

    width_figure: str = "60%"
    width_details: str = "40%"

    # For thumbnails
    web_title: str = ""
    web_description: str = ""
    web_icon: str = ""


@dataclass
class DBConfig:
    uri: str = ""
    database_id: str = "cvpr2023"
    container_id: str = "Container-01"


@dataclass
class Config:
    project: ProjectConfig
    pipeline: PipelineConfig
    scraper: ScraperConfig
    files: MedatadaFileConfig
    embedding: EmbeddingConfig
    summary: SummaryConfig
    webapp: WebAppConfig
    db: DBConfig

    def __post_init__(self):
        if self.project.max_papers < 0:
            self.project.max_papers = 2**31 - 1
        with open(self.summary.prompt_file, "r", encoding="utf-8") as f:
            self.summary.prompt = f.read()
        with open(self.summary.function_schema_file, "r", encoding="utf-8") as f:
            # clear formatting
            self.summary.function_schema = json.load(f)

        self.files.reduced_feature_file = (
            self.project.dataname + "/" + "reduced_features.pickle"
        )
        self.files.papers_file = self.project.dataname + "/" + "papers.pickle"
        self.files.data_frame_file = self.project.dataname + "/" + "data_frame.pickle"

        for key in self.embedding.keys:
            self.files.embeddings_files[key] = self.embedding_path(label=key)

        self.project.stub_names = []
        for key, value in vars(self.project).items():
            if key.startswith("_stub"):
                self.project.stub_names.append(value)

    def embedding_path(self, label: str) -> str:
        return str(
            Path(self.project.dataname)
            / (
                label
                + "-"
                + self.project.dataname
                + "-"
                + self.embedding.model
                + ".npy"
            )
        )

    def reduced_feature_path(self, label: str, method: str, dim: int) -> str:
        return str(
            Path(self.project.dataname)
            / (
                label
                + "-"
                + self.project.dataname
                + "-"
                + self.embedding.model
                + "-"
                + method
                + "-"
                + f"{str(dim)}d"
                + ".npy"
            )
        )
