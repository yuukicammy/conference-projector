from dataclasses import dataclass, field

from typing import List, Any, Dict
import json
from pathlib import Path


def stab_dict_factory(items: list[tuple[str, str]]) -> dict[str, str]:
    """dict_factory for stabs"""
    adict = {}
    for key, value in items:
        adict[key] = value

    return adict


@dataclass
class ProjectConfig:
    _name: str = "paper-viz"
    _shared_vol: str = "paper-viz-vol"
    _stab_embedding: str = f"{_name}-embedding"
    _stab_html_parser: str = f"{_name}-html-parser"
    _stab_paper_image: str = f"{_name}-extract-paper-image"
    _stab_summary: str = f"{_name}-summary"
    _stab_tfboard_webapp: str = f"{_name}-tfboard-webapp"
    _stab_pipeline: str = f"{_name}-pipeline"
    _stab_webapp: str = f"{_name}-webapp"
    _stab_db: str = f"{_name}-db"
    _stab_test: str = f"{_name}-test"
    stab_names: List[str] = field(default_factory=list)
    stab_files: List[str] = field(default_factory=list)
    num_workers: int = 0
    max_papers: int = None
    dataname: str = "cvpr2023"


@dataclass
class PipelineConfig:
    deplpoy_stubs: bool = True
    download_data_locally: bool = True
    initialize_volume: bool = False
    run_embed: bool = True
    run_html_parse: bool = True
    run_paper_image: bool = True
    run_summarize: bool = True


@dataclass
class MedatadaFileConfig:
    local_output_dir: str = "data"
    json_file: str = ""
    save_json: bool = False
    json_indent: int = 4
    tsv_file: str = "test.tsv"
    embeddings_files: Dict[str, str] = field(default_factory=dict)

    # The following file paths are defined in __post_init__()
    reduced_feature_file: str = ""
    papers_file: str = "" 
    data_frame_file: str = ""

    image_name_width: int = 4
    image_max_size: int = 1000
    force_extract_image: bool = False


@dataclass
class HTMLParserConfig:
    base_url: str = field(default_factory=str)
    path_papers: str = field(default_factory=str)
    suffix_abst: str = field(default_factory=str)
    suffix_item: str = field(default_factory=str)
    suffix_pdf: str = field(default_factory=str)
    suffix_title: str = field(default_factory=str)
    prefix_abst: str = field(default_factory=str)
    prefix_item: str = field(default_factory=str)
    prefix_pdf: str = field(default_factory=str)
    prefix_title: str = field(default_factory=str)
    prefix_arxiv: str = field(default_factory=str)
    suffix_arxiv: str = field(default_factory=str)


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
    title: str = "Papers Projector: CVPR 2023"
    num_neighborhoods: int = 0
    num_text_nodes: int = 0

    max_chars_long: int = 100
    max_chars_short: int = 100
    max_hight: str = "1000px"

    num_colors: int = 2000
    size_title: int = 20
    size_code: int = 12
    size_default: int = 12
    size_fig_title_large: int = 16
    size_fig_title_small: int = 12
    margin_title_bottom: str = "20px"
    margine_default: str = "10px"
    color_selected: str = "#895b8a"
    color_not_selected: str = "#c099a0"
    color_fig_title: str = ""
    node_size_default: int = 8
    node_symbol_clicked: str = "circle"
    node_symbol_default: str = "circle"
    node_symbol_selected: str = "circle"
    text_details_default: str = ""
    embedding_options: list = field(default_factory=list)
    dimension_options: list = field(default_factory=list)
    dimension_reduction_options: list = field(default_factory=list)
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
    init_cache: bool = False
    width_figure: str = "60%"
    width_details: str = "40%"
    web_title:str=""
    web_description:str=""
    web_icon:str = ""
    title_url:str=""

@dataclass
class DBConfig:
    uri: str = ""
    database_id: str = "cvpr2023"
    container_id: str = "Container-01"


@dataclass
class Config:
    project: ProjectConfig
    pipeline: PipelineConfig
    files: MedatadaFileConfig
    html_parser: HTMLParserConfig
    embedding: EmbeddingConfig
    summary: SummaryConfig
    webapp: WebAppConfig
    db: DBConfig

    def __post_init__(self):
        self.files.json_path = f"{self.project.dataname}/{self.project.dataname}.json"
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

        self.project.stab_names = []
        for key, value in vars(self.project).items():
            if key.startswith("_stab"):
                self.project.stab_names.append(value)

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
