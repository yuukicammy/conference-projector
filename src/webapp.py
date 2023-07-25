# 1) Change CONFIG_FILE
# 2) Change PROD_STUB_NAME
# 3) Change CONFIG_FILE in webapp_image_builder.py

from pathlib import Path
from typing import Dict, Any, List, Tuple
import modal

from dash import html
import dash_bootstrap_components as dbc

from .config import Config, ProjectConfig
from .webapp_image_builder import build_modal_image
from .webapp_layout import Layout
from .webapp_data import ContainerData, WebappData

SHARED_ROOT = "/root/.cache"
CONFIG_FILE = "configs/defaults.toml"

PROD_STUB_NAME = "conference-projector"

modal_image = (
    modal.Image.debian_slim(force_build=False)
    .apt_install("git")
    .pip_install(
        [
            "flask",
            "dacite",
            "dash",
            "toml",
            "dash-bootstrap-components",
            "scipy",
            "scikit-learn",
            "umap-learn",
            "seaborn",
        ]
    )
    .run_function(
        build_modal_image,
        force_build=False,
        cpu=12,
        memory=10240,
        network_file_systems={
            SHARED_ROOT: modal.NetworkFileSystem.persisted(ProjectConfig._shared_vol)
        },
        mounts=[
            modal.Mount.from_local_dir(
                Path(__file__).parent.parent / "configs", remote_path="/root/configs"
            ),
            modal.Mount.from_local_dir(
                Path(__file__).parent.parent / "data/prompts",
                remote_path="/root/data/prompts",
            ),
        ],
    )
)

stub = modal.Stub(
    PROD_STUB_NAME if "prod" in ProjectConfig._name else ProjectConfig._stub_webapp,
    image=modal_image,
    mounts=[
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "configs", remote_path="/root/configs"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "assets", remote_path="/root/src/assets"
        ),
        modal.Mount.from_local_dir(
            Path(__file__).parent.parent / "data/prompts",
            remote_path="/root/data/prompts",
        ),
    ],
)

stub.cache = modal.Dict.new()

if stub.is_inside():
    from typing import Dict, Any, List, Tuple
    import dacite
    import toml
    import numpy as np
    import dash
    from dash import dcc
    from dash import html
    import dash_bootstrap_components as dbc
    from dash.dependencies import Input, Output, State
    import pandas as pd
    import plotly
    import plotly.express as px
    from scipy.spatial import cKDTree


@stub.cls(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.persisted(ProjectConfig._shared_vol)
    },
)
class DashApp:
    def __init__(self, config: Config):
        self.config = config
        self.container_data = ContainerData(config=self.config)
        self.layout = Layout(config=self.config, container_data=self.container_data)
        self.app_data = WebappData(
            config=self.config, container_data=self.container_data
        )

        app_description = self.config.webapp.web_description
        app_title = self.config.webapp.web_title
        app_image = self.config.webapp.web_icon

        metas = [
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"property": "twitter:card", "content": "summary_large_image"},
            {"property": "twitter:url", "content": "https://www.wealthdashboard.app/"},
            {"property": "twitter:title", "content": app_title},
            {"property": "twitter:description", "content": app_description},
            {"property": "twitter:image", "content": app_image},
            {"property": "og:title", "content": app_title},
            {"property": "og:type", "content": "website"},
            {"property": "og:description", "content": app_description},
            {"property": "og:image", "content": app_image},
        ]

        self.app = dash.Dash(
            __name__,
            meta_tags=metas,
            title=app_title,
            external_stylesheets=[
                dbc.icons.FONT_AWESOME,
                "https://codepen.io/chriddyp/pen/bWLwgP.css",
                "/root/assets/style.css",
                dbc.themes.BOOTSTRAP,
            ],
            # comment two following lines for local tests
            routes_pathname_prefix="/",
            requests_pathname_prefix="/",
            serve_locally=False,
        )
        self.app.layout = self.get_layout()

        # Update the figure and the entire screen.
        self.app.callback(
            [
                Output("url", "search"),
                Output("scatter-plot", "figure"),
                Output("graph-area", "style"),
                Output("details-row", "children"),
                Output("details-row", "style"),
                Output("viewall-nclicks", "n_clicks"),
                Output("scatter-plot", "clickData"),
            ],
            [
                Input("scatter-plot", "clickData"),
                Input("embeddings", "value"),
                Input("viewall-nclicks", "n_clicks"),
            ],
            [
                State("url", "search"),
                State("shared-data", "data"),
                State("url", "pathname"),
            ],
        )(self.update)

        # Updated paper description
        self.app.callback(
            [Output("recommendation", "children"), Output("shared-data", "data")],
            [Input("details-option", "value")],
            [State("url", "search"), State("url", "pathname")],
        )(self.update_details)

    @modal.method()
    def get_server(self):
        return self.app.server

    @modal.method()
    def get_layout(self) -> html.Div:
        return self.layout.screen_layout

    @modal.method()
    def k_nearest(self, index: int, feature_name: str) -> Tuple[List[float], List[int]]:
        if stub.app.cache.contains(
            f"indices-{str(index)}-{feature_name}"
        ) and stub.app.cache.contains(f"distances-{str(index)}-{feature_name}"):
            indices = stub.app.cache[f"indices-{str(index)}-{feature_name}"]
            distances = stub.app.cache[f"distances-{str(index)}-{feature_name}"]
        else:
            distances, indices = self.container_data.trees[feature_name].query(
                self.container_data.reduced_features[feature_name][index, :],
                k=self.config.webapp.num_neighborhoods,
                workers=self.config.project.num_workers,
            )
            stub.app.cache[f"indices-{str(index)}-{feature_name}"] = indices
            stub.app.cache[f"distances-{str(index)}-{feature_name}"] = distances
        return distances, indices

    @modal.method()
    def update_df_center_node(self, index: int, feature_name: str) -> List[int]:
        distances, indices = self.k_nearest(index=index, feature_name=feature_name)
        self.app_data.update_center(index=index, distances=distances, indices=indices)
        return indices

    @modal.method()
    @staticmethod
    def parse_search(search: str) -> Dict[str, str | int]:
        from urllib.parse import parse_qs

        res = {
            "node": None,
            "key": None,
            "method": None,
            "dim": None,
        }
        if search is None or len(search) == 0:
            return res
        parsed_query = parse_qs(search[1:])  # skip '?'
        if parsed_query.get("node"):
            res["node"] = int(parsed_query.get("node", [""])[0])
        else:
            res["node"] = None
        if parsed_query.get("e"):
            res["key"] = parsed_query.get("e", [""])[0]
        else:
            res["key"] = None
        if parsed_query.get("m"):
            res["method"] = parsed_query.get("m", [""])[0]
        else:
            res["method"] = None
        if parsed_query.get("d"):
            res["dim"] = int(parsed_query.get("d", [""])[0])
        else:
            res["dim"] = None
        return res

    def parse_pathname(self, pathname: str) -> Dict[str, str | bool]:
        # /cvpr2023/en or /en pr /ja or /cvpr2023/ja
        if pathname is None or len(pathname) == 0:
            return {"en_mode": False, "conference": ""}
        values = pathname.split("/")
        en_mode = False
        if "en" in values:
            en_mode = True
        conference = ""
        for option in self.config.webapp.conference_options:
            if option["value"] in values:
                conference = option["value"]
                break
        return {"en_mode": en_mode, "conference": conference}

    @modal.method()
    @staticmethod
    def make_search(
        node: int = None,
        key: str = None,
        method: str = None,
        dim: int = None,
    ):
        from urllib.parse import urlencode

        params = {
            "node": node,
            "e": key,
            "m": method,
            "d": dim,
        }
        filtered_params = {k: v for k, v in params.items() if v is not None}
        return "" if len(filtered_params) == 0 else "?" + urlencode(filtered_params)

    @modal.method()
    def update(
        self,
        clicked_data: Dict[str, Any],
        key: str,
        # method: str = "umap",
        # dim: int = 2,
        view_all_nclicks: int = 0,
        search: str = None,
        shared_data: Dict[str, Any] = {},
        pathname: str = None,
    ):
        method = "umap"
        dim = 2
        path_dict = self.parse_pathname(pathname=pathname)
        en_mode = path_dict["en_mode"]

        if 0 < view_all_nclicks:
            # Show all nodes
            return (
                "",
                self.layout.default_figure(
                    df=self.app_data.df,
                    key=key,
                    method=method,
                    dim=dim,
                    en_mode=en_mode,
                ),
                {
                    "width": "100%",
                    "minWidth": "700px",
                    "minHeight": "400px",
                },
                None,
                {"width": "0%"},
                0,
                None,
            )

        feature_name = f"{key}_{method}_{str(dim)}"
        print(f"update_chart {feature_name}")

        params = self.parse_search(search=search)

        if isinstance(shared_data, dict) and "options" in shared_data.keys():
            options = shared_data["options"]
        else:
            options = (
                self.container_data.default_options_en
                if en_mode
                else self.container_data.default_options
            )

        if (
            clicked_data is not None
            and "points" in clicked_data.keys()
            and 0 < len(clicked_data["points"])
        ):
            print(clicked_data)
            index = int(clicked_data["points"][0]["customdata"][-1])
            indices = self.update_df_center_node(index=index, feature_name=feature_name)
            return (
                self.make_search(
                    node=index,
                    key=key,
                    method=method,
                    dim=dim,
                ),
                self.layout.selected_figure(
                    df=self.app_data.df,
                    key=key,
                    method=method,
                    dim=dim,
                    indices=indices,
                ),
                {
                    "width": self.config.webapp.width_figure,
                },
                dbc.Row(
                    [
                        self.layout.selected_paper_block(
                            df=self.app_data.df, index=index, en_mode=en_mode
                        ),
                        self.layout.recommendation_block(
                            options=options,
                            all_options=self.container_data.info_opts_en
                            if en_mode
                            else self.container_data.info_opts,
                        ),
                    ]
                ),
                {"width": self.config.webapp.width_details},
                dash.no_update,
                None,
            )

        if params["key"] != key or params["method"] != method or params["dim"] != dim:
            print(search)
            search = self.make_search(
                node=params["node"],
                key=key,
                method=method,
                dim=dim,
            )
            print(search)
            if params["node"] is None:
                return (
                    search,
                    self.layout.default_figure(
                        df=self.app_data.df,
                        key=key,
                        method=method,
                        dim=dim,
                        en_mode=en_mode,
                    ),
                    {
                        "width": "100%",
                        "minWidth": "700px",  # Set the minimum width of the graph container
                        "minHeight": "400px",
                    },
                    None,
                    {"width": "0%"},
                    dash.no_update,
                    dash.no_update,
                )
            else:
                indices = self.update_df_center_node(
                    index=params["node"], feature_name=feature_name
                )
                return (
                    search,
                    self.layout.selected_figure(
                        df=self.app_data.df,
                        key=key,
                        method=method,
                        dim=dim,
                        indices=indices,
                    ),
                    {
                        # "width": self.config.webapp.width_figure,
                        "minWidth": "400px",  # Set the minimum width of the graph container
                        "minHeight": "400px",
                    },
                    dbc.Row(
                        [
                            self.layout.selected_paper_block(
                                df=self.app_data.df,
                                index=params["node"],
                                en_mode=en_mode,
                            ),
                            self.layout.recommendation_block(
                                options=options,
                                all_options=self.container_data.info_opts_en
                                if en_mode
                                else self.container_data.info_opts,
                            ),
                        ]
                    ),
                    {
                        "width": self.config.webapp.width_details,
                        "minWidth": "700px",  # Set the minimum width of the graph container
                        "minHeight": "700px",
                    },
                    dash.no_update,
                    dash.no_update,
                )
        print("NO update")
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    @modal.method()
    def update_details(
        self, options: List[str] = None, search: str = None, pathname: str = None
    ):
        print("update_details")
        try:
            params = self.parse_search(search=search)
            en_mode = True if pathname.startswith("/en") else False
            index = params["node"]
            key = params["key"]
            method = params["method"]
            dim = params["dim"]
            feature_name = f"{key}_{method}_{str(dim)}"
            _, indices = self.k_nearest(index=index, feature_name=feature_name)
            return [
                self.layout.description_block(
                    df=self.app_data.df,
                    indices=indices[1:],
                    options=options,
                    start_rank=1,
                    im_width="50%",
                    en_mode=en_mode,
                )
            ], {"options": options}
        except Exception as e:
            print(e)
            return [dbc.Row()], {"options": options}


@stub.function(
    network_file_systems={
        SHARED_ROOT: modal.NetworkFileSystem.persisted(ProjectConfig._shared_vol)
    },
    # cpu=1,
    memory=5120,
    # keep_warm=5,
)
@modal.wsgi_app()
def wrapper():
    import dacite
    import toml

    print(CONFIG_FILE)
    config = dacite.from_dict(data_class=Config, data=toml.load(CONFIG_FILE))

    return DashApp(config=config).get_server()
