from typing import Dict, Any, List, Tuple

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly
import plotly.express as px
from dash.dependencies import Input, Output, State

from .config import Config
from .webapp_data import ContainerData
from .utils import insert_line_breaks


class Layout:
    def __init__(self, config: Config, container_data: ContainerData) -> None:
        self.config = config
        self.container_data = container_data

    def paper_title_block(
        self, df: pd.core.frame.DataFrame, index: int, rank: int
    ) -> dbc.Row:
        title_text = (
            f'{str(rank)}.  {df.at[index, "title"]}'
            if 0 < rank
            else f'{df.at[index, "title"]}'
        )
        return dbc.Row(
            [
                dbc.Col(
                    dcc.Link(
                        title_text,
                        href=df.at[index, "pdf_url"],
                        id="paper-title",
                        target="_blank",
                        style={
                            "fontSize": 16,
                            "margin-bottom": "5px",
                        },  # "color": "#316745",
                    ),
                    width="90%",
                ),
            ]
        )

    def paper_info_base(
        self, df: pd.core.frame.DataFrame, index: int, en_mode: bool = False
    ) -> dbc.Row:
        contents = [
            dbc.Col(
                dcc.Markdown(
                    f'#### {self.container_data.properties["task_en"]["description"]}\n'
                    f'{df.at[index, "task_en"]}\n'
                    f'#### {self.container_data.properties["topic_en"]["description"]}\n'
                    f'{df.at[index, "topic_en"]}\n'
                    f'#### {self.container_data.properties["application_en"]["description"]}\n'
                    f'{df.at[index, "application_en"]}'
                )
            ),
        ]
        if not en_mode:
            contents.append(
                dbc.Col(
                    dcc.Markdown(
                        f'#### {self.container_data.properties["task_ja"]["description"]}\n'
                        f'{df.at[index, "task_ja"]}\n'
                        f'#### {self.container_data.properties["topic_ja"]["description"]}\n'
                        f'{df.at[index, "topic_ja"]}\n'
                        f'#### {self.container_data.properties["application_ja"]["description"]}\n'
                        f'{df.at[index, "application_ja"]}'
                    )
                )
            )
        return dbc.Row(contents)

    def paper_block(
        self,
        df: pd.core.frame.DataFrame,
        index: int,
        rank: int,
        options: List[str],
        en_mode: bool = False,
    ) -> dbc.Row:
        title_block = self.paper_title_block(df=df, index=index, rank=rank)
        info_base = self.paper_info_base(df=df, index=index, en_mode=en_mode)

        description = ""
        for key in self.container_data.info_order:
            if key in options:
                if key == "abstract":
                    description += f"#### Abstract\n"
                    description += f'{df.at[index, "abstract"]}\n'
                if key in self.container_data.properties.keys():
                    description += (
                        f'#### {self.container_data.properties[key]["description"]}\n'
                    )
                    description += f"{df.at[index, key]}\n"
        if len(df.at[index, "award"]) == 0:
            return dbc.Row([title_block, info_base, dcc.Markdown(description)])
        else:
            award = dbc.Row(
                dcc.Markdown(f'#### ★ {df.at[index, "award"]}'),
                style={"color": "#824880"},
            )
            return dbc.Row([title_block, award, info_base, dcc.Markdown(description)])

    def description_block(
        self,
        df: pd.core.frame.DataFrame,
        indices: List[int],
        options: List[str],
        start_rank: int,
        im_width="100%",
        en_mode: bool = False,
    ) -> dbc.Row:
        content = []
        if indices is None or len(indices) == 0:
            content = [dcc.Markdown(f"### {self.config.webapp.text_top_description}")]
        else:
            for i, index in enumerate(indices):
                rank = i + start_rank
                if index < self.container_data.num_data:
                    info = self.paper_block(
                        df=df, index=index, rank=rank, options=options, en_mode=en_mode
                    )
                    if "image" in options:
                        im = self.container_data.get_image(index=index)
                        if im is not None:
                            info = html.Div(
                                children=[
                                    info,
                                    html.Img(
                                        src=im,
                                        height="100px",
                                        width=im_width,
                                    ),
                                ]
                            )
                    content.append(info)

        return dbc.Row(content)

    def figure_info(self, key: str, method: str, n_nodes: int = None) -> str:
        if n_nodes is None:
            n_nodes = self.container_data.num_data
        embed_label = [
            item["label"]
            for item in self.config.webapp.embedding_options
            if item.get("value") == key
        ][0]
        method_label = [
            item["label"]
            for item in self.config.webapp.dimension_reduction_options
            if item.get("value") == method
        ][0]
        info_text = f"Conference: {self.config.webapp.text_conference}, #Papers: {n_nodes}, Perspective: {embed_label}, Method: {method_label}, Model: {self.config.embedding.model}"
        return info_text

    def default_figure(
        self,
        df: pd.core.frame.DataFrame,
        key: str,
        method: str,
        dim: int,
        en_mode: bool,
    ):
        feature_name = f"{key}_{method}_{str(dim)}"
        info = self.figure_info(key=key, method=method)

        fig = px.scatter(
            data_frame=df,
            x=f"{feature_name}_x",
            y=f"{feature_name}_y",
            text=df["award_text"],
            hover_data={  # Id must be the last for clicked node analysis.
                "Title": df["long_title"],
                "Task": df["long_task_en"],
                "Topic": df["long_topic_en"],
                "Application": df["long_application_en"],
                "Selection": df["award"],
                "TaskJa": df["long_task_ja"],
                "TopicJa": df["long_topic_ja"],
                "ApplicationJa": df["long_application_ja"],
                "Id": df["id"],
            },
            color=df["award_label"],
            color_discrete_sequence=px.colors.qualitative.Pastel,
            category_orders=self.container_data.legend_orders,
        )
        if en_mode:
            fig.update_traces(
                hovertemplate="<b>Title</b>: %{customdata[0]}<br><b>Task</b>: %{customdata[1]}<br><b>Topic</b>: %{customdata[2]}<br><b>Application</b>: %{customdata[3]}<br><b>Selection</b>: %{customdata[4]}<br>Id: %{customdata[8]}",
            )
        else:
            fig.update_traces(
                hovertemplate="<b>Title</b>: %{customdata[0]}<br><b>Task</b>: %{customdata[1]}%{customdata[5]}<br><b>Topic</b>: %{customdata[2]}%{customdata[6]}<br><b>Application</b>: %{customdata[3]}%{customdata[7]}<br><b>Selection</b>: %{customdata[4]}<br>Id: %{customdata[8]}",
            )
        fig.update_traces(textposition="top center")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        fig.update_layout(
            overwrite=True,
            font=dict(
                # family="Courier New, monospace",
                size=12,  # Set the font size here
                color="#43676b",
            ),
            annotations=[
                dict(
                    text=info,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.1,
                    showarrow=False,
                    font=dict(size=12, color=self.config.webapp.color_fig_title),
                )
            ],
            # showlegend=False,
        )
        return fig

    def selected_figure(
        self,
        df: pd.core.frame.DataFrame,
        key: str,
        method: str,
        dim: int,
        indices: List[int] = None,
    ) -> plotly.graph_objs._figure.Figure:
        indices.sort()
        feature_name = f"{key}_{method}_{str(dim)}"
        info = self.figure_info(
            key=key,
            method=method,
            n_nodes=None if indices is None else len(indices),
        )
        info = insert_line_breaks(
            text=info, max_chars_per_line=60, prefix="", suffix="", newline="<br>"
        )
        if indices is not None:
            target_df = df.iloc[indices]
        else:
            target_df = df
        if dim == 2:
            fig = px.scatter(
                data_frame=target_df,
                x=f"{feature_name}_x",
                y=f"{feature_name}_y",
                text="text",
                hover_data={  # Id must be the last for clicked node analysis.
                    "Title": target_df["short_title"],
                    "Task": target_df["short_task_en"],
                    "Topic": target_df["short_topic_en"],
                    "Application": target_df["short_application_en"],
                    self.config.webapp.label_distance: True,
                    "Selection": target_df["award"],
                    # "TaskJa": target_df["short_task_ja"],
                    # "TopicJa": target_df["short_topic_ja"],
                    "Id": target_df["id"],  # Id must be the last in customdata
                },
                template="ggplot2",
                color=self.config.webapp.label_distance,
                color_continuous_scale="magenta_r",
            )
        # elif dim == 3:
        #     fig = px.scatter_3d(
        #         data_frame=target_df,
        #         x=f"{feature_name}_x",
        #         y=f"{feature_name}_y",
        #         z=f"{feature_name}_z",
        #         text="text",
        #         hover_data={
        #             "Title": target_df["short_title"],
        #             "Category": target_df["short_category"],
        #             "Application": target_df["short_application"],
        #             self.config.webapp.label_distance: True,
        #             "Selection": target_df["award"],
        #             "Id": target_df["id"],  # Id must be the last in customdata
        #         },
        #         template="ggplot2",
        #         color=self.config.webapp.label_distance,
        #         color_continuous_scale="magenta_r",
        #     )
        fig.update_traces(textposition="top center")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        fig.update_traces(
            hovertemplate="<b>Title</b>: %{customdata[0]}<br><b>Task</b>: %{customdata[1]}<br><b>Topic</b>: %{customdata[2]}<br><b>Application</b>: %{customdata[3]}<br><b>Distance</b>: %{customdata[4]}<br><b>Selection</b>: %{customdata[5]}<br>Id: %{customdata[6]}"
        )
        fig.update_layout(
            overwrite=True,
            showlegend=False,
            font=dict(
                # family="Courier New, monospace",
                size=12,  # Set the font size here
                color="#43676b",
            ),
            annotations=[
                dict(
                    text=info,
                    xref="paper",
                    yref="paper",
                    x=-0.2,
                    y=1.1,
                    showarrow=False,
                    font=dict(size=12, color=self.config.webapp.color_fig_title),
                )
            ],
        )
        return fig

    def selected_paper_block(
        self, df: pd.core.frame.DataFrame, index: int, en_mode: bool = False
    ) -> dbc.Col:
        content = self.description_block(
            df=df,
            indices=[index],
            options=list(  # remove description_en because it is almost same as the abstract.
                filter(
                    lambda x: x != "description_en",
                    self.container_data.info_opts_vals_en
                    if en_mode
                    else self.container_data.info_opt_vals,
                )
            ),
            start_rank=0,
            im_width="100%",
            en_mode=en_mode,
        )
        return dbc.Col(
            dbc.Row(
                [
                    self.top_margin,
                    dbc.Badge(
                        self.config.webapp.text_selection_description,
                        color="secondary",
                        className="mr-1",  # boostrap margen right 0.25rem
                        style={
                            "width": "98%",
                            # "margin-bottom": "5px",
                            "text-align": "center",
                            "fontSize": 12,
                            "vertical-align": "middle",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "3px",
                        },
                    ),
                    dcc.Loading(
                        type="default",
                        children=[
                            html.Div(
                                id="selected-paper",
                                children=[content],
                                style={
                                    "maxHeight": self.config.webapp.max_hight,
                                    "overflow": "scroll",
                                    "margin-bottom": "5px"
                                    # "fontFamily": [
                                    #     "UDFont",
                                    #     "Helvetica Neue",
                                    #     "Helvetica",
                                    #     "Hiragino Sans",
                                    #     "Hiragino Kaku Gothic ProN",
                                    #     "Arial",
                                    #     "Yu Gothic",
                                    #     "Meiryo",
                                    # ],
                                },
                            )
                        ],
                    ),
                ]
            )
        )

    def recommendation_block(
        self,
        options: List[str],
        all_options: List[Dict[str, str]],
    ) -> dbc.Col:
        return dbc.Col(
            dbc.Row(
                [
                    self.top_margin,
                    dbc.Badge(
                        self.config.webapp.label_options,
                        color="secondary",
                        className="mr-1",  # boostrap margen right 0.25rem
                        style={
                            "width": "98%",
                            "margin-bottom": "5px",
                            "text-align": "center",
                            "fontSize": 12,
                            "vertical-align": "middle",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "5px",
                        },
                    ),
                    dcc.Checklist(
                        id="details-option",
                        options=all_options,
                        value=options,
                        labelStyle={
                            "display": "inline-block",
                            "color": "DarkSlateGray",
                            "fontSize": 12,
                            "margin-right": "20px",
                            "margin-left": "5px",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "5px",
                        },
                    ),
                    dbc.Badge(
                        self.config.webapp.text_recommendation_description,
                        color="secondary",
                        className="mr-1",  # boostrap margen right 0.25rem
                        style={
                            "width": "98%",
                            "margin": "auto",
                            "text-align": "center",
                            "fontSize": 12,
                            "vertical-align": "middle",
                        },
                    ),
                    self.recommendation_contents,
                ]
            )
        )

    @property
    def navbar(self) -> dbc.Navbar:
        return dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.NavbarBrand(
                                    self.config.webapp.title,
                                    # className="ms-3",
                                    href=self.config.webapp.title_url,
                                    external_link=True,
                                    style={
                                        "color": "#FFFFFF",
                                        "fontSize": self.config.webapp.size_title,
                                    },
                                )
                            ),
                            dbc.Col(
                                dbc.DropdownMenu(
                                    children=[
                                        dbc.DropdownMenuItem(
                                            item["label"],
                                            href=item["href"],
                                            external_link=True,
                                            style={
                                                "fontSize": self.config.webapp.size_title,
                                                "font-weight": "bold",
                                                "color": "#43676b",
                                            },
                                        )
                                        for item in self.config.webapp.conference_options
                                    ],
                                    nav=True,
                                    in_navbar=True,
                                    label=self.config.webapp.text_conference,
                                    style={
                                        "color": "#FFFFFF",
                                        "fontSize": self.config.webapp.size_title,
                                    },
                                )
                            ),
                        ],
                        align="center",
                        className="g-0",
                        style={
                            "fontSize": self.config.webapp.size_title,
                            "font-weight": "bold",
                        },
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "EN",
                                                href=self.config.webapp.title_url
                                                + "/en",
                                                style={
                                                    "fontSize": self.config.webapp.size_code,
                                                    "color": "#824880",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        )
                                    ),
                                    dbc.Col(
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "JP",
                                                href=self.config.webapp.title_url,
                                                style={
                                                    "fontSize": self.config.webapp.size_code,
                                                    "color": "#824880",
                                                    "margin-right": "10px",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        )
                                    ),
                                    dbc.Col(
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "GitHub",
                                                href="https://github.com/yuukicammy/conference-projector",
                                                style={
                                                    "fontSize": self.config.webapp.size_code,
                                                    "color": "#6c757d",
                                                    "font-weight": "bold",
                                                },
                                            )
                                        )
                                    ),
                                ],
                                className="g-3 ms-auto flex-nowrap mt-3 mt-md-0",
                                align="center",
                            ),
                            id="navbar-collapse",
                            is_open=False,
                            navbar=True,
                        )
                    ),
                ]
            ),
            color="dark",
            dark=True,
            className="dash-header",
        )

    @property
    def footer(self) -> dcc.Markdown:
        return dcc.Markdown(
            "© 2023 yuukicammy",
            style={
                "display": "flex",
                "justifyContent": "center",
                "fontSize": 8,
            },
        )

    @property
    def embedding_selector(self) -> dcc.RadioItems:
        return dcc.RadioItems(
            id="embeddings",
            options=self.config.webapp.embedding_options,
            value=self.config.webapp.default_embedding_value,
            labelStyle={
                "display": "inline-block",
                "color": "DarkSlateGray",
                "fontSize": self.config.webapp.size_default,
                "margin-right": "20px",
            },
            style={
                "fontSize": self.config.webapp.size_default,
            },
        )

    @property
    def top_margin(self) -> html.Div:
        return html.Div(
            style={
                "height": "10px",
            },
        )

    @property
    def reduction_algo_selector(self) -> dbc.Col:
        return dbc.Col(
            [
                dbc.Badge(
                    self.config.webapp.label_projection_algorithm,
                    color="secondary",
                    className="mr-1",  # boostrap margen right 0.25rem
                    style={
                        "width": "100%",
                        "fontSize": self.config.webapp.size_default,
                    },
                ),
                dcc.RadioItems(
                    id="reduction-algorithm",
                    options=self.config.webapp.dimension_reduction_options,
                    value="umap",
                    labelStyle={
                        "display": "inline-block",
                        "color": "DarkSlateGray",
                        "fontSize": self.config.webapp.size_default,
                        "margin-right": "20px",
                    },
                    style={
                        "align": "center",
                        "fontSize": self.config.webapp.size_default,
                        "display": "flex",
                        "justifyContent": "center",
                    },
                ),
            ]
        )

    @property
    def dimension_selector(self) -> dbc.Col:
        return dbc.Col(
            [
                dbc.Badge(
                    self.config.webapp.label_dimension,
                    color="secondary",
                    className="mr-1",  # boostrap margen right 0.25rem
                    style={
                        "width": "100%",
                        "fontSize": self.config.webapp.size_default,
                    },
                ),
                dcc.RadioItems(
                    id="dimensions",
                    options=self.config.webapp.dimension_options,
                    value=2,
                    labelStyle={
                        "display": "inline-block",
                        "color": "DarkSlateGray",
                        "fontSize": self.config.webapp.size_default,
                        "margin-right": "20px",
                        "align": "center",
                    },
                    style={
                        "align": "center",
                        "fontSize": self.config.webapp.size_default,
                        "display": "flex",
                        "justifyContent": "center",
                    },
                ),
            ]
        )

    @property
    def recommendation_contents(self) -> dcc.Loading:
        return dcc.Loading(
            type="default",
            children=[
                html.Div(
                    id="recommendation",
                    style={
                        "maxHeight": self.config.webapp.max_hight,
                        "overflow": "scroll",
                        # "fontFamily": [
                        #     "UDFont",
                        #     "Helvetica Neue",
                        #     "Helvetica",
                        #     "Hiragino Sans",
                        #     "Hiragino Kaku Gothic ProN",
                        #     "Arial",
                        #     "Yu Gothic",
                        #     "Meiryo",
                        # ],
                    },
                ),
            ],
        )

    @property
    def detail_view(self) -> dbc.Col:
        return dbc.Col(id="details-row", children=[], width="0%")

    @property
    def graph_view(self) -> dbc.Col:
        return dbc.Col(
            id="graph-area",
            style={
                "minWidth": "700px",  # Set the minimum width of the graph container
                "minHeight": "400px",
            },
            children=[
                dbc.Row(
                    [
                        dbc.Row(self.top_margin),
                        dbc.Row(
                            [
                                dcc.Markdown(
                                    f"## {self.config.webapp.text_concern_description}",
                                ),
                                self.embedding_selector,
                            ],
                        ),
                        dbc.Row(
                            dcc.Loading(
                                type="circle",
                                children=[
                                    dcc.Graph(
                                        id="scatter-plot",
                                        config={
                                            "displayModeBar": False,  # Hide the toolbar (including the legend)
                                        },
                                    )
                                ],
                            )
                        ),
                        # self.reduction_algo_selector,
                        # self.dimension_selector,
                        # html.Div(
                        #     style={
                        #         "height": "5px",
                        #     },
                        # ),
                        dbc.Row(
                            html.Center(
                                html.Button(
                                    "View ALL",
                                    id="viewall-nclicks",
                                    n_clicks=5,
                                    style={
                                        "width": "90%",
                                        "align": "center",
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                    },
                                )
                            )
                        ),
                    ]
                ),
            ],
        )

    @property
    def body_layout(self) -> dbc.Container:
        return dbc.Container(
            children=[
                dcc.Store(id="shared-data"),
                dbc.Row(
                    [
                        self.graph_view,
                        self.detail_view,
                    ]
                ),
            ],
        )

    @property
    def screen_layout(self) -> html.Div:
        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                self.navbar,
                self.body_layout,
                self.footer,
            ]
        )
