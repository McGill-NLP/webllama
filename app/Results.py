#!/usr/bin/env python3
from collections import defaultdict
import os
import time
from datetime import datetime
import json
import random
import string
import shutil
import traceback
import sys
from pathlib import Path
import textwrap as tw

import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd

import weblinx as wt


sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.utils import show_overlay

def remove_latex(x):
    if isinstance(x, tuple):
        return tuple(map(remove_latex, x))

    if not isinstance(x, str):
        return x

    if x.startswith("Unnamed"):
        return ""

    if "}" not in x:
        return x

    return x.rpartition("}")[0].partition("{")[2]


def load_and_clean_df(path):
    df = pd.read_csv(path, index_col=0, header=[0, 1])

    df.index.name = "model"
    df.columns = df.columns.map(remove_latex)
    df.index = df.index.map(remove_latex)

    df = df.reset_index().set_index(["model", "intent"])

    return df


def build_cond_df(base_dir="analysis/data/tables/"):
    base_dir = Path(base_dir).resolve()
    tables_dir = base_dir / "results"

    cond_df = pd.concat(
        axis=1,
        objs=[
            load_and_clean_df(tables_dir / "results_general_grouped_intents.csv"),
            load_and_clean_df(tables_dir / "results_text_grouped_intents.csv"),
            load_and_clean_df(tables_dir / "results_elem_grouped_intents.csv"),
        ],
    )

    return cond_df


def build_uncond_df(base_dir="analysis/data/tables/"):
    base_dir = Path(base_dir).resolve()
    tables_dir = base_dir / "results_unconditional"

    uncond_df = pd.concat(
        axis=1,
        objs=[
            load_and_clean_df(
                base_dir / "results" / "results_general_grouped_intents.csv"
            ),
            load_and_clean_df(tables_dir / "results_text_grouped_intents.csv"),
            load_and_clean_df(tables_dir / "results_elem_grouped_intents.csv"),
        ],
    )

    return uncond_df


@st.cache_data(ttl=60 * 2)
def build_dataframe(score_path, choice):
    with open(score_path) as f:
        scores = json.load(f)

    for score in scores:
        replacements = [
            ("website", "test-web"),
            ("blind", "test-vis"),
            ("subcategory", "test-cat"),
            ("geography", "test-geo"),
            ("dev", "dev-deprecated"),
            ("indomain", "test-indomain"),
        ]
        for original, new in replacements:
            score["split"] = score["split"].replace(original, new)
    df = pd.DataFrame(scores)

    if choice != "Conditional":
        df["score"] = df["unconditional_score"]
    df.pop("unconditional_score")

    dff = df.pivot(
        index=["intent", "project_name", "model_name"],
        columns=["split", "metric"],
        values="score",
    )

    return dff


def add_test_avg_inplace(df, fillna_with=0):
    splits = df.columns.get_level_values(0).unique().tolist()
    test_splits = [x for x in splits if x.startswith("test") and not x.endswith("iid")]
    metrics = df.columns.get_level_values(1).unique().tolist()
    # We need to take the mean of the test splits for each metric in metrics, we call this test-avg
    for metric in metrics:
        test_scores = [df[(split, metric)] for split in test_splits]
        test_df = pd.concat(test_scores, axis=1)
        if fillna_with is not None:
            test_df = test_df.fillna(fillna_with)

        df[("test-avg", metric)] = test_df.mean(axis=1)


def preset_to_values():
    return {
        "All approximate": {
            "intent": [
                "overall",
                # "change",
                "click",
                "load",
                "say",
                # "scroll",
                "submit",
                "textinput",
            ],
            "metric": ["overall", "iou", "chrf", "urlf"],
        },
        "Group Approximate": {
            "intent": [
                "overall",
                "text-group",
                "element-group",
            ],
            "metric": ["overall", "intent-match", "iou", "chrf-urlf"],
        },
        "All intent-match": {
            "intent": [
                "change",
                "click",
                "load",
                "say",
                "scroll",
                "submit",
                "textinput",
            ],
            "metric": ["intent-match"],
        },
        "change": {
            "intent": ["change"],
            "metric": ["intent-match", "iou"],
        },
        "click": {
            "intent": ["click"],
            "metric": ["intent-match", "iou"],
        },
        "say": {
            "intent": ["say"],
            "metric": ["intent-match", "chrf"],
        },
        "textinput": {
            "intent": ["textinput"],
            "metric": ["intent-match", "iou", "chrf"],
        },
        "load": {
            "intent": ["load"],
            "metric": ["intent-match", "urlf"],
        },
        "submit": {
            "intent": ["submit"],
            "metric": ["intent-match", "iou"],
        },
        
    }

def latex_sort_func(name):
    if not (name.endswith('B') or name.endswith("M")):
        return name, 0

    left, sep, right = name.rpartition("-")

    num = float(right[:-1])

    if right.endswith("B"):
        rest = 1e9
    elif right.endswith("M"):
        rest = 1e6
    else:
        rest = 1
    
    num = num * rest
    
    if left.startswith("MindAct"):
        left = 0
    elif left.startswith("Flan"):
        left = 1
    elif left.startswith("Pix2Struct"):
        left = 2
    elif left.startswith("Fuyu"):
        left = 3
    elif left.startswith("Sheared"):
        left = 4
    elif left.startswith("Llama"):
        left = 5
    elif left.startswith("GPT"):
        left = 6
    
    return left, num
    
        
        

@st.cache_data(ttl=60 * 2)
def filter_models_by_project(projects, df):
    # reset all indices except project_name
    df = df.copy().reset_index().set_index("project_name")
    # filter by project
    rem_models = df.loc[projects]["model_name"].unique().tolist()
    return rem_models


def run(score_path="modeling/results/aggregated_scores.json"):
    st.title("Results Table Viewer")

    presets = preset_to_values()

    # Either choose cond or uncond
    with st.sidebar:
        use_two_cols = st.checkbox(
            "Use two columns", value=True, help="Use two columns for the dropdowns"
        )

        pivot_intent_index = st.checkbox(
            "Show intent as column",
            value=True,
            help="Whether to show intent as column, or keep it as index",
        )

        choice = st.radio(
            "Results wrt matched intent",
            ["Conditional", "Unconditional"],
            help=(
                "Conditional: only count samples where the predicted intent matches the reference "
                "intent (when there is no match, the sample is discarded)"
                "Unconditional: counts all samples (when there is no match, the score is set to 0)"
            ),
            index=1,
        )

        preset_choice = st.selectbox("Metric/Intent Preset", list(presets.keys()), index=0)

        remove_na = st.checkbox(
            "Drop cols with only NaN", value=True
        )

        remove_zero = st.checkbox(
            "Drop cols with only 0", value=True
        )

    if use_two_cols:
        col1, col2 = st.columns(2)
    else:
        col1 = col2 = st.columns(1)[0]

    
    df = build_dataframe(score_path, choice)

    add_test_avg_inplace(df)

    splits = df.columns.get_level_values("split").unique().tolist().copy()
    metrics = df.columns.get_level_values("metric").unique().tolist()
    models = df.index.get_level_values("model_name").unique().tolist()
    intents = df.index.get_level_values("intent").unique().tolist()
    projects = df.index.get_level_values("project_name").unique().tolist()
    
    default_splits = ["valid"]
    default_intents = presets[preset_choice]["intent"]
    default_metrics = presets[preset_choice]["metric"]

    default_projects = ["llama_ft"]

    splits = col1.multiselect("Split", splits, default=default_splits)
    metrics = col1.multiselect("Metric", metrics, default=default_metrics)
    intents = col2.multiselect("Intent", intents, default=default_intents)
    sort_by_container = col2.container()
    projects = col1.multiselect("Project", projects, default=default_projects)

    remaining_models = filter_models_by_project(projects=projects, df=df)
    models = col2.multiselect("Model", remaining_models, default=remaining_models)

    if len(projects) == 0:
        st.error("Please select at least one project")
        st.stop()

    if len(models) == 0:
        st.error("Please select at least one model")
        st.stop()

    cols = pd.MultiIndex.from_product([splits, metrics], names=["split", "metric"])
    # remove all cols not in dff
    cols = cols.intersection(df.columns)

    idx = pd.MultiIndex.from_product(
        [intents, projects, models], names=["intent", "project_name", "model_name"]
    )
    # remove all idx not in dff
    idx = idx.intersection(df.index)

    dff = df.loc[idx, cols]

    if pivot_intent_index:
        dff = dff.reset_index("intent").pivot(columns="intent")
    
        if remove_na:
            dff = dff.dropna(axis=1, how="all")
        if remove_zero:
            dff = dff.loc[:, (dff != 0).any(axis=0)]

    with sort_by_container:
        sort_by = st.selectbox("Sort by", dff.columns.tolist())
    
    # Sort by
    if sort_by:
        dff = dff.sort_values(sort_by, ascending=False)

    # swap order of column indices so that we have, in order, split, intent, metric
    dff = dff.swaplevel(1,2, axis=1)

    
    with st.expander("Latex Table"):
        dropped_col_indices = st.multiselect(
            "Drop columns levels", dff.columns.names, default=['split']
        )
        use_shorthand = st.checkbox("Use shorthand", value=False)
        use_custom_sorting = st.checkbox("Use custom sorting", value=True)
        remove_column_names = st.checkbox("Remove column names", value=True)
        merge_index = st.checkbox("Merge index", value=False)

        # dropdown
        custom_index_names = st.selectbox(
            "Custom index names", ["None", "Appendix"], index=0
        )

        custom_column_names = st.selectbox(
            "Custom column names", ["None", "Appendix"], index=0
        )
        
        # Rename metrics to symbols for latex
        with open("app/data/latex/column_name_map.json") as f:
            column_name_map = json.load(f)
        
        with open("app/data/latex/index_name_map.json") as f:
            index_name_map = json.load(f)
        
        with open("app/data/latex/hide_list.json") as f:
            hide_list = json.load(f)
        with open("app/data/latex/shortcut_maps.json") as f:
            shortcut_maps = json.load(f)

        if custom_index_names == "Appendix":
            with open("app/data/latex/custom/appendix/index_name_map.json") as f:
                custom_index_name_map = json.load(f)
            # update index name map with custom names
            index_name_map.update(custom_index_name_map)
        
        if custom_column_names == "Appendix":
            with open("app/data/latex/custom/appendix/column_name_map.json") as f:
                custom_column_name_map = json.load(f)
            # update column name map with custom names
            column_name_map.update(custom_column_name_map)

        # Remove rows from dff_latex if the project_name index and model_name index are in hide_list
        dff_latex = dff.copy()
        dff_latex = dff_latex.reset_index()
        for project_name, model_name in hide_list:
            dff_latex = dff_latex[~((dff_latex["project_name"] == project_name) & (dff_latex["model_name"] == model_name))]
        dff_latex = dff_latex.set_index(["project_name", "model_name"])
        dff_latex = dff_latex.rename(columns=column_name_map)
        dff_latex = dff_latex.rename(index=index_name_map)
        
        
        for i in dropped_col_indices:
            dff_latex.columns = dff_latex.columns.droplevel(i)

        # Convert all column index names to Capitalized
        dff_latex.columns.names = [x.capitalize() for x in dff_latex.columns.names]

        # Sort by index level 0
        if use_custom_sorting:
            dff_latex = dff_latex.sort_index(
                key=lambda index: index.map(latex_sort_func), ascending=True,
            )
        else:
            dff_latex = dff_latex.sort_index(ascending=True)
        
        if merge_index:
            # Join the multiindex into a single index separated by -
            dff_latex.index = dff_latex.index.map(lambda x: " - ".join(x))
        
        if remove_column_names:
            dff_latex.columns.names = [None] * len(dff_latex.columns.names)

        # should be at 4 decimal places
        # Multiply by 100 to get percentage
        dff_latex = dff_latex * 100
        dff_latex = dff_latex.to_latex(float_format="{:0.2f}".format)
        
        if use_shorthand:
            for full_value, shortcut in shortcut_maps.items():
                dff_latex = dff_latex.replace(full_value, shortcut)

        st.code(dff_latex, language="latex")

    with st.expander("Markdown Table"):
        st.code(dff.round(4).to_markdown(), language="markdown")

    with st.expander("Results Table", expanded=True):
        st.table(dff)

    # Best models
    if not pivot_intent_index:
        # We need to pivot the table to get the best model per intent
        dff = dff.reset_index("intent").pivot(columns="intent")

    best_df = pd.concat([dff.idxmax(), dff.max()], axis=1)
    # Set name of best_df.multiindex indexes
    best_df.columns = ["best_model", "best_score"]

    best_df["project_name"] = best_df["best_model"].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    best_df["best_model"] = best_df["best_model"].apply(lambda x: x[1] if isinstance(x, tuple) else x)
    # Reorder columns
    best_df = best_df[["project_name", "best_model", "best_score"]]

    with st.expander("Best Models", expanded=True):
        st.table(best_df)


if __name__ == "__main__":
    try:
        st.set_page_config(layout="wide")
    except:
        pass
    # run = protect_with_authentication(run)
    run()
