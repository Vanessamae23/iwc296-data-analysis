from dataclasses import dataclass, field
from typing import Any, Optional, Union, List
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import plotly.graph_objects as go
from collections import OrderedDict

@dataclass
class ColumnDef:
    name: str
    precision: int
    formula: str

@dataclass
class FunctionVariable:
    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class PlotColumn:
    column: str
    func: list[FunctionVariable] = field(default_factory=list)


class Variable(ABC):
    name: str # represents each figure plot

    @abstractmethod
    def plot(self, versions: "Versions") -> go.Figure:
        ...

    @staticmethod
    def from_yaml(yaml_dict) -> "Variable":
        pass

@dataclass
class PlotVariable(Variable):
    ALLOWED_PLOT_TYPES = {"Scatter", "Bar"}

    name: str
    column: str
    x_column: Optional[str] = "Date"
    plot_type: str = "Scatter"
    plot_source: Optional[str] = None
    filter_var: Optional[str] = None
    filter_value: Optional[Any] = None
    func: list[FunctionVariable] = field(default_factory=list)
    plot_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.plot_type not in self.ALLOWED_PLOT_TYPES:
            raise ValueError(f"PlotVariable only supports {self.ALLOWED_PLOT_TYPES}, got '{self.plot_type}'")

    @classmethod
    def from_yaml(cls, name: str, cfg: dict) -> "PlotVariable":
        func_cfg = cfg.get("func", [])
        if isinstance(func_cfg, str):
            func_cfg = [{"name": func_cfg}]
        elif isinstance(func_cfg, dict):
            func_cfg = [func_cfg]
        funcs = [FunctionVariable(
            name=f["name"] if isinstance(f, dict) else f,
            args=f.get("args", []) if isinstance(f, dict) else [],
            kwargs=f.get("kwargs", {}) if isinstance(f, dict) else {},
        ) for f in func_cfg]
        return cls(
            name=name,
            column=cfg.get("column", ""),
            x_column=cfg.get("x_column"),
            plot_type=cfg.get("plot_type", "Scatter"),
            plot_source=cfg.get("plot_source"),
            filter_var=cfg.get("filter_var"),
            filter_value=cfg.get("filter_value"),
            func=funcs,
            plot_kwargs=cfg.get("plot_kwargs", {}),
        )

    def _apply_func(self, series: pd.Series) -> pd.Series:
        result = series
        for step in self.func:
            func = getattr(result, step.name)
            result = func(*step.args, **step.kwargs)
        return result

    def get_dataframe(self, default_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.plot_source:
            df = pd.read_csv(self.plot_source)
        elif default_df is not None:
            df = default_df.copy()
        else:
            raise ValueError(f"No data source for {self.name}")
        if self.filter_var and self.filter_value is not None:
            df = df[df[self.filter_var] == self.filter_value]
        return df

    def build_trace(self, default_df: Optional[pd.DataFrame] = None) :
        df = self.get_dataframe(default_df)
        y_vals = df[self.column]
        if self.func:
            y_vals = self._apply_func(y_vals)
        x_vals = df[self.x_column] if self.x_column else None
        if self.plot_type == "Scatter":
            return go.Scatter(x=x_vals, y=y_vals, **self.plot_kwargs)
        else:  # Bar
            return go.Bar(x=x_vals, y=y_vals, **self.plot_kwargs)

    def plot(self, versions: "Versions") -> go.Figure:
        version_df_map = versions.get_version_df_map()
        traces = []
        for version_name, version_df in version_df_map.items():
            trace = self.build_trace(version_df)
            trace.name = version_name
            traces.append(trace)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=self.name,
            hovermode='x unified',
            font=dict(family='Arial, sans-serif'),
            plot_bgcolor='#fff',
            showlegend=True,
            legend=dict(title="Version"),
        )
        return fig

@dataclass
class CustomPlotVariable(Variable):
    name: str = ""
    plot_type: str = "Scatter"
    plot_cols: dict[str, PlotColumn] = field(default_factory=dict)
    plot_kwargs: dict[str, Any] = field(default_factory=dict)
    plot_source: Optional[str] = None
    filter_var: Optional[str] = None
    filter_value: Optional[Any] = None

    @classmethod
    def from_yaml(cls, name: str, cfg: dict) -> "CustomPlotVariable":
        plot_cols = {}
        for key, col_cfg in cfg.get("plot_cols", {}).items():
            if isinstance(col_cfg, str):
                plot_cols[key] = PlotColumn(column=col_cfg)
            else:
                func_cfg = col_cfg.get("func", [])
                if isinstance(func_cfg, str):
                    func_cfg = [{"name": func_cfg}]
                elif isinstance(func_cfg, dict):
                    func_cfg = [func_cfg]
                funcs = [FunctionVariable(
                    name=f["name"] if isinstance(f, dict) else f,
                    args=f.get("args", []) if isinstance(f, dict) else [],
                    kwargs=f.get("kwargs", {}) if isinstance(f, dict) else {},
                ) for f in func_cfg]
                plot_cols[key] = PlotColumn(column=col_cfg["column"], func=funcs)
        return cls(
            name=name,
            plot_type=cfg.get("plot_type", "Scatter"),
            plot_cols=plot_cols,
            plot_kwargs=cfg.get("plot_kwargs", {}),
            plot_source=cfg.get("plot_source"),
            filter_var=cfg.get("filter_var"),
            filter_value=cfg.get("filter_value"),
        )

    def _apply_func(self, series: pd.Series, funcs: list[FunctionVariable]) -> pd.Series:
        result = series
        for step in funcs:
            func = getattr(result, step.name)
            result = func(*step.args, **step.kwargs)
        return result

    def get_dataframe(self, default_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.plot_source:
            df = pd.read_csv(self.plot_source)
        elif default_df is not None:
            df = default_df.copy()
        else:
            raise ValueError(f"No data source for {self.name}")
        if self.filter_var and self.filter_value is not None:
            df = df[df[self.filter_var] == self.filter_value]
        return df

    def build_trace(self, default_df: Optional[pd.DataFrame] = None) :
        df = self.get_dataframe(default_df)
        col_data = {}
        for key, col in self.plot_cols.items():
            vals = df[col.column]
            if col.func:
                vals = self._apply_func(vals, col.func)
            col_data[key] = vals
        trace_cls = getattr(go, self.plot_type)
        return trace_cls(**col_data, **self.plot_kwargs)

    def plot(self, versions: "Versions") -> go.Figure:
        version_df_map = versions.get_version_df_map()
        traces = []
        for version_name, version_df in version_df_map.items():
            trace = self.build_trace(version_df)
            trace.name = version_name
            traces.append(trace)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=self.name,
            hovermode='x unified',
            font=dict(family='Arial, sans-serif'),
            plot_bgcolor='#fff',
            showlegend=True,
            legend=dict(title="Version"),
        )
        return fig

@dataclass
class WindowPlotVariable(Variable):
    window: int


class SummaryType(Enum):
    PCT = "pct"


class Base(ABC):
    @abstractmethod
    def calculate(self, df: pd.DataFrame):
        pass

class BaseY(Base):
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

    def calculate(self, df: pd.DataFrame):
        return df[self.val1] + df[self.val2]

@dataclass
class Version:
    name: str
    def get_df(self) -> pd.DataFrame:
        pass

@dataclass
class Versions:
    ref: str
    versions: list[Version] # this is actly a list of Version # remove the ref df

    def get_version_df_map(self):
        version_name_to_df = {}
        for version in self.versions:
            version_name_to_df[version.name] = version.get_df()
        return version_name_to_df

@dataclass
class SummaryVariable:
    format: Optional[str]
    column: str
    summary_type: SummaryType
    func: str
    show_diff: bool = False
    params: list[str] = field(default_factory=list) # the parameter to the computer from columns

    def calculate(self, versions: Versions, summary_df: pd.DataFrame):
        computer_class = None
        if self.summary_type == SummaryType.PCT:
            computer_class = BaseY(self.params[0], self.params[1])

        if computer_class is None:
            raise ValueError("Summary type not supported")


        # 2. Collect values for each version
        # We use a dictionary where key = Version Name, value = Calculated Result
        row_values = {}
        for version_name, version_df in self.df.items():
            # Logic: Perform the calculation for this specific version
            # This assumes computer_class can take the version_df as input
            val = computer_class.calculate(version_df)
            row_values[version_name] = val

        # 3. Insert into summary_df
        # We set the row label as self.column and the columns as the version names
        summary_df.loc[self.column] = pd.NA
        for version_name, value in row_values.items():
            summary_df.at[self.column, version_name] = value

        if self.show_diff:
            ref_name = versions.ref
            ref_val = summary_df.at[self.column, ref_name]

            # Calculate diff for all OTHER versions against the ref
            other_versions = [v for v in versions.versions if v.name != versions.ref]

            for version in other_versions:
                target_val = summary_df.at[self.column, version.name]
                diff_col_name = f"Diff ({version.name} vs {ref_name})"

                # Use .at to ensure the result is placed in the specific diff column
                summary_df.at[self.column, diff_col_name] = target_val - ref_val

@dataclass
class ModelMapping:
    mapper: dict[str, str] = field(default_factory=dict)

@dataclass
class ModelGroup:
    variable_groups: dict[str, list[Variable]] = field(default_factory=dict)

class ModelSummary:
    # Results
    versions: Versions
    summary_df: Optional[pd.DataFrame]
    summary_variables: list[SummaryVariable] = []
    plots: List[Union[CustomPlotVariable, PlotVariable, WindowPlotVariable]] = []

    def __init__(self, versions: Versions, variables: list[Variable] = []):
        self.versions = versions
        for variable in variables:
            if isinstance(variable, SummaryVariable):
                self.summary_variables.append(variable)
            elif isinstance(variable, PlotVariable):
                pass
    
    def prepare_report(self) -> tuple[OrderedDict[str, go.Figure], pd.DataFrame]:
        for summary_variable in self.summary_variables:
            summary_variable.calculate(self.versions, self.summary_df)

        figs: OrderedDict[str, go.Figure] = OrderedDict()
        for plotter in self.plots:
            fig = plotter.plot(self.versions)
            figs[plotter.name] = fig

        return (figs, self.summary_df)

class ConfigLoader:
    yaml_content: Any

    def get_variables(self):
        pass

def prepare_columns(df: pd.DataFrame, column_defs: list[ColumnDef]):
    transformed_df = df
    for column_def in column_defs:
        df[column_def.name] = ...

    return transformed_df