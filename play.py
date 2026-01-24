import numpy as np
import importlib
import sys
import base64
from run_functions import interactively_input_arguments_for_function


def resolve_func(func_reference_name):
    split_func = func_reference_name.rsplit(".", 1)
    if len(split_func) == 1:
        raise Exception(
            "%s should include filename as well as function e.g. filename.funcname or module.filename.funcname"
            % func_reference_name
        )
    funcname = split_func.pop()
    funcsource = split_func.pop()

    # stop overzealous interpreter tripping up
    func = None

    # imports have to be done in main
    try:
        mod = importlib.import_module(funcsource)
    except ImportError:
        raise Exception(
            "NOT FOUND: Module %s specified for function reference %s\n"
            % (funcsource, func_reference_name)
        )

    func = getattr(mod, funcname, None)

    if func is None:
        raise Exception(
            "NOT FOUND: function %s in module %s  specified for function reference %s"
            % (funcname, mod, func_reference_name)
        )

    return func

def main(func_name):
    func = getattr(np, func_name)
    args, kwargs = interactively_input_arguments_for_function(func, func_reference_name)
    plot = func(*args, **kwargs)

    print("Plot:", plot)

if __name__ == "__main__":
    # if len(sys.argv) == 1:
    #     print(
    #         "Enter the name of a function with full pathname eg systems.basesystem.System"
    #     )
    #     exit()

    # func_reference_name = sys.argv[1]
    # main(func_reference_name)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import pandas as pd
    import re

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/Mining-BTC-180.csv")

    for i, row in enumerate(df["Date"]):
        p = re.compile(" 00:00:00")
        datetime = p.split(df["Date"][i])[0]
        df.iloc[i, 1] = datetime

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Mining-revenue-USD"],
            mode="lines",
            name="mining revenue"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Hash-rate"],
            mode="lines",
            name="hash-rate-TH/s"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)",
                        "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD",
                        "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[df[k].tolist() for k in df.columns[1:]],
                align = "left")
        ),
        row=1, col=1
    )
    fig.update_layout(
        height=800,
        autosize=True,
        showlegend=False,
        title_text="Bitcoin mining stats for 180 days",
    )

    fig.show()

