from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
import yaml
import dash_ag_grid as dag
import importlib

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

def df_func(series, func_cfg):
    if isinstance(func_cfg, str):
        func_cfg = [{"name": func_cfg}]
    elif isinstance(func_cfg, dict):
        func_cfg = [func_cfg]
    result = series
    for step in func_cfg:
        if isinstance(step, str):
            result = getattr(result, step)()
        else:
            name = step["name"]
            args = step.get("args", [])
            kwargs = step.get("kwargs", {})
            result = getattr(result, name)(*args, **kwargs)
    return result

def go_func(plot_type: str = "Scatter", **kwargs):
    func = getattr(go, plot_type)
    return func(**kwargs)

app = Dash()

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

DEFAULT_YAML = """\
DEFAULT:
    plot_type: Ohlc
    plot_cols:
        x:
            column: Date
        open:
            column: AAPL.Open
        high:
            column: AAPL.High
        low:
            column: AAPL.Low
        close:
            column: AAPL.Close
            func:
              - name: rolling
                args: [5]
                kwargs:
                    min_periods: 1
              - name: mean
    plot_kwargs:
        hoverinfo: text+x+y
SECOND:
    plot_type: Ohlc
    plot_cols:
        x:
            column: Date
        open:
            column: AAPL.Open
        high:
            column: AAPL.High
        low:
            column: AAPL.Low
        close:
            column: AAPL.Close
            func:
              - name: rolling
                args: [5]
                kwargs:
                    min_periods: 1
              - name: mean
    plot_kwargs:
        hoverinfo: text+x+y
NVDA:                                                                         
    plot_source: "/Users/vanessamae/Documents/playground/python/NVDA.csv"     
    plot_type: Scatter                                                        
    plot_cols:                                                                
        x:                                                                    
            column: timestamp                                                 
        y:                                                                    
            column: open
"""

columnDefs = columnDefs = [{'field': col_name, 'headerName': col_name } for col_name in df.columns]


grid = dag.AgGrid(
    id="getting-started-sort",
    rowData=df.to_dict("records"),
    columnDefs=columnDefs,
    dashGridOptions={"suppressFieldDotNotation": True},
)

app.layout = html.Div([
    html.Div([
        html.Div([grid], style={
            'marginBottom': '16px',
            'marginTop': '16px',
            'display': 'block',
        }),
        html.Label('Config', style={
            'fontWeight': '600',
            'fontSize': '20px',
            'color': '#444',
            'marginBottom': '6px',
            'display': 'block',
        }),
        dcc.Textarea(
            id='yaml-input',
            value=DEFAULT_YAML,
            style={
                'width': '100%',
                'height': '160px',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'padding': '10px',
                'border': '1px solid #ccc',
                'borderRadius': '4px',
                'resize': 'vertical',
                'backgroundColor': '#fff',
                'lineHeight': '1.5',
            },
        ),
        html.Button('Render', id='render-btn', n_clicks=0, style={
            'marginTop': '10px',
            'padding': '8px 24px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '13px',
            'fontWeight': '600',
            'color': '#fff',
            'backgroundColor': '#1a1a1a',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
        }),
        html.Div(id='yaml-error', style={
            'color': '#d32f2f',
            'fontSize': '12px',
            'marginTop': '6px',
        }),
    ], style={
        'padding': '16px 20px',
        'backgroundColor': '#fafafa',
        'borderBottom': '1px solid #e0e0e0',
        'borderRadius': '6px 6px 0 0',
    }),

    dcc.Graph(id='indicator-graphic', style={'padding': '0 10px'}),

], style={
    'fontFamily': 'Arial, sans-serif',
    'maxWidth': '1200px',
    'margin': '30px auto',
    'backgroundColor': '#fff',
    'borderRadius': '6px',
    'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
    'border': '1px solid #e0e0e0',
})

@callback(
    Output('indicator-graphic', 'figure'),
    Output('yaml-error', 'children'),
    Input('render-btn', 'n_clicks'),
    State('yaml-input', 'value'))
def update_graph(n_clicks, yaml_str):
    empty = go.Figure()

    try:
        config = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        return empty, f"YAML parse error: {e}"

    if not isinstance(config, dict):
        return empty, "Invalid YAML: expected a mapping"

    from plotly.subplots import make_subplots

    titles = list(config.keys())
    n = len(titles)
    n_cols = 4
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for idx, (title, graph_cfg) in enumerate(config.items()):
        if not isinstance(graph_cfg, dict):
            continue

        row = 1 + idx // n_cols
        col = 1 + idx % n_cols

        plot_type = graph_cfg.get('plot_type', 'Scatter')
        plot_source = graph_cfg.get('plot_source', None)
        if plot_source:
            dff = pd.read_csv(plot_source)
        else:
            dff = df
        filter_var = graph_cfg.get('filter_var', None)
        filter_value = graph_cfg.get('filter_value', None)
        if filter_var and filter_value:
            dff = dff[dff[filter_var] == filter_value]

        plot_cols = graph_cfg.get('plot_cols', {})
        col_data = {}
        for key, cfg in plot_cols.items():
            if isinstance(cfg, str):
                col_data[key] = dff[cfg]
            else:
                vals = dff[cfg['column']]
                func = cfg.get('func', None)
                if func:
                    vals = df_func(vals, func)
                col_data[key] = vals

        plot_kwargs = graph_cfg.get('plot_kwargs', {})
        trace = go_func(plot_type, **col_data, **plot_kwargs)
        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        height=400 * n_rows,
        margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
        hovermode='closest',
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#fafafa',
        showlegend=False,
    )
    fig.update_xaxes(gridcolor='#e0e0e0')
    fig.update_yaxes(gridcolor='#e0e0e0')

    return fig, ''


if __name__ == '__main__':
    app.run(debug=True)
