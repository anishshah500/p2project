import os
import sys
import dash
import flask
import dash_table as dt

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_table import DataTable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analytics import *

a = Analytics()
tickers = a.dc.get_tickers()

indexes = ["^GSPC", "^IXIC", "^RUT"]
default_tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMD", "AMZN", "LLY", "JPM"]

index_options = [
    {"label": "S&P 500", "value": "^GSPC"},
    {"label": "Russell 2000", "value": "^RUT"},
    {"label": "Nasdaq 100", "value": "^IXIC"},
]

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom style for card headers
card_header_style = {
    'backgroundColor': '#f8f9fa',
    'fontWeight': 'bold',
    'textAlign': 'center',
    'fontSize': '1.25rem',
    'padding': '10px'
}

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Multi-Variate Index Regression"), className="mb-4")),

        dbc.Row(dbc.Col(dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Regression Input Parameters', style=card_header_style),
                    dbc.Row(
                        [
                            dbc.Col(html.H6('Choose Index'), width=3),
                            dbc.Col(dcc.Dropdown(id="index", options=index_options, value="^GSPC"), width=9)
                        ],
                        className="mb-3"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.H6('Select securities to explain daily performance'), width=3),
                            dbc.Col(dcc.Dropdown(
                                options=[{"label": ticker, "value": ticker} for ticker in tickers],
                                value=default_tickers,
                                id="explain_securities",
                                multi=True
                            ), width=9)
                        ],
                        className="mb-3"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.H6('Date Range'), width=3),
                            dbc.Col(dcc.DatePickerRange(id="date-picker", start_date="2023-01-01", end_date="2023-12-31"), width=9)
                        ],
                        className="mb-3"
                    ),
                    dbc.Row(dbc.Col(dbc.Button("Submit", id="submit-button", n_clicks=0, color="primary", className="w-100"))),
                ]
            ),
            className="mb-4"
        ))),

        dbc.Row(dbc.Col(dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Regression Performance', style=card_header_style),
                    html.Div(id='out1')
                ]
            ),
            className="mb-4"
        ))),

        dbc.Row(dbc.Col(dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Regression Coefficients', style=card_header_style),
                    html.Div(id='out2')
                ]
            ),
            className="mb-4"
        ))),

        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Regression Plot', style=card_header_style),
                            dcc.Graph(id="regression-graph")
                        ]
                    )
                ), width=6),
                dbc.Col(dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Top 10 Securities', style=card_header_style),
                            html.Div(id='out3')
                        ]
                    )
                ), width=6)
            ],
            className="mb-4"
        )
    ],
    fluid=True
)

@app.callback(
    Output("out1", "children"),
    Output("out2", "children"),
    Output("regression-graph", "figure"),
    Output("out3", "children"),
    [Input("submit-button", "n_clicks")],
    [
        State("index", "value"),
        State("explain_securities", "value"),
        State("date-picker", "start_date"),
        State("date-picker", "end_date")
    ]
)
def update_dashboard(n_clicks, index, explain_securities, start_date, end_date):
    if n_clicks > 0:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Pull data
        a.set_data_df(start_date, end_date)

        model, fig = a.perform_regression(index, explain_securities)
        
        table1 = model.summary2().tables[0]
        table2 = model.summary2().tables[1].reset_index().rename(columns={"index": "Ticker"}).round(3)

        top_10_explain_tickers_df = a.get_top_n_explaining_tickers(index, indexes).round(4)

        table1_table = DataTable(
            columns=[{"name": str(i), "id": str(i)} for i in table1.columns],
            data=table1.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px'
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold'
            },
            page_action='native',
            page_size=25,
            filter_action='native',
            sort_action='native'
        )

        table2_table = DataTable(
            columns=[{"name": str(i), "id": str(i)} for i in table2.columns],
            data=table2.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px'
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold'
            },
            page_action='native',
            page_size=25,
            filter_action='native',
            sort_action='native'
        )

        top_10_explain_tickers_df_table = DataTable(
            columns=[{"name": str(i), "id": str(i)}
                     for i in top_10_explain_tickers_df.columns],
            data=top_10_explain_tickers_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '5px'
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold'
            },
            page_action='native',
            page_size=25,
            filter_action='native',
            sort_action='native'
        )

        return table1_table, table2_table, fig, top_10_explain_tickers_df_table

    return [], [], go.Figure(), []

if __name__ == '__main__':
    app.run_server(debug=True)
    