import os
import sys
import dash
import flask
import dash_table as dt

from dash import dcc, Input, Output, State
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
app = dash.Dash(__name__, server= server)


# App layout
app.layout = html.Div(
    [
        html.H1("Multi-Variate Index Regression"),
        html.Div(id='input-section', children = 
                [
                    html.H4('Choose Index',style={'display':'inline-block','margin-right':20}),
                    dcc.Dropdown(id="index", options=index_options, value="^GSPC"),
                    html.H4('Select securities to explain daily performance',style={'display':'inline-block','margin-right':20}),
                    dcc.Dropdown(
                        tickers,
                        default_tickers,
                        id="explain_securities",
                        multi=True
                    ),
                    html.H4('Date Range',style={'display':'inline-block','margin-right':20}),
                    dcc.DatePickerRange(id="date-picker", start_date="2023-01-01", end_date="2023-12-31"),
                    html.Br(),
                    html.Button(id="submit-button", n_clicks=0, children="Submit")
                ]),
        html.H4('Regression Performance',style={'display':'inline-block','margin-right':20}),
        html.Div(id='out1'),
        html.H4('Regression Coefficients',style={'display':'inline-block','margin-right':20}),
        html.Div(id='out2'),
        html.H4('Regression plot',style={'display':'inline-block','margin-right':20}),
        dcc.Graph(id="regression-graph")
    ])

@app.callback(
    Output("out1", "children"),
    Output("out2", "children"),
    Output("regression-graph", "figure"),
    [Input("submit-button", "n_clicks")],
    [
    dash.dependencies.State("index", "value"), 
    dash.dependencies.State("explain_securities", "value"), 
    dash.dependencies.State("date-picker", "start_date"), 
    dash.dependencies.State("date-picker", "end_date")
    ] 
)
def update_dashboard(n_clicks, index, explain_securities, start_date, end_date):
	# Index explain using constituents app
    if n_clicks > 0:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Pull data
        a.set_data_df(start_date, end_date)

        model, fig = a.perform_regression(index, explain_securities)
        
        table1 = model.summary2().tables[0]
        table2 = model.summary2().tables[1].reset_index().rename(columns={"index": "Ticker"}).round(3)

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

        return table1_table, table2_table, fig

    return [], [], go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)