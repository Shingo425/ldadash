#!/usr/bin/env python3
import base64

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import os
import sys
import io
import json
import pickle

sys.path.append("./import_py")
import util


#import logging

# logging
#logging.basicConfig(filename='logfile/logger.log', level=logging.INFO)
# デフォルトのスタイルをアレンジ
common_style = {'font-family': 'Meiryo UI'}
# アップロード部分のスタイル
upload_style = {
    'width': '60%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '0 auto'
            }

color_list =["red",
            "purple",
            "fuchsia",
            "green",
            "lime",
            "olive",
            "yellow",
            "navy",
            "blue",
            "teal",
            "aqua"]
topic_list = [
    4,5,6,7,8
    ]

# bootstrapを使用
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

APIkey = os.environ["TWEET_API_KEY"] if "TWEET_API_KEY" in os.environ.keys() else ""
APItoken = os.environ["TWEET_API_TOKEN"] if "TWEET_API_TOKEN" in os.environ.keys() else ""
twitter_bool =APIkey != "" and APItoken != ""

# レイアウトの作成
app.layout = dbc.Container(
    [
     # Title
    dbc.Row(
        dbc.Col(
        html.H1('Topic Model Application')
        ),className="text-center my-3"
        ),

    dbc.Row(
        [dbc.Col([
            dbc.Checklist(
                id="check-sjis",
                options=[{"label":"ファイルのアップロード・ダウンロードをsjisにする","value":"sjis"}],
                )],xl=4,md = 6,className="d-flex align-items-end"
            ),
            dbc.Col([
            dbc.Checklist(
                id="check-gettweet",
                options=[{"label":"twitterから取得","value":"twitterから取得"}],
                switch=True,
                style= {} if twitter_bool else {"display":"None"}
                )],xl=4,md = 2,className="d-flex align-items-end"
            ),
            
            ]),
# ファイルのアップロード前のテキスト            
    dbc.Row([
        dbc.Col(html.P("解析したいテキストファイル(csv) または 学習済みpklファイル(ldadash_output.pkl)を入力してください。"
                       ,id="preupload-text"),className="text-center mt-4")        
        ]),

            
        # ファイルアップロードの部分を作る

    dbc.Row(
        dbc.Col(dcc.Upload(
            id='upload-data',
            children = html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style=upload_style,
            # Allow multiple files to be uploaded
            # # これないとおそらくエラーになる
            multiple = True
        )),className = "my-3",id= "upload-data-row"
        ),

    # tweet取得部分を作成
    dbc.Row([
        dbc.Col(html.P("過去7日間のtwitterデータを取得します。検索にRTを含めたくない場合は、検索クエリの最後に「-RT」をつけてください。"
                       ,id="twitter-text"),className="text-center")        
        ]),
    dbc.Row([
        # クエリ
        dbc.Col([
            html.Label("検索クエリ"),
            dbc.Input(
                id = "input-query"
                )],md = 3
            ),

        # 詳細設定
        dbc.Col([
            html.Label("取得件数"),
            dcc.Dropdown(
                id = "dropdown-maxnum",
                options = [{"label":i*100,"value":i*100} for i in range(1,450)]
                )],md = 2,xs = 9
            ),
        # 実行ボタン
        dbc.Col([
            dbc.Button(
                "取得", id="gettweet-exec",color="success"
                )],md = 1,xs = 3,className="d-flex align-items-end"
            )
        ],className="mx-auto my-4 justify-content-center",id = "gettweet-row"),



    
    dcc.Loading(id="loading-0",
        children=[html.Div([

    # ファイルアップロード後の設定
    dbc.Row([dbc.Col(html.P(id="upload-text"),className="text-center",md=9 if twitter_bool else 12),
             dbc.Col(html.Div([
                    html.Button("twitterデータのダウンロード", id="btn-twitter"),
                    Download(id="download-twitter")]),md=3),
             
             ]),
    dbc.Row([
        # テキスト対象項目
        dbc.Col([
            html.Label("テキスト解析対象項目(必須)",id="label-textcol"),
           dcc.Dropdown(
              id = "dropdown-textcol",
              )],md = 3
           ),
        # ラベル項目
        dbc.Col([
            html.Label("ラベル項目",id="label-labelcol"),
           dcc.Dropdown(
              id = "dropdown-labelcol",

              )],md = 3
           ),

        # トピック数
        dbc.Col([
            html.Label("トピック数(必須)",id="label-topicnum"),
            dcc.Dropdown(
                id = "dropdown-topicnum",
                options = [{"label":i,"value":i} for i in topic_list],
                )],md = 3
            ),

        # 詳細設定
        dbc.Col([
            dbc.Checklist(
                id="check-shousai",
                options=[{"label":"詳細設定","value":"詳細設定"}],
                switch=True
                )],md = 2,xs = 9,className="d-flex align-items-end"
            ),
        # 実行ボタン
        dbc.Col([
            dbc.Button(
                "実行", id="button-exec",color="success"
                )],md = 1,xs = 3,className="d-flex align-items-end"
            )
        ],className="mx-auto my-4 justify-content-center",id = "afterupload-row"),
    # 詳細
    # stopword入力
    dbc.Row([
        dbc.Col(
            dbc.FormGroup(id = "input-stopword")   
        )
    ]),
    # 品詞
    dbc.Row([
        dbc.Col(
            dbc.FormGroup(id = "check-hinshi")   
        )
    ]),
    dbc.Row([html.Div(id = 'intermediate-value0', style={'display': 'none'})]),
    dbc.Row([html.Div(id='intermediate-value-pkl', style={'display': 'none'})])
    ])]),

    # ダウンロード
    dbc.Row([
    dbc.Col([
    html.Div([
        html.Button("学習済みpklのダウンロード", id="btn-pkl"),
        Download(id="download-pkl")]),
    ],className="d-flex justify-content-end")]),

    dbc.Row([
        dbc.Col(
            dcc.Loading(id="loading-1",
                    children=[html.Div([
                        html.Label(id ="label-scatter-plot"),
                        dcc.Graph(id='scatter-plot',style={'height':'600px'})
                                ]
                        )]),md=6),
        dbc.Col(
            dcc.Loading(id="loading-2",
                children=[html.Div([
                    html.Label(id="label-bar-plot"),
                    dcc.Graph(id='bar-plot',style={'height':'600px'})]),
                    # 中間ファイル(表示しない)
                    dbc.Row([html.Div(id='intermediate-value', style={'display': 'none'})])                    
                    ]
                ),md=6)
        
        ]),
        # ダウンロード
    # テーブル
    dbc.Row([
        dbc.Col([
        dcc.Loading(id="loading-3",
            children=[
            dbc.Row(
                dbc.Col([
                html.Div([
                    html.Button("トピックごとの単語分布のダウンロード", id="btn-topicword"),
                    Download(id="download-topicword")]),
                html.Div([
                    html.Button("文書とトピック付与のダウンロード", id="btn-doctopic"),
                    Download(id="download-doctopic")])
                ]
                ,className="d-flex justify-content-end")
            ),                
                
                
                html.Label(id="output-caption"),
                dash_table.DataTable(
                    id="output-data",
                    style_table={
    #                                                    'overflowX': 'scroll',
                                    'overflowY': 'scroll',
                                    'maxHeight': '500px',
                                    
                                },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'},
                    style_cell = {
                        'textAlign': 'left',
                        'textOverflow': 'ellipsis',
                        },
                    style_data={
                        'whiteSpace': 'normal',
                        },                                                
                    style_cell_conditional=[
                        {'if': {'column_id': 'text'},
                         'width': '80%'},
                        {'if': {'column_id': 'label'},
                         'width': '10%'}],
                    css=[{
                        'selector': '.dash-spreadsheet td div',
                        'rule': '''
                        line-height: 15px;
                        max-height: 30px; min-height: 30px; height: 30px;
                        display: block;
                        overflow-y: hidden;
                        '''
                        }],
                    ),
        # 中間ファイル(表示しない)
        dbc.Row([html.Div(id='mmm', style={'display': 'none'})])                    
                
                
                ])
        ])],
            
        style={
            "width":"100%",
        }
        )],
    style=common_style,
    fluid=True
)


# データをロードし、項目指定などのドロップダウンを表示
@app.callback([
    # 中間ファイル(入力データ)
    Output('intermediate-value0', 'children'),
    # 「(入力ファイル)を読み込みました」というテキスト
    Output("upload-text","children"),
    # dropdownの項目
    Output('dropdown-textcol',"options"),
    Output('dropdown-labelcol',"options"),
    # dropdownの初期値を選択
    Output('dropdown-textcol',"value"),
    Output('dropdown-labelcol',"value"),
    Output("dropdown-topicnum","value"),

    # データ入力前は要素を隠しておくためにstyleを指定 

    Output('afterupload-row',"style"),
    Output('button-exec',"style"),
    Output('btn-twitter',"style"),
    
    
    Output("intermediate-value-pkl","children")
    ],
    [Input('upload-data', 'contents'),
     Input("gettweet-exec","n_clicks")],
    [State('upload-data', 'filename'),
     State("input-query","value"),
     State("dropdown-maxnum","value"),
     State("check-sjis","value"),     
     ]
)
def update_dataload(contents,gettweetflg, filenames,query,maxnum,check_sjis):    
    np.random.seed(1)
    # データは読みこまない
    #df = pd.read_csv("../data/livedoornews_short.csv")
    # コールバックが起こるがまだデータはアップロードされていないので、例外処理を行う

    # アップロードしたデータテーブルの中身を読み込む
    if contents is None and gettweetflg is None:
        return [dash.no_update,dash.no_update ,
                dash.no_update,dash.no_update ,
                dash.no_update,dash.no_update, dash.no_update,              
                {"display":"None"},{"display":"None"},{"display":"None"},dash.no_update]
    
    if contents is not None:
        data,trainedflg,error_flg = util.parse_contents(contents[0], filenames[0],check_sjis)
        if error_flg==1:
            return [dash.no_update,"ファイルの形式はcsvまたはpklを指定してください。",dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update]
        if error_flg==2:
            return [dash.no_update,"ファイルの読み込み時にエラーが発生しました。文字コードなどを見直してください。",dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update]

            
        
        if trainedflg is False:
            df = data
            # テキスト項目のdropdown作成
            textcol = [{'label': i, 'value': i} for i in df.columns[df.dtypes=="object"]]
            # ラベル項目のdroprown作成
            labelcol = [{'label': i, 'value': i} for i in df.columns]
            labelcol.append({'label': "(指定なし)", 'value': "(指定なし)"})
            return [df.to_json(), f"{filenames[0]}({len(df)}レコード)を読み込みました",textcol, labelcol,
                    df.columns[0],"(指定なし)",6,
                    {},{},{"display":"None"},dash.no_update]
        else:
            # テキスト項目のdropdown作成
            textcol = [{'label': "text", 'value': "text"}]
            # ラベル項目のdroprown作成
            labelcol = [{'label': "label", 'value': "label"}]
            
            return [dash.no_update, "",textcol, labelcol,
                    "text","label",data["topicnum"],
                    {},{"display":"None"},{"display":"None"},json.dumps(data)]
    
    else:
        success,df = util.gettweet(APIkey,APItoken,query,maxnum)
        if success:
            textcol = [{'label': i, 'value': i} for i in df.columns[df.dtypes=="object"]]
            # ラベル項目のdroprown作成
            labelcol = [{'label': i, 'value': i} for i in df.columns]
            labelcol.append({'label': "(指定なし)", 'value': "(指定なし)"})
            return [df.to_json(), f"twitterからクエリ「{query}」を{len(df)}レコード読み込みました",textcol, labelcol,
                        "text","user.name",6,
                        {},{},{},dash.no_update]
        else:
            return [dash.no_update,"APIkeyまたはAPItokenが間違っています。",dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update,
                    dash.no_update,dash.no_update,dash.no_update,dash.no_update]

        

# 詳細(ストップワードと品詞)を指定
@app.callback([
    Output('input-stopword', 'children'),
    Output('check-hinshi', 'children'),    
    Output('input-stopword', 'style'), 
    Output('check-hinshi', 'style'),    
    ],
    [Input('check-shousai', 'value')]
)
def update_shousai(check_shousai): 
    if check_shousai:
        inputstopword = [dbc.Label("Stop Word"),
             dbc.Input(placeholder="カンマ区切りで入力してください。",
                       type="text",
                       value="")
            ]
        checkhinshi =[
                dbc.Label("使用する品詞を選択してください"),
                dbc.Checklist(
                    options=[
                        {"label": "名詞/一般", "value": "名詞/一般"},
                        {"label": "名詞/固有名詞", "value": "名詞/固有名詞"},
                        {"label": "動詞/自立", "value": "動詞/自立"},
                        {"label": "形容詞/自立", "value": "形容詞/自立"},
                        {"label": "名詞/形容動詞語幹", "value": "名詞/形容動詞語幹"},                        
                    ],
                    value=["名詞/一般","名詞/固有名詞"],
                    id="checklist-inline-input",
                    inline=True,
                ),
            ]
        return [inputstopword, checkhinshi,
                {"display":"block"},{"display":"block"}]
    else:
        return [dash.no_update,dash.no_update,{"display":"None"},{"display":"None"}]

# twitterを指定
@app.callback([
    Output('twitter-text', 'style'),
    Output('gettweet-row', 'style'),
    Output('upload-data-row', 'style'),   
    Output('preupload-text',"style")
    ],
    [Input('check-gettweet', 'value')]
)
def update_gettweet(check_gettweet): 
    if check_gettweet:
        return [{},{},{"display":"None"},{"display":"None"}]
    else:
        return [{"display":"None"},{"display":"None"},{},{}]


    
# 形態素解析とLDAを実行し、scatter-plotを作成
@app.callback([
    Output('intermediate-value', 'children'),
    Output('scatter-plot','figure'),
    Output("label-scatter-plot","children"),
    Output("label-bar-plot","children"),
    Output("mmm","children")
    ],
    [Input("button-exec","n_clicks"),
     Input("intermediate-value-pkl","children")],
    [State("dropdown-textcol","value"),
     State("dropdown-labelcol","value"),
     State("dropdown-topicnum","value"),
     State("input-stopword","children"),
     State("check-hinshi","children"),
     State('intermediate-value0', 'children')]
)
def update_result1(n,jsondata,text_col,label_col,topicnum,stop_word_c,hinshi_c,contents): 
#    text_col,label_col,id_col,topicnum,stop_word,hinshi="body","media","(指定なし)",6,[],["名詞/一般","名詞/固有名詞"]
    np.random.seed(1)
    # データは読みこまない
    #df = pd.read_csv("../ldadash/data/livedoornews.csv")
    # コールバックが起こるがまだデータはアップロードされていないので、例外処理を行う

    if contents is None and jsondata is None:
        raise dash.exceptions.PreventUpdate


    if jsondata is None:
        if stop_word_c is None:
            stop_word = []
        else:
            stop_word= stop_word_c[1]["props"]["value"].split(",")
        if hinshi_c is None:
            hinshi = ["名詞/一般","名詞/固有名詞"]
        else:
            hinshi = hinshi_c[1]["props"]["value"]
        plot_df,topic_word_df = util.train(contents,text_col,hinshi,stop_word,topicnum,label_col)

    else:
        data = json.loads(jsondata)
        plot_df,topic_word_df = pd.read_json(data["plot_df"]),pd.read_json(data["topic_word_df"])
        stop_word,hinshi = data["stop_word"],data["hinshi"]
        
        
    plot_df_by_topic = pd.concat([plot_df.groupby("topic").mean(),
                                  plot_df.groupby("topic").size().rename("size")],1).reset_index()
    plot_df_by_topic["size_plot"] = plot_df_by_topic["size"]/len(plot_df)*10000
    plot_df_by_topic["color"] = color_list[:topicnum]
    
    scatter_fig = px.scatter(plot_df_by_topic, x='x1', y = 'x2',
               color = 'topic',
               color_discrete_map={f"topic{i+1}":j for i,j in enumerate(color_list)},
               custom_data = ["topic"],
               hover_data={
                   "x1":False,
                   "x2":False,
                   "topic":True,
                   "size":True,
                   "size_plot":False
                   },
               text="topic",
               size="size_plot")
    scatter_fig.update_traces(textposition='top center')
    scatter_fig = scatter_fig.update_layout(clickmode='event+select',showlegend=False)    
    
    dicts = {"topic_word_df":topic_word_df.to_json(),
             "plot_df":plot_df.to_json(),
             "topicnum":topicnum,
             "stop_word":stop_word,
             "hinshi":hinshi
             }
    return [json.dumps(dicts),
            scatter_fig,
            "トピックの可視化",
            "トピック-単語分布(縦軸：単語、横軸：トピックに対する単語の構成割合)",""]
# トピック-単語分布を作成
# scatterで選んだトピック以外をグレーにする
@app.callback([
     Output('bar-plot', 'figure'),
    ],
    [
     Input('scatter-plot', 'selectedData'),
     Input('intermediate-value', 'children')],
     [State("dropdown-topicnum","value")]
)
def update_result2(selectdata,intermediate_value,topicnum):
    n_cluster=topicnum
    if intermediate_value is None:
        raise dash.exceptions.PreventUpdate
    data = json.loads(intermediate_value)
    topic_word_df = pd.read_json(data["topic_word_df"])
    
    colnum = (n_cluster+1)//2
    bar_fig = make_subplots(rows=2, cols=colnum,subplot_titles=[f"topic{i+1}" for i in range(n_cluster)])
    if selectdata:
        select_topic = selectdata["points"][0]["customdata"][0]
        color_list_barplot = {f"topic{i+1}":c if f"topic{i+1}"==select_topic else "lightgray" for i,c in enumerate(color_list)}
    else:
        color_list_barplot = {f"topic{i+1}":c for i,c in enumerate(color_list)}
    for i,(topic,gdf) in enumerate(topic_word_df.groupby("topic")):
        gdf = gdf.sort_values("score")
        bar_fig.add_trace(go.Bar(y=gdf["word"], x=gdf["score"],
                                 marker_color=color_list_barplot[topic],
                                 name=f"topic{topic}",
                                 orientation='h'),
                      row=i//colnum+1, col=i%colnum+1)
    bar_fig.update_layout(showlegend=False)
    return [bar_fig]

# テーブルを表示
# scatterで選んだトピックのみに絞る
@app.callback([
    Output('output-data', 'data'),
     Output('output-data', 'columns'),
     Output("output-caption","children"),
    Output("btn-pkl","style"),
     Output("btn-topicword","style"),
     Output("btn-doctopic","style")
    ],
    [Input('scatter-plot', 'selectedData'),
     Input('intermediate-value', 'children')]
)
def update_table(selectdata,intermediate_value):
    if intermediate_value is None :
        return [dash.no_update,dash.no_update,dash.no_update,
                {"display":"None"},{"display":"None"},{"display":"None"}]
    data = json.loads(intermediate_value)
    plot_df = pd.read_json(data["plot_df"])
    if selectdata:
        select_topic = selectdata["points"][0]["customdata"][0]
        plot_df = plot_df[plot_df["topic"]==select_topic].sort_values("score",ascending=False)
        output_caption = f"{select_topic}のテーブル"
    else:
        output_caption = "テーブル"
    plot_df = plot_df[["id","topic","score","label","text"]]
    columns=[{"name": i, "id": i} for i in plot_df.columns]
    data=plot_df.to_dict('records')

    return [data,columns,output_caption,{"display":"block"},{"display":"block"},{"display":"block"}]


# twitterデータをダウンロード
@app.callback(Output("download-twitter", "data"),
              [Input("btn-twitter", "n_clicks")],
              [State("intermediate-value0","children"),
               State("check-sjis","value")])
def downloadtwitter(n_nlicks,intermediate_value,check_sjis):
    if n_nlicks is None:
        raise dash.exceptions.PreventUpdate

    tweet_df = pd.read_json(intermediate_value)
    io1 = io.StringIO()
    enc = "cp932" if check_sjis else "utf8"
    tweet_df.to_csv(io1,index=False,encoding=enc)
    str_outputs = io1.getvalue().encode(enc,errors="ignore")
    return dict(content=base64.b64encode(str_outputs).decode(),
                 filename="tweet_output.csv",base64=True)


# 学習済みpickleデータをダウンロード
@app.callback(Output("download-pkl", "data"),
              [Input("btn-pkl", "n_clicks")],
              [State("intermediate-value","children")])
def downloadpicke(n_nlicks,intermediate_value):
    if n_nlicks is None:
        raise dash.exceptions.PreventUpdate

    data = json.loads(intermediate_value)
    io1 = io.BytesIO()
    pickle.dump(data, io1)
    return dict(content=base64.b64encode(io1.getvalue()).decode(),
                 filename="ldadash_output.pkl",base64=True)


# トピック-単語分布をダウンロード
@app.callback(Output("download-topicword", "data"),
              [Input("btn-topicword", "n_clicks")],
              [State("intermediate-value","children"),
               State("check-sjis","value")])
def downloadtopicword(n_nlicks,intermediate_value,check_sjis):
    if n_nlicks is None:
        raise dash.exceptions.PreventUpdate
    data = json.loads(intermediate_value)
    topic_word_df = pd.read_json(data["topic_word_df"])
    io1 = io.StringIO()
    enc = "cp932" if check_sjis else "utf8"
    topic_word_df.to_csv(io1,index=False,encoding=enc)    
    str_outputs = io1.getvalue().encode(enc,errors="ignore")
    return dict(content=base64.b64encode(str_outputs).decode(),
                 filename="tweet_output.csv",base64=True)

# 文書-トピック分布をダウンロード
@app.callback(Output("download-doctopic", "data"),
              [Input("btn-doctopic", "n_clicks")],
              [State("intermediate-value","children"),
               State("check-sjis","value")])
def downloaddoctopic(n_nlicks,intermediate_value,check_sjis):
    if n_nlicks is None:
        raise dash.exceptions.PreventUpdate
    data = json.loads(intermediate_value)
    plot_df = pd.read_json(data["plot_df"])
    io1 = io.StringIO()
    enc = "cp932" if check_sjis else "utf8"
    plot_df.to_csv(io1,index=False,encoding=enc)    
    str_outputs = io1.getvalue().encode(enc,errors="ignore")
    return dict(content=base64.b64encode(str_outputs).decode(),
                 filename="tweet_output.csv",base64=True)
if __name__ == '__main__':
    app.run_server(debug=False,host="0.0.0.0",port=8050)

