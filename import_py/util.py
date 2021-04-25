# -*- coding: utf-8 -*-
import io
#!/usr/bin/env python3
import base64


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import json
import pickle
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import re

import sys
sys.path.append("./import_py")
import preprocess
import lda



def gettweet(APIkey,APItoken,query,maxnum):
    token_url = 'https://api.twitter.com/oauth2/token'
    client = BackendApplicationClient(client_id=APIkey)
    oauth = OAuth2Session(client=client)
    try:
        token = oauth.fetch_token(token_url=token_url,client_id=APIkey,client_secret=APItoken)
    except:
        return False,""
        
    # エスケープ文字を呼び出し先で使うのでraw文字列で記入
    
    headers = {
        "Authorization": "Bearer {}".format(token.get('access_token')),
        "Content-Type": "application/json"}  
    tweets_list = []
    next_param = '?q={query}&count=100'.format(query=query)
    for i in range(maxnum//100):
        try:
            endpoint="https://api.twitter.com/1.1/search/tweets.json"
            endpoint += next_param
            res = requests.get(
                endpoint,
                headers=headers)
            
            if res.status_code == 200: #正常通信出来た場合
                searchTweets = json.loads(res.text) 
                tweets_list.append(pd.io.json.json_normalize(searchTweets["statuses"]))
                if "next_results" in searchTweets["search_metadata"].keys():
                    next_param=searchTweets["search_metadata"]["next_results"]
                else:
                    break
        except:
            pass    

    
    return True,pd.concat(tweets_list,ignore_index=True)
    


# アップロードしたファイルをデータフレームとして読み込むための関数
def parse_contents(content, filename, check_sjis):
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            str_inputs = io.StringIO(decoded.decode("cp932" if check_sjis else "utf-8"))
            data = pd.read_csv(str_inputs)
            trainedflg = False
            error_flg=0
            
        elif re.match("ldadash_output.*?\.pkl",filename):
            data = pickle.loads(decoded)
            trainedflg = True
            error_flg=0            
        else:
            data = None
            trainedflg = None
            error_flg=1
    except Exception as e:
        data = None
        trainedflg = None
        
        error_flg=2
        
    # データフレームの中身を送る
    return data,trainedflg,error_flg

def train(data,text_col,hinshi,stop_word,topicnum,label_col):
    
    df = pd.read_json(data)  
    df["id"] = np.arange(len(df))
    
    # テキスト前処理
    df["text"] = df[text_col]
    df["rep_tweet"] = df["text"].fillna("").map(preprocess.normalize_neologd)
#    logging.info("parse start")
    
    # MECABPATHの引数があればそれを指定
    MECABPATH = os.environ["MECABPATH"] if "MECABPATH" in dict(os.environ).keys() else None
    MECABPATH =  MECABPATH if MECABPATH else None

    df_parse = preprocess.create_df_parse(df,"id","rep_tweet",n_jobs=1,
                                          verbose=0,param=None)
#    logging.info("parse finish")
    hinshi_list = [h.split("/") for h in hinshi]
    df_parse_work = preprocess.create_df_parse_work(df_parse, "id",
                                                    hinshi_list=hinshi_list,
                                                    stop_word=stop_word)
    
    # ldaの実行
#    logging.info("lda start")
    lda_ins = lda.LDA_cluster(df_parse_work, "id", n_cluster=topicnum)
    lda_ins.fit()
#    logging.info("lda finish")
    # 整形
    plot_df = lda_ins._create_df_result()   
    X_embedded = TSNE(n_components=2).fit_transform(plot_df.iloc[:,2:].values)
    plot_df["x1"] = X_embedded[:,0]
    plot_df["x2"] = X_embedded[:,1]
    plot_df["text"] = df["text"]
    plot_df["id"] = df["id"]
    if label_col is None or label_col == "(指定なし)":
        plot_df["label"] = 0
    else:
        plot_df["label"] = df[label_col]
    
    plot_df = plot_df.rename(columns={"main_topic":"topic","main_topic_score":"score"})
    
    lists = []
    for i in range(topicnum):
        temp_df = pd.DataFrame(lda_ins.lda.show_topic(i),columns=["word","score"])
        temp_df["topic"] = f"topic{i+1}"
        lists.append(temp_df)
    topic_word_df = pd.concat(lists,ignore_index=True)
    return plot_df,topic_word_df
