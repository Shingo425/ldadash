# -*- coding: utf-8 -*-
"""
自然言語処理における、前処理関係をまとめたファイル.
sentiment_analysis.pyでも利用している.
インポート（import preprocess as pp）して使う想定.
"""
import re
import pandas as pd
import numpy as np
import unicodedata
import MeCab
import emoji
from joblib import Parallel, delayed
import os

def convert_tz2JST(Series,before='UTC'):
    """
    時刻をJSTに変更
    
    Parameters
    ---------------------------
    series : pandas.Series
        datetime型
    before : str
        変換前のタイムゾーン
        
    """
    res = pd.DatetimeIndex(Series).tz_localize(before)\
            .tz_convert('Asia/Tokyo').tz_localize(None)
    return res

def categorize_tweet(Series):
    """
    RT,@(mention|replay),plainに分類する.
    ※stemming_tweetを掛ける前のSeriesに使う
    """
    pat_URL = re.compile(
        r"h?ttps?:?//[0-9a-zA-Z%@$/_:#&,;='\(\)\.\-\!\*\?\+]+ ?"
        )
    pat_res = re.compile("^@[a-zA-Z0-9_]+")
    pat_RT = re.compile("^RT @[@a-zA-Z0-9_ ]+:")
    temp = Series.map(lambda s:unicodedata.normalize('NFKC', s))
    res = pd.Series(
        np.full(temp.shape[0],'plain'),
        name='flag',
        index=temp.index
    )
    res = res.mask(temp.str.contains(pat_URL),"media")\
            .mask(temp.str.contains(pat_res),"@")\
            .mask(temp.str.contains(pat_RT),"RT")

    return res.astype("category")

def stemming_tweet(Series):
    """
    ツイートのステミングを行う.
    * URL,mention先,hashtagをマスクする
    * アルファベットを小文字に統一
    * 数字を0に統一
    * 改行を削除
    """
    reg_mention = re.compile("@[0-9a-zA-Z_]+")
    reg_URL = re.compile(
        r"h?ttps?:?//[0-9a-zA-Z%@$/_:#&,;='\(\)\.\-\!\*\?\+]+ ?"
    )
    reg_tag = re.compile(r"#.+?\s")
    res = Series.str.replace(reg_mention,"")\
                    .str.replace(reg_URL,"")\
                    .str.replace(reg_tag,"")\
                    .str.replace("[0-9]","0")\
                    .str.replace("\n","").str.lower()
    return res

def split_sentence_tweet(df,textcol=""):
    """
    ツイート（文章）を文に分割する
    
    Parameters
    -----------------------------------
    df : pandas.DataFrame
    textcol : str
        分析対象テキストの列名を指定
    
    Returns
    -----------------------------------
    res : pandas.DataFrame
        文を縦持ちしたデータフレーム。
        original_index列が入力元のindexと紐づく
    
    """
    # 区切り文字にする絵文字の辞書と記号のリストを準備
    set_emoji = set(emoji.UNICODE_EMOJI_ALIAS.keys())
    set_kigou = set(["☆","✩","★","♪","♬","♩","♫","♡","♥","◎"])
    set_trasn_delimiter = set_emoji.union(set_kigou)
    # 顔文字っぽい表現
    reg_kaomoji = re.compile(r"\([^ぁ-んァ-ネハ-ン]+\)")    
    # 区切り文字を指定
    punct = "|".join(["!",r"\?","。","…","#"])
    
    temp = df[textcol].map(
        lambda s: "".join("。" if c in set_trasn_delimiter else c for c in s))\
        .map(
        lambda s: re.sub(reg_kaomoji,"。",s))\
        .str.replace(r"\.{2,}","。")\
        .str.split(punct).map(
        lambda ls :"$".join([s for s in ls if s != ""])
        ).str.split("$",expand=True)
    temp.columns = [col +1 for col in temp.columns]
    temp.index = temp.index.set_names("original_index")
    temp = temp.reset_index().melt(
        id_vars="original_index",var_name="Sentence#",
        value_name="Sentence").dropna().sort_values(
        ["original_index","Sentence#"])
    res = df.merge(temp,how="inner",
                   left_index=True,
                   right_on="original_index").reset_index(drop=True)
    return res.loc[:,["original_index","Text","Sentence#","Sentence"]]

#neologd_正規化
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

# #形態素解析用モジュール
# def parse(tweet_temp,path=""):
#     """
#         mecabで形態素解析してリストで返す
#     input:
#         tweet_temp :文字列
#         path       :システム辞書のパスを指定（デフォルトから変更する場合）
#     output:
#         形態素解析結果のリスト
#     """
#     t = MeCab.Tagger(path)
#     temp1 = t.parse(tweet_temp)
#     temp2 = temp1.split("\n")[:-2]
#     t_list = [ [i.split("\t")[0]] + i.split("\t")[1].split(",") for i in temp2]
#     return t_list

# Taggerは作成時間がかかるため、関数の外で呼び出す形に変更

def parse(Text,param=None):
    """
    mecabで形態素解析してリストで返す

    Parameters
    ----------------------------------------
    Text : str
        形態素解析したい文字列
    param : str
        MeCav.Taggerに入れるパラメータ

    Returns
    ----------------------------------------
    t_list : 2D-list-of-str
        形態素解析結果のリスト
        一単語につき以下の情報を内側のリストで保有
        [ 表層形,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音]
    """
    if param:
        t_p = MeCab.Tagger(param)
    else:
        t_p = MeCab.Tagger()
    temp1 = t_p.parse(Text)
    temp2 = temp1.split("\n")[:-2]
    t_list = [ [i.split("\t")[0]] + i.split("\t")[1].split(",") for i in temp2]
    return t_list

def create_df_parse(df,id_col,text_col,n_jobs=-1,verbose=0,param=None):
    """
    df_parse(mecabの形態素解析結果をtidy-dataで保持したDataFrame)を作成

    Parameters
    -----------
    df : pandas.DataFrame
        分析対象列のテキスト列を持つデータフレーム
        ※テキスト列はpreprocess.normalize_neologd適用後想定
    id_col : str or list-of-str
        Textのunique IDの列名
    text_cols : str
        形態素解析対象列の列名
    n_jobs : int,defalut -1
        並列化コア数.defalutは最大
    verbose : int,defalut 0
        並列処理状況の可視化頻度.defalut表示なし
    param: str,None
        形態素解析時のMecab.Tagger()に入れるパラメータ

    Returns
    -----------
    df_parse : pandas.DataFrame
        mecabの形態素解析結果をtidy-dataで保持したDataFrame
    """
    if isinstance(id_col,str):
        id_col = [id_col]
    
    # 並列処理用関数
    def parallel_process(text,ids,param=None):
        if type(text)==str:
            lists = []
            for word in parse(text,param):
                try:
                    lists.append(
                        ids+[word[0], word[7], word[8], word[9],
                        word[1], word[2], word[3], word[4]]
                    )
                except:
                    pass

            df=pd.DataFrame(
                lists,
                columns= id_col + [
                    "word","genkei", "yomi","hatuon",
                    "hinshi", "shousai1","shousai2","shousai3"
                ]
            )
            return df
        else:
            return None

    # 並列処理実行
    if n_jobs != 1:
        df_parse = pd.concat(
            Parallel(n_jobs=n_jobs,verbose=verbose)([
                delayed(parallel_process)(text,ids.tolist())
                for text,ids in zip(
                    df[text_col].values,
                    df[id_col].values)
            ]), ignore_index=True)
    else:
        df_parse = pd.concat(
            [
                parallel_process(text,ids.tolist())
                for text,ids in zip(
                    df[text_col].values,
                    df[id_col].values)
            ], ignore_index=True)        
    return df_parse

def create_df_parse_work(df_parse,id_col,hinshi_list=None,add_word=None,stop_word=None,drop_dup=True):
    """
    df_parse_work（必要な品詞など、条件抽出した作業用DataFrame）を作成

    Parameters
    -----------
    df_parse : pandas.DataFrame
        mecabの形態素解析結果をtidy-dataで保持したDataFrame
    id_col : str or list-of-str
        Textのunique IDの列名
    hinshi_list : 2D-list-of-str
        抽出する品詞リスト.
        内側のリストに[品詞、詳細1]や[品詞、詳細1,詳細2]と記載
    add_word : list-of-str
        抽出する単語のリスト.品詞リストに存在しなくとも抽出したい単語を指定.
    stop_word : list-of-str
        抽出したくない単語のリスト.品詞リストに存在しても抽出したくない単語を指定.
    drop_dup : bool,defalut True
        一文書内に重複単語があった場合、削除するか。defalut削除する.

    Returns
    -----------
    df_parse_work : pandas.DataFrame
       必要な品詞など、条件抽出した作業用DataFrame
    """
    if isinstance(id_col,str):
        id_col = [id_col]    

    # add_wordの抽出
    if add_word:
        flag_extract = df_parse["genkei"].isin(add_word)
    else:
        flag_extract = pd.Series(False,index=df_parse.set_index(id_col).index).values

    # 品詞リストの抽出
    if hinshi_list:
        for hinshi in hinshi_list:
            if len(hinshi)==1:
                temp=df_parse["hinshi"]==hinshi[0]
            elif len(hinshi)==2:
                temp=(df_parse["hinshi"]==hinshi[0]) & (df_parse["shousai1"]==hinshi[1])
            elif len(hinshi)==3:
                temp=(df_parse["hinshi"]==hinshi[0]) & (df_parse["shousai1"]==hinshi[1]) \
                        & (df_parse["shousai2"]==hinshi[2])
            elif len(hinshi)==4:
                temp=(df_parse["hinshi"]==hinshi[0]) & (df_parse["shousai1"]==hinshi[1]) \
                        & (df_parse["shousai2"]==hinshi[2]) & (df_parse["shousai3"]==hinshi[3])
            else:
                raise ValueError("hinshi_listの指定が不正です。")
            flag_extract = flag_extract | temp
    else:
        # 品詞の指定がない⇒全品詞抽出
        flag_extract = np.full(flag_extract.shape[0],True)

    # stop_word除外とDataFrame抽出
    if stop_word:
        df_parse_work=df_parse[(~df_parse["genkei"].isin(stop_word)) \
                                 & (~df_parse["word"].isin(stop_word))
                                 & flag_extract]
    else :
        df_parse_work=df_parse[flag_extract]

    # drop_dup
    if drop_dup:
        df_parse_work=df_parse_work.drop_duplicates(
            subset=id_col+["genkei"],keep="first").reset_index(drop=True)

    return df_parse_work

def extract_df_parse_work(df_parse_work,in_word):
    """
    df_parse_work(またはdf_parse)から,指定単語のみ抽出する

    Parameters
    -----------
    df_parse_work : pandas.DataFrame
       必要な品詞など、条件抽出した作業用DataFrame

    Returns
    -----------
    df_parse_work : pandas.DataFrame
       必要な品詞など、条件抽出した作業用DataFrame
    """
    return df_parse_work[df_parse_work["genkei"].isin(in_word)]

def transform_bi_word(df_parse_work,id_col,bi_word,n_jobs=-1,verbose=0):
    """
    df_prase_work(またはdf_parse)内の指定単語をバイワード化する.

    Parameters
    -----------
    df_parse_work : pandas.DataFrame
       必要な品詞など、条件抽出した作業用DataFrame
    id_col : str or list-of-str
        Textのunique IDの列名
    bi_word : list-of-str
        バイワード化する単語のリスト.原形で指定.
        ※["ない"]なら「直前のword・ない」でバイワード化する
    n_jobs : int,defalut -1
        並列化コア数.defalutは最大
    verbos : int,defalut 0
        並列処理状況の可視化頻度.defalut表示なし

    Returns
    -----------
    df_parse_work : pandas.DataFrame
       必要な品詞など、条件抽出した作業用DataFrame
    """
    # 並列化用関数
    def make_bi_word(sdf,bi_word):
        
        doc=sdf["genkei"].copy().values
        outside_doc_col = sdf.columns.to_list()
        outside_doc_col.remove("genkei")
        outside_doc = sdf.copy()[outside_doc_col].values
        
        for n in range(len(doc)-1):
            if doc[n+1] in bi_word and doc[n]!="*":
                doc[n]=doc[n]+"・"+doc[n+1]
        
        bi_word = set(bi_word)
        retain_indices=[i for i, word in enumerate(doc) if not word in bi_word]
        
        doc=doc[retain_indices]
        outside_doc = outside_doc[retain_indices]
        
        res1=pd.Series(doc,name="genkei")
        res2=pd.DataFrame(outside_doc,columns=outside_doc_col)
        sres = pd.concat([res1,res2],axis=1).reindex(sdf.columns,axis=1)

        return sres

    g = df_parse_work.groupby(id_col)

    res = pd.concat(
            Parallel(n_jobs=n_jobs,verbose=verbose)([
                delayed(make_bi_word)(sdf,bi_word)
                for _,sdf in g
                ])
        )
    return res
