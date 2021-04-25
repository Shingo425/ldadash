# -*- coding: utf-8 -*-
"""
LDA（トピック分析）を行う一連の処理をまとめたファイル.
※インポートしてLDAクラスのインスタンスメソッド____を実行する想定.on
"""
import pandas as pd
import numpy as np
import gensim
import matplotlib.pyplot as plt
import japanize_matplotlib
from wordcloud import WordCloud
import os

# 日本語フォントのpathを指定(wordclud文字化け対策)
fpath=os.path.relpath(
    os.path.join(
        os.path.dirname(__file__),
        "fonts",
        "ipaexg.ttf"
        ),
    os.getcwd()
    )

class LDA_cluster(object):
    """
    トピック分析にあたり、処理結果や関数を保持するクラス。

    Attributes
    ----------
    df_parse_work : pandas.DataFrame
        mecabの形態素解析結果をtidy-dataで保持したDataFrame.
        ※必要な品詞など、条件抽出した後のもの.
    df_result : pandas.DataFrame
        文書単位のトピック分類結果/トピック所属確率が格納されたデータフレーム.     
    id_col : str or list-of-str
        分析対象Textのunique IDの列名.
        ※id_colの値が同じ単語を「一文書」として扱う.
    n_cluster : int 
        分類するトピック数.
    use_tfidf : bool
        TF-IDF値を元にLDAをするか.
    lda : gensim.models.ldamodel.LdaModel
        gensimのldaモデルクラスのインスタンス.LDA作成に利用している.

    Methods
    ----------
    fit : LDAモデルをトレーニングし文書分類を実行.
    fit_load : 事前にトレーニングしたLDAモデルをロードし文書分類を実行.
    show_* : LDAモデルのトレーニング結果の確認用.
    save_model : トレーニングしたLDAモデルを保存.
    save_result_csv : df_result(文書分類結果を格納したデータフレーム)をcsv保存.

    See Also
    -----------
    gensim.models.LdaModel : LDA作成に利用している.
    """

    def __init__(self,df_parse_work,id_col,n_cluster,use_tfidf=False,keep_n=10000,no_below=1,no_above=0.8):
        """
        Parameters
        -----------
        df_parse_work : pandas.DataFrame
            mecabの形態素解析結果をtidy-dataで保持したDataFrame.
            ※必要な品詞など、条件抽出した後のもの.
        id_col : str
            分析対象Textのunique IDの列名.
            ※id_colの値が同じ単語を「一文書」として扱う.
        n_cluster : int
            分類するトピック数.
        use_tfidf : bool,defalut False
            TF-IDF値を元にLDAをするか.defalutは非使用(bag-of-words)
        keep_n : int,dafalut 10000
            単語出現頻度上位何単語までを使用するか.defalutは上位1万単語
        no_below : int,defalut 1
            出現文書数が閾値以上になる単語のみを使用.defalutは1文書以上(足切りなし)
        no_above : int,dafalut 0.8
            出現文書数/全文書数が閾値以下になるような単語のみを使用.
            defalutは8割以下
        """
        self.df_parse_work = df_parse_work
        if isinstance(id_col,str):
            id_col = [id_col]
        self.id_col = id_col
        self.n_cluster = n_cluster
        self.use_tfidf = use_tfidf
        self.keep_n = keep_n
        self.no_below = no_below
        self.no_above = no_above
        self.docs, self.index = self._transform_parse2docs() # docsとidexを作成

    def _transform_parse2docs(self):
        """
        df_parse_workからdocsとindexを作成
        """
        filter_word=self.df_parse_work["genkei"].value_counts()[:self.keep_n].index
        self.df_parse_work=self.df_parse_work[self.df_parse_work["genkei"].isin(filter_word)]
        df_docs=self.df_parse_work.groupby(self.id_col)["genkei"].apply(list)
        docs=df_docs.values.tolist()
        index=df_docs.index.values.tolist()
        return docs,index

    def fit(self,n_cluster=None,use_tfidf=None,no_below=None,no_above=None):
        """
        LDAモデルをトレーニング.
        ※引数を指定したらイニシャライズ時の値を更新してトレーニングできる.

        Parameters
        ------------
        n_cluster : int,defalut None
            分類するトピック数を更新する.
            defalutはNone(イニシャライズ時の値を利用)
        use_tfidf : bool,defalut None
            TF-IDF値を元にLDAをするかの値を更新する.
            defalutはNone(イニシャライズ時の値を利用)
        no_below : int,defalut None
            出現文書数が閾値以上になる単語のみを使用.defalutはNone(イニシャライズ時の値を利用)
        no_above : int,dafalut None
            出現文書数/全文書数が閾値以下になるような単語のみを使用.
            defalutはNone(イニシャライズ時の値を利用)
        """
        # 値を更新
        if n_cluster:
            self.n_cluster = n_cluster
        if use_tfidf:
            self.use_tfidf = use_tfidf
        if no_below:
            self.no_below = no_below
        if no_above:
            self.no_above = no_above
        # gensimを使ってトレーニング
        self.dictionary = gensim.corpora.Dictionary(self.docs)
        # 辞書を指定した閾値でフィルタリングしている
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]
        if self.use_tfidf:
            tfidf = gensim.models.TfidfModel(self.corpus)
            self.corpus = tfidf[self.corpus]
        self.lda = gensim.models.LdaModel(
                        corpus=self.corpus,
                        id2word=self.dictionary,
                        num_topics=self.n_cluster, 
                        minimum_probability=0.001,
                        passes=20, 
                        update_every=0, 
                        chunksize=10000,
                        random_state=1
                        )
        # 結果を格納
        self.df_result = self._create_df_result()

    def fit_load(self,path):
        """
        事前にトレーニングしたLDAモデルをロードし文書分類を実行.

        Parameters
        -----------
        path : str
            ロードするモデルのpath.
            ※末尾に何もついてない(.id2wordや.stateなどがついていない)ファイルを指定.
        """
        # gensimでモデルをロード
        self.lda = gensim.models.LdaModel.load(path)
        self.dictionary = self.lda.id2word
        self.n_cluster = self.lda.num_topics
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]
        if self.use_tfidf:
            tfidf = gensim.models.TfidfModel(self.corpus)
            self.corpus = tfidf[self.corpus]
        # 結果を格納
        self.df_result = self._create_df_result()

    def _create_df_result(self):
        """
        文書分類結果を格納したDataFrameを作成.

        Returns
        --------
        df_result : pandas.DataFrame
            文書単位のトピック分類結果/トピック所属確率が格納されたデータフレーム.
        """
        # 文書毎のトピック所属確率を格納したDataFrameを作成
        corpus_lda = self.lda[self.corpus]
        arr = gensim.matutils.corpus2dense(
                corpus_lda,
                num_terms=self.n_cluster
                ).T
        res1 = pd.concat([
            pd.DataFrame(arr),
            pd.DataFrame(self.index,columns=self.id_col)
            ],axis=1).set_index(self.id_col)
        res1.columns = ["topic{}".format(col+1) for col in res1.columns]
        self._df_result_topic_probability = res1
        # 文書毎のメイントピックとスコアを格納したDataFrameを作成
        res2 = res1.idxmax(axis=1).rename("main_topic")
        res3 = res1.max(axis=1).rename("main_topic_score")
        self._df_result_main_topic = pd.concat([res2,res3],axis=1)
        # 上記結果を結合して返す
        return self._df_result_main_topic.merge(
            self._df_result_topic_probability,
            how="inner",
            left_index=True,
            right_index=True)

    def save_model(self,path):
        """
        トレーニングしたLDAモデルを保存.

        Parameters
        -----------
        path : str
            保存するパスを文字列で指定.
        """
        self.lda.save(path)

    def save_result_csv(self,path,df_text=None):
        """
        df_result(文書分類結果を格納したデータフレーム)をcsv保存.
        ※df_textを指定した場合は結果を付与して出力.

        Parameters
        -----------
        path : str
            保存するパスを文字列で指定.
        df_text : pandas.DataFrame 
            分析対象テキスト列を持つデータフレーム.
            ※id_col列も持つ
        """
        if isinstance(df_text,pd.core.frame.DataFrame) and self.id_col in df_text.columns :
            out = df_text.merge(
                self.df_result,
                how="left",
                left_on=self.id_col,
                right_index=True
                )
        else:
            out = self.df_result

        out.to_csv(path,encoding="utf-8-sig")

    def show_topic_size(self):
        """
        トピックに所属する文書数を棒グラフで表示
        """
        temp = self._df_result_main_topic.groupby(
            "main_topic"
            ).size().rename("docment size")[::-1]
        ax = temp.plot.barh(
            title="各トピックに所属する文書数"
            )
        ax.set_xlabel("document size")
        plt.show()

    def show_topic_bar(self,topn=10,save_dir=None):
        """
        トピックの単語分布を棒グラフで表示.

        Parameters
        -----------
        topn : int,dafalut 10
            上位何単語を表示するか.defalutは10件.
        save_dir : str
            グラフ画像の保存ディレクトリ,defalutはNone.
            ※ディレクトリを指定したら直下にグラフ画像を保存する.
        """
        df_temp = pd.concat([
            pd.DataFrame(
                self.lda.show_topic(i,topn),
                columns=["word","ratio"]
                ).assign(topic="topic_{}".format(i+1))
            for i in range(self.n_cluster)
            ])
        x_lim = df_temp["ratio"].max() * 1.2

        figs = dict()
        # 表示
        for t,sdf in df_temp.groupby("topic"):
            temp = sdf[::-1].plot.barh(
                x="word",y="ratio",
                title="{}の単語の構成比率".format(t),
                xlim=(0,x_lim)
                )
            figs[t]=temp.get_figure()
            plt.show()
        # 保存
        if save_dir:
            for t,fig in figs.items():
                fig.savefig(
                    os.path.join(
                        save_dir,
                        "topic_bar_{}.png".format(t)
                        )
                    )

    def show_topic_wordcloud(self,topn=20,save_dir=None):
        """
        トピックの単語分布をワードクラウドで表示する.

        Parameters
        -----------
        topn : int,dafalut 20
            上位何単語を表示するか.defalutは20件.
        save_dir : str
            グラフ画像の保存ディレクトリ,defalutはNone.
            ※ディレクトリを指定したら直下にグラフ画像を保存する.
        """
        df_temp = pd.concat([
            pd.DataFrame(
                self.lda.show_topic(i,topn),
                columns=["word","ratio"]
                ).assign(topic="topic_{:02}".format(i))
            for i in range(self.n_cluster)
            ])
        x_lim = df_temp["ratio"].max() * 1.2

        figs = dict()
        # 表示
        for t,sdf in df_temp.groupby("topic"):
            wc = sdf.set_index("word")["ratio"]
            wordc = WordCloud(
                background_color='white',
                font_path=fpath,
                min_font_size=15,
                max_font_size=100,
                width=500,
                height=300,
                prefer_horizontal=1
                )

            wordc.generate_from_frequencies(wc)
            fig = plt.figure(figsize=[10,10])
            plt.imshow(wordc,interpolation='bilinear')
            plt.title("{}の構成単語".format(t))
            plt.axis("off")
            figs[t] = fig
            plt.show()

        # 保存
        if save_dir:
            for t,fig in figs.items():
                fig.savefig(
                    os.path.join(
                        save_dir,
                        "topic_wordcloud_{}.png".format(t)
                        )
                    )    

    def show_symbolic_text(self,df_text,text_col,topn=10):
        """
        各トピックの象徴的な文書を表示する.

        Parameters
        -----------
        df_text : pandas.DataFrame 
            分析対象テキスト列を持つデータフレーム.
            ※id_col列も持つ
        text_col : str
            テキスト列の列名.
        topn : int,dafalut 10
            トピック毎に何件の文書を表示するか.defalutは10件.
        """
        df_text_add = df_text.merge(
            self.df_result,
            how="inner",
            left_on=self.id_col,
            right_index = True,
            ).loc[:,[text_col,"main_topic","main_topic_score"]]

        g = df_text_add.groupby("main_topic")

        for i in range(self.n_cluster):
            gname = "topic_{:02}".format(i)
            print(
                "*************************************"
                "*************************************"
                "*************************************"
                )
            print(gname)
            print(
                "\n".join(
                    self.lda.show_topics()[i][1].split(" + ")
                    ),
                "\n"
                )
            temp = g.get_group(gname).drop("main_topic", axis=1)
            temp = temp.sort_values(
                "main_topic_score",
                ascending=False
                ).head(topn)
            for c,(text,score) in enumerate(temp.values):
                print(
                    "{} top{} score: {:.5f}".format(gname,c+1,score),
                    text,
                    "\n",
                    sep="\n"
                    )
            




