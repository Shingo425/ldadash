# ldadash
ldadashは、テキストを入力してクラスタリングを行い、可視化するツールです。
自然言語処理のトピックモデル(lda)を使用しています。
また、Twitterデータ(要API申請)にも対応しているため、
気になった単語のTwitterの可視化も行うことができます。

## デモ
youtubeをご確認ください。

## 使用方法

### docker(推奨)
twitter取得機能なし
```bash
docker run -it -p 8050:8050 shingo425/ldadash
```
twitter取得機能あり(Twitter APIの申請が必要です。)
```bash
docker run -it -p 8050:8050 -e TWEET_API_KEY=<your tweet api key> -e TWEET_API_TOKEN=<your tweet api token> shingo425/ldadash
```

起動させた後、ブラウザに「localhost:8080」を入力してください。
※ dockerのメモリ割り当てが足りないと、メモリエラーになる場合があります。割り当てるメモリは大きめに設定してください。

### conda + git clone
```
conda create -n ldadash python=3.7
git clone https://github.com/Shingo425/ldadash.git
cd ldadash
pip install -r requirements.txt
python ldadash.py
```
※ twitter取得機能を使用する場合、以下の環境変数を設定してください。
```
TWEET_API_KEY=<your tweet api key>
TWEET_API_TOKEN=<your tweet api token>
```


