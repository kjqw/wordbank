# データの取得方法

Wordbankからデータを取得する方法の例を以下に示します。

## 手動でのダウンロード方法

1. [Wordbankのデータページ](https://wordbank.stanford.edu/data/?name=instrument_data)にアクセス
2. **Language** で「English (American)」を選択
3. **Form** で「WG」を選択
4. 「Get Data!」ボタンをクリック
5. 表示されたページで「Download data」をクリックし、データをダウンロード
6. `data/`にダウンロードした`wordbank_instrument_data.csv`を配置

## コマンドラインを使用したダウンロード方法

linuxでコマンドラインから直接ダウンロードするにはプロジェクトのルートで例えば以下を実行

```sh
wget -O data/wordbank_instrument_data.csv https://wordbank-shiny.com/instrument_data/_w_638744f1daca870ab8b87084850b4ffc221372064f79be30/session/c0dcbc0571c2d93aaae7f4a210a3e865/download/download_data?w=638744f1daca870ab8b87084850b4ffc221372064f79be30
```
認証エラーの場合は以下を実行（セキュリティに注意）
```sh
wget --no-check-certificate -O data/wordbank_instrument_data.csv https://wordbank-shiny.com/instrument_data/_w_638744f1daca870ab8b87084850b4ffc221372064f79be30/session/c0dcbc0571c2d93aaae7f4a210a3e865/download/download_data?w=638744f1daca870ab8b87084850b4ffc221372064f79be30
```
