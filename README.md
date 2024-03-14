# コメント
このプロジェクトは未完成です。作成者はgitをよくわかっていないので不備があると思います。
# 環境構築
devcontainerを使わないのなら`conda`環境等で`environment.yml`から必要ライブラリをインストールすれば動くはずです。また、`Dockerfile`が私のPC内にあるローカルなイメージを参照しているので
```
docker build -t pytorch:cuda wordbank/.devcontainer/
```
などとやってもビルドはできないはずです。（あとで改善）


`.gitignore`の影響で`data/`と`tmp/`がないので手動でディレクトリを作ってください
# 実行の流れ
1. `how_to_get_data.md`を参考にして、`data/`フォルダに[Wordbank](https://wordbank.stanford.edu)のデータを`wordbank_instrument_data.csv`として入れる
2. `convert_data.ipynb`で前処理
3. `vae.ipynb`でvaeの学習
4. `analyze_model.ipynb`で学習後のモデルの分析
