# コメント
- `analyze_latent_space.ipynb`まで実行すれば、`sample_result`内のgifのようなものが得られるはず
- 関数のコメントが不十分で読みにくい
- 作成者はgitをよくわかっていないので不備がある可能性あり
- `.gitignore`の影響で`data/`や`tmp/`などがないので、エラーが出たら手動でディレクトリを作ってください
# 環境構築
devcontainerを使わないのなら`conda`環境等で`environment.yml`から必要ライブラリをインストールすれば動くはずです。また、`Dockerfile`が私のPC内にあるローカルなイメージを参照しているので
```
docker build -t pytorch:cuda wordbank/.devcontainer/
```
などとやってもビルドはできないはずです。（あとで改善）



# 実行の流れ
1. `how_to_get_data.md`を参考にして、`data/`フォルダに[Wordbank](https://wordbank.stanford.edu)のデータを`wordbank_instrument_data.csv`として入れる
2. `convert_data.ipynb`で前処理
3. `vae.ipynb`でvaeの学習
4. `analyze_model.ipynb`で学習後のモデルの分析
5. `analyze_latent_space.ipynb`で潜在空間の分析
