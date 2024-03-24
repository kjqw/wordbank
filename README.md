# コメント
- `analyze_latent_space.ipynb`まで実行すれば、`sample_result`内のgifのようなものが得られるはず
- 作成者はgitをよくわかっていないので不備がある可能性あり
# 環境構築
`conda`環境等で`environment.yml`から必要ライブラリをインストールすれば動くはずです。

# 実行の流れ
1. データの準備 
- `how_to_get_data.md`を参考にして、`data/`フォルダに[Wordbank](https://wordbank.stanford.edu)のデータを`wordbank_instrument_data.csv`として入れる
2. データの前処理
- `convert_data.ipynb`を実行
3. VAEの学習
- `vae.ipynb`を実行、もしくは学習済みデータを使用
- **補足**: `sample_result/best_model.pth`を`tmp/`に入れて作成者と同じ学習済みモデルを使うこともできる。潜在空間の分析の際の点の位置などがずれないで済む。
4. 学習後のモデルの分析
- `analyze_model.ipynb`を実行
5. 潜在空間の分析
- `analyze_latent_space.ipynb`を実行
