# ビルド方法
# docker build -t 名前:タグ Dockerfileの親ディレクトリのパス
# 例
# docker build -t pytorch:cuda ~/python/pytorch/.devcontainer/

FROM python:mamba

RUN mamba create -y -n pytorch numpy scipy pandas matplotlib ipykernel pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pytorch:cuda
