import pickle

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# GPUが利用可能な場合はGPUを使用し、そうでない場合はCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    """
    変分オートエンコーダー(VAE)の実装。

    エンコーダーとデコーダーのニューラルネットワークを含むVAEモデルです。
    エンコーダーは入力データを潜在空間のパラメータに変換し、デコーダーはこの潜在表現から元のデータを再構築します。

    Attributes
    ----------
    encoder : nn.Sequential
        エンコーダーネットワーク。
    decoder : nn.Sequential
        デコーダーネットワーク。
    """

    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(680, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 680),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        潜在変数をサンプリングする。

        平均と対数分散のパラメータを使用して、潜在空間からランダムな点をサンプリングします。

        Parameters
        ----------
        mu : torch.Tensor
            平均ベクトル。
        log_var : torch.Tensor
            対数分散ベクトル。

        Returns
        -------
        torch.Tensor
            サンプリングされた潜在変数。
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        モデルの順伝播を行う。

        入力データをエンコーダーに通し、潜在空間のパラメータ（平均と対数分散）を取得します。
        その後、このパラメータを使用して潜在空間からサンプリングし、サンプリングされた潜在変数をデコーダーに通して
        元のデータを再構成します。

        Parameters
        ----------
        x : torch.Tensor
            モデルに入力されるデータ。

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            再構成されたデータ、潜在空間の平均ベクトル、潜在空間の対数分散ベクトルを含むタプル。
        """

        x = self.encoder(x)
        mu, log_var = x.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var


def loss_function(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    VAEの損失関数を計算する。

    この関数は、再構成されたデータと元のデータの間の二項交差エントロピー損失と、
    潜在空間における正規分布とのKLダイバージェンスを計算して、VAEの全損失を算出します。

    Parameters
    ----------
    recon_x : torch.Tensor
        再構成されたデータ。モデルによって生成された出力。
    x : torch.Tensor
        元のデータ。損失を計算する際の基準となる入力データ。
    mu : torch.Tensor
        潜在空間における平均ベクトル。エンコーダーによって出力されます。
    log_var : torch.Tensor
        潜在空間における対数分散ベクトル。エンコーダーによって出力されます。

    Returns
    -------
    torch.Tensor
        計算された損失。再構成誤差とKLダイバージェンスの和。
    """

    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(
    model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer
) -> None:
    """
    VAEモデルの訓練を行う。

    モデルを訓練データで訓練し、誤差逆伝播を使用してモデルのパラメータを更新します。

    Parameters
    ----------
    model : nn.Module
        訓練するモデル。
    data_loader : DataLoader
        訓練データローダー。
    optimizer : optim.Optimizer
        オプティマイザ。

    Returns
    -------
    None
    """

    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    average_loss = train_loss / len(data_loader.dataset)
    return average_loss


def load_data(data_names: list[str]) -> list:
    """
    指定された名前のデータファイルをロードする。

    指定されたファイル名リストに基づき、pickleファイル形式で保存されたデータをロードし、リスト形式で返します。

    Parameters
    ----------
    data_names : list[str]
        ロードするデータファイルの名前のリスト。各ファイル名は拡張子を除く形式で指定します。

    Returns
    -------
    list
        ロードされたデータオブジェクトのリスト。データはファイル名リストに対応する順番で格納されます。

    Notes
    -----
    この関数は、`tmp`ディレクトリ内の`.pkl`ファイルを対象とします。指定されたファイルが存在しない場合、
    `pickle.load`メソッドによるエラーが発生する可能性があります。
    """

    data: list = []
    for name in data_names:
        filename = f"tmp/{name}.pkl"
        with open(filename, "rb") as file:
            data.append(pickle.load(file))
    return data
