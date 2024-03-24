import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utilities_base import VAE, load_data

word_count = 680


def child_id_to_data(child_id: int) -> list[int]:
    child_id_dict, data = load_data(["child_id_dict", "data"])

    data_ids = [i[0] for i in child_id_dict[child_id]]
    return data[data_ids]


# 単語データを潜在変数に変換
def x_to_z(model: VAE, xs: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xs = torch.tensor(xs.astype(np.float32)).to(device)
        zs = model.encoder(xs)
        mu, log_var = zs.chunk(2, dim=1)
        z_points = mu.cpu()
        z_points = np.array(z_points)
        return z_points


# 潜在変数を単語データに変換
def z_to_x(model: VAE, z: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        z = torch.tensor(z.astype(np.float32)).to(device)
        xs = model.decoder(z)
        xs = np.array(xs.cpu())
        return xs


def category_to_num(categories: list[str]) -> list[int]:
    category_dict = load_data(["category_dict"])[0]
    nums = []
    if categories == ["all"]:
        return list(range(word_count))
    for category in categories:
        nums.extend([i[0] for i in category_dict[category]])
    return nums


def get_vocabulary(xs: np.ndarray, categories: list[str] = ["all"]) -> np.ndarray:
    nums = category_to_num(categories)
    return np.sum(xs[..., nums], axis=-1)


def plot_origin(model: VAE, ax: Axes) -> None:
    all_0s = np.zeros((1, 680))
    z0 = x_to_z(model, all_0s)
    all_1s = np.ones((1, 680))
    z1 = x_to_z(model, all_1s)
    ax.scatter(z0[:, 0], z0[:, 1], color="blue", label="all 0s", marker="x")
    ax.scatter(z1[:, 0], z1[:, 1], color="red", label="all 1s", marker="*")
    ax.legend()


def set_labels(ax: Axes, title: str = "") -> None:
    ax.set_xlabel(r"$z_{1}$")
    ax.set_ylabel(r"$z_{2}$")
    ax.set_title(title)


def plot_x(model: VAE, xs: np.ndarray, ax: Axes, color: str = "tab:blue") -> None:
    zs = x_to_z(model, xs)
    ax.scatter(zs[:, 0], zs[:, 1], s=0.2, color=color)

    plot_origin(model, ax)


def plot_x_with_age(model: VAE, data_ids: list[int], fig: Figure, ax: Axes) -> None:
    data, data_id_dict = load_data(["data", "data_id_dict"])
    xs = data[data_ids]
    ages = np.array([data_id_dict[i][1] for i in data_ids])
    zs = x_to_z(model, xs)
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=ages, cmap="turbo", s=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("age")

    plot_origin(model, ax)
    set_labels(ax, "age")


def plot_x_with_vocabulary(
    model: VAE,
    data_ids: list[int],
    fig: Figure,
    ax: Axes,
    categories: list[str] = ["all"],
) -> None:
    data = load_data(["data"])[0]
    xs = data[data_ids]
    vocabulary = get_vocabulary(xs, categories)
    zs = x_to_z(model, xs)
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=vocabulary, cmap="turbo", s=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("vocabulary")

    plot_origin(model, ax)
    set_labels(ax, ", ".join(categories))


# 潜在空間の格子点
def make_lattice_points(
    z1_start: float,
    z1_end: float,
    z2_start: float,
    z2_end: float,
    spacing: float,
) -> np.float32:
    z1 = np.arange(z1_start, z1_end + spacing, spacing)
    z2 = np.arange(z2_start, z2_end + spacing, spacing)

    return np.meshgrid(z1, z2)


def plot_vocabulary(
    model: VAE,
    z_mashgrid: list[np.ndarray, np.ndarray],
    fig: Figure,
    ax: Axes,
    categories: list[str] = ["all"],
) -> None:
    z1, z2 = z_mashgrid
    zs = np.dstack((z1, z2))
    xs = z_to_x(model, zs)
    vocabulary = get_vocabulary(xs, categories)
    cmap = ax.pcolormesh(z1, z2, vocabulary, cmap="turbo")
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label("vocabulary")
    set_labels(ax, ", ".join(categories))


def plot_arrow(model: VAE, data_ids: list[int], ax: Axes) -> None:
    data = load_data(["data"])[0]
    xs = data[data_ids]
    zs = x_to_z(model, xs)
    for z1, z2 in zip(zs[0:, :], zs[1:, :]):
        ax.annotate(
            "",
            xy=z2,
            xytext=z1,
            arrowprops=dict(arrowstyle="->", color="black"),
        )
