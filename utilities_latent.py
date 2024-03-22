import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

import utilities_plot as up
from utilities_base import VAE, load_data


def get_symmetric_points(P1: np.ndarray, P2: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    与えられた2点P1, P2を通る直線に対して、点Qの対称点を計算する。

    Parameters:
    P1 (np.ndarray): 直線を定義する最初の点。
    P2 (np.ndarray): 直線を定義する2番目の点。
    Q (np.ndarray): 対称点を見つけたい点。

    Returns:
    np.ndarray: 点Qの対称点
    """
    # P1とP2を結ぶベクトルを計算
    P1P2_vector = P2 - P1
    # P1からQへのベクトルを計算
    P1Q_vector = Q - P1

    # P1QベクトルがP1P2ベクトルに投影されたときのスカラーを計算
    projection_scalar = np.dot(P1Q_vector, P1P2_vector) / np.dot(
        P1P2_vector, P1P2_vector
    )
    # Qの対称点を求めるためのベクトルを計算
    symmetric_vector = projection_scalar * P1P2_vector - P1Q_vector

    # Qの直線に対する対称点を計算
    symmetric_point = Q + symmetric_vector * 2

    return symmetric_point


def get_expectation_from_zs_with_category(
    model: VAE, zs: np.ndarray
) -> dict[str, np.ndarray]:
    xs = up.z_to_x(model, zs)
    index_dict = {}
    category_dict = load_data(["category_dict"])[0]
    for key, val in category_dict.items():
        index_min = min([i[0] for i in val])
        index_max = max([i[0] for i in val])
        index_dict[key] = (index_min, index_max)
    category_vocabulary = {}
    for key, val in index_dict.items():
        category_vocabulary[key] = xs[:, val[0] : val[1] + 1]
    return category_vocabulary


def set_config_expectation_plot(image_num: int) -> dict[str, tuple[Figure, Axes]]:
    figs = {}
    for i in range(image_num):
        figs[f"expectation_{i}"] = plt.subplots()
    for fig, ax in figs.values():
        ax.set_xticks([j for j in range(22)])
        ax.set_xticklabels([chr(65 + j) for j in range(22)])
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Expectation")
        for i, v in enumerate(ax.get_xticklabels()):
            v.set_color(f"C{i}")
    return figs


def plot_expectation_with_category(
    category_vocabulary: dict[str, np.ndarray], figs: dict[str, tuple[Figure, Axes]]
) -> None:
    for i, (key, val) in enumerate(category_vocabulary.items()):
        for j in range(val.shape[0]):
            figs[f"expectation_{j}"][1].scatter(
                [i] * val.shape[1], val[j, :], color=f"C{i}"
            )
            plt.close()


def save_expectation_plot_with_category(
    model: VAE, zs: np.ndarray, output_path: Path
) -> None:
    category_vocabulary = get_expectation_from_zs_with_category(model, zs)
    image_num = category_vocabulary["sounds"].shape[0]
    figs = set_config_expectation_plot(image_num)
    plot_expectation_with_category(category_vocabulary, figs)
    for i in range(image_num):
        figs[f"expectation_{i}"][0].savefig(output_path / f"expectation_{i}.png")


def get_expectation_diff_from_zs_with_category(
    model: VAE, zs1: np.ndarray, zs2: np.ndarray
) -> dict[str, np.ndarray]:
    category_vocabulary1 = get_expectation_from_zs_with_category(model, zs1)
    category_vocabulary2 = get_expectation_from_zs_with_category(model, zs2)
    category_vocabulary_diff = {}
    for key in category_vocabulary1.keys():
        category_vocabulary_diff[key] = (
            category_vocabulary2[key] - category_vocabulary1[key]
        )

    return category_vocabulary_diff


def set_config_expectation_diff_plot(
    image_num: int, point1: str, point2: str
) -> dict[str, tuple[Figure, Axes]]:
    figs = {}
    for i in range(image_num):
        figs[f"expectation_diff_{i}"] = plt.subplots()
    for fig, ax in figs.values():
        ax.set_xticks([j for j in range(22)])
        ax.set_xticklabels([chr(65 + j) for j in range(22)])
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(f"Expectation Differece {point1} - {point2}")
        ax.axhline(y=0, color="grey", linestyle="--")
        for i, v in enumerate(ax.get_xticklabels()):
            v.set_color(f"C{i}")
    return figs


def plot_expectation_diff_with_category(
    category_vocabulary: dict[str, np.ndarray], figs: dict[str, tuple[Figure, Axes]]
) -> None:
    for i, (key, val) in enumerate(category_vocabulary.items()):
        for j in range(val.shape[0]):
            figs[f"expectation_diff_{j}"][1].scatter(
                [i] * val.shape[1], val[j, :], color=f"C{i}"
            )
            plt.close()


def save_expectation_diff_plot_with_category(
    model: VAE,
    zs1: np.ndarray,
    zs2: np.ndarray,
    point1: str,
    point2: str,
    output_path: Path,
) -> None:
    category_vocabulary = get_expectation_diff_from_zs_with_category(model, zs1, zs2)
    image_num = category_vocabulary["sounds"].shape[0]
    figs = set_config_expectation_diff_plot(image_num, point1, point2)
    plot_expectation_diff_with_category(category_vocabulary, figs)
    for i in range(image_num):
        figs[f"expectation_diff_{i}"][0].savefig(
            output_path / f"expectation_diff_{i}.png"
        )


def save_zs_plot(model: VAE, zs: np.ndarray, output_path: Path) -> None:
    data = load_data(["data"])[0]
    figs = {}
    figs["points"] = plt.subplots()
    up.plot_x(model, data, figs["points"][1])
    for i, v in enumerate(zs):
        sc = figs["points"][1].scatter(v[0], v[1], color="tab:orange", label=f"point")
        figs["points"][1].legend()
        figs["points"][0].savefig(output_path / f"point_{i}.png")
        sc.remove()
        plt.close()


def save_zs_diff_plot(
    model: VAE,
    zs1: np.ndarray,
    zs2: np.ndarray,
    point1: str,
    point2: str,
    output_path: Path,
) -> None:
    data = load_data(["data"])[0]
    figs = {}
    figs["points"] = plt.subplots()
    up.plot_x(model, data, figs["points"][1])
    for i in range(len(zs1)):
        sc1 = figs["points"][1].scatter(
            *zs1[i], color="tab:orange", label=point1, marker="o"
        )
        sc2 = figs["points"][1].scatter(
            *zs2[i], color="tab:green", label=point2, marker="+"
        )
        figs["points"][1].legend()
        figs["points"][0].savefig(output_path / f"point_diff_{i}.png")
        sc1.remove()
        sc2.remove()
        plt.close()


def make_combined_gif(
    image_dirs1: list[str], image_dirs2: list[str], output_path: Path
) -> None:
    new_frames = []

    # 画像ファイルからフレームを読み込み、横に並べて新しいフレームを作成
    for img_file_expectation, img_file_point in zip(image_dirs1, image_dirs2):
        img_expectation = Image.open(img_file_expectation)
        img_point = Image.open(img_file_point)

        # 両画像のサイズを取得
        width_e, height_e = img_expectation.size
        width_p, height_p = img_point.size

        # 新しい画像のサイズを計算（幅は2つの画像のうち大きい方、高さは合計）
        new_width = max(width_e, width_p)
        new_height = height_e + height_p

        # 新しい画像を作成
        new_img = Image.new("RGB", (new_width, new_height))

        # 新しい画像に元の画像を貼り付け
        new_img.paste(img_expectation, (0, 0))
        new_img.paste(img_point, (0, height_e))

        # 新しいフレームをリストに追加
        new_frames.append(new_img)

    # 新しいフレームのリストからGIFを作成
    new_frames[0].save(
        output_path,
        save_all=True,
        append_images=new_frames[1:],
        duration=100,  # 各フレームの表示時間（ミリ秒）
        loop=0,  # ループ回数（0は無限ループ）
    )
