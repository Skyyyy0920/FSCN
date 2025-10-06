#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb


def find_key_col(cols):
    """从诊断表列名中鲁棒地找出 subjectkey 列。"""
    lowered = [c.lower() for c in cols]
    candidates = ["subjectkey", "src_subject_id", "subject_id", "subject", "id", "name"]
    for cand in candidates:
        if cand in lowered:
            return cols[lowered.index(cand)]
    raise KeyError("未在诊断表中找到 subjectkey / src_subject_id 等主键列，请检查列名。")


def load_labels(label_csv: Path):
    """读取多标签 CSV，返回：id -> 标签list；以及标签列顺序。"""
    df = pd.read_csv(label_csv)
    key_col = find_key_col(df.columns)
    # 其余列全部视为标签列（保留原顺序）
    label_cols = [c for c in df.columns if c != key_col]

    # 将 555/888 -> -1，其余尽量转为 int；缺失 NaN 也用 -1
    lab = (
        df[label_cols]
        .replace({555: -1, 888: -1})
        .astype("Int64")  # 先用可空整型
        .fillna(-1)
        .astype(int)  # 再转为普通 int
    )

    id2labels = dict(zip(df[key_col].astype(str), lab.values.tolist()))
    return id2labels, label_cols, key_col


def read_mat(csv_path: Path):
    """
    读取 FC/SC 邻接矩阵 csv 文件为 numpy 数组.
    兼容情况:
      - 没有表头 (纯数字矩阵)
      - 有列名和行名 (第0列是行名, 第一行是列名)
    """
    # 先读一次
    df = pd.read_csv(csv_path)

    # 如果第一列是非数字（通常是行名，比如 V1, V2...）
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        df = pd.read_csv(csv_path, index_col=0)

    arr = df.values.astype(float)
    return arr


def build_data_dict(fc_dir: Path, sc_dir: Path, label_csv: Path, out_pkl: Path):
    id2labels, label_cols, key_col = load_labels(label_csv)

    # 以文件名（不含扩展名）作为 ID，例如 NDAR_INV0A9K5L4R
    fc_files = {p.stem: p for p in fc_dir.glob("*.csv")}
    sc_files = {p.stem: p for p in sc_dir.glob("*.csv")}

    # 取三者交集（同时具备 FC、SC、标签）
    common_ids = sorted(set(fc_files) & set(sc_files) & set(id2labels))

    if not common_ids:
        print("没有可用的共同 ID，请检查文件夹与诊断表是否对应。", file=sys.stderr)
        return

    data_dict = {}
    skipped = {"missing_fc": [], "missing_sc": [], "missing_label": []}

    for idx, sid in tqdm(enumerate(common_ids), total=len(common_ids)):
        try:
            fc = read_mat(fc_files[sid])
        except Exception as e:
            skipped["missing_fc"].append(sid)
            continue
        try:
            sc = read_mat(sc_files[sid])
        except Exception as e:
            skipped["missing_sc"].append(sid)
            continue
        labels = id2labels.get(sid)
        if labels is None:
            skipped["missing_label"].append(sid)
            continue

        data_dict[idx] = {
            "FC": fc,
            "SC": sc,
            "label": list(map(int, labels)),  # 确保是 Python int
            "name": sid,
        }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 保存完成：{out_pkl}  | 样本数：{len(data_dict)}")
    if any(v for v in skipped.values()):
        print("⚠️ 跳过的 ID 概览：", {k: len(v) for k, v in skipped.items()})


def main():
    ap = argparse.ArgumentParser(description="Build data_dict.pkl from FC/SC and diagnosis CSV.")
    ap.add_argument("--fc_dir", type=Path, default=r"W:\Brain Analysis\data\data\raw\FC_ABCD",
                    help="FC 文件夹路径（内含若干 NDAR_*.csv）")
    ap.add_argument("--sc_dir", type=Path, default=r"W:\Brain Analysis\data\data\raw\SC_ABCD",
                    help="SC 文件夹路径（内含若干 NDAR_*.csv）")
    ap.add_argument("--label_csv", type=Path,
                    default=r"W:\Brain Analysis\data\data\raw\abcd_diagnosis_sum_baseline_n11976.csv",
                    help="abcd_diagnosis_sum 的 CSV 路径")
    ap.add_argument("--out", type=Path, default=Path(r"W:\Brain Analysis\data\data\data_dict.pkl"),
                    help="输出 pkl 路径")
    args = ap.parse_args()

    build_data_dict(args.fc_dir, args.sc_dir, args.label_csv, args.out)


if __name__ == "__main__":
    main()
