import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import nlopt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from src.data_prep import *
from src.nominal_speed_func import *
FT_TO_M = 0.3048

################################################################
#マクロ挙動の計算
################################################################
#　スナップショットからの密度、平均速度を計算する。
# 密度は車種ごとに計算。
#　スナップショットからの密度、平均速度を計算する。
# 密度は"車種ごと"に計算。

def compute_macroscopic_data(df, lanes=(2, 3, 4), classes=(2, 3), snapshot_step_frames=5,
                                    L_m=None, n_lanes=None, frame_interval_ms=100):
    """
    一定間隔(デフォルト5フレーム=0.5秒)でスナップショットを取り、
    マクロ密度(veh/km/lane)・平均速度(km/h)・流率(veh/h/lane)を計算する。

    df: trajectoryデータ(関数内でlanes/classesに絞る前のデータでも可)
    lanes: 対象車線
    classes: 対象車種(2=car, 3=truck)
    snapshot_step_frames: 何フレームおきにスナップショットを取るか(デフォルトは5フレーム(論文値))
    L_m: 実際のデータのy座標の範囲から自動計算する 
    n_lanes: 車線数(デフォルトは3)
    """
    #snaphshot間隔を時間に直す
    step_ms = frame_interval_ms * snapshot_step_frames  # 500ms

    # time_periodごとに、スナップショットを撮る時刻の集合を作成 
    valid_times = set()
    #time_periodごとにグループ分けし１ブロックずつループ処理
    for tp, group in df.groupby('time_period'):
        # 観測時間の最小値(t_min)・最大値(t_max)を取得
        t_min = int(group['Global_Time'].min())
        t_max = int(group['Global_Time'].max())
        
        # t_min始まりの0.5秒(500ms)ごとの時刻配列を作成し、集合に追加
        snapshot_times_tp = set(range(t_min, t_max + 1, step_ms))
        #上で作った集合に追加し、全時間区間のスナップショットを撮る時間リストを作成（globaltimeは重複しないためデータの重複もないはず）
        valid_times.update(snapshot_times_tp)

    # データを車線と車種で絞り込む
    target = df[df['Lane_ID'].isin(lanes) & df['v_Class'].isin(classes)].copy()

    #車線の長さ・車線数
    #L_mが引数で与えられなかった場合、ｙ座標の最大値と最小値の差から計算
    if L_m is None:
        L_m = (target['Local_Y'].max() - target['Local_Y'].min()) * FT_TO_M
        
    #n_lanesが引数で与えられなかった場合、laneの数を計算
    if n_lanes is None:
        n_lanes = len(lanes)

    L_km = L_m / 1000

    print(f"区間長: {L_m:.1f} m, 車線数: {n_lanes}")

    # gloobaltimeが以前作ったスナップショットを撮る時間リストに入っている行を抽出
    snap_rows = target[target['Global_Time'].isin(valid_times)]

    #抽出したスナップショットを撮るデータを時間と車種でグループ化し、グループごとに集計処理
    agg = snap_rows.groupby(['time_period', 'Global_Time', 'v_Class']).agg(
        n_vehicles=('Vehicle_ID', 'size'),#車種ごとの車両数　Vehicle_ID 列のデータ行数（size）を数えて、n_vehicles（車両数）という列名で保存
        speed_kmh=('v_Vel', lambda s: (s * FT_TO_M * 3.6).mean())
        #平均速度　feet/s -> m/s -> km/h　v_Vel 列を単位換算して平均値（mean）を計算し、speed_kmh（平均速度）という列名で保存　
    ).reset_index()

    agg['density_veh_km'] = agg['n_vehicles'] / (L_km * n_lanes)#マクロ密度の計算　面積で車両数を割る
    agg['flow_veh_h'] = agg['density_veh_km'] * agg['speed_kmh']#流量の計算　
    print("\n--マクロ密度・平均速度を計算--")

    return agg

#マクロデータの時間推移をプロットするコード
def plot_macro_timeseries(df_macro_data, col, ylabel, car_class=2, truck_class=3):
    """
    マクロデータ(compute_macroscopic_dataの出力)の指定した列を、time_periodごとに
    分けて、車種別(車・トラック)の折れ線グラフで表示する。

    df_macro_data: compute_macroscopic_data の返り値
    col: 表示する列名(例: 'density_veh_km', 'speed_kmh')
    ylabel: y軸のラベル
    car_class, truck_class: 車・トラックのv_Classの値

    その車種が0台のスナップショットは元データに行自体が存在しない(pandasの
    groupbyの仕様)ため、0で埋めてから描画する(埋めないと折れ線が飛んでしまう)。
    """
    class_labels = {car_class: 'Car', truck_class: 'Truck'}
    class_colors = {car_class: 'tab:blue', truck_class: 'tab:orange'}
    periods = sorted(df_macro_data['time_period'].unique())

    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 4 * len(periods)))
    if len(periods) == 1:
        axes = [axes]

    for ax, tp in zip(axes, periods):
        tpd = df_macro_data[df_macro_data['time_period'] == tp]
        t0 = tpd['Global_Time'].min()
        all_times = sorted(tpd['Global_Time'].unique())  # このtime_period内の全スナップショット時刻

        for cls in [car_class, truck_class]:
            d = tpd[tpd['v_Class'] == cls].set_index('Global_Time').reindex(all_times)
            d[col] = d[col].fillna(0.0)  # 0台で行が欠落している時刻を0で埋める
            elapsed = (d.index - t0) / 1000
            ax.plot(elapsed, d[col], label=class_labels[cls], color=class_colors[cls], linewidth=1)

        ax.set_title(f'time_period {tp}')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Elapsed time (s)')
    plt.tight_layout()
    plt.show()