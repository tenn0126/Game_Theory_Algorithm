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
from scipy.optimize import brentq

from src.data_prep import *
from src.nominal_speed_func import *
from src.macroscopic_data import *

#################################################################################
# 論文Eq(7)の残差を求める関数（その時のu*がどれくらい式(7)の等式を満たしているか）
# # ※ underwood_density_speed, logistic_density_speed は
#    nominal_speed_func.py に定義済みのものをそのまま使う
# a1=1（cc追従）、 b2=1（tt追従）
def _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1):
    rho_tot = rho1 + rho2

    #u*の時のu1,u2の逆関数を計算　u1_invは密度
    u1_inv = logistic_density_speed(
        u, car_params['ub'], car_params['uf'], car_params['rhoc'],
        car_params['theta1'], car_params['theta2']
    )
    u2_inv = underwood_density_speed(u, truck_params['uf'], truck_params['rhoc'])

    #sigma_j=1^j=2を崩して書いた形
    term1 = (rho1 / u1_inv) * (rho1 / 1.0 + rho2 / a2)
    term2 = (rho2 / u2_inv) * (rho1 / b1 + rho2 / 1.0)

    #eq(7)の左辺と右辺の差を返す。0に近いほど式(7)を満たす
    return (term1 + term2) / rho_tot - 1.0

#brent法でu*を求める関数
def solve_1pipe_speed(rho1, rho2, car_params, truck_params, a2, b1,
                       u_lower=None, u_upper=None, margin=1e-3):
    """
    論文Eq(7)をBrent法（scipy.optimize.brentq）で解き、1-pipe速度u*を求める。

    rho1, rho2: マクロ密度（veh/km）。クラス1=車、クラス2=トラック
    car_params: fit_nominal_speed_logistic の戻り値（C-C）
    truck_params: fit_nominal_speed_underwood の戻り値（T-T）
    a2: fit_scaling_parameter_logistic の戻り値の'scaling_param'
    b1: fit_scaling_parameter_underwood の戻り値の'scaling_param'
    u_lower, u_upper: 探索区間。省略時は車のub・両クラスのufから自動決定
    """
    if rho1 + rho2 <= 0:
        return np.nan  # 車もトラックもいないスナップショットは対象外

    if u_lower is None or u_upper is None:
        lowers, uppers = [], []
        if rho1 > 0:
            #lowers.append(car_params['ub'] + margin)
            lowers.append(margin)#より頑健な下限　
            # 車の"uf"は密度→-∞の極限値で、密度0でも到達できない。
            # 密度0での実際の到達可能な最高速度 f(0) を上限にする。
            car_speed_at_zero = logistic_speed_density(
                0.0, car_params['ub'], car_params['uf'], car_params['rhoc'],
                car_params['theta1'], car_params['theta2']
            )
            uppers.append(car_speed_at_zero - margin)
        if rho2 > 0:
            lowers.append(margin)#下限は定義できない
            uppers.append(truck_params['uf'] - margin)  # Underwoodはg(0)=ufなので問題なし
        if u_lower is None:
            u_lower = max(lowers)
        if u_upper is None:
            u_upper = min(uppers)

    #uだけを引数に持つ関数を新たに作る
    f = lambda u: _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1)
    #ブレント法でf(u)=0となるuを求める
    return brentq(f, u_lower, u_upper)


# マクロ密度のスナップショット表全体(rho1,rho2の組み合わせ)に対して1-pipe速度を計算
def compute_1pipe_speed_for_snapshots(df_macro, car_params, truck_params, a2, b1,
                                        car_class=2, truck_class=3):
    """
    df_macro: compute_macroscopic_data の返り値
              （time_period, Global_Time, v_Class, density_veh_km, ... を含む）
    """
    #縦持ちのデータを横並びに変換　
    pivot = df_macro.pivot_table(
        index=['time_period', 'Global_Time'],#timeperiodとGlobal_Timeが同じなら同じ行にまとめる
        columns='v_Class',#クラスを新たな列名にする
        values=['density_veh_km', 'speed_kmh']#その時の密度を入れる
    )

    #2段の列名を1段のわかりやすい名前に変換
    col_map = {
        ('density_veh_km', car_class): 'rho1',
        ('density_veh_km', truck_class): 'rho2',
        ('speed_kmh', car_class): 'u1_obs',    #車の観測平均速度 ũ1
        ('speed_kmh', truck_class): 'u2_obs',  #トラックの観測平均速度 ũ2
    }
    pivot.columns = [col_map[c] for c in pivot.columns]

    #密度は「いない=0」でよいが、速度は「いない=定義できない」のでNaNのまま残す
    pivot[['rho1', 'rho2']] = pivot[['rho1', 'rho2']].fillna(0.0)
    pivot = pivot.reset_index()

    #1行ずつ1-pipe速度を計算する関数
    #もし、下限と上限の範囲で0をはさめず、解が存在しない場合(ブレント法のエラーになる場合)も止まらないようにnanを返す
    def _solve_row(row):
        try:
            return solve_1pipe_speed(row['rho1'], row['rho2'], car_params, truck_params, a2, b1)
        except ValueError:
            return np.nan

    pivot['rho_tot'] = pivot['rho1'] + pivot['rho2']
    pivot['u_1pipe'] = pivot.apply(_solve_row, axis=1)#すべての行に適応　結果を新しい列として追加

    print(f"1-pipe速度を計算: {len(pivot):,}スナップショット中、"
          f"{pivot['u_1pipe'].notna().sum():,}件で解を取得")

    return pivot

KM_TO_MILE = 1.60934  # 1 mile = 1.60934 km

#1-pipe速度のヒートマップを作成するための共通処理関数
def _bin_and_aggregate_1pipe(df, rho1_col, rho2_col, bin_width1, bin_width2,
                               max_rho1, max_rho2, min_count, value_col='u_1pipe'):
    """rho1_col, rho2_colの密度をビンに区切り、各ビンのvalue_colの平均をまとめる共通処理。"""
    bins1 = np.arange(0, max_rho1 + bin_width1, bin_width1)
    bins2 = np.arange(0, max_rho2 + bin_width2, bin_width2)

    df = df.copy()
    df['_bin1'] = pd.cut(df[rho1_col], bins=bins1, include_lowest=True)
    df['_bin2'] = pd.cut(df[rho2_col], bins=bins2, include_lowest=True)

    cats1 = df['_bin1'].cat.categories
    cats2 = df['_bin2'].cat.categories

    grid = df.groupby(['_bin2', '_bin1'], observed=True).agg(
        mean_val=(value_col, 'mean'), count=(value_col, 'size')  # ← 'u_1pipe'決め打ちをvalue_colに
    ).reset_index()

    heat = grid.pivot(index='_bin2', columns='_bin1', values='mean_val').reindex(index=cats2, columns=cats1)  # mean_u→mean_val
    counts = grid.pivot(index='_bin2', columns='_bin1', values='count').reindex(index=cats2, columns=cats1)
    heat = heat.where(counts >= min_count)

    return bins1, bins2, heat


#論文図11に類似した1-pipe速度のヒートマップを作成する関数
def plot_1pipe_speed_heatmap(df_1pipe, bin_width1=1.0, bin_width2=1.0,
                               max_rho1=None, max_rho2=None, min_count=3):
    """
    1-pipe速度のヒートマップを、km単位版とmile単位版(論文の単位)を横並びで表示する。

    bin_width1, bin_width2はkm単位での指定値。mile版は、rho1・rho2を先に
    mile換算してから、ビン幅もmile換算した値(bin_width * KM_TO_MILE)で、
    km版とは独立にビン分割・集計し直す(km版の結果の軸ラベルを
    書き換えているだけではない)。

    max_rho1, max_rho2を省略した場合、実データの最大値(km単位)から自動で決める。
    min_count未満のビンはデータ不足として表示しない(論文のgray areaに相当)。
    """
    df = df_1pipe.dropna(subset=['u_1pipe']).copy()

    if max_rho1 is None:
        max_rho1 = df['rho1'].max()
    if max_rho2 is None:
        max_rho2 = df['rho2'].max()

    # --- km版：km単位のrho1, rho2をそのままビン分割 ---
    bins1_km, bins2_km, heat_kmh = _bin_and_aggregate_1pipe(
        df, 'rho1', 'rho2', bin_width1, bin_width2, max_rho1, max_rho2, min_count
    )

    # --- mile版：rho1, rho2をmileに変換してから、mile換算のビン幅で独立に再分割 ---
    df['rho1_mile'] = df['rho1'] * KM_TO_MILE
    df['rho2_mile'] = df['rho2'] * KM_TO_MILE
    bin_width1_mile = bin_width1 * KM_TO_MILE
    bin_width2_mile = bin_width2 * KM_TO_MILE
    max_rho1_mile = max_rho1 * KM_TO_MILE
    max_rho2_mile = max_rho2 * KM_TO_MILE

    bins1_mile, bins2_mile, heat_kmh_in_mile_bins = _bin_and_aggregate_1pipe(
        df, 'rho1_mile', 'rho2_mile', bin_width1_mile, bin_width2_mile,
        max_rho1_mile, max_rho2_mile, min_count
    )
    heat_mph = heat_kmh_in_mile_bins / KM_TO_MILE  # 速度はkm/h -> mph に変換

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    mesh0 = axes[0].pcolormesh(bins1_km, bins2_km, heat_kmh.values, cmap='plasma', vmin=0, shading='flat')
    fig.colorbar(mesh0, ax=axes[0], label='1-pipe speed (km/h)')
    axes[0].set_xlabel('Density class 1 - Car (veh/km)')
    axes[0].set_ylabel('Density class 2 - Truck (veh/km)')
    axes[0].set_title('1-pipe Speed Heatmap (km)')

    mesh1 = axes[1].pcolormesh(bins1_mile, bins2_mile, heat_mph.values, cmap='plasma', vmin=0, shading='flat')
    fig.colorbar(mesh1, ax=axes[1], label='1-pipe speed (mph)')
    axes[1].set_xlabel('Density class 1 - Car (veh/mile)')
    axes[1].set_ylabel('Density class 2 - Truck (veh/mile)')
    axes[1].set_title('1-pipe Speed Heatmap (mile, paper units)')

    plt.tight_layout()
    plt.show()

    return heat_kmh, heat_mph

#############################################################################
#1-pipe速度の時間変化をプロットする関数
def plot_1pipe_timeseries(df_1pipe, col='u_1pipe', ylabel='1-pipe speed (km/h)', color='tab:green'):
    periods = sorted(df_1pipe['time_period'].unique())
    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 4 * len(periods)))
    if len(periods) == 1:
        axes = [axes]

    for ax, tp in zip(axes, periods):
        tpd = df_1pipe[df_1pipe['time_period'] == tp].sort_values('Global_Time')
        t0 = tpd['Global_Time'].min()
        elapsed = (tpd['Global_Time'] - t0) / 1000
        ax.plot(elapsed, tpd[col], color=color, linewidth=1)
        ax.set_title(f'time_period {tp}')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)  # y軸を0から始める
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Elapsed time (s)')
    plt.tight_layout()
    plt.show()

def plot_timeseries_overlay(df, col, ylabel, title=None):
    """
    dfの指定した列を、time_periodごとに色分けして1枚のグラフに重ねて表示する
    (各time_period自身の開始時刻からの経過秒数を横軸にする)。

    df: time_period, Global_Time, colを含むデータフレーム
    col: 表示する列名
    ylabel: y軸のラベル
    title: グラフのタイトル(省略時はcol名)
    """
    df_sorted = df.sort_values(['time_period', 'Global_Time'])

    fig, ax = plt.subplots(figsize=(14, 5))

    for tp, group in df_sorted.groupby('time_period'):
        elapsed_sec = (group['Global_Time'] - group['Global_Time'].min()) / 1000
        ax.plot(elapsed_sec, group[col], label=f'time_period {tp}', alpha=0.8, linewidth=1)

    ax.set_xlabel('Elapsed time (s, from each time_period start)')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'{col} time series')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_timeseries_overlay(df, col, ylabel, title=None, hline=None, ylim=None):
    """
    dfの指定した列を、time_periodごとに色分けして1枚のグラフに重ねて表示する
    (各time_period自身の開始時刻からの経過秒数を横軸にする)。

    df: time_period, Global_Time, colを含むデータフレーム
    col: 表示する列名
    ylabel: y軸のラベル
    title: グラフのタイトル(省略時はcol名)
    hline: 指定すると、その値に水平の目印線を引く(例: sの0基準線)
    ylim: 指定すると、y軸の表示範囲を(min, max)で固定する
    """
    df_sorted = df.sort_values(['time_period', 'Global_Time'])

    fig, ax = plt.subplots(figsize=(14, 5))

    for tp, group in df_sorted.groupby('time_period'):
        elapsed_sec = (group['Global_Time'] - group['Global_Time'].min()) / 1000
        ax.plot(elapsed_sec, group[col], label=f'time_period {tp}', alpha=0.8, linewidth=1)

    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=1)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel('Elapsed time (s, from each time_period start)')
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'{col} time series')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#############################################################################
#最小道路占有
#############################################################################
def compute_min_road_share(df_1pipe, car_params, truck_params):
    """
    論文Eq(4)：最小道路占有 pi* = rho_i / ui^-1(u*) を計算する。

    ui^-1は「クラス自身の名目速度関数」の逆関数（_eq7_residual内のu1_inv,
    u2_invと同じ呼び出し。a2・b1によるスケーリングはここでは使わない）。

    df_1pipe: compute_1pipe_speed_for_snapshots の返り値
              （rho1, rho2, u_1pipe列を含む）
    car_params: C-Cの名目速度関数パラメータ
    truck_params: T-Tの名目速度関数パラメータ
    """
    df = df_1pipe.copy()

    # u*におけるクラス1・クラス2それぞれの"密度" u_i^-1(u*)
    u1_inv = logistic_density_speed(
        df['u_1pipe'], car_params['ub'], car_params['uf'], car_params['rhoc'],
        car_params['theta1'], car_params['theta2']
    )
    u2_inv = underwood_density_speed(df['u_1pipe'], truck_params['uf'], truck_params['rhoc'])

    df['p1_star'] = df['rho1'] / u1_inv
    df['p2_star'] = df['rho2'] / u2_inv

    n_valid = df[['p1_star', 'p2_star']].notna().all(axis=1).sum()
    print(f"最小道路占有を計算: {len(df):,}スナップショット中、{n_valid:,}件で計算完了")

    return df    

#############################################################################
#協力余剰
#############################################################################
def compute_cooperation_surplus(df_pstar):
    """
    論文Definition 1：協力余剰 s := 1 - p1* - p2* を計算する。

    df_pstar: compute_min_road_share の返り値（p1_star, p2_star列を含む）
    """
    df = df_pstar.copy()
    df['s'] = 1.0 - df['p1_star'] - df['p2_star']

    n_valid = df['s'].notna().sum()
    n_positive = (df['s'] > 0).sum()
    print(f"協力余剰sを計算: {len(df):,}スナップショット中、{n_valid:,}件で計算完了"
          f"（うちs>0: {n_positive:,}件）")

    return df

def plot_cooperation_surplus_heatmap(df_s, bin_width1=1.0, bin_width2=1.0,
                                       max_rho1=None, max_rho2=None, min_count=3):
    """
    協力余剰sのヒートマップを、km単位版とmile単位版(論文の単位)を横並びで表示する。
    ビン分割の方式はplot_1pipe_speed_heatmapと同じ(mile版は独立に再ビン分割)。
    カラーマップは論文Fig.12に合わせてviridis、0を起点とした連続カラーにしている。
    """
    df = df_s.dropna(subset=['s']).copy()

    if max_rho1 is None:
        max_rho1 = df['rho1'].max()
    if max_rho2 is None:
        max_rho2 = df['rho2'].max()

    # --- km版 ---
    bins1_km, bins2_km, heat_s_km = _bin_and_aggregate_1pipe(
        df, 'rho1', 'rho2', bin_width1, bin_width2, max_rho1, max_rho2, min_count,
        value_col='s'
    )

    # --- mile版：rho1, rho2をmileに変換してから、mile換算のビン幅で独立に再分割 ---
    df['rho1_mile'] = df['rho1'] * KM_TO_MILE
    df['rho2_mile'] = df['rho2'] * KM_TO_MILE
    bin_width1_mile = bin_width1 * KM_TO_MILE
    bin_width2_mile = bin_width2 * KM_TO_MILE
    max_rho1_mile = max_rho1 * KM_TO_MILE
    max_rho2_mile = max_rho2 * KM_TO_MILE

    bins1_mile, bins2_mile, heat_s_mile = _bin_and_aggregate_1pipe(
        df, 'rho1_mile', 'rho2_mile', bin_width1_mile, bin_width2_mile,
        max_rho1_mile, max_rho2_mile, min_count, value_col='s'
    )

    # km版・mile版で共通の色スケール(0起点、絶対値最大に合わせる)
    vmax = np.nanmax(np.concatenate([heat_s_km.values.ravel(), heat_s_mile.values.ravel()]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    mesh0 = axes[0].pcolormesh(bins1_km, bins2_km, heat_s_km.values, cmap='viridis', vmin=0, vmax=vmax, shading='flat')
    fig.colorbar(mesh0, ax=axes[0], label='Cooperation surplus s')
    axes[0].set_xlabel('Density class 1 - Car (veh/km)')
    axes[0].set_ylabel('Density class 2 - Truck (veh/km)')
    axes[0].set_title('Cooperation Surplus Heatmap (km)')

    mesh1 = axes[1].pcolormesh(bins1_mile, bins2_mile, heat_s_mile.values, cmap='viridis', vmin=0, vmax=vmax, shading='flat')
    fig.colorbar(mesh1, ax=axes[1], label='Cooperation surplus s')
    axes[1].set_xlabel('Density class 1 - Car (veh/mile)')
    axes[1].set_ylabel('Density class 2 - Truck (veh/mile)')
    axes[1].set_title('Cooperation Surplus Heatmap (mile, paper units)')

    plt.tight_layout()
    plt.show()

    return heat_s_km, heat_s_mile
#############################################################################
#テスト用関数
#############################################################################
# 論文Fig.11のカラーバーから読み取った、speed(mph) -> RGBの対応表
calibration = [
    (0.5, (17, 6, 136)), (1.0, (28, 5, 140)), (1.5, (34, 6, 143)), (2.0, (39, 4, 146)),
    (2.5, (48, 4, 149)), (3.0, (53, 5, 153)), (3.5, (57, 4, 154)), (4.0, (63, 4, 156)),
    (4.5, (68, 3, 159)), (5.0, (73, 2, 161)), (5.5, (78, 2, 162)), (6.0, (81, 3, 164)),
    (6.5, (89, 3, 166)), (7.0, (95, 1, 167)), (7.5, (98, 0, 167)), (8.0, (105, 0, 169)),
    (8.5, (106, 0, 168)), (9.0, (113, 0, 168)), (9.5, (117, 2, 168)), (10.0, (123, 2, 169)),
    (10.5, (129, 4, 168)), (11.0, (131, 5, 167)), (11.5, (137, 8, 166)), (12.0, (141, 12, 164)),
    (12.5, (147, 14, 165)), (13.0, (150, 18, 159)), (13.5, (157, 22, 161)), (14.0, (159, 25, 156)),
    (14.5, (161, 28, 154)), (15.0, (166, 33, 152)), (15.5, (170, 36, 146)), (16.0, (172, 39, 146)),
    (16.5, (180, 44, 148)), (17.0, (182, 47, 141)), (17.5, (187, 51, 136)), (18.0, (190, 55, 134)),
    (18.5, (192, 57, 133)), (19.0, (196, 62, 125)), (19.5, (199, 65, 126)), (20.0, (201, 67, 121)),
    (20.5, (204, 72, 119)), (21.0, (207, 77, 115)), (21.5, (211, 80, 114)), (22.0, (214, 84, 113)),
    (22.5, (216, 87, 107)), (23.0, (219, 91, 106)), (23.5, (220, 94, 103)), (24.0, (223, 97, 100)),
    (24.5, (226, 101, 96)), (25.0, (228, 106, 94)), (25.5, (231, 110, 91)), (26.0, (233, 113, 88)),
    (26.5, (235, 118, 85)), (27.0, (237, 121, 82)), (27.5, (241, 127, 78)), (28.0, (241, 130, 77)),
    (28.5, (243, 133, 73)), (29.0, (246, 138, 72)), (29.5, (245, 141, 68)), (30.0, (247, 146, 66)),
    (30.5, (250, 152, 63)), (31.0, (250, 155, 61)), (31.5, (249, 160, 59)), (32.0, (252, 164, 54)),
    (32.5, (252, 168, 53)), (33.0, (253, 173, 50)), (33.5, (251, 178, 47)), (34.0, (255, 183, 45)),
    (34.5, (254, 188, 45)), (35.0, (252, 193, 41)), (35.5, (252, 199, 37)), (36.0, (254, 202, 38)),
    (36.5, (254, 208, 37)), (37.0, (251, 214, 35)), (37.5, (250, 218, 37)), (38.0, (248, 224, 36)),
    (38.5, (246, 229, 37)), (39.0, (246, 235, 39)), (39.5, (243, 242, 38)), (40.0, (239, 249, 31)),
]

def rgb_to_speed(rgb, calibration=calibration, max_dist=25.0):
    """
    RGB値を対応表の中で最も近い色のspeed値(mph)に変換する。

    距離(ユークリッド距離)がmax_distを超える場合、plasmaのカラースケール上の
    色とは考えにくい(グレーの無データ領域や、境界のアンチエイリアシングなど)
    としてNaNを返す。
    """
    r, g, b = rgb
    best_value, best_dist = None, float('inf')
    for value, (cr, cg, cb) in calibration:
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_value = value

    if best_dist > max_dist:
        return float('nan')
    return best_value



