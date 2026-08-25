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


################################################################
################################################################
#平滑化フィルタ関数
################################################################
################################################################
def savgol_filter_numpy(y, window_length, polyorder, deriv=0, delta=1.0):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < window_length:
        return y.copy()

    half_window = (window_length - 1) // 2
    x = np.arange(-half_window, half_window + 1)
    A = np.vander(x, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(A)[deriv] * (math.factorial(deriv) / (delta ** deriv))

    smoothed_interior = np.convolve(y, coeffs[::-1], mode='valid')

    result = y.copy()
    result[half_window: n - half_window] = smoothed_interior
    return result

def smooth_spacing(df, window_length=11, polyorder=3, spacing_col='Space_Headway', max_gap_frames=1):
    """
    車間距離(Space_Headway)を、追従ペア(temp_pair_id)ごとに独立して
    Savitzky-Golayフィルタで平滑化する。
    同一車両でも、別々の追従区間をまたいで平滑化しないよう、
    Vehicle_ID/time_periodではなくtemp_pair_idでグループ化する。
    """
    df_paired = make_temp_pair_id(df, max_gap_frames=max_gap_frames)

    def _smooth(s):
        return pd.Series(
            savgol_filter_numpy(s.values, window_length=window_length, polyorder=polyorder),
            index=s.index
        )

    df_paired[f'{spacing_col}_smooth'] = (
        df_paired.dropna(subset=['temp_pair_id'])
        .groupby('temp_pair_id')[spacing_col]
        .transform(_smooth)
    )

    return df_paired.drop(columns='temp_pair_id')


################################################################
################################################################
#車間距離時間系列プロット関数
################################################################
################################################################
def plot_space_headway_time_series(df, vehicle_id, time_period=None, time_col='relative_sec',
                                     use_meters=True, space_headway_col='Space_Headway', ax=None):
    mask = df['Vehicle_ID'] == vehicle_id
    if time_period is not None:
        mask &= df['time_period'] == time_period

    sub = df[mask].copy().sort_values('Frame_ID')

    if sub.empty:
        print(f"エラー: Vehicle_ID {vehicle_id} (time_period={time_period}) のデータが見つかりません。")
        return

    if time_period is None and sub['time_period'].nunique() > 1:
        print(f"警告: Vehicle_ID {vehicle_id} は複数のtime_periodにまたがっています。")
        return

    FT_TO_M = 0.3048
    if use_meters:
        headway = sub[space_headway_col] * FT_TO_M
        unit_label = 'm'
    else:
        headway = sub[space_headway_col]
        unit_label = 'ft'

    if time_col == 'relative_sec':
        x_data = (sub['Frame_ID'] - sub['Frame_ID'].min()) * 0.1
        x_label = 'Time elapsed (seconds)'
    elif time_col == 'Frame_ID':
        x_data = sub['Frame_ID']
        x_label = 'Frame ID'
    else:
        x_data = sub[time_col]
        x_label = time_col

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(x_data, headway, color='tab:blue', linewidth=1.5, label=f'Vehicle {vehicle_id}')
    ax.set_title(f'{space_headway_col} (Vehicle {vehicle_id}, time_period={time_period})')
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'Space Headway ({unit_label})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()



################################################################
################################################################
#異常な車間距離を排除する
################################################################
################################################################

def filter_invalid_headway(df, min_spacing_m=4.5, ft_to_m=0.3048):
    """
    Space_Headway(ft)に基づいて異常な小車間距離データをフィルタリングする関数
    """
    min_spacing_ft = min_spacing_m / ft_to_m
    df_filtered = df.copy()

    before = len(df_filtered)
    df_filtered = df_filtered[df_filtered['Space_Headway'] > min_spacing_ft]
    after = len(df_filtered)

    print(f"spacing <= {min_spacing_m}m ({min_spacing_ft:.2f}ft) の行を除外: {before:,} -> {after:,} (除外 {before-after:,}行)")
    
    return df_filtered


################################################################
################################################################
#ミクロ密度の計算
################################################################
################################################################

def compute_micro_density(df, ft_to_m=0.3048):
    """
    車間距離(ft)および速度(ft/s)からミクロ密度を算出する関数
    """
    df_computed = df.copy()

    # 1000m / (Space_Headway * ft_to_m)
    df_computed['mdensity_veh_km'] = 1000 / (df_computed['Space_Headway'] * ft_to_m)

    return df_computed

################################################################
################################################################
#マクロ挙動の計算
################################################################
################################################################
#　スナップショットからの密度、平均速度を計算する。
# 密度は車種ごとに計算。
def compute_macroscopic_snapshots(df, lanes=(2, 3, 4), classes=(2, 3), snapshot_step_frames=5,
                                    L_m=None, n_lanes=None, frame_interval_ms=100):
    """
    Algorithm 1 Step 2: 一定間隔(デフォルト5フレーム=0.5秒)でスナップショットを取り、
    マクロ密度(veh/km/lane)・平均速度(km/h)・流率(veh/h/lane)を計算する。

    df: 車線・車種で絞り込む前の全車両trajectoryデータ(関数内でlanes/classesに絞る)
    lanes: 対象車線
    classes: 対象車種(2=car, 3=truck)
    snapshot_step_frames: 何フレームおきにスナップショットを取るか
    L_m: 区間長(m)1,640feet = 1,640 * 0.3048 = 499.872mメタデータではこう書いているが、実際のデータのy座標の範囲から自動計算する 見えない加工の可能性を避けるために（カメラの画角で切られている部分があるとか）
    n_lanes: 車線数。3
    """
    FT_TO_M = 0.3048
    step_ms = frame_interval_ms * snapshot_step_frames  # 500ms

    # time_periodごとに、スナップショットを撮る時刻の集合を作成 
    valid_times = set()
    for tp, group in df.groupby('time_period'):
        # 車種を絞り込む前の全データから最小値(t_min)・最大値(t_max)を取得
        t_min = int(group['Global_Time'].min())
        t_max = int(group['Global_Time'].max())
        
        # t_min始まりの0.5秒(500ms)ごとの時刻配列を作成し、集合に追加
        snapshot_times_tp = set(range(t_min, t_max + 1, step_ms))
        valid_times.update(snapshot_times_tp)

    # データを車線と車種で絞り込む
    target = df[df['Lane_ID'].isin(lanes) & df['v_Class'].isin(classes)].copy()

    #車線の長さ・車線数
    #メタデータは使わないい　論文でもメタデータとは異なる長さが使われていた。
    if L_m is None:
        L_m = (target['Local_Y'].max() - target['Local_Y'].min()) * FT_TO_M
    if n_lanes is None:
        n_lanes = len(lanes)
    L_km = L_m / 1000

    print(f"区間長: {L_m:.1f} m, 車線数: {n_lanes}")


    #絞り込み後のデータ(target)から、生成した0.5s刻みリスト(valid_times)に含まれる行だけを抽出←ちょっと冗長
    # snapshot_mask = target['Global_Time'].isin(valid_times)
    # snap_rows = target[snapshot_mask]

    # snapshot_mask という変数を作らず、直接1行でフィルタリング
    snap_rows = target[target['Global_Time'].isin(valid_times)]

    agg = snap_rows.groupby(['time_period', 'Global_Time', 'v_Class']).agg(
        n_vehicles=('Vehicle_ID', 'size'),#車種ごとの車両数
        speed_kmh=('v_Vel', lambda s: (s * FT_TO_M * 3.6).mean())#平均速度　feet/s -> m/s -> km/h
    ).reset_index()

    agg['density_veh_km'] = agg['n_vehicles'] / (L_km * n_lanes)#マクロ密度の計算
    agg['flow_veh_h'] = agg['density_veh_km'] * agg['speed_kmh']#流量の計算

    return agg

################################################################
################################################################
#ロジスティックモデルでのパラメータ推定 (speed target)
################################################################
################################################################
def logistic_speed_density(rho, ub, uf, rhoc, theta1, theta2):
    """
    論文Eq(17)のロジスティック速度-密度モデル。
    f(rho) = ub + (uf - ub) / (1 + exp((rho-rhoc)/theta1))^theta2
    """
    exp_term = np.exp(np.clip((rho - rhoc) / theta1, -100, 100))  # オーバーフロー防止
    return ub + (uf - ub) / (1 + exp_term) ** theta2


def fit_nominal_speed_logistic(df, mode_id, density_col='mdensity_veh_km', speed_col='v_Vel',
                                 lower_bounds=None, upper_bounds=None, init_params=None,
                                 max_eval=15000, test_size=0.3, random_state=42):
    """
    論文Algorithm 1 Step 1(a): 指定した追従タイプ(mode_id)のデータから、
    ロジスティックモデルのパラメータ(ub, uf, rhoc, theta1, theta2)を
    ISRES(改良型確率的ランキング進化戦略)で推定する。

    パラメータの並び順: [ub, uf, rhoc, theta1, theta2]
    - ub: 平均走行速度(km/h)     - uf: 自由流速度(km/h)
    - rhoc: 臨界密度(veh/km)     - theta1: 曲線の伸び(密度と同じ単位)
    - theta2: 曲線の歪み(無次元)

    lower_bounds/upper_bounds/init_paramsを渡せば探索範囲・初期値を制約できる。
    """
    # ① 対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        raise ValueError(f"mode_id={mode_id} に該当するデータがありません。")

    # ② 速度をft/s -> km/hに変換(密度は既にveh/km前提)
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    # ③ 70:30で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # ④ 目的関数：学習データに対するL1誤差の合計(論文のl(theta)の定義通り)　gradは勾配情報だが今回は使用しない
    def objective(x, grad):
        ub, uf, rhoc, theta1, theta2 = x
        if uf <= ub:  # 自由流速度が平均速度より遅い、という物理的にありえない組合せは弾く
            return 1e10
        pred = logistic_speed_density(X_train, ub, uf, rhoc, theta1, theta2)
        return np.sum(np.abs(Y_train - pred))

    # ⑤ 探索範囲・初期値のデフォルト(論文Table4の実測値を参考に設定。指定があれば上書き)
    if lower_bounds is None:
        lower_bounds = [0.0, 20.0, 5.0, 0.5, 0.05]     # [ub, uf, rhoc, theta1, theta2]
    if upper_bounds is None:
        upper_bounds = [60.0, 130.0, 150.0, 30.0, 3.0]
    if init_params is None:
        init_params = [15.0, 80.0, 30.0, 10.0, 1.0]

    # ⑥ ISRES(GN_ISRES)で最適化
    opt = nlopt.opt(nlopt.GN_ISRES, 5)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_params = opt.optimize(init_params)

    # ⑦ テストデータでMAEを評価
    ub, uf, rhoc, theta1, theta2 = best_params
    Y_pred_test = logistic_speed_density(X_test, ub, uf, rhoc, theta1, theta2)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)

    print(f"--- mode_id={mode_id} 推定結果 ---")
    print(f"ub={ub:.3f} km/h, uf={uf:.3f} km/h, rhoc={rhoc:.3f} veh/km, "
          f"theta1={theta1:.3f}, theta2={theta2:.3f}")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'ub': ub, 'uf': uf, 'rhoc': rhoc,
            'theta1': theta1, 'theta2': theta2, 'test_mae': test_mae}


################################################################
################################################################
#ロジスティックモデルでのパラメータ推定 (density target)
################################################################
################################################################
def logistic_density_speed(v, ub, uf, rhoc, theta1, theta2):
    """逆関数: 速度 v から 密度 rho を予測

    rho = rhoc + theta1 * ln( ((uf - ub) / (v - ub))^(1/theta2) - 1 )
    """
    # 数学的に定義できない領域（v <= ub または v >= uf）に対する数値的クリップ
    eps = 1e-4
    v_clipped = np.clip(v, ub + eps, uf - eps)

    # 累乗項の計算
    ratio = (uf - ub) / (v_clipped - ub)
    inner_term = (ratio ** (1.0 / theta2)) - 1.0

    # 対数の中身が 0 以下にならないようにクリップ
    inner_term = np.clip(inner_term, 1e-10, None)

    return rhoc + theta1 * np.log(inner_term)


def fit_nominal_density_logistic(
    df,
    mode_id,
    density_col='mdensity_veh_km',
    speed_col='v_Vel',
    lower_bounds=None,
    upper_bounds=None,
    init_params=None,
    max_eval=15000,
    test_size=0.3,
    random_state=42,
):
    """速度を入力として密度を予測し、密度の誤差(L1 Loss)を最小化する最適化関数"""
    # ① データの抽出
    data = (
        df[df['mode_id'] == mode_id]
        .dropna(subset=[density_col, speed_col])
        .copy()
    )
    if len(data) == 0:
        raise ValueError(f'mode_id={mode_id} に該当するデータがありません。')

    # ② 速度を ft/s -> km/h に変換
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data['speed_kmh'].values  # ★ 入力: 速度 (km/h)
    Y = data[density_col].values  # ★ ターゲット: 密度 (veh/km)

    # ③ 学習・評価データに分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # ④ 目的関数：密度のL1誤差の合計
    def objective(x, grad):
        ub, uf, rhoc, theta1, theta2 = x

        # 物理的な制約違反に対するペナルティ
        if uf <= ub:
            return 1e10

        # 速度 X_train から 密度 pred_rho を予測
        pred_rho = logistic_density_speed(X_train, ub, uf, rhoc, theta1, theta2)

        # 実測密度 Y_train との絶対誤差の合計
        return np.sum(np.abs(Y_train - pred_rho))

    # ⑤ 探索範囲・初期値の設定
    if lower_bounds is None:
        lower_bounds = [0.0, 20.0, 5.0, 0.5, 0.05]
    if upper_bounds is None:
        upper_bounds = [60.0, 130.0, 150.0, 100.0, 100.0]
    if init_params is None:
        init_params = [15.0, 120.0, 30.0, 10.0, 1.0]

    # Boundsの範囲外エラー(invalid_argument)防止用クランプ
    init_params = np.clip(init_params, lower_bounds, upper_bounds).tolist()

    # ⑥ ISRESによる最適化
    opt = nlopt.opt(nlopt.GN_ISRES, 5)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_params = opt.optimize(init_params)

    # ⑦ テストデータで密度のMAEを評価
    ub, uf, rhoc, theta1, theta2 = best_params
    Y_pred_test = logistic_density_speed(
        X_test, ub, uf, rhoc, theta1, theta2
    )
    test_mae_density = mean_absolute_error(Y_test, Y_pred_test)

    # （参考）予測した密度から算出した速度でのMAEも計算
    v_pred_test = logistic_speed_density(
        Y_test, ub, uf, rhoc, theta1, theta2
    )
    test_mae_speed = mean_absolute_error(X_test, v_pred_test)

    print(f'--- mode_id={mode_id} 密度ターゲットでの推定結果 ---')
    print(
        f'ub={ub:.3f} km/h, uf={uf:.3f} km/h, rhoc={rhoc:.3f} veh/km, '
        f'theta1={theta1:.3f}, theta2={theta2:.3f}'
    )
    print(f'テスト密度 MAE: {test_mae_density:.3f} veh/km')
    print(f'テスト速度 MAE: {test_mae_speed:.3f} km/h')

    return {
        'mode_id': mode_id,
        'ub': ub,
        'uf': uf,
        'rhoc': rhoc,
        'theta1': theta1,
        'theta2': theta2,
        'test_mae_density': test_mae_density,
        'test_mae_speed': test_mae_speed,
    }




################################################################
################################################################
#ミクロ速度密度散布図と推定モデルの関数プロット
################################################################
#################################################################
def plot_model_logistic(
    df,
    mode_id,
    params,
    params2,
    #params3,
    mode_name,
    label1='Estimated Model',
    label2='Estimated Model in paper',
    #label3='Estimated Model (ALL)',
):
    ub, uf, rhoc, theta1, theta2 = params
    ub2, uf2, rhoc2, theta12, theta22 = params2
    #ub3, uf3, rhoc3, theta13, theta23 = params3

    # 1. 指定した mode_id のデータだけに絞り込み（SettingWithCopyWarning防止のため copy）
    df_mode = df[df['mode_id'] == mode_id].copy()

    # 2. speed_kmh 列を df_mode 自体に確実に作成
    if 'speed_kmh' not in df_mode.columns:
        df_mode['speed_kmh'] = df_mode['v_Vel'] * 0.3048 * 3.6

    # 0〜250 veh/km の範囲で100点の密度グリッドを作成
    rho_range = np.linspace(0, 250, 100)

    # 予測速度の計算
    v_pred = logistic_speed_density(rho_range, ub, uf, rhoc, theta1, theta2)
    v_pred2 = logistic_speed_density(rho_range, ub2, uf2, rhoc2, theta12, theta22)
    #v_pred3 = logistic_speed_density(rho_range, ub3, uf3, rhoc3, theta13, theta23)

    # --- ある密度における速度の中央値 ---
    bin_width = 5.0
    max_density = 250
    bins = np.arange(0, max_density + bin_width, bin_width)
    df_mode['density_bin'] = pd.cut(
        df_mode['mdensity_veh_km'], bins=bins, include_lowest=True
    )
    bin_stats = df_mode.groupby('density_bin', observed=True).agg(
        median_speed=('speed_kmh', 'median'), count=('speed_kmh', 'count')
    )
    bin_centers = [interval.mid for interval in bin_stats.index]

    # --- ある速度における密度の中央値 ---
    bin_width_v = 5.0
    max_speed = 120
    bins_v = np.arange(0, max_speed + bin_width_v, bin_width_v)
    df_mode['speed_bin'] = pd.cut(
        df_mode['speed_kmh'], bins=bins_v, include_lowest=True
    )
    bin_stats_v = df_mode.groupby('speed_bin', observed=True).agg(
        median_density=('mdensity_veh_km', 'median')
    )
    bin_centers_v = [interval.mid for interval in bin_stats_v.index]

    # --- 描画処理 ---
    plt.figure(figsize=(9, 6))

    # 実測データ散布図
    plt.scatter(
        df_mode['mdensity_veh_km'],
        df_mode['speed_kmh'],
        s=3,
        alpha=0.3,
        color='steelblue',
        label='Actual Data',
    )

    # 各モデル曲線（引数で受け取った label1, label2, label3 を適用）
    plt.plot(
        rho_range,
        v_pred,
        color='red',
        linewidth=2,
        label=label1,
    )
    # plt.plot(
    #     rho_range,
    #     v_pred3,
    #     color='green',
    #     linewidth=2,
    #     linestyle='-.',
    #     label=label3,
    # )
    plt.plot(
        rho_range,
        v_pred2,
        color='orange',
        linewidth=2,
        linestyle='--',
        label=label2,
    )

    # 中央値プロット
    plt.plot(
        bin_centers,
        bin_stats['median_speed'],
        color='black',
        linewidth=2.5,
        marker='o',
        markersize=4,
        label=f'Median Speed (bin={bin_width} veh/km)',
    )
    plt.plot(
        bin_stats_v['median_density'],
        bin_centers_v,
        color='purple',
        linewidth=2,
        linestyle=':',
        marker='s',
        markersize=3,
        label=f'Median Density (bin={bin_width_v} km/h)',
    )

    # 軸・タイトルの設定
    plt.xlabel('Density (veh/km)')
    plt.ylabel('Speed (km/h)')
    plt.xlim(0, 250)
    plt.ylim(0, 120)
    plt.title(f'{mode_name} (n={len(df_mode):,} frames)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_model_logistic_3params(
    df,
    mode_id,
    params,
    params2,
    params3,
    mode_name,
    label1='Estimated Model',
    label2='Estimated Model in paper',
    label3='Estimated Model (ALL)',
):
    ub, uf, rhoc, theta1, theta2 = params
    ub2, uf2, rhoc2, theta12, theta22 = params2
    ub3, uf3, rhoc3, theta13, theta23 = params3

    # 1. 指定した mode_id のデータだけに絞り込み（SettingWithCopyWarning防止のため copy）
    df_mode = df[df['mode_id'] == mode_id].copy()

    # 2. speed_kmh 列を df_mode 自体に確実に作成
    if 'speed_kmh' not in df_mode.columns:
        df_mode['speed_kmh'] = df_mode['v_Vel'] * 0.3048 * 3.6

    # 0〜250 veh/km の範囲で100点の密度グリッドを作成
    rho_range = np.linspace(0, 250, 100)

    # 予測速度の計算
    v_pred = logistic_speed_density(rho_range, ub, uf, rhoc, theta1, theta2)
    v_pred2 = logistic_speed_density(rho_range, ub2, uf2, rhoc2, theta12, theta22)
    v_pred3 = logistic_speed_density(rho_range, ub3, uf3, rhoc3, theta13, theta23)

    # --- ある密度における速度の中央値 ---
    bin_width = 5.0
    max_density = 250
    bins = np.arange(0, max_density + bin_width, bin_width)
    df_mode['density_bin'] = pd.cut(
        df_mode['mdensity_veh_km'], bins=bins, include_lowest=True
    )
    bin_stats = df_mode.groupby('density_bin', observed=True).agg(
        median_speed=('speed_kmh', 'median'), count=('speed_kmh', 'count')
    )
    bin_centers = [interval.mid for interval in bin_stats.index]

    # --- ある速度における密度の中央値 ---
    bin_width_v = 5.0
    max_speed = 120
    bins_v = np.arange(0, max_speed + bin_width_v, bin_width_v)
    df_mode['speed_bin'] = pd.cut(
        df_mode['speed_kmh'], bins=bins_v, include_lowest=True
    )
    bin_stats_v = df_mode.groupby('speed_bin', observed=True).agg(
        median_density=('mdensity_veh_km', 'median')
    )
    bin_centers_v = [interval.mid for interval in bin_stats_v.index]

    # --- 描画処理 ---
    plt.figure(figsize=(9, 6))

    # 実測データ散布図
    plt.scatter(
        df_mode['mdensity_veh_km'],
        df_mode['speed_kmh'],
        s=3,
        alpha=0.3,
        color='steelblue',
        label='Actual Data',
    )

    # 各モデル曲線（引数で受け取った label1, label2, label3 を適用）
    plt.plot(
        rho_range,
        v_pred,
        color='red',
        linewidth=2,
        label=label1,
    )
    plt.plot(
        rho_range,
        v_pred3,
        color='green',
        linewidth=2,
        linestyle='-.',
        label=label3,
    )
    plt.plot(
        rho_range,
        v_pred2,
        color='orange',
        linewidth=2,
        linestyle='--',
        label=label2,
    )

    # 中央値プロット
    plt.plot(
        bin_centers,
        bin_stats['median_speed'],
        color='black',
        linewidth=2.5,
        marker='o',
        markersize=4,
        label=f'Median Speed (bin={bin_width} veh/km)',
    )
    plt.plot(
        bin_stats_v['median_density'],
        bin_centers_v,
        color='purple',
        linewidth=2,
        linestyle=':',
        marker='s',
        markersize=3,
        label=f'Median Density (bin={bin_width_v} km/h)',
    )

    # 軸・タイトルの設定
    plt.xlabel('Density (veh/km)')
    plt.ylabel('Speed (km/h)')
    plt.xlim(0, 250)
    plt.ylim(0, 120)
    plt.title(f'{mode_name} (n={len(df_mode):,} frames)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

################################################################
################################################################
#Underwoodモデルでのパラメータ推定
################################################################
#################################################################

def underwood_speed_density(rho, uf, rhoc):
    """
    論文Eq(18)のアンダーウッド速度-密度モデル。
    g(rho) = uf * exp(-rho / rhoc)
    """
    return uf * np.exp(np.clip(-rho / rhoc, -100, 100))  # オーバーフロー防止


def fit_nominal_speed_underwood(df, mode_id, density_col='mdensity_veh_km', speed_col='v_Vel',
                                  lower_bounds=None, upper_bounds=None, init_params=None,
                                  max_eval=15000, test_size=0.3, random_state=42):
    """
    論文Algorithm 1 Step 1(a): 指定した追従タイプ(mode_id)のデータから、
    アンダーウッドモデルのパラメータ(uf, rhoc)を
    ISRES(改良型確率的ランキング進化戦略)で推定する。トラッククラス用。

    パラメータの並び順: [uf, rhoc]
    - uf: 自由流速度(km/h)      - rhoc: 臨界密度(veh/km)

    lower_bounds/upper_bounds/init_paramsを渡せば探索範囲・初期値を制約できる。
    """
    # ① 対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        raise ValueError(f"mode_id={mode_id} に該当するデータがありません。")

    # ② 速度をft/s -> km/hに変換(密度は既にveh/km前提)
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    # ③ 70:30で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # ④ 目的関数：学習データに対するL1誤差の合計(論文のl(theta)の定義通り) gradは今回は使用しない
    def objective(x, grad):
        uf, rhoc = x
        pred = underwood_speed_density(X_train, uf, rhoc)
        return np.sum(np.abs(Y_train - pred))

    # ⑤ 探索範囲・初期値のデフォルト(論文Table4の実測値を参考に設定。指定があれば上書き)
    #    論文の値(km換算): T-T → uf≈68.5, rhoc≈25.9 / T-C → uf≈96.7, rhoc≈49.4
    if lower_bounds is None:
        lower_bounds = [20.0, 5.0]        # [uf, rhoc]
    if upper_bounds is None:
        upper_bounds = [150.0, 200.0]
    if init_params is None:
        init_params = [80.0, 40.0]

    # ⑥ ISRES(GN_ISRES)で最適化
    opt = nlopt.opt(nlopt.GN_ISRES, 2)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_params = opt.optimize(init_params)

    # ⑦ テストデータでMAEを評価
    uf, rhoc = best_params
    Y_pred_test = underwood_speed_density(X_test, uf, rhoc)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)

    print(f"--- mode_id={mode_id} 推定結果(アンダーウッド) ---")
    print(f"uf={uf:.3f} km/h, rhoc={rhoc:.3f} veh/km")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'uf': uf, 'rhoc': rhoc, 'test_mae': test_mae}

################################################################
################################################################
#ミクロ速度密度散布図と推定モデルの関数プロット Underwoodモデル用
################################################################
#################################################################
def plot_model_underwood(
    df,
    mode_id,
    params,
    params2,
    params3,
    mode_name,
    label1='Estimated Model',
    label2='Estimated Model in paper',
    label3='Estimated Model (ALL)',
):
    uf, rhoc = params
    uf2, rhoc2= params2
    uf3, rhoc3= params3

    # 1. 指定した mode_id のデータだけに絞り込み（SettingWithCopyWarning防止のため copy）
    df_mode = df[df['mode_id'] == mode_id].copy()

    # 2. speed_kmh 列を df_mode 自体に確実に作成
    if 'speed_kmh' not in df_mode.columns:
        df_mode['speed_kmh'] = df_mode['v_Vel'] * 0.3048 * 3.6

    # 0〜250 veh/km の範囲で100点の密度グリッドを作成
    rho_range = np.linspace(0, 250, 100)

    # 予測速度の計算
    v_pred = underwood_speed_density(rho_range, uf, rhoc)
    v_pred2 = underwood_speed_density(rho_range, uf2, rhoc2)
    v_pred3 = underwood_speed_density(rho_range, uf3, rhoc3)

    # --- ある密度における速度の中央値 ---
    bin_width = 5.0
    max_density = 250
    bins = np.arange(0, max_density + bin_width, bin_width)
    df_mode['density_bin'] = pd.cut(
        df_mode['mdensity_veh_km'], bins=bins, include_lowest=True
    )
    bin_stats = df_mode.groupby('density_bin', observed=True).agg(
        median_speed=('speed_kmh', 'median'), count=('speed_kmh', 'count')
    )
    bin_centers = [interval.mid for interval in bin_stats.index]

    # --- ある速度における密度の中央値 ---
    bin_width_v = 5.0
    max_speed = 120
    bins_v = np.arange(0, max_speed + bin_width_v, bin_width_v)
    df_mode['speed_bin'] = pd.cut(
        df_mode['speed_kmh'], bins=bins_v, include_lowest=True
    )
    bin_stats_v = df_mode.groupby('speed_bin', observed=True).agg(
        median_density=('mdensity_veh_km', 'median')
    )
    bin_centers_v = [interval.mid for interval in bin_stats_v.index]

    # --- 描画処理 ---
    plt.figure(figsize=(9, 6))

    # 実測データ散布図
    plt.scatter(
        df_mode['mdensity_veh_km'],
        df_mode['speed_kmh'],
        s=3,
        alpha=0.3,
        color='steelblue',
        label='Actual Data',
    )

    # 各モデル曲線（引数で受け取った label1, label2, label3 を適用）
    plt.plot(
        rho_range,
        v_pred,
        color='red',
        linewidth=2,
        label=label1,
    )
    plt.plot(
        rho_range,
        v_pred3,
        color='green',
        linewidth=2,
        linestyle='-.',
        label=label3,
    )
    plt.plot(
        rho_range,
        v_pred2,
        color='orange',
        linewidth=2,
        linestyle='--',
        label=label2,
    )

    # 中央値プロット
    plt.plot(
        bin_centers,
        bin_stats['median_speed'],
        color='black',
        linewidth=2.5,
        marker='o',
        markersize=4,
        label=f'Median Speed (bin={bin_width} veh/km)',
    )
    plt.plot(
        bin_stats_v['median_density'],
        bin_centers_v,
        color='purple',
        linewidth=2,
        linestyle=':',
        marker='s',
        markersize=3,
        label=f'Median Density (bin={bin_width_v} km/h)',
    )

    # 軸・タイトルの設定
    plt.xlabel('Density (veh/km)')
    plt.ylabel('Speed (km/h)')
    plt.xlim(0, 250)
    plt.ylim(0, 120)
    plt.title(f'{mode_name} (n={len(df_mode):,} frames)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

################################################################
################################################################
#スケーリングパラメータ推定
################################################################
#################################################################




