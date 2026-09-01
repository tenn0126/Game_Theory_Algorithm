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
#平滑化フィルタ関数　（使用しない）
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
#車間距離時間系列プロット関数
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
#異常な車間距離を排除する
################################################################

def filter_invalid_headway(df, min_spacing_m=4.5, ft_to_m=0.3048):
    """
    Space_Headway(ft)に基づいて異常な小車間距離データをフィルタリングする関数
    """
    min_spacing_ft = min_spacing_m / ft_to_m
    df_filtered = df.copy()

    before = len(df_filtered)

    # フィルタリング条件: Space_Headway > min_spacing_ft の行だけを残す
    df_filtered = df_filtered[df_filtered['Space_Headway'] > min_spacing_ft]
    after = len(df_filtered)

    print("\n--閾値以下の車頭距離データを除外--")
    print(f"spacing <= {min_spacing_m}m ({min_spacing_ft:.2f}ft) の行を除外: {before:,} -> {after:,} (除外 {before-after:,}行)")

    
    return df_filtered


################################################################
#ミクロ密度の計算
################################################################
def compute_micro_density(df, ft_to_m=0.3048):
    """
    車間距離(ft)からミクロ密度を算出する関数
    """
    df_computed = df.copy()

    # 1000m / (Space_Headway * ft_to_m)
    #ミクロ密度は車頭距離の逆数
    df_computed['mdensity_veh_km'] = 1000 / (df_computed['Space_Headway'] * ft_to_m)
    print("\n--ミクロ密度を計算--")
    return df_computed


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
################################################################
#ロジスティックモデルでのパラメータ推定 (speed target)
################################################################
def logistic_speed_density(rho, ub, uf, rhoc, theta1, theta2):
    """
    論文Eq(17)のロジスティック速度-密度モデル。
    f(rho) = ub + (uf - ub) / (1 + exp((rho-rhoc)/theta1))^theta2
    """
    exp_term = np.exp(np.clip((rho - rhoc) / theta1, -100, 100))  # expの中が大きくなるとすぐにオーバーフローするため防止策を入れておく
    return ub + (uf - ub) / ((1 + exp_term) ** theta2)


def fit_nominal_speed_logistic(df, mode_id, density_col='mdensity_veh_km', speed_col='v_Vel',
                                 lower_bounds=None, upper_bounds=None, init_params=None,
                                 max_eval=15000, test_size=0.3, random_state=42):
    """
    指定した追従タイプ(mode_id)のデータから、
    ロジスティックモデルのパラメータ(ub, uf, rhoc, theta1, theta2)を
    ISRES(改良型確率的ランキング進化戦略)で推定する。

    パラメータの並び順: [ub, uf, rhoc, theta1, theta2]
    - ub: 平均走行速度(km/h)     - uf: 自由流速度(km/h)
    - rhoc: 臨界密度(veh/km)     - theta1: 曲線の伸び(密度と同じ単位)
    - theta2: 曲線の歪み(無次元)

    lower_bounds/upper_bounds/init_paramsを渡せば探索範囲・初期値に制約を与える。
    """
    
    # 対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        print(f"mode_id={mode_id} に該当するデータがありません。")
        return None
    
    #速度をft/s -> km/hに変換(密度は既にveh/km前提 compute_micro_densityで計算済み)　
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    #7:3で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    #目的関数：学習データに対するL1誤差の合計(論文のl(theta)の定義通り)　gradは勾配情報だがISRESは使用しない
    #xはアルゴリズムが試行する5つのパラメータのリスト
    def objective(x, grad):
        ub, uf, rhoc, theta1, theta2 = x
        if uf <= ub:  # 自由流速度が平均速度より遅い、という物理的にありえない組合せは弾く
            return 1e10 #巨大な誤差値を返す
        pred = logistic_speed_density(X_train, ub, uf, rhoc, theta1, theta2)#予測速度を計算
        return np.sum(np.abs(Y_train - pred))#誤差の絶対値の足し合わせを求める

    #探索範囲・初期値のデフォルト(指定があれば上書き)
    if lower_bounds is None:
        lower_bounds = [0.0, 20.0, 5.0, 0.5, 0.05]     # [ub, uf, rhoc, theta1, theta2]
    if upper_bounds is None:
        upper_bounds = [60.0, 130.0, 150.0, 100.0, 100.0]
    if init_params is None:
        init_params = [15.0, 80.0, 30.0, 10.0, 1.0]

    # Boundsの範囲外エラー(invalid_argument)防止用クランプ
    init_params = np.clip(init_params, lower_bounds, upper_bounds).tolist()

    #ISRES(GN_ISRES)で最適化
    #GN_ISRES,GN_DIRECT,LN_COBYLA,LD_MMA　等別手法
    opt = nlopt.opt(nlopt.GN_ISRES, 5)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(objective)#objectiveを最小化する
    opt.set_maxeval(max_eval)#打ち切りの上限
    best_params = opt.optimize(init_params)#初期値リストの値からスタートして探索を実行

    #テストデータでMAEを評価　絶対平均誤差
    ub, uf, rhoc, theta1, theta2 = best_params
    Y_pred_test = logistic_speed_density(X_test, ub, uf, rhoc, theta1, theta2)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)

    print(f"\n--mode_id={mode_id}の名目速度関数paramを推定--")
    print(f"--- mode_id={mode_id} 推定結果 ---")
    print(f"ub={ub:.3f} km/h, uf={uf:.3f} km/h, rhoc={rhoc:.3f} veh/km, "
          f"theta1={theta1:.3f}, theta2={theta2:.3f}")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'ub': ub, 'uf': uf, 'rhoc': rhoc,
            'theta1': theta1, 'theta2': theta2, 'test_mae': test_mae}

################################################################
#ロジスティックモデルでのパラメータ推定 (density target)
################################################################
#ロジスティックモデルを逆関数にして、速度から密度を予測する関数を定義
def logistic_density_speed(v, ub, uf, rhoc, theta1, theta2):
    """
    逆関数: 速度 v から 密度 rho を予測
    rho = rhoc + theta1 * ln( ((uf - ub) / (v - ub))^(1/theta2) - 1 )
    """
    # 数学的に定義できない領域（v <= ub または v >= uf）に対する数値的クリップ
    #0で割り算することを防ぐ
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
    # データの抽出
    data = (
        df[df['mode_id'] == mode_id]
        .dropna(subset=[density_col, speed_col])
        .copy()
    )

    if len(data) == 0:
        print(f"mode_id={mode_id} に該当するデータがありません。")
        return None

    # 速度を ft/s -> km/h に変換
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data['speed_kmh'].values  # 入力: 速度 (km/h)
    Y = data[density_col].values  # ターゲット: 密度 (veh/km)

    #学習・評価データに分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # 目的関数：密度のL1誤差の合計
    def objective(x, grad):
        ub, uf, rhoc, theta1, theta2 = x

        # 物理的な制約違反に対するペナルティ
        if uf <= ub:
            return 1e10

        # 速度 X_train から 密度 pred_rho を予測
        pred_rho = logistic_density_speed(X_train, ub, uf, rhoc, theta1, theta2)

        # 実測密度 Y_train との絶対誤差の合計
        return np.sum(np.abs(Y_train - pred_rho))

    # 探索範囲・初期値の設定
    if lower_bounds is None:
        lower_bounds = [0.0, 20.0, 5.0, 0.5, 0.05]
    if upper_bounds is None:
        upper_bounds = [60.0, 130.0, 150.0, 100.0, 100.0]
    if init_params is None:
        init_params = [15.0, 120.0, 30.0, 10.0, 1.0]

    # Boundsの範囲外エラー(invalid_argument)防止用クランプ
    init_params = np.clip(init_params, lower_bounds, upper_bounds).tolist()

    # ISRESによる最適化
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
#ミクロ速度密度散布図と推定モデルの関数プロット
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
    def _extract_params(p):
        if isinstance(p, dict):
         # 辞書の場合はキー名で安全に取り出し
            return (p['ub'], p['uf'], p['rhoc'], p['theta1'], p['theta2'])
        return p  # 従来のリストやタプルの場合はそのまま


    ub, uf, rhoc, theta1, theta2 = _extract_params(params)
    ub2, uf2, rhoc2, theta12, theta22 = _extract_params(params2)
    # ub3, uf3, rhoc3, theta13, theta23 = _extract_params(params3)

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
    def _extract_params(p):
            if isinstance(p, dict):
             # 辞書の場合はキー名で安全に取り出し
                return (p['ub'], p['uf'], p['rhoc'], p['theta1'], p['theta2'])
            return p  # 従来のリストやタプルの場合はそのまま
    
    
    ub, uf, rhoc, theta1, theta2 = _extract_params(params)
    ub2, uf2, rhoc2, theta12, theta22 = _extract_params(params2)
    ub3, uf3, rhoc3, theta13, theta23 = _extract_params(params3)

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

def plot_model_logistic_scaling(
    df,
    mode_id,
    base_params,
    direct_params,
    scaling_param,
    mode_name,
    label_base="Base Model (e.g. Mode 1)",
    label_direct="Direct Fit Model",
    label_scaled="Scaled Base Model",
    max_density=250,
    max_speed=120,
):
    """ロジスティックモデルの描画比較関数: 1. 実測データ散布図 & 中央値 2.

    ベースモデル (スケーリングなし名目速度関数) 3. 単独推定モデル (5パラメータ直接推定) 4.
    スケーリングモデル (ベースモデルの密度を a でスケーリング: f(rho / a))

    Parameters:
    -----------
    df : pd.DataFrame
        軌跡データフレーム
    mode_id : int
        プロット対象の追従モードID (例: 2)
    base_params : dict or list/tuple
        基準となる名目速度関数のパラメータ (fit_nominal_speed_logistic の戻り値など)
    direct_params : dict or list/tuple
        当該モードで直接5パラメータ推定した結果
    scaling_param : dict, float, or int
        fit_scaling_parameter_logistic の戻り値 (辞書) または スケーリング係数 a
    mode_name : str
        グラフタイトルに表示するモード名
    """

    # --- パラメータ展開ヘルパー ---
    def _extract_5params(p):
        if isinstance(p, dict):
            return (p["ub"], p["uf"], p["rhoc"], p["theta1"], p["theta2"])
        return p

    def _extract_scale(s):
        if isinstance(s, dict):
            return s.get("scaling_param", s.get("a", s.get("b", 1.0)))
        return float(s)

    # 5パラメータの展開
    ub_base, uf_base, rhoc_base, th1_base, th2_base = _extract_5params(
        base_params
    )
    ub_dir, uf_dir, rhoc_dir, th1_dir, th2_dir = _extract_5params(direct_params)
    scale_a = _extract_scale(scaling_param)

    # 1. 指定した mode_id のデータだけに絞り込み
    df_mode = df[df["mode_id"] == mode_id].copy()
    if len(df_mode) == 0:
        raise ValueError(f"mode_id={mode_id} のデータが存在しません。")

    # 2. speed_kmh 列の生成
    if "speed_kmh" not in df_mode.columns:
        df_mode["speed_kmh"] = df_mode["v_Vel"] * 0.3048 * 3.6

    # 密度グリッドの作成
    rho_range = np.linspace(0, max_density, 200)

    # 予測速度の計算
    # ① ベースモデルそのまま
    v_base = logistic_speed_density(
        rho_range, ub_base, uf_base, rhoc_base, th1_base, th2_base
    )
    # ② 当該モードの単独推定モデル
    v_direct = logistic_speed_density(
        rho_range, ub_dir, uf_dir, rhoc_dir, th1_dir, th2_dir
    )
    # ③ ベースモデルをスケーリングしたモデル: f(rho / a)
    v_scaled = logistic_speed_density(
        rho_range / scale_a, ub_base, uf_base, rhoc_base, th1_base, th2_base
    )

    # --- ある密度における速度の中央値 ---
    bin_width = 5.0
    bins = np.arange(0, max_density + bin_width, bin_width)
    df_mode["density_bin"] = pd.cut(
        df_mode["mdensity_veh_km"], bins=bins, include_lowest=True
    )
    bin_stats = df_mode.groupby("density_bin", observed=True).agg(
        median_speed=("speed_kmh", "median"), count=("speed_kmh", "count")
    )
    valid_speed = bin_stats[bin_stats["count"] >= 5]  # データ数5件以上
    bin_centers = [interval.mid for interval in valid_speed.index]

    # --- ある速度における密度の中央値 ---
    bin_width_v = 5.0
    bins_v = np.arange(0, max_speed + bin_width_v, bin_width_v)
    df_mode["speed_bin"] = pd.cut(
        df_mode["speed_kmh"], bins=bins_v, include_lowest=True
    )
    bin_stats_v = df_mode.groupby("speed_bin", observed=True).agg(
        median_density=("mdensity_veh_km", "median"),
        count=("mdensity_veh_km", "count"),
    )
    valid_density = bin_stats_v[bin_stats_v["count"] >= 5]
    bin_centers_v = [interval.mid for interval in valid_density.index]

    # --- 描画処理 ---
    plt.figure(figsize=(10, 6.5))

    # 実測データ散布図
    plt.scatter(
        df_mode["mdensity_veh_km"],
        df_mode["speed_kmh"],
        s=3,
        alpha=0.25,
        color="steelblue",
        label="Actual Data",
    )

    # ① ベースモデル (点線・グレー)
    plt.plot(
        rho_range,
        v_base,
        color="gray",
        linewidth=1.8,
        linestyle=":",
        label=label_base,
    )

    # ② 単独推定モデル (オレンジ破線)
    plt.plot(
        rho_range,
        v_direct,
        color="darkorange",
        linewidth=2.0,
        linestyle="--",
        label=label_direct,
    )

    # ③ スケーリングモデル (赤実線)
    plt.plot(
        rho_range,
        v_scaled,
        color="crimson",
        linewidth=2.5,
        label=f"{label_scaled} (scale={scale_a:.3f})",
    )

    # 中央値プロット
    plt.plot(
        bin_centers,
        valid_speed["median_speed"],
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=f"Median Speed (bin={bin_width} veh/km)",
    )
    plt.plot(
        valid_density["median_density"],
        bin_centers_v,
        color="purple",
        linewidth=1.8,
        linestyle="-.",
        marker="s",
        markersize=3,
        label=f"Median Density (bin={bin_width_v} km/h)",
    )

    # 軸・タイトルの設定
    plt.xlabel("Density (veh/km)")
    plt.ylabel("Speed (km/h)")
    plt.xlim(0, max_density)
    plt.ylim(0, max_speed)
    plt.title(f"{mode_name} (n={len(df_mode):,} frames)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
################################################################
#Underwoodモデルでのパラメータ推定
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
    指定した追従タイプ(mode_id)のデータから、
    アンダーウッドモデルのパラメータ(uf, rhoc)を
    ISRES(改良型確率的ランキング進化戦略)で推定する。トラッククラス用。

    パラメータの並び順: [uf, rhoc]
    - uf: 自由流速度(km/h)      - rhoc: 臨界密度(veh/km)

    lower_bounds/upper_bounds/init_paramsを渡せば探索範囲・初期値を制約できる。
    """

    # 対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        print(f"mode_id={mode_id} に該当するデータがありません。")
        return None

    #速度をft/s -> km/hに変換(密度は既にveh/km前提)
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    #70:30で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    #目的関数：学習データに対するL1誤差の合計(論文のl(theta)の定義通り) gradはISRESは使用しない
    def objective(x, grad):
        uf, rhoc = x
        pred = underwood_speed_density(X_train, uf, rhoc)
        return np.sum(np.abs(Y_train - pred))#実測値と推測値の差の絶対値の総和

    #探索範囲・初期値のデフォルト(指定があれば上書き)
    #論文の値(km換算): T-T → uf≈68.5, rhoc≈25.9 / T-C → uf≈96.7, rhoc≈49.4
    if lower_bounds is None:
        lower_bounds = [20.0, 5.0]        # [uf, rhoc]
    if upper_bounds is None:
        upper_bounds = [150.0, 100.0]
    if init_params is None:
        init_params = [80.0, 40.0]

    #ISRES(GN_ISRES)で最適化
    opt = nlopt.opt(nlopt.GN_ISRES, 2)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_params = opt.optimize(init_params)

    #テストデータでMAEを評価 絶対平均誤差
    uf, rhoc = best_params
    Y_pred_test = underwood_speed_density(X_test, uf, rhoc)
    test_mae = mean_absolute_error(Y_test, Y_pred_test)
    
    print(f"\n--mode_id={mode_id}の名目速度関数paramを推定--")
    print(f"--- mode_id={mode_id} 推定結果(アンダーウッド) ---")
    print(f"uf={uf:.3f} km/h, rhoc={rhoc:.3f} veh/km")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'uf': uf, 'rhoc': rhoc, 'test_mae': test_mae}

################################################################
#ミクロ速度密度散布図と推定モデルの関数プロット Underwoodモデル用
#################################################################
def plot_model_underwood(
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
    def _extract_params(p):
                if isinstance(p, dict):
                 # 辞書の場合はキー名で安全に取り出し
                    return (p['uf'], p['rhoc'])
                return p  # 従来のリストやタプルの場合はそのまま
    uf, rhoc = _extract_params(params)
    uf2, rhoc2= _extract_params(params2)
    #uf3, rhoc3= _extract_params(params3)

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
    #v_pred3 = underwood_speed_density(rho_range, uf3, rhoc3)

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


def plot_model_underwood_3params(
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
    def _extract_params(p):
                if isinstance(p, dict):
                 # 辞書の場合はキー名で安全に取り出し
                    return (p['uf'], p['rhoc'])
                return p  # 従来のリストやタプルの場合はそのまま
    uf, rhoc = _extract_params(params)
    uf2, rhoc2= _extract_params(params2)
    uf3, rhoc3= _extract_params(params3)

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
        v_pred2,
        color='orange',
        linewidth=2,
        linestyle='--',
        label=label2,
    )

    plt.plot(
        rho_range,
        v_pred3,
        color='green',
        linewidth=2,
        linestyle='-.',
        label=label3,
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


def plot_model_underwood_scaling(
    df,
    mode_id,
    base_params,
    direct_params,
    scaling_param,
    mode_name,
    label_base="Base Model (e.g. Mode 4: Truck-Truck)",
    label_direct="Direct Fit Model",
    label_scaled="Scaled Base Model",
    max_density=250,
    max_speed=120,
):
    """Underwoodモデルの描画比較関数: 1. 実測データ散布図 & 中央値 2.

    ベースモデル (スケーリングなし Underwood 名目速度関数) 3. 単独推定モデル (uf, rhoc
    を直接推定) 4. スケーリングモデル (ベースモデルの密度を b でスケーリング: f(rho / b))

    Parameters:
    -----------
    df : pd.DataFrame
        軌跡データフレーム
    mode_id : int
        プロット対象の追従モードID (例: 3 = Truck-Car)
    base_params : dict, list, or tuple
        基準となる名目速度関数のパラメータ (uf, rhoc)
    direct_params : dict, list, or tuple
        当該モードで直接2パラメータ推定した結果 (uf, rhoc)
    scaling_param : dict, float, or int
        fit_scaling_parameter_underwood の戻り値 (辞書) または スケーリング係数 b
    mode_name : str
        グラフタイトルに表示するモード名
    """

    # --- パラメータ展開ヘルパー ---
    def _extract_2params(p):
        if isinstance(p, dict):
            return (p["uf"], p["rhoc"])
        return p

    def _extract_scale(s):
        if isinstance(s, dict):
            return s.get("scaling_param", s.get("b", s.get("b1", 1.0)))
        return float(s)

    # パラメータの展開 (Underwoodは uf, rhoc の2変数)
    uf_base, rhoc_base = _extract_2params(base_params)
    uf_dir, rhoc_dir = _extract_2params(direct_params)
    scale_b = _extract_scale(scaling_param)

    # 1. 指定した mode_id のデータだけに絞り込み
    df_mode = df[df["mode_id"] == mode_id].copy()
    if len(df_mode) == 0:
        raise ValueError(f"mode_id={mode_id} のデータが存在しません。")

    # 2. speed_kmh 列の生成
    if "speed_kmh" not in df_mode.columns:
        df_mode["speed_kmh"] = df_mode["v_Vel"] * 0.3048 * 3.6

    # 密度グリッドの作成
    rho_range = np.linspace(0, max_density, 200)

    # 予測速度の計算
    # ① ベースモデルそのまま
    v_base = underwood_speed_density(rho_range, uf_base, rhoc_base)
    # ② 当該モードの単独推定モデル
    v_direct = underwood_speed_density(rho_range, uf_dir, rhoc_dir)
    # ③ ベースモデルをスケーリングしたモデル: f(rho / b)
    v_scaled = underwood_speed_density(rho_range / scale_b, uf_base, rhoc_base)

    # --- ある密度における速度の中央値 ---
    bin_width = 5.0
    bins = np.arange(0, max_density + bin_width, bin_width)
    df_mode["density_bin"] = pd.cut(
        df_mode["mdensity_veh_km"], bins=bins, include_lowest=True
    )
    bin_stats = df_mode.groupby("density_bin", observed=True).agg(
        median_speed=("speed_kmh", "median"), count=("speed_kmh", "count")
    )
    valid_speed = bin_stats[bin_stats["count"] >= 5]
    bin_centers = [interval.mid for interval in valid_speed.index]

    # --- ある速度における密度の中央値 ---
    bin_width_v = 5.0
    bins_v = np.arange(0, max_speed + bin_width_v, bin_width_v)
    df_mode["speed_bin"] = pd.cut(
        df_mode["speed_kmh"], bins=bins_v, include_lowest=True
    )
    bin_stats_v = df_mode.groupby("speed_bin", observed=True).agg(
        median_density=("mdensity_veh_km", "median"),
        count=("mdensity_veh_km", "count"),
    )
    valid_density = bin_stats_v[bin_stats_v["count"] >= 5]
    bin_centers_v = [interval.mid for interval in valid_density.index]

    # --- 描画処理 ---
    plt.figure(figsize=(10, 6.5))

    # 実測データ散布図
    plt.scatter(
        df_mode["mdensity_veh_km"],
        df_mode["speed_kmh"],
        s=3,
        alpha=0.25,
        color="steelblue",
        label="Actual Data",
    )

    # ① ベースモデル (点線・グレー)
    plt.plot(
        rho_range,
        v_base,
        color="gray",
        linewidth=1.8,
        linestyle=":",
        label=label_base,
    )

    # ② 単独推定モデル (オレンジ破線)
    plt.plot(
        rho_range,
        v_direct,
        color="darkorange",
        linewidth=2.0,
        linestyle="--",
        label=label_direct,
    )

    # ③ スケーリングモデル (赤実線)
    plt.plot(
        rho_range,
        v_scaled,
        color="crimson",
        linewidth=2.5,
        label=f"{label_scaled} (scale={scale_b:.3f})",
    )

    # 中央値プロット
    plt.plot(
        bin_centers,
        valid_speed["median_speed"],
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=f"Median Speed (bin={bin_width} veh/km)",
    )
    plt.plot(
        valid_density["median_density"],
        bin_centers_v,
        color="purple",
        linewidth=1.8,
        linestyle="-.",
        marker="s",
        markersize=3,
        label=f"Median Density (bin={bin_width_v} km/h)",
    )

    # 軸・タイトルの設定
    plt.xlabel("Density (veh/km)")
    plt.ylabel("Speed (km/h)")
    plt.xlim(0, max_density)
    plt.ylim(0, max_speed)
    plt.title(f"{mode_name} (n={len(df_mode):,} frames)")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
################################################################
#スケーリングパラメータ推定
#################################################################
def fit_scaling_parameter_logistic(df, mode_id, nominal_params,
                                     density_col='mdensity_veh_km', speed_col='v_Vel',
                                     lower_bound=0.1, upper_bound=5.0, init_param=1.0,
                                     max_eval=15000, test_size=0.3, random_state=42):
    """
    推定済みの名目速度関数f(=f11)を基準に、
    スケーリングパラメータ a2 を ISRES で推定する。
    論文Eq(20): f12(rho) = f(rho / a2)

    nominal_params: Step 1(a)で推定したロジスティックのパラメータ
                    (fit_nominal_speed_logistic の戻り値をそのまま渡す)
    lower_bound/upper_bound/init_param: スケーリングパラメータの探索範囲・初期値
    """
    #対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        raise ValueError(f"mode_id={mode_id} に該当するデータがありません。")

    #速度をft/s -> km/hに変換(密度は既にveh/km前提)
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    #7:3で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    #名目速度関数のパラメータを取り出す(これらは固定。推定するのはスケーリングのみ)
    ub = nominal_params['ub']
    uf = nominal_params['uf']
    rhoc = nominal_params['rhoc']
    theta1 = nominal_params['theta1']
    theta2 = nominal_params['theta2']

    #目的関数：密度を a で割ってから名目速度関数に入れ、L1誤差の合計を取る(Eq(20))
    def objective(x, grad):
        a = x[0]
        pred = logistic_speed_density(X_train / a, ub, uf, rhoc, theta1, theta2)#rhoをaで割ったものを入力
        return np.sum(np.abs(Y_train - pred))

    #ISRES(GN_ISRES)で最適化(推定するのはスケーリングパラメータ1つだけ)
    #    論文Table5の実測値: a2 = 0.4528
    opt = nlopt.opt(nlopt.GN_ISRES, 1)
    opt.set_lower_bounds([lower_bound])
    opt.set_upper_bounds([upper_bound])
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_param = opt.optimize([init_param])

    #テストデータでMAEを評価
    a = best_param[0]
    Y_pred_test = logistic_speed_density(X_test / a, ub, uf, rhoc, theta1, theta2)#rhoをaで割ったものを入力
    test_mae = mean_absolute_error(Y_test, Y_pred_test)

    print(f"--- mode_id={mode_id} スケーリングパラメータ推定結果 ---")
    print(f"a = {a:.4f}")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'scaling_param': a, 'test_mae': test_mae}


def fit_scaling_parameter_underwood(df, mode_id, nominal_params,
                                     density_col='mdensity_veh_km', speed_col='v_Vel',
                                     lower_bound=0.1, upper_bound=5.0, init_param=1.0,
                                     max_eval=15000, test_size=0.3, random_state=42):
    """
    推定済みの名目速度関数f(=f22)を基準に、
    スケーリングパラメータ b1 を ISRES で推定する。
    論文Eq(20): f21(rho) = f(rho / b1)

    nominal_params: Step 1(a)で推定したUnderwoodのパラメータ
                    (fit_nominal_speed_underwood の戻り値をそのまま渡す)
    lower_bound/upper_bound/init_param: スケーリングパラメータの探索範囲・初期値
    """
    #対象の追従タイプ(mode_id)に絞り、欠損を除く
    data = df[df['mode_id'] == mode_id].dropna(subset=[density_col, speed_col]).copy()
    if len(data) == 0:
        raise ValueError(f"mode_id={mode_id} に該当するデータがありません。")

    #速度をft/s -> km/hに変換(密度は既にveh/km前提)
    data['speed_kmh'] = data[speed_col] * 0.3048 * 3.6

    X = data[density_col].values   # 密度(veh/km)
    Y = data['speed_kmh'].values   # 速度(km/h)

    #7:3で学習・評価データに分割(論文と同じ比率)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    #名目速度関数のパラメータを取り出す(これらは固定。推定するのはスケーリングのみ)
    uf = nominal_params['uf']
    rhoc = nominal_params['rhoc']

    #目的関数：密度を b で割ってから名目速度関数に入れ、L1誤差の合計を取る(Eq(20))
    def objective(x, grad):
        b = x[0]
        pred = underwood_speed_density(X_train / b, uf, rhoc)#rhoをbで割ったものを入力
        return np.sum(np.abs(Y_train - pred))

    #ISRES(GN_ISRES)で最適化(推定するのはスケーリングパラメータ1つだけ)
    #    論文Table5の実測値: b1 = 2.5996
    opt = nlopt.opt(nlopt.GN_ISRES, 1)
    opt.set_lower_bounds([lower_bound])
    opt.set_upper_bounds([upper_bound])
    opt.set_min_objective(objective)
    opt.set_maxeval(max_eval)
    best_param = opt.optimize([init_param])

    #テストデータでMAEを評価
    b = best_param[0]
    Y_pred_test = underwood_speed_density(X_test / b, uf, rhoc)#rhoをaで割ったものを入力
    test_mae = mean_absolute_error(Y_test, Y_pred_test)

    print(f"--- mode_id={mode_id} スケーリングパラメータ推定結果 ---")
    print(f"b = {b:.4f}")
    print(f"テストMAE: {test_mae:.3f} km/h")

    return {'mode_id': mode_id, 'scaling_param': b, 'test_mae': test_mae}


################################################################
#データの確認
################################################################
def plot_density_speed_by_mode(df, pair_types=(1, 2, 3, 4), ft_to_m=0.3048, figsize=(12, 10)):
    """
    ペアタイプ（mode_id）ごとに密度-速度の散布図を2x2で並べて表示する関数。
    mdensity_veh_km, mode_id, v_Velを持つDataFrame（compute_micro_density実行後）に対して使う。
    速度(v_Vel, ft/s)はここでkm/hにその場で変換してプロットする（df自体は変更しない）。
    """
    mode_map = {1: 'C-C', 2: 'C-T', 3: 'T-C', 4: 'T-T'}

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, pt in zip(axes.flat, pair_types):
        sub = df[df['mode_id'] == pt]
        speed_kmh = sub['v_Vel'] * ft_to_m * 3.6

        ax.scatter(sub['mdensity_veh_km'], speed_kmh, s=3, alpha=0.3, color='steelblue')
        ax.set_xlabel('micro_Density (veh/km)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(f'{mode_map.get(pt, pt)} (n={len(sub):,} frames)')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fundamental_diagram(
    snapshots,
    classes=(2, 3),
    class_labels={2: 'Car', 3: 'Truck'},
    bin_width=5.0,
    max_density=100,
    show_mean_trend=True,
    scatter_color='steelblue',
    trend_color='red',
):
    """マクロスナップショットデータから車種ごとの基本図 (q-k図: 密度 vs 流量)

    を描画する関数

    Parameters:
    -----------
    snapshots : pd.DataFrame
        compute_macroscopic_data の返り値 (density_veh_km, flow_veh_h, v_Class
        を含む)
    classes : tuple or list
        プロット対象の車種クラス (デフォルト: (2, 3))
    class_labels : dict
        車種コードと表示名の対応辞書
    bin_width : float
        平均トレンド線を計算する際の密度ビン幅 (veh/km/lane)
    max_density : float
        X軸の最大表示範囲 (veh/km/lane)
    show_mean_trend : bool
        ビンごとの平均流量トレンド線を描画するかどうか
    """
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))

    # クラスが1つの場合でもイテレーションできるようにリスト化
    if n_classes == 1:
        axes = [axes]

    # ビン分割用
    bins = np.arange(0, max_density + bin_width, bin_width)

    for ax, cls in zip(axes, classes):
        sub = snapshots[snapshots['v_Class'] == cls].copy()
        cls_name = class_labels.get(cls, f'Class {cls}')

        if len(sub) == 0:
            ax.set_title(f'{cls_name} (No Data)')
            continue

        # ① 散布図 (実測スナップショット)
        ax.scatter(
            sub['density_veh_km'],
            sub['flow_veh_h'],
            s=8,
            alpha=0.35,
            color=scatter_color,
            label='Snapshot Data',
        )

        # ② 密度区間ごとの平均流量トレンド線
        if show_mean_trend:
            sub['density_bin'] = pd.cut(
                sub['density_veh_km'], bins=bins, include_lowest=True
            )
            bin_stats = sub.groupby('density_bin', observed=True).agg(
                mean_flow=('flow_veh_h', 'mean'), count=('flow_veh_h', 'count')
            )
            # サンプル数が一定以上ある区間のみプロット
            valid_stats = bin_stats[bin_stats['count'] >= 5]
            bin_centers = [interval.mid for interval in valid_stats.index]

            ax.plot(
                bin_centers,
                valid_stats['mean_flow'],
                color=trend_color,
                linewidth=2.5,
                marker='o',
                markersize=4,
                label=f'Mean Flow (bin={bin_width})',
            )

        # 軸・装飾
        ax.set_xlabel('Density (veh/km/lane)')
        ax.set_ylabel('Flow (veh/h/lane)')
        ax.set_xlim(0, max_density)
        ax.set_ylim(bottom=0)
        ax.set_title(f'{cls_name} (n_snapshots={len(sub):,})')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_macroscopic_speed_density(
    df_macro,
    bin_width=5.0,
    max_density=100,
    min_samples=5,
    class_labels={2: 'Car', 3: 'Truck'},
    class_colors={2: 'steelblue', 3: 'darkorange'},
):
    """
    マクロ密度 (density_veh_km) と 平均速度 (speed_kmh)
    の関係を車種別にプロットする関数
    Parameters:
    -----------
    df_macro : pd.DataFrame
        compute_macroscopic_data の返り値 (density_veh_km, speed_kmh,
        v_Class を含む)
    bin_width : float
        密度をグループ化するビン幅 (veh/km/lane)
    max_density : float
        X軸の最大表示範囲 (veh/km/lane)
    min_samples : int
        平均速度を計算する際に必要な最小スナップショット数 (ノイズ除去用)
    class_labels : dict
        v_Class のラベル表示名
    class_colors : dict
        v_Class ごとのプロット色
    """
    plt.figure(figsize=(9, 6))

    # 1. 密度ビンの設定 (0 ~ max_density)
    bins = np.arange(0, max_density + bin_width, bin_width)

    # 2. 車種ごとにプロット
    for v_cls in sorted(df_macro['v_Class'].unique()):
        label_name = class_labels.get(v_cls, f'v_Class={v_cls}')
        color_code = class_colors.get(v_cls, 'gray')

        # 対象車種に絞り込み
        sub = df_macro[df_macro['v_Class'] == v_cls].copy()
        if len(sub) == 0:
            continue

        # ① 生スナップショットデータの散布図（薄く背景に表示）
        plt.scatter(
            sub['density_veh_km'],
            sub['speed_kmh'],
            s=8,
            alpha=0.15,
            color=color_code,
            label=f'{label_name} (Snapshots)',
        )

        # ② 密度ビンの割り当てとビンごとの平均速度集計
        sub['density_bin'] = pd.cut(
            sub['density_veh_km'], bins=bins, include_lowest=True
        )

        bin_stats = sub.groupby('density_bin', observed=True).agg(
            mean_speed=('speed_kmh', 'mean'), count=('speed_kmh', 'count')
        )

        # サンプル数が少ない区間を除外
        valid_stats = bin_stats[bin_stats['count'] >= min_samples]
        bin_centers = [interval.mid for interval in valid_stats.index]

        # ③ ビンごとの平均速度プロット（太線 + マーカー）
        plt.plot(
            bin_centers,
            valid_stats['mean_speed'],
            marker='o',
            markersize=5,
            linewidth=2.5,
            color=color_code,
            label=f'{label_name} Mean Speed',
        )

    # 3. グラフ装飾
    plt.xlabel('Macroscopic Density (veh/km/lane)')
    plt.ylabel('Mean Speed (km/h)')
    plt.title('Macroscopic Speed vs. Density (by Vehicle Class)')
    plt.xlim(0, max_density)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()