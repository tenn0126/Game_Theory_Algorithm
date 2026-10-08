#パイプラインで使用する関数をまとめたファイル
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
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
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# from src.data_prep import *
# from src.nominal_speed_func import *
# from src.macroscopic_data import *
# from src.cooperation_surplus import *

FT_TO_M = 0.3048
#################################################################################
#step0:データ準備
#################################################################################
#軌跡データの読み込み
def load_trajectory_data():
    #csvファイルではなくtxtファイルを読み込む
    path_1 = Path('../data/raw/trajectories-0400-0415.txt')
    path_2 = Path('../data/raw/trajectories-0500-0515.txt')
    path_3 = Path('../data/raw/trajectories-0515-0530.txt')

    columns = [
        'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y',
        'Global_X', 'Global_Y', 'v_Length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc',
        'Lane_ID', 'Preceding', 'Following', 'Space_Headway', 'Time_Headway'
    ]

    df_1 = pd.read_csv(path_1, sep=r'\s+', header=None, names=columns)
    df_2 = pd.read_csv(path_2, sep=r'\s+', header=None, names=columns)
    df_3 = pd.read_csv(path_3, sep=r'\s+', header=None, names=columns)

    # 時間帯の識別列を追加 (id の重複があるため)
    df_1['time_period'] = 1  # 4:00-4:15
    df_2['time_period'] = 2  # 5:00-5:15
    df_3['time_period'] = 3  # 5:15-5:30

    # 結合
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    # データの確認
    print(df.shape)
    df.head()
    return df

def remove_flicker_noise(df, col='Preceding', threshold=5):
    """
    先行車ID(Preceding)の短時間フリッカリングを除去する。
    「長いA → threshold以下のB → 長いA」の形（前後の値が完全一致する
    短いブロック）のみを対象とし、直前の値(A)で補間する。
    A→B→C→...→Aのような連鎖的な変化は対象外（そのまま残す）。
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    # 車両・時間帯・値のいずれかが変わったら新しいブロックの開始
    is_new_block = (
        (df['Vehicle_ID'] != df['Vehicle_ID'].shift()) |
        (df['time_period'] != df['time_period'].shift()) |
        (df[col] != df[col].shift())
    )
    #それぞれのブロックに特有のIDを付与していく
    df['_block_id'] = is_new_block.cumsum()

    #ブロックID事でグループ化し、そのグループが何フレーム続いているかなどの情報を取り出す
    blocks = df.groupby('_block_id').agg(
        Vehicle_ID=('Vehicle_ID', 'first'),
        time_period=('time_period', 'first'),
        value=(col, 'first'),
        frame_count=(col, 'size'),
    )

    # 前後ブロックは同一車両・時間帯の中でのみ参照する　現在ブロックの一個前と一個後の両方を取得する
    blocks['prev_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(1)
    blocks['next_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(-1)

    # フリッカリング判定：短いブロックの前後の値が完全一致する場合はノイズとして判定
    is_flicker = (
        (blocks['frame_count'] <= threshold)
        & (blocks['prev_value'] == blocks['next_value'])
        & blocks['prev_value'].notna()
    )

    #フリッカリングと判定された部分は直前の値をそのまま残し、それ以外はnanにしてしまう
    replacement = blocks['prev_value'].where(is_flicker)
    #ノイズ判定された行はprevの値(replacementに入っている)で埋められる　 mapした結果nanの行は元の値で埋める
    df[col] = df['_block_id'].map(replacement).fillna(df[col])
    # 補正した行に印を付ける（後で車間距離を座標から計算し直すため）
    df[f'_flicker_{col}'] = df['_block_id'].map(is_flicker).fillna(False).astype(bool)
    #block_idは不要なので消す   
    df = df.drop(columns='_block_id')

    print("\n--フリッカリング除去--")
    print(f"補正されたブロック数: {is_flicker.sum():,}")
    print(f"行数: {len(df):,}")

    return df

def recompute_headway_after_flicker(df, flag_cols=('_flicker_Preceding', '_flicker_Lane_ID')):
    """
    フリッカリング補正した行の Space_Headway / Time_Headway を、補正後の先行車(Preceding)の
    座標から計算し直す。
      Space_Headway = 先行車の Local_Y − 自車の Local_Y   (ft, どちらも前端中心 → NGSIMと同じ定義)
      Time_Headway  = Space_Headway / 自車速度 v_Vel      (s, 停止中はNGSIMと同じ 9999.99)
    次の行は計算できないので NaN にする（後段の filter_invalid_headway で除外される）:
      Preceding=0 / 先行車がその時刻に存在しない / 距離が0以下 / 先行車が別の車線にいる
    """
    df = df.copy()
    flag_cols = [c for c in flag_cols if c in df.columns]
    flag = df[flag_cols].any(axis=1)

    # 各時刻の全車両の位置と車線（先行車として突き合わせる）
    lead = df[['time_period', 'Global_Time', 'Vehicle_ID', 'Local_Y', 'Lane_ID']].rename(
        columns={'Vehicle_ID': 'Preceding', 'Local_Y': 'Local_Y_lead', 'Lane_ID': 'Lane_ID_lead'})
    tgt = df.loc[flag, ['time_period', 'Global_Time', 'Preceding', 'Local_Y', 'v_Vel', 'Lane_ID']]
    m = (tgt.reset_index()
            .merge(lead, on=['time_period', 'Global_Time', 'Preceding'], how='left')
            .set_index('index')
            .loc[tgt.index])

    new_space = m['Local_Y_lead'] - m['Local_Y']
    same_lane = m['Lane_ID_lead'] == m['Lane_ID']
    base_ok = (m['Preceding'] != 0) & new_space.notna() & (new_space > 0)
    ok = base_ok & same_lane

    new_space = new_space.where(ok)
    new_time = pd.Series(np.where(m['v_Vel'] > 0, new_space / m['v_Vel'], 9999.99),
                         index=m.index).where(ok)

    old_space = df.loc[flag, 'Space_Headway']
    df.loc[flag, 'Space_Headway'] = new_space
    df.loc[flag, 'Time_Headway'] = new_time

    changed = ok & ((new_space - old_space).abs() > 1.0)
    print("\n--フリッカリング補正行の車間距離を座標から再計算--")
    print(f"対象行: {int(flag.sum()):,}  再計算できた行: {int(ok.sum()):,}  NaNにした行: {int((~ok).sum()):,}")
    print(f"  NaNの内訳: 先行車が別車線 {int((base_ok & ~same_lane).sum()):,}行 / "
          f"先行車なし・位置不明・距離0以下 {int((~base_ok).sum()):,}行")
    print(f"  元の値から1ft超変わった行: {int(changed.sum()):,}  (元の値が0だった行: {int((old_space == 0).sum()):,})")
    
    return df

def give_pair_id(df, max_gap_frames=10, frame_interval_ms=100):
    """
    追従ペアIDを付与する関数

    区切り条件（いずれか1つでも満たせば追従の切れ目）：
    - Vehicle_IDが変わった
    - time_periodが変わった
    - Lane_IDが変わった（自車の車線変更）
    - Precedingが変わった（先行車の交代・消失）
    - 直前の行とのGlobal_Timeの差が許容範囲を超える（=フレーム欠損が大きすぎる）

    max_gap_frames: 連続して許容する欠損フレーム数(デフォルト10=1秒分まで許容)
    """
    #データを使いやすい形にソート
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    #区切り条件に当てはまる部分に旗を立てていく
    vehicle_changed      = df['Vehicle_ID']  != df['Vehicle_ID'].shift()
    time_period_changed  = df['time_period'] != df['time_period'].shift()
    lane_changed          = df['Lane_ID']     != df['Lane_ID'].shift()
    preceding_changed     = df['Preceding']   != df['Preceding'].shift()

    # 直前の行との時間差(ms)。フレーム欠損が大きすぎる場合も区切りとして扱う
    time_gap = df['Global_Time'] - df['Global_Time'].shift()
    max_allowed_gap_ms = (max_gap_frames + 1) * frame_interval_ms
    #データの端でtime_gapが欠損値になる可能性もあるため、欠損値の時か許容フレームを超えるときに旗を立てる
    gap_too_large = (time_gap > max_allowed_gap_ms) | time_gap.isna()

    #どれか一つでも条件がtrueならば追従の切れ目
    break_point = (
        vehicle_changed | time_period_changed | lane_changed |
        preceding_changed | gap_too_large
    )

    #切れ目ごとに数字を変えていく　これがそれぞれのペアが持つ固有のIDとなる
    df['pair_id'] = break_point.cumsum()

    # Preceding=0の行はpair_idをNaNにする（追従関係がないため）後での分析対象外
    preceding_zero = df['Preceding'] == 0
    df.loc[preceding_zero, 'pair_id'] = None

    print("\n--ペアIDを付与--")
    print(f"総行数: {len(df):,}")
    print(f"追従ペア数: {df['pair_id'].nunique():,}")
    return df

def give_mode_id(df, original_df):
    """
    追従ペアにモードIDを付与する関数
    
    モードID（ego_class, leader_class）：
    (c-c)=1, (c-t)=2, (t-c)=3
    (t-t)=4, (others)=5,
    車両クラス:
    auto=1,car=2,truck=3
    """
    mode_map = {
        (2,2): 1, (2,3): 2, (3,2): 3,
        (3,3): 4, (2,1): 5, (3,1): 5,
        (1,1): 5, (1,2): 5, (1,3): 5
    }
    
    # 元データから前方車両の車種を取得
    #元データから車両id,車種、timeperiodの3列を抜き出す　重複するものは消しておく
    leader_class = original_df[['Vehicle_ID', 'v_Class','time_period']].drop_duplicates()
    #名前を変更
    leader_class = leader_class.rename(columns={'Vehicle_ID': 'Preceding', 'v_Class': 'leader_class'})
    
    # 結合 mergeも元のdfとは違う新しいdataframeを返すので.copy()はなくてOK
    df = df.merge(leader_class, on=['Preceding', 'time_period'], how='left')
    #後続車のクラスを示す行を新たに作成
    df['ego_class'] = df['v_Class']
    
    # モードIDを付与（NaNの場合はNoneを返す）
    def get_mode_id(row):
        if pd.isna(row['leader_class']):
            return None
        return mode_map.get((int(row['ego_class']), int(row['leader_class'])))

    #1行ずつmodeidを取得し、mode_id列に入れていく
    df['mode_id'] = df.apply(get_mode_id, axis=1)
    
    # 確認
    pairs_count = df.dropna(subset=['pair_id']).groupby(['pair_id', 'mode_id']).size().reset_index()
    summary = pairs_count.groupby('mode_id').size()
    print("\n--モードID別を付与--")
    
    print(f"総行数: {len(df):,}")
    print(f"追従ペア数: {df['pair_id'].nunique():,}")
    print(summary)
    
    return df

def filter_valid_pairs(df):
    """
    pair_idとmode_idが付与されている行のみ残す関数
    （追従関係がない行を除外）
    """
    df_filtered = df.dropna(subset=['pair_id', 'mode_id'])
    
    print("\n--追従していない軌跡データを削除--")
    print(f"フィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"ペア数: {df_filtered['pair_id'].nunique():,}")
    
    return df_filtered

def filter_lane(df, lanes=[2, 3, 4]):
    """
    車線フィルタ
    lanes: 使用する車線のリスト（デフォルト: Lane 2,3,4）
    """
    df_filtered = df[df['Lane_ID'].isin(lanes)].copy()


    print("\n--車線フィルタ(2,3,4レーンを残す)--")
    print(f"フィルタ前: {len(df):,}行")
    print(f"フィルタ後: {len(df_filtered):,}行")
    print(f"除外された行数: {len(df) - len(df_filtered):,}行")
    print(f"追従ペア数: {df_filtered['pair_id'].nunique():,}")

    print(f"\n車線の分布:")
    print(df_filtered['Lane_ID'].value_counts().sort_index())
    
    return df_filtered

def filter_mode(df, modes=[1,2,3,4]):
    """
    特定のモードIDのみを残すフィルタ（ペア単位で安全に処理）
    modes: 使用するモードIDのリスト
    デフォルト: [1,2,3,4]
    """
    
    #各ペアIDの代表モードIDを取得（最初の行の値を使う）ペアIDでグループ化しそのグループのmodeを見る
    pair_modes = df.dropna(subset=['pair_id']).groupby('pair_id')['mode_id'].first()
    
    #指定したモード(1,2,3,4)に該当する「ペアIDのリスト」を作成
    valid_pair_ids = pair_modes[pair_modes.isin(modes)].index
    
    #元のデータから、有効なペアIDの行を丸ごと残す
    df_filtered = df[df['pair_id'].isin(valid_pair_ids)]
    
    print("\n--車種フィルタ(1,2,3,4の追従モードを残す)--")
    print(f"フィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    print(f"除外されたペア数: {df['pair_id'].nunique() - df_filtered['pair_id'].nunique():,}")
    print(f"フィルタ後行数: {len(df_filtered):,}行")
    print(f"\nモードID別ペア数:")
    pairs_count = df_filtered.groupby(['pair_id', 'mode_id']).size().reset_index()
    print(pairs_count.groupby('mode_id').size())
    
    return df_filtered

def filter_acceleration(df, max_acc=1.0):
    """
    加速度フィルタ
    max_acc: 最大加速度・減速度（m/s²）（デフォルト: 1.0 m/s²）
    """
    # 閾値をfeet/s²に変換
    max_acc_feet = max_acc / 0.3048
    
    # 閾値より低い加速度の絶対値を持つデータを抽出 ペアは関係ない処理
    df_filtered = df[df['v_Acc'].abs() <= max_acc_feet]

    print("\n--加速度フィルタ--")
    #print(f"閾値: {max_acc} m/s² = {max_acc_feet:.4f} feet/s²")
    print(f"フィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"フィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    
    return df_filtered

def filter_duration(df, min_duration=60):
    """
    追従時間フィルタ
    min_duration: 最小追従時間（秒）（デフォルト: 60秒）
    """
    # 各ペアの追従時間を計算 ペアIDでグループ化し、グループ内での追従時間の最大値と最小値の差が追従時間
    pair_duration = df.groupby('pair_id')['Global_Time'].agg(
        duration=lambda x: (x.max() - x.min()) / 1000  + 0.1 # ミリ秒→秒　
    )
    
    # min_duration以上のペアIDを取得
    valid_pair_ids = pair_duration[pair_duration['duration'] >= min_duration].index
    
    # 有効なペアIDの行だけ残す
    df_filtered = df[df['pair_id'].isin(valid_pair_ids)]

    print("\n--追従時間フィルタ--")
    print(f"フィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    print(f"除外されたペア数: {df['pair_id'].nunique() - df_filtered['pair_id'].nunique():,}")
    print(f"フィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    
    return df_filtered

def apply_cutoff(df, cutoff=10):
    """
    カットオフウィンドウ
    cutoff: カットオフ時間（秒）（デフォルト: 10秒）
    """
    cutoff_ms = cutoff * 1000  # 秒→ミリ秒
    
    # ペアごとの最小・最大時刻を計算
    pair_times = df.groupby('pair_id')['Global_Time'].agg(
        min_time='min',
        max_time='max'
    ).reset_index()
    
    # 元のデータに結合　各行にそのペアの最大時間と最小時間の列が追加される
    df_merged = df.merge(pair_times, on='pair_id')
    
    # カットオフ条件でフィルタリング　最大最小時間の列は不要なので消してしまう
    df_filtered = df_merged[
        (df_merged['Global_Time'] >= df_merged['min_time'] + cutoff_ms) &
        (df_merged['Global_Time'] <= df_merged['max_time'] - cutoff_ms)
    ].drop(columns=['min_time', 'max_time'])

    print("\n--カットオフウィンドウ--")
    print(f"カットオフ前の行数: {len(df):,}")
    print(f"カットオフ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"カットオフ前のペア数: {df['pair_id'].nunique():,}")
    print(f"カットオフ後のペア数: {df_filtered['pair_id'].nunique():,}")
    
    return df_filtered

#################################################################################
#step1:名目速度関数・スケーリングパラメータの推定
################################################################################
#異常な車間距離を排除する
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


#ミクロ密度の計算
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

#ロジスティックモデルでのパラメータ推定 (speed target)
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

#Underwoodモデルでのパラメータ推定
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

#スケーリングパラメータ推定
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



#################################################################################
#step2:マクロデータ計算
################################################################################
#　スナップショットからの密度、平均速度を計算する。
# 密度は車種ごとに計算。
#　スナップショットからの密度、平均速度を計算する。
# 密度は"車種ごと"に計算。

def compute_macroscopic_data(df, lanes=(2, 3, 4), classes=(2, 3), snapshot_step_frames=5,
                                    L_m=None, n_lanes=None, frame_interval_ms=100):
    """
    一定間隔(デフォルト5フレーム=0.5秒)でスナップショットを取り、車種ごとに
    マクロ密度(veh/km/lane)・平均速度(km/h)・速度の標準偏差(km/h)・流率(veh/h/lane)を計算する。

    スナップショット時刻は全time_period共通で「Global_Timeが500msの倍数」とする。
    (period2と3はGlobal_Timeが約180秒重なっているが、全期間で同じ時刻を使うため
     刻みがずれて余分なスナップショットが混ざることはない。重なり区間そのものは
     両期間に残るので、先頭・末尾の除外で別途処理する)

    df: trajectoryデータ(関数内でlanes/classesに絞る前のデータでも可)
    lanes: 対象車線
    classes: 対象車種(2=car, 3=truck)
    snapshot_step_frames: 何フレームおきにスナップショットを取るか(論文: 0.5秒)
    L_m: 区間長(m)。Noneなら対象データのLocal_Yの範囲から自動計算
    n_lanes: 車線数。Noneならlanesの数

    戻り値の列:
      time_period, Global_Time, v_Class,
      n_vehicles     : スナップショット内の台数
      speed_kmh      : 平均速度(論文Eq.10)
      speed_std_kmh  : 速度の標準偏差(不偏, ddof=1)。1台だけの時刻はNaN
      density_veh_km : マクロ密度(論文Eq.9)
      flow_veh_h     : 流率
    """
    #snaphshot間隔を時間に直す
    step_ms = frame_interval_ms * snapshot_step_frames  # 500ms

    # time_periodごとに、スナップショットを撮る時刻の集合を作成 
    #valid_times = set()

    #スナップショットを撮る配列作成は使用しないように変更(26/10/03)
    # #time_periodごとにグループ分けし１ブロックずつループ処理
    # for tp, group in df.groupby('time_period'):
    #     # 観測時間の最小値(t_min)・最大値(t_max)を取得
    #     t_min = int(group['Global_Time'].min())
    #     t_max = int(group['Global_Time'].max())
        
    #     # t_min始まりの0.5秒(500ms)ごとの時刻配列を作成し、集合に追加
    #     snapshot_times_tp = set(range(t_min, t_max + 1, step_ms))
    #     #上で作った集合に追加し、全時間区間のスナップショットを撮る時間リストを作成（globaltimeは重複しないためデータの重複もないはず）
    #     valid_times.update(snapshot_times_tp)

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

    #スナップショットリストを使わないよう変更(26/10/03)
    # # gloobaltimeが以前作ったスナップショットを撮る時間リストに入っている行を抽出
    # snap_rows = target[target['Global_Time'].isin(valid_times)]

    # スナップショット時刻(Global_Timeが500msの倍数)の行だけ抽出
    snap_rows = target[target['Global_Time'] % step_ms == 0].copy()
    #速度をkm/hへ変換 (feet/s -> m/s -> km/h)
    snap_rows['v_kmh'] = snap_rows['v_Vel'] * FT_TO_M * 3.6
    
    #抽出したスナップショットを撮るデータを時間と車種でグループ化し、グループごとに集計処理
    agg = snap_rows.groupby(['time_period', 'Global_Time', 'v_Class']).agg(
        n_vehicles=('Vehicle_ID', 'size'),#車種ごとの車両数　Vehicle_ID 列のデータ行数（size）を数えて、n_vehicles（車両数）という列名で保存
        speed_kmh=('v_kmh', 'mean'),#平均速度　feet/s -> m/s -> km/h　v_Vel 列を単位換算は先に計算されている。平均値（mean）を計算し、speed_kmh（平均速度）という列名で保存　
        speed_std_kmh=('v_kmh', 'std'),   # ddof=1。1台だけの時刻はNaN　標準偏差を計算
    ).reset_index()

    agg['density_veh_km'] = agg['n_vehicles'] / (L_km * n_lanes)#マクロ密度の計算　面積で車両数を割る
    agg['flow_veh_h'] = agg['density_veh_km'] * agg['speed_kmh']#流量の計算　
    print("\n--マクロ密度・平均速度を計算--")

    return agg

#################################################################################
#step3:協力余剰
################################################################################
# 論文Eq(7)の残差を求める関数（その時のu*がどれくらい式(7)の等式を満たしているか）
# # ※ underwood_density_speed, logistic_density_speed は
#    nominal_speed_func.py に定義済みのものをそのまま使う
# a1=1（cc追従）、 b2=1（tt追従）

#ロジスティックモデルを逆関数にして、速度から密度を予測する関数
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

#Underwoodモデルを逆関数にして、速度から密度を予測する関数
def underwood_density_speed(v, uf, rhoc):
    """
    逆関数: 速度 v から 密度 rho を予測
    g(rho) = uf * exp(-rho/rhoc) を rho について解くと、
    rho = rhoc * ln(uf / v)
    """
    # 数学的に定義できない領域（v <= 0 または v >= uf）に対する数値的クリップ
    eps = 1e-6
    v_clipped = np.clip(v, eps, uf - eps)

    return rhoc * np.log(uf / v_clipped)

# def _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1):
#     rho_tot = rho1 + rho2

#     #u*の時のu1,u2の逆関数を計算　u1_invは密度
#     u1_inv = logistic_density_speed(
#         u, car_params['ub'], car_params['uf'], car_params['rhoc'],
#         car_params['theta1'], car_params['theta2']
#     )
#     u2_inv = underwood_density_speed(u, truck_params['uf'], truck_params['rhoc'])

#     #sigma_j=1^j=2を崩して書いた形
#     term1 = (rho1 / u1_inv) * (rho1 / 1.0 + rho2 / a2)
#     term2 = (rho2 / u2_inv) * (rho1 / b1 + rho2 / 1.0)

#     #eq(7)の左辺と右辺の差を返す。0に近いほど式(7)を満たす
#     return (term1 + term2) / rho_tot - 1.0

def _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1, scaling):
    rho_tot = rho1 + rho2

    # u*におけるクラス自身の名目速度関数の逆関数（=密度）
    u1_inv = logistic_density_speed(
        u, car_params['ub'], car_params['uf'], car_params['rhoc'],
        car_params['theta1'], car_params['theta2']
    )
    u2_inv = underwood_density_speed(u, truck_params['uf'], truck_params['rhoc'])

    if scaling == 'printed':
        c_ct, c_tc = 1.0 / a2, 1.0 / b1
    elif scaling == 'inverted':
        c_ct, c_tc = a2, b1
    else:
        raise ValueError(f"scaling は {_VALID_SCALINGS} のどちらかを指定してください")

    term1 = (rho1 / u1_inv) * (rho1 + rho2 * c_ct)   # 車(クラス1)の項
    term2 = (rho2 / u2_inv) * (rho1 * c_tc + rho2)   # トラック(クラス2)の項
    return (term1 + term2) / rho_tot - 1.0


# #brent法でu*を求める関数
# def solve_1pipe_speed(rho1, rho2, car_params, truck_params, a2, b1,
#                        u_lower=None, u_upper=None, margin=1e-3):
#     """
#     論文Eq(7)をBrent法（scipy.optimize.brentq）で解き、1-pipe速度u*を求める。

#     rho1, rho2: マクロ密度（veh/km）。クラス1=車、クラス2=トラック
#     car_params: fit_nominal_speed_logistic の戻り値（C-C）
#     truck_params: fit_nominal_speed_underwood の戻り値（T-T）
#     a2: fit_scaling_parameter_logistic の戻り値の'scaling_param'
#     b1: fit_scaling_parameter_underwood の戻り値の'scaling_param'
#     u_lower, u_upper: 探索区間。省略時は車のub・両クラスのufから自動決定
#     """
#     if rho1 + rho2 <= 0:
#         return np.nan  # 車もトラックもいないスナップショットは対象外

#     if u_lower is None or u_upper is None:
#         lowers, uppers = [], []
#         if rho1 > 0:
#             #lowers.append(car_params['ub'] + margin)
#             lowers.append(margin)#より頑健な下限　
#             # 車の"uf"は密度→-∞の極限値で、密度0でも到達できない。
#             # 密度0での実際の到達可能な最高速度 f(0) を上限にする。
#             car_speed_at_zero = logistic_speed_density(
#                 0.0, car_params['ub'], car_params['uf'], car_params['rhoc'],
#                 car_params['theta1'], car_params['theta2']
#             )
#             uppers.append(car_speed_at_zero - margin)
#         if rho2 > 0:
#             lowers.append(margin)#下限は定義できない
#             uppers.append(truck_params['uf'] - margin)  # Underwoodはg(0)=ufなので問題なし
#         if u_lower is None:
#             u_lower = max(lowers)
#         if u_upper is None:
#             u_upper = min(uppers)

#     #uだけを引数に持つ関数を新たに作る
#     f = lambda u: _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1)
#     #ブレント法でf(u)=0となるuを求める
#     return brentq(f, u_lower, u_upper)


_VALID_SCALINGS = ('printed', 'inverted')

def _check_scaling(scaling):
    """scalingの指定ミスを、try/exceptに飲み込まれる前に検出する。"""
    if scaling not in _VALID_SCALINGS:
        raise ValueError(f"scaling は {_VALID_SCALINGS} のどちらかを指定してください (指定値: {scaling!r})")


def solve_1pipe_speed(rho1, rho2, car_params, truck_params, a2, b1, scaling,
                      u_lower=None, u_upper=None, margin=1e-3):
    """
    論文Eq(7)をBrent法（scipy.optimize.brentq）で解き、1-pipe速度u*を求める。
    rho1, rho2: マクロ密度（veh/km/lane）。クラス1=車、クラス2=トラック
    scaling: 'printed' か 'inverted'（デフォルトなし。必ず明示する）
    """
    _check_scaling(scaling)

    if rho1 + rho2 <= 0:
        return np.nan  # 車もトラックもいないスナップショットは対象外

    if u_lower is None:
        # ロジスティックの逆関数はub付近でクリップされ一定値になるため、
        # 0付近から探索しても解はub以上で一意に求まる
        u_lower = margin

    if u_upper is None:
        uppers = []
        if rho1 > 0:
            # 車のufは到達できない漸近値なので、密度0での実際の最高速度f(0)を上限にする
            car_speed_at_zero = logistic_speed_density(
                0.0, car_params['ub'], car_params['uf'], car_params['rhoc'],
                car_params['theta1'], car_params['theta2']
            )
            uppers.append(car_speed_at_zero - margin)
        if rho2 > 0:
            uppers.append(truck_params['uf'] - margin)  # Underwoodはg(0)=uf
        u_upper = min(uppers)

    f = lambda u: _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1, scaling)
    return brentq(f, u_lower, u_upper)


# # マクロ密度のスナップショット表全体(rho1,rho2の組み合わせ)に対して1-pipe速度を計算
# def compute_1pipe_speed_for_snapshots(df_macro, car_params, truck_params, a2, b1,
#                                         car_class=2, truck_class=3):
#     """
#     df_macro: compute_macroscopic_data の返り値
#               （time_period, Global_Time, v_Class, density_veh_km, ... を含む）
#     """
#     #縦持ちのデータを横並びに変換　
#     pivot = df_macro.pivot_table(
#         index=['time_period', 'Global_Time'],#timeperiodとGlobal_Timeが同じなら同じ行にまとめる
#         columns='v_Class',#クラスを新たな列名にする
#         values=['density_veh_km', 'speed_kmh', 'speed_std_kmh', 'n_vehicles']#その時の密度を入れる
#     )

#     #2段の列名を1段のわかりやすい名前に変換
#     col_map = {
#         ('density_veh_km', car_class): 'rho1',
#         ('density_veh_km', truck_class): 'rho2',
#         ('speed_kmh', car_class): 'u1_obs',    #車の観測平均速度 ũ1
#         ('speed_kmh', truck_class): 'u2_obs',  #トラックの観測平均速度 ũ2
#         ('speed_std_kmh', car_class): 'u1_std',
#         ('speed_std_kmh', truck_class): 'u2_std',
#         ('n_vehicles', car_class): 'n1',
#         ('n_vehicles', truck_class): 'n2'
#     }
#     pivot.columns = [col_map[c] for c in pivot.columns]

#     # #密度は「いない=0」でよいが、速度は「いない=定義できない」のでNaNのまま残す
#     # pivot[['rho1', 'rho2']] = pivot[['rho1', 'rho2']].fillna(0.0)
#     #密度と台数は「いない=0」、速度と標準偏差は「定義できない」のでNaNのまま
#     pivot[['rho1', 'rho2', 'n1', 'n2']] = pivot[['rho1', 'rho2', 'n1', 'n2']].fillna(0.0)
#     pivot = pivot.reset_index()

#     #1行ずつ1-pipe速度を計算する関数
#     #もし、下限と上限の範囲で0をはさめず、解が存在しない場合(ブレント法のエラーになる場合)も止まらないようにnanを返す
#     def _solve_row(row):
#         try:
#             return solve_1pipe_speed(row['rho1'], row['rho2'], car_params, truck_params, a2, b1)
#         except ValueError:
#             return np.nan

#     pivot['rho_tot'] = pivot['rho1'] + pivot['rho2']
#     pivot['u_1pipe'] = pivot.apply(_solve_row, axis=1)#すべての行に適応　結果を新しい列として追加

#     print(f"1-pipe速度を計算: {len(pivot):,}スナップショット中、"
#           f"{pivot['u_1pipe'].notna().sum():,}件で解を取得")

#     return pivot
# マクロ密度のスナップショット表全体に対して1-pipe速度を計算
def compute_1pipe_speed_for_snapshots(df_macro, car_params, truck_params, a2, b1, scaling,
                                      car_class=2, truck_class=3):
    """
    df_macro: compute_macroscopic_data の返り値
              （time_period, Global_Time, v_Class, density_veh_km, speed_kmh を含む）
    戻り値: rho1, rho2, u1_obs, u2_obs, rho_tot, u_1pipe 列を持つ表
    """
    _check_scaling(scaling)  # ループに入る前に検査（_solve_rowのexceptに飲み込ませない）

    # 縦持ち → 横持ち（密度と観測平均速度の両方を残す）
    pivot = df_macro.pivot_table(
        index=['time_period', 'Global_Time'],
        columns='v_Class',
        values=['density_veh_km', 'speed_kmh', 'speed_std_kmh', 'n_vehicles']
    )

    col_map = {
        ('density_veh_km', car_class): 'rho1',
        ('density_veh_km', truck_class): 'rho2',
        ('speed_kmh', car_class): 'u1_obs',    # 車の観測平均速度 ũ1
        ('speed_kmh', truck_class): 'u2_obs',  # トラックの観測平均速度 ũ2
        ('speed_std_kmh', car_class): 'u1_sd',    #車の速度の標準偏差
        ('speed_std_kmh', truck_class): 'u2_sd',  #トラックの速度の標準偏差
        ('n_vehicles', car_class): 'n1',          #車の台数
        ('n_vehicles', truck_class): 'n2',        #トラックの台数
    }
    # 対象外クラス（例: バイク）が混ざっていても落ちないよう、対象列だけ残す
    pivot = pivot[[c for c in pivot.columns if c in col_map]]
    pivot.columns = [col_map[c] for c in pivot.columns]

    # 片方のクラスが一度も出てこないデータでも列を用意する
    for col in ['rho1', 'rho2', 'u1_obs', 'u2_obs']:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # 密度は「いない=0」、速度は「いない=定義できない」のでNaNのまま
    pivot[['rho1', 'rho2', 'n1', 'n2']] = pivot[['rho1', 'rho2', 'n1', 'n2']].fillna(0.0)
    pivot = pivot.reset_index()

    # 解が挟めない(brentqの符号エラー)場合のみNaNにする
    def _solve_row(row):
        try:
            return solve_1pipe_speed(row['rho1'], row['rho2'], car_params, truck_params,
                                     a2, b1, scaling)
        except ValueError:
            return np.nan

    pivot['rho_tot'] = pivot['rho1'] + pivot['rho2']
    pivot['u_1pipe'] = pivot.apply(_solve_row, axis=1)

    n_ok = pivot['u_1pipe'].notna().sum()
    n_need = (pivot['rho_tot'] > 0).sum()
    print(f"1-pipe速度を計算(scaling={scaling}): {len(pivot):,}スナップショット中、{n_ok:,}件で解を取得")
    if n_ok < n_need:
        print(f"  ※警告: 解が得られなかったスナップショットが {n_need - n_ok:,}件あります")

    return pivot



KM_TO_MILE = 1.60934  # 1 mile = 1.60934 km


#最小道路占有
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

#協力余剰
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

#################################################################################
#step4:交通レジーム判定
################################################################################
# Step 4: 交通状態(traffic regime)の判定  論文Eq(11)
#   2-pipe : ũ1 >= u* かつ ũ2 >= u* (ただし両方"="は同時に成り立たない)
#   1-pipe : ũ1 = ũ2 = u*
#   上のどちらでもない → non-Pareto-efficient NE
# 実データの速度は連続値で"="が厳密には成り立たないので、|ũi - u*| <= eps を"="とみなす
# (論文にはepsの値は書かれていない)

def _resolve_eps(eps_kmh):
    """eps_kmhを (車のε, トラックのε) の組に変換する。
    数値 → 両クラス共通 / (eps1, eps2) → クラス別 / estimate_eps_from_seの戻り値(dict) → その'eps1','eps2'"""
    if isinstance(eps_kmh, dict):
        return eps_kmh['eps1'], eps_kmh['eps2']
    if isinstance(eps_kmh, (tuple, list)):
        if len(eps_kmh) != 2:
            raise ValueError("eps_kmh をタプルで渡す場合は (車のε, トラックのε) の2要素にしてください")
        return eps_kmh[0], eps_kmh[1]
    return eps_kmh, eps_kmh


def classify_traffic_regime(df_s, eps_kmh, u1_floor=None, u1_col='u1_obs', u2_col='u2_obs',
                            u_star_col='u_1pipe', s_col='s'):
    """
    u1_floor: 車の名目速度関数の下限速度（ロジスティックの ub, km/h）。
              指定すると ũ1 < u1_floor の時刻を 'out-of-domain'（モデルの適用範囲外）として
              判定対象(Pcoopの分母)から外す。Noneなら外さない（従来どおり）。
    """
    eps1, eps2 = _resolve_eps(eps_kmh)
    df = df_s.copy()
    df['du1'] = df[u1_col] - df[u_star_col]
    df['du2'] = df[u2_col] - df[u_star_col]

    valid = df[['du1', 'du2']].notna().all(axis=1)
    # モデルの適用範囲外: 観測した車の平均速度が、名目速度関数の下限より遅い
    if u1_floor is not None:
        out_of_domain = valid & (df[u1_col] < u1_floor)
    else:
        out_of_domain = pd.Series(False, index=df.index)
    judged = valid & ~out_of_domain

    eq1 = df['du1'].abs() <= eps1
    eq2 = df['du2'].abs() <= eps2
    ge1 = df['du1'] >= -eps1
    ge2 = df['du2'] >= -eps2
    is_1pipe = judged & eq1 & eq2
    is_2pipe = judged & ge1 & ge2 & ~(eq1 & eq2)

    df['regime'] = 'non-NE'
    df.loc[is_2pipe, 'regime'] = '2-pipe'
    df.loc[is_1pipe, 'regime'] = '1-pipe'
    df.loc[out_of_domain, 'regime'] = 'out-of-domain'
    df.loc[~valid, 'regime'] = 'undefined'

    counts = df['regime'].value_counts()
    n_tot = int(judged.sum())   # Pcoopの分母 = 2-pipe + 1-pipe + non-NE（範囲外は含めない）
    n_2pipe = int(counts.get('2-pipe', 0))
    print(f"交通状態を判定(eps: 車 {eps1:.2f}, トラック {eps2:.2f} km/h): 判定対象 {n_tot:,}件 "
          f"(undefined: {int(counts.get('undefined', 0)):,}件, "
          f"適用範囲外 ũ1<{u1_floor}: {int(counts.get('out-of-domain', 0)):,}件)")
    for r in ['2-pipe', '1-pipe', 'non-NE']:
        n = int(counts.get(r, 0))
        pct = 100 * n / n_tot if n_tot > 0 else float('nan')
        print(f"  {r:>10}: {n:,}件 ({pct:.2f}%)")
    if n_tot > 0:
        print(f"  Pcoop = {100 * n_2pipe / n_tot:.2f}% (論文: 13.83%)")
    if s_col in df.columns:
        print(f"  平均s: 2-pipe = {df.loc[df['regime'] == '2-pipe', s_col].mean():.4f}, "
              f"1-pipe = {df.loc[df['regime'] == '1-pipe', s_col].mean():.4f} (論文: 0.15, 0.11)")
    return df


# 立ち上がり部分(各time_periodの最初のskip_sec秒)を除外する補助関数
# Step4以降(集計・プロット・αの推定)で同じデータ範囲を使うために、Step4の前にかける
def drop_warmup(df, skip_sec=100):
    """
    df: time_period, Global_Time列を含むスナップショット表
    skip_sec: 各time_periodの開始からこの秒数未満の行を除外する
    """
    t0 = df.groupby('time_period')['Global_Time'].transform('min')
    elapsed_sec = (df['Global_Time'] - t0) / 1000
    out = df[elapsed_sec >= skip_sec].copy()
    print(f"各time_periodの最初の{skip_sec}秒を除外: {len(df):,}件 → {len(out):,}件")
    return out

# 終わり際(各time_periodの最後のskip_sec秒)を除外する補助関数
# NGSIMは記録終了までに区間を出なかった車両が記録されないため、
# 各time_periodの最後の100〜200秒で台数(密度)が過小になる
def drop_cooldown(df, skip_sec=200):
    """
    df: time_period, Global_Time列を含むスナップショット表
    skip_sec: 各time_periodの終了時刻からこの秒数以内の行を除外する
    """
    t_end = df.groupby('time_period')['Global_Time'].transform('max')
    remaining_sec = (t_end - df['Global_Time']) / 1000
    out = df[remaining_sec >= skip_sec].copy()
    print(f"各time_periodの最後の{skip_sec}秒を除外: {len(df):,}件 → {len(out):,}件")
    return out

#############################################################################
# Step 5: 協力性(collective cooperativeness)の存在判定  論文4.3.3
#############################################################################
#   存在する ⇔ s > 0 かつ 交通状態が2-pipe
def check_cooperation_existence(df_regime, s_col='s', regime_col='regime'):
    """
    df_regime: classify_traffic_regime の返り値（s, regime列を含む）

    戻り値: df_regime に 'coop_exists'(bool) 列を追加したコピー
            Step6(αの推定)では df[df['coop_exists']] を使う
    """
    df = df_regime.copy()

    is_2pipe = df[regime_col] == '2-pipe'
    s_pos = df[s_col] > 0
    df['coop_exists'] = is_2pipe & s_pos

    #判定対象(undefinedを除く)での集計
    judged = ~df[regime_col].isin(['undefined', 'out-of-domain'])   # 適用範囲外も判定対象から除く
    n_tot = int(judged.sum())
    n_2pipe = int(is_2pipe.sum())
    n_exist = int(df['coop_exists'].sum())
    n_spos = int((s_pos & judged).sum())
    #命題1では 2-pipe ⇔ s>0 なので、2-pipeなのにs<=0の行はモデルと観測が合っていない行
    n_2pipe_sneg = int((is_2pipe & ~s_pos).sum())

    def pct(n, d):
        return 100 * n / d if d > 0 else float('nan')

    print(f"協力性の存在判定: 判定対象 {n_tot:,}件")
    print(f"  s>0             : {n_spos:,}件 ({pct(n_spos, n_tot):.2f}%)")
    print(f"  2-pipe          : {n_2pipe:,}件 ({pct(n_2pipe, n_tot):.2f}%)  ← 論文Pcoop=13.83%")
    print(f"  2-pipe かつ s>0 : {n_exist:,}件 ({pct(n_exist, n_tot):.2f}%)  ← 協力性あり(Step6で使う)")
    print(f"  2-pipe かつ s<=0: {n_2pipe_sneg:,}件 ({pct(n_2pipe_sneg, n_2pipe):.2f}% of 2-pipe)  ← 命題1と不整合")

    
    return df

#############################################################################
# Step 6: 余剰分割係数λの推定  論文Eq(12)-(14), Section 5.2.3
#   ※Pythonでは lambda が予約語なので、変数名は lam を使う
#############################################################################
# 論文Eq(13)：λを与えたときの各クラスの予測速度
#   û1 = u1( ρ1 / (p1* + λ s) ),  û2 = u2( ρ2 / (p2* + (1-λ) s) )
#   u1, u2 はクラス自身の名目速度関数(C-C: ロジスティック, T-T: アンダーウッド)
def predict_speed_with_lambda(df, lam, car_params, truck_params):
    p1 = df['p1_star'] + lam * df['s']
    p2 = df['p2_star'] + (1.0 - lam) * df['s']

    # λが[0,1]の外だと道路占有が0以下になることがある（例: λ=2 で p2 = p2* - s < 0）。
    # 負の道路占有は物理的に無意味なので、その行の予測はNaNにする
    p1 = p1.where(p1 > 0)
    p2 = p2.where(p2 > 0)

    #入力の密度を変更し、u1_hatはその密度のときの速度の値が記録される
    u1_hat = logistic_speed_density(
        df['rho1'] / p1, car_params['ub'], car_params['uf'], car_params['rhoc'],
        car_params['theta1'], car_params['theta2']
    )
    u2_hat = underwood_speed_density(df['rho2'] / p2, truck_params['uf'], truck_params['rhoc'])
    return u1_hat, u2_hat


# 論文Eq(12)：損失関数 L(λ) = (1/T) Σt [ Σi wi |ũit - ûit| ]^2
#   各時刻で「重み付き絶対誤差の2クラス合計」を2乗し、時間で平均する
def surplus_split_loss(lam, df, car_params, truck_params, w1=0.5, w2=0.5):
    u1_hat, u2_hat = predict_speed_with_lambda(df, lam, car_params, truck_params)
    err_t = w1 * np.abs(df['u1_obs'] - u1_hat) + w2 * np.abs(df['u2_obs'] - u2_hat)
    # 1行でも道路占有が無効ならそのλ全体を無効にする
    # （pandasのmeanはNaNを無視するので、そのままだと無効な行を捨てた平均になってしまう）
    if err_t.isna().any():
        return np.inf
    return np.mean(err_t ** 2)


# 評価指標：クラス別MAEと重み付きMAE（論文Table 6）
def surplus_split_mae(lam, df, car_params, truck_params, w1=0.5, w2=0.5):
    u1_hat, u2_hat = predict_speed_with_lambda(df, lam, car_params, truck_params)
    mae1 = np.mean(np.abs(df['u1_obs'] - u1_hat))
    mae2 = np.mean(np.abs(df['u2_obs'] - u2_hat))
    return {'mae1': mae1, 'mae2': mae2, 'mae_weighted': w1 * mae1 + w2 * mae2}


# 論文Eq(14)：λ̂ = argmin L(λ) の1次元有界問題をBrent法で解く
#   bounds: λの探索範囲。論文の定義(Definition 2)は λ∈[0,1] なので既定値は(0,1)
def fit_surplus_split(df, car_params, truck_params, w1=0.5, w2=0.5, bounds=(0.0, 1.0)):
    res = minimize_scalar(
        surplus_split_loss, bounds=bounds, method='bounded',
        args=(df, car_params, truck_params, w1, w2)
    )
    return res.x


def estimate_surplus_split(df_coop, car_params, truck_params, w1=0.5, w2=0.5,
                           test_size=0.3, k=10, random_state=42, bounds=(0.0, 1.0)):
    """
    協力性ありのスナップショット(2-pipe かつ s>0)で余剰分割係数λを推定する。
    論文と同じく 70%/30% に分割し、学習データで k-fold 交差検証(k=10)を行う。

    df_coop: check_cooperation_existence の返り値（coop_exists, rho1, rho2, p1_star, p2_star, s,
             u1_obs, u2_obs 列を含む）
    car_params, truck_params: u*・p*の計算に使ったものと同じパラメータを渡すこと
    w1, w2: クラスの重み（論文はclass-based: w1=w2=0.5）
    bounds: λの探索範囲。既定値(0,1)は論文の定義どおり。
            境界解かどうか確かめたいときだけ (-1, 2) などに広げる

    戻り値: dict
      'lambda_hat'  : 学習データ全体で推定したλ
      'test_mae'    : テストデータでのクラス別MAE・重み付きMAE（km/h）
      'cv'          : 各foldのλ・検証MAEのDataFrame（論文Fig.16に相当）
      'train','test': 分割後のデータ
    """
    cols = ['rho1', 'rho2', 'p1_star', 'p2_star', 's', 'u1_obs', 'u2_obs']
    data = df_coop[df_coop['coop_exists']].dropna(subset=cols)
    print(f"λの推定に使うスナップショット(2-pipe かつ s>0): {len(data):,}件  (探索範囲: {bounds})")
    if len(data) < k:
        raise ValueError(f"データが{len(data)}件しかなく、{k}-fold交差検証ができません")

    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    # k-fold交差検証（学習データをk分割し、k-1個で推定→残り1個で評価をk回）
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        lam_f = fit_surplus_split(tr, car_params, truck_params, w1, w2, bounds=bounds)
        mae_f = surplus_split_mae(lam_f, va, car_params, truck_params, w1, w2)
        cv_rows.append({'fold': fold, 'lambda': lam_f, **mae_f})
    cv = pd.DataFrame(cv_rows)

    # 学習データ全体でλを推定し、テストデータで評価
    lam_hat = fit_surplus_split(train, car_params, truck_params, w1, w2, bounds=bounds)
    test_mae = surplus_split_mae(lam_hat, test, car_params, truck_params, w1, w2)

    # 探索範囲の端に張り付いていたら警告（真の最小点が範囲の外にある可能性）
    if min(abs(lam_hat - bounds[0]), abs(lam_hat - bounds[1])) < 1e-3:
        print(f"  ※警告: λ_hat={lam_hat:.4f} が探索範囲{bounds}の端にあります（境界解）。"
              f"plot_surplus_split_loss で範囲外の損失の形を確認してください")

    print(f"  学習 {len(train):,}件 / テスト {len(test):,}件")
    print(f"  {k}-fold CV: λ = {cv['lambda'].mean():.4f} ± {cv['lambda'].std():.4f}, "
          f"重み付きMAE = {cv['mae_weighted'].mean():.3f} ± {cv['mae_weighted'].std():.3f} km/h")
    print(f"  λ_hat = {lam_hat:.4f}  (論文: 0.8067)")
    print(f"  テストMAE: 車 {test_mae['mae1']:.3f}, トラック {test_mae['mae2']:.3f}, "
          f"重み付き {test_mae['mae_weighted']:.3f} km/h")
    print(f"           = 車 {test_mae['mae1'] / KM_TO_MILE:.3f}, トラック {test_mae['mae2'] / KM_TO_MILE:.3f}, "
          f"重み付き {test_mae['mae_weighted'] / KM_TO_MILE:.3f} mph  (論文: 2.32, 1.45, 1.89 mph)")

    return {'lambda_hat': lam_hat, 'test_mae': test_mae, 'cv': cv, 'train': train, 'test': test}

# 論文Eq(15)(16)：台数比で正規化した分割係数と公平性指標
#   Pi = Ni*PCEi / Σ Nj*PCEj,  λ̃1 = λ̂/P1,  λ̃2 = (1-λ̂)/P2,  δ = |λ̃1 - λ̃2|
def compute_equity_metric(lam_hat, n_car, n_truck, pce_car=1.0, pce_truck=1.5):
    w_car, w_truck = n_car * pce_car, n_truck * pce_truck
    P1 = w_car / (w_car + w_truck)
    P2 = w_truck / (w_car + w_truck)
    lam1 = lam_hat / P1
    lam2 = (1.0 - lam_hat) / P2
    delta = abs(lam1 - lam2)
    print(f"P1 = {P1:.4f}, P2 = {P2:.4f}")
    print(f"λ~1 = {lam1:.3f}, λ~2 = {lam2:.3f}, δ = {delta:.3f}  (論文: 0.84, 4.82, 3.98)")
    return {'P1': P1, 'P2': P2, 'lambda_tilde1': lam1, 'lambda_tilde2': lam2, 'delta': delta}


def _resolve_eps(eps_kmh):
    """eps_kmhを (車のε, トラックのε) の組に変換する。
    数値 → 両クラス共通 / (eps1, eps2) → クラス別 / estimate_eps_from_seの戻り値(dict) → その'eps1','eps2'"""
    if isinstance(eps_kmh, dict):
        return eps_kmh['eps1'], eps_kmh['eps2']
    if isinstance(eps_kmh, (tuple, list)):
        if len(eps_kmh) != 2:
            raise ValueError("eps_kmh をタプルで渡す場合は (車のε, トラックのε) の2要素にしてください")
        return eps_kmh[0], eps_kmh[1]
    return eps_kmh, eps_kmh


def classify_traffic_regime(df_s, eps_kmh, u1_col='u1_obs', u2_col='u2_obs', u_star_col='u_1pipe', s_col='s'):
    """
    論文Eq(11)で交通状態を判定する。
      「=」: |ũi - u*| <= eps_i,  「≥」: ũi - u* >= -eps_i
      2-pipe: ≥1 かつ ≥2 かつ not(=1 かつ =2) / 1-pipe: =1 かつ =2 / それ以外: non-NE

    df_s: u1_obs, u2_obs, u_1pipe 列を含むスナップショット表
    eps_kmh: 観測速度と1-pipe速度が「等しい」とみなす許容幅(km/h)。次のどれでも渡せる
        - 数値                 : 車・トラック共通の固定値          例) 5
        - (eps1, eps2)         : 車・トラック別の固定値            例) (1.8, 6.3)
        - estimate_eps_from_seの戻り値 : 標準誤差から決めた車・トラック別のε

    戻り値: df_s に以下の列を追加したコピー
      'du1', 'du2' : ũi - u*  (km/h)
      'regime'     : '2-pipe' / '1-pipe' / 'non-NE' / 'undefined'
                     'undefined' = 片方のクラスがいない、またはu*が解けなかったスナップショット
    """
    eps1, eps2 = _resolve_eps(eps_kmh)
    df = df_s.copy()

    #観測速度とu*の差を計算する
    df['du1'] = df[u1_col] - df[u_star_col]
    df['du2'] = df[u2_col] - df[u_star_col]

    #両クラスの観測速度とu*が存在する行だけ判定対象にする
    valid = df[['du1', 'du2']].notna().all(axis=1)

    #判定に使う条件（車はeps1、トラックはeps2）
    eq1 = df['du1'].abs() <= eps1   #ũ1 = u* とみなせる
    eq2 = df['du2'].abs() <= eps2   #ũ2 = u* とみなせる
    ge1 = df['du1'] >= -eps1        #ũ1 >= u* (許容幅込み)
    ge2 = df['du2'] >= -eps2        #ũ2 >= u* (許容幅込み)

    is_1pipe = valid & eq1 & eq2
    is_2pipe = valid & ge1 & ge2 & ~(eq1 & eq2)   #両方"="の場合は1-pipeなので除く

    df['regime'] = 'non-NE'
    df.loc[is_2pipe, 'regime'] = '2-pipe'
    df.loc[is_1pipe, 'regime'] = '1-pipe'
    df.loc[~valid, 'regime'] = 'undefined'

    #集計の表示  論文Eq(21): Pcoop = N_2pipe / N_tot
    counts = df['regime'].value_counts()
    n_tot = int(valid.sum())
    n_2pipe = int(counts.get('2-pipe', 0))
    print(f"交通状態を判定(eps: 車 {eps1:.2f}, トラック {eps2:.2f} km/h): 判定対象 {n_tot:,}件 "
          f"(対象外 undefined: {int(counts.get('undefined', 0)):,}件)")
    for r in ['2-pipe', '1-pipe', 'non-NE']:
        n = int(counts.get(r, 0))
        pct = 100 * n / n_tot if n_tot > 0 else float('nan')
        print(f"  {r:>10}: {n:,}件 ({pct:.2f}%)")
    if n_tot > 0:
        print(f"  Pcoop = {100 * n_2pipe / n_tot:.2f}% (論文: 13.83%)")

    #regime別の協力余剰sの平均（論文Fig.13の水平線: 2-pipe 0.15, 1-pipe 0.11）
    if s_col in df.columns:
        mean_s_2pipe = df.loc[df['regime'] == '2-pipe', s_col].mean()
        mean_s_1pipe = df.loc[df['regime'] == '1-pipe', s_col].mean()
        print(f"  平均s: 2-pipe = {mean_s_2pipe:.4f}, 1-pipe = {mean_s_1pipe:.4f} "
              f"(論文: 0.15, 0.11)")
    return df

#################################################################################
#パイプラインを頭から実行する関数
#  prepare_data : Step 0(前処理)とStep 2(マクロデータ)。パラメータに依存しないので1回だけ実行
#  run_pipeline : Step 3〜6。パラメータ・scaling・εを変えて何度でも実行できる
#################################################################################
def prepare_data(lanes=(2, 3, 4), classes=(2, 3), snapshot_step_frames=5):
    """
    Step 0（データ読み込み・前処理）とStep 2（マクロデータ）を実行する。
    どちらも名目速度関数のパラメータに依存しないので、1回だけ実行して使い回す。

    戻り値: dict
      'df'       : 元データ（Step 2のマクロデータの元）
      'df_final' : フィルタ後の追従データ（Step 1の推定・Eq.15の台数P_iに使う）
      'df_macro' : compute_macroscopic_data の返り値（先頭・末尾の除外前）
    """
    #step 0
    df = load_trajectory_data()  # データの読み込み
    df_rem_noise = remove_flicker_noise(df, col='Preceding', threshold=10)  # フリッカリングノイズを除去
    df_clean = remove_flicker_noise(df_rem_noise, col='Lane_ID', threshold=10)
    df_clean = recompute_headway_after_flicker(df_clean)  # 車頭距離を再計算
    df_with_pairs = give_pair_id(df_clean)  # ペアID付与
    df_with_mode = give_mode_id(df_with_pairs, df)  # モードID付与
    df_valid = filter_valid_pairs(df_with_mode)  # 追従していない行を除外
    df_lane = filter_lane(df_valid)  # 車線フィルタ（Lanes 2, 3, 4のみ）
    df_mode = filter_mode(df_lane)  # モードフィルタ（車種を限定）
    df_duration = filter_duration(df_mode)  # 追従時間フィルタ
    df_cutoff = apply_cutoff(df_duration)  # カットオフウィンドウ
    df_final = filter_acceleration(df_cutoff).copy()  # 加速度フィルタ

    #step 2（元データdfのレーン2〜4の全車両を使う）
    df_macro = compute_macroscopic_data(df, lanes=lanes, classes=classes,
                                        snapshot_step_frames=snapshot_step_frames)

    return {'df': df, 'df_final': df_final, 'df_macro': df_macro}


def _scaling_value(x):
    """a2, b1 を数値でも fit_scaling_parameter_* の戻り値(dict)でも受け取れるようにする"""
    return x['scaling_param'] if isinstance(x, dict) else float(x)


def run_pipeline(data, car_params, truck_params, a2, b1, scaling, eps_kmh,
                 skip_sec_warmup=100, skip_sec_cooldown=200,
                 n_car=None, n_truck=None, pce_truck=1.5,
                 w1=0.5, w2=0.5, test_size=0.3, k=10, random_state=42, bounds=(0.0, 1.0)):
    """
    Step 3〜6 を同じパラメータで一括実行する。

    data: prepare_data の返り値
    car_params: 車(C-C)の名目速度関数パラメータ（ub, uf, rhoc, theta1, theta2 を含む辞書）
    truck_params: トラック(T-T)の名目速度関数パラメータ（uf, rhoc を含む辞書）
    a2, b1: スケーリングパラメータ。数値でも fit_scaling_parameter_* の戻り値(dict)でもよい
    scaling: 'printed'（論文の記載どおり ρj/aj）/ 'inverted'（ρj·aj）
    eps_kmh: 交通状態判定の許容幅(km/h)。数値（共通）か (車のε, トラックのε)
    skip_sec_warmup, skip_sec_cooldown: 各time_periodの最初・最後に除外する秒数
    n_car, n_truck: Eq.15の台数。Noneなら df_final の車両数（論文と同じ数え方）
    w1, w2, test_size, k, random_state, bounds: λの推定に関する設定（estimate_surplus_split と同じ）

    戻り値: dict
      'df_s'     : Step 3の結果（u*, p1*, p2*, s）
      'df_regime': Step 4の結果（regime列）
      'df_coop'  : Step 5の結果（coop_exists列）
      'lambda'   : Step 6の結果（estimate_surplus_split の返り値。推定できなければNone）
      'equity'   : Eq.15/16の結果（推定できなければNone）
      'summary'  : 主な数値をまとめた辞書（設定の比較用）
    """
    a2 = _scaling_value(a2)
    b1 = _scaling_value(b1)
    print(f"===== scaling={scaling}, a2={a2:.4f}, b1={b1:.4f}, eps={eps_kmh}, "
          f"除外: 最初{skip_sec_warmup}秒・最後{skip_sec_cooldown}秒 =====")

    #step 3: 先頭・末尾を除外してから u* → p* → s
    df_macro = drop_warmup(data['df_macro'], skip_sec=skip_sec_warmup)
    df_macro = drop_cooldown(df_macro, skip_sec=skip_sec_cooldown)
    df_1pipe = compute_1pipe_speed_for_snapshots(df_macro, car_params, truck_params,
                                                 a2, b1, scaling)
    df_pstar = compute_min_road_share(df_1pipe, car_params, truck_params)  # u*と同じパラメータを使う
    df_s = compute_cooperation_surplus(df_pstar)

    #step 4
    df_regime = classify_traffic_regime(df_s, eps_kmh=eps_kmh)

    #step 5
    df_coop = check_cooperation_existence(df_regime)

    #step 6: 協力性ありの時刻がk件未満だとk-fold交差検証ができないので飛ばす
    res, equity = None, None
    n_coop = int(df_coop['coop_exists'].sum())
    if n_coop >= k:
        res = estimate_surplus_split(df_coop, car_params, truck_params, w1=w1, w2=w2,
                                     test_size=test_size, k=k, random_state=random_state,
                                     bounds=bounds)
        if n_car is None or n_truck is None:
            veh = data['df_final'][['time_period', 'Vehicle_ID', 'v_Class']].drop_duplicates()
            n_car = int((veh['v_Class'] == 2).sum())
            n_truck = int((veh['v_Class'] == 3).sum())
        print(f"Eq.15の台数: 車 {n_car}台, トラック {n_truck}台  (論文: 1401台, 39台)")
        equity = compute_equity_metric(res['lambda_hat'], n_car=n_car, n_truck=n_truck,
                                       pce_truck=pce_truck)
    else:
        print(f"協力性ありのスナップショットが{n_coop}件（{k}件未満）のため、λは推定しません")

    #設定どうしを比べやすいように主な数値をまとめる
    judged = df_regime[~df_regime['regime'].isin(['undefined', 'out-of-domain'])]
    n_tot = len(judged)
    summary = {
        'scaling': scaling, 'a2': a2, 'b1': b1, 'eps_kmh': eps_kmh,
        'n_judged': n_tot,
        'Pcoop_%': 100 * (judged['regime'] == '2-pipe').mean() if n_tot else np.nan,
        'share_1pipe_%': 100 * (judged['regime'] == '1-pipe').mean() if n_tot else np.nan,
        'mean_s_2pipe': judged.loc[judged['regime'] == '2-pipe', 's'].mean(),
        'mean_s_1pipe': judged.loc[judged['regime'] == '1-pipe', 's'].mean(),
        'n_coop': n_coop,
        'lambda_hat': res['lambda_hat'] if res else np.nan,
        'mae_weighted_kmh': res['test_mae']['mae_weighted'] if res else np.nan,
        'zeta': equity['delta'] if equity else np.nan,
    }

    return {'df_s': df_s, 'df_regime': df_regime, 'df_coop': df_coop,
            'lambda': res, 'equity': equity, 'summary': summary}
##############################################################################
#可視化用　プロット用関数
##############################################################################
#ミクロ速度密度散布図と推定モデルの関数プロット
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


def _bin_and_aggregate_1pipe(df, rho1_col, rho2_col, bin_width1, bin_width2,
                               max_rho1, max_rho2, min_count, value_col='u_1pipe',
                               min_rho1=0.6, min_rho2=0.6):
    """rho1_col, rho2_colの密度をビンに区切り、各ビンのvalue_colの平均をまとめる共通処理。"""
    bins1 = np.arange(min_rho1, max_rho1 + bin_width1, bin_width1)
    bins2 = np.arange(min_rho2, max_rho2 + bin_width2, bin_width2)

    df = df.copy()
    df['_bin1'] = pd.cut(df[rho1_col], bins=bins1, include_lowest=True)
    df['_bin2'] = pd.cut(df[rho2_col], bins=bins2, include_lowest=True)

    cats1 = df['_bin1'].cat.categories
    cats2 = df['_bin2'].cat.categories

    grid = df.groupby(['_bin2', '_bin1'], observed=True).agg(
        mean_val=(value_col, 'mean'), count=(value_col, 'size')
    ).reset_index()

    heat = grid.pivot(index='_bin2', columns='_bin1', values='mean_val').reindex(index=cats2, columns=cats1)
    counts = grid.pivot(index='_bin2', columns='_bin1', values='count').reindex(index=cats2, columns=cats1)
    heat = heat.where(counts >= min_count)

    return bins1, bins2, heat


def plot_1pipe_speed_heatmap(df_1pipe, bin_width1=1.0, bin_width2=1.0,
                               max_rho1=None, max_rho2=None, min_count=3,
                               min_rho1=0.6, min_rho2=0.6):
    """
    1-pipe速度のヒートマップを、km単位版とmile単位版(論文の単位)を横並びで表示する。

    min_rho1, min_rho2: 密度の下限(km単位)。論文Fig.11がρ=1未満を表示していないのに
    合わせてデフォルトを1にしている。mile版はmin_rho1*KM_TO_MILEとして換算される
    (他のbin_width1・max_rho1と同じ扱い)。
    """
    df = df_1pipe.dropna(subset=['u_1pipe']).copy()

    if max_rho1 is None:
        max_rho1 = df['rho1'].max()
    if max_rho2 is None:
        max_rho2 = df['rho2'].max()

    # --- km版 ---
    bins1_km, bins2_km, heat_kmh = _bin_and_aggregate_1pipe(
        df, 'rho1', 'rho2', bin_width1, bin_width2, max_rho1, max_rho2, min_count,
        min_rho1=min_rho1, min_rho2=min_rho2
    )

    # --- mile版：rho1, rho2をmileに変換してから、mile換算のビン幅・下限で独立に再分割 ---
    df['rho1_mile'] = df['rho1'] * KM_TO_MILE
    df['rho2_mile'] = df['rho2'] * KM_TO_MILE
    bin_width1_mile = bin_width1 * KM_TO_MILE
    bin_width2_mile = bin_width2 * KM_TO_MILE
    max_rho1_mile = max_rho1 * KM_TO_MILE
    max_rho2_mile = max_rho2 * KM_TO_MILE
    min_rho1_mile = min_rho1 * KM_TO_MILE
    min_rho2_mile = min_rho2 * KM_TO_MILE

    bins1_mile, bins2_mile, heat_kmh_in_mile_bins = _bin_and_aggregate_1pipe(
        df, 'rho1_mile', 'rho2_mile', bin_width1_mile, bin_width2_mile,
        max_rho1_mile, max_rho2_mile, min_count,
        min_rho1=min_rho1_mile, min_rho2=min_rho2_mile
    )
    heat_mph = heat_kmh_in_mile_bins / KM_TO_MILE

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



def plot_surplus_by_regime(df_regime, s_col='s', regime_col='regime',
                           colors=None, show_mean=True, skip_sec=100,
                           ylim=(-0.1, 0.1), title=None):
    """
    論文Fig.13風：協力余剰sの時系列を、交通状態(2-pipe / 1-pipe / non-Pareto)で色分けして表示する。
    time_periodごとに1段ずつ並べ、横軸は各time_periodの開始時刻からの経過秒数。

    df_regime: classify_traffic_regime の返り値（time_period, Global_Time, s, regime列を含む）
    colors: regimeごとの色の辞書（省略時は 2-pipe=赤, 1-pipe=青, non-Pareto=灰）
    show_mean: Trueなら 2-pipe / 1-pipe それぞれの平均sを水平の破線で表示（全time_periodでの平均）
    skip_sec: 各time_periodの開始からこの秒数までは除外する（区間に車両が入りきる前の立ち上がり部分）
              平均sの計算からも除外する。0にすると除外しない
    ylim: y軸の表示範囲 (min, max)。Noneなら自動
    """
    if colors is None:
        colors = {'2-pipe': 'tab:red', '1-pipe': 'tab:blue', 'non-NE': 'tab:gray'}

    df_sorted = df_regime.sort_values(['time_period', 'Global_Time']).copy()

    #各time_periodの開始時刻からの経過秒数を計算し、立ち上がり部分を除外
    t0 = df_sorted.groupby('time_period')['Global_Time'].transform('min')
    df_sorted['elapsed_sec'] = (df_sorted['Global_Time'] - t0) / 1000
    n_before = len(df_sorted)
    df_sorted = df_sorted[df_sorted['elapsed_sec'] >= skip_sec]
    print(f"各time_periodの最初の{skip_sec}秒を除外: {n_before:,}件 → {len(df_sorted):,}件")

    periods = df_sorted['time_period'].unique()

    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 3.5 * len(periods)),
                             sharey=True, squeeze=False)
    axes = axes[:, 0]

    #全time_periodを通したregime別の平均s（除外後のデータで計算。論文: 2-pipe 0.15, 1-pipe 0.11）
    means = {r: df_sorted.loc[df_sorted[regime_col] == r, s_col].mean() for r in ['2-pipe', '1-pipe']}

    for ax, tp in zip(axes, periods):
        g = df_sorted[df_sorted['time_period'] == tp]
        t = g['elapsed_sec'].to_numpy()
        s = g[s_col].to_numpy()
        reg = g[regime_col].to_numpy()

        #隣り合う2点を結ぶ線分を作り、左端の点のregimeで色を付ける
        pts = np.column_stack([t, s])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_reg = reg[:-1]
        #sがNaNの線分と、undefined(色の辞書にないregime)の線分は描かない → 途切れて見える
        keep = np.isfinite(segs).all(axis=(1, 2)) & np.isin(seg_reg, list(colors))
        lc = LineCollection(segs[keep], colors=[colors[r] for r in seg_reg[keep]], linewidths=1)
        ax.add_collection(lc)
        ax.autoscale_view()

        if show_mean:
            for r, m in means.items():
                if np.isfinite(m):
                    ax.axhline(m, color=colors[r], linestyle='--', linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.6)

        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(f'time_period {tp}')
        ax.set_ylabel('Cooperation surplus s')
        ax.grid(alpha=0.3)

    #凡例（線の色 = regime、破線 = 平均s）
    handles = [Line2D([0], [0], color=c, linewidth=2, label=r) for r, c in colors.items()]
    if show_mean:
        handles += [Line2D([0], [0], color=colors[r], linestyle='--',
                           label=f'mean s ({r}) = {m:.3f}')
                    for r, m in means.items() if np.isfinite(m)]
    axes[0].legend(handles=handles, loc='upper right', ncol=2, fontsize=9)

    axes[-1].set_xlabel('Elapsed time (s, from each time_period start)')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    print("regime別の平均s: " + ", ".join(f"{r}={m:.3f}" for r, m in means.items())
          + "  (論文: 2-pipe=0.15, 1-pipe=0.11)")


# 論文Fig.17風：λを0〜1で動かしたときの損失関数の形
def plot_surplus_split_loss(df, car_params, truck_params, lam_hat=None, w1=0.5, w2=0.5, n_grid=201):
    lams = np.linspace(-1.0, 2.0, n_grid)
    losses = [surplus_split_loss(l, df, car_params, truck_params, w1, w2) for l in lams]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, losses, color='tab:blue')
    if lam_hat is not None:
        ax.scatter([lam_hat], [surplus_split_loss(lam_hat, df, car_params, truck_params, w1, w2)],
                   color='red', zorder=3, label=f'λ̂ = {lam_hat:.4f}')
        ax.legend()
    ax.set_xlabel('Surplus split factor λ')
    ax.set_ylabel('Loss L(λ)  ((km/h)$^2$)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

