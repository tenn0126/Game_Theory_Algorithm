import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

#追従ペアIDを付与する関数
def give_pair_id(df):
    """
    追従ペアIDを付与する関数
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    # 追従が途切れるタイミングを検出（直前の行と比較して変化したら旗を立てる）
    vehicle_changed      = df['Vehicle_ID']  != df['Vehicle_ID'].shift()
    time_period_changed  = df['time_period'] != df['time_period'].shift()
    lane_changed         = df['Lane_ID']     != df['Lane_ID'].shift()
    preceding_changed    = df['Preceding']   != df['Preceding'].shift()

    # どれか1つでもTrueなら追従の切れ目
    break_point = vehicle_changed | time_period_changed | lane_changed | preceding_changed

    # ペアIDを付与
    df['pair_id'] = break_point.cumsum()

    # Preceding=0の行はpair_idをNaNにする（追従関係がないため）
    preceding_zero = df['Preceding'] == 0
    df.loc[preceding_zero, 'pair_id'] = None

    return df

#追従ペアIDを付与する関数
def give_mode_id(df, original_df):
    """
    追従ペアにモードIDを付与する関数
    
    モードID（ego_class, leader_class）：
    (c-c)=1, (c-t)=2, (t-c)=3
    (t-t)=4, (others)=5,
    """
    mode_map = {
        (2,2): 1, (2,3): 2, (3,2): 3,
        (3,3): 4, (2,1): 5, (3,1): 5,
        (1,1): 5, (1,2): 5, (1,3): 5
    }
    
    # 元データから前方車両の車種を取得
    leader_class = original_df[['Vehicle_ID', 'v_Class','time_period']].drop_duplicates()
    leader_class = leader_class.rename(columns={'Vehicle_ID': 'Preceding', 'v_Class': 'leader_class'})
    
    # 結合
    df = df.merge(leader_class, on=['Preceding', 'time_period'], how='left')
    df['ego_class'] = df['v_Class']
    
    # モードIDを付与（NaNの場合はNoneを返す）
    def get_mode_id(row):
        if pd.isna(row['leader_class']):
            return None
        return mode_map.get((int(row['ego_class']), int(row['leader_class'])))
    
    df['mode_id'] = df.apply(get_mode_id, axis=1)
    
    # 確認
    pairs_count = df.dropna(subset=['pair_id']).groupby(['pair_id', 'mode_id']).size().reset_index()
    summary = pairs_count.groupby('mode_id').size()
    print("モードID別ペア数:")
    print(summary)
    
    return df

def filter_valid_pairs(df):
    """
    pair_idとmode_idが付与されている行のみ残す関数
    （追従関係がない行を除外）
    """
    df_filtered = df.dropna(subset=['pair_id', 'mode_id'])
    
    print(f"フィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"\nペア数: {df_filtered['pair_id'].nunique():,}")
    
    return df_filtered

def filter_lane(df, lanes=[2, 3, 4]):
    """
    車線フィルタ
    lanes: 使用する車線のリスト（デフォルト: Lane 2,3,4）
    """
    df_filtered = df[df['Lane_ID'].isin(lanes)]
    
    print(f"フィルタ前: {len(df):,}行")
    print(f"フィルタ後: {len(df_filtered):,}行")
    print(f"除外された行数: {len(df) - len(df_filtered):,}行")
    print(f"\n車線の分布:")
    print(df_filtered['Lane_ID'].value_counts().sort_index())
    
    return df_filtered

def filter_mode(df, modes=[1,2,3,4]):
    """
    特定のモードIDのみを残すフィルタ（ペア単位で安全に処理）
    modes: 使用するモードIDのリスト
    デフォルト: [1,2,3,4]
    """
    
    # 1. 各ペアIDの代表モードIDを取得（最初の行の値を使う）
    pair_modes = df.dropna(subset=['pair_id']).groupby('pair_id')['mode_id'].first()
    
    # 2. 指定したモード(1,2,3,4)に該当する「ペアIDのリスト」を作成
    valid_pair_ids = pair_modes[pair_modes.isin(modes)].index
    
    # 3. 元のデータから、その有効なペアIDの行を丸ごと残す
    df_filtered = df[df['pair_id'].isin(valid_pair_ids)]
    
    
    print(f"フィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    print(f"除外されたペア数: {df['pair_id'].nunique() - df_filtered['pair_id'].nunique():,}")
    print(f"\nモードID別ペア数:")
    pairs_count = df_filtered.groupby(['pair_id', 'mode_id']).size().reset_index()
    print(pairs_count.groupby('mode_id').size())
    
    return df_filtered

def assign_and_count_pair_type(df):
    """
    車種の組み合わせ（ペアタイプ：C-C, C-Tなど）を付与し、集計結果を出力する関数
    """
    # 元のデータを直接書き換えないようにコピーを作成
    df_out = df.copy()

    # 1. 辞書の定義
    class_map = {2: 'C', 3: 'T', 1: 'M'}
    
    # 2. pair_type列の追加（★高速化ポイント）
    # map関数を使って、列全体を一気に文字（C, T, M）に変換して結合します
    ego_str = df_out['ego_class'].map(class_map).fillna('?')
    leader_str = df_out['leader_class'].map(class_map).fillna('?')
    df_out['pair_type'] = ego_str + '-' + leader_str

    # 3. 出力順番を指定
    order = ['C-C', 'C-T', 'C-M', 'T-T', 'T-C', 'T-M', 'M-M', 'M-C', 'M-T']

    # 4. ペア数の集計
    pairs_count = df_out.dropna(subset=['pair_id']).groupby(['pair_id', 'pair_type']).size().reset_index()
    pairs_summary = pairs_count.groupby('pair_type').size().reindex(order, fill_value=0)

    # 5. 結果の出力
    print("追従ペア数:")
    print(pairs_summary)

    return df_out

def filter_duration(df, min_duration=60):
    """
    追従時間フィルタ
    min_duration: 最小追従時間（秒）（デフォルト: 60秒）
    """
    # 各ペアの追従時間を計算
    pair_duration = df.groupby('pair_id')['Global_Time'].agg(
        duration=lambda x: (x.max() - x.min()) / 1000  # ミリ秒→秒
    )
    
    # min_duration以上のペアIDを取得
    valid_pair_ids = pair_duration[pair_duration['duration'] >= min_duration].index
    
    # 有効なペアIDの行だけ残す
    df_filtered = df[df['pair_id'].isin(valid_pair_ids)]
    
    print(f"フィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    print(f"除外されたペア数: {df['pair_id'].nunique() - df_filtered['pair_id'].nunique():,}")
    print(f"\nフィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    
    return df_filtered

def filter_acceleration(df, max_acc=1.0):
    """
    加速度フィルタ
    max_acc: 最大加速度・減速度（m/s²）（デフォルト: 1.0 m/s²）
    """
    # 閾値をfeet/s²に変換
    max_acc_feet = max_acc / 0.3048
    
    # 閾値を超える行を除外
    df_filtered = df[df['v_Acc'].abs() <= max_acc_feet]
    
    print(f"閾値: {max_acc} m/s² = {max_acc_feet:.4f} feet/s²")
    print(f"\nフィルタ前の行数: {len(df):,}")
    print(f"フィルタ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"\nフィルタ前のペア数: {df['pair_id'].nunique():,}")
    print(f"フィルタ後のペア数: {df_filtered['pair_id'].nunique():,}")
    
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
    
    # 元のデータに結合
    df_merged = df.merge(pair_times, on='pair_id')
    
    # カットオフ条件でフィルタリング
    df_filtered = df_merged[
        (df_merged['Global_Time'] >= df_merged['min_time'] + cutoff_ms) &
        (df_merged['Global_Time'] <= df_merged['max_time'] - cutoff_ms)
    ].drop(columns=['min_time', 'max_time'])
    
    print(f"カットオフ前の行数: {len(df):,}")
    print(f"カットオフ後の行数: {len(df_filtered):,}")
    print(f"除外された行数: {len(df) - len(df_filtered):,}")
    print(f"\nカットオフ前のペア数: {df['pair_id'].nunique():,}")
    print(f"カットオフ後のペア数: {df_filtered['pair_id'].nunique():,}")
    
    return df_filtered