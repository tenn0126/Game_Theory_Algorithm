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
    df['_block_id'] = is_new_block.cumsum()

    blocks = df.groupby('_block_id').agg(
        Vehicle_ID=('Vehicle_ID', 'first'),
        time_period=('time_period', 'first'),
        value=(col, 'first'),
        frame_count=(col, 'size'),
    )

    # 前後ブロックは同一車両・時間帯の中でのみ参照する
    blocks['prev_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(1)
    blocks['next_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(-1)

    # フリッカリング判定：短いブロックの前後の値が完全一致する場合のみ
    is_flicker = (
        (blocks['frame_count'] <= threshold)
        & (blocks['prev_value'] == blocks['next_value'])
        & blocks['prev_value'].notna()
    )

    replacement = blocks['prev_value'].where(is_flicker)
    df[col] = df['_block_id'].map(replacement).fillna(df[col])
    df = df.drop(columns='_block_id')

    print("フリッカリング除去完了")
    print(f"補正されたブロック数: {is_flicker.sum():,}")
    print(f"行数: {len(df):,}")

    return df

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



####################################################################################################
######################################################################################################
#ペアID付与がなくてもできる関数
######################################################################################################
######################################################################################################
def give_mode_id_rows(df, original_df):
    """
    順序非依存版：pair_idを介さず、egoとleaderの車種からmode_idを付与する。
    
    モードID（ego_class, leader_class）：
    (c-c)=1, (c-t)=2, (t-c)=3
    (t-t)=4, (others)=5,
    """
    mode_map = {
        (2,2): 1, (2,3): 2, (3,2): 3,
        (3,3): 4, (2,1): 5, (3,1): 5,
        (1,1): 5, (1,2): 5, (1,3): 5
    }

    # 元データから前方車両(Preceding)の車種を取得
    leader_class = original_df[['Vehicle_ID', 'v_Class', 'time_period']].drop_duplicates()
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

    print(f"[give_mode_id_rows] mode_id付与完了: {len(df):,}行")
    print(df['mode_id'].value_counts(dropna=False).sort_index())

    return df

def filter_valid_rows(df):
    """
    順序非依存版：追従関係がない行(Preceding=0)や、
    leader_classが解決できずmode_idがNaNの行を除外する。
    """
    mask = (df['Preceding'] != 0) & df['mode_id'].notna()
    df_filtered = df[mask]

    print(f"[filter_valid_rows] フィルタ前: {len(df):,}行 → フィルタ後: {len(df_filtered):,}行 (除外 {len(df)-len(df_filtered):,}行)")

    return df_filtered

def filter_lane_rows(df, lanes=[2, 3, 4]):
    """
    順序非依存版：Lane_IDが対象車線に含まれない行を削除する。
    pair_idの存在を前提としない。1行単位で判定できる条件。
    """
    mask = df['Lane_ID'].isin(lanes)
    df_filtered = df[mask]

    print(f"[filter_lane_rows] フィルタ前: {len(df):,}行 → フィルタ後: {len(df_filtered):,}行 (除外 {len(df)-len(df_filtered):,}行)")

    return df_filtered


def filter_mode_rows(df, modes=[1, 2, 3, 4]):
    """
    順序非依存版：mode_idが対象カテゴリに含まれない行を削除する。
    mode_idはego_class(自車の車種)とleader_class(Precedingの車種)
    という1行単位の情報から決まるため、pair_idを介さず直接判定できる。
    ただし give_mode_id 実行後（mode_id列が存在する状態）である必要はある。
    """
    mask = df['mode_id'].isin(modes)
    df_filtered = df[mask]

    print(f"[filter_mode_rows] フィルタ前: {len(df):,}行 → フィルタ後: {len(df_filtered):,}行 (除外 {len(df)-len(df_filtered):,}行)")

    return df_filtered

def filter_acceleration_rows(df, max_acc=1.0):
    """
    順序非依存版：加速度フィルタ
    max_acc: 最大加速度・減速度（m/s²）（デフォルト: 1.0 m/s²）
    v_Accは1行単位で分かる値なので、pair_idを介さず直接判定できる。
    """
    max_acc_feet = max_acc / 0.3048

    mask = df['v_Acc'].abs() <= max_acc_feet
    df_filtered = df[mask]

    print(f"[filter_acceleration_rows] 閾値: {max_acc} m/s² = {max_acc_feet:.4f} feet/s²")
    print(f"フィルタ前: {len(df):,}行 → フィルタ後: {len(df_filtered):,}行 (除外 {len(df)-len(df_filtered):,}行)")

    return df_filtered

def make_temp_pair_id(df, max_gap_frames=1, frame_interval_ms=100):
    """
    その場限りの使い捨てペアID(temp_pair_id)を計算する。
    順序非依存フィルタを何段か重ねた後の"残っている行"に対して、
    都度これを呼び出して「連続した追従区間」を再定義する想定。

    区切り条件（いずれか1つでも満たせば新しいペアの開始）：
    - Vehicle_IDが変わった
    - time_periodが変わった
    - Lane_IDが変わった（自車の車線変更）
    - Precedingが変わった（先行車の交代・消失）
    - 直前の行とのGlobal_Timeの差が許容範囲を超える（=欠損フレームが多すぎる）

    max_gap_frames: 連続して許容する欠損フレーム数
                    (例: 1なら、1フレーム抜けている程度(=200ms間隔)までは許容し、
                     2フレーム以上抜けている(=300ms以上)場合は区切る)
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    vehicle_changed     = df['Vehicle_ID']  != df['Vehicle_ID'].shift()
    time_period_changed = df['time_period'] != df['time_period'].shift()
    lane_changed         = df['Lane_ID']     != df['Lane_ID'].shift()
    preceding_changed    = df['Preceding']   != df['Preceding'].shift()

    # 直前の行との時間差(ms)。通常は frame_interval_ms(=100ms)ずつ進むはず
    time_gap = df['Global_Time'] - df['Global_Time'].shift()
    max_allowed_gap_ms = (max_gap_frames + 1) * frame_interval_ms
    gap_too_large = (time_gap > max_allowed_gap_ms) | time_gap.isna()

    break_point = vehicle_changed | time_period_changed | lane_changed | preceding_changed | gap_too_large

    df['temp_pair_id'] = break_point.cumsum()

    # Preceding=0の行はペアとして扱わない
    preceding_zero = df['Preceding'] == 0
    df.loc[preceding_zero, 'temp_pair_id'] = None

    return df

def filter_duration_rows(df, min_duration=60, max_gap_frames=1, frame_interval_ms=100):
    """
    順序非依存版：追従時間フィルタ
    その場でtemp_pair_idを計算し、min_duration秒未満のペアの行を削除する。

    min_duration: 最小追従時間（秒）
    max_gap_frames, frame_interval_ms: make_temp_pair_idにそのまま渡すパラメータ
    """
    df_paired = make_temp_pair_id(df, max_gap_frames=max_gap_frames, frame_interval_ms=frame_interval_ms)

    valid_rows = df_paired.dropna(subset=['temp_pair_id'])
    pair_duration = valid_rows.groupby('temp_pair_id')['Global_Time'].agg(
        # (max-min)だとNフレームの継続がN-1フレーム分としてカウントされるので、1フレーム分補正
        duration=lambda x: (x.max() - x.min()) / 1000 + frame_interval_ms / 1000
    )

    valid_pair_ids = pair_duration[pair_duration['duration'] >= min_duration].index
    df_filtered = df_paired[df_paired['temp_pair_id'].isin(valid_pair_ids)].drop(columns='temp_pair_id')

    print(f"[filter_duration_rows] フィルタ前の行数: {len(df):,} → フィルタ後: {len(df_filtered):,} (除外 {len(df)-len(df_filtered):,}行)")
    print(f"使い捨てペア数: {df_paired['temp_pair_id'].nunique():,} → {min_duration}秒以上のペア数: {len(valid_pair_ids):,}")

    return df_filtered



def apply_cutoff_rows(df, cutoff=10, max_gap_frames=1, frame_interval_ms=100):
    """
    順序非依存版：前後カットオフウィンドウ
    その場でtemp_pair_idを計算し、各ペアの最初と最後のcutoff秒を除外する。

    cutoff: カットオフ時間（秒）
    max_gap_frames, frame_interval_ms: make_temp_pair_idにそのまま渡すパラメータ
    """
    cutoff_ms = cutoff * 1000 # 秒→ミリ秒

    df_paired = make_temp_pair_id(df, max_gap_frames=max_gap_frames, frame_interval_ms=frame_interval_ms)
    valid_rows = df_paired.dropna(subset=['temp_pair_id'])

    pair_times = valid_rows.groupby('temp_pair_id')['Global_Time'].agg(
        min_time='min', max_time='max'
    ).reset_index()

    df_merged = valid_rows.merge(pair_times, on='temp_pair_id')
    mask = (
        (df_merged['Global_Time'] >= df_merged['min_time'] + cutoff_ms) &
        (df_merged['Global_Time'] <= df_merged['max_time'] - cutoff_ms)
    )
    df_filtered = df_merged[mask].drop(columns=['min_time', 'max_time', 'temp_pair_id'])

    print(f"[apply_cutoff_rows] カットオフ前の行数: {len(df):,} → カットオフ後: {len(df_filtered):,} (除外 {len(df)-len(df_filtered):,}行)")
    print(f"使い捨てペア数: {df_paired['temp_pair_id'].nunique():,} → カットオフ後も行が残るペア数: {df_merged.loc[mask, 'temp_pair_id'].nunique():,}")

    return df_filtered


def assign_pair_type(df):
    """
    順序非依存版：ego_class/leader_classからpair_type(C-C, C-Tなど)を付与する。
    1行単位で決まる値なので、pair_idを介さず直接計算できる。
    """
    df_out = df.copy()
    class_map = {2: 'C', 3: 'T', 1: 'M'}

    ego_str = df_out['ego_class'].map(class_map).fillna('?')
    leader_str = df_out['leader_class'].map(class_map).fillna('?')
    df_out['pair_type'] = ego_str + '-' + leader_str

    return df_out


def count_pair_type(df, max_gap_frames=1, frame_interval_ms=100):
    """
    pair_type別のペア数（行数ではなくユニークな追従区間の数）を集計する。
    その場でtemp_pair_idを計算して数える。
    事前に assign_pair_type(df) で pair_type 列が付与されている必要がある。
    """
    df_paired = make_temp_pair_id(df, max_gap_frames=max_gap_frames, frame_interval_ms=frame_interval_ms)
    valid = df_paired.dropna(subset=['temp_pair_id'])

    order = ['C-C', 'C-T', 'C-M', 'T-T', 'T-C', 'T-M', 'M-M', 'M-C', 'M-T']
    pair_types = valid.groupby('temp_pair_id')['pair_type'].first()
    summary = pair_types.value_counts().reindex(order, fill_value=0)

    print("追従ペア数:")
    print(summary)

    return summary


######################################################################################################
######################################################################################################
#ペアID付与が前提の関数
######################################################################################################
######################################################################################################

def give_mode_id_with_pair_id(df, original_df):
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



def filter_lane_with_pair_id(df, lanes=[2, 3, 4]):
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

def filter_mode_with_pair_id(df, modes=[1,2,3,4]):
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


def filter_duration_with_pair_id(df, min_duration=60):
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

def filter_acceleration_with_pair_id(df, max_acc=1.0):
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



def apply_cutoff_with_pair_id(df, cutoff=10):
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


def assign_and_count_pair_type_with_pair_id(df):
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
