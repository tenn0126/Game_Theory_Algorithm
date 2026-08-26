import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
    #block_idは不要なので消す   
    df = df.drop(columns='_block_id')

    print("\n--フリッカリング除去--")
    print(f"補正されたブロック数: {is_flicker.sum():,}")
    print(f"行数: {len(df):,}")

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




###################################################################
###################################################################
#ローパスフィルタ
###################################################################
###################################################################

def smooth_speed(df, window_length=11, polyorder=3, speed_col='v_Vel'):
    """
    車両(Vehicle_ID, time_period)ごとに独立してSavitzky-Golayフィルタをかけ、
    速度の高周波ノイズ(NGSIM特有の階段状の疑似定速パターン)を平滑化する。

    window_length: フィルタ窓の長さ(フレーム数, 奇数)。10Hzサンプリングなので
                    11なら約1.1秒, 21なら約2.1秒の窓になる
    polyorder: 窓内でフィットする多項式の次数(window_length未満である必要がある)
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    def _smooth(s):
        if len(s) < window_length:
            return s  # 窓より短い軌跡は平滑化しない
        return pd.Series(
            savgol_filter(s.values, window_length=window_length, polyorder=polyorder),
            index=s.index
        )

    df[f'{speed_col}_smooth'] = df.groupby(['Vehicle_ID', 'time_period'])[speed_col].transform(_smooth)

    return df

def recompute_acceleration(df, speed_col='v_Vel_smooth'):
    """平滑化した速度の差分から加速度を再計算する(ft/s^2)。"""
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()
    g = df.groupby(['Vehicle_ID', 'time_period'])
    dt = g['Global_Time'].diff() / 1000  # ms -> s
    dv = g[speed_col].diff()
    df['v_Acc_recalc'] = dv / dt
    return df

def smooth_acceleration(df, window_length=11, polyorder=3, acc_col='v_Acc'):
    """
    車両(Vehicle_ID, time_period)ごとに独立してSavitzky-Golayフィルタをかけ、
    加速度(v_Acc)の高周波ノイズ/スパイクを平滑化する。
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()

    def _smooth(s):
        if len(s) < window_length:
            return s
        return pd.Series(
            savgol_filter(s.values, window_length=window_length, polyorder=polyorder),
            index=s.index
        )

    df[f'{acc_col}_smooth'] = df.groupby(['Vehicle_ID', 'time_period'])[acc_col].transform(_smooth)

    return df


def moving_average_acceleration(df, window=11, acc_col='v_Acc'):
    """比較用：単純な窓内平均による加速度平滑化。"""
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()
    df[f'{acc_col}_ma'] = df.groupby(['Vehicle_ID', 'time_period'])[acc_col].transform(
        lambda s: s.rolling(window=window, center=True, min_periods=1).mean()
    )
    return df



###################################################################################
###################################################################################
#データの分析用関数　（実際のデータ処理には使わない）
###################################################################################
###################################################################################import pandas as pd

def analyze_pair_termination_reasons(df, pair_col='pair_id', duration_col=None, frame_interval=0.1):
    """
    各ペアの最終フレームの直後に何が起きて終了したかを分類する。
    - lane_changed: 車線変更
    - preceding_zero: 先行車を見失った(Preceding=0)
    - preceding_changed(real): 0を経由せず別の実車両IDに切り替わった
    - end_of_trajectory: 車両の記録自体がそこで終わっている
    """
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()
    grp = df.groupby(['Vehicle_ID', 'time_period'])
    df['next_Lane_ID'] = grp['Lane_ID'].shift(-1)
    df['next_Preceding'] = grp['Preceding'].shift(-1)

    pairs = df.dropna(subset=[pair_col]).groupby(pair_col).agg(
        duration_sec=(pair_col, 'size'),
        Lane_ID=('Lane_ID', 'last'),
        Preceding=('Preceding', 'last'),
        next_Lane_ID=('next_Lane_ID', 'last'),
        next_Preceding=('next_Preceding', 'last'),
    )
    pairs['duration_sec'] *= frame_interval

    def reason(row):
        if pd.isna(row['next_Preceding']):
            return 'end_of_trajectory'
        if row['next_Lane_ID'] != row['Lane_ID']:
            return 'lane_changed'
        if row['next_Preceding'] == 0:
            return 'preceding_zero'
        if row['next_Preceding'] != row['Preceding']:
            return 'preceding_changed(real)'
        return 'unknown'

    pairs['end_reason'] = pairs.apply(reason, axis=1)

    # 継続時間帯別に終了理由の内訳を見る
    pairs['duration_bucket'] = pd.cut(
        pairs['duration_sec'], bins=[0, 5, 20, 60, float('inf')],
        labels=['<5s', '5-20s', '20-60s', '>=60s']
    )
    summary = pd.crosstab(pairs['duration_bucket'], pairs['end_reason'])
    print(summary)
    print("\n=== 割合 ===")
    print(summary.div(summary.sum(axis=1), axis=0).round(3))

    return pairs


def plot_mode_distribution(df, figsize=(8, 6)):
    """
    ペアタイプ（mode_id）別のペア数を棒グラフで可視化する関数。
    pair_id, mode_id列を持つDataFrame（give_mode_id_rows等の実行後）に対して使う。

    行単位ではなくペア単位（pair_idごとに代表のmode_idを1つ）でカウントする。
    同じペアの複数フレームを重複カウントしないため。
    """
    mode_labels = {1: 'C-C', 2: 'C-T', 3: 'T-C', 4: 'T-T', 5: 'Other'}

    # ペアごとに代表のmode_idを1つ取り出す（同一ペア内でmode_idは変化しない前提）
    pair_modes = df.dropna(subset=['pair_id']).groupby('pair_id')['mode_id'].first()
    counts = pair_modes.value_counts().sort_index()
    labels = [mode_labels.get(m, str(int(m))) for m in counts.index]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts.values, color='skyblue')

    # 値をバーの上に表示
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
                str(int(bar.get_height())), ha='center', va='bottom')

    ax.set_xlabel('Pair type (mode_id)')
    ax.set_ylabel('Number of pairs')
    ax.set_title('Pair Type Distribution')
    plt.tight_layout()
    plt.show()

    return counts

def plot_lane_distribution(df, lanes=[2, 3, 4], figsize=(8, 6)):
    """
    車線ごとの車両クラス別分布（Car/Truck台数）を棒グラフで可視化する関数。
    Vehicle_ID, time_period, Lane_ID, v_Classを持つDataFrameに対して使う。

    Vehicle_ID+time_periodで重複除去してから台数を数える
    （同一車両の複数フレームを重複カウントしないため）。
    """
    base_df = df[df['Lane_ID'].isin(lanes)]
    base_df = base_df.drop_duplicates(subset=['Vehicle_ID', 'time_period', 'Lane_ID'])
    lane_class = base_df.groupby(['Lane_ID', 'v_Class'])['Vehicle_ID'].nunique().reset_index()

    # pivotでLane_ID×v_Classのクロス集計表にする（データがない組み合わせは0台で埋める）
    pivot_df = lane_class.pivot(index='Lane_ID', columns='v_Class', values='Vehicle_ID').fillna(0)

    x = pivot_df.index.to_numpy()
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    # v_Class: 2=Car, 3=Truck。片方が0台で列自体が存在しなくてもエラーを回避
    cars_vals = pivot_df[2].values if 2 in pivot_df.columns else np.zeros(len(x))
    trucks_vals = pivot_df[3].values if 3 in pivot_df.columns else np.zeros(len(x))

    bars_car = ax.bar(x - width / 2, cars_vals, width, label='Car', color='skyblue')
    bars_truck = ax.bar(x + width / 2, trucks_vals, width, label='Truck', color='orange')

    for bar in bars_car:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha='center', va='bottom')
    for bar in bars_truck:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha='center', va='bottom')

    ax.set_xlabel('Lane ID')
    ax.set_ylabel('Number of vehicles')
    ax.set_xticks(x)
    ax.legend()
    ax.set_title('Lane Distribution')
    plt.tight_layout()
    plt.show()

    return pivot_df

def plot_duration_histogram(df, pair_col='pair_id', duration_threshold=60):
    """
    追従ペアの追従時間（秒）の分布をヒストグラムで可視化する関数。
    全体分布と、閾値付近を見やすくした0-120秒の拡大版を並べて表示する。
    """
    valid = df.dropna(subset=[pair_col])
    pair_duration = valid.groupby(pair_col)['Global_Time'].agg(
        duration_sec=lambda x: (x.max() - x.min()) / 1000 + 0.1
    )

    print(f"ペア総数: {len(pair_duration):,}")
    print(f"追従時間0秒のペア数: {(pair_duration['duration_sec'] == 0).sum():,}")
    print(f"{duration_threshold}秒以上: {(pair_duration['duration_sec'] >= duration_threshold).sum():,}")
    print(f"{duration_threshold}秒未満: {(pair_duration['duration_sec'] < duration_threshold).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 全体分布
    axes[0].hist(pair_duration['duration_sec'], bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(duration_threshold, color='red', linestyle='--', label=f'{duration_threshold}s threshold')
    axes[0].set_xlabel('Car-following duration (s)')
    axes[0].set_ylabel('Number of pairs')
    axes[0].set_title('Duration distribution (all pairs, pre-filter)')
    axes[0].legend()

    # 0〜120秒の拡大
    near = pair_duration[pair_duration['duration_sec'] <= 120]
    axes[1].hist(near['duration_sec'], bins=60, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(duration_threshold, color='red', linestyle='--', label=f'{duration_threshold}s threshold')
    axes[1].set_xlabel('Car-following duration (s)')
    axes[1].set_ylabel('Number of pairs')
    axes[1].set_title('Zoomed (0-120s)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return pair_duration

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


def count_mode_by_stage(stages):
    """
    stages: dict of {stage_name: dataframe}, each df must have 'pair_id' and 'mode_id' columns.
    Counts unique pairs per mode_id at each stage.
    """
    mode_labels = {1: 'C-C', 2: 'C-T', 3: 'T-C', 4: 'T-T', 5: 'Others'}
    records = []
    for name, df in stages.items():
        valid = df.dropna(subset=['pair_id'])
        pair_modes = valid.groupby('pair_id')['mode_id'].first()

        row = {'stage': name}
        for mode_id, label in mode_labels.items():
            row[label] = int((pair_modes == mode_id).sum())
        row['Unresolved'] = int(pair_modes.isna().sum())
        row['Total'] = len(pair_modes)
        records.append(row)

    return pd.DataFrame(records)

def count_mode_rows_by_stage(stages):
    """
    stages: dict of {stage_name: dataframe}, each df must have 'mode_id' column.
    Counts rows (not unique pairs) per mode_id at each stage.
    """
    mode_labels = {1: 'C-C', 2: 'C-T', 3: 'T-C', 4: 'T-T', 5: 'Others'}
    records = []
    for name, df in stages.items():
        row = {'stage': name}
        for mode_id, label in mode_labels.items():
            row[label] = int((df['mode_id'] == mode_id).sum())
        row['Unresolved'] = int(df['mode_id'].isna().sum())
        row['Total'] = len(df)
        records.append(row)

    return pd.DataFrame(records)

def analyze_zero_gaps(df, col='Preceding'):
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()
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
    blocks['prev_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(1)
    blocks['next_value'] = blocks.groupby(['Vehicle_ID', 'time_period'])['value'].shift(-1)

    zero_blocks = blocks[blocks['value'] == 0].copy()
    zero_blocks['same_leader_after'] = (
        (zero_blocks['prev_value'] == zero_blocks['next_value']) & zero_blocks['prev_value'].notna()
    )

    print(f"Preceding=0のブロック数: {len(zero_blocks):,}")

    bins = [0, 5, 10, 20, 50, float('inf')]
    labels = ['<=5f(補正済)', '6-10f', '11-20f', '21-50f', '51f+']
    zero_blocks['gap_bucket'] = pd.cut(zero_blocks['frame_count'], bins=bins, labels=labels)

    print("\n=== 0区間の長さ別: 前後で同じ先行車に戻る割合 ===")
    print(zero_blocks.groupby('gap_bucket')['same_leader_after'].agg(['sum', 'count', 'mean']))

    return zero_blocks



def analyze_preceding_zero_duration(df):
    """
    Preceding=0（先行車なし）が何フレーム連続しているかの分布を出力する関数
    """
    print("Preceding=0 の連続フレーム数を集計中...")
    
    # 時間帯、車両ID、時間順にソート
    df = df.sort_values(['time_period', 'Vehicle_ID', 'Global_Time']).copy()
    
    # Precedingが0かどうかのフラグ
    is_zero = df['Preceding'] == 0
    
    # グループ分けの境界を定義
    # 車両が変わった時、時間帯が変わった時、または0と0以外が切り替わったタイミング
    boundary = (
        (df['Vehicle_ID'] != df['Vehicle_ID'].shift(1)) |
        (df['time_period'] != df['time_period'].shift(1)) |
        (is_zero != is_zero.shift(1))
    )
    
    # 連続するブロックごとにIDを振る
    df['block_id'] = boundary.cumsum()
    
    # Preceding=0 のブロックのみを抽出して、各ブロックのフレーム数（行数）をカウント
    zero_durations = df[is_zero].groupby('block_id').size()
    
    # -----------------------
    # 結果の出力
    # -----------------------
    print("\n=== Preceding=0 の持続フレーム数 統計量 ===")
    print(zero_durations.describe())
    
    print("\n=== 短い持続フレーム数（1〜20フレーム）の出現回数 ===")
    print(zero_durations[zero_durations <= 20].value_counts().sort_index())
    
    # -----------------------
    # ヒストグラムの描画
    # -----------------------
    plt.figure(figsize=(10, 5))
    # 極端に長い追従切れ（数千フレームなど）は除外し、200フレーム（20秒）以下を可視化
    zero_durations[zero_durations <= 200].hist(bins=100, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Preceding=0 Durations (<= 200 frames)")
    plt.xlabel("Duration (frames, 1 frame = 0.1s)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    return zero_durations