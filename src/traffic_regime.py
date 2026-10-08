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


from src.data_prep import *
from src.nominal_speed_func import *
from src.macroscopic_data import *
from src.cooperation_surplus import *


#################################################################################
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

#############################################################################
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


# 論文Fig.17風：λを動かしたときの損失関数の形
#   lam_range: 描く範囲。論文Fig.17に合わせて既定値は(-1, 2)。推定範囲[0,1]は色付きで表示
def plot_surplus_split_loss(df, car_params, truck_params, lam_hat=None, w1=0.5, w2=0.5,
                            n_grid=301, lam_range=(-1.0, 2.0)):
    lams = np.linspace(lam_range[0], lam_range[1], n_grid)
    losses = np.array([surplus_split_loss(l, df, car_params, truck_params, w1, w2) for l in lams])
    losses[~np.isfinite(losses)] = np.nan   # 道路占有が負になる無効なλは描かない

    n_invalid = int(np.isnan(losses).sum())
    if n_invalid > 0:
        print(f"道路占有が0以下になり無効なλ: {n_invalid}/{n_grid}点（図では空白）")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(0.0, 1.0, color='tab:green', alpha=0.08, label='estimation range [0, 1]')
    ax.plot(lams, losses, color='tab:blue')
    if lam_hat is not None:
        ax.scatter([lam_hat], [surplus_split_loss(lam_hat, df, car_params, truck_params, w1, w2)],
                   color='red', zorder=3, label=f'λ̂ = {lam_hat:.4f}')
    ax.set_xlabel('Surplus split factor λ')
    ax.set_ylabel('Loss L(λ)  ((km/h)$^2$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

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