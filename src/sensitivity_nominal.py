#感度分析②: 名目速度関数のパラメータを1つずつ少しだけ動かす（OAT: one-at-a-time）
#
# 目的: 「ほんの少し値が変わっただけで、sの符号・レジーム・協力性の判定が大きく変わる」
#       パラメータがあるかを探す。あわせてフィットMAEの変化も出し、
#       「結果は大きく変わるのにMAEはほとんど変わらない」（＝データで決まらない）かを見る。
#
# 動かすもの: C-C ロジスティックの ub, uf, rhoc, theta1, theta2 と T-T Underwood の uf, rhoc（7個）
# 固定するもの: a2, b1, eps, 観測状態（rho1, rho2, ũ1, ũ2）
#
# 方針（sensitivity_scaling と同じ）
#   ・観測状態は pipeline_step3（run_pipeline の Step 3 と同じ関数呼び出し）で1回だけ作る
#   ・u*, p*, s は sensitivity_scaling.surplus_at（パイプラインとの一致確認済みの二分法版）
#   ・レジーム判定は whole_pipeline.classify_traffic_regime と同じ Eq.11 を配列で行う（表示なし）
#
# 単位: 密度 veh/km/lane、速度 km/h（パイプライン全体と同じ）
# 図のラベルは英語（matplotlibに日本語フォントを設定していないため）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.whole_pipeline import logistic_speed_density, underwood_speed_density
from src.sensitivity_scaling import surplus_at

FT_S_TO_KMH = 0.3048 * 3.6

# 動かすパラメータ（(クラス, キー), 表示名）
PARAMS = [
    (('car', 'ub'), 'C-C ub'),
    (('car', 'uf'), 'C-C uf'),
    (('car', 'rhoc'), 'C-C rhoc'),
    (('car', 'theta1'), 'C-C theta1'),
    (('car', 'theta2'), 'C-C theta2'),
    (('truck', 'uf'), 'T-T uf'),
    (('truck', 'rhoc'), 'T-T rhoc'),
]

# 既定の倍率: ±1%, ±5%, ±10%
DEFAULT_FACTORS = (0.90, 0.95, 0.99, 1.01, 1.05, 1.10)


#################################################################################
# パラメータを1つだけ動かす
def perturb(car_params, truck_params, key, factor):
    """
    key = ('car', 'theta1') のように指定し、その値だけ factor 倍した (car, truck) のコピーを返す。
    元のdictは変更しない。必要なキーだけ取り出すので、fit_* の戻り値(dict)をそのまま渡してよい。
    """
    car = {k: float(car_params[k]) for k in ('ub', 'uf', 'rhoc', 'theta1', 'theta2')}
    truck = {k: float(truck_params[k]) for k in ('uf', 'rhoc')}
    cls, name = key
    target = car if cls == 'car' else truck
    target[name] = target[name] * factor
    return car, truck


#################################################################################
# フィットMAE（NGSIMのミクロ密度データに対する当てはまり）
def prepare_mae_data(df_mdensity, density_col='mdensity_veh_km', speed_col='v_Vel'):
    """
    compute_micro_density の返り値から、追従タイプ(mode_id 1〜4)ごとの (密度, 速度km/h) の配列を取り出す。
    パラメータを変えるたびに表を引き直さないよう、最初に1回だけ作っておく。
      1: C-C, 2: C-T, 3: T-C, 4: T-T
    """
    out = {}
    for mode_id in (1, 2, 3, 4):
        d = df_mdensity.loc[df_mdensity['mode_id'] == mode_id, [density_col, speed_col]].dropna()
        out[mode_id] = (d[density_col].values, d[speed_col].values * FT_S_TO_KMH)
    return out


def fit_mae(mae_data, car, truck, a2, b1):
    """
    4つの追従タイプのMAE (km/h)。C-T, T-C は Step 1 の Eq.20 と同じ f(ρ/a2), g(ρ/b1)。
    パラメータの推定ではないので train/test に分けず全データで評価する。
    """
    f = lambda r: logistic_speed_density(r, car['ub'], car['uf'], car['rhoc'],
                                         car['theta1'], car['theta2'])
    g = lambda r: underwood_speed_density(r, truck['uf'], truck['rhoc'])

    def _mae(mode_id, model):
        x, y = mae_data[mode_id]
        return float(np.mean(np.abs(y - model(x)))) if len(y) else np.nan

    return {
        'mae_cc': _mae(1, f),
        'mae_ct': _mae(2, lambda r: f(r / a2)),
        'mae_tc': _mae(3, lambda r: g(r / b1)),
        'mae_tt': _mae(4, g),
    }


#################################################################################
# 1つのパラメータセットを評価する
def classify_regime(du1, du2, eps_kmh):
    """whole_pipeline.classify_traffic_regime と同じ Eq.11 の判定（表示なし・配列版、epsは両クラス共通）"""
    valid = np.isfinite(du1) & np.isfinite(du2)
    eq1, eq2 = np.abs(du1) <= eps_kmh, np.abs(du2) <= eps_kmh
    ge1, ge2 = du1 >= -eps_kmh, du2 >= -eps_kmh
    regime = np.full(du1.shape, 'non-NE', dtype=object)
    regime[valid & ge1 & ge2 & ~(eq1 & eq2)] = '2-pipe'
    regime[valid & eq1 & eq2] = '1-pipe'
    regime[~valid] = 'undefined'
    return regime


def evaluate(states_both, car, truck, a2, b1, eps_kmh):
    """
    両クラスがいるスナップショット(states_both)について、スナップショットごとの結果を返す。
    戻り値: dict(u_1pipe, p1_star, p2_star, s, du1, du2, regime, coop)
      coop = s>0 かつ 2-pipe（Step 5 の「協力性あり」）
    """
    r = surplus_at(states_both, car, truck, a2, b1, 'printed')
    r['du1'] = states_both['u1_obs'].values - r['u_1pipe']
    r['du2'] = states_both['u2_obs'].values - r['u_1pipe']
    r['regime'] = classify_regime(r['du1'], r['du2'], eps_kmh)
    r['coop'] = (r['s'] > 0) & (r['regime'] == '2-pipe')
    return r


def summarize(r):
    """スナップショットごとの結果を1行の要約にする（判定対象 = u*が解けたスナップショット）"""
    judged = r['regime'] != 'undefined'
    n = int(judged.sum())
    s = r['s'][judged]
    reg = r['regime'][judged]
    frac = lambda m: float(np.mean(m)) if n else np.nan
    med = lambda x: float(np.median(x)) if n else np.nan
    return {
        'n_judged': n,
        's_median': med(s),
        's_q10': float(np.quantile(s, 0.10)) if n else np.nan,
        's_q90': float(np.quantile(s, 0.90)) if n else np.nan,
        'frac_s_pos': frac(s > 0),
        'u_median': med(r['u_1pipe'][judged]),
        'du1_median': med(r['du1'][judged]),
        'du2_median': med(r['du2'][judged]),
        'share_2pipe': frac(reg == '2-pipe'),     # 論文の Pcoop（2-pipe の割合）
        'share_1pipe': frac(reg == '1-pipe'),
        'share_nonNE': frac(reg == 'non-NE'),
        'share_coop': frac(r['coop'][judged]),    # s>0 かつ 2-pipe の割合
    }


def compare_to_base(r, r0):
    """
    基準からの「判定の入れ替わり」をスナップショット単位で数える。
    割合が同じでも、中身のスナップショットが入れ替わっていることがあるため。
    """
    both = (r['regime'] != 'undefined') & (r0['regime'] != 'undefined')
    n = int(both.sum())
    frac = lambda m: float(np.mean(m[both])) if n else np.nan
    return {
        'flip_s_sign': frac(np.sign(r['s']) != np.sign(r0['s'])),   # sの符号が反転
        'flip_regime': frac(r['regime'] != r0['regime']),           # レジームが変わった
        'flip_coop': frac(r['coop'] != r0['coop']),                 # 協力性あり/なしが変わった
    }


#################################################################################
# OAT（1つずつ動かす）表
def oat_table(states, car_params, truck_params, a2, b1, eps_kmh, mae_data,
              factors=DEFAULT_FACTORS, params=PARAMS, base_name=None):
    """
    params の各パラメータを factors 倍に1つずつ動かし（他は基準のまま）、要約・入れ替わり・MAEを並べた表を返す。
    states: pipeline_step3 の戻り値（両クラスがいるスナップショットだけを使う）
    1行目(param='base', factor=1.0)が基準。d_* 列は基準からの差。
    """
    states_both = states[(states['rho1'] > 0) & (states['rho2'] > 0)]
    base_car, base_truck = perturb(car_params, truck_params, ('car', 'ub'), 1.0)
    r0 = evaluate(states_both, base_car, base_truck, a2, b1, eps_kmh)
    row0 = {'base': base_name, 'param': 'base', 'factor': 1.0, 'value': np.nan,
            **summarize(r0), **compare_to_base(r0, r0), **fit_mae(mae_data, base_car, base_truck, a2, b1)}
    rows = [row0]

    for key, label in params:
        for fac in factors:
            car, truck = perturb(car_params, truck_params, key, fac)
            r = evaluate(states_both, car, truck, a2, b1, eps_kmh)
            value = (car if key[0] == 'car' else truck)[key[1]]
            rows.append({'base': base_name, 'param': label, 'factor': fac, 'value': value,
                         **summarize(r), **compare_to_base(r, r0),
                         **fit_mae(mae_data, car, truck, a2, b1)})

    df = pd.DataFrame(rows)
    for col in ['s_median', 'frac_s_pos', 'u_median', 'share_2pipe', 'share_1pipe', 'share_coop',
                'mae_cc', 'mae_ct', 'mae_tc', 'mae_tt']:
        df[f'd_{col}'] = df[col] - row0[col]
    # そのパラメータが効く追従タイプのMAEの変化の合計
    #（C-Cのパラメータは C-C と C-T、T-Tのパラメータは T-T と T-C に効く）
    is_car = df['param'].str.startswith('C-C')
    df['d_mae_own'] = np.where(is_car, df['d_mae_cc'] + df['d_mae_ct'], df['d_mae_tt'] + df['d_mae_tc'])
    df.loc[df['param'] == 'base', 'd_mae_own'] = 0.0
    return df


def flip_summary(df_oat, pct):
    """
    倍率 1±pct/100 の2行のうち、絶対値の大きい方をパラメータごとに1行にまとめる（一覧表用）。
    """
    sub = df_oat[np.isclose(np.abs(df_oat['factor'] - 1.0), pct / 100)]
    cols = ['flip_s_sign', 'flip_regime', 'flip_coop', 'd_frac_s_pos', 'd_share_2pipe',
            'd_share_coop', 'd_u_median', 'd_mae_own']
    out = sub.groupby('param', sort=False)[cols].agg(lambda x: x.loc[x.abs().idxmax()])
    out.columns = [f'{c} (±{pct}%)' for c in cols]
    return out


#################################################################################
# 可視化
def plot_oat_lines(df_oat, metrics, title=None):
    """
    横軸 = 倍率(%)、縦軸 = 指標。パラメータごとに1本の線。metrics の数だけ横に並べる。
    """
    fig, axs = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False)
    base = df_oat[df_oat['param'] == 'base']
    for ax, m in zip(axs[0], metrics):
        for label, d in df_oat[df_oat['param'] != 'base'].groupby('param', sort=False):
            d = pd.concat([d, base]).sort_values('factor')
            ax.plot(100 * (d['factor'] - 1), d[m], marker='o', ms=3, label=label)
        ax.axvline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('parameter change (%)')
        ax.set_title(m)
    axs[0][-1].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_flip_bars(df_oat, pct, title=None):
    """
    ±pct% 動かしたとき、判定が入れ替わったスナップショットの割合（左3枚）と、
    そのときのMAEの変化（右1枚）をパラメータごとの棒で描く。
    """
    sub = df_oat[np.isclose(np.abs(df_oat['factor'] - 1.0), pct / 100)]
    labels = [lab for _, lab in PARAMS if lab in set(sub['param'])]
    panels = [('flip_s_sign', 'share of snapshots: s sign flipped'),
              ('flip_regime', 'share: regime changed'),
              ('flip_coop', 'share: cooperation judgement changed'),
              ('d_mae_own', 'change in fit MAE (km/h)')]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
    y = np.arange(len(labels))
    for ax, (col, name) in zip(axs, panels):
        for k, (fac, color) in enumerate([(1 - pct / 100, 'tab:blue'), (1 + pct / 100, 'tab:red')]):
            d = sub[np.isclose(sub['factor'], fac)].set_index('param').reindex(labels)
            ax.barh(y + (k - 0.5) * 0.38, d[col], height=0.38, color=color, label=f'x{fac:.2f}')
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(name, fontsize=10)
        if col.startswith('flip'):
            ax.set_xlim(0, 1)
    axs[-1].legend(fontsize=8)
    fig.suptitle(title or f'one parameter changed by ±{pct}%')
    fig.tight_layout()
    return fig


def plot_effect_vs_mae(df_oat, effect='flip_coop', pct=5, ax=None, title=None):
    """
    横軸 = MAEの変化、縦軸 = 判定の入れ替わり。左上に来るパラメータほど
    「データへの当てはまりはほとんど変わらないのに、結論が大きく変わる」（＝見直し候補）。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.2))
    sub = df_oat[np.isclose(np.abs(df_oat['factor'] - 1.0), pct / 100)]
    for (label, d), c in zip(sub.groupby('param', sort=False), plt.cm.tab10.colors):
        ax.scatter(d['d_mae_own'], d[effect], color=c, s=40, label=label)
        for _, row in d.iterrows():
            ax.annotate(f"x{row['factor']:.2f}", (row['d_mae_own'], row[effect]),
                        textcoords='offset points', xytext=(4, 3), fontsize=7)
    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('change in fit MAE (km/h)  [C-C+C-T or T-T+T-C]')
    ax.set_ylabel(effect)
    ax.set_title(title or f'±{pct}%: effect vs MAE change')
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5))
    return ax
