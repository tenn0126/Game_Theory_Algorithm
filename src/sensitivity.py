import numpy as np
import pandas as pd

from src.nominal_speed_func import (
    logistic_speed_density, logistic_density_speed,
    underwood_speed_density, underwood_density_speed,
)

#################################################################################
# 感度分析 Step 0: 協力余剰sを大量のパラメータセットで高速に評価するための基盤
#
# 既存の compute_1pipe_speed_for_snapshots は1行ずつbrentqを呼ぶため、
# パラメータ1セットごとに数秒かかる。ここでは
#   ・Eq(7)の残差がuについて単調増加(u_i^-1(u)が単調減少)であることを使い、
#     全スナップショット(または全グリッド点)を同時に二分法で解く
#   ・パラメータは1つのdict(param set)にまとめ、u*・p*・sで同じ値を必ず使う
# ことで、1セットあたりの計算をミリ秒オーダーにする。
#
# 単位: 密度 veh/km/lane、速度 km/h（既存コードと同じ）
#################################################################################

SCALINGS = ('printed', 'inverted')


def make_param_set(car_params, truck_params, a2, b1, name=None):
    """
    感度分析で動かす9個のパラメータを1つのdictにまとめる。
    Eq(7)のスケーリング形では、C-T・T-Cの関数はa2・b1を通じてしか効かないため、
    名目速度関数はC-C(car_params)とT-T(truck_params)の2つだけ持てばよい。

    car_params: ub, uf, rhoc, theta1, theta2 を含むdict（fit_nominal_speed_logisticの戻り値でもよい）
    truck_params: uf, rhoc を含むdict（fit_nominal_speed_underwoodの戻り値でもよい）
    """
    return {
        'name': name,
        'car': {k: float(car_params[k]) for k in ('ub', 'uf', 'rhoc', 'theta1', 'theta2')},
        'truck': {k: float(truck_params[k]) for k in ('uf', 'rhoc')},
        'a2': float(a2),
        'b1': float(b1),
    }


def _check_scaling(scaling):
    if scaling not in SCALINGS:
        raise ValueError(f"scaling は {SCALINGS} のどちらかを指定してください (指定値: {scaling!r})")


def _scaling_coefs(ps, scaling):
    """Eq(7)の和 Σ_j ρ_j·c_j で、相手クラスの密度にかける係数 (c_ct, c_tc) を返す。"""
    _check_scaling(scaling)
    if scaling == 'printed':      # 論文Eq.7の記載どおり rho_j/a_j, rho_j/b_j
        return 1.0 / ps['a2'], 1.0 / ps['b1']
    return ps['a2'], ps['b1']     # inverted: rho_j*a_j, rho_j*b_j


def _u1_inv(u, car):
    return logistic_density_speed(u, car['ub'], car['uf'], car['rhoc'], car['theta1'], car['theta2'])


def _u2_inv(u, truck):
    return underwood_density_speed(u, truck['uf'], truck['rhoc'])


#################################################################################
# Eq(7)の残差（配列対応版）
def eq7_residual(u, rho1, rho2, ps, scaling):
    c_ct, c_tc = _scaling_coefs(ps, scaling)
    rho_tot = rho1 + rho2
    term1 = (rho1 / _u1_inv(u, ps['car'])) * (rho1 + rho2 * c_ct)    # 車(クラス1)の項
    term2 = (rho2 / _u2_inv(u, ps['truck'])) * (rho1 * c_tc + rho2)  # トラック(クラス2)の項
    return (term1 + term2) / rho_tot - 1.0


#################################################################################
# 1-pipe速度u*を全点同時に二分法で解く
def solve_1pipe_speed_vec(rho1, rho2, ps, scaling, n_iter=60, margin=1e-3):
    """
    rho1, rho2: 同じ形の配列（スナップショット列でも2次元グリッドでもよい）
    探索区間は既存の solve_1pipe_speed と同じ:
      下限 = margin
      上限 = min(車がいれば f(0)-margin, トラックがいれば uf-margin)
    区間の両端で残差の符号が変わらない点（brentqならValueError）はNaNを返す。
    n_iter=60 で区間幅(~120 km/h)は 1e-16 km/h 程度まで縮む。
    """
    rho1 = np.asarray(rho1, dtype=float)
    rho2 = np.asarray(rho2, dtype=float)
    rho1, rho2 = np.broadcast_arrays(rho1, rho2)
    car, truck = ps['car'], ps['truck']

    car_top = logistic_speed_density(0.0, car['ub'], car['uf'], car['rhoc'],
                                     car['theta1'], car['theta2']) - margin
    truck_top = truck['uf'] - margin
    upper = np.where(rho1 > 0, car_top, np.inf)
    upper = np.minimum(upper, np.where(rho2 > 0, truck_top, np.inf))
    lower = np.full(rho1.shape, margin)

    valid = (rho1 + rho2 > 0) & np.isfinite(upper)

    with np.errstate(divide='ignore', invalid='ignore'):
        r_lo = eq7_residual(lower, rho1, rho2, ps, scaling)
        r_hi = eq7_residual(np.where(valid, upper, 1.0), rho1, rho2, ps, scaling)
        valid &= (np.sign(r_lo) != np.sign(r_hi)) | (r_lo == 0) | (r_hi == 0)

        lo = lower.copy()
        hi = np.where(valid, upper, lower)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            r_mid = eq7_residual(mid, rho1, rho2, ps, scaling)
            # 残差はuについて単調増加 → r_mid<0 なら解はmidより上
            go_up = r_mid < 0
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)

    return np.where(valid, 0.5 * (lo + hi), np.nan)


#################################################################################
# u* → p1*, p2* → s を一括計算（Eq.4, Definition 1）
def compute_surplus_vec(rho1, rho2, ps, scaling, **solve_kw):
    """
    戻り値: dict(u_1pipe, p1_star, p2_star, s)  いずれも rho1 と同じ形の配列
    u*とp*に同じパラメータが使われることを関数の形で保証する。
    """
    u = solve_1pipe_speed_vec(rho1, rho2, ps, scaling, **solve_kw)
    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = np.asarray(rho1, dtype=float) / _u1_inv(u, ps['car'])
        p2 = np.asarray(rho2, dtype=float) / _u2_inv(u, ps['truck'])
    return {'u_1pipe': u, 'p1_star': p1, 'p2_star': p2, 's': 1.0 - p1 - p2}


#################################################################################
# 出力①: (rho1, rho2)平面上のsの"場"（観測データに依存しない、パラメータだけで決まる量）
def surplus_field(ps, scaling, rho1_max=80.0, rho2_max=8.0, n1=161, n2=81):
    """
    rho1 ∈ [0, rho1_max], rho2 ∈ [0, rho2_max] の格子上で u*, p*, s を計算する。
    戻り値: dict(rho1, rho2 = 1次元の軸, u_1pipe, p1_star, p2_star, s = shape (n2, n1))
            行がrho2、列がrho1（pcolormesh(rho1, rho2, s) でそのまま描ける向き）
    """
    r1 = np.linspace(0.0, rho1_max, n1)
    r2 = np.linspace(0.0, rho2_max, n2)
    R1, R2 = np.meshgrid(r1, r2)
    out = compute_surplus_vec(R1, R2, ps, scaling)
    out['rho1'], out['rho2'] = r1, r2
    return out


#################################################################################
# 出力②③のための観測状態表
def build_observed_states(df_macro, car_class=2, truck_class=3, on_grid_only=True,
                          step_ms=500, head_s=0, tail_s=0):
    """
    compute_macroscopic_data の返り値から、スナップショットごとの
    rho1, rho2, u1_obs, u2_obs の表を作る（compute_1pipe_speed_for_snapshotsのpivotと同じ）。
    観測状態はパラメータに依存しないので、感度分析では1回だけ作って使い回す。

    on_grid_only: 各time_periodの開始時刻から step_ms 間隔の格子に乗るスナップショットだけ残す。
                  （period 2/3 の Global_Time が重なっているため、compute_macroscopic_data の
                    出力には格子から100msずれたスナップショットが混ざる。2026-09-30レビュー参照）
    head_s, tail_s: 各time_periodの先頭・末尾の秒数を除外（区間が空の立ち上がり・終了直前）
    """
    pivot = df_macro.pivot_table(
        index=['time_period', 'Global_Time'], columns='v_Class',
        values=['density_veh_km', 'speed_kmh']
    )
    col_map = {
        ('density_veh_km', car_class): 'rho1',
        ('density_veh_km', truck_class): 'rho2',
        ('speed_kmh', car_class): 'u1_obs',
        ('speed_kmh', truck_class): 'u2_obs',
    }
    pivot.columns = [col_map[c] for c in pivot.columns]
    pivot[['rho1', 'rho2']] = pivot[['rho1', 'rho2']].fillna(0.0)
    df = pivot.reset_index()

    n0 = len(df)
    g = df.groupby('time_period')['Global_Time']
    t0, t1 = g.transform('min'), g.transform('max')
    keep = pd.Series(True, index=df.index)
    if on_grid_only:
        keep &= ((df['Global_Time'] - t0) % step_ms) == 0
    if head_s > 0:
        keep &= (df['Global_Time'] - t0) / 1000 >= head_s
    if tail_s > 0:
        keep &= (t1 - df['Global_Time']) / 1000 >= tail_s
    df = df[keep].reset_index(drop=True)
    df['rho_tot'] = df['rho1'] + df['rho2']

    print(f"観測状態表: {n0:,}件 → {len(df):,}件 "
          f"(on_grid_only={on_grid_only}, head_s={head_s}, tail_s={tail_s})")
    return df


def classify_regime_vec(du1, du2, eps_kmh):
    """
    traffic_regime.classify_traffic_regime と同じ判定（Eq.11）を、表示なしで配列に対して行う。
    du_i = ũ_i - u*。戻り値: '2-pipe' / '1-pipe' / 'non-NE' / 'undefined' の配列
    """
    du1 = np.asarray(du1, dtype=float)
    du2 = np.asarray(du2, dtype=float)
    valid = np.isfinite(du1) & np.isfinite(du2)
    eq1, eq2 = np.abs(du1) <= eps_kmh, np.abs(du2) <= eps_kmh
    ge1, ge2 = du1 >= -eps_kmh, du2 >= -eps_kmh

    regime = np.full(du1.shape, 'non-NE', dtype=object)
    regime[valid & ge1 & ge2 & ~(eq1 & eq2)] = '2-pipe'
    regime[valid & eq1 & eq2] = '1-pipe'
    regime[~valid] = 'undefined'
    return regime


def evaluate_observed(states, ps, scaling, eps_kmh=None):
    """
    観測状態表(build_observed_states)に u*, p*, s を付ける。eps_kmhを渡すと regime も付ける。
    statesは変更せずコピーを返す。
    """
    df = states.copy()
    out = compute_surplus_vec(df['rho1'].values, df['rho2'].values, ps, scaling)
    for k, v in out.items():
        df[k] = v
    if eps_kmh is not None:
        df['du1'] = df['u1_obs'] - df['u_1pipe']
        df['du2'] = df['u2_obs'] - df['u_1pipe']
        df['regime'] = classify_regime_vec(df['du1'], df['du2'], eps_kmh)
    return df


def summarize_surplus(df_eval, eps_kmh=None):
    """
    出力②③の要約指標を1行のdictにまとめる（パラメータスイープの1行分）。
      ② s分布: 解けた件数, sの中央値・平均・10/90%点, P(s>0)
      ③ regime: Pcoop(=2-pipe/判定対象), 1-pipe率, 2-pipeでの平均s,
               命題1チェック(2-pipeなのにs<=0の件数。0でなければ理論と矛盾)
    s分布は両クラスがいるスナップショットに限定する（片方だけならsは恒等的に0のため）。
    """
    both = (df_eval['rho1'] > 0) & (df_eval['rho2'] > 0)
    s = df_eval.loc[both, 's']
    s_ok = s.dropna()
    row = {
        'n_both': int(both.sum()),
        'n_solved': int(s_ok.size),
        's_median': s_ok.median(),
        's_mean': s_ok.mean(),
        's_q10': s_ok.quantile(0.10),
        's_q90': s_ok.quantile(0.90),
        'frac_s_pos': (s_ok > 0).mean() if s_ok.size else np.nan,
        'u_star_median': df_eval.loc[both, 'u_1pipe'].median(),
    }
    if eps_kmh is not None:
        if 'regime' not in df_eval.columns:
            raise ValueError("regime列がありません。evaluate_observed に eps_kmh を渡してください")
        reg = df_eval['regime']
        judged = reg != 'undefined'
        n_tot = int(judged.sum())
        is2 = reg == '2-pipe'
        row.update({
            'n_judged': n_tot,
            'pcoop': is2.sum() / n_tot if n_tot else np.nan,
            'frac_1pipe': (reg == '1-pipe').sum() / n_tot if n_tot else np.nan,
            's_mean_2pipe': df_eval.loc[is2, 's'].mean(),
            'n_2pipe_s_nonpos': int((is2 & (df_eval['s'] <= 0)).sum()),
            'du1_median': df_eval.loc[judged, 'du1'].median(),
            'du2_median': df_eval.loc[judged, 'du2'].median(),
        })
    return row


#################################################################################
# MAE: そのパラメータセットがNGSIMの追従データにどれだけ合っているか
def compute_fit_mae(ps, df_mdensity, scaling, density_col='mdensity_veh_km', speed_col='v_Vel'):
    """
    ミクロ密度データ(compute_micro_densityの返り値)に対する、4つの追従タイプのMAE(km/h)。
    パラメータを推定するわけではないので、train/testに分けず全データで評価する。

      mae_cc: f(ρ)        vs C-C (mode 1)
      mae_tt: g(ρ)        vs T-T (mode 4)
      mae_ct: f12(ρ)      vs C-T (mode 2)
      mae_tc: g21(ρ)      vs T-C (mode 3)

    f12, g21 の形は scaling に合わせる:
      printed : f(ρ/a2), g(ρ/b1)   （Step1のEq.20と同じ、a2・b1の推定に使った形）
      inverted: f(ρ·a2), g(ρ·b1)   （Eq.7のinverted解釈を名目速度関数に戻した形）
    inverted のときStep1で推定したa2・b1は前提が整合しないため、mae_ct/mae_tcが悪化するのは
    それ自体がアルゴリズムの不整合を表す。比較用に mae_ct_eq20 / mae_tc_eq20（常にρ/a）も返す。
    """
    _check_scaling(scaling)
    car, truck = ps['car'], ps['truck']
    f = lambda r: logistic_speed_density(r, car['ub'], car['uf'], car['rhoc'], car['theta1'], car['theta2'])
    g = lambda r: underwood_speed_density(r, truck['uf'], truck['rhoc'])

    def _xy(mode_id):
        d = df_mdensity.loc[df_mdensity['mode_id'] == mode_id, [density_col, speed_col]].dropna()
        return d[density_col].values, d[speed_col].values * 0.3048 * 3.6   # ft/s → km/h

    def _mae(y, pred):
        return float(np.mean(np.abs(y - pred))) if len(y) else np.nan

    x1, y1 = _xy(1)
    x2, y2 = _xy(2)
    x3, y3 = _xy(3)
    x4, y4 = _xy(4)
    a2, b1 = ps['a2'], ps['b1']
    if scaling == 'printed':
        pred_ct, pred_tc = f(x2 / a2), g(x3 / b1)
    else:
        pred_ct, pred_tc = f(x2 * a2), g(x3 * b1)

    return {
        'mae_cc': _mae(y1, f(x1)),
        'mae_tt': _mae(y4, g(x4)),
        'mae_ct': _mae(y2, pred_ct),
        'mae_tc': _mae(y3, pred_tc),
        'mae_ct_eq20': _mae(y2, f(x2 / a2)),
        'mae_tc_eq20': _mae(y3, g(x3 / b1)),
    }
