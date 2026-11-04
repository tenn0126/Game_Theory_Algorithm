import contextlib
import io
import warnings

import nlopt
import numpy as np
import pandas as pd

from src.nominal_speed_func import (
    fit_nominal_speed_logistic, fit_nominal_speed_underwood,
    fit_scaling_parameter_logistic, fit_scaling_parameter_underwood,
)
from src.ring_equilibrium import (
    CAR, TRUCK, PAIRS, make_ov_nominal, make_paper_nominal, h_eq, u_nominal,
    at_speed_cap, eq7_speed, separation_speed, surplus_at, true_1pipe_speed,
)
from src.uncertainty import _argmin_1d

#################################################################################
# Q1 ②a・②b の実験用の関数
#
#   格子の計算 : 真の名目速度関数での v_true, s_true と、アルゴリズムの u*, s を密度の格子で計算する
#   フィット   : 真の曲線から点を取り、論文形（または a2, b1 だけ）をフィットする
#   評価       : Δu* = u* − v_true、Δs = s_alg − s_true、符号の一致率
#################################################################################

FT_S_TO_KMH = 0.3048 * 3.6
MODE_ID = {(1, 1): 1, (1, 2): 2, (2, 1): 3, (2, 2): 4}   #NGSIMのmode_idと同じ番号
SIGN_EPS = 1e-6                                          #|s_true| がこれ以下なら符号を比べない（s=0 の計算誤差のため）


#################################################################################
# 1: 混ぜたことによる車間の増減 B(v)
#################################################################################
def cross_spacing_excess(nominal, v):
    """
    B(v) = h_CT(v) + h_TC(v) − h_CC(v) − h_TT(v) (m)。
    円環の式は n1·h_CC + n2·h_TT + m·B(v) = L と書けるので、v_true では s = m·B(v_true) / L になる。
    """
    h = {p: float(h_eq(nominal, *p, v)) for p in PAIRS}
    return h[(1, 2)] + h[(2, 1)] - h[(1, 1)] - h[(2, 2)]


#################################################################################
# 2: 密度の格子での計算（行が rho2、列が rho1）
#################################################################################
def truth_grid(nominal, r1s, r2s, m_frac=None):
    """
    正解の v_true, s_true。上限に張り付いた点と解なしの点は NaN。
    m_frac: None ならランダム混合、値を指定すると m = m_frac · min(ρ1, ρ2)
    """
    V = np.full((len(r2s), len(r1s)), np.nan)
    S = V.copy()
    for b, r2 in enumerate(r2s):
        for a, r1 in enumerate(r1s):
            m = None if m_frac is None else m_frac * min(r1, r2)
            v = true_1pipe_speed(nominal, r1, r2, m)
            if np.isnan(v) or at_speed_cap(nominal, r1, r2, v):
                continue
            V[b, a] = v
            S[b, a] = float(surplus_at(nominal, r1, r2, v)['s'])
    return V, S


def surplus_grid(nominal, r1s, r2s, U):
    """速度の格子 U での s。surplus_at は密度が1点ずつなのでループで計算する。"""
    S = np.full(U.shape, np.nan)
    for b, r2 in enumerate(r2s):
        for a, r1 in enumerate(r1s):
            if np.isfinite(U[b, a]):
                S[b, a] = float(surplus_at(nominal, r1, r2, U[b, a])['s'])
    return S


def eq7_grid(algo_nominal, r1s, r2s, scaling='printed'):
    """論文形の名目速度関数を入れた Eq.7 の u* と s。"""
    U = eq7_speed(algo_nominal, r1s[None, :], r2s[:, None], scaling)
    return U, surplus_grid(algo_nominal, r1s, r2s, U)


def own_curve_eq7_grid(algo_nominal, r1s, r2s):
    """
    C-C・T-T は真の曲線、C-T・T-C は h_ij = h_ii / a_ij としたときの Eq.7 の u* と s。
    form='scale' のOV形は同じ関係を持つので、ランダム混合の円環の解がそのまま Eq.7 の解になる。
    """
    U = np.array([[true_1pipe_speed(algo_nominal, r1, r2) for r1 in r1s] for r2 in r2s])
    return U, surplus_grid(algo_nominal, r1s, r2s, U)


def ratio_line(nominal, rhos, p2, m_frac=None):
    """
    トラック比率 p2 = ρ2 / (ρ1 + ρ2) を固定し、全体密度 rhos を動かしたときの v_true, s_true, u_sep。
    m_frac: None ならランダム混合、値を指定すると m = m_frac · min(ρ1, ρ2)
    上限に張り付いた点と解なしの点は NaN。
    """
    v, s, u_sep = (np.full(len(rhos), np.nan) for _ in range(3))
    for k, rho in enumerate(rhos):
        r1, r2 = (1 - p2) * rho, p2 * rho
        m = None if m_frac is None else m_frac * min(r1, r2)
        vk = true_1pipe_speed(nominal, r1, r2, m)
        u_sep[k] = separation_speed(nominal, r1, r2)
        if np.isnan(vk) or at_speed_cap(nominal, r1, r2, vk):
            continue
        v[k] = vk
        s[k] = float(surplus_at(nominal, r1, r2, vk)['s'])
    return {'v': v, 's': s, 'u_sep': u_sep}


def eq7_ratio_line(algo_nominal, rhos, p2, scaling='printed'):
    """ratio_line と同じ密度の線で、Eq.7 の u* と s を計算する。"""
    r1, r2 = (1 - p2) * rhos, p2 * rhos
    u = eq7_speed(algo_nominal, r1, r2, scaling)
    s = np.full(len(rhos), np.nan)
    for k in range(len(rhos)):
        if np.isfinite(u[k]):
            s[k] = float(surplus_at(algo_nominal, r1[k], r2[k], u[k])['s'])
    return {'v': u, 's': s}


#################################################################################
# 3: 真の曲線へのフィット
#################################################################################
def sample_curves(nominal, rho_min=2.0, rho_max=100.0, n=400, v_min=1.0):
    """
    真の4本の曲線の上に点を取り、NGSIMのフィット関数にそのまま渡せる表にする。ノイズはなし。
    密度は等間隔。ほぼ停止した点（v < v_min）は使わない。
    """
    rho = np.linspace(rho_min, rho_max, n)
    rows = []
    for p, mode_id in MODE_ID.items():
        v = np.asarray(u_nominal(nominal, *p, rho), dtype=float)
        keep = v >= v_min
        rows.append(pd.DataFrame({'mode_id': mode_id, 'mdensity_veh_km': rho[keep], 'v_Vel': v[keep] / FT_S_TO_KMH}))
    return pd.concat(rows, ignore_index=True)


def fit_paper_form(df, seed=0):
    """NGSIMと同じ手順（C-C → T-T → a2 → b1）で論文形をフィットする。探索範囲はフィット関数の既定値。"""
    nlopt.srand(seed)   #ISRESは確率的な探索なので乱数を固定する
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cc = fit_nominal_speed_logistic(df, 1)
        tt = fit_nominal_speed_underwood(df, 4)
        a2 = fit_scaling_parameter_logistic(df, 2, cc)
        b1 = fit_scaling_parameter_underwood(df, 3, tt)
    return {'nominal': make_paper_nominal(cc, tt, a2['scaling_param'], b1['scaling_param']),
            'cc': cc, 'tt': tt, 'a2': a2['scaling_param'], 'b1': b1['scaling_param'],
            'mae_cc': cc['test_mae'], 'mae_tt': tt['test_mae'],
            'mae_ct': a2['test_mae'], 'mae_tc': b1['test_mae']}


def fit_scaling_only(nominal, df):
    """C-C・T-T は真の曲線のまま、a2, b1 だけをフィットする。戻り値は form='scale' のOV形とフィット値。"""
    d = df[df['mode_id'] == MODE_ID[(CAR, TRUCK)]]
    x, y = d['mdensity_veh_km'].values, d['v_Vel'].values * FT_S_TO_KMH
    a2, _ = _argmin_1d(lambda a: np.sum(np.abs(y - u_nominal(nominal, CAR, CAR, x / a))), 0.1, 5.0)
    d = df[df['mode_id'] == MODE_ID[(TRUCK, CAR)]]
    x, y = d['mdensity_veh_km'].values, d['v_Vel'].values * FT_S_TO_KMH
    b1, _ = _argmin_1d(lambda b: np.sum(np.abs(y - u_nominal(nominal, TRUCK, TRUCK, x / b))), 0.1, 5.0)
    algo = make_ov_nominal(vmax=nominal['vmax'], length=nominal['length'], shape=nominal['shape'],
                           form='scale', scale={(CAR, TRUCK): a2, (TRUCK, CAR): b1})
    return algo, a2, b1


#################################################################################
# 4: 評価
#################################################################################
def compare(V, S, U, s, v_below=None):
    """
    正解 (V, S) とアルゴリズム (U, s) の差をまとめる。
    v_below: 指定すると v_true がこの値より低い点だけで集計する
    sign_match: |s_true| > SIGN_EPS の点での符号の一致率（真の s が0なら NaN）
    false_signal: |s_alg| > 0.01 の割合（真の s が0のときの偽の信号の大きさを見る）
    """
    truth_ok = np.isfinite(V) if v_below is None else (np.isfinite(V) & (V < v_below))
    ok = truth_ok & np.isfinite(U)
    du, ds = (U - V)[ok], (s - S)[ok]
    signed = np.abs(S[ok]) > SIGN_EPS
    return {
        'n_truth': int(truth_ok.sum()),
        'solved': ok.sum() / truth_ok.sum() if truth_ok.any() else np.nan,
        'du_mean': du.mean(), 'du_abs_max': np.abs(du).max(),
        'ds_mean': ds.mean(), 'ds_abs_max': np.abs(ds).max(),
        's_true_pos': np.mean(S[ok] > SIGN_EPS), 's_alg_pos': np.mean(s[ok] > 0),
        's_alg_min': s[ok].min(), 's_alg_max': s[ok].max(),
        'sign_match': np.mean(np.sign(s[ok][signed]) == np.sign(S[ok][signed])) if signed.any() else np.nan,
        'false_signal': np.mean(np.abs(s[ok]) > 0.01),
    }
