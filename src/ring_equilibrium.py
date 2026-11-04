import numpy as np
from scipy.optimize import brentq

from src.nominal_speed_func import (
    logistic_speed_density, logistic_density_speed,
    underwood_speed_density, underwood_density_speed,
)
from src.sensitivity import make_param_set, solve_1pipe_speed_vec

#################################################################################
# ダミーデータ検証の共通道具: 円環道路の定常状態ソルバー
#
#   層1 名目速度関数: 追従ペア4種類の、定常状態での速度と車間の関係 V_ij
#   層2 円環の設定: 台数 n1, n2、周長 L、車線ごとの並び m
#   層3 定常状態 : 正解の1-pipe速度 v_true、u_sep、2-pipe速度、任意の u での p*, s
#   （層4 OVの時間発展は Q2 で追加する）
#
# 円環の定常状態では同じ車線の全車が同じ速度 v で走り、各車の車間は h_ij(v)。
# よって 1車線の速度は  Σ_ij n_ij · h_ij(v) = L  の解。
# 円環では C-T ペア数 = T-C ペア数 = m（トラックの塊の数）なので
#   n_CC = n1 − m, n_CT = n_TC = m, n_TT = n2 − m
# ランダム混合 m = n1·n2/N を入れると、論文形の名目速度関数では印刷版Eq.7と同じ式になる。
#
# 単位: 速度 km/h、密度 veh/km/lane、車間（前端間）h は m、周長 L は m
# クラス番号: 1 = 乗用車、2 = トラック（論文と同じ）。ペアは (自車, 先行車)
#################################################################################

CAR, TRUCK = 1, 2
PAIRS = ((1, 1), (1, 2), (2, 2), (2, 1))

#OV関数の一般形の既定値（x は先行車との隙間 (m)）
#  V(x) = v0 · [tanh(x/l0 − c) + tanh(c)]  → x=0 で V=0、最高速度は v0·(1 + tanh(c))
#  無次元形 tanh(x−2)+tanh(2) に、長さの尺度 l0 を入れたもの。l0 は仮置き
OV_SHAPE = {'l0': 12.5, 'c': 2.0}


#################################################################################
# 1: 名目速度関数の定義
#   名目速度関数は dict で表す。nominal['kind'] が 'paper' か 'ov' かで計算式を切り替える
#     h_eq(nominal, i, j, v)       : 速度 v (km/h) での平衡車間 (m)
#     v_eq(nominal, i, j, h)       : 車間 h (m) での平衡速度 (km/h)  ＝ OVの V_ij(h)
#     v_range(nominal, i, j)       : h_eq が有限で単調に定義される速度範囲 (v_lo, v_hi)
#     u_nominal(nominal, i, j, rho): 名目速度関数 u_ij(ρ) = V_ij(1/ρ)
#     R_eq(nominal, i, j, v)       : 速度 v で走るときの密度 R_ij(v) = 1/h_ij(v)
#################################################################################
#モデルの定義
def make_paper_nominal(car_params, truck_params, a2, b1, margin=1e-3):
    """
    論文形の名目速度関数。scaling pramを使った4種類分の各パラメータ
    u_ij(ρ) = u_i(ρ / a_j)
    h_ij(v) = 1 / (a_j · R_i(v))。
    """
    return {
        'kind': 'paper',
        'ps': make_param_set(car_params, truck_params, a2, b1),
        'scale': {(1, 1): 1.0, (1, 2): float(a2), (2, 1): float(b1), (2, 2): 1.0},
        'margin': margin,
    }


def make_ov_nominal(vmax=None, length=None, shape=None, form='gap', scale=None):
    """
    OV形の名目速度関数。クラス i の最適速度関数は一般形
        V_i(x) = v0_i · [tanh(x/l0 − c) + tanh(c)]   (m/s、負になる所は0で打ち切り)
    で、形 (l0, c) は全クラス共通、最高速度だけクラスごとに変える
    （v0_i = vmax_i / (1 + tanh(c))）。既定値の vmax, length, l0 ,cは仮置き。

    先行車への依存のさせ方 form: 
      'gap'   : x = h − ℓ_j（速度は先行車との隙間で決まる。隙間0で停止）。
                運転の仕方は先行車によらず、車長の差だけが C-T と C-C の車間差を作る。
                → 仮定1・仮定2 をどちらも満たさない（②b, ②b-2）
      'scale' : x = a_ij · h − ℓ_i。V_ij(h) = V_ii(a_ij · h)
                → スケーリング（仮定2）は満たすが形（仮定1）は満たさない（②b-1）
        2通り作ってどちらかを実験内容によって選択する。
    """
    if form not in ('gap', 'scale'):
        raise ValueError("form は 'gap' か 'scale' を指定してください")
    if form == 'scale' and scale is None:
        raise ValueError("form='scale' のときは scale={(1,2): a2, (2,1): b1} を渡してください")
    shape = dict(OV_SHAPE if shape is None else shape)#shape はcとl0の値を意味する
    vmax = {CAR: 120.0, TRUCK: 90.0} if vmax is None else vmax#値は仮置き
    return {
        'kind': 'ov',
        'shape': shape,
        'vmax': {k: float(v) for k, v in vmax.items()},                     # km/h　確実に小数で扱うため
        'length': {CAR: 4.5, TRUCK: 12.0} if length is None else dict(length),  # m
        'form': form,
        'scale': {(1, 1): 1.0, (2, 2): 1.0, **(scale or {})},
    }

#名目速度関数を扱うための関数
def h_eq(nominal, i, j, v):
    """
        速度v→車間h 速度vで走るのに必要な車間hを計算
        kindでモデルを分類し、それぞれに合った関数を呼び出す
    """
    if nominal['kind'] == 'paper':
        return _h_eq_paper(nominal, i, j, v)
    return _h_eq_ov(nominal, i, j, v)


def v_eq(nominal, i, j, h):
    """
        車間h→速度v 車間hの時の定常速度
        kindでモデルを分類し、それぞれに合った関数を呼び出す
    """
    if nominal['kind'] == 'paper':
        return _v_eq_paper(nominal, i, j, h)
    return _v_eq_ov(nominal, i, j, h)


def v_range(nominal, i, j):
    """
        速度の上限下限を計算
    """
    if nominal['kind'] == 'paper':
        return _v_range_paper(nominal, i, j)
    return _v_range_ov(nominal, i, j)


def u_nominal(nominal, i, j, rho):
    """
        名目速度関数：速度→密度
        h = 1000/ρ
    """
    rho = np.asarray(rho, dtype=float)#配列にそろえる
    with np.errstate(divide='ignore'):#rho=0の時にinfが返ってきても計算が止まらないように
        return v_eq(nominal, i, j, 1000.0 / rho)#密度(veh/km)を車間(m)に変換し、v_eqへ投げる

def R_eq(nominal, i, j, v):
    """
        速度→密度
    """    
    return 1000.0 / h_eq(nominal, i, j, v)#車間(m)を密度(veh/km)に変える


#----- 論文形の名目速度関数 -----
def _R_own_paper(nominal, i, v):
    """
    速度→密度を求める関数
    """
    if i == CAR:
        c = nominal['ps']['car']
        return logistic_density_speed(v, c['ub'], c['uf'], c['rhoc'], c['theta1'], c['theta2'])
    t = nominal['ps']['truck']
    return underwood_density_speed(v, t['uf'], t['rhoc'])


def _u_own_paper(nominal, i, rho):
    """
    密度→速度を求める関数
    """
    if i == CAR:
        c = nominal['ps']['car']
        return logistic_speed_density(rho, c['ub'], c['uf'], c['rhoc'], c['theta1'], c['theta2'])
    t = nominal['ps']['truck']
    return underwood_speed_density(rho, t['uf'], t['rhoc'])


def _h_eq_paper(nominal, i, j, v):
    """
    速度→車間を求める関数 u_ij⁻¹(v) = a_ij *u_i⁻¹(v)
    1000/ijペアのスケーリングパラメータ*速度vで走るときの密度
    """
    return 1000.0 / (nominal['scale'][(i, j)] * _R_own_paper(nominal, i, np.asarray(v, dtype=float)))


def _v_eq_paper(nominal, i, j, h):
    """
    車間→速度を求める関数 u_ij(ρ) = u_i(ρ/a_ij)
    密度(veh/km)=1000/車間(m)
    """
    rho = 1000.0 / np.asarray(h, dtype=float)
    return _u_own_paper(nominal, i, rho / nominal['scale'][(i, j)])


def _v_range_paper(nominal, i, j):
    """
    計算できる範囲 端からmargineだけ内側に設定
    """
    #logistic: ub < v < f(0)（ufは密度→-∞の極限で到達しない）。ub以下は逆関数がクリップされる
    #Underwood: 0 < v < uf（密度→∞でも速度0にならない）
    margin = nominal['margin']
    if i == CAR:
        c = nominal['ps']['car']
        top = logistic_speed_density(0.0, c['ub'], c['uf'], c['rhoc'], c['theta1'], c['theta2'])
        return c['ub'] + margin, top - margin
    return margin, nominal['ps']['truck']['uf'] - margin


#----- OV形の名目速度関数 -----
def _v0(nominal, i):
    """
    最高速度を式の係数v0に変換 
    xが十分に大きいとtanhは1に近づくため速度の上限は v0 * (1 + tanh c)
    速度の上限 v0 * (1 + tanh c)がv_maxに一致するようv0を逆算
    """
    return nominal['vmax'][i] / 3.6 / (1 + np.tanh(nominal['shape']['c']))   # m/s(3.6で割ることでkm/hをm/hに変換)


#車間 h → OV関数の引数 x
def _x_of_h(nominal, i, j, h):
    """
    車間h→式の入力x gap(x=h-l_j) scale(a_ij*h-l_i)
    """
    if nominal['form'] == 'gap':
        return h - nominal['length'][j]
    return nominal['scale'][(i, j)] * h - nominal['length'][i]


def _h_of_x(nominal, i, j, x):
    """
    式の入力x→車間h gap(x=h-l_j) scale(x=a_ij*h-l_i)
    """
    if nominal['form'] == 'gap':
        return x + nominal['length'][j]
    return (x + nominal['length'][i]) / nominal['scale'][(i, j)]


def _v_eq_ov(nominal, i, j, h):
    """
    車間→速度 x = h − ℓ_j
    """
    l0, c = nominal['shape']['l0'], nominal['shape']['c']
    x = _x_of_h(nominal, i, j, np.asarray(h, dtype=float))
    v = _v0(nominal, i) * (np.tanh(x / l0 - c) + np.tanh(c))
    return np.maximum(v, 0.0) * 3.6#負の速度を防ぐ


def _h_eq_ov(nominal, i, j, v):
    """
    速度→車間
    """
    l0, c = nominal['shape']['l0'], nominal['shape']['c']
    w = np.asarray(v, dtype=float) / 3.6 / _v0(nominal, i) - np.tanh(c)   # = tanh(x/l0 − c)
    x = l0 * (np.arctanh(np.clip(w, -1 + 1e-15, 1 - 1e-15)) + c)
    return _h_of_x(nominal, i, j, x)


def _v_range_ov(nominal, i, j):
    """
    計算可能な範囲
    """
    #v=0（渋滞で停止）から最高速度の手前まで
    return 0.0, nominal['vmax'][i] * (1 - 1e-9)


#################################################################################
# 2: 円環の設定
#################################################################################
def pair_counts(n1, n2, m=None):
    """
    1車線の円環でのペア数 {(i,j): n_ij} を返す。
    m: トラックの塊の数（＝C-Tペア数＝T-Cペア数）。
       None ならランダム混合 m = n1·n2/N（連続値。Eq.7が暗に置いている仮定）。
       整数で与える場合は 1 ≤ m ≤ min(n1, n2)（片方が0台なら m=0）。
    n1, n2 も連続値（＝単位長さあたりの台数）でよい。
    """
    n1, n2 = float(n1), float(n2)
    N = n1 + n2
    if m is None:
        m = n1 * n2 / N if N > 0 else 0.0
    elif n1 > 0 and n2 > 0:
        if not (1 <= m <= min(n1, n2)):
            raise ValueError(f"m は 1〜min(n1, n2)={min(n1, n2):g} の範囲で指定してください (m={m})")
    elif m != 0:
        raise ValueError("片方のクラスが0台のときは m=0 です")
    return {(1, 1): n1 - m, (1, 2): float(m), (2, 2): n2 - m, (2, 1): float(m)}


def m_range(n1, n2):
    """
    その台数でとりうるmの一覧
    整数の並びとして取り得る m の一覧（隊列 m=1 〜 交互 m=min(n1,n2)）
    """
    if n1 == 0 or n2 == 0:
        return [0]
    return list(range(1, int(min(n1, n2)) + 1))


#################################################################################
# 3: 定常状態の計算
#################################################################################
def lane_speed(nominal, counts, L):
    """
    1車線の円環の定常速度 v (km/h)：Σ_ij n_ij · h_ij(v) = L を Brent法で解く。
    counts: pair_counts の戻り値、L: 周長 (m)
    詰め込みすぎ（最も遅い速度でも車間の合計が L を超える）なら NaN。
    速度の上限でも車間の合計が L に届かないときは上限の速度を返す（at_speed_cap で判定できる）。
    """
    used = [p for p in PAIRS if counts[p] > 0]#実在するペアだけ残す　存在しないペアの最小値で速度の制限を行わないようにするため
    if not used:
        return np.nan
    lo = max(v_range(nominal, *p)[0] for p in used)#すべての下限の中の最大値
    hi = min(v_range(nominal, *p)[1] for p in used)#上限のなかの最小値
    f = lambda v: sum(counts[p] * float(h_eq(nominal, *p, v)) for p in used) - L#全車間の合計-周長　f=0が定常速度
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:#lo と　hiがf=0をはさんでほしい
        return np.nan            # 密度が高すぎて解なし　
    if f_hi < 0:
        return hi                # 速度の上限まで来ても車間が足りない（実質自由流の上限）
    return brentq(f, lo, hi, xtol=1e-12)#ブレント法で解く


def at_speed_cap(nominal, rho1, rho2, v):
    """
    速度 v が v_range の上限に張り付いているか。
    OV形では tanh が最高速度に漸近するので、上限付近では車間が変わっても速度がほぼ変わらない。
    ここでの p*, s は上限の取り方で値が変わる（数値的に決まらない）ので、結果として使わない。
    """
    used = [i for i, r in ((CAR, rho1), (TRUCK, rho2)) if r > 0]#rho>0の物だけ
    return v >= min(v_range(nominal, i, i)[1] for i in used)#usedの中の最大値のうち最小の値に達しているか


def ring_state(nominal, lanes, L):
    """
    K車線の円環（車線変更なし、各車線は独立した1車線の円環）の定常状態。
    lanes: 車線ごとの (n1, n2, m) のリスト。m=None はランダム混合
    戻り値: dict(rho1, rho2 [veh/km/lane], u1, u2 [クラス平均速度 km/h], lane_v [車線ごとの速度])
    クラス平均速度は台数で重み付けした空間平均速度（スナップショットの ũ_i に対応）。
    """
    K = len(lanes)#車線数
    lane_v, n1_tot, n2_tot, s1, s2 = [], 0.0, 0.0, 0.0, 0.0#初期化
    for n1, n2, m in lanes:#車線ごとに計算
        v = lane_speed(nominal, pair_counts(n1, n2, m), L)#定常速度
        lane_v.append(v)
        n1_tot += n1#乗用車の総台数　平均密度のため
        n2_tot += n2#トラックの総台数
        s1 += n1 * v#乗用車の台数*速度の合計　平均速度のため
        s2 += n2 * v#トラックの台数＊速度の合計
    L_km = K * L / 1000.0#L(m)をkmに直す
    return {
        'rho1': n1_tot / L_km, 'rho2': n2_tot / L_km,#密度
        'u1': s1 / n1_tot if n1_tot > 0 else np.nan,#乗用車の平均速度
        'u2': s2 / n2_tot if n2_tot > 0 else np.nan,#トラックの平均速度
        'lane_v': np.array(lane_v),
    }


def true_1pipe_speed(nominal, rho1, rho2, m_per_km=None):
    """
    混走1車線（1-pipe）の正解速度 v_true。密度で指定する版（周長1 km の円環として解く）。
    m_per_km: 1 kmあたりのトラックの塊の数。None ならランダム混合（論文の仮定）。
    """
    if m_per_km is None:
        counts = pair_counts(rho1, rho2, None)
    else:
        counts = _counts_continuous(rho1, rho2, m_per_km)

    return lane_speed(nominal, counts, 1000.0)

def _counts_continuous(rho1, rho2, m):
    #密度で m を直接与える版。範囲チェックは 0 ≤ m ≤ min(ρ1, ρ2)
    if not (0 <= m <= min(rho1, rho2) + 1e-12):
        raise ValueError("m は 0〜min(rho1, rho2) の範囲で指定してください")
    return {(1, 1): rho1 - m, (1, 2): m, (2, 2): rho2 - m, (2, 1): m}


def separated_speeds(nominal, rho1, rho2, p2):
    """
    道路シェアを指定したときの2-pip速度。各車両は同車種しか追従しない
    p2 = トラックの道路シェア
    p1=1-p2
    """
    if rho1 > 0:
        rho1_lane = rho1 / (1 - p2)
        u1 = float(u_nominal(nominal, CAR, CAR, rho1_lane))
    else:
        u1 = np.nan

    if rho2 > 0:
        rho2_lane = rho2 / p2
        u2 = float(u_nominal(nominal, TRUCK, TRUCK, rho2_lane))
    else:
        u2 = np.nan

    return u1, u2


def surplus_at(nominal, rho1, rho2, u):
    """
    速度 u を与えたときの
    最小道路シェア p_i* = ρ_i / R_ii(u) （分離して速度uで走るのに必要な道幅）
    協力余剰 s = 1 − p1* − p2*（Eq.4, Def.1）
    u に v_true を入れれば正解の s、Eq.7 の u* を入れればアルゴリズムの s になる。
    nominal にはアルゴリズム側の名目速度関数（フィット結果の論文形の名目速度関数）を渡してもよい。
    """
    u = np.asarray(u, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        # 乗用車に必要な車線の割合
        if rho1 > 0:
            R1 = R_eq(nominal, CAR, CAR, u)#速度uで走るのに必要な密度
            p1 =rho1 / R1#速度uを保つのに必要な車線割合
        else:
            p1 = 0.0 * u#uが配列の時も結果を配列にするためuをかける

        # トラックに必要な車線の割合
        if rho2 > 0:
            R2 = R_eq(nominal, TRUCK, TRUCK, u)
            p2 = rho2 / R2
        else:
            p2 = 0.0 * u

    s = 1.0 - p1 - p2
    return {'p1_star': p1, 'p2_star': p2, 's': s}


def separation_speed(nominal, rho1, rho2):
    """
    u_sep: 両クラスが分離して同じ速度で走ると道路をちょうど使い切る速度
    （ρ1/R11(u) + ρ2/R22(u) = 1、すなわち s(u) = 0 となる u）。
    """
    used = [i for i, r in ((CAR, rho1), (TRUCK, rho2)) if r > 0]
    lo = max(v_range(nominal, i, i)[0] for i in used)
    hi = min(v_range(nominal, i, i)[1] for i in used)
    #R_ii(u) は u について単調減少 → s(u) は単調減少 → f = −s は単調増加
    f = lambda u: -float(surplus_at(nominal, rho1, rho2, u)['s'])
    if f(lo) > 0:
        return np.nan
    if f(hi) < 0:
        return hi                # 低密度で、遅い方のクラスの最高速度でも道路が余る
    return brentq(f, lo, hi, xtol=1e-12)


def eq7_speed(algo_nominal, rho1, rho2, scaling='printed'):
    """
    アルゴリズム側の Eq.7 の u*（印刷版 'printed' / 反転版 'inverted'）。
    algo_nominal: 論文形の名目速度関数（真の名目速度関数が論文形ならそれ自身、OVならフィット結果）
    既存の src.sensitivity.solve_1pipe_speed_vec をそのまま呼ぶ。
    """
    return solve_1pipe_speed_vec(rho1, rho2, algo_nominal['ps'], scaling)
