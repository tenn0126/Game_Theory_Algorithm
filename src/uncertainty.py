#不確かさ解析: s の符号はデータのばらつき（推定誤差）で反転するか
#
# 方針（A＋D の組み合わせ）
#   1. フィットだけをブートストラップ（車両単位・追従タイプごと）して、自由パラメータの ± と連動(共分散)を求める
#   2. 方法A（誤差伝播・数値微分）で各 (rho1, rho2) の s の ± と、どのパラメータ由来かの内訳を出す
#   3. 1 の各パラメータで s を計算し直す全体ブートストラップ（方法D）で、2 の結果と答え合わせする
#
# 引き直すのはフィット用の追従データ(df_final由来)だけ。マクロデータは s を「どこで読むか」なので固定する。
#
# フィットの設定は notebook の strong_constraint と同じ:
#   車(C-C, mode 1): ub, uf, rhoc, theta1 を論文値に固定、theta2 だけ推定
#   トラック(T-T, mode 4): uf を論文値に固定、rhoc だけ推定
#   a2(C-T, mode 2), b1(T-C, mode 3): Eq.20 の形 f(ρ/a) で推定（探索範囲 0.1〜5）
# 自由パラメータはどれも1個ずつなので、ISRES の代わりに「粗い格子 → 黄金分割」の1次元探索で解く。
# ISRES は乱数を使うので、ブートストラップに最適化のばらつきが混ざるのを防ぐ目的もある。
# （check_reference_fit で、元のISRESの結果と一致することを確認してから使う）
#
# 単位: 密度 veh/km/lane、速度 km/h（パイプライン全体と同じ）
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nominal_speed_func import logistic_speed_density, underwood_speed_density
from src.sensitivity import make_param_set, compute_surplus_vec

FT_S_TO_KMH = 0.3048 * 3.6

# 自由パラメータの名前（この順番でベクトル・共分散行列を作る）
FREE_PARAMS = ('car_theta2', 'truck_rhoc', 'a2', 'b1')
MODE_NAME = {1: 'C-C', 2: 'C-T', 3: 'T-C', 4: 'T-T'}


def default_config():
    """
    固定値と探索範囲。notebook の results_*_strong_constraint を作ったときと同じ設定。
    固定値を変えたシナリオ（例: 論文値に固定しない）は、このdictを書き換えて渡す。
    """
    return {
        'car_fixed': {'ub': 12.762, 'uf': 118.367, 'rhoc': 12.676, 'theta1': 4.995},
        'truck_fixed': {'uf': 68.478},
        'bounds': {
            'car_theta2': (0.001, 100.0),
            'truck_rhoc': (1.0, 100.0),
            'a2': (0.1, 5.0),
            'b1': (0.1, 5.0),
        },
    }


#################################################################################
# フィット用データの準備
def prepare_fit_data(df_mdensity, unit_cols=('time_period', 'Vehicle_ID'),
                     density_col='mdensity_veh_km', speed_col='v_Vel'):
    """
    compute_micro_density の返り値から、追従タイプ(mode_id)ごとに
      x: 密度, y: 速度(km/h), unit: 引き直しの単位(追従車)の番号 0..n_units-1
    を作る。
    unit_cols: 引き直しの単位。Vehicle_ID は time_period をまたいで使い回されるので組で区別する。
               同じ車の0.1秒ごとの行はまとめて入れる・抜く（行単位で引くと ± が小さく出すぎる）。
    """
    fd = {}
    for m in (1, 2, 3, 4):
        d = df_mdensity.loc[df_mdensity['mode_id'] == m, [density_col, speed_col, *unit_cols]].dropna()#各モード毎に必要な列を持ってくる
        unit = d.groupby(list(unit_cols), sort=False).ngroup().values#車両ごとでグループ化し番号を振る　同一運転手のデータは似やすいので一緒に扱う
        fd[m] = {
            'x': d[density_col].values.astype(float),
            'y': d[speed_col].values.astype(float) * FT_S_TO_KMH,
            'unit': unit,
            'n_units': int(unit.max()) + 1,
        }
    print('フィット用データ（引き直しの単位 = ' + ' × '.join(unit_cols) + '）')
    for m, v in fd.items():
        print(f"  mode {m} ({MODE_NAME[m]}): {len(v['x']):>8,} 行, {v['n_units']:>5,} 台")
    return fd


def draw_weights(fd, rng):
    """
    ブートストラップ1回分の行の重みを作る。追従タイプごとに、その中の台数と同じ台数を重複ありで引く
    （層別ブートストラップ。T-T のように台数が少ないタイプが丸ごと消えるのを防ぐ）。
    重み = その行の車が何回引かれたか。0回なら重み0（その車の行は使わない）。
    """
    w = {}
    for m, v in fd.items():
        n = v['n_units']
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)#0以上n(mode=aの台数)未満の整数を台数分ランダムに引く　何回ひかれたかを数える
        w[m] = counts[v['unit']].astype(float)#
    return w


#################################################################################
# 1次元の最小化（粗い格子で谷を見つけてから黄金分割で詰める）
def _argmin_1d(obj, lo, hi, n_grid=30, n_golden=40):
    """
    L1誤差は谷が複数あることがあるので、まず対数等間隔の格子で一番低い点を探し、
    その両隣の区間を黄金分割で詰める。戻り値: (最小点, 最小値)
    """
    grid = np.geomspace(lo, hi, n_grid)#lo からhiまでをngrid個に分ける　等対数間隔で
    vals = np.array([obj(g) for g in grid])#n_grid個分の目的関数を計算し配列にする
    i = int(np.argmin(vals))#最小値が何番目かを返す
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]#最小値の左隣をa、右隣をbにし、本当の最小値はこの間に存在する
    gr = (math.sqrt(5) - 1) / 2#黄金比に関係する値　0.618
    c, d = b - gr * (b - a), a + gr * (b - a)#区間abの間に調べる点を二つ置く
    fc, fd_ = obj(c), obj(d)#目的関数の値
    for _ in range(n_golden):
        if fc < fd_:
            b, d, fd_ = d, c, fc#右はしを
            c = b - gr * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd_
            d = a + gr * (b - a)
            fd_ = obj(d)
    x = 0.5 * (a + b)#十分に狭めた中央を最終的な答えとする
    return x, obj(x)


#################################################################################
# 4つの自由パラメータを順番に推定（Step 1 と同じ順番: 名目速度関数 → スケーリング）
def fit_all(fd, cfg=None, weights=None):
    """
    weights: draw_weights の戻り値。None なら全行の重み1（元データでの推定）。
    戻り値: dict(car_theta2, truck_rhoc, a2, b1)
    """
    cfg = cfg or default_config()
    car_fx, trk_fx, bnd = cfg['car_fixed'], cfg['truck_fixed'], cfg['bounds']
    W = weights or {m: np.ones(len(v['x'])) for m, v in fd.items()}#重みがなければ全行１で計算

    # (1) 車 C-C: theta2 だけ動くので、(1+exp(...)) の対数を先に計算しておくと1回の評価がexp1回で済む
    x, y, w = fd[1]['x'], fd[1]['y'], W[1]#mode=1
    log1pe = np.log1p(np.exp(np.clip((x - car_fx['rhoc']) / car_fx['theta1'], -100, 100)))#theta2を変えても変わらない値を先に計算
    span = car_fx['uf'] - car_fx['ub']
    theta2, _ = _argmin_1d(#目的関数の損失が最小になるthetaを探索
        lambda t: np.sum(w * np.abs(y - (car_fx['ub'] + span * np.exp(-t * log1pe)))),
        *bnd['car_theta2'])
    car = {**car_fx, 'theta2': theta2}#使ったtheta2を追加

    # (2) トラック T-T: rhoc だけ推定
    x, y, w = fd[4]['x'], fd[4]['y'], W[4]
    rhoc, _ = _argmin_1d(
        lambda r: np.sum(w * np.abs(y - underwood_speed_density(x, trk_fx['uf'], r))),
        *bnd['truck_rhoc'])
    truck = {**trk_fx, 'rhoc': rhoc}

    # (3) a2: C-T のデータに f(ρ/a2)（同じ回で推定した車の関数を使う）
    x, y, w = fd[2]['x'], fd[2]['y'], W[2]
    a2, _ = _argmin_1d(
        lambda a: np.sum(w * np.abs(y - logistic_speed_density(x / a, car['ub'], car['uf'], car['rhoc'],
                                                               car['theta1'], car['theta2']))),
        *bnd['a2'])

    # (4) b1: T-C のデータに g(ρ/b1)
    x, y, w = fd[3]['x'], fd[3]['y'], W[3]
    b1, _ = _argmin_1d(
        lambda b: np.sum(w * np.abs(y - underwood_speed_density(x / b, truck['uf'], truck['rhoc']))),
        *bnd['b1'])

    return {'car_theta2': theta2, 'truck_rhoc': rhoc, 'a2': a2, 'b1': b1}#推定されたパラメータ


def check_reference_fit(fd, reference, cfg=None, test_size=0.3, random_state=42):
    """
    元のISRESの推定（notebookの results_*_strong_constraint）と一致するかの確認。
    元の推定は7:3分割の学習データ(7割)だけを使っているので、同じ分割を再現した重みでも推定する。
    reference: dict(car_theta2, truck_rhoc, a2, b1)
    """
    from sklearn.model_selection import train_test_split

    w_train = {}
    for m, v in fd.items():
        idx = np.arange(len(v['x']))
        tr, _ = train_test_split(idx, test_size=test_size, random_state=random_state)
        w = np.zeros(len(idx))
        w[tr] = 1.0
        w_train[m] = w
    full = fit_all(fd, cfg)
    train = fit_all(fd, cfg, w_train)
    return pd.DataFrame({'ISRES (元の推定)': reference, '今回の方法・学習7割': train, '今回の方法・全データ': full}).loc[list(FREE_PARAMS)]


#################################################################################
# 手順1: フィットのブートストラップ
def bootstrap_fit(fd, B=200, seed=0, cfg=None, verbose=True):
    """
    B回引き直して fit_all をやり直す。戻り値: DataFrame（1行=1回、列=FREE_PARAMS + 探索範囲の端に張り付いたか）
    """
    cfg = cfg or default_config()
    rng = np.random.default_rng(seed)#再現のため
    rows = []#結果をためておくリスト
    for b in range(B):#b回繰り返す
        est = fit_all(fd, cfg, draw_weights(fd, rng))#重みを引き、パラメータを推定し直す
        for k in FREE_PARAMS:#4つのパラメータを順にみていく
            lo, hi = cfg['bounds'][k]
            # 端から相対0.5%以内なら「張り付いた」とみなす
            est[f'at_bound_{k}'] = bool(est[k] <= lo * 1.005 or est[k] >= hi * 0.995)
        rows.append(est)#結果をリストに追加
        if verbose and (b + 1) % max(B // 10, 1) == 0:#進み具合を表示
            print(f'  {b + 1}/{B}')
    return pd.DataFrame(rows)#辞書のリストをpandasの表にして返す


def summarize_boot_params(df_boot, ref):
    """パラメータごとの ± と、端に張り付いた割合"""
    P = df_boot[list(FREE_PARAMS)]#パラメータの列だけ取り出す
    out = pd.DataFrame({#辞書から新しい表を作成
        'ref': pd.Series(ref)[list(FREE_PARAMS)],
        'boot_mean': P.mean(),#b回分の推定値の列ごと（パラメータごと）の平均
        'sd (±)': P.std(ddof=1),#推定値の標準偏差
        'q2.5': P.quantile(0.025),#下から2.5%の位置の値
        'q97.5': P.quantile(0.975),#下から97.5%の位置の値
        'at_bound_rate': [df_boot[f'at_bound_{k}'].mean() for k in FREE_PARAMS],#張り付いた回数
    })
    out['sd/ref'] = out['sd (±)'] / out['ref']#相対的な+-
    return out


#################################################################################
# パラメータ → s
def to_param_set(p, cfg=None):
    """自由パラメータのdict(またはSeries)と固定値から sensitivity.make_param_set の形を作る"""
    cfg = cfg or default_config()
    car = {**cfg['car_fixed'], 'theta2': float(p['car_theta2'])}#固定値と自由値を合わせる
    truck = {**cfg['truck_fixed'], 'rhoc': float(p['truck_rhoc'])}
    return make_param_set(car, truck, p['a2'], p['b1'])


def s_of(p, rho1, rho2, scaling, cfg=None):#sを計算
    return compute_surplus_vec(rho1, rho2, to_param_set(p, cfg), scaling)['s']


def make_grid(rho1_max=80.0, rho2_max=8.0, n1=80, n2=40):
    """
    s を評価する (rho1, rho2) の格子。行がrho2、列がrho1（sensitivity.surplus_field と同じ向き）
    片方の密度が0だと s は恒等的に0（符号が決まらない）なので、0は含めない。
    """
    r1 = np.linspace(rho1_max / n1, rho1_max, n1)
    r2 = np.linspace(rho2_max / n2, rho2_max, n2)
    R1, R2 = np.meshgrid(r1, r2)
    return {'rho1': r1, 'rho2': r2, 'R1': R1, 'R2': R2}


def _norm_cdf(z):#P(s>0)を計算する　分布のうちs＞０の面積を計算する
    z = np.asarray(z, dtype=float)
    out = np.full(z.shape, np.nan)#結果をここに書きこむよう
    ok = np.isfinite(z)#infやnanでないところはtrue
    out[ok] = 0.5 * (1.0 + np.vectorize(math.erf)(z[ok] / math.sqrt(2.0)))
    return out


#################################################################################
# 手順2: 方法A（誤差伝播）
def delta_method(ref, cov, rho1, rho2, scaling, cfg=None, rel_step=1e-3):
    """
    s の ± を「感度 × パラメータのズレ」で計算する。
      感度 g_k = ∂s/∂p_k を中心差分で数値微分（p_k を ±rel_step×p_k だけ動かす）
      Var(s) ≈ g^T Σ g      （Σ = ブートストラップで求めたパラメータの共分散行列。連動も含む）
      内訳 share_k = g_k (Σ g)_k / Var(s)   （合計すると1。連動の分は関係するパラメータに振り分けられる。
                                            負の値は「そのパラメータとの連動で打ち消し合っている」意味）
      P(s>0) ≈ Φ(s / sd)   （s が正規分布に従うと仮定したとき）
    ref: 基準のパラメータ(dict)、cov: FREE_PARAMS の順の共分散行列 (k×k)
    """
    cov = np.asarray(cov, dtype=float)
    p0 = {k: float(ref[k]) for k in FREE_PARAMS}
    s0 = s_of(p0, rho1, rho2, scaling, cfg)
    grads = []
    for k in FREE_PARAMS:
        h = rel_step * abs(p0[k])
        s_up = s_of({**p0, k: p0[k] + h}, rho1, rho2, scaling, cfg)
        s_dn = s_of({**p0, k: p0[k] - h}, rho1, rho2, scaling, cfg)
        grads.append((s_up - s_dn) / (2 * h))
    G = np.stack(grads)                                  # (k, ...)
    SG = np.tensordot(cov, G, axes=(1, 0))               # (Σ g) の各成分
    var = np.sum(G * SG, axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        share = G * SG / var
        z = s0 / sd
    # 参考: 連動を無視して「二乗して足してルート」で合わせた場合の ±
    sd_indep = np.sqrt(np.tensordot(np.diag(cov), G ** 2, axes=(0, 0)))
    return {'s': s0, 'sd': sd, 'sd_indep': sd_indep, 'grad': G, 'share': share, 'p_pos': _norm_cdf(z)}


#################################################################################
# 手順3: 方法D（全体のブートストラップ）
def bootstrap_surplus(df_boot, rho1, rho2, scaling, cfg=None):
    """ブートストラップの各回のパラメータで s を計算し直す。戻り値: shape (B, *rho1.shape)"""
    return np.stack([s_of(row, rho1, rho2, scaling, cfg) for _, row in df_boot[list(FREE_PARAMS)].iterrows()])


def summarize_boot_surplus(S):
    """
    S: bootstrap_surplus の戻り値。
    P(s>0) は解けた回(NaNでない回)の中での割合。
    """
    ok = np.isfinite(S)
    n_ok = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        p_pos = np.where(n_ok > 0, np.sum((S > 0) & ok, axis=0) / n_ok, np.nan)
    return {
        'p_pos': p_pos,
        'sd': np.nanstd(S, axis=0, ddof=1),
        'q025': np.nanquantile(S, 0.025, axis=0),
        'q975': np.nanquantile(S, 0.975, axis=0),
        'median': np.nanmedian(S, axis=0),
        'solved_rate': n_ok / S.shape[0],
    }


def sign_class(p_pos, level=0.95):
    """P(s>0) から 'robust +' / 'robust -' / 'ambiguous' に分ける（level=0.95 なら 0.95以上 / 0.05以下）"""
    p = np.asarray(p_pos, dtype=float)
    out = np.full(p.shape, 'undefined', dtype=object)
    ok = np.isfinite(p)
    out[ok] = 'ambiguous'
    out[ok & (p >= level)] = 'robust +'
    out[ok & (p <= 1 - level)] = 'robust -'
    return out


#################################################################################
# 図
def plot_p_pos_map(grid, p_pos, ax=None, title=None, states=None):
    """P(s>0) の地図。0.05 / 0.5 / 0.95 の等高線つき（境界の"帯"の太さを見る）"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    m = ax.pcolormesh(grid['rho1'], grid['rho2'], p_pos, cmap='RdBu', vmin=0, vmax=1, shading='auto')
    cs = ax.contour(grid['rho1'], grid['rho2'], p_pos, levels=[0.05, 0.5, 0.95],
                    colors=['k', 'k', 'k'], linestyles=[':', '-', ':'], linewidths=1)
    ax.clabel(cs, fmt='%.2f', fontsize=7)
    if states is not None:
        ax.scatter(states['rho1'], states['rho2'], s=2, c='gray', alpha=0.3, lw=0, label='observed')
    plt.colorbar(m, ax=ax, label='P(s>0)')
    ax.set_xlabel('rho1 car (veh/km/lane)')
    ax.set_ylabel('rho2 truck (veh/km/lane)')
    ax.set_title(title or 'P(s>0)')
    return ax


def plot_share_maps(grid, share, title=None):
    """方法Aの内訳: s の分散のうち各パラメータ由来の割合"""
    fig, axs = plt.subplots(1, len(FREE_PARAMS), figsize=(4.2 * len(FREE_PARAMS), 3.6), sharey=True)
    for ax, k, sh in zip(axs, FREE_PARAMS, share):
        m = ax.pcolormesh(grid['rho1'], grid['rho2'], sh, cmap='PuOr', vmin=-1, vmax=1, shading='auto')
        ax.set_title(k)
        ax.set_xlabel('rho1')
    axs[0].set_ylabel('rho2')
    fig.colorbar(m, ax=axs, label='share of Var(s)')
    if title:
        fig.suptitle(title)
    return fig


def plot_compare_A_D(resA, resD, title=None):
    """方法Aと方法Dの答え合わせ: sd 同士、P(s>0) 同士の散布図"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    sa, sd_ = resA['sd'].ravel(), resD['sd'].ravel()
    ok = np.isfinite(sa) & np.isfinite(sd_)
    axs[0].scatter(sd_[ok], sa[ok], s=4, alpha=0.4)
    lim = np.nanmax([sa[ok].max(), sd_[ok].max()]) * 1.05 if ok.any() else 1
    axs[0].plot([0, lim], [0, lim], 'r--', lw=1)
    axs[0].set_xlabel('sd of s  (D: bootstrap)')
    axs[0].set_ylabel('sd of s  (A: delta method)')
    pa, pd_ = resA['p_pos'].ravel(), resD['p_pos'].ravel()
    ok = np.isfinite(pa) & np.isfinite(pd_)
    axs[1].scatter(pd_[ok], pa[ok], s=4, alpha=0.4)
    axs[1].plot([0, 1], [0, 1], 'r--', lw=1)
    axs[1].set_xlabel('P(s>0)  (D: bootstrap)')
    axs[1].set_ylabel('P(s>0)  (A: normal approx.)')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
