#感度分析①: スケーリングパラメータ (a2, b1) を格子上で動かしたときの u*, p1*, p2*, s
#
# 方針（いつものパイプラインと違う処理が混ざらないように）
#   ・データの準備(先頭・末尾の除外、横持ちへの変換)は whole_pipeline の関数をそのまま呼ぶ
#     → run_pipeline の Step 3 と全く同じスナップショット集合を使う
#   ・Eq.7の残差・名目速度関数の逆関数も whole_pipeline のものを使う
#   ・変えているのは「brentqで1行ずつ解く」部分だけ（全スナップショットを同時に二分法で解く）
#     → check_against_pipeline で、パイプラインの結果と一致することを必ず確認してから使う
#
# 単位: 密度 veh/km/lane、速度 km/h（パイプライン全体と同じ）。
#   a2, b1, p*, s は無次元なので単位系に依存しない。論文値は results_*_paper_km（換算済み）を使う。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.whole_pipeline import (
    drop_warmup, drop_cooldown,
    compute_1pipe_speed_for_snapshots, compute_min_road_share, compute_cooperation_surplus,
    _eq7_residual, _check_scaling,
    logistic_speed_density, logistic_density_speed, underwood_density_speed,
)


#################################################################################
# パイプラインのStep 3（run_pipelineと同じ順番・同じ関数）
def pipeline_step3(data, car_params, truck_params, a2, b1, scaling,
                   skip_sec_warmup=100, skip_sec_cooldown=200):
    """
    run_pipeline の Step 3 部分だけを取り出したもの（中身は同じ関数呼び出し）。
    戻り値の表（rho1, rho2, u1_obs, u2_obs, ...）を感度分析の観測状態として使い回す。
    """
    df_macro = drop_warmup(data['df_macro'], skip_sec=skip_sec_warmup)  #各時間における立ち上がりデータを削除
    df_macro = drop_cooldown(df_macro, skip_sec=skip_sec_cooldown)  #各時間における立ち下がりデータを削除
    df_1pipe = compute_1pipe_speed_for_snapshots(df_macro, car_params, truck_params,
                                                 a2, b1, scaling) #1-pipe速度の計算
    df_pstar = compute_min_road_share(df_1pipe, car_params, truck_params)#最小道路占有
    return compute_cooperation_surplus(df_pstar)#協調余剰の計算


#################################################################################
# u* → p1*, p2* → s を全スナップショット同時に計算
def solve_u_star(rho1, rho2, car_params, truck_params, a2, b1, scaling, n_iter=60, margin=1e-3):
    """
    solve_1pipe_speed（brentq版）と同じ探索区間・同じ残差関数を、二分法で配列ごと解く。
      下限 = margin
      上限 = min(車がいれば f(0)-margin, トラックがいれば uf-margin)
    区間の両端で残差の符号が同じ点（brentqならValueError→NaN）はNaNを返す。
    残差は u について単調増加（u_i^-1(u)が単調減少）なので二分法で解ける。
    """
    _check_scaling(scaling)
    rho1 = np.asarray(rho1, dtype=float)
    rho2 = np.asarray(rho2, dtype=float) #numpy の配列としてまとめて計算できるようにするらしい

    car_top = logistic_speed_density(0.0, car_params['ub'], car_params['uf'], car_params['rhoc'],#探索区間の上限を決定 u*がとりうる最高速度をクラスごとに求める
                                     car_params['theta1'], car_params['theta2']) - margin
    truck_top = truck_params['uf'] - margin
    upper = np.full(rho1.shape, np.inf) #スナップショットごとの上限　初めは全部infで埋める
    upper = np.where(rho1 > 0, np.minimum(upper, car_top), upper) #車がいるスナップショットだけ上限をcartopまで引き下げ
    upper = np.where(rho2 > 0, np.minimum(upper, truck_top), upper)#トラックいるスナップショットだけさらにtrucktopまで引き下げ
    lower = np.full(rho1.shape, margin)#下限　はmrginで埋める ほぼ０

    valid = (rho1 + rho2 > 0) & np.isfinite(upper)#解けるスナップショットかの判定　どちらもいないものは対象がいないので除外　上限がinfのものも除外
    hi = np.where(valid, upper, 1.0)# 対象外は上限がinfのままなので、適当に1.0で埋める
    lo = lower.copy()#下限の配列をコピー　二分法で中を書き換えるためlowerとは別配列

    def res(u):#残差を計算する
        with np.errstate(divide='ignore', invalid='ignore'):
            return _eq7_residual(u, rho1, rho2, car_params, truck_params, a2, b1, scaling)

    r_lo, r_hi = res(lo), res(hi)#区間で両端の符号が逆なら解があるので、符号が同じものは除外する　０だとそれ自体が解なので除外しない
    valid &= (np.sign(r_lo) != np.sign(r_hi)) | (r_lo == 0) | (r_hi == 0)

    for _ in range(n_iter):#二分法本体
        mid = 0.5 * (lo + hi) #中点を計算
        go_up = res(mid) < 0          # 残差<0 なら解はmidより上
        lo = np.where(go_up, mid, lo) #loをmidに更新するかどうかの判定　残差<0ならloをmidに更新
        hi = np.where(go_up, hi, mid) #hiをmidに更新するかどうかの判定　残差<0ならhiはmidに更新しない

    return np.where(valid, 0.5 * (lo + hi), np.nan) #最後に残った区間の真ん中をu*の解として返す。解けなかったスナップショットはNaN


#スナップショットの表を受け取り、u* → p1*, p2* → s）を一度に計算
def surplus_at(states, car_params, truck_params, a2, b1, scaling):
    """
    states（pipeline_step3の戻り値）の各スナップショットについて u*, p1*, p2*, s を返す。
    p* の式は compute_min_road_share と同じ（クラス自身の名目速度関数の逆関数。a2・b1は使わない）。
    """
    rho1, rho2 = states['rho1'].values, states['rho2'].values #密度を配列で取り出す
    u = solve_u_star(rho1, rho2, car_params, truck_params, a2, b1, scaling) #二分法でu*を解く
    p1 = rho1 / logistic_density_speed(u, car_params['ub'], car_params['uf'], car_params['rhoc'],#最小道路占有
                                       car_params['theta1'], car_params['theta2'])
    p2 = rho2 / underwood_density_speed(u, truck_params['uf'], truck_params['rhoc'])
    return {'u_1pipe': u, 'p1_star': p1, 'p2_star': p2, 's': 1.0 - p1 - p2} #協力余剰ｓを返す


#################################################################################
# パイプラインとの一致確認（感度分析の前に必ず実行する）　理論的にあってそうならなくてもいいかな
def check_against_pipeline(data, car_params, truck_params, scaling, ab_list,
                           skip_sec_warmup=100, skip_sec_cooldown=200, tol=1e-6):
    """
    ab_list の (a2, b1) それぞれについて、pipeline_step3（brentq版）と surplus_at（二分法版）の
    u*, p1*, p2*, s を比べる。一致しなければ AssertionError。
    """
    rows = []
    for a2, b1 in ab_list:
        ref = pipeline_step3(data, car_params, truck_params, a2, b1, scaling,
                             skip_sec_warmup, skip_sec_cooldown)
        new = surplus_at(ref, car_params, truck_params, a2, b1, scaling)
        row = {'a2': a2, 'b1': b1}
        for k in ['u_1pipe', 'p1_star', 'p2_star', 's']:
            r, n = ref[k].values, new[k]
            row[f'nan_mismatch_{k}'] = int((np.isnan(r) != np.isnan(n)).sum())
            both = ~np.isnan(r) & ~np.isnan(n)
            row[f'maxdiff_{k}'] = float(np.max(np.abs(r[both] - n[both]))) if both.any() else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)

    nan_cols = [c for c in out.columns if c.startswith('nan_mismatch')]
    diff_cols = [c for c in out.columns if c.startswith('maxdiff')]
    assert (out[nan_cols] == 0).all().all(), f"解けた/解けないの判定がパイプラインと違います\n{out}"
    assert (out[diff_cols] < tol).all().all(), f"値がパイプラインと一致しません\n{out}"
    print(f"パイプラインとの一致を確認 (scaling={scaling}, {len(ab_list)}組, 最大差 "
          f"{out[diff_cols].values.max():.2e})")
    return out


#################################################################################
# (a2, b1) 格子のスイープ
def log_grid(lo=0.25, hi=4.0, n=41):#調べるabの組
    """対数等間隔の格子。lo*hi=1 かつ n が奇数なら 1.0 がちょうど含まれる。"""
    return np.geomspace(lo, hi, n)#比が一定になるように点を並べる　倍の間隔が同じになるよう対数を使用1.072 倍


def sweep_scaling(states, car_params, truck_params, scaling, a2_grid, b1_grid):
    """
    各 (a2, b1) について、両クラスがいるスナップショットだけを対象に要約する。
    （片方のクラスしかいない時刻は Eq.7 から s=0 が恒等的に成り立つので除く）

    戻り値の列:
      a2, b1, n_both (両クラスがいる件数), n_solved (u*が解けた件数),
      frac_s_pos (s>0の割合), s_median, s_q10, s_q90,
      u_median (u*の中央値 km/h), p1_median, p2_median
    """
    both = ((states['rho1'] > 0) & (states['rho2'] > 0)).values #両方のクラスがいるスナップショットだけを対象にする(これは実質ゲームが発生していない組の除外では？)
    sub = states[both]
    rows = []
    for a2 in a2_grid:#格子を二重ループで回す
        for b1 in b1_grid:
            r = surplus_at(sub, car_params, truck_params, a2, b1, scaling)
            ok = ~np.isnan(r['s'])#解けた点だけ取り出す
            s = r['s'][ok]
            rows.append({#1行に要約
                'a2': a2, 'b1': b1,
                'n_both': int(both.sum()), 'n_solved': int(ok.sum()),
                'frac_s_pos': float(np.mean(s > 0)) if ok.any() else np.nan,#s>0のスナップショット割合
                's_median': float(np.median(s)) if ok.any() else np.nan,#sの中央値
                's_q10': float(np.quantile(s, 0.10)) if ok.any() else np.nan,#下から10%
                's_q90': float(np.quantile(s, 0.90)) if ok.any() else np.nan,#下から90%
                'u_median': float(np.median(r['u_1pipe'][ok])) if ok.any() else np.nan,#中央値
                'p1_median': float(np.median(r['p1_star'][ok])) if ok.any() else np.nan,#
                'p2_median': float(np.median(r['p2_star'][ok])) if ok.any() else np.nan,#
            })
    return pd.DataFrame(rows)


#################################################################################
# 可視化
def plot_sweep(df_sweep, col, ax=None, marks=None, cmap='RdBu', center=None, title=None):
    """
    (a2, b1) 平面に col の値を色で描く。軸は対数。
    marks: {'名前': (a2, b1)} の辞書（論文値・自分の推定値などに印を付ける）
    center: 色の中心にする値（例: s_median なら 0）。None なら最小〜最大
    a2=1, b1=1 の線（スケーリングなし）を点線で描く。
    RdBuは赤から白を経て青に変わる。正負を見るのに適する
    """
    #表を二次元に並べ直す　41＊41の表にする
    piv = df_sweep.pivot(index='b1', columns='a2', values=col)
    #axが渡されていなければ、新しい図を作る。zは色を付ける値のnumpy配列
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    z = piv.values

    #色の範囲を決める
    if center is not None:#centerがあるときは、一番遠い値までの距離mを求め、色の範囲をcenter +-mにする
        m = np.nanmax(np.abs(z - center))
        vmin, vmax = center - m, center + m
    else:#最大値から最小値をそのまま色の範囲にする
        vmin, vmax = np.nanmin(z), np.nanmax(z)
    
    #色を塗る
    pc = ax.pcolormesh(piv.columns, piv.index, z, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    plt.colorbar(pc, ax=ax, label=col)#横に色と値の対応を表示
    #境目の線を引く　値がちょうどcenterになる位置に黒の等高線を引く
    if center is not None and np.nanmin(z) < center < np.nanmax(z):
        ax.contour(piv.columns, piv.index, z, levels=[center], colors='k', linewidths=1.2)
    #境目を描く　スケーリングなしの場合を引くa2=b1=1の線
    ax.axvline(1.0, color='gray', ls=':', lw=1)
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    for name, (a2, b1) in (marks or {}).items():
        ax.plot(a2, b1, marker='*', ms=12, mec='k', mfc='yellow', ls='none')
        ax.annotate(name, (a2, b1), textcoords='offset points', xytext=(6, 6), fontsize=9)
    #軸を整える
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel('a2 (car→truck)')
    ax.set_ylabel('b1 (truck→car)')
    ax.set_title(title or col)
    return ax


#################################################################################
# (rho1, rho2) 平面上の s の地図（架空の密度格子）
def surplus_map(car_params, truck_params, a2, b1, scaling, rho1_max=80.0, rho2_max=20.0,
                n1=161, n2=101):
    """
    rho1 ∈ [0, rho1_max], rho2 ∈ [0, rho2_max] の格子上で u*, p1*, p2*, s を計算する。
    u*, p*, s は (rho1, rho2) とパラメータだけで決まる（観測速度を使わない）ので、
    観測されていない密度の組でも同じ式で計算できる。計算は surplus_at をそのまま使う。

    戻り値: dict(rho1, rho2 = 1次元の軸, u_1pipe, p1_star, p2_star, s = shape (n2, n1))
            行がrho2、列がrho1（pcolormesh(rho1, rho2, s) でそのまま描ける向き）
    """
    #格子を作る
    r1 = np.linspace(0.0, rho1_max, n1) #0から最大値までにn1個の点を等間隔に並べる
    r2 = np.linspace(0.0, rho2_max, n2)
    R1, R2 = np.meshgrid(r1, r2)#r1,r2を組み合わせた二次元配列を作る

    #二次元格子をrevalで一列に並べ、観測スナップショットの表と同じ形にする
    grid = pd.DataFrame({'rho1': R1.ravel(), 'rho2': R2.ravel()})

    #surplus_atでu*p1*p2*sを計算し、返り値を元のn1*n2の形に戻す
    out = {k: v.reshape(R1.shape)
           for k, v in surplus_at(grid, car_params, truck_params, a2, b1, scaling).items()}
    out['rho1'], out['rho2'] = r1, r2#1次元の軸も一緒に返す
    return out


def plot_surplus_map(m, states=None, ax=None, title=None, smax=None, u_levels=None):
    """
    surplus_map の結果を描く。
      色: s（青=s>0, 赤=s<0）  黒の太線: s=0  灰色の細線: u* の等高線(km/h)
      states を渡すと、観測スナップショット（両クラスがいるもの）を点で重ねる
      → 点の外側は名目速度関数の外挿であることに注意
    """
    #描く場所がな書ければ新たに作る
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    s = m['s']
    if smax is None:
        smax = np.nanmax(np.abs(s))
    pc = ax.pcolormesh(m['rho1'], m['rho2'], s, cmap='RdBu', vmin=-smax, vmax=smax, shading='nearest')
    plt.colorbar(pc, ax=ax, label='s')
    #s=0になる境界線を引く　sが正の時と負の時があるときだけ線を引く
    if np.nanmin(s) < 0 < np.nanmax(s):
        ax.contour(m['rho1'], m['rho2'], s, levels=[0], colors='k', linewidths=1.8)

    #u*の等高線を描く　u_levelsの指定がなければ切りのいい8段階を選ぶ
    cs = ax.contour(m['rho1'], m['rho2'], m['u_1pipe'], levels=u_levels or 8,
                    colors='gray', linewidths=0.7)
    ax.clabel(cs, fmt='%.0f', fontsize=7)
    #観測点を重ねる
    if states is not None:
        b = states[(states['rho1'] > 0) & (states['rho2'] > 0)]
        ax.scatter(b['rho1'], b['rho2'], s=2, c='k', alpha=0.25, lw=0)

    #軸を整える
    ax.set_xlabel('rho1 car (veh/km/lane)')
    ax.set_ylabel('rho2 truck (veh/km/lane)')
    ax.set_title(title or '')
    return ax


#################################################################################
# s=0 の境界を (a2, b1) 平面で理論式と比べる
#
# printed形の Eq.7 から、恒等的に
#   s = (rho1*rho2/rho_tot) * [ (1/a2 - 1)/R1(u*) + (1/b1 - 1)/R2(u*) ],  R_i = u_i^-1（クラス自身の逆関数）
# が成り立つ。よって s=0 の境界は
#   1/a2 - 1 = (1 - 1/b1) * kappa(u*),   kappa(u) = R1(u)/R2(u)（同じ速度での車とトラックの密度の比）
# Li(2022) のように基準の速度関数が共通なら kappa=1 で、境界は 1/a2 + 1/b1 = 2（組成に依存しない）。
def density_ratio(u, car_params, truck_params):
    """kappa(u) = R1(u)/R2(u): 同じ速度 u での、車の密度 / トラックの密度"""
    r1 = logistic_density_speed(u, car_params['ub'], car_params['uf'], car_params['rhoc'],
                                car_params['theta1'], car_params['theta2'])
    r2 = underwood_density_speed(u, truck_params['uf'], truck_params['rhoc'])
    return r1 / r2


def b1_on_boundary(a2, kappa):
    """1/a2 - 1 = (1 - 1/b1)*kappa を b1 について解く。解がない（1/b1<=0）ところは NaN"""
    inv_b1 = 1.0 - (1.0 / np.asarray(a2, dtype=float) - 1.0) / kappa
    with np.errstate(divide='ignore'):
        return np.where(inv_b1 > 0, 1.0 / inv_b1, np.nan)


def boundary_b1_exact(states, car_params, truck_params, a2_values, b1_lo=0.05, b1_hi=50.0, n_iter=50):
    """
    両クラスがいる各スナップショットについて、a2 ごとに s=0 となる b1 を二分法（対数軸）で求める（printed形）。
    b1 の区間の両端で s の符号が同じなら NaN。
    戻り値: dict(b1 = shape (len(a2_values), n_snap), u_star = 境界上の u*, kappa = 境界上の kappa)
    """
    sub = states[(states['rho1'] > 0) & (states['rho2'] > 0)]
    n = len(sub)
    b1_out = np.full((len(a2_values), n), np.nan)
    u_out = np.full((len(a2_values), n), np.nan)

    def s_of(a2, b1):
        return surplus_at(sub, car_params, truck_params, a2, b1, 'printed')['s']

    for i, a2 in enumerate(a2_values):
        lo, hi = np.full(n, np.log(b1_lo)), np.full(n, np.log(b1_hi))
        s_lo = s_of(a2, np.exp(lo))
        s_hi = s_of(a2, np.exp(hi))
        valid = np.isfinite(s_lo) & np.isfinite(s_hi) & (np.sign(s_lo) != np.sign(s_hi))
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            s_mid = s_of(a2, np.exp(mid))
            same = np.sign(s_mid) == np.sign(s_lo)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        b1 = np.exp(0.5 * (lo + hi))
        b1_out[i] = np.where(valid, b1, np.nan)
        u_out[i] = np.where(valid, surplus_at(sub, car_params, truck_params, a2, b1, 'printed')['u_1pipe'], np.nan)

    return {'b1': b1_out, 'u_star': u_out, 'kappa': density_ratio(u_out, car_params, truck_params)}


def plot_boundary_comparison(df_sweep, exact, a2_values, kappa_fixed, ax=None, marks=None, title=None):
    """
    (a2, b1) 平面（対数軸）に s=0 の境界を重ねる。
      黒の実線   : スイープの frac_s_pos = 0.5 の等高線（plot_sweep と同じもの）
      青の線と帯 : スナップショットごとの厳密な境界の中央値と 10〜90%
      緑の破線   : kappa を定数 kappa_fixed にした理論式
      赤の点線   : Li(2022) の共通関数の場合（kappa=1, 1/a2 + 1/b1 = 2）
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    piv = df_sweep.pivot(index='b1', columns='a2', values='frac_s_pos')
    ax.contourf(piv.columns, piv.index, piv.values, levels=[-0.01, 0.5, 1.01],
                colors=['#f4c7c3', '#c6dbef'], alpha=0.6)
    ax.contour(piv.columns, piv.index, piv.values, levels=[0.5], colors='k', linewidths=2)

    b1 = exact['b1']
    ok = np.isfinite(b1).sum(axis=1) > 0
    q10, q50, q90 = (np.full(len(a2_values), np.nan) for _ in range(3))
    q10[ok], q50[ok], q90[ok] = np.nanquantile(b1[ok], [0.1, 0.5, 0.9], axis=1)
    ax.fill_between(a2_values, q10, q90, color='tab:blue', alpha=0.25, lw=0)
    ax.plot(a2_values, q50, color='tab:blue', lw=1.5, label='exact s=0 (median, 10-90%)')
    ax.plot(a2_values, b1_on_boundary(a2_values, kappa_fixed), 'g--', lw=1.5,
            label=f'theory, kappa={kappa_fixed:.2f} fixed')
    ax.plot(a2_values, b1_on_boundary(a2_values, 1.0), 'r:', lw=1.8, label='Li (common function, kappa=1)')
    for name, (a2, b1v) in (marks or {}).items():
        ax.plot(a2, b1v, marker='*', ms=12, mec='k', mfc='yellow', ls='none')
        ax.annotate(name, (a2, b1v), textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(a2_values.min(), a2_values.max())
    ax.set_ylim(0.25, 4.0)
    ax.set_xlabel('a2 (car→truck)')
    ax.set_ylabel('b1 (truck→car)')
    ax.set_title(title or 's=0 boundary (blue area: s>0 majority)')
    ax.legend(fontsize=7, loc='lower left')
    return ax
