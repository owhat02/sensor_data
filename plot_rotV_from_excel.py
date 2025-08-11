# ROTATION_VECTOR (same-timestamp spreading by index)
# - rotation_vector: x, y, z, a(=w) 시각화
# - 같은 timestamp에 여러 샘플이 있으면, 인덱스 순서대로 T..T_next 구간에 균등 분할해 배치
# - 체크박스로 각 시리즈 보이기/숨기기
# - x축 시간 레이블 강제 표시(YYYY-MM-DD HH:MM:SS[.ffffff])

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

# import matplotlib
# matplotlib.use("TkAgg")  # GUI 이슈 있으면 주석 해제

# ========= 필요한 부분만 수정 =========
EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data(rotationv)"   # 회전벡터 시트명
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"
A_COL        = "a"        # 스칼라 성분(=w) 열 이름

SHOW_A        = True      # a(w) 표시 여부
SMOOTH_WINDOW = 0
SAVE_PNG_PATH = r"rotation_plot_with_a.png"  # 자동 저장 끄려면 None

# 선택: 쿼터니언 정규화(각 행에서 ||q||=1). 기록값 보존 원하면 False 유지
NORMALIZE_QUAT = False
# 같은 timestamp 블록 펼치기 방식: 'median' 또는 'jitter'
SPREAD_FALLBACK = 'median'  # 'jitter'로 바꾸면 T_next 없는 경우 1ms 지터만 적용
JITTER_US = 1000
# ====================================

# --- 같은 timestamp 블록을 펼쳐 시각화용 시간열 생성 ---
def build_plot_time_by_timestamp(df: pd.DataFrame, time_col: str,
                                 fallback: str = 'median', jitter_us: int = 1000) -> pd.Series:
    """
    같은 timestamp 내 여러 샘플을 인덱스(유입 순서)대로 T..T_next 구간에 균등 분할해 배치.
    - fallback='median': T_next 없거나 Δt<=0이면 전체 median Δt 사용
    - fallback='jitter': 그런 경우 고정 지터(기본 1ms)만큼만 벌려 배치
    """
    t = pd.to_datetime(df[time_col])

    # 고유 시간과 그 다음 고유 시간 매핑
    uniq = t.drop_duplicates(keep='first').reset_index(drop=True)
    next_map = pd.Series(uniq.shift(-1).values, index=uniq.values)  # {T: T_next}
    t_next = t.map(next_map)

    # 전체 median Δt (fallback용)
    med_dt = t.sort_values().diff().median()
    if pd.isna(med_dt) or med_dt == pd.Timedelta(0):
        med_dt = pd.Timedelta(seconds=1)

    # 같은 timestamp 그룹 내 등수/크기 (경계에 딱 붙지 않게)
    grp  = df.groupby(time_col, sort=False, dropna=False)
    rank = grp.cumcount()                      # 0..n-1
    size = grp[time_col].transform('size')     # n
    frac = (rank + 1) / (size + 1)             # (0,1) 사이

    # Δt 유효화: 음수/NaT/0 -> fallback 처리
    dt = (t_next - t)
    dt = dt.where(dt > pd.Timedelta(0), pd.NaT)
    if fallback == 'median':
        dt = dt.fillna(med_dt)
    else:  # 'jitter'
        dt = dt.fillna(pd.Timedelta(microseconds=jitter_us))

    offset = dt * frac
    return (t + offset).astype('datetime64[ns]')

def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return series.rolling(window, min_periods=max(1, window//2)).mean()
    return series

def setup_time_axis(ax, tseries: pd.Series):
    tseries = pd.to_datetime(tseries)
    has_us = (tseries.dt.microsecond != 0).any()
    fmt = "%Y-%m-%d %H:%M:%S.%f" if has_us else "%Y-%m-%d %H:%M:%S"
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.DateFormatter(fmt)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

def main():
    # 1) 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 2) 시간 파싱(예외 포맷 보정)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2",
            regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 3) 필수 열 정리 + 안정 정렬(원래 순서 보존)
    base_cols = [TIME_COL, X_COL, Y_COL, Z_COL]
    df = df.dropna(subset=base_cols).copy()
    df['_ord'] = np.arange(len(df))  # 유입 순서 보존
    df = df.sort_values([TIME_COL, '_ord'], kind='mergesort')

    # 4) 숫자 변환 & 스무딩
    x = moving_average(pd.to_numeric(df[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)

    a = None
    has_a = (A_COL in df.columns) and SHOW_A
    if has_a:
        a = moving_average(pd.to_numeric(df[A_COL], errors="coerce").astype(float), SMOOTH_WINDOW)

    # 5) (선택) 쿼터니언 정규화
    if NORMALIZE_QUAT and has_a and a is not None:
        norm = np.sqrt(x**2 + y**2 + z**2 + a**2)
        norm = norm.replace({0: np.nan})
        x, y, z, a = x / norm, y / norm, z / norm, a / norm

    # 6) 시각화용 시간열(같은 timestamp 블록 펼치기) — 모든 시리즈 공통 x축
    t_plot = build_plot_time_by_timestamp(df, TIME_COL, fallback=SPREAD_FALLBACK, jitter_us=JITTER_US)

    # 7) 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)

    lines, labels = [], []

    ln_x, = ax.plot(t_plot, x, marker="o", linestyle="-", markersize=3, label="x")
    lines.append(ln_x); labels.append("x")
    ln_y, = ax.plot(t_plot, y, marker="o", linestyle="-", markersize=3, label="y")
    lines.append(ln_y); labels.append("y")
    ln_z, = ax.plot(t_plot, z, marker="o", linestyle="-", markersize=3, label="z")
    lines.append(ln_z); labels.append("z")

    if has_a and a is not None:
        ln_a, = ax.plot(t_plot, a, marker="o", linestyle="-", markersize=3, label="a (w)")
        lines.append(ln_a); labels.append("a (w)")

    ax.set_xlabel("Time")
    ax.set_ylabel("ROTATION_VECTOR quaternion (unitless)")
    ax.set_title("ROTATION_VECTOR (x, y, z{} over time)".format(", a(w)" if has_a and a is not None else ""))
    ax.grid(True, alpha=0.3)

    setup_time_axis(ax, t_plot)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 체크박스
    rax = plt.axes([0.84, 0.4, 0.12, 0.2])
    states = [ln.get_visible() for ln in lines]
    check = CheckButtons(rax, labels, states)

    def on_check(label):
        idx = labels.index(label)
        ln = lines[idx]
        ln.set_visible(not ln.get_visible()); plt.draw()

    check.on_clicked(on_check)

    # 보조 컬럼 정리(선택)
    if '_ord' in df.columns:
        df.drop(columns=['_ord'], inplace=True)

    # 자동 저장(원하면 경로 유지, 끄려면 None)
    if SAVE_PNG_PATH:
        fig.savefig(SAVE_PNG_PATH, dpi=150)
        print(f"[✓] saved: {os.path.abspath(SAVE_PNG_PATH)}")

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()

if __name__ == "__main__":
    main()