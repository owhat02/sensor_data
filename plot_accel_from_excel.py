# ACCELEROMETER (동시간 샘플 펼치기 적용: timestamp 블록을 인덱스 순서대로 등분)
# - 같은 timestamp에 샘플이 여러 개면, 그 다음 고유 시간까지 간격을 등분해 배치
# - x/y/z/|a| 모두 같은 x축(t_plot)을 사용

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

# import matplotlib
# matplotlib.use("TkAgg")  # GUI 이슈 있으면 주석 해제

EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data(accelerometer)"
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"

SHOW_MAGNITUDE = True
SMOOTH_WINDOW  = 0
SAVE_PNG_PATH  = None  # 자동 저장 끔

# --------- 핵심: timestamp 블록을 펼쳐 시각화용 시간열 생성 ---------
def build_plot_time_by_timestamp(df: pd.DataFrame, time_col: str,
                                 fallback: str = 'median',  # 'median' 또는 'jitter'
                                 jitter_us: int = 1000) -> pd.Series:
    """
    같은 timestamp 내 여러 샘플을 인덱스(유입 순서)대로 T..T_next 구간에 균등 분할해 배치.
    - fallback='median': T_next가 없거나 Δt<=0이면 전체 median Δt 사용
    - fallback='jitter': 그런 경우 고정 지터(기본 1ms)만큼만 벌려 배치
    반환: 시각화용 시간열(pd.Series, datetime64[ns])
    """
    t = pd.to_datetime(df[time_col])

    # 고유 시간과 그 다음 고유 시간 매핑
    uniq = t.drop_duplicates(keep='first').reset_index(drop=True)
    next_map = pd.Series(uniq.shift(-1).values, index=uniq.values)   # {T: T_next}
    t_next = t.map(next_map)

    # 전체 median Δt (fallback용)
    med_dt = t.sort_values().diff().median()
    if pd.isna(med_dt) or med_dt == pd.Timedelta(0):
        med_dt = pd.Timedelta(seconds=1)

    # 같은 timestamp 그룹 내 등수/크기 (경계에 딱 붙지 않게)
    grp  = df.groupby(time_col, sort=False, dropna=False)
    rank = grp.cumcount()                         # 0..n-1
    size = grp[time_col].transform('size')        # n
    frac = (rank + 1) / (size + 1)                # (0,1) 사이

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
    # 1) 데이터 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 2) 시간 파싱(예외 포맷 보정 포함)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        # "yyyy-mm-dd hh:mm:ss:ms" -> "yyyy-mm-dd hh:mm:ss.ms"
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2", regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 3) 유효행만 남기고, 원래 순서를 보존하며 "안정 정렬"
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy()
    df['_ord'] = np.arange(len(df))  # 유입 순서 보존
    df = df.sort_values([TIME_COL, '_ord'], kind='mergesort')

    # 4) 시각화용 시간열 생성: 같은 timestamp 블록 펼치기 (모든 시리즈 공통)
    t_plot = build_plot_time_by_timestamp(df, TIME_COL, fallback='median')  # 필요시 'jitter'

    # 5) 숫자 변환 & 스무딩
    x = moving_average(pd.to_numeric(df[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    mag = np.sqrt(x**2 + y**2 + z**2) if SHOW_MAGNITUDE else None

    # 6) 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)  # 체크박스 자리

    lines, labels = [], []

    ln_x, = ax.plot(t_plot, x, marker="o", linestyle="-", markersize=3, label=X_COL)
    lines.append(ln_x); labels.append(X_COL)
    ln_y, = ax.plot(t_plot, y, marker="o", linestyle="-", markersize=3, label=Y_COL)
    lines.append(ln_y); labels.append(Y_COL)
    ln_z, = ax.plot(t_plot, z, marker="o", linestyle="-", markersize=3, label=Z_COL)
    lines.append(ln_z); labels.append(Z_COL)
    if SHOW_MAGNITUDE and mag is not None:
        ln_m, = ax.plot(t_plot, mag, marker="o", linestyle="-", markersize=3, label="|a|")
        lines.append(ln_m); labels.append("|a|")

    ax.set_xlabel("Time")
    ax.set_ylabel("Acceleration")
    ax.set_title("Accelerometer (x, y, z over time)")
    ax.grid(True, alpha=0.3)

    setup_time_axis(ax, t_plot)

    # (보조) 범례
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 7) 체크박스
    rax = plt.axes([0.84, 0.4, 0.12, 0.2])
    states = [ln.get_visible() for ln in lines]
    check = CheckButtons(rax, labels, states)

    def on_check(label):
        idx = labels.index(label)
        ln = lines[idx]
        ln.set_visible(not ln.get_visible())
        plt.draw()

    check.on_clicked(on_check)

    # 보조 컬럼 정리(선택)
    if '_ord' in df.columns:
        df.drop(columns=['_ord'], inplace=True)

    # 자동 저장 끔
    if SAVE_PNG_PATH:
        fig.savefig(SAVE_PNG_PATH, dpi=150)
        print(f"[✓] saved: {os.path.abspath(SAVE_PNG_PATH)}")

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()

if __name__ == "__main__":
    main()