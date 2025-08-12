# ACCELEROMETER (동시간 샘플 펼치기 적용 + 날짜별 분리 플로팅)
# - 같은 timestamp에 샘플이 여러 개면, 그 다음 고유 시간까지 간격을 등분해 배치
# - 날짜별로 잘라서 각각의 figure에 그리되, x축은 시간(시:분:초[.us])만 표시

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data(accelerometer)"
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"

SHOW_MAGNITUDE = True
SMOOTH_WINDOW  = 0
SAVE_PNG_PATH  = None  # 예: r"./out_plots"  # 폴더 지정하면 날짜별 PNG 저장

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

def setup_time_axis_time_of_day(ax, tseries: pd.Series):
    """날짜별 플롯이므로 시간(HH:MM:SS[.us])만 표시"""
    tseries = pd.to_datetime(tseries)
    has_us = (tseries.dt.microsecond != 0).any()
    fmt = "%H:%M:%S.%f" if has_us else "%H:%M:%S"
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.DateFormatter(fmt)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_one_day(df_day: pd.DataFrame, day_label: pd.Timestamp):
    df_day = df_day.copy()
    df_day['_ord'] = np.arange(len(df_day))
    df_day = df_day.sort_values([TIME_COL, '_ord'], kind='mergesort')

    t_plot = build_plot_time_by_timestamp(df_day, TIME_COL, fallback='median')

    x = moving_average(pd.to_numeric(df_day[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df_day[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df_day[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    mag = np.sqrt(x**2 + y**2 + z**2) if SHOW_MAGNITUDE else None

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)

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

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"Accelerometer — {day_label.strftime('%Y-%m-%d')} (x, y, z over time)")
    ax.grid(True, alpha=0.3)
    setup_time_axis_time_of_day(ax, t_plot)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # ✅ (1) 이 figure에 명시적으로 축을 추가
    rax = fig.add_axes([0.84, 0.4, 0.12, 0.2])
    states = [ln.get_visible() for ln in lines]
    check = CheckButtons(rax, labels, states)

    def on_check(label):
        idx = labels.index(label)
        ln = lines[idx]
        ln.set_visible(not ln.get_visible())
        # ✅ (2) 이 figure만 다시 그리기
        fig.canvas.draw_idle()

    check.on_clicked(on_check)

    # ✅ (3) 위젯/데이터 참조 유지해서 GC 방지
    fig._lines = lines
    fig._labels = labels
    fig._check = check

    plt.tight_layout(rect=(0, 0, 0.82, 1))

def main():
    # 0) 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 1) 시간 파싱(예외 포맷 보정 포함)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2", regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 2) 유효행만
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy()

    # 3) 날짜 키로 그룹핑 (00:00 기준 정규화)
    df['__date__'] = pd.to_datetime(df[TIME_COL]).dt.normalize()

    # 4) 날짜별 플롯 생성
    for day_key, df_day in df.groupby('__date__', sort=True):
        plot_one_day(df_day, day_key)

    plt.show()  # 모든 날짜의 figure를 한 번에 표시

if __name__ == "__main__":
    main()