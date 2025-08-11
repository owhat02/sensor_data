# plot_accel_from_excel_toggle_fixed_time.py
# - 체크박스로 x/y/z/|a| 보이기/숨기기
# - x축 시간 레이블을 연-월-일 시:분:초(필요시 마이크로초)로 강제 표시

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

# import matplotlib
# matplotlib.use("TkAgg")  # GUI 이슈 있으면 주석 해제


EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data(linear_acc)"
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"

SHOW_MAGNITUDE = True
SMOOTH_WINDOW  = 0
SAVE_PNG_PATH  = r"accel_plot.png"

def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return series.rolling(window, min_periods=max(1, window//2)).mean()
    return series

def setup_time_axis(ax, tseries: pd.Series):
    """x축 시간 포매터/로케이터 강제 설정 (연도만 뜨는 문제 방지)"""
    tseries = pd.to_datetime(tseries)
    # 데이터에 마이크로초라도 있으면 .%f까지 표시
    has_us = (tseries.dt.microsecond != 0).any()
    fmt = "%Y-%m-%d %H:%M:%S.%f" if has_us else "%Y-%m-%d %H:%M:%S"

    # 기간에 맞춰 눈금 간격 대략 자동, 포맷은 강제
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.DateFormatter(fmt)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    # 레이블이 겹치면 자동 회전
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
            r"\1.\2",
            regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 유효행만, 시간 정렬
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy().sort_values(TIME_COL)

    # 3) 스무딩
    x = moving_average(df[X_COL].astype(float), SMOOTH_WINDOW)
    y = moving_average(df[Y_COL].astype(float), SMOOTH_WINDOW)
    z = moving_average(df[Z_COL].astype(float), SMOOTH_WINDOW)
    mag = np.sqrt(x**2 + y**2 + z**2) if SHOW_MAGNITUDE else None

    # 4) 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)  # 체크박스 공간

    lines, labels = [], []

    ln_x, = ax.plot(df[TIME_COL], x, marker="o", linestyle="-", markersize=3, label=X_COL)
    lines.append(ln_x); labels.append(X_COL)
    ln_y, = ax.plot(df[TIME_COL], y, marker="o", linestyle="-", markersize=3, label=Y_COL)
    lines.append(ln_y); labels.append(Y_COL)
    ln_z, = ax.plot(df[TIME_COL], z, marker="o", linestyle="-", markersize=3, label=Z_COL)
    lines.append(ln_z); labels.append(Z_COL)
    if SHOW_MAGNITUDE and mag is not None:
        ln_m, = ax.plot(df[TIME_COL], mag, marker="o", linestyle="-", markersize=3, label="|a|")
        lines.append(ln_m); labels.append("|a|")

    ax.set_xlabel("Time")
    ax.set_ylabel("LINEAR_ACCELERATION")
    ax.set_title("LINEAR_ACCELERATION (x, y, z over time)")
    ax.grid(True, alpha=0.3)

    # ★ 시간축 강제 포맷
    setup_time_axis(ax, df[TIME_COL])

    # (보조) 범례
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 5) 체크박스
    rax = plt.axes([0.84, 0.4, 0.12, 0.2])
    states = [ln.get_visible() for ln in lines]
    check = CheckButtons(rax, labels, states)

    def on_check(label):
        idx = labels.index(label)
        ln = lines[idx]
        ln.set_visible(not ln.get_visible())
        plt.draw()

    check.on_clicked(on_check)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()