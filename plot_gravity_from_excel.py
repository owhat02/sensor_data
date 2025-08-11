# plot_gravity_toggle.py
# - gravity 센서: x, y, z (m/s^2) 시간축 그래프
# - 체크박스로 각 축 보이기/숨기기
# - x축 라벨은 "YYYY-MM-DD HH:MM:SS[.ffffff]"로 강제 표시
# - 자동 저장 꺼둠 (SAVE_PNG_PATH=None)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

# import matplotlib
# matplotlib.use("TkAgg")  # 체크박스 클릭 안 되면 주석 해제

# ====== 여기만 네 파일에 맞게 수정 ======
EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data_f_g"   # 시트명
TIME_COL     = "measured_at"            # 시간 컬럼
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"

SHOW_MAGNITUDE = False   # |g|=sqrt(x^2+y^2+z^2)도 보고 싶으면 True
SMOOTH_WINDOW  = 0       # 이동평균 창(표본 수). 0이면 미적용
SAVE_PNG_PATH  = None    # 자동 저장 끔 (원하면 "gravity_plot.png" 등으로 지정)
# ======================================

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

    # 2) 시간 파싱(예외 포맷 보정)
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

    # 3) 숫자 변환 & 스무딩
    x = moving_average(pd.to_numeric(df[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)

    mag = None
    if SHOW_MAGNITUDE:
        mag = np.sqrt(x**2 + y**2 + z**2)

    # 4) 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)  # 체크박스 자리

    lines, labels = [], []

    ln_x, = ax.plot(df[TIME_COL], x, marker="o", linestyle="-", markersize=3, label="x")
    lines.append(ln_x); labels.append("x")
    ln_y, = ax.plot(df[TIME_COL], y, marker="o", linestyle="-", markersize=3, label="y")
    lines.append(ln_y); labels.append("y")
    ln_z, = ax.plot(df[TIME_COL], z, marker="o", linestyle="-", markersize=3, label="z")
    lines.append(ln_z); labels.append("z")

    if SHOW_MAGNITUDE and mag is not None:
        ln_m, = ax.plot(df[TIME_COL], mag, marker="o", linestyle="-", markersize=3, label="|g|")
        lines.append(ln_m); labels.append("|g|")

    ax.set_xlabel("Time")
    ax.set_ylabel("Gravity (m/s²)")
    ax.set_title("GRAVITY (x, y, z over time)")
    ax.grid(True, alpha=0.3)

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

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()

if __name__ == "__main__":
    main()