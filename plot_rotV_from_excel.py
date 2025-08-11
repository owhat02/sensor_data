# plot_rotation_vector_with_a.py
# - rotation_vector: x, y, z, a(=w) 시각화
# - 체크박스로 각 시리즈 보이기/숨기기
# - x축 시간 레이블을 연-월-일 시:분:초(필요시 마이크로초)로 표시

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
# B_COL      = "b"        # 고정 246 → 사용 안 함

SHOW_A       = True       # a(w) 표시 여부
SMOOTH_WINDOW= 0
SAVE_PNG_PATH= r"rotation_plot_with_a.png"

# 선택: 쿼터니언 정규화(각 행마다 ||q||=1로 맞춤). 기록값 보존 원하면 False 유지
NORMALIZE_QUAT = False
# ====================================

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

    # 3) 필수 열 정리
    base_cols = [TIME_COL, X_COL, Y_COL, Z_COL]
    df = df.dropna(subset=base_cols).copy().sort_values(TIME_COL)

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
        # 0으로 나누기 방지
        norm = norm.replace({0: np.nan})
        x, y, z, a = x / norm, y / norm, z / norm, a / norm

    # 6) 플롯
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)

    lines, labels = [], []

    ln_x, = ax.plot(df[TIME_COL], x, marker="o", linestyle="-", markersize=3, label="x")
    lines.append(ln_x); labels.append("x")
    ln_y, = ax.plot(df[TIME_COL], y, marker="o", linestyle="-", markersize=3, label="y")
    lines.append(ln_y); labels.append("y")
    ln_z, = ax.plot(df[TIME_COL], z, marker="o", linestyle="-", markersize=3, label="z")
    lines.append(ln_z); labels.append("z")

    if has_a and a is not None:
        ln_a, = ax.plot(df[TIME_COL], a, marker="o", linestyle="-", markersize=3, label="a (w)")
        lines.append(ln_a); labels.append("a (w)")

    ax.set_xlabel("Time")
    ax.set_ylabel("ROTATION_VECTOR quaternion (unitless)")
    ax.set_title("ROTATION_VECTOR (x, y, z{} over time)".format(", a(w)" if has_a and a is not None else ""))
    ax.grid(True, alpha=0.3)

    setup_time_axis(ax, df[TIME_COL])

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 체크박스
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