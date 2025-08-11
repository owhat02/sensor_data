# plot_gyro_with_calcs.py
# - 자이로스코프 x,y,z(각속도)와 계산값을 시각화
#   위: 각속도(rad/s) x,y,z, |ω|
#   아래: 적분 각도(deg) θx, θy, θz, θ_total
# - 체크박스로 보이기/숨기기 토글

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates

# import matplotlib
# matplotlib.use("TkAgg")  # 체크박스 클릭 안 되면 주석 해제

# ========= 여기만 네 파일에 맞게 수정 =========
EXCEL_PATH   = r"해부학적자세.xlsx"
SHEET_NAME   = "sensor_data_gyr"   # 시트명
TIME_COL     = "measured_at"
X_COL, Y_COL, Z_COL = "x", "y", "z"       # 열 이름

SMOOTH_WINDOW      = 0      # 이동평균 창(표본 수). 0이면 미적용
BIAS_EST_SAMPLES   = 0      # 초기 N표본의 중앙값으로 바이어스 제거(0=끄기)
SAVE_PNG_PATH      = None   # 자동 저장 끔 (원하면 "gyro_plot.png")
# ============================================

def moving_average(s: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return s.rolling(window, min_periods=max(1, window//2)).mean()
    return s

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

    # 2) 시간 파싱(+ 예외 포맷 보정)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        # "yyyy-mm-dd hh:mm:ss:ms" -> "yyyy-mm-dd hh:mm:ss.ms"
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2",
            regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 유효행 + 정렬
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy().sort_values(TIME_COL)

    # 3) 숫자 변환
    wx = pd.to_numeric(df[X_COL], errors="coerce").astype(float)  # rad/s 가정
    wy = pd.to_numeric(df[Y_COL], errors="coerce").astype(float)
    wz = pd.to_numeric(df[Z_COL], errors="coerce").astype(float)

    # 4) (선택) 바이어스 제거: 초기 N표본 중앙값 빼기
    if BIAS_EST_SAMPLES and BIAS_EST_SAMPLES > 0 and len(df) >= BIAS_EST_SAMPLES:
        bx = wx.iloc[:BIAS_EST_SAMPLES].median()
        by = wy.iloc[:BIAS_EST_SAMPLES].median()
        bz = wz.iloc[:BIAS_EST_SAMPLES].median()
        wx, wy, wz = wx - bx, wy - by, wz - bz

    # 5) (선택) 스무딩
    wx, wy, wz = (moving_average(wx, SMOOTH_WINDOW),
                  moving_average(wy, SMOOTH_WINDOW),
                  moving_average(wz, SMOOTH_WINDOW))

    # 6) 계산값
    # 각속도 크기 |ω|
    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)

    # 샘플 간 Δt(초)
    t = pd.to_datetime(df[TIME_COL])
    dt = t.diff().dt.total_seconds().fillna(0)
    # 음수/0 비정상값 보호: 0보다 큰 값만 유지, 나머지는 0
    dt = dt.where(dt > 0, 0)

    # 적분 각도(rad) → deg
    theta_x = np.degrees((wx * dt).cumsum())
    theta_y = np.degrees((wy * dt).cumsum())
    theta_z = np.degrees((wz * dt).cumsum())
    theta_total = np.degrees((w_mag * dt).cumsum())  # 근사 총 회전량(방향 무시)

    # 7) 플롯: 두 개 서브플롯 (위: 각속도, 아래: 적분 각도)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(right=0.84)  # 체크박스 자리

    # --- 위: 각속도(rad/s) ---
    lines1, labels1 = [], []
    ln_x, = ax1.plot(t, wx, marker="o", linestyle="-", markersize=2, label="ωx (rad/s)")
    lines1.append(ln_x); labels1.append("ωx (rad/s)")
    ln_y, = ax1.plot(t, wy, marker="o", linestyle="-", markersize=2, label="ωy (rad/s)")
    lines1.append(ln_y); labels1.append("ωy (rad/s)")
    ln_z, = ax1.plot(t, wz, marker="o", linestyle="-", markersize=2, label="ωz (rad/s)")
    lines1.append(ln_z); labels1.append("ωz (rad/s)")
    ln_m, = ax1.plot(t, w_mag, marker="o", linestyle="-", markersize=2, label="|ω| (rad/s)")
    lines1.append(ln_m); labels1.append("|ω| (rad/s)")

    ax1.set_title("GYROSCOPE – Angular velocity")
    ax1.set_ylabel("rad/s")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # --- 아래: 적분 각도(deg) ---
    lines2, labels2 = [], []
    thx, = ax2.plot(t, theta_x, marker="o", linestyle="-", markersize=2, label="θx (deg)")
    lines2.append(thx); labels2.append("θx (deg)")
    thy, = ax2.plot(t, theta_y, marker="o", linestyle="-", markersize=2, label="θy (deg)")
    lines2.append(thy); labels2.append("θy (deg)")
    thz, = ax2.plot(t, theta_z, marker="o", linestyle="-", markersize=2, label="θz (deg)")
    lines2.append(thz); labels2.append("θz (deg)")
    tht, = ax2.plot(t, theta_total, marker="o", linestyle="-", markersize=2, label="θ_total (deg)")
    lines2.append(tht); labels2.append("θ_total (deg)")

    ax2.set_title("Integrated angle (drift-prone)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("deg")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 시간축 포맷(연/월/일 시:분:초[.us])
    setup_time_axis(ax2, t)

    # 체크박스 – 위/아래 각각
    rax1 = plt.axes([0.86, 0.56, 0.12, 0.32])
    check1 = CheckButtons(rax1, labels1, [ln.get_visible() for ln in lines1])

    def on_check1(label):
        ln = lines1[labels1.index(label)]
        ln.set_visible(not ln.get_visible()); plt.draw()

    check1.on_clicked(on_check1)

    rax2 = plt.axes([0.86, 0.12, 0.12, 0.32])
    check2 = CheckButtons(rax2, labels2, [ln.get_visible() for ln in lines2])

    def on_check2(label):
        ln = lines2[labels2.index(label)]
        ln.set_visible(not ln.get_visible()); plt.draw()

    check2.on_clicked(on_check2)

    plt.tight_layout(rect=(0, 0, 0.84, 1))
    plt.show()

if __name__ == "__main__":
    main()
