# GYROSCOPE — 날짜별 개별 창 + (날짜/데이터 조건별) 동시간 펼치기 + 밀리초 2자리 + 체크박스 고정
# - 위: 각속도(rad/s) x,y,z, |ω|
# - 아래: 적분 각도(deg) θx, θy, θz, θ_total
# - 날짜별로 분리하여 각기 다른 figure로 표시
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

EXCEL_PATH   = r"손목돌리기.xlsx"
SHEET_NAME   = "sensor_data_gyr"   # 시트명
TIME_COL     = "measured_at"
X_COL, Y_COL, Z_COL = "x", "y", "z"

SMOOTH_WINDOW      = 0      # 이동평균 창(표본 수). 0이면 미적용
BIAS_EST_SAMPLES   = 0      # 초기 N표본 중앙값으로 바이어스 제거(0=끄기)
SAVE_PNG_DIR       = None   # 예: r"./out_gyro"  (None이면 저장 안 함)
USE_PLOT_TIME_FOR_INTEGRATION = False  # True면 적분 Δt를 t_plot 기준으로 계산

# --- 같은 timestamp 블록을 펼쳐 시각화용 시간열 생성 ---
def build_plot_time_by_timestamp(df: pd.DataFrame, time_col: str,
                                 fallback: str = 'median',  # 'median' or 'jitter'
                                 jitter_us: int = 1000) -> pd.Series:
    """
    같은 timestamp 내 여러 샘플을 인덱스(유입 순서)대로 T..T_next 구간에 균등 분할해 배치.
    - fallback='median': T_next 없거나 Δt<=0이면 전체 median Δt 사용
    - fallback='jitter': 그런 경우 고정 지터(기본 1ms)만 적용
    (주의) 하루 단위 데이터에 적용 권장
    """
    t = pd.to_datetime(df[time_col])

    uniq = t.drop_duplicates(keep='first').reset_index(drop=True)
    next_map = pd.Series(uniq.shift(-1).values, index=uniq.values)  # {T: T_next}
    t_next = t.map(next_map)

    med_dt = t.sort_values().diff().median()
    if pd.isna(med_dt) or med_dt == pd.Timedelta(0):
        med_dt = pd.Timedelta(seconds=1)

    grp  = df.groupby(time_col, sort=False, dropna=False)
    rank = grp.cumcount()                      # 0..n-1
    size = grp[time_col].transform('size')     # n
    frac = (rank + 1) / (size + 1)             # (0,1) 경계 회피

    dt = (t_next - t)
    dt = dt.where(dt > pd.Timedelta(0), pd.NaT)
    if fallback == 'median':
        dt = dt.fillna(med_dt)
    else:  # 'jitter'
        dt = dt.fillna(pd.Timedelta(microseconds=jitter_us))

    offset = dt * frac
    return (t + offset).astype('datetime64[ns]')

def moving_average(s: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return s.rolling(window, min_periods=max(1, window//2)).mean()
    return s

def setup_time_of_day_axis(ax, tseries: pd.Series):
    """시:분:초(.ms 2자리) 또는 시:분:초 포맷으로 표시"""
    tseries = pd.to_datetime(tseries)
    has_us = (tseries.dt.microsecond != 0).any()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    if has_us:
        # 예: 12:34:56.78  (마이크로초 6자리 중 앞 2자리만 표시)
        def short_ms(x, pos):
            s = mdates.num2date(x).strftime("%H:%M:%S.%f")
            return s[:-4]  # 뒤 4자리 잘라 두 자리만 남김
        ax.xaxis.set_major_formatter(FuncFormatter(short_ms))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def should_spread_for_day(df_day: pd.DataFrame, day_key: pd.Timestamp) -> bool:
    """
    날짜 규칙:
    - (해당 연도) 8월 4일 이전: 무조건 펼치기(True)
    - (해당 연도) 8월 11일 이후: 펼치기 금지(False, 기록된 대로)
    """
    year = day_key.year
    cut_a = pd.Timestamp(f"{year}-08-04")
    cut_b = pd.Timestamp(f"{year}-08-11")

    if day_key < cut_a:
        return True
    if day_key >= cut_b:
        return False

    t = pd.to_datetime(df_day[TIME_COL])
    no_us = (t.dt.microsecond == 0).all()
    has_dups = t.duplicated(keep=False).any()
    return bool(no_us and has_dups)

def integrate_angles(wx: pd.Series, wy: pd.Series, wz: pd.Series,
                     t_for_dt: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """rad/s * dt -> rad 누적 후 degrees 변환"""
    dt = pd.to_datetime(t_for_dt).diff().dt.total_seconds().fillna(0)
    dt = dt.where(dt > 0, 0)  # 음수/0 보호
    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    theta_x = np.degrees((wx * dt).cumsum())
    theta_y = np.degrees((wy * dt).cumsum())
    theta_z = np.degrees((wz * dt).cumsum())
    theta_total = np.degrees((w_mag * dt).cumsum())
    return theta_x, theta_y, theta_z, theta_total

def plot_one_day(df_day: pd.DataFrame, day_key: pd.Timestamp):
    # 유입 순서 보존 + 정렬
    df_day = df_day.copy()
    df_day['_ord'] = np.arange(len(df_day))
    df_day = df_day.sort_values([TIME_COL, '_ord'], kind='mergesort')

    # 원시 각속도(rad/s)
    wx = pd.to_numeric(df_day[X_COL], errors="coerce").astype(float)
    wy = pd.to_numeric(df_day[Y_COL], errors="coerce").astype(float)
    wz = pd.to_numeric(df_day[Z_COL], errors="coerce").astype(float)

    # 바이어스 제거(선택)
    if BIAS_EST_SAMPLES and BIAS_EST_SAMPLES > 0 and len(df_day) >= BIAS_EST_SAMPLES:
        wx -= wx.iloc[:BIAS_EST_SAMPLES].median()
        wy -= wy.iloc[:BIAS_EST_SAMPLES].median()
        wz -= wz.iloc[:BIAS_EST_SAMPLES].median()

    # 스무딩(선택)
    wx, wy, wz = moving_average(wx, SMOOTH_WINDOW), moving_average(wy, SMOOTH_WINDOW), moving_average(wz, SMOOTH_WINDOW)

    # 날짜/데이터 조건에 따라 t_plot 결정
    if should_spread_for_day(df_day, day_key):
        t_plot = build_plot_time_by_timestamp(df_day, TIME_COL, fallback='median')
    else:
        t_plot = pd.to_datetime(df_day[TIME_COL])

    # 적분 Δt 기준 선택
    t_raw = pd.to_datetime(df_day[TIME_COL])
    t_for_dt = t_plot if USE_PLOT_TIME_FOR_INTEGRATION else t_raw

    # 적분 각도 계산
    theta_x, theta_y, theta_z, theta_total = integrate_angles(wx, wy, wz, t_for_dt)
    w_mag = np.sqrt(wx**2 + wy**2 + wz**2)

    # tz 제거 후 시각화
    t_dt = pd.to_datetime(t_plot).dt.tz_localize(None)

    # ---- Figure 생성(날짜별 창 분리)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(right=0.84)  # 체크박스 자리

    # 창 제목에 날짜 표기
    try:
        fig.canvas.manager.set_window_title(f"{day_key.strftime('%Y-%m-%d')} — Gyroscope")
    except Exception:
        pass

    # --- 위: 각속도(rad/s) ---
    lines1, labels1 = [], []
    ln_x, = ax1.plot(t_dt, wx, marker="o", linestyle="-", markersize=2, label="ωx (rad/s)")
    lines1.append(ln_x); labels1.append("ωx (rad/s)")
    ln_y, = ax1.plot(t_dt, wy, marker="o", linestyle="-", markersize=2, label="ωy (rad/s)")
    lines1.append(ln_y); labels1.append("ωy (rad/s)")
    ln_z, = ax1.plot(t_dt, wz, marker="o", linestyle="-", markersize=2, label="ωz (rad/s)")
    lines1.append(ln_z); labels1.append("ωz (rad/s)")
    ln_m, = ax1.plot(t_dt, w_mag, marker="o", linestyle="-", markersize=2, label="|ω| (rad/s)")
    lines1.append(ln_m); labels1.append("|ω| (rad/s)")

    ax1.set_title("GYROSCOPE – Angular velocity")
    ax1.set_ylabel("rad/s")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # --- 아래: 적분 각도(deg) ---
    lines2, labels2 = [], []
    thx, = ax2.plot(t_dt, theta_x, marker="o", linestyle="-", markersize=2, label="θx (deg)")
    lines2.append(thx); labels2.append("θx (deg)")
    thy, = ax2.plot(t_dt, theta_y, marker="o", linestyle="-", markersize=2, label="θy (deg)")
    lines2.append(thy); labels2.append("θy (deg)")
    thz, = ax2.plot(t_dt, theta_z, marker="o", linestyle="-", markersize=2, label="θz (deg)")
    lines2.append(thz); labels2.append("θz (deg)")
    tht, = ax2.plot(t_dt, theta_total, marker="o", linestyle="-", markersize=2, label="θ_total (deg)")
    lines2.append(tht); labels2.append("θ_total (deg)")

    ax2.set_title("Integrated angle (drift-prone)")
    ax2.set_xlabel("Time of day")
    ax2.set_ylabel("deg")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 시간축 포맷(밀리초 2자리) — 공유 x축이라 아래 축만 설정
    setup_time_of_day_axis(ax2, t_dt)

    # 체크박스(위/아래) — figure에 부착 + draw_idle + GC 방지
    rax1 = fig.add_axes([0.86, 0.56, 0.12, 0.32])
    check1 = CheckButtons(rax1, labels1, [ln.get_visible() for ln in lines1])
    def on_check1(label):
        ln = lines1[labels1.index(label)]
        ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()
    check1.on_clicked(on_check1)

    rax2 = fig.add_axes([0.86, 0.12, 0.12, 0.32])
    check2 = CheckButtons(rax2, labels2, [ln.get_visible() for ln in lines2])
    def on_check2(label):
        ln = lines2[labels2.index(label)]
        ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()
    check2.on_clicked(on_check2)

    # 참조 보존(GC 방지)
    fig._lines1, fig._labels1, fig._check1 = lines1, labels1, check1
    fig._lines2, fig._labels2, fig._check2 = lines2, labels2, check2

    plt.tight_layout(rect=(0, 0, 0.84, 1))

    return fig

def main():
    # 1) 데이터 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 2) 시간 파싱(+ 예외 포맷 보정)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        # "yyyy-mm-dd hh:mm:ss:ms" -> "yyyy-mm-dd hh:mm:ss.ms"
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2", regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 3) 유효행만 사용
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy()

    # ✔ 날짜 키 생성 (정규화해서 00:00 기준으로 같은 날 묶기)
    df['__date__'] = pd.to_datetime(df[TIME_COL]).dt.normalize()

    # 4) 날짜별 개별 figure 생성 후 즉시 show()
    figs = []
    for day_key, df_day in df.groupby('__date__', sort=True):
        fig = plot_one_day(df_day, day_key)
        figs.append(fig)
        try:
            fig.show()  # 백엔드에 따라 창 분리가 확실해짐
        except Exception:
            pass

    plt.show()  # 일부 백엔드는 마지막에 한 번 더 필요

if __name__ == "__main__":
    main()
