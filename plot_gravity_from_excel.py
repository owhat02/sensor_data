# GRAVITY — 날짜별 분리 + 동시간 펼치기 + 체크박스 고정 + 시간축(밀리초 2자리) 표시
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

EXCEL_PATH   = r"벽밀기.xlsx"
SHEET_NAME   = "sensor_data_f_g"
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"

SHOW_MAGNITUDE = False
SMOOTH_WINDOW  = 0
SAVE_PNG_DIR   = None 


def build_plot_time_by_timestamp(df: pd.DataFrame, time_col: str,
                                 fallback: str = 'median',
                                 jitter_us: int = 1000) -> pd.Series:
    """
    같은 timestamp 블록을 T..T_next 사이에 인덱스 비율(frac)대로 균등 분할해 배치.
    (주의) 이 함수는 '하루분 데이터'에 적용하는 걸 권장 (다음날로 넘어가며 Δt가 커지는 것 방지)
    """
    t = pd.to_datetime(df[time_col])

    uniq = t.drop_duplicates(keep='first').reset_index(drop=True)
    next_map = pd.Series(uniq.shift(-1).values, index=uniq.values)   # {T: T_next}
    t_next = t.map(next_map)

    med_dt = t.sort_values().diff().median()
    if pd.isna(med_dt) or med_dt == pd.Timedelta(0):
        med_dt = pd.Timedelta(seconds=1)

    grp  = df.groupby(time_col, sort=False, dropna=False)
    rank = grp.cumcount()
    size = grp[time_col].transform('size')
    frac = (rank + 1) / (size + 1)

    dt = (t_next - t)
    dt = dt.where(dt > pd.Timedelta(0), pd.NaT)
    if fallback == 'median':
        dt = dt.fillna(med_dt)
    else:
        dt = dt.fillna(pd.Timedelta(microseconds=jitter_us))

    offset = dt * frac
    return (t + offset).astype('datetime64[ns]')

def moving_average(s: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return s.rolling(window, min_periods=max(1, window//2)).mean()
    return s

def setup_time_of_day_axis(ax, tseries: pd.Series):
    """시:분:초(.ms 2자리) 포맷으로 표시"""
    tseries = pd.to_datetime(tseries)
    has_us = (tseries.dt.microsecond != 0).any()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    if has_us:
        # 예: 12:34:56.78  (마이크로초 6자리 중 앞 2자리만 표시)
        def short_ms(x, pos):
            s = mdates.num2date(x).strftime("%H:%M:%S.%f")
            return s[:-4]  # 뒤 4자리 잘라서 두 자리만 남김
        ax.xaxis.set_major_formatter(FuncFormatter(short_ms))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_one_day(df_day: pd.DataFrame, day_key: pd.Timestamp):
    # 유입 순서 보존 + 정렬
    df_day = df_day.copy()
    df_day['_ord'] = np.arange(len(df_day))
    df_day = df_day.sort_values([TIME_COL, '_ord'], kind='mergesort')

    # 동시간 펼치기 (하루 데이터에만 적용!)
    t_plot = build_plot_time_by_timestamp(df_day, TIME_COL, fallback='median')

    # 숫자화 & 스무딩
    x = moving_average(pd.to_numeric(df_day[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df_day[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df_day[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    mag = np.sqrt(x**2 + y**2 + z**2) if SHOW_MAGNITUDE else None

    # tz 제거 (있을 경우) 후 바로 datetime Series 사용
    t_dt = pd.to_datetime(t_plot).dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)

    lines, labels = [], []
    ln_x, = ax.plot(t_dt, x, linestyle="-", marker="o", markersize=3, label="x")
    lines.append(ln_x); labels.append("x")
    ln_y, = ax.plot(t_dt, y, linestyle="-", marker="o", markersize=3, label="y")
    lines.append(ln_y); labels.append("y")
    ln_z, = ax.plot(t_dt, z, linestyle="-", marker="o", markersize=3, label="z")
    lines.append(ln_z); labels.append("z")
    if SHOW_MAGNITUDE and mag is not None:
        ln_m, = ax.plot(t_dt, mag, linestyle="-", marker="o", markersize=3, label="|g|")
        lines.append(ln_m); labels.append("|g|")

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Gravity (m/s²)")
    ax.set_title(f"GRAVITY — {day_key.strftime('%Y-%m-%d')} (x, y, z)")
    ax.grid(True, alpha=0.3)

    setup_time_of_day_axis(ax, t_dt)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # 체크박스 (figure에 부착 + draw_idle + GC 방지)
    rax = fig.add_axes([0.84, 0.4, 0.12, 0.2])
    states = [ln.get_visible() for ln in lines]
    check = CheckButtons(rax, labels, states)

    def on_check(label):
        idx = labels.index(label)
        ln = lines[idx]
        ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(on_check)
    fig._lines = lines; fig._labels = labels; fig._check = check  # 참조 보존

    plt.tight_layout(rect=(0, 0, 0.82, 1))

    # (옵션) 저장
    if SAVE_PNG_DIR:
        ensure_dir(SAVE_PNG_DIR)
        out_path = os.path.join(SAVE_PNG_DIR, f"{day_key.strftime('%Y%m%d')}.png")
        fig.savefig(out_path, dpi=150)

def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 시간 파싱 (예외 포맷 보정)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2", regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 유효행만 사용
    df = df.dropna(subset=[TIME_COL, X_COL, Y_COL, Z_COL]).copy()

    # ✔ 날짜 키 생성 (정규화해서 00:00 기준으로 같은 날 묶기)
    df['__date__'] = pd.to_datetime(df[TIME_COL]).dt.normalize()

    # 날짜별로 개별 figure 생성
    for day_key, df_day in df.groupby('__date__', sort=True):
        plot_one_day(df_day, day_key)

    plt.show()

if __name__ == "__main__":
    main()
