# ROTATION_VECTOR — 날짜별 개별 창 + (날짜/데이터 조건별) 동시간 펼치기 + 체크박스 + 시간축(밀리초 2자리)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# import matplotlib
# matplotlib.use("TkAgg")  # 여러 창이 한 창으로 합쳐 보이면 잠깐 켜서 테스트

# ========= 설정 =========
EXCEL_PATH   = r"손목돌리기.xlsx"
SHEET_NAME   = "sensor_data(rotationv)"   # 회전벡터 시트명
TIME_COL     = "measured_at"
X_COL        = "x"
Y_COL        = "y"
Z_COL        = "z"
A_COL        = "a"        # 스칼라 성분(=w) 열 이름

SHOW_A        = True      # a(w) 표시 여부
SMOOTH_WINDOW = 0
SAVE_PNG_DIR  = None      # 예: r"./out_rotationv"  (None이면 저장 안 함)

# 선택: 쿼터니언 정규화(각 행에서 ||q||=1). 기록값 보존 원하면 False 유지
NORMALIZE_QUAT = False
# 같은 timestamp 블록 펼치기 방식: 'median' 또는 'jitter'
SPREAD_FALLBACK = 'median'
JITTER_US = 1000
# =======================

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

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def should_spread_for_day(df_day: pd.DataFrame, day_key: pd.Timestamp) -> bool:
    """
    날짜 규칙:
    - (해당 연도) 8월 4일 이전은 무조건 펼치기(True)
    - (해당 연도) 8월 11일 이후는 펼치기 금지(False, 기록된 대로)
    """
    year = day_key.year
    cut_a = pd.Timestamp(f"{year}-08-04")
    cut_b = pd.Timestamp(f"{year}-08-11")

    if day_key < cut_a:
        return True
    if day_key >= cut_b:
        return False

    # 자동 판별 구간: 8/4 ~ 8/10
    t = pd.to_datetime(df_day[TIME_COL])
    no_us = (t.dt.microsecond == 0).all()
    has_dups = t.duplicated(keep=False).any()
    return bool(no_us and has_dups)

def plot_one_day(df_day: pd.DataFrame, day_key: pd.Timestamp):
    # 유입 순서 보존 + 정렬
    df_day = df_day.copy()
    df_day['_ord'] = np.arange(len(df_day))
    df_day = df_day.sort_values([TIME_COL, '_ord'], kind='mergesort')

    # 날짜/데이터 조건에 따라 시간열 결정
    if should_spread_for_day(df_day, day_key):
        t_plot = build_plot_time_by_timestamp(df_day, TIME_COL, fallback=SPREAD_FALLBACK, jitter_us=JITTER_US)
    else:
        t_plot = pd.to_datetime(df_day[TIME_COL])

    # 숫자화 & 스무딩
    x = moving_average(pd.to_numeric(df_day[X_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    y = moving_average(pd.to_numeric(df_day[Y_COL], errors="coerce").astype(float), SMOOTH_WINDOW)
    z = moving_average(pd.to_numeric(df_day[Z_COL], errors="coerce").astype(float), SMOOTH_WINDOW)

    a = None
    has_a_col = (A_COL in df_day.columns) and SHOW_A
    if has_a_col:
        a = moving_average(pd.to_numeric(df_day[A_COL], errors="coerce").astype(float), SMOOTH_WINDOW)

    # (선택) 쿼터니언 정규화
    if NORMALIZE_QUAT and has_a_col and a is not None:
        norm = np.sqrt(x**2 + y**2 + z**2 + a**2)
        norm = norm.replace({0: np.nan})
        x, y, z, a = x / norm, y / norm, z / norm, a / norm

    # tz 제거 (있을 경우) 후 datetime Series 사용
    t_dt = pd.to_datetime(t_plot).dt.tz_localize(None)

    # 플롯 — 날짜마다 '새' figure 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.82)

    # 창 제목에 날짜 표기
    try:
        fig.canvas.manager.set_window_title(f"{day_key.strftime('%Y-%m-%d')} — Rotation Vector")
    except Exception:
        pass

    lines, labels = [], []
    ln_x, = ax.plot(t_dt, x, marker="o", linestyle="-", markersize=3, label="x")
    lines.append(ln_x); labels.append("x")
    ln_y, = ax.plot(t_dt, y, marker="o", linestyle="-", markersize=3, label="y")
    lines.append(ln_y); labels.append("y")
    ln_z, = ax.plot(t_dt, z, marker="o", linestyle="-", markersize=3, label="z")
    lines.append(ln_z); labels.append("z")

    if has_a_col and a is not None:
        ln_a, = ax.plot(t_dt, a, marker="o", linestyle="-", markersize=3, label="a (w)")
        lines.append(ln_a); labels.append("a (w)")

    ax.set_xlabel("Time of day")
    ax.set_ylabel("ROTATION_VECTOR quaternion (unitless)")
    ax.set_title("ROTATION_VECTOR (x, y, z{} over time)".format(", a(w)" if has_a_col and a is not None else ""))
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

    # (옵션) 날짜별 PNG 저장
    if SAVE_PNG_DIR:
        ensure_dir(SAVE_PNG_DIR)
        out_path = os.path.join(SAVE_PNG_DIR, f"{day_key.strftime('%Y%m%d')}.png")
        fig.savefig(out_path, dpi=150)

    return fig  # 메인에서 fig.show()로 각 창 표시

def main():
    # 1) 로드
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # 2) 시간 파싱(예외 포맷 보정)
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
        df[TIME_COL] = df[TIME_COL].str.replace(
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}):(\d+)$",
            r"\1.\2", regex=True
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", infer_datetime_format=True)

    # 3) 필수 열 정리
    base_cols = [TIME_COL, X_COL, Y_COL, Z_COL]
    if A_COL in df.columns and SHOW_A:
        base_cols.append(A_COL)
    df = df.dropna(subset=base_cols).copy()

    # ✔ 날짜 키 생성 (정규화해서 00:00 기준으로 같은 날 묶기)
    df['__date__'] = pd.to_datetime(df[TIME_COL]).dt.normalize()

    # 4) 날짜별로 개별 figure 생성 후 즉시 show()
    figs = []
    for day_key, df_day in df.groupby('__date__', sort=True):
        fig = plot_one_day(df_day, day_key)
        figs.append(fig)
        try:
            fig.show()  # 백엔드에 따라 창 분리가 확실해짐
        except Exception:
            pass

    plt.show()  # 일부 백엔드에서는 마지막에 한 번 더 필요

if __name__ == "__main__":
    main()
