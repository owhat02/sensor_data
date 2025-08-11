import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def load_and_plot(file_path):
    # 1) 엑셀에서 sensor_data 시트 읽기
    df = pd.read_excel(file_path, sheet_name='sensor_data')
    df['measured_at'] = pd.to_datetime(df['measured_at'])
    df['date'] = df['measured_at'].dt.date
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    # 2) 개별 날짜별 산점도 (Figure 1–4)
    unique_dates = sorted(df['date'].unique())
    for idx, single_date in enumerate(unique_dates, start=1):
        grp = df[df['date'] == single_date]
        plt.figure(idx, figsize=(10,5))
        plt.scatter(grp['measured_at'], grp['magnitude'], alpha=0.6)
        plt.xlabel('Timestamp')
        plt.ylabel('Acceleration Magnitude (√(x² + y² + z²))')
        plt.title(f'Accelerometer: {single_date}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

    # 3) 통합 산점도 (Figure 5) — x축에 “time of day”만, 날짜별 색상
    plt.figure(5, figsize=(12,6))
    colors = ['red','blue','purple','green']  # 7/30, 7/31, 8/1, 8/4 순서

    for color, single_date in zip(colors, unique_dates):
        grp = df[df['date'] == single_date]
        # time-only axis: 임의의 기준일(1900-01-01)에 각 시간 붙이기
        times = pd.to_datetime(grp['measured_at'].dt.strftime('%H:%M:%S'),
                               format='%H:%M:%S')
        plt.scatter(times, grp['magnitude'],
                    alpha=0.7, color=color, label=str(single_date))

    # x축 포맷: 오직 시간만 보이도록
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.HourLocator())  # 한 시간 단위 눈금
    plt.xlabel('Time of Day')
    plt.ylabel('Acceleration Magnitude (√(x² + y² + z²))')
    plt.title('Accelerometer: Combined by Time of Day')
    plt.legend(title='Date')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    load_and_plot('힐레이즈.xlsx')
