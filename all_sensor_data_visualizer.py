import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_all_sensors_long(file_path):
    # 1) 데이터 로드, measured_at 파싱
    df = pd.read_excel(file_path, sheet_name='sensor_data', parse_dates=['measured_at'])
    
    # 2) 디버깅용 출력: 실제 센서 이름과 측정시간 확인
    print(">> Unique sensor_name values:", df['sensor_name'].unique())
    print(">> measured_at examples:\n", df['measured_at'].head())
    
    # 3) 플롯 준비
    plt.figure(figsize=(12,6))
    cmap = plt.get_cmap('tab10')
    
    # 4) 파일에 실제 들어있는 sensor_name 리스트로 순회
    sensors = df['sensor_name'].unique()
    for idx, sensor in enumerate(sensors):
        sub = df[df['sensor_name'] == sensor]
        if sub.empty:
            continue
        
        # 5) measured_at 에 시간 정보가 없다면 00:00:00 이 찍힙니다.
        #    시간 정보만 추출해서 plotting 할 경우:
        times = pd.to_datetime(
            sub['measured_at'].dt.strftime('%H:%M:%S'),
            format='%H:%M:%S'
        )
        
        # 6) magnitude 계산
        mag = np.sqrt(sub['x']**2 + sub['y']**2 + sub['z']**2)
        
        # 7) 플롯
        plt.plot(
            times, mag,
            marker='o', linestyle='-',
            label=sensor, color=cmap(idx)
        )
    
    # 8) 축 포맷팅 (시간만 표시)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    
    plt.xlabel('Time of Day')
    plt.ylabel('Magnitude (√(x² + y² + z²))')
    plt.title('Sensor Magnitudes Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_all_sensors_long('해부학적자세.xlsx')
