import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_plot(file_path):
    # 1) 엑셀에서 sensor_data 시트 읽기
    df = pd.read_excel(file_path, sheet_name='sensor_data')

    # 2) 측정 시간 컬럼을 datetime 타입으로 변환
    df['measured_at'] = pd.to_datetime(df['measured_at'])
    df['date'] = df['measured_at'].dt.date

    # 3) 가속도 벡터 크기 계산 (optional)
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    #시간 vs 벡터 크기
    for single_date, group in df.groupby('date'):
        plt.figure(figsize=(10, 5))
        plt.scatter(group['measured_at'], group['magnitude'], alpha=0.6)
        plt.xlabel('Timestamp')
        plt.ylabel('Acceleration Magnitude (√(x² + y² + z²))')
        plt.title(f'Accelerometer: {single_date}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    load_and_plot('밸런스 스탠드.xlsx')
