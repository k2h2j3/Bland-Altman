import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chardet

# 파일 인코딩 확인
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

# 파일 경로
file_path = '/content/drive/MyDrive/20240924/incheon_standard.csv'

# 인코딩 감지 및 출력
detected_encoding = detect_encoding(file_path)
print(f"Detected encoding: {detected_encoding}")

# CSV 파일 로드
data = pd.read_csv(file_path, encoding=detected_encoding)

# 열 이름 확인 및 데이터 추출
print("Column names:", data.columns)
method1 = data.iloc[:, 0].values
method2 = data.iloc[:, 1].values

# 차이와 평균 계산
diff = method1 - method2
mean = (method1 + method2) / 2

# 차이의 평균과 표준편차 계산
md = np.mean(diff)
sd = np.std(diff, axis=0)

# 일치도 한계 계산 (차이의 평균 ± 1.96 * 차이의 표준편차)
loa_upper = md + 1.96 * sd
loa_lower = md - 1.96 * sd

# Bland-Altman 플롯 그리기
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.scatter(mean, diff, alpha=0.5)
ax.axhline(md, color='gray', linestyle='--', label='Mean difference')
ax.axhline(loa_upper, color='red', linestyle='--', label='Upper LOA')
ax.axhline(loa_lower, color='red', linestyle='--', label='Lower LOA')

ax.set_xlabel('Mean of two methods')
ax.set_ylabel('Difference between two methods')
ax.set_title("Bland-Altman Plot")

# 선 레이블 추가
ax.text(ax.get_xlim()[1], md, f'{md:.2f}', va='center', ha='left', backgroundcolor='w')
ax.text(ax.get_xlim()[1], loa_upper, f'{loa_upper:.2f}', va='center', ha='left', backgroundcolor='w')
ax.text(ax.get_xlim()[1], loa_lower, f'{loa_lower:.2f}', va='center', ha='left', backgroundcolor='w')

ax.legend()
plt.tight_layout()
plt.show()

print(f"평균오차: {md:.2f}")
print(f"표준편차: {sd:.2f}")
print(f"상한치(Upper Limit of Agreement): {loa_upper:.2f}")
print(f"하한치(Lower Limit of Agreement): {loa_lower:.2f}")
