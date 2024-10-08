from google.colab import drive

import csv
from collections import Counter

drive.mount('/content/drive')

def count_first_column_values(csv_file_path):
    values = []

    # CSV 파일 읽기
    with open(csv_file_path, 'r', newline='', encoding='EUC-KR') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 행 건너뛰기

        # 첫 번째 열의 값들을 리스트에 추가 (2행부터)
        for row in csv_reader:
            if row:  # 빈 행 건너뛰기
                values.append(row[0])

    # 값들의 개수 세기
    value_counts = Counter(values)

    return dict(value_counts)

def main():
    # CSV 파일 경로 설정
    csv_file_path_incheonUniv = '/content/drive/MyDrive/20240924/incheon_standard.csv'  
    csv_file_path_samsung = '/content/drive/MyDrive/20240924/samsung_standard.csv'

    try:
        # 함수 실행 및 결과 출력
        result = count_first_column_values(csv_file_path_incheonUniv)
        print("인천대 데이터 값들의 개수:")
        for value, count in result.items():
            print(f"{value}: {count}")
    except FileNotFoundError:
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
    except csv.Error as e:
        print(f"CSV 파일을 처리하는 중 오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

    try:
        # 함수 실행 및 결과 출력
        result = count_first_column_values(csv_file_path_samsung)
        print("삼성 데이터 값들의 개수:")
        for value, count in result.items():
            print(f"{value}: {count}")
    except FileNotFoundError:
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
    except csv.Error as e:
        print(f"CSV 파일을 처리하는 중 오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()

