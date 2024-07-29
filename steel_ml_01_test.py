import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# 소수점 자리수 표시 
pd.options.display.float_format = '{:.5f}'.format

# 경고 메세지 제거 
import warnings
warnings.filterwarnings('ignore')

#--- 분석목표 --- 
# 황삭기 작업 소요 영향 요소 파악 
# 과거 실적데이터 이용하여 만든 예측 모델로 모든 재공들의 작업 소요시간미리 파악, 황삭기 시간 기반 부하 상황 확인

# --- 실습 내용 --- 
# 황삭기 실적데이터 이해 
# 탐색적 데이터 분석 (EDA), 시각화를 통한 데이터 특성 파악
# 이상 데이터 처리 및 파생변수 생성
# 데이터 분리(train, test)
# 예측모델(Regression) 학습 및 평가
# 하이퍼파라미터 튜닝

# --- 실습 데이터 ---
# 황삭기는 Bite, Drill 등 공구로 표면을 깎아내어 소재의 조직 및 물리적 성질 변화없이 형상만 변화시키는 가공설비
# 실습데이터는 황상설비 1개의 3년치 MES작업실적데이터
# 1) 소재별 특성데이터, 2) 가공 후 작업 결과 데이터
# 1) 소재별 특성데이터 - 소재 사이즈(외경, 길이), 투입량(중량, 수량), 가공후 목표 외경, 강종, 작업표준(이동속도/절입량)
# 2) 가공 후 작업 결과 데이터는 작업시작/종료시간, 작업후 생산 사이즈(외경길이), 생산량(중량, 수량)
# 2)의 경우 작업 전 소재는 알수 없는 정보로 추측모델 생성시 제외

# --- 데이터 구성---
# csv, 27,100개 


# 데이터 불러오기 
## 생산 소요시간 로그 데이터 읽어오기 
location = 'steel_ai_01_on.csv'
df = pd.read_csv(location, header = 0)

col = df.select_dtypes([int, float]).columns
