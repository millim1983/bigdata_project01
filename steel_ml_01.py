import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  

# ----------- 사전 설정 --------------# 
# 소수점 자리수 표시 
pd.options.display.float_format = '{:.5f}'.format

# 경고 메세지 제거 
import warnings
warnings.filterwarnings('ignore')




# ----------- 데이터 준비 ----------- # 

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

# 데이터 확인
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.nunique())

# 데이터 타입 변경

df = df.astype({'STEEL_CATEGORY' : 'category'})  # STEEL_CATEGORY 값이 6개이므로 타입을 category로 변경
df = df.astype({'WORK_SHAPE':'object'})  # WORK_SHAPE은 숫자이지만, 작업조를 나타냄 1근, 2근 3근 > object로 변경

# 데이터 타입 확인 
print(df.info())

# 수치형 데이터 요약 
print(df.describe())


# 데이터 탐색 
# 2-1 중복데이터 처리
# 2-2 종속변수 생성
# 2-3 개별 분포 확인
# 2-4 소요시간에 따른 분포 확인 
# 2-5 상관관계 확인 

# 2-1 중복데이터 처리
# 전체 데이터수 확인 
print('젅체 데이터 수 :', df.shape)  # 27150, 10

# 중복데이터 수 
print(df[df.duplicated()].shape)  # 9, 10

# 중복 제거 후 데이터 수 
df = df.drop_duplicates()
print(df.shape)   # 27141, 10  
print(df.info())  # entries : 27141, index : 0 to 27149 문제 없나? 

# 2-2 종속변수 생성

# 작업소요시간(분) = 작업종료시간 - 작업시간 
# diff = work_end_dt - work_start_dt
df['work_start_dt_ns'] = pd.to_datetime(df['WORK_START_DT'])
df['work_end_dt_ns'] = pd.to_datetime(df['WORK_END_DT'])
print(df.info())  

df['diff'] = (df['work_end_dt_ns'] - df['work_start_dt_ns']).dt.total_seconds().div(60).astype(int)


print(df[['diff', 'work_start_dt_ns', 'work_end_dt_ns']].head())
print(df['diff'].tail())


# 2-3 개별 분포 확인 

## 모든 수치형 변수에 대한 히스토그램 확인 
### 수치형 컬럼만 df1로 따로 저장하기 
df1 = df.select_dtypes([int, float])  # df에서 데이타 타입이 int, float만 셀렉트하여 df1에 저장 

### 해당 컬럼 확인 
print('데이터 타입이 int, float만 셀렉트')
print(df.select_dtypes([int, float]).info())
print('데이터 타입이 int, float인 컬럼', df1.columns)


## 모든 수치형 변수에 대한 히스토그램 확인
# 각 컬럼별 값의 빈도수 x 축 = 값의 범위, y축은 = 특정 구간에 속하는 데이터 개수나 비율 


for i, col in enumerate(df1.columns) : 
    # print(i, col)  # i = 0 1 2 3 4, col = INPUT_ED INPUT_LENGTH INPUT_QTY DIRECTION_ED OUTPUT_ED
    plt.figure(i)  # 그래프를 그릴 figure 지정 - figure0, figure1, figure2 ... 
    sns.histplot(x = col, data = df1)


## countplot 함수로 개별분포 확인 x축 = 값의 카테고리(범주), y = 데이터 개수 
print(df.nunique())   # INPUT_QTY 은 데이터값 종류가 9개 (int, float 컬럼 중 가장 작다)
#sns.countplot(df['INPUT_QTY']) 


# # 그래프 설정
fig, ax = plt.subplots(ncols=2)  # 1행, 2열의 서브플롯을 갖는 figure 객체와 그에 해당하는 2개의 ax 객체(그래프 축 객체)를 설정 
fig.set_figwidth(15) 

# # 전체 소요시간 분포
sns.histplot(df['diff'], ax=ax[0])


# 0분 초과 120분 이하 소요시간 분포 
sns.histplot(df.loc[(df['diff'] <=120) & (df['diff'] >0)]['diff'],bins = 20, ax = ax[1])  
# bins 는 막대기 갯수 = 데이터 구간 분할 갯수, 
# ax = 그래프 축이고 뒤에 붙는 [0] [1] 등 번호는 인덱스로 figure 내 그래프 그릴 순서 
# 객체 ax[0] = 분할 된 figure 에서 첫번째에 그려라, ax[1] .. 두번째에 그려라 



plt.show()  # 반드시 마지막에 붙여야모든 그래프가 그려짐 
