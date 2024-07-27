#자주 사용하는 라이브러리 임포트
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#소숫점 자리수 표시 방법 지정
pd.options.display.float_format = '{:.5f}'.format

#경고메세지 제거
import warnings
warnings.filterwarnings('ignore')

from mpl_toolkits.mplot3d import Axes3D

location = "https://raw.githubusercontent.com/millim1983/bigdata_project01/main/steel_ai_01_on.csv"
df_tmp = pd.read_csv(location, header=0)
df_tmp.head()


df_tmp = df_tmp.astype({'STEEL_CATEGORY':'category'})
df_tmp = df_tmp.astype({'WORK_SHAPE':'object'})

df_tmp = df_tmp.drop_duplicates()

df_tmp['diff'] = (pd.to_datetime(df_tmp['WORK_END_DT']) - pd.to_datetime(df_tmp['WORK_START_DT'])).dt.total_seconds().div(60).astype(int)
df_tmp[['diff','WORK_START_DT','WORK_END_DT']].head()


# dt.total_seconds() : 전체 초 , div(60) : 60으로 나눔 = 분으로 변환
c= df_tmp.corr(numeric_only = True) #corr:상관계수를 구하는 pandas 내장 함수

# 데이터 준비
labels = c.columns   # INPUT_ED, INPUT_LENGTH, INPUT_QTY, DRECTION_ED, OUTPUT_ED, diff 
x = np.arange(c.shape[0])   # c.shape[0] = 6, [0, 1, 2, 3, 4, 5]
y = np.arange(c.shape[1])   # c.shape[1] = 6, [0, 1, 2, 3, 4, 5]
x, y = np.meshgrid(x, y)  
z = np.zeros_like(x)   # x랑 동일한 0으로된 배열을 만들어 달라 [0,0,0,0,0,0]

# 높이 값 생성
height = c.values  # c = df.corr() 상관관계의 값

# 컬러맵 생성
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=np.min(height), vmax=np.max(height))
colors = cmap(norm(height))

# 그래프 생성
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D 바 플롯
#surf = ax.plot_surface(x, y, height, facecolors = colors, shade = False)
surf = ax.bar3d(x.flatten(), y.flatten(), z.flatten(), 0.2, 0.2, height.flatten(), shade=True, cmap='viridis')


# 컬러바 추가
m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#m.set_array(height)
m.set_array([])

fig.colorbar(m, ax=ax, shrink=0.5, aspect=5)  # ax 인자를 추가하여 컬러바를 생성할 축을 지정

# 축 설정
ax.set_xlabel('df.columns')
ax.set_ylabel('df.columns')
ax.set_zlabel('df.corr()')
plt.title('3D Surface Plot with Color Mapping')



plt.ion()
plt.show()
