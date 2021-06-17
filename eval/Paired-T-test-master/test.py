# import pandas as mypandas
# from scipy import stats as mystats
# myData=mypandas.read_csv('.\datasets\Diet.csv')
# before=myData.Before
# after=myData.After
# print(mystats.ttest_rel(before,after))
import pandas as mypandas
from scipy import stats as mystats

#%%

myData=mypandas.read_csv(".\datasets\\test.csv")

#%%

B1=myData.Brand_1
B2=myData.Brand_2

#%%

print(mystats.ttest_rel(B1,B2))