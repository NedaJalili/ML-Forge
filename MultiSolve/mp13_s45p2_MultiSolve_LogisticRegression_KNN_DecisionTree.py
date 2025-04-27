#با توجه به ديتاست، چه سايز لباسي مناسب فردي با وزن 69kg و قد 169cm است؟
#فايل T_shirt.xlsx
#داده ها برچسب دارند. برچسب ها گسسته هستند. میتوانیم از الگوریتم های طبقه بندی استفاده کنیم.

#فراخواني کتابخانه براي ساماندهي  داده ها و کار با آنها
import pandas as pd
#خواندن داده ها از فايل مربوطه 
data=pd.read_excel("T_shirt.xlsx")

#کسب اطلاعات از داده ها
print(data.shape)#ابعاد داه ها
print(data)#خود فايل
print(data.describe())# اطلاعات سازمان يافته از داده ها

#تقسيم داده ها به ويژگي و برچسب
x=data.iloc[:,:-1].values #ويژگي ها
y=data.iloc[:,-1].values #برچسب ها

#فراخواني کتابخانه براي تقسيم داده ها به دو بخش آموزش و آزمون
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)#تقسيم داده ها به دو بخش آموزش و آزمون


#LogisticRegression

#فراخواني الگوريتم مورد نياز
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()#تعريف مدل
model.fit(xtrain,ytrain)#برازش مدل روي داده هاي آموزش

ypred=model.predict([[169,69]])#پيش بيني براي داده مورد نظر
print("LogisticRegression : ",ypred)
#-----------------------------------------

#DecisionTree

#فراخواني الگوريتم مورد نياز
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()#تعريف مدل
model.fit(xtrain,ytrain)#برازش مدل روي داده هاي آموزش
ypred=model.predict([[169,69]])#پيش بيني براي داده مورد نظر
print("DecisionTreeClassifier : ",ypred)
#----------------------------------------------

#KNN

#فراخواني الگوريتم مورد نياز
from sklearn.neighbors import KNeighborsClassifier
#در استفاده از الگوريتم KNN بايد اول تعداد همسايه بهينه را پيدا کنيم.
#در اين مسيله فرض ميکنيم که تعداد همسايه بهينه 1 همسايه است.

model=KNeighborsClassifier(n_neighbors=1)#تعريف مدل
model.fit(xtrain,ytrain)#برازش مدل روي داده هاي آموزش
ypred=model.predict([[169,69]])#پيش بيني براي داده مورد نظر
print("KNeighborsClassifier : ",ypred)

















