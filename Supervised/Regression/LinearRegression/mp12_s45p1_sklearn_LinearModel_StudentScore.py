#پيش بيني نمرات دانشجويان بر اساس مدت زمان مطالعه (برحسب ساعت)‌(با فرض اينکه نمره دانشجويان فقط به ساعت مطالعه وابسته است)
# کدام الگوريتم مناسب است؟
#داده ها برچسب دارند>الگوريتم هاي بانظارت----برچسب ها پيوسته هستند > يکي از الگوريتم هاي مناسب LinearRegressionاست


#فراخواني کتابخانه براي ساماندهي  داده ها و کار با آنها
import pandas as pd
#خواندن داده ها از فايل مربوطه 
data=pd.read_csv("student_scores.csv")

#کسب اطلاعات از داده ها
print(data.shape)#ابعاد داه ها
print(data)#خود فايل
print(data.describe)# اطلاعات سازمان يافته از داده ها

#فراخواني کتابخانه براي ترسيم نمودار
import matplotlib.pyplot as plt
data.plot(x="Hours",y="Scores",style="o")#ترسيم نمودار داده  ها
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


#تا اینجا از داده ها اطلاعات کسب کردیم. رفتار داده ها را بررسی کردیم  
#و در نهایت با توجه به رفتار داده ها که رفتاری خطی است و نیز دانستن اینکه داده ها برچسب دارند، تصمیم گرفتیم که از الگوریتمLinearRegressionاستفاده کنیم

#-------------------------------------------------

x=data.iloc[:,:-1].values #تعريف ويژگي ها:ستون اول از دو ستون
y=data.iloc[:,-1] #تعريف برچسب ها

#فراخواني کتابخانه براي تقسيم داده ها به دو بخش آموزش و آزمون
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)#تقسيم داده ها به دو بخش آموزش و آزمون

#فراخواني الگوريتم مورد نياز
from sklearn.linear_model import LinearRegression
model=LinearRegression()#تعريف مدل
model.fit(xtrain,ytrain)#برازش مدل روي داده هاي آموزش
ypred=model.predict(xtest)#پيش بيني مدل براي داده هاي آزمون

#-------------------------------------------------
#شاخص هاي ارزيابي را براي اين مدل بررسي کنيد

import  numpy as np
from sklearn import metrics
print("mean=",np.mean(y))
print("mae=",metrics.mean_absolute_error(ytest,ypred))
print("mse=",metrics.mean_squared_error(ytest,ypred))
print("rmse=",np.sqrt(metrics.mean_squared_error(ytest,ypred)))

#چون مدل ما خطي است، مي توان شيب و عرض از مبدا را هم استخراج کرد
print(model.intercept_)#عرض از مبدا
print(model.coef_)#شيب
