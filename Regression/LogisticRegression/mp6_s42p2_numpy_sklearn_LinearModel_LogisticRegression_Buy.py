#اطلاعات مربوط به فروش عطری با قیمت 150 و 250 هزار تومان در جدول قرار گرفته است. با توجه به این اطلاعات آقای 26 ساله با سطح درآمد 5 میلیون تومانی عطر 250 تومانی را خرید می کند؟
#جنسیت )آقا 1 / خانم 0 )
#خرید 1 / عدم خرید


#فراخواني ماژول نامپاي براي استفاده از داده ها و تعريف داده ها به صورت آرايه
import numpy as np
data=np.array([[18,1,1,250,0],[20,0,1,150,0],[22,0,2,150,1],[23,1,2,250,1],
[24,1,3,150,0],[24,0,2,250,0],[24,0,3,150,1],[25,0,2,250,0],
[25,0,1,150,0],[25,1,3,150,1],[25,1,5,250,1],[25,0,5,250,1],
[26,0,1,150,0],[26,0,2,250,0],[26,0,3,150,1],[27,1,3,250,1],
[27,1,5,250,1],[27,1,7,250,1],[27,0,5,150,1],[27,0,7,250,1],
[27,0,7,150,1],[28,1,3,250,0],[28,1,5,250,1],[28,1,7,250,1],
[28,1,10,250,1],[29,0,10,250,1],[30,1,5,150,1],[30,0,7,250,1]])

print(data)

#تقسيم داده ها به دو بخش ويژگي و برچسب
x=data[:,:-1]
y=data[:,-1]

#تقسيم داده ها به دو بخش آموزشي و تستي
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)

#فراخواني رگرسيون لجستيک براي کار داده هاي با برچسب هاي گسسته
from sklearn.linear_model import LogisticRegression
#تعريف و آموزش مدل
model=LogisticRegression()
model.fit(xtrain,ytrain)

#پيش بيني کلي 
ypred=model.predict(xtest)
print("ypred",ypred)

#پيش بيني خريد يک مورد مد نظر
theMan=[[26,1,5,250]]
ypred_theMan=model.predict(theMan)
print("ypred_theMan : ",ypred_theMan)

#مقايسه نتايج واقعي و نتايج پيش بيني شده
import pandas as pd
df=pd.DataFrame({"ytest":ytest,"ypred":ypred})
print(df)

#بررسي  دقيق تر نتايج با توابع مورد نياز
from sklearn import metrics
# نمایش ماتریس درهم‌ریختگی
print(metrics.confusion_matrix(ytest,ypred))
# نمایش گزارش طبقه‌بندی (precision, recall, f1-score)
print(metrics.classification_report(ytest,ypred))
# محاسبه و نمایش دقت مدل
print('Accuracy : ',metrics.accuracy_score(ytest,ypred))
