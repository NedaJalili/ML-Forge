#با توجه به داده ها قیمت یک آپارتمان 70 متری، یک اتاق، طبقه دوم، سال ساخت در منطقه 2 را تخمین بزنید


#فراخواني ماژول پانداز براي کار با ديتا
import pandas as pd
data=pd.read_excel("houses.xlsx")

#تقسيم داده ها به دو بخش ويژگي و برچسب
x=data[["meterage","rooms","floor","year","region"]]
y=data["price"]

#فراخواني  رگرسيون خطي
from sklearn.linear_model import LinearRegression

#ايجاد و آموزش مدل
model=LinearRegression()
model.fit(x,y)

#پيش بيني کردن مدل
ypred=model.predict(x)
print("ypred : ",ypred)

#پيش بيني براي مورد مدنظر
terget_apartment=pd.DataFrame([[70, 1, 2, 17, 2]], columns=["meterage",
"rooms", "floor", "year", "region"])
ypred1=model.predict(terget_apartment)
print("price of the terget apartment :",ypred1)
