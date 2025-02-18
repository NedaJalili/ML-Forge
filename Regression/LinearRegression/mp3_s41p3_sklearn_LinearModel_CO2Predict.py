#سوال : با توجه به اطلاعات داده شده از خودروها که در فایل اکسل است ،‌ میزان CO2 خودرویی با حجم ۱۳۰۰ و وزن ۲۳۰۰ چقدر است؟ تولید شده برای 

import pandas as pd #فراخواني ماژول pandas براي کار با داده ها
from sklearn.linear_model import LinearRegression # براي برازش خطي

# خواندن داده‌ها
data = pd.read_excel("car.xlsx")

# انتخاب ویژگی‌ها و هدف
X = data[["Volume", "Weight"]]
y = data["CO2"]

# ایجاد و آموزش مدل
model = LinearRegression()
model.fit(X, y)

# تبدیل داده جدید به DataFrame
new_car = pd.DataFrame([[1300, 2300]], columns=["Volume", "Weight"])
ypred = model.predict(new_car)

print("Predicted CO2 emission:", ypred[0])

#استخراج ضرايب معادله
print(model.coef_)
