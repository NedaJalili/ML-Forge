#   سوال: چگونه می‌توان یک مدل رگرسیون خطی ساخت که داده‌های یک بعدی را برازش کند و دقت آن را ارزیابی کرد؟ 

#  در این کد از تابع np.poly1d برای ساخت مدل استفاده شده است

import numpy as np  # کتابخانه numpy را برای کار با آرایه‌های عددی وارد می‌کنم.

# داده‌های x و y را به صورت آرایه تعریف می‌کنم.
x = np.array([0, 2, 4, 6, 8, 10, 12])  
y = np.array([0, 2, 4, 6, 8, 10, 12])  

# مدل خطی را با استفاده از polyfit و poly1d از کتابخانه numpy می‌سازم.
# این مدل یک چندجمله‌ای درجه ۱ (یعنی یک خط) را روی داده‌ها برازش می‌کند.
model = np.poly1d(np.polyfit(x, y, 1))  

# از مدل برای پیش‌بینی مقادیر y در نقاط x استفاده می‌کنم.
ypred = model(x)  

# نمایش مقادیر واقعی y و مقادیر پیش‌بینی شده ypred
print("y =", y)  
print("ypred =", ypred)  

# محاسبه و نمایش ضریب تعیین (R2 Score) برای ارزیابی دقت مدل
from sklearn.metrics import r2_score  
print("r2 =", r2_score(y, ypred))  

# رسم نمودار داده‌ها و خط برازش شده
import matplotlib.pyplot as plt  
plt.scatter(x, y, label="data")  # رسم نقاط داده‌ای
plt.plot(x, y, label="fit")  # رسم خط برازش شده
plt.xlabel("x")  # برچسب محور x
plt.ylabel("y")  # برچسب محور y
plt.legend()  # نمایش راهنمای نمودار
plt.show()  # نمایش نمودار
 
