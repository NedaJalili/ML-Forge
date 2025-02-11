#   سوال: چگونه می‌توان یک مدل رگرسیون خطی ساخت که داده‌های یک بعدی را برازش کند و دقت آن را ارزیابی کرد؟ 



import numpy as np  # کتابخانه‌ای برای کار با آرایه‌های عددی

# تعریف داده‌های ورودی (x) و خروجی (y)
x = np.array([0, 2, 4, 6, 8, 10, 12])
y = np.array([0, 2, 4, 6, 8, 10, 12])

# تبدیل x به آرایه دو بعدی برای استفاده در مدل
X = x[:, np.newaxis]

# نمایش آرایه‌های x و X برای بررسی شکل داده‌ها
print(x)
print(X)

from sklearn.linear_model import LinearRegression  # ایمپورت مدل رگرسیون خطی

# ساخت مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل با داده‌های موجود
model.fit(X, y)

# پیش‌بینی مقدار y بر اساس ورودی X
ypred = model.predict(X)

from sklearn.metrics import r2_score  # ایمپورت متریک R² برای ارزیابی مدل

# محاسبه و نمایش مقدار R² (ضریب تعیین) برای بررسی دقت مدل
print("r2 = ", r2_score(y, ypred))

import matplotlib.pyplot as plt  # ایمپورت کتابخانه رسم نمودار

# رسم نقاط داده‌های واقعی
plt.scatter(x, y, label="data")

# رسم خط برازش شده توسط مدل
plt.plot(X, ypred, label="fit")

# تنظیم برچسب‌های محورهای نمودار
plt.xlabel("x")
plt.ylabel("y")

# نمایش راهنمای نمودار
plt.legend()

# نمایش نمودار
plt.show()
