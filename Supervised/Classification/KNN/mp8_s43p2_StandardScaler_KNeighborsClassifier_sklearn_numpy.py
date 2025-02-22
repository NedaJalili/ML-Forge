#داده هاي کانديداهاي پذيرش در يک دانشگاه داده شده است
#تعداد همسايه هاي بهينه را  به دست آوريد

# وارد کردن کتابخانه‌های مورد نیاز
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# داده‌های کاندیداها
candidates = {
    'gmat': [780, 750, 690, 710, 680, 730, 690, 720, 740, 690, 610, 690, 710, 680, 770, 610, 580, 650, 540, 590, 620, 600, 550, 550, 570, 670, 660, 580, 650, 660, 640, 620, 660, 660, 680, 650, 670, 580, 590, 690],
    'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.7, 2.3, 3.3, 3.3, 1.7, 2.7, 3.7, 3.7, 3.3, 3.3, 3, 2.7, 3.7, 2.7, 2.3, 3.3, 2, 2.3, 2.7, 3, 3.3, 3.7, 2.3, 3.7, 3.3, 3, 2.7, 4, 3.3, 3.3, 2.3, 2.7, 3.3, 1.7, 3.7],
    'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5, 1, 3, 5, 6, 4, 3, 1, 4, 6, 2, 3, 2, 1, 4, 1, 2, 6, 4, 2, 6, 5, 1, 2, 4, 6, 5, 1, 2, 1, 4, 5],
    'admitted': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
}

# تبدیل داده‌ها به DataFrame
data = pd.DataFrame(candidates)

# ویژگی‌ها و برچسب‌ها را جدا می‌کنیم
x = data[['gmat', 'gpa', 'work_experience']]
y = data['admitted']

# استانداردسازی داده‌ها برای مقایسه بهتر
scaler = StandardScaler()
scaler.fit(x)  # یادگیری مقادیر استاندارد از داده‌ها
x_scaled = scaler.transform(x)  # اعمال استانداردسازی به داده‌ها

# تقسیم داده‌ها به مجموعه‌های آموزش و آزمون
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# آماده‌سازی مدل KNN و محاسبه خطاها برای k‌های مختلف
error = []  # لیستی برای ذخیره خطاها
max_len = len(xtrain)  # تعداد نمونه‌های آموزشی
for k in range(1, max_len):
    model = KNeighborsClassifier(n_neighbors=k)  # ساخت مدل KNN با k همسایه
    model.fit(xtrain, ytrain)  # آموزش مدل با داده‌های آموزشی
    ypred = model.predict(xtest)  # پیش‌بینی برچسب‌ها برای داده‌های آزمون
    error.append(np.mean(ypred != ytest))  # محاسبه خطا: تعداد پیش‌بینی‌های نادرست تقسیم بر تعداد کل پیش‌بینی‌ها

# رسم نمودار خطاها
plt.plot(range(1, max_len), error)  # رسم خطای مدل برای k‌های مختلف
plt.xlabel("n_neighbors")  # برچسب محور افقی
plt.ylabel("error")  # برچسب محور عمودی
plt.title("Error vs. Number of Neighbors")  # عنوان نمودار
plt.show()  # نمایش نمودار
