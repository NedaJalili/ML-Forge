import pandas as pd  # اضافه کردن کتابخانه Pandas برای کار با داده‌ها

# تعریف یک دیکشنری شامل اطلاعات داوطلبان و ویژگی‌های آن‌ها
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
}

# تبدیل دیکشنری به DataFrame برای پردازش داده‌ها
df = pd.DataFrame(candidates, columns=['gmat', 'gpa', 'work_experience', 'admitted'])

# جدا کردن ویژگی‌های ورودی (X) و برچسب خروجی (y)
X = df[['gmat', 'gpa', 'work_experience']]  # ویژگی‌های ورودی شامل نمره GMAT، معدل (GPA) و سابقه کار
y = df['admitted']  # متغیر خروجی نشان‌دهنده پذیرش یا عدم پذیرش داوطلب

# اضافه کردن کتابخانه‌های موردنیاز برای آموزش مدل درخت تصمیم
from sklearn.model_selection import train_test_split  # تقسیم داده‌ها به دو بخش آموزشی و آزمایشی
from sklearn.tree import DecisionTreeClassifier  # اضافه کردن DecisionTreeClassifier برای ساخت مدل درخت تصمیم

# تقسیم داده‌ها به مجموعه آموزشی (۷۵٪) و آزمایشی (۲۵٪) با مقدار تصادفی ثابت
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ایجاد مدل درخت تصمیم
model = DecisionTreeClassifier()

# آموزش مدل بر روی داده‌های آموزشی
model.fit(X_train, y_train)

# محاسبه و نمایش دقت مدل بر روی داده‌های آموزشی
print("Training set accuracy: {:.3f}".format(model.score(X_train, y_train)))

# محاسبه و نمایش دقت مدل بر روی داده‌های آزمایشی
print("Test set accuracy: {:.3f}".format(model.score(X_test, y_test)))

# پیش‌بینی برچسب کلاس برای نمونه‌های آزمایشی
y_pred = model.predict(X_test)

# نمایش خروجی پیش‌بینی شده برای داده‌های آزمایشی
print(y_pred)

# اضافه کردن کتابخانه NumPy برای محاسبه خطای پیش‌بینی
import numpy as np

# محاسبه میزان خطای پیش‌بینی با تفریق مقدار واقعی از مقدار پیش‌بینی شده
percentage_error_tree = (y_test - y_pred)

# محاسبه میانگین خطای پیش‌بینی شده
print(np.mean(percentage_error_tree))

# اضافه کردن کتابخانه‌های لازم برای نمایش درخت تصمیم
import matplotlib.pyplot as plt  # اضافه کردن Matplotlib برای رسم نمودار
from sklearn.tree import plot_tree  # اضافه کردن plot_tree برای نمایش درخت تصمیم

# رسم نمودار درخت تصمیم آموزش‌داده‌شده
plot_tree(model)

# نمایش نمودار درخت تصمیم
plt.show()
