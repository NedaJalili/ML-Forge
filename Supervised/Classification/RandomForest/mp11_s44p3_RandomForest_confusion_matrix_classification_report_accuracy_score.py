<<<<<<< HEAD
#داده هاي کانديداهاي پذيرش در يک دانشگاه داده شده است


# اضافه کردن کتابخانه pandas برای کار با داده‌ها
import pandas as pd 

# تعریف یک دیکشنری شامل اطلاعات داوطلبان و ویژگی‌های آن‌ها
candidates = {
    'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
    'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
    'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
    'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
} 

# تبدیل دیکشنری به DataFrame برای پردازش داده‌ها
df = pd.DataFrame(candidates, columns=['gmat', 'gpa', 'work_experience', 'admitted'])

# جدا کردن ویژگی‌های ورودی (X) و برچسب خروجی (y)
X = df[['gmat', 'gpa', 'work_experience']] # ویژگی‌های ورودی شامل نمره GMAT، معدل (GPA) و سابقه کار
y = df['admitted'] # متغیر خروجی نشان‌دهنده پذیرش یا عدم پذیرش داوطلب

# اضافه کردن کتابخانه‌های موردنیاز برای تقسیم داده‌ها به دو مجموعه آموزشی و آزمایشی
from sklearn.model_selection import train_test_split

# تقسیم داده‌ها به دو بخش آموزشی و آزمایشی (۷۰٪ آموزش، ۳۰٪ تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# اضافه کردن کتابخانه RandomForestClassifier برای ساخت مدل جنگل تصادفی
from sklearn.ensemble import RandomForestClassifier

# ایجاد مدل جنگل تصادفی با ۵۰ درخت تصمیم‌گیری
model = RandomForestClassifier(n_estimators=50)

# آموزش مدل بر روی داده‌های آموزشی
model.fit(X_train, y_train)

# پیش‌بینی برچسب کلاس برای داده‌های آزمایشی
y_pred = model.predict(X_test)

# نمایش خروجی پیش‌بینی شده برای داده‌های آزمایشی
print(y_pred)

# اضافه کردن کتابخانه‌های موردنیاز برای ارزیابی مدل
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# محاسبه و نمایش ماتریس سردرگمی (Confusion Matrix)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

# محاسبه و نمایش گزارش ارزیابی (Precision, Recall, F1-score)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

# محاسبه و نمایش دقت مدل (Accuracy)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)
=======
#داده هاي کانديداهاي پذيرش در يک دانشگاه داده شده است


# اضافه کردن کتابخانه pandas برای کار با داده‌ها
import pandas as pd 

# تعریف یک دیکشنری شامل اطلاعات داوطلبان و ویژگی‌های آن‌ها
candidates = {
    'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
    'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
    'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
    'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
} 

# تبدیل دیکشنری به DataFrame برای پردازش داده‌ها
df = pd.DataFrame(candidates, columns=['gmat', 'gpa', 'work_experience', 'admitted'])

# جدا کردن ویژگی‌های ورودی (X) و برچسب خروجی (y)
X = df[['gmat', 'gpa', 'work_experience']] # ویژگی‌های ورودی شامل نمره GMAT، معدل (GPA) و سابقه کار
y = df['admitted'] # متغیر خروجی نشان‌دهنده پذیرش یا عدم پذیرش داوطلب

# اضافه کردن کتابخانه‌های موردنیاز برای تقسیم داده‌ها به دو مجموعه آموزشی و آزمایشی
from sklearn.model_selection import train_test_split

# تقسیم داده‌ها به دو بخش آموزشی و آزمایشی (۷۰٪ آموزش، ۳۰٪ تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# اضافه کردن کتابخانه RandomForestClassifier برای ساخت مدل جنگل تصادفی
from sklearn.ensemble import RandomForestClassifier

# ایجاد مدل جنگل تصادفی با ۵۰ درخت تصمیم‌گیری
model = RandomForestClassifier(n_estimators=50)

# آموزش مدل بر روی داده‌های آموزشی
model.fit(X_train, y_train)

# پیش‌بینی برچسب کلاس برای داده‌های آزمایشی
y_pred = model.predict(X_test)

# نمایش خروجی پیش‌بینی شده برای داده‌های آزمایشی
print(y_pred)

# اضافه کردن کتابخانه‌های موردنیاز برای ارزیابی مدل
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# محاسبه و نمایش ماتریس سردرگمی (Confusion Matrix)
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

# محاسبه و نمایش گزارش ارزیابی (Precision, Recall, F1-score)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

# محاسبه و نمایش دقت مدل (Accuracy)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)
>>>>>>> 8e665af (Your commit message)
