#هدف: ایجاد یک مدل رگرسیون لجستیک در پایتون تا تعیین کند که آیا داوطلبان در یک دانشگاه معتبر پذیرفته می شوند یا خیر.
#دو نتیجه احتمالی وجود دارد: پذیرفته شده ) 1 ( در مقابل رد شده ) 0 )
#سپس می توان یک رگرسیون لجستیک در پایتون ایجاد کرد ، جایی که : متغیر وابسته نشان می دهد که آیا فرد پذیرفته می شود. و 3 متغیر مستقل نمره GMAT ،معدل ) GPA (و سالها سابقه کار هستند



import pandas as pd  # کتابخانه pandas برای پردازش داده‌ها

# تعریف داده‌های مربوط به داوطلبان پذیرش
candidates = {
    'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
    'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
    'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
    'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
}

# تبدیل داده‌ها به DataFrame
data = pd.DataFrame(candidates)

# انتخاب ویژگی‌ها (GMAT، GPA و تجربه کاری) و متغیر هدف (پذیرفته شده یا نه)
X = data[['gmat', 'gpa', 'work_experience']]
y = data['admitted']

from sklearn.model_selection import train_test_split  # برای تقسیم داده‌ها به مجموعه‌های آموزشی و تستی

# تقسیم داده‌ها به مجموعه‌های آموزشی (80٪) و تستی (20٪)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression  # مدل رگرسیون لجستیک

# ایجاد و آموزش مدل رگرسیون لجستیک
model = LogisticRegression()
model.fit(X_train, y_train)

# پیش‌بینی پذیرش داوطلبان با داده‌های تستی
y_pred = model.predict(X_test)
print("ypred : ", y_pred)  # نمایش پیش‌بینی‌ها

# ایجاد DataFrame برای مقایسه نتایج واقعی و پیش‌بینی‌شده
df = pd.DataFrame({"ytest": y_test, "ypred": y_pred})
print(df)

from sklearn.metrics import confusion_matrix, classification_report  # ابزارهای ارزیابی مدل

# نمایش ماتریس درهم‌ریختگی (Confusion Matrix)
print(confusion_matrix(y_test, y_pred))

# نمایش گزارش ارزیابی مدل شامل معیارهایی مانند دقت، یادآوری و F1-Score
print(classification_report(y_test, y_pred))
