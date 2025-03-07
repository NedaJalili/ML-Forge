#یک مدل درخت تصمیم (Decision Tree) برای طبقه‌بندی داده‌ها پیاده‌سازی کنید.
#ابتدا داده‌های ویژگی و برچسب‌های کلاس را تعریف کنید، سپس مدل DecisionTreeClassifier را روی این داده‌ها آموزش دهید. 
#در نهایت، خروجی پیش‌بینی مدل برای یک نمونه جدید را چاپ کرده و ساختار درخت تصمیم را ترسیم کنید.

import matplotlib.pyplot as plt#اضافه کردن کتابخانه‌ موردنیاز برای رسم نمودار
from sklearn.tree import DecisionTreeClassifier#اضافه کردن DecisionTreeClassifier برای ساخت مدل درخت تصمیم
from sklearn.tree import plot_tree#   اضافه کردن plot_tree برای رسم درخت تصمیم

#تقسيم داده ها به دو بخش ويژگي و برچسب
x =[[19.5,3,1,3],[16.5,0,1,4],[15,0,0,3],[17,2,1,2.5],[18.5,2,0,2.5],[15.5,1,1,2.5],[19,3,1,3]]#ويژگي
y = [1,0,0,1,1,0,1]#برچسب

model= DecisionTreeClassifier()#  ایجاد یک مدل درخت تصمیم از طریق DecisionTreeClassifier
model.fit(x, y) # آموزش مدل روی داده‌های ورودی (x) و خروجی (y)
print(model.predict([[15.5,5,0,3]]))#  پیش‌بینی کلاس یک نمونه جدید با ویژگی‌های مشخص
plot_tree(model)#  نمایش گرافیکی ساختار درخت تصمیم
plt.show()#نمایش نمودار درخت تصمیم
