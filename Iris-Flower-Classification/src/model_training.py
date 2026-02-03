import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# كلاس مسؤول عن: تحميل البيانات، تقسيمها، تدريب النموذج، وتقييمه
class IrisModelTrainer:

    def __init__(self, csv_path="iris.csv"):     # تهيئة الكلاس وتجهيز المتغيرات الأساسية

        self.csv_path = csv_path        # حفظ مسار ملف البيانات
        self.df = None           # متغير لحفظ البيانات بعد قراءتها

        # متغيرات لحفظ بيانات التدريب والاختبار
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None         # متغير لحفظ النموذج بعد التدريب


    def load_data(self):      # دالة لقراءة البيانات من ملف CSV وتجهيز X (الخصائص) و y (النوع)

        # قراءة ملف CSV
        self.df = pd.read_csv(self.csv_path)

        if "Id" in self.df.columns: # حذف عمود Id لأنه ما نحتاجه في التدريب
            self.df = self.df.drop("Id", axis=1)

        X = self.df.drop("Species", axis=1)     # X = كل الأعمدة ماعدا Species

        y = self.df["Species"]         # y = عمود النوع Species

        return X, y

    def split_data(self, test_size=0.2, random_state=42): # دالة لتقسيم البيانات إلى تدريب واختبار باستخدام train_test_split

        X, y = self.load_data()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,  # نسبة بيانات الاختبار
            random_state=random_state, # تثبيت النتائج
            stratify=y  # عشان يكون توزيع الأنواع متوازن في train و test
        )

        print("Data has been split into training and testing sets ✅ .")

    def train_model(self):  # دالة لتدريب نموذج Decision Tree على بيانات التدريب

        # لو لسه ما سوينا split، نسويه أول
        if self.X_train is None or self.y_test is None:
            self.split_data()

        # نختار خوارزمية Decision Tree
        self.model = DecisionTreeClassifier(random_state=42)
        # تدريب النموذج باستخدام بيانات التدريب
        self.model.fit(self.X_train, self.y_train)

        print("Model has been trained successfully ✅.")

    def evaluate(self):     # دالة لاختبار النموذج على بيانات الاختبار وقياس الدقة

        if self.model is None:     # لو النموذج غير مدرّب، ندربه أول
            self.train_model()

        # التنبؤ بأنواع الزهور باستخدام بيانات الاختبار
        y_pred = self.model.predict(self.X_test)

        # حساب دقة النموذج
        acc = accuracy_score(self.y_test, y_pred)

        print(f"📊Model accuracy: {acc:.2f} ")
        return acc
#اختبار الكلاس
if __name__ == "__main__":
   # إنشاء كائن من الكلاس
   trainer = IrisModelTrainer(csv_path="iris.csv")

   # تقسيم البيانات
   trainer.split_data()

   # تدريب النموذج
   trainer.train_model()

   # تقييم النموذج
   trainer.evaluate()
