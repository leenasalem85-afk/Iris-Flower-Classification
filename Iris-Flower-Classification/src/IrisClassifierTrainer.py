import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


# كلاس مسؤول عن: تحميل البيانات، تقسيمها، تدريب النموذج، وتقييم الدقة
class IrisClassifierTrainer:

    def __init__(self, csv_path="Iris.csv"):
        # حفظ مسار ملف البيانات
        self.csv_path = csv_path
        
        # متغيرات سيتم تعبئتها لاحقاً
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.accuracy = None

    # تحميل البيانات وتجهيز X و y
    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

        # حذف عمود Id إذا موجود لأنه غير مفيد للنموذج
        if "Id" in self.df.columns:
            self.df = self.df.drop("Id", axis=1)

        # X = الخصائص ، y = نوع الزهرة
        X = self.df.drop("Species", axis=1)
        y = self.df["Species"]
        return X, y

    # تقسيم البيانات إلى تدريب واختبار
    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.load_data()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # موازنة توزيع الأنواع
        )
        print("Data successfully split into train/test sets.")

    # تدريب النموذج (fit)
    def fit(self):
        # إذا ما تم تقسيم البيانات، نقسّمها الآن
        if self.X_train is None:
            self.split_data()

        # اختيار نموذج Decision Tree
        self.model = DecisionTreeClassifier(random_state=42)

        # تدريب النموذج على بيانات التدريب
        self.model.fit(self.X_train, self.y_train)
        print("Model fitted successfully.")

        joblib.dump(self.model, "iris_model.pkl")
        print("Model saved successfully as iris_model.pkl ")

    # دالة train تستدعي fit (حسب متطلبات المشروع)
    def train(self):
        print("Training model...")
        self.fit()

    # اختبار النموذج وحساب الدقة
    def evaluate(self):
        # إذا ما تم التدريب، ندرب النموذج الآن
        if self.model is None:
            self.train()

        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        return self.accuracy

    # طباعة الدقة
    def print_accuracy(self):
        if self.accuracy is None:
            self.evaluate()
        print(f"Model Accuracy = {self.accuracy * 100:.2f}%")

# تشغيل الكلاس للتجربة
if __name__ == "__main__":
    trainer = IrisClassifierTrainer(csv_path="Iris.csv")
    trainer.train()
    trainer.evaluate()
    trainer.print_accuracy()