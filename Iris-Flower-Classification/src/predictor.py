import joblib

class IrisPredictor:
    def __init__(self, trained_model):
        self.model = trained_model

    def predict(self, sample):
        return self.model.predict([sample])[0]

    def predict_proba(self, sample):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba([sample])[0]
        else:
            return "Model does not support probability predictions."

# تحميل النموذج
trained_model = joblib.load("iris_model.pkl")

# إنشاء كائن التنبؤ
predictor = IrisPredictor(trained_model)

# عينات جديدة
sample1 = [5.1, 3.5, 1.4, 0.2]
sample2 = [6.0, 2.9, 4.5, 1.5]
sample3 = [7.6, 3.0, 6.6, 2.1]

# التنبؤ
print("Prediction for sample1:", predictor.predict(sample1))
print("Prediction for sample2:", predictor.predict(sample2))
print("Prediction for sample3:", predictor.predict(sample3))

print("Prediction probabilities for sample1:", predictor.predict_proba(sample1))
print("Prediction probabilities for sample2:", predictor.predict_proba(sample2))
print("Prediction probabilities for sample3:", predictor.predict_proba(sample3))
