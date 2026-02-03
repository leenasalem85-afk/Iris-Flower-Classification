import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

from predictor import IrisPredictor  # load prediction class


# Load trained model
loaded_model = joblib.load("iris_model.pkl")

# Create predictor object
predictor = IrisPredictor(loaded_model)


def predict_species():
    """Get input values, run prediction, and display result."""
    try:
        # Read input values
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())

        # Create DataFrame with correct feature names
        sample_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )

        # Predict species
        species = loaded_model.predict(sample_df)[0]

        # Predict probabilities (if available)
        if hasattr(loaded_model, "predict_proba"):
            probs = loaded_model.predict_proba(sample_df)[0]
            classes = loaded_model.classes_
            prob_lines = [f"{cls}: {p:.2f}" for cls, p in zip(classes, probs)]
            probs_text = "\n".join(prob_lines)
        else:
            probs_text = "No probability data."

        # Display result (3 lines: setosa, versicolor, virginica)
        result_text = f"Predicted species: {species}\n\n{probs_text}"
        result_label.config(text=result_text)

    except ValueError:
        messagebox.showerror("Input error", "Enter numeric values only.")


# Create GUI window
root = tk.Tk()
root.title("Iris Flower Classifier")
root.geometry("380x380")  # taller window to show all lines

# Input fields
tk.Label(root, text="Sepal Length (cm):").pack(pady=(10, 0))
entry_sepal_length = tk.Entry(root)
entry_sepal_length.pack()

tk.Label(root, text="Sepal Width (cm):").pack(pady=(10, 0))
entry_sepal_width = tk.Entry(root)
entry_sepal_width.pack()

tk.Label(root, text="Petal Length (cm):").pack(pady=(10, 0))
entry_petal_length = tk.Entry(root)
entry_petal_length.pack()

tk.Label(root, text="Petal Width (cm):").pack(pady=(10, 0))
entry_petal_width = tk.Entry(root)
entry_petal_width.pack()

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_species)
predict_button.pack(pady=15)

# Result label (expanded to show all result lines)
result_label = tk.Label(root, text="", justify="left", anchor="w")
result_label.pack(pady=10, fill="both")

# Run GUI
if __name__ == "__main__":
    root.mainloop()