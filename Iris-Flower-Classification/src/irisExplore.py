import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Iris.csv")

if "Id" in data.columns:
    data.drop("Id", axis=1, inplace=True)

print("First 3 rows:\n", data.head(3), "\n")
print("Last 3 rows:\n", data.tail(3), "\n")

print("Dataset info:")
data.info()
print("\nSummary statistics:")
print(data.describe())

print("\nNumber of samplse in each class:")
print(data["Species"].value_counts(), "\n")

data.hist(figsize=(10, 6), color="#9D7BB0")
plt.show()

purples = {"Iris-setosa": "#B19CD9", "Iris-versicolor": "#9D7BB0", "Iris-virginica": "#6A0DAD"}

plt.figure(figsize=(8, 6))
for species, color in purples.items():
    subset = data[data["Species"] == species]
    plt.scatter(subset["PetalLengthCm"], subset["PetalWidthCm"], label=species, color=color)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for species, color in purples.items():
    subset = data[data["Species"] == species]
    plt.scatter(subset["SepalLengthCm"], subset["SepalWidthCm"], label=species, color=color)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")
plt.legend()
plt.show()
