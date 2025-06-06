Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import numpy as np
... import pandas as pd
... from sklearn.model_selection import train_test_split
... from sklearn.tree import DecisionTreeClassifier
... from sklearn.metrics import classification_report
... 
... # Step 1: Generate synthetic data
... def generate_data(num_samples=500):
...     np.random.seed(42)
...     data = []
...     labels = []
...     for _ in range(num_samples):
...         # Simulate different driving styles
...         style = np.random.choice(['aggressive', 'normal', 'cautious'])
...         
...         if style == 'aggressive':
...             speed = np.random.normal(100, 10)       # higher speed
...             acceleration = np.random.normal(4, 1)   # stronger acceleration
...             steering = np.random.normal(30, 5)      # sharper turns
...         elif style == 'normal':
...             speed = np.random.normal(70, 8)
...             acceleration = np.random.normal(2, 0.5)
...             steering = np.random.normal(15, 3)
...         else:  # cautious
...             speed = np.random.normal(50, 5)
...             acceleration = np.random.normal(1, 0.2)
...             steering = np.random.normal(5, 2)
... 
...         data.append([speed, acceleration, steering])
...         labels.append(style)
...     
...     df = pd.DataFrame(data, columns=['speed', 'acceleration', 'steering_angle'])
...     df['label'] = labels
...     return df
... 
... # Step 2: Prepare dataset
... df = generate_data()
X = df[['speed', 'acceleration', 'steering_angle']]
y = df['label']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Step 5: Predict & evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Optional: predict a new sample
new_sample = np.array([[85, 3, 20]])
predicted_style = clf.predict(new_sample)
print(f"Predicted driving style for new sample: {predicted_style[0]}")
