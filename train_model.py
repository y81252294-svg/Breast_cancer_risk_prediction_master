import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    df = pd.read_csv('data/data.csv')
    df = df.drop(columns=['id'], errors='ignore')

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis'].map({'B': 0, 'M': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train_scaled, y_train)

    predictions = clf.predict(X_test_scaled)

    print('Accuracy:', accuracy_score(y_test, predictions))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, predictions))
    print('\nClassification Report:\n')
    print(classification_report(y_test, predictions, target_names=['Benign', 'Malignant']))


if __name__ == '__main__':
    main()
