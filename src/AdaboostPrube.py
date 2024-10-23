import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Carga del dataset seleccionado
file_path = r'diabetes_dataset00.csv'
data = pd.read_csv(file_path)

# Seleccion de las columnas base para el analisis
features = ['Blood Glucose Levels', 'Insulin Levels', 'BMI', 'Family History', 'Age',
            'Genetic Markers', 'Cholesterol Levels', 'Physical Activity', 'Dietary Habits',
            'Blood Pressure', 'Waist Circumference', 'Previous Gestational Diabetes']
target = 'Target'

# Filtrar las columnas
X = data[features]
y = data[target]

# Preprocesamiento de etiquetas (codificación de la variable Target)
le = LabelEncoder()
y = le.fit_transform(y)

# Identificacion de las columnas categóricas y numéricas
categorical_columns = ['Family History', 'Genetic Markers', 'Physical Activity', 'Dietary Habits',
                       'Previous Gestational Diabetes']
numerical_columns = ['Blood Glucose Levels', 'Insulin Levels', 'BMI', 'Age', 'Cholesterol Levels',
                     'Blood Pressure', 'Waist Circumference']

# Crear transformadores para las columnas categóricas y numéricas
#  Usamos OneHotEncoder para las columnas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Dividir el dataset en el conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Definir el pipeline con preprocesamiento y el modelo AdaBoost
adaboost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        n_estimators=100,
        learning_rate=1,
        random_state=42,
        algorithm= 'SAMME'
    ))
])

# Realizar validación cruzada con el pipeline
scores = cross_val_score(adaboost_pipeline, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Entrenar el modelo
adaboost_pipeline.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = adaboost_pipeline.predict(X_test)

# Imprimir el informe de clasificación
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Crear una matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calcular la importancia de las características
feature_importance = adaboost_pipeline.named_steps['classifier'].feature_importances_
feature_names = (numerical_columns +
                 list(adaboost_pipeline.named_steps['preprocessor']
                      .transformers_[1][1].get_feature_names_out(categorical_columns)))

# Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, orient='h')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()