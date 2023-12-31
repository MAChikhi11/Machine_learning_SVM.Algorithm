import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QComboBox, QFileDialog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

class SVMApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('SVM Classifier App')
        self.setGeometry(300, 300, 400, 300)

        # Paramètres SVM
        self.C_label = QLabel('C value:')
        self.C_entry = QLineEdit('30.0')
        self.gamma_label = QLabel('Gamma value:')
        self.gamma_entry = QLineEdit('0.003')
        self.kernel_label = QLabel('Kernel:')
        self.kernel_combobox = QComboBox()
        self.kernel_combobox.addItems(['linear', 'poly', 'rbf'])
        self.kernel_combobox.currentIndexChanged.connect(self.handle_kernel_change)

        # Paramètre degree pour kernel 'poly'
        self.degree_label = QLabel('Degree value:')
        self.degree_entry = QLineEdit('3')
        self.degree_label.setEnabled(False)
        self.degree_entry.setEnabled(False)

        # Bouton d'entraînement du modèle
        self.train_button = QPushButton('Train the model')
        self.train_button.clicked.connect(self.train_model)

        # Bouton pour charger les données depuis un fichier CSV
        self.load_data_button = QPushButton('Load data from Iris CSV file')
        self.load_data_button.clicked.connect(self.load_data)

        # Afficher la matrice de confusion et l'accuracy
        self.confusion_label = QLabel('Confusion matrix:')
        self.accuracy_label = QLabel('Accuracy:')

        # Section de prédiction
        self.prediction_label = QLabel('Prediciton:')
        self.sepal_length_label = QLabel('sepal length (cm):')
        self.sepal_length_entry = QLineEdit()
        self.sepal_width_label = QLabel('sepal width (cm):')
        self.sepal_width_entry = QLineEdit()
        self.petal_length_label = QLabel('petal length (cm):')
        self.petal_length_entry = QLineEdit()
        self.petal_width_label = QLabel('petal width (cm):')
        self.petal_width_entry = QLineEdit()

        # Bouton de prédiction
        self.predict_button = QPushButton('Prédire')
        self.predict_button.clicked.connect(self.predict)

        # Résultat de la prédiction
        self.result_label = QLabel('')

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_data_button)
        layout.addWidget(self.C_label)
        layout.addWidget(self.C_entry)
        layout.addWidget(self.gamma_label)
        layout.addWidget(self.gamma_entry)
        layout.addWidget(self.kernel_label)
        layout.addWidget(self.kernel_combobox)
        layout.addWidget(self.degree_label)
        layout.addWidget(self.degree_entry)
        layout.addWidget(self.train_button)
        layout.addWidget(self.confusion_label)
        layout.addWidget(self.accuracy_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.sepal_length_label)
        layout.addWidget(self.sepal_length_entry)
        layout.addWidget(self.sepal_width_label)
        layout.addWidget(self.sepal_width_entry)
        layout.addWidget(self.petal_length_label)
        layout.addWidget(self.petal_length_entry)
        layout.addWidget(self.petal_width_label)
        layout.addWidget(self.petal_width_entry)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        # Data variables
        self.data = None

    def handle_kernel_change(self):
        # Disable or enable the gamma and degree entry based on the selected kernel
        selected_kernel = self.kernel_combobox.currentText()
        self.gamma_label.setEnabled(selected_kernel != 'linear')
        self.gamma_entry.setEnabled(selected_kernel != 'linear')
        self.degree_label.setEnabled(selected_kernel == 'poly')
        self.degree_entry.setEnabled(selected_kernel == 'poly')

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Charger les données depuis un fichier CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            # Charger les données à partir du fichier CSV
            self.data = pd.read_csv(file_path)
            self.result_label.setText(f"Chargé avec succès depuis {file_path}")

    def train_model(self):
        if self.data is None:
            self.result_label.setText("Veuillez charger les données d'abord.")
            return

        X = self.data.drop('variety', axis=1, errors='ignore')
        y = self.data['variety']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=16)

        svm_kernel = self.kernel_combobox.currentText()
        gamma_value = float(self.gamma_entry.text()) if svm_kernel != 'poly' else 1
        degree_value = int(self.degree_entry.text()) if svm_kernel == 'poly' else 1

        self.svm_model = SVC(
            C=float(self.C_entry.text()),
            gamma=gamma_value,
            kernel=svm_kernel,
            degree=degree_value,
            decision_function_shape='ovo'
        )

        # Ensure feature names are provided during fitting
        self.svm_model.fit(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        self.svm_model.fit(X_train_scaled, y_train)

        # Transform the test set with feature names
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        y_pred = self.svm_model.predict(X_test_scaled)

        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        self.confusion_label.setText(f"Matrice de confusion:\n{confusion}")
        self.accuracy_label.setText(f"Accuracy: {accuracy}")

        self.scaler = scaler


    def predict(self):
        if self.data is None:
            self.result_label.setText("Veuillez charger les données d'abord.")
            return

        column_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        values = [
            float(self.sepal_length_entry.text()),
            float(self.sepal_width_entry.text()),
            float(self.petal_length_entry.text()),
            float(self.petal_width_entry.text())
        ]

        data_to_predict = [values]
        scaled_data = self.scaler.transform(data_to_predict)

        prediction = self.svm_model.predict(scaled_data)

        self.result_label.setText(f"Classe prédite : {prediction[0]}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SVMApp()
    ex.show()
    sys.exit(app.exec_())

#%%
