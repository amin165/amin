import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Titanic dataset.csv')
#  Identifier les types de données  
print("Partie 1 : Introduction au Dataset")
print(df.shape)
print(df)
print("Types de données dans le dataset :")
print(df.dtypes) 

#  Visualiser les 5 premières lignes du dataset
print("\n 5 Premières lignes du dataset :")
print(df.head())  # Affiche les 5 premières lignes

#  Vérifier les valeurs manquantes
print("\nNombre de valeurs manquantes par colonne :")
print(df.isnull().sum())



print("Partie 2 : Nettoyage des Données")
  # Imputation avec la moyenne
mean_imputer = SimpleImputer(strategy='mean')  
 # Imputation avec la médiane
median_imputer = SimpleImputer(strategy='median')
col_num = df.select_dtypes(include=['float64', 'int64']).columns

# Imputation avec la moyenne
df[col_num] = mean_imputer.fit_transform(df[col_num])

print(" voir les résultats après imputation avec la moyenne ")
print(df[col_num])

df[col_num] = median_imputer.fit_transform(df[col_num])
print(" Voir les résultats après imputation avec la médiane ")
print(df[col_num])

mode_imputer = SimpleImputer(strategy='most_frequent')
col_cate = df.select_dtypes(include=['object']).columns

df[col_cate] = mode_imputer.fit_transform(df[col_cate])
print(" voir les résultats après imputation avec la mode ")
print(df[col_cate])
print("Supprimer les colonnes avec plus de 40 % de valeurs manquantes")

missing_percentage = df.isnull().mean() * 100

# Filtrer les colonnes avec plus de 40 % de valeurs manquantes
columns_to_drop = missing_percentage[missing_percentage > 40].index

# Supprimer les colonnes avec plus de 40 % de valeurs manquantes
df = df.drop(columns=columns_to_drop)
print(df)

# 1. Visualisation avec boxplot pour détecter les valeurs aberrantes
for column in col_num:
    plt.figure(figsize=(1, 1))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot de {column}')
    plt.show()
# 2. Détection des valeurs aberrantes et gestion :
# Définir les critères pour détecter les valeurs aberrantes 
Q1 = df[col_num].quantile(0.25)
Q3 = df[col_num].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifiez les lignes où il y a des valeurs aberrantes
outliers = ((df[col_num] < lower_bound) | (df[col_num] > upper_bound))
# Option 1: Remplacer les valeurs aberrantes par la médiane de chaque colonne
df_cleaned_median = df.copy()

for column in col_num:
    # Remplacer les valeurs aberrantes par la médiane
    df_cleaned_median[column] = df_cleaned_median[column].where(outliers[column], df[column].median())

# Option 2: Supprimer les lignes avec des valeurs aberrantes
df_cleaned_no_outliers = df.copy()

# Supprimer les lignes où au moins une valeur aberrante est présente
df_cleaned_no_outliers = df_cleaned_no_outliers[outliers.any(axis=1)]
print("Afficher les boxplots après modification")
for column in col_num:
    plt.figure(figsize=(1, 1))
    sns.boxplot(x=df_cleaned_median[column])
    plt.title(f'Boxplot après remplacement par médiane pour {column}')
    plt.show()

# Voir les premières lignes après gestion des valeurs aberrantes 
print("Data après remplacement des valeurs aberrantes par la médiane :")
print(df_cleaned_median)
# Voir les premières lignes après suppression des valeurs aberrantes
print("Data après suppression des  valeurs aberrantes :")
print(df_cleaned_no_outliers)






cat_columns = df.select_dtypes(include=['object']).columns
print("\nColonnes catégorielles :\n", cat_columns)


# Nous appliquerons One-Hot Encoding sur 'Sex' et 'Embarked'
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Appliquer l'encodage numérique pour les variables ordinales
# Supposons que 'Pclass' est une variable ordinale (1: 1ère classe, 2: 2ème classe, 3: 3ème classe)
df_encoded['Pclass'] = df_encoded['Pclass'].astype(int)

# Afficher les premières lignes du dataframe encodé
print("\nDataframe après encodage :\n", df_encoded.head())

# Afficher les colonnes du dataframe après encodage
print("\nColonnes après encodage :\n", df_encoded.columns)

onehot_encoder = OneHotEncoder(sparse_output=False)  
nominal_columns = ['Sex', 'Embarked']  
df_nominal_encoded = pd.DataFrame(onehot_encoder.fit_transform(df[nominal_columns]),columns=onehot_encoder.get_feature_names_out(nominal_columns))

# Ajouter ces nouvelles colonnes au DataFrame
df = pd.concat([df, df_nominal_encoded], axis=1)

# Supprimer les colonnes nominales initiales
df = df.drop(columns=nominal_columns)

df['Pclass'] = df['Pclass'].map({1: 1, 2: 2, 3: 3})

# Vérifiez l'état du DataFrame après transformation
print(df)
print(df[['Pclass']])

nume_columns = df.select_dtypes(include=[np.number]).columns
print("Colonnes numériques avant transformation:")
print(df[nume_columns])

print(" Normalisation")
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df[nume_columns]), columns=nume_columns)
print(df_normalized)

print(" Standardisation")
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df[nume_columns]), columns=nume_columns)
print(df_standardized)

# 3. Comparer les distributions avant et après transformation

for col in nume_columns:
    plt.figure(figsize=(12, 4))

    # Histogramme avant transformation
    plt.subplot(1, 3, 1)
    plt.hist(df[col].dropna(), bins=20, alpha=0.7, color='blue', label='Original')
    plt.title(f'Distribution de {col} (Original)')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

    # Histogramme après Normalisation
    plt.subplot(1, 3, 2)
    plt.hist(df_normalized[col].dropna(), bins=20, alpha=0.7, color='green', label='Normalisé')
    plt.title(f'Distribution de {col} (Normalisation)')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

    # Histogramme après Standardisation
    plt.subplot(1, 3, 3)
    plt.hist(df_standardized[col].dropna(), bins=20, alpha=0.7, color='red', label='Standardisé')
    plt.title(f'Distribution de {col} (Standardisation)')
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.tight_layout()
    plt.show()


# Variable cible
target = 'Survived'
# Afficher la distribution des classes dans 'Survived'
print("Distribution des classes avant sous-échantillonnage :")
print(df[target].value_counts())

X = df.drop(columns=[target])  # Toutes les colonnes sauf la cible
y = df[target] # Variable cible
# Diviser en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un objet RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Appliquer le sous-échantillonnage sur l'ensemble d'entraînement
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Vérifier la distribution des classes après sous-échantillonnage
print("Distribution des classes après sous-échantillonnage :")
print(pd.Series(y_resampled).value_counts())

# Séparer les caractéristiques (X) et la cible (y)
X = df.drop(columns=['Survived'])
y = df['Survived']
print(X_train.dtypes)
X = X.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')
# Encoder les colonnes catégoriques en colonnes numériques (One-Hot Encoding)
print("les columns existe")
print(X.columns)

print("Distribution des classes avant SMOTE :")
print(y.value_counts())

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Appliquer SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Vérifier la distribution des classes après sur-échantillonnage
print("Distribution des classes après sur-échantillonnage (SMOTE) :")
print(pd.Series(y_train_smote).value_counts())







