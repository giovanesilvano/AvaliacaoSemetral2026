import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

base = fetch_ucirepo(id=14)

atributos = base.data.features.copy()
alvo = base.data.targets.squeeze()

mapa_ajustes = {
    "tumor-size": {
        "9-May": "5-9",
        "14-Oct": "10-14"
    },
    "inv-nodes": {
        "5-Mar": "3-5",
        "8-Jun": "6-8",
        "11-Sep": "9-11",
        "14-Dec": "12-14"
    }
}

atributos = atributos.replace(mapa_ajustes)

print("\nFormato dos dados:", atributos.shape)
print("\nContagem de nulos:")
print(atributos.isnull().sum())

print("\nFrequência por classe:")
print(alvo.value_counts())

atr_train, atr_val, alvo_train, alvo_val = train_test_split(
    atributos,
    alvo,
    test_size=0.20,
    random_state=42,
    stratify=alvo
)

print("\nConjunto de treino:", atr_train.shape)
print("Conjunto de teste:", atr_val.shape)

lista_colunas = atributos.columns.tolist()

transformador = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(steps=[
                (
                    "preenchimento",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "codificador",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            lista_colunas
        )
    ]
)

candidatos = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {
            "clf__C": [0.1, 1, 10]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_leaf": [1, 3, 5]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, None]
        }
    ),

    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [3, 5, 7, 9]
        }
    ),

    "Support Vector Machine": (
        SVC(),
        {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"]
        }
    )
}

divisor_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lista_resultados = []
campeao = None
pontuacao_campeao = -1
nome_campeao = ""

for titulo, (estimador, grade) in candidatos.items():

    print(f"\nIniciando treino de: {titulo}")

    fluxo = Pipeline(steps=[
        ("preparo", transformador),
        ("clf", estimador)
    ])

    procura = GridSearchCV(
        estimator=fluxo,
        param_grid=grade,
        cv=divisor_cv,
        scoring="accuracy",
        n_jobs=-1
    )

    procura.fit(atr_train, alvo_train)

    pontuacao = procura.best_score_

    print(f"Média de acurácia na validação cruzada: {pontuacao:.4f}")
    print("Hiperparâmetros ideais:", procura.best_params_)

    lista_resultados.append({
        "Algoritmo": titulo,
        "Acurácia (CV)": pontuacao
    })

    if pontuacao > pontuacao_campeao:
        pontuacao_campeao = pontuacao
        campeao = procura.best_estimator_
        nome_campeao = titulo

ranking = pd.DataFrame(lista_resultados)

ranking = ranking.sort_values(
    by="Acurácia (CV)",
    ascending=False
)

print("\n===== RANKING DOS ALGORITMOS =====")
print(ranking.to_string(index=False))

print("\nAlgoritmo vencedor:", nome_campeao)
print(f"Acurácia (CV) do vencedor: {pontuacao_campeao:.4f}")

predicoes = campeao.predict(atr_val)

acuracia_final = accuracy_score(
    alvo_val,
    predicoes
)

print("\n===== RESULTADO NO CONJUNTO DE TESTE =====")

print(f"Acurácia obtida: {acuracia_final:.4f}")

print("\nRelatório detalhado:")
print(
    classification_report(
        alvo_val,
        predicoes,
        zero_division=0
    )
)

rotulos = [
    "no-recurrence-events",
    "recurrence-events"
]

print("\nMatriz de confusão:")
print(
    confusion_matrix(
        alvo_val,
        predicoes,
        labels=rotulos
    )
)

os.makedirs("saidas", exist_ok=True)

joblib.dump(
    campeao,
    "saidas/modelo_campeao.joblib"
)

ranking.to_csv(
    "saidas/ranking_algoritmos.csv",
    index=False
)

print("\nModelo armazenado com êxito!")
print("Arquivo do modelo: saidas/modelo_campeao.joblib")
print("Arquivo do ranking: saidas/ranking_algoritmos.csv")