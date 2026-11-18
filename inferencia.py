import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

ARQUIVO_CAMPEAO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saidas",
    "modelo_campeao.joblib"
)

if not os.path.exists(ARQUIVO_CAMPEAO):
    print("Arquivo do modelo não localizado!")
    print("Rode antes o script de treino: python treinamento.py")
    raise SystemExit(1)

campeao = joblib.load(ARQUIVO_CAMPEAO)

base = fetch_ucirepo(id=14)
atributos = base.data.features.copy()

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

lista_colunas = atributos.columns.tolist()

alternativas = {}

for col in lista_colunas:
    alternativas[col] = sorted(
        atributos[col].dropna().astype(str).unique().tolist()
    )

print("\n===== PREDITOR DE CÂNCER DE MAMA =====")
print("Preencha as informações da nova paciente.\n")

dados_paciente = {}

for col in lista_colunas:

    possiveis = alternativas[col]

    print(f"\nVariável: {col}")

    for posicao, item in enumerate(possiveis, start=1):
        print(f"{posicao} - {item}")

    while True:
        resposta = input("Selecione uma alternativa: ").strip()

        if resposta.isdigit():
            posicao = int(resposta)

            if 1 <= posicao <= len(possiveis):
                item = possiveis[posicao - 1]

                if col == "deg-malig":
                    dados_paciente[col] = int(item)
                else:
                    dados_paciente[col] = item

                break

        print("Alternativa inválida. Tente de novo.")

entrada_modelo = pd.DataFrame(
    [dados_paciente],
    columns=lista_colunas
)

predicao = campeao.predict(entrada_modelo)[0]

print("\n===== RESULTADO DA PREDIÇÃO =====")

print("\nInformações fornecidas:")
print(entrada_modelo.to_string(index=False))

print("\nClasse estimada:", predicao)

if predicao == "no-recurrence-events":
    print("Interpretação: NÃO há indício de recorrência.")

elif predicao == "recurrence-events":
    print("Interpretação: HÁ indício de recorrência.")
