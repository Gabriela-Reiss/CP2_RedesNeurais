# 🧠 **CP2 - Redes Neurais e Visão Computacional**

Este projeto tem como objetivo o **desenvolvimento e a aplicação de Redes Neurais Artificiais (RNA)** para **classificação e regressão**, além da criação e teste de um **modelo de Visão Computacional** capaz de diferenciar imagens de **gatos e cachorros**.

---

## 📂 Estrutura do Projeto

O trabalho está dividido em **duas partes principais**, cada uma em um notebook distinto:

| Parte | Arquivo | Descrição |
|-------|----------|------------|
| **Parte 1** | `Parte_1_Redes_Neurais.ipynb` | Modelos de Classificação e Regressão utilizando Redes Neurais e algoritmos clássicos de Machine Learning |
| **Parte 2** | `Parte_2_Visão_Computacional.ipynb` | Criação e teste de um modelo de visão computacional baseado em YOLOv8 para detecção de gatos e cachorros |

---

## 🧩 **Parte 1 — Redes Neurais e Modelos de ML**

### 🔹 Tecnologias Utilizadas
- **Python**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **NumPy** e **Pandas**
- **Matplotlib** / **Seaborn**
- **Google Colab**

---

### 🧪 **Exercício 1 — Classificação Multiclasse**

**Dataset:** [Wine Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)

**Etapas:**
1. Importação das bibliotecas necessárias.  
2. Leitura do arquivo `wine.data` (disponível no repositório).  
3. Criação de uma Rede Neural utilizando **Keras**.  
4. Desenvolvimento de modelos de classificação:
   - `RandomForestClassifier`
   - `LogisticRegression`
5. Comparação dos resultados entre os modelos e a rede neural.

---

### 📈 **Exercício 2 — Regressão**

**Dataset:** [California Housing Dataset - Scikit-Learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

**Etapas:**
1. Importação das bibliotecas necessárias.  
2. Criação de uma Rede Neural utilizando **Keras**.  
3. Desenvolvimento de modelos de regressão:
   - `LinearRegression`
   - `RandomForestRegressor`
4. Comparação de desempenho entre os modelos e a rede neural.

---

## 🖼️ **Parte 2 — Visão Computacional**

Nesta parte foi desenvolvido um **modelo de classificação de imagens** para diferenciar **gatos e cachorros**, utilizando **um modelo pré-treinado YOLOv8n (640x640)** disponibilizado pelo **Roboflow**.

### Dataset utilizado
Para realização de testes do modelo, utilizamos o dataset do Kaggle: [Cats And Dogs](https://www.kaggle.com/datasets/marquis03/cats-and-dogs/data)

### ⚙️ **Configuração do Workflow**

<img width="669" height="563" alt="image" src="https://github.com/user-attachments/assets/13e84168-2bc6-420a-a007-1b27ff044831" />

O modelo foi configurado e hospedado no **Roboflow**, sendo posteriormente integrado ao **Google Colab** via API.

---

### 🔗 **Integração com o Roboflow (Colab)**

Para realizar a inferência do modelo, é necessário autenticar a API do Roboflow:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="{SUA_API_KEY}"  # Substitua pela chave da organização
)
```

### ☁️ Comparação com o Azure Computer Vision
Além do modelo desenvolvido com o Roboflow (YOLOv8), foi realizada uma integração com o serviço Azure Computer Vision, a fim de comparar os resultados entre os dois sistemas de visão computacional.

O Azure Computer Vision foi utilizado no mesmo notebook da Parte 2 (Parte_2_Visão_Computacional.ipynb) para:
- Analisar as mesmas imagens utilizadas no modelo do Roboflow;
- Comparar a precisão, tempo de resposta e qualidade das detecções;
- Avaliar vantagens e limitações entre uma solução customizada (YOLOv8) e uma solução pronta de nuvem (Azure).

Essa comparação permitiu compreender melhor como modelos pré-treinados customizáveis (YOLOv8) se comportam em relação a serviços cognitivos prontos para uso (Azure Computer Vision).

Para a criação desse recurso, criamos diretamente pelo Portal do Azure, e para seu uso no colab adicionamos o Endpoint e a Key do recurso:
```python
key = "{API_KEY}"
endpoint = "{ENDPOINT_AZ_COMPUTER_VISION}"
```

## 🎥 **Vídeo demonstrativo**

**Link Youtube:** https://youtu.be/bYFx-rRQM80?si=EMnabShqzh3tfmhM

## 👨‍💻 **Grupo Desenvolvedor**

- Gabriela de Sousa Reis - RM558830
- Laura Amadeu Soares - RM556690
- Raphael Lamaison Kim - RM557914
