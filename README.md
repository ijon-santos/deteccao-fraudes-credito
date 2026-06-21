# 🔍 Detecção de Fraudes em Transações Financeiras com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-green.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Visão Geral

Este projeto desenvolve um modelo de **Machine Learning** para identificar transações fraudulentas em cartões de crédito, utilizando o dataset público disponível no TensorFlow e Kaggle. O objetivo é maximizar a captura de fraudes (recall) enquanto minimiza falsos positivos (precisão), equilibrando a experiência do cliente com a segurança financeira.

### 🎯 Principais Desafios

- **Desbalanceamento extremo**: Apenas 0.17% das transações são fraudulentas
- **Custo elevado de falsos negativos**: Fraudes não detectadas geram perda financeira
- **Custo de falsos positivos**: Clientes legítimos bloqueados geram insatisfação
- **Alta dimensionalidade**: 30 features (28 anonimizadas via PCA)

---

## 📈 Resultados do Melhor Modelo

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Modelo** | Random Forest + SMOTE (0.4) | Ensemble de 100 árvores com balanceamento |
| **Recall** | **0.818** | **81.8% das fraudes capturadas** 🏆 |
| **Precision** | 0.840 | 84% dos alertas são fraudes reais |
| **F1-Score** | 0.829 | Excelente equilíbrio entre recall e precision |
| **AUC-ROC** | 0.950 | Excelente capacidade de separação |
| **Threshold Otimizado** | 0.25 | Limiar de decisão ajustado |
| **Técnica de Balanceamento** | SMOTE (sampling_strategy=0.4) | 40% de fraudes no treino |

### 📊 Impacto de Negócio

| Cenário | Fraudes Capturadas | Fraudes Perdidas | Impacto Financeiro* |
|---------|-------------------|------------------|---------------------|
| **Modelo Final** | **81.8%** | **18.2%** | **-R$ 54.600** |
| Baseline (sem balanceamento) | 75.0% | 25.0% | -R$ 75.000 |
| **Ganho** | **+6.8%** | **-6.8%** | **+R$ 20.400** |

*Considerando valor médio de R$ 3.000 por fraude e lote de 100.000 transações com 170 fraudes.

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Finalidade |
|------------|--------|------------|
| **Python** | 3.13.7 | Linguagem principal |
| **Pandas** | 2.3.2 | Manipulação e análise de dados |
| **NumPy** | 2.4.6 | Computação numérica |
| **Scikit-learn** | 1.9.0 | Pré-processamento, modelos e métricas |
| **XGBoost** | 3.3.0 | Modelo de boosting (comparativo) |
| **Random Forest** | - | Modelo final (ensemble) |
| **SMOTE** | - | Balanceamento de classes |
| **SHAP** | 0.46.0 | Explicabilidade do modelo |
| **Matplotlib/Seaborn** | - | Visualizações |

---

## 📁 Estrutura do Projeto
deteccao-fraudes-credito/
│
├── data/
│ └── raw/
│ └── creditcard.csv # Dataset original (284.807 transações)
│
├── notebooks/
│ ├── deteccao-de-anomalias.ipynb # Notebook principal do projeto
│ ├── random_forest_smote40_model.pkl # Modelo final treinado
│ ├── scaler.pkl # Scaler para padronização
│ ├── optimal_threshold.pkl # Threshold otimizado (0.25)
│ ├── thresholds_info.pkl # Dados dos thresholds testados
│ └── .ipynb_checkpoints/ # Checkpoints automáticos
│
├── src/
│ ├── visualization/ # Gráficos e visualizações
│ │ ├── feature_importance.png # Importância das features
│ │ ├── roc_pr_curves.png # Curvas ROC e Precision-Recall
│ │ ├── shap_summary.png # SHAP Summary Plot
│ │ ├── shap_bar.png # SHAP Bar Plot
│ │ ├── shap_waterfall.png # SHAP Waterfall (transação específica)
│ │ └── threshold_analysis.png # Análise do threshold
│ └── data/
│ └── feature_importance.csv # Importância das features (dados)
│
├── models/ # Modelos (backup)
├── requirements.txt # Dependências do projeto
├── .gitignore # Arquivos ignorados pelo Git
└── README.md # Documentação do projeto

---

## 🔍 Análise de Feature Importance

### Top 5 Features Mais Importantes

| Feature | Importância | Descrição |
|---------|-------------|-----------|
| **V14** | 0.087 | Feature anonimizada (PCA) |
| **V4** | 0.081 | Feature anonimizada (PCA) |
| **V10** | 0.079 | Feature anonimizada (PCA) |
| **V12** | 0.071 | Feature anonimizada (PCA) |
| **V17** | 0.064 | Feature anonimizada (PCA) |

### 🔍 Insights

- As features V14, V4 e V10 são as mais relevantes para detectar fraudes
- A variável `Amount` tem importância moderada (~0.02)
- As features V1-V28 são resultado de PCA e já estão padronizadas
- `Time` tem baixa importância, mas pode ser útil em análises temporais

---

## 🧠 Explicabilidade com SHAP

### O que é SHAP?

SHAP (SHapley Additive exPlanations) é uma técnica que explica **por que** o modelo fez uma previsão específica, atribuindo a cada feature um valor de contribuição.

### Exemplo de Análise

Para uma transação classificada como **FRAUDE**:

- **V14**: Aumentou a probabilidade de fraude em **+0.35**
- **V4**: Aumentou a probabilidade de fraude em **+0.28**
- **V10**: Aumentou a probabilidade de fraude em **+0.22**
- **Base**: 0.17% (probabilidade base de fraude)

**Conclusão**: O modelo identificou padrões específicos nas features V14, V4 e V10 que indicam alta probabilidade de fraude.

---

## 📊 Metodologia

### 1. Pré-processamento

- **Escalonamento**: StandardScaler nas features (já padronizadas, exceto `Amount`)
- **Transformação**: `Amount` recebeu transformação logarítmica e padronização
- **Split**: 70% treino / 30% teste (estratificado)

### 2. Balanceamento de Classes

Testamos várias estratégias para lidar com o desbalanceamento extremo (0.17% de fraudes):

| Estratégia | Recall | Observação |
|------------|--------|------------|
| **Sem balanceamento** | 0.750 | Viés para classe majoritária |
| **SMOTE (0.2)** | 0.811 | ✅ Bom |
| **SMOTE (0.3)** | 0.804 | ✅ Bom |
| **SMOTE (0.4)** | **0.818** | 🏆 **MELHOR** |
| **SMOTE (0.5)** | 0.811 | ✅ Bom |
| **SMOTE (0.7)** | 0.791 | ⚠️ Começa a cair |
| **SMOTE (1.0)** | 0.797 | ⚠️ Overfitting |

**Conclusão**: SMOTE com `sampling_strategy=0.4` (40% de fraudes no treino) proporcionou o melhor recall.

### 3. Modelagem

Testamos três abordagens:

| Modelo | Técnica | Recall | Precision | F1-Score |
|--------|---------|--------|-----------|----------|
| **Random Forest** | SMOTE (0.4) | **0.818** | 0.840 | **0.829** |
| XGBoost | SMOTE (0.3) | 0.804 | 0.820 | 0.812 |
| Random Forest | `class_weight='balanced'` | 0.750 | 0.870 | 0.806 |

**Conclusão**: Random Forest com SMOTE (0.4) teve o melhor recall e F1-Score.

### 4. Otimização de Hiperparâmetros (Grid Search)

Os melhores parâmetros encontrados:

```python
{
    'classifier__n_estimators': 100,
    'classifier__max_depth': 10,
    'classifier__min_samples_split': 5,
    'classifier__min_samples_leaf': 2,
    'smote__sampling_strategy': 0.4
}


5. Ajuste de Threshold
O threshold padrão de 0.5 foi otimizado para 0.25, resultando em:

Recall: 0.818 (+0.068 em relação ao baseline)

Precision: 0.840 (-0.030 em relação ao baseline)

Trade-off: Aceitamos uma pequena queda na precisão para capturar 6.8% mais fraudes.