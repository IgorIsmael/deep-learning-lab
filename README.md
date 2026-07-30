# Deep Learning com Python: do NumPy ao TensorFlow

Este repositório apresenta minha jornada prática de aprendizado em **Deep Learning com Python**, começando pelos fundamentos matemáticos e computacionais até a construção de um projeto completo de regressão com uma rede neural.

O objetivo foi compreender não apenas como utilizar bibliotecas prontas, mas também o que acontece internamente durante o treinamento de uma rede neural.

Ao longo dos notebooks, desenvolvi conceitos como tensores, operações matriciais, funções de ativação, forward pass, função de perda, gradiente, backpropagation, normalização, validação, regularização e avaliação de modelos.

---

## Objetivos do projeto

Este projeto foi desenvolvido com os seguintes objetivos:

* aprender os fundamentos de Deep Learning;
* compreender como os dados são representados em vetores, matrizes e tensores;
* implementar manualmente partes de uma rede neural utilizando NumPy;
* entender o funcionamento do forward pass e da backpropagation;
* construir redes neurais com TensorFlow e Keras;
* preparar corretamente conjuntos de treino, validação e teste;
* evitar vazamento de dados durante a normalização;
* analisar overfitting, underfitting e generalização;
* avaliar modelos de regressão utilizando diferentes métricas;
* comparar arquiteturas e técnicas de regularização;
* salvar modelos e artefatos para utilização futura.

---

## Estrutura do repositório

```text
deep-learning-python/
│
├── 01_introducao_deep_learning.ipynb
├── 02_numpy_fundamentos.ipynb
├── 03_operacoes_numpy.ipynb
├── 04_funcoes_ativacao.ipynb
├── 05_treinamento_numpy.ipynb
├── 06_primeira_rede_tensorflow.ipynb
├── 07_projeto_01_previsao_precos.ipynb
├── app/
│   ├── app.py
│   └── artefatos_portateis.json
├── requirements.txt
└── README.md
```

---

## Conteúdo dos notebooks

### 01 — Introdução ao Deep Learning

O primeiro notebook apresenta uma introdução aos principais conceitos de Deep Learning.

São abordados temas como:

* inteligência artificial;
* Machine Learning;
* Deep Learning;
* redes neurais artificiais;
* entradas, pesos e saídas;
* aprendizado supervisionado;
* aplicações de redes neurais.

O objetivo desta etapa foi compreender onde o Deep Learning se encontra dentro da área de inteligência artificial e como uma rede neural aprende a partir dos dados.

---

### 02 — Fundamentos de NumPy

Neste notebook são apresentados os fundamentos do NumPy, biblioteca utilizada para computação numérica em Python.

Os principais conceitos estudados foram:

* criação de arrays;
* escalares;
* vetores;
* matrizes;
* tensores;
* dimensões com `ndim`;
* formato dos dados com `shape`;
* tipos de dados;
* indexação e manipulação de arrays.

Esses conceitos são essenciais porque redes neurais trabalham internamente com grandes estruturas numéricas.

---

### 03 — Operações com NumPy

O terceiro notebook aprofunda as operações matemáticas utilizadas em redes neurais.

Foram estudadas:

* soma e subtração de arrays;
* multiplicação elemento a elemento;
* multiplicação matricial;
* produto entre entradas e pesos;
* transposição de matrizes;
* broadcasting;
* cálculo da operação linear de um neurônio.

Uma das principais expressões estudadas foi:

```text
Z = XW + b
```

Onde:

* `X` representa as entradas;
* `W` representa os pesos;
* `b` representa o bias;
* `Z` representa o resultado da combinação linear.

---

### 04 — Funções de ativação

Neste notebook foram implementadas e analisadas funções de ativação utilizadas em redes neurais.

Entre elas:

* função linear;
* ReLU;
* Sigmoid;
* Tanh.

Também foi estudado o papel das funções de ativação na introdução de não linearidade na rede.

Sem funções de ativação não lineares, várias camadas densas se comportariam como uma única transformação linear, limitando a capacidade de aprendizado do modelo.

---

### 05 — Treinamento de uma rede com NumPy

Neste notebook foi implementado manualmente o processo de treinamento de uma rede neural utilizando apenas NumPy.

O fluxo desenvolvido foi:

```text
Entradas
    ↓
Forward Pass
    ↓
Previsão
    ↓
Cálculo da Loss
    ↓
Gradientes
    ↓
Backpropagation
    ↓
Atualização dos pesos
```

Foram estudados:

* inicialização de pesos;
* bias;
* Mean Squared Error;
* Gradient Descent;
* derivadas;
* backpropagation;
* learning rate;
* épocas de treinamento;
* atualização dos parâmetros.

Também foi criada uma rede com arquitetura:

```text
2 entradas
    ↓
3 neurônios
    ↓
1 saída
```

Após o treinamento, a rede conseguiu aprender a relação presente nos dados e gerar previsões próximas dos valores esperados.

---

### 06 — Primeira rede neural com TensorFlow

Depois da implementação manual com NumPy, o mesmo processo foi desenvolvido utilizando TensorFlow e Keras.

Foram abordados:

* criação de modelos com `Sequential`;
* camadas `Dense`;
* funções de ativação;
* compilação do modelo;
* otimizador Adam;
* learning rate;
* treinamento com `model.fit`;
* avaliação com `model.evaluate`;
* previsões com `model.predict`;
* separação de treino, validação e teste;
* normalização com `StandardScaler`;
* comparação entre redes de tamanhos diferentes.

Foi utilizado um dataset artificial com aproximadamente mil exemplos, seguindo uma relação matemática com adição de ruído.

Os experimentos mostraram que uma rede maior não é necessariamente melhor. Em alguns casos, um modelo menor apresentou melhor capacidade de generalização.

---

# Projeto 01 — Previsão de preços de imóveis

O último notebook apresenta um projeto completo de regressão utilizando o dataset **California Housing**.

O objetivo foi construir uma rede neural capaz de prever o valor médio de imóveis a partir de características socioeconômicas e geográficas.

---

## Dataset

O California Housing possui:

* 20.640 observações;
* 8 características de entrada;
* 1 variável alvo.

As variáveis incluem informações como:

* renda média;
* idade média das residências;
* quantidade média de cômodos;
* população;
* ocupação média;
* latitude;
* longitude;
* valor médio dos imóveis.

---

## Etapas do projeto

O projeto percorreu um fluxo completo de Machine Learning:

1. carregamento do dataset;
2. exploração inicial;
3. análise de tipos e dimensões;
4. verificação de valores ausentes;
5. verificação de dados duplicados;
6. análise da distribuição do alvo;
7. análise de correlações;
8. separação entre features e target;
9. divisão entre treino, validação e teste;
10. padronização das características;
11. construção da rede neural;
12. treinamento com Early Stopping;
13. avaliação no conjunto de teste;
14. análise de previsões;
15. análise de resíduos e maiores erros;
16. comparação entre valores reais e previstos;
17. cálculo de MSE, MAE, RMSE e R².

---

## Modelo V1

A primeira arquitetura utilizada foi:

```text
8 entradas
    ↓
64 neurônios — ReLU
    ↓
32 neurônios — ReLU
    ↓
16 neurônios — ReLU
    ↓
1 saída
```

O modelo possuía aproximadamente 3.201 parâmetros treináveis.

O treinamento utilizou:

* otimizador Adam;
* learning rate de 0,001;
* MSE como função de perda;
* MAE como métrica;
* batch size de 32;
* limite de 500 épocas;
* Early Stopping com paciência de 20 épocas.

---

## Resultados do Modelo V1

Nos resultados registrados durante o projeto, o Modelo V1 alcançou aproximadamente:

| Métrica | Resultado |
| ------- | --------: |
| MSE     |    0,2660 |
| MAE     |    0,3467 |
| RMSE    |    0,5158 |
| R²      |    0,7970 |

O treinamento foi interrompido após 93 épocas, sendo que o melhor resultado de validação ocorreu aproximadamente na época 73.

O valor de `R² = 0,7970` indica que o modelo conseguiu explicar cerca de 79,7% da variação observada nos valores do conjunto de teste.

---

## Experimentos adicionais

Após o primeiro modelo, o notebook foi ampliado com uma sequência de experimentos controlados.

Para aumentar a reprodutibilidade, foram definidas seeds para:

* Python;
* NumPy;
* TensorFlow.

Os experimentos incluíram:

### Modelo V1 controlado

Recriação da arquitetura original utilizando seeds e condições padronizadas.

### Modelo menor

Arquitetura reduzida:

```text
8 entradas
    ↓
32 neurônios
    ↓
16 neurônios
    ↓
1 saída
```

O objetivo foi verificar se uma rede mais simples conseguiria alcançar desempenho semelhante utilizando menos parâmetros.

### Regularização L2

Aplicação de penalização L2 sobre os pesos das camadas para reduzir a dependência de valores excessivamente altos.

### Dropout

Utilização de camadas `Dropout(0.2)`, desativando aleatoriamente aproximadamente 20% das ativações durante o treinamento.

### L2 e Dropout

Combinação das duas técnicas para analisar se a regularização conjunta melhora a generalização ou provoca underfitting.

---

## Comparação dos modelos

Os modelos são comparados utilizando:

* MSE;
* MAE;
* RMSE;
* R²;
* quantidade de épocas;
* quantidade de parâmetros;
* comportamento da curva de validação.

O principal critério utilizado para a seleção automática é o menor RMSE no conjunto de teste.

No entanto, a escolha final também deve considerar a simplicidade da arquitetura e a diferença real de desempenho entre os modelos.

Uma rede menor pode ser preferível quando apresenta desempenho semelhante utilizando menos parâmetros e menor custo computacional.

---

## Salvamento dos artefatos

O notebook também apresenta como salvar:

* o melhor modelo no formato `.keras`;
* o `StandardScaler` utilizado na preparação dos dados;
* a tabela de comparação dos modelos em CSV;
* os artefatos do projeto em um arquivo ZIP.

Também é realizado um teste de carregamento do modelo e do scaler para gerar uma nova previsão.

Esse processo simula uma etapa de inferência que poderia ser utilizada posteriormente em uma API ou aplicação web.

---

## Limitações do projeto

Entre as limitações identificadas estão:

* o alvo do dataset possui um teto nos valores mais altos;
* os dados representam distritos, e não imóveis individuais;
* correlação não representa causalidade;
* uma única divisão dos dados não representa toda a variabilidade possível;
* o erro percentual é instável quando os valores reais são pequenos;
* pequenas diferenças podem ocorrer entre ambientes computacionais;
* o conjunto de teste foi utilizado na comparação dos modelos com finalidade didática.

---

## Tecnologias utilizadas

* Python;
* NumPy;
* Pandas;
* Matplotlib;
* Scikit-learn;
* TensorFlow;
* Keras;
* Joblib;
* Jupyter Notebook;
* Google Colab.

---

## Como executar

Os notebooks podem ser executados no Google Colab, Jupyter Notebook ou VS Code.

### Google Colab

1. Faça o download do notebook desejado.
2. Acesse o Google Colab.
3. Selecione **Arquivo > Fazer upload de notebook**.
4. Execute as células na ordem apresentada.

### Ambiente local

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta:

```bash
cd NOME_DO_REPOSITORIO
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow joblib jupyter
```

Inicie o Jupyter:

```bash
jupyter notebook
```

## Aplicação web com Streamlit

A interface em `app/app.py` usa os oito campos na mesma ordem do Notebook 07:
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`,
`Latitude` e `Longitude`. Antes da inferência, ela aplica o
`StandardScaler` treinado e converte a saída do modelo, expressa em centenas
de milhares de dólares, para uma estimativa em dólares.

### Preparar os artefatos

A pasta `app/` já inclui `artefatos_portateis.json`, um fallback textual para
que o deploy abra e gere estimativas mesmo antes de executar o notebook. O
fallback utiliza os parâmetros de padronização do California Housing e uma
regressão linear educacional; ele não representa a rede neural treinada no
Notebook 07. Por ser JSON, pode ser revisado e incluído em pull requests sem
limitações de arquivos binários.

Para publicar a rede neural do projeto, execute o Notebook 07 até a seção
**Salvando o modelo e o scaler**. Depois, substitua os dois artefatos da pasta
da aplicação pelos arquivos gerados:

```bash
cp artefatos_projeto_01/melhor_modelo_california_housing.keras app/
cp artefatos_projeto_01/standard_scaler_california_housing.pkl app/
```

Os dois nomes precisam ser mantidos e os binários devem ser adicionados
diretamente ao repositório fora deste PR (ou armazenados com Git LFS). Quando
ambos estão presentes, a aplicação prioriza o modelo Keras e o `StandardScaler`
reais; quando estão ausentes, utiliza o JSON portátil. Nos dois casos, a ordem
das features é validada ao iniciar.

### Executar localmente

O projeto solicita Python 3.12 por meio de `.python-version`. A aplicação usa
o backend NumPy do Keras somente para inferência, portanto não depende da
disponibilidade de um pacote TensorFlow compatível com a versão de Python do
servidor. Na raiz do repositório, execute:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/app.py
```

No Windows PowerShell, ative o ambiente com
`.venv\Scripts\Activate.ps1`. O Streamlit abrirá a aplicação em
`http://localhost:8501`.

### Publicar no Streamlit Community Cloud

1. Envie este repositório, incluindo os dois artefatos, para o GitHub.
2. Entre em [share.streamlit.io](https://share.streamlit.io/) com sua conta do GitHub.
3. Selecione **Create app** e escolha o repositório e a branch desejados.
4. Em **Main file path**, informe `app/app.py`.
5. Em **Advanced settings**, selecione Python 3.12 e clique em **Deploy**. Se a
   aplicação existente foi criada com outra versão, exclua-a e crie o deploy
   novamente para que a mudança de runtime seja aplicada.

O `requirements.txt` na raiz será detectado automaticamente e não instala o
runtime completo do TensorFlow: modelos sequenciais com camadas densas, como o
deste projeto, podem ser carregados para inferência pelo backend NumPy do Keras.
O JAX está declarado explicitamente porque o Keras utiliza algumas de suas
operações internas mesmo nesse backend. O Keras só é importado quando os dois
artefatos binários reais estão presentes; o fallback JSON não depende dele.
Os artefatos não
contêm segredos e devem fazer parte do commit usado no deploy. Caso um deles
esteja ausente ou incompatível, a interface exibirá uma mensagem de erro em vez
de tentar fazer uma previsão incorreta.

---

## Principais aprendizados

Este projeto mostrou que construir uma rede neural não significa apenas adicionar mais camadas e neurônios.

Um fluxo correto de Deep Learning envolve:

```text
Entendimento dos dados
        ↓
Preparação
        ↓
Modelo de referência
        ↓
Treinamento
        ↓
Avaliação
        ↓
Análise de erros
        ↓
Experimentos controlados
        ↓
Comparação
        ↓
Seleção do modelo
        ↓
Salvamento
        ↓
Inferência
```

Também foi possível compreender que:

* modelos maiores não são automaticamente melhores;
* normalização deve ser ajustada somente nos dados de treinamento;
* treino, validação e teste possuem funções diferentes;
* Early Stopping ajuda a evitar treinamento desnecessário;
* regularização pode melhorar a generalização;
* regularização excessiva pode provocar underfitting;
* métricas devem ser analisadas em conjunto;
* a simplicidade do modelo também deve ser considerada.

---

## Próximos passos

Os próximos estudos desta jornada serão:

* validação cruzada;
* ajuste de hiperparâmetros;
* Batch Normalization;
* comparação entre otimizadores;
* estratégias de learning rate;
* primeiro projeto de classificação;
* matriz de confusão;
* precisão, recall e F1-score;
* redes neurais convolucionais;
* visão computacional;
* publicação de modelos em aplicações.

---

## Autor

**Igor Ismael de Souza Silva**

Estudante e desenvolvedor com interesse em tecnologia, análise de dados, inteligência artificial e desenvolvimento de sistemas.

Este repositório faz parte da minha jornada prática de aprendizado e construção de portfólio em Deep Learning.
