# 🐾 IAnimal: Classificação e Reconhecimento de Imagens com Deep Learning

Projeto desenvolvido para identificação e classificação automatizada de animais utilizando técnicas de **Visão Computacional** e **Deep Learning**. O projeto explora **Transfer Learning** e **Fine-Tuning** a partir da rede neural convolucional **MobileNetV2**, voltado a aplicações como monitoramento de fauna e ciências ambientais.

---

## 📌 Visão Geral do Projeto

O modelo foi treinado e avaliado utilizando o dataset **Animals-10** (com 10 classes distintas de fauna), adaptado para tradução e classificação em Português:

| Classe em Italiano (Original) | Tradução em Português |
| :--- | :--- |
| `cane` | Cachorro |
| `cavallo` | Cavalo |
| `elefante` | Elefante |
| `farfalla` | Borboleta |
| `gallina` | Galinha |
| `gatto` | Gato |
| `mucca` | Vaca |
| `pecora` | Ovelha |
| `ragno` | Aranha |
| `scoiattolo` | Esquilo |

---

## 🧠 Arquitetura do Modelo

1. **Backbone**: `MobileNetV2` pré-treinado no dataset ImageNet (pesos congelados na etapa inicial).
2. **Data Augmentation**: Camadas de aumento de dados em tempo de execução (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`).
3. **Classificador Customizado**:
   - `Conv2D` e `MaxPooling2D`
   - Camadas `Flatten` e `Dense` com ativação ReLU
   - `Dropout` para regularização e prevenção de overfitting
   - Camada final `Dense(10, activation='softmax')`
4. **Treinamento & Fine-Tuning**:
   - Fase 1: Treinamento do classificador com a base congelada.
   - Fase 2: Ajuste fino (*Fine-Tuning*) descongelando as camadas superiores da MobileNetV2 com taxa de aprendizado reduzida (`lr=1e-5`) e `EarlyStopping`.

---

## 📁 Estrutura de Diretórios

```
IAnimal/
├── Animais/                        # Imagens de exemplo para testes rápidos de inferência
│   ├── aranha.webp
│   ├── borboleta.jpg
│   ├── cachorro.webp
│   ├── cavalo.jpg
│   ├── elefante.jpg
│   ├── esquilo.jpg
│   ├── galinha.jpg
│   ├── gatoTeste2.webp
│   ├── ovelha.jpg
│   └── vaca.jpeg
├── modelos_treinados/              # Pesos do modelo treinado (.h5)
│   └── animal_detector_animals10_fine_tuned_model.h5
├── identificando_imagem.py         # Script para inferência e visualização de novas imagens
├── treinamento.py                  # Pipeline completo de treino e fine-tuning
├── divisao_pastas.py               # Utilitário para separação treino (80%) e validação (20%)
├── contagem.py                     # Utilitário para verificar volume de imagens por classe
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação do projeto
```

---

## 🚀 Como Executar

### 1. Pré-requisitos e Instalação

Clone o repositório e configure um ambiente virtual:

```bash
git clone https://github.com/Bren0-lz/ianimal-detector.git
cd ianimal-detector

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Testar o Modelo com Imagens Existentes

Para rodar a inferência e exibir o painel de previsões com as fotos de teste da pasta `Animais/`:

```bash
python identificando_imagem.py
```

O script carregará o modelo treinado `animal_detector_animals10_fine_tuned_model.h5` e gerará um grid visual exibindo a imagem, classe prevista e o percentual de confiança.

### 3. Re-treinar o Modelo (Opcional)

Caso queira treinar o modelo a partir do dataset bruto:

1. Baixe o dataset [Animals-10 no Kaggle](https://www.kaggle.com/datasets/alessiasimon/animals-10) e extraia na pasta `raw-img`.
2. Execute a divisão em treino/validação:
   ```bash
   python divisao_pastas.py
   ```
3. Inicie o treinamento:
   ```bash
   python treinamento.py
   ```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras** (Deep Learning & Redes Neurais Convolucionais)
- **NumPy** (Processamento matricial e vetorial)
- **Matplotlib** (Visualização de métricas e previsões)
- **Pillow** (Manipulação de imagens)
