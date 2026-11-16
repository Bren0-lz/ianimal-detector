import tensorflow as tf
# Importações de modelos e camadas
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import Input

# Importações de aplicações e pré-processamento (MOVIDAS PARA O TOPO)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # <<<< MOVIDO PARA CÁ

# Importações de otimizadores, utilitários de dados e callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image as keras_image_preprocessing # Para carregar imagens manualmente
from tensorflow.keras.callbacks import EarlyStopping

# Importações gerais
import matplotlib.pyplot as plt
import numpy as np
import os

print("TensorFlow Version:", tf.__version__)

# --- Mapeamento de Nomes de Classes (Italiano para Português) ---
raw_class_names_it = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
                      'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'] 

portuguese_class_names = ['cachorro', 'cavalo', 'elefante', 'borboleta', 'galinha',
                          'gato', 'vaca', 'ovelha', 'aranha', 'esquilo'] 

class_name_map = dict(zip(raw_class_names_it, portuguese_class_names))

# O número de classes será detectado dinamicamente das pastas.

# --- Configurações de Treinamento ---
IMG_HEIGHT_RES = 96 
IMG_WIDTH_RES = 96
IMG_CHANNELS = 3 
BATCH_SIZE = 32 
EPOCHS = 20 
FINE_TUNE_EPOCHS = 30 
TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS 


# --- Configurações para EarlyStopping ---
early_stopping_callback = EarlyStopping(monitor='val_accuracy', 
                                        patience=7, 
                                        restore_best_weights=True,
                                        min_delta=0.001, 
                                        verbose=1)


# --- Carregando o Dataset Animals-10 de Pastas Separadas de Treino/Validação ---
SPLIT_DATASET_DIR = 'data_split' # <<<<<<< AJUSTE ESTE CAMINHO!

print(f"\nCarregando dataset de imagens dos diretórios separados: {SPLIT_DATASET_DIR}")

# Carregamento do dataset de treino (tf.data.Dataset)
train_ds_initial = image_dataset_from_directory(
    directory=os.path.join(SPLIT_DATASET_DIR, 'train'), 
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT_RES, IMG_WIDTH_RES),
    interpolation='bicubic',
    batch_size=BATCH_SIZE,
    shuffle=True, 
    seed=None 
)

# --- CORREÇÃO FINAL PARA NOMES DE CLASSES E NÚMERO DE CLASSES (DINÂMICO) ---
dataset_class_names = train_ds_initial.class_names 
print(f"Classes carregadas pelo Keras (nomes das pastas): {dataset_class_names}")

portuguese_class_names_map_dynamic = {}
known_mappings = {
    'cane': 'cachorro', 'cavallo': 'cavalo', 'elefante': 'elefante', 'farfalla': 'borboleta', 
    'gallina': 'galinha', 'gatto': 'gato', 'mucca': 'vaca', 'pecora': 'ovelha', 
    'ragno': 'aranha', 'scoiattolo': 'esquilo'
}

for class_name_it in dataset_class_names:
    portuguese_class_names_map_dynamic[class_name_it] = known_mappings.get(class_name_it, class_name_it)

final_portuguese_class_names = [portuguese_class_names_map_dynamic[name_it] for name_it in dataset_class_names]
NUM_CLASSES_ANIMALS_10 = len(dataset_class_names) 
print(f"Número de classes detectadas: {NUM_CLASSES_ANIMALS_10}")
print(f"Nomes de classes finais em português: {final_portuguese_class_names}")


# --- FUNÇÃO DE PRÉ-PROCESSAMENTO (Definida AGORA, após imports) ---
def preprocess_for_mobilenet(image, label=None): # label é opcional para carregamento manual
    # preprocess_input espera valores de pixel entre 0-255 e os converte para [-1, 1].
    image = preprocess_input(image) 
    return (image, label) if label is not None else image

# Aplicar o pré-processamento ao dataset de treino (tf.data.Dataset)
train_dataset = train_ds_initial.map(preprocess_for_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# --- Carregar o Dataset de Validação para Arrays NumPy ---
print("\nCarregando dataset de validação para arrays NumPy (para garantir diversidade)...")
validation_images = []
validation_labels = []

validation_root_path = os.path.join(SPLIT_DATASET_DIR, 'validation')
if not os.path.exists(validation_root_path):
    print(f"ERRO CRÍTICO: Pasta de validação não encontrada: {validation_root_path}")
    print("Por favor, verifique se 'data_split/validation' realmente existe e contém as subpastas de classes.")
    exit()

# Itera sobre cada pasta de classe dentro do diretório de validação
for class_idx, class_name_it in enumerate(dataset_class_names):
    class_folder_path = os.path.join(validation_root_path, class_name_it)
    if not os.path.exists(class_folder_path):
        print(f"AVISO: Pasta de validação para classe '{class_name_it}' não encontrada: {class_folder_path}")
        continue

    for img_name in os.listdir(class_folder_path):
        img_path = os.path.join(class_folder_path, img_name)
        # Verifica se é um arquivo (e não uma subpasta oculta, etc.)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Carrega a imagem e redimensiona
                img = keras_image_preprocessing.load_img(img_path, target_size=(IMG_HEIGHT_RES, IMG_WIDTH_RES))
                img_array = keras_image_preprocessing.img_to_array(img) # Valores 0-255 float32
                
                # Aplica o pré-processamento do MobileNetV2 AQUI
                img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))[0] # [0] para remover a dimensão do lote
                
                validation_images.append(img_preprocessed)
                # Cria o rótulo one-hot manualmente
                label_one_hot = np.zeros(NUM_CLASSES_ANIMALS_10)
                label_one_hot[class_idx] = 1
                validation_labels.append(label_one_hot)
            except Exception as e:
                print(f"Erro ao carregar ou processar imagem {img_path}: {e}")

# Converte listas para arrays NumPy
validation_images_np = np.array(validation_images)
validation_labels_np = np.array(validation_labels)

# Embaralha os dados NumPy após carregá-los
shuffle_indices = np.arange(len(validation_images_np))
np.random.shuffle(shuffle_indices)
validation_images_np = validation_images_np[shuffle_indices]
validation_labels_np = validation_labels_np[shuffle_indices]


print(f"Dataset de validação carregado para NumPy: {len(validation_images_np)} imagens.")
if validation_images_np.size > 0: # Verifica se o array não está vazio antes de tentar np.unique
    unique_classes_in_numpy = np.unique(np.argmax(validation_labels_np, axis=1))
    if len(unique_classes_in_numpy) == NUM_CLASSES_ANIMALS_10:
        print("VERIFICAÇÃO BEM-SUCEDIDA: O dataset de validação NumPy contém TODAS as classes.")
    else:
        print(f"ALERTA: O dataset de validação NumPy NÃO contém todas as classes ({len(unique_classes_in_numpy)} de {NUM_CLASSES_ANIMALS_10}). Verifique suas pastas de validação.")
        missing_classes_it = [dataset_class_names[i] for i in range(NUM_CLASSES_ANIMALS_10) if i not in unique_classes_in_numpy]
        missing_classes_pt = [portuguese_class_names_map_dynamic.get(name_it, "Desconhecido") for name_it in missing_classes_it]
        print(f"Classes ausentes: {missing_classes_pt}")

else:
    print("ERRO CRÍTICO: O dataset de validação NumPy está vazio. Nenhuma imagem foi carregada.")

print("--- Fim do Carregamento e Diagnóstico do Dataset de Validação ---")


# --- Construção do Modelo (Transfer Learning) ---
print("\nCarregando o modelo pré-treinado MobileNetV2 (pesos do ImageNet)...")
base_model = MobileNetV2(input_shape=(IMG_HEIGHT_RES, IMG_WIDTH_RES, IMG_CHANNELS),
                         include_top=False, 
                         weights='imagenet')

base_model.trainable = False 
print("Camadas do MobileNetV2 congeladas (fase inicial).")

model = Sequential([
    Input(shape=(IMG_HEIGHT_RES, IMG_WIDTH_RES, IMG_CHANNELS)),
    RandomFlip("horizontal"), 
    RandomRotation(0.1),     
    RandomZoom(0.1),         
    RandomContrast(0.2), 
    base_model, 
    Flatten(),  
    Dense(128, activation='relu'), 
    Dropout(0.5), 
    Dense(NUM_CLASSES_ANIMALS_10, activation='softmax') 
])

print("\nResumo da arquitetura do modelo antes do ajuste fino:")
model.summary()


# --- FASE 1: Treinamento Inicial (apenas as camadas densas) ---
print("\n--- FASE 1: Treinamento Inicial das Camadas Densas ---")
print("Compilando o modelo para treinamento inicial...")
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treina apenas as novas camadas densas.
history_initial = model.fit(train_dataset, 
                            epochs=EPOCHS, 
                            validation_data=(validation_images_np, validation_labels_np), # <<<<< USANDO ARRAYS NUMPY!
                            callbacks=[early_stopping_callback], 
                            verbose=1)

print("\nTreinamento Inicial Concluído!")


# --- FASE 2: Ajuste Fino (Fine-tuning) ---
print("\n--- FASE 2: Iniciando o Ajuste Fino (Fine-tuning) do MobileNetV2 ---")

fine_tune_at = len(base_model.layers) - 30 
print(f"Total de camadas no MobileNetV2: {len(base_model.layers)}")
print(f"Descongelando as últimas {len(base_model.layers) - fine_tune_at} camadas do MobileNetV2 para ajuste fino...")

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False 
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True  


print("Recompilando o modelo com uma taxa de aprendizado muito baixa para ajuste fino...")
model.compile(optimizer=Adam(learning_rate=0.000005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("\nResumo da arquitetura do modelo após descongelar para ajuste fino:")
model.summary() 

print(f"\nContinuando o treinamento por mais {FINE_TUNE_EPOCHS} épocas (ajuste fino)...")
history_fine_tune = model.fit(train_dataset, 
                              epochs=TOTAL_EPOCHS, 
                              initial_epoch=history_initial.epoch[-1] + 1 if history_initial.epoch else EPOCHS, 
                              validation_data=(validation_images_np, validation_labels_np), # <<<<< USANDO ARRAYS NUMPY!
                              callbacks=[early_stopping_callback], 
                              verbose=1)

print("\nAjuste Fino Concluído!")


# --- Combinar Históricos para Visualização ---
acc = history_initial.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy']

loss = history_initial.history['loss'] + history_initial.history['loss']
val_loss = history_initial.history['val_loss'] + history_fine_tune.history['val_loss'] # Corrigido aqui


# --- Visualizar o Histórico de Treinamento Combinado ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Acurácia de Treino')
plt.plot(val_acc, label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Acurácia do Modelo Durante o Treinamento (Total)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Perda de Treino')
plt.plot(val_loss, label='Perda de Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda do Modelo Durante o Treinamento (Total)')
plt.legend()

plt.tight_layout()
plt.show()


# --- Salvando o Modelo Final (após ajuste fino) ---
model_filename_fine_tuned = 'animal_detector_animals10_fine_tuned_model.h5' 

print(f"\nSalvando o modelo FINAL (após ajuste fino) como '{model_filename_fine_tuned}'...")
try:
    model.save(model_filename_fine_tuned)
    print("Modelo salvo com sucesso!")
except Exception as e:
    print(f"Erro ao salvar o modelo: {e}")


# --- Avaliação Final do Modelo ---
print("\nAvaliando o modelo FINAL (após ajuste fino) com os dados de validação (Animals-10)...")
loss, accuracy = model.evaluate(validation_images_np, validation_labels_np, verbose=2) # <<<<< USANDO ARRAYS NUMPY!

print(f"Acurácia final no conjunto de validação (Animals-10, ajuste fino): {accuracy*100:.2f}%")
print(f"Perda final no conjunto de validação (Animals-10, ajuste fino): {loss:.4f}")


# --- Fazendo Previsões Visuais em Amostras do Dataset de Validação ---
print("\nFazendo previsões em algumas imagens do dataset de validação (Animals-10, ajuste fino)...")

# NOVO: Definir num_predictions_to_show, num_cols e num_rows AQUI, no TOPO DO BLOCO.
# Isso garante que estejam sempre definidos antes de serem usados para a figura.
num_predictions_to_show = min(25, len(validation_images_np)) # Plota até 25, ou o que tiver
num_cols = 5 # Exibir 5 imagens por linha
num_rows = (num_predictions_to_show + num_cols - 1) // num_cols # Calcula o número de linhas

# Agora a figura pode ser criada, pois num_cols e num_rows estão definidas.
plt.figure(figsize=(num_cols * 4, num_rows * 3)) 

# O restante da lógica de amostragem e plotagem permanece igual
# Embaralha os índices do array NumPy para pegar amostras aleatórias para visualização
display_indices = np.random.choice(len(validation_images_np), num_predictions_to_show, replace=False)


for i, idx in enumerate(display_indices):
    image_to_plot = validation_images_np[idx] # Pega do array NumPy
    true_label_one_hot = validation_labels_np[idx] # Pega do array NumPy
    true_label_index = np.argmax(true_label_one_hot)
    
    image_for_display = (image_to_plot + 1) / 2.0 
    
    true_label_name_it = dataset_class_names[true_label_index] 
    true_label_name_pt = portuguese_class_names_map_dynamic.get(true_label_name_it, "Desconhecido") 

    image_for_prediction_final = np.expand_dims(image_to_plot, axis=0) 
    
    predictions = model.predict(image_for_prediction_final, verbose=0)
    predicted_probabilities = predictions[0] 
    predicted_label_index = np.argmax(predicted_probabilities)
    
    predicted_label_name_it = dataset_class_names[predicted_label_index]
    predicted_label_name_pt = portuguese_class_names_map_dynamic.get(predicted_label_name_it, "Desconhecido")
    
    confidence = predicted_probabilities[predicted_label_index] * 100 

    plt.subplot(num_rows, num_cols, i + 1)
    plt.xticks([]) 
    plt.yticks([]) 
    plt.grid(False)
    plt.imshow(image_for_display) 

    color = 'green' if predicted_label_index == true_label_index else 'red'
    title = f"Real: {true_label_name_pt}\nPredito: {predicted_label_name_pt} ({confidence:.2f}%)"
    plt.xlabel(title, color=color)

plt.suptitle("Previsões do Modelo em Imagens de Validação (Animals-10 - Ajuste Fino)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\nProcesso de treinamento e avaliação com Animals-10 (Ajuste Fino) concluído!")