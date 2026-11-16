import tensorflow as tf
from tensorflow.keras.models import load_model # Para carregar o modelo salvo
from tensorflow.keras.preprocessing import image as keras_image_preprocessing # Para carregar e manipular imagens
import numpy as np
import matplotlib.pyplot as plt # Para exibir as imagens e previsões
import os # Para verificar a existência de arquivos

print("TensorFlow Version:", tf.__version__)

# --- 1. Definir o Caminho do Modelo Final e as Nossas Classes ---
# ESTE CAMINHO DEVE APONTAR para o modelo que foi salvo APÓS o ajuste fino (fine-tuning)
MODEL_PATH = os.path.join('modelos_treinados', 'animal_detector_animals10_fine_tuned_model.h5')
if not os.path.exists(MODEL_PATH) and os.path.exists('animal_detector_animals10_fine_tuned_model.h5'):
    MODEL_PATH = 'animal_detector_animals10_fine_tuned_model.h5'

# NOMES EM PORTUGUÊS: Devem ser EXATAMENTE os mesmos usados no script de treinamento,
# na mesma ordem em que as pastas são lidas (ordem alfabética).
# Confirme esta lista com base no que foi impresso no seu treinamento (`dataset_class_names`).
# Ex: ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
# E seu respectivo mapeamento para português.
raw_class_names_it_detected = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
                               'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'] # Ajuste se necessário!

portuguese_class_names_map = {
    'cane': 'cachorro', 'cavallo': 'cavalo', 'elefante': 'elefante', 'farfalla': 'borboleta', 
    'gallina': 'galinha', 'gatto': 'gato', 'mucca': 'vaca', 'pecora': 'ovelha', 
    'ragno': 'aranha', 'scoiattolo': 'esquilo'
} # Ajuste se necessário!

# A lista final de nomes em português, na ordem correta para o modelo
final_portuguese_class_names = [portuguese_class_names_map[name_it] for name_it in raw_class_names_it_detected]


# --- 2. Parâmetros de Imagem esperados pelo modelo ---
# Estes devem corresponder exatamente ao tamanho de entrada que seu modelo foi treinado.
IMG_HEIGHT_RES = 96 
IMG_WIDTH_RES = 96
IMG_CHANNELS = 3 # RGB para imagens coloridas

# --- 3. Carregar o Modelo Salvo ---
print(f"Carregando o modelo de '{MODEL_PATH}'...")
# Verifica se o arquivo do modelo existe antes de tentar carregar
if not os.path.exists(MODEL_PATH):
    print(f"Erro: O arquivo do modelo '{MODEL_PATH}' não foi encontrado.")
    print("Por favor, certifique-se de que o script de treinamento foi executado com sucesso")
    print("e salvou o modelo final neste local e com este nome.")
    exit() # Sai do script se o modelo não for encontrado

try:
    loaded_model = load_model(MODEL_PATH) # Carrega o modelo do arquivo .h5
    print("Modelo carregado com sucesso!")
    loaded_model.summary() # Opcional: Mostra um resumo do modelo carregado para verificação
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit() # Sai do script se houver erro ao carregar o modelo

# --- 4. Função para Pré-processar Imagens e Fazer Previsões ---
# Importa a função de pré-processamento específica do MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_single_image(img_path, model, class_names_pt, target_height, target_width, target_channels):
    """
    Carrega uma imagem do disco, pré-processa-a para o formato esperado pelo modelo
    (redimensiona e aplica normalização do MobileNetV2) e faz uma previsão.

    Args:
        img_path (str): Caminho completo para o arquivo da imagem.
        model (keras.Model): O modelo de IA carregado.
        class_names_pt (list): Lista de nomes das classes em português (final_portuguese_class_names).
        target_height (int): Altura esperada da imagem pelo modelo.
        target_width (int): Largura esperada da imagem pelo modelo.
        target_channels (int): Número de canais de cor esperados (3 para RGB).

    Returns:
        tuple: (Imagem para exibição, Nome da classe prevista, Confiança)
               Retorna None se houver erro no carregamento/pré-processamento da imagem.
    """
    print(f"\nProcessando imagem: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"Erro: Arquivo de imagem '{img_path}' não encontrado.")
        return None, None, None # Retorna None para indicar erro

    try:
        # Carrega a imagem, redimensionando-a para o tamanho alvo.
        img = keras_image_preprocessing.load_img(img_path, target_size=(target_height, target_width))
        # Converte a imagem carregada para um array NumPy (valores entre 0-255).
        img_array = keras_image_preprocessing.img_to_array(img)
        
        # Adiciona uma dimensão extra no início para criar um "lote" de uma única imagem.
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Aplica o pré-processamento específico do MobileNetV2.
        img_preprocessed = preprocess_input(img_array_expanded)

    except Exception as e:
        print(f"Erro ao carregar ou pré-processar a imagem: {e}")
        return None, None, None # Retorna None para indicar erro

    # Faz a previsão usando o modelo.
    predictions = model.predict(img_preprocessed)
    # Pega as probabilidades para a única imagem no lote.
    predicted_probabilities = predictions[0] 
    # Encontra o índice da classe com a maior probabilidade (a previsão do modelo).
    predicted_label_index = np.argmax(predicted_probabilities)
    # Obtém o nome legível da classe prevista em português.
    predicted_label_name = class_names_pt[predicted_label_index] 
    # Calcula a confiança da previsão (probabilidade da classe mais alta em porcentagem).
    confidence = predicted_probabilities[predicted_label_index] * 100

    return img, predicted_label_name, confidence

# --- Exemplo de Uso: Testando com Suas Próprias Imagens ---
if __name__ == "__main__":
    # <<<<<<< MUDE ESTA LISTA COM OS CAMINHOS PARA SUAS PRÓPRIAS IMAGENS! >>>>>>>
    # Coloque aqui os caminhos completos para as fotos que você quer testar.
    # Ex: 'C:/Users/SeuUsuario/Imagens/Animais/minha_vaca.jpg'
    # Ex: '/home/seu_usuario/fotos/minha_aranha.png'
    #
    # Lembre-se que o modelo foi treinado APENAS nas 10 classes de animais que você definiu.
    # Se você fornecer uma imagem de algo que NÃO é um desses 10 animais,
    # o modelo AINDA ASSIM tentará classificá-la como um dos 10 animais que conhece.
    
    image_paths = [
        os.path.join('Animais', 'gatoTeste2.webp'), 
        os.path.join('Animais', 'cachorro.webp'), 
        os.path.join('Animais', 'serHumanoTeste.jpg'),
        os.path.join('Animais', 'elefante.jpg'),
        os.path.join('Animais', 'cavalo.jpg'),
        os.path.join('Animais', 'aranha.webp'),
        os.path.join('Animais', 'borboleta.jpg'),
        os.path.join('Animais', 'elefante.webp'),
        os.path.join('Animais', 'esquilo.jpg'),  
        os.path.join('Animais', 'galinha.jpg'),   
        os.path.join('Animais', 'ovelha.jpg'),
        os.path.join('Animais', 'vaca.jpeg'),
        # Adicione mais caminhos de imagens conforme necessário
    ]

    # Configura a figura para exibir as imagens
    num_cols = 5 # Número de colunas no grid de exibição
    num_rows = (len(image_paths) + num_cols - 1) // num_cols # Calcula o número de linhas necessárias
    
    plt.figure(figsize=(num_cols * 4, num_rows * 3)) # Tamanho da figura (largura x altura)
    plot_index = 1 # Para controlar a posição dos subplots

    for img_path in image_paths:
        # Chama a função para prever a imagem
        img_to_display, predicted_name, confidence = predict_single_image(
            img_path, loaded_model, final_portuguese_class_names, IMG_HEIGHT_RES, IMG_WIDTH_RES, IMG_CHANNELS
        )
        
        # Se a imagem foi processada com sucesso
        if img_to_display is not None:
            # Cria um subplot para a imagem atual
            plt.subplot(num_rows, num_cols, plot_index)
            plt.xticks([]) # Remove os ticks do eixo X
            plt.yticks([]) # Remove os ticks do eixo Y
            plt.grid(False) # Remove a grade
            plt.imshow(img_to_display) # Exibe a imagem

            # Define o título do subplot com a previsão e a confiança
            title = f"Predito: {predicted_name} ({confidence:.2f}%)"
            plt.xlabel(title, color='blue') # O texto da previsão será azul
            plot_index += 1 # Prepara para o próximo subplot
        else:
            # Mensagem se a imagem não pôde ser processada
            print(f"Não foi possível processar a imagem: {img_path}")

    # Título geral para a figura
    plt.suptitle("Previsões do Modelo em Suas Próprias Imagens (Animals-10 - Ajuste Fino)", fontsize=16)
    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    plt.show() # Exibe a janela com as imagens e previsões

    print("\nTeste com suas próprias imagens concluído!")
    print("Lembre-se de que o modelo foi treinado para as 10 categorias de animais especificadas.")
    print("Avalie se ele está fazendo as previsões esperadas para suas imagens.")