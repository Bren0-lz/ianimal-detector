import os
import shutil
import random

source_dir = 'raw-img' # Sua pasta atual com as 10 subpastas
target_dir = 'data_split'    # Nova pasta onde train/validation serão criados

train_ratio = 0.8 # 80% para treino, 20% para validação

# Criar as pastas de destino
os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'validation'), exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images) # Embaralha as imagens dentro da classe

        num_train = int(len(images) * train_ratio)
        train_images = images[:num_train]
        validation_images = images[num_train:]

        # Criar subpastas para a classe dentro de train/validation
        os.makedirs(os.path.join(target_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'validation', class_name), exist_ok=True)

        # Mover imagens para as pastas de treino
        for img_name in train_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(target_dir, 'train', class_name, img_name))

        # Mover imagens para as pastas de validação
        for img_name in validation_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(target_dir, 'validation', class_name, img_name))

        print(f"Classe {class_name}: {len(train_images)} para treino, {len(validation_images)} para validação.")

print("Divisão de dataset concluída!")