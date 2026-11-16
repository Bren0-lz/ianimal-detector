import os
validation_root = 'data_split/validation'
print("Contagem de imagens nas pastas de validação:")
for class_folder in os.listdir(validation_root):
    class_path = os.path.join(validation_root, class_folder)
    if os.path.isdir(class_path):
        num_files = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        print(f"- {class_folder}: {num_files} imagens")