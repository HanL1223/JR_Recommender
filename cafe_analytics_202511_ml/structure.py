import os

structure = {
    'data': ['raw', 'train','test'],
    'notebooks': [],
    'pipeline':['train','test'],
    'src': [],
    'steps':[],
    'reports': ['figures'],
    'config': [],
    'model':['base','tuned']
}

print("Creating folder structure...")
for folder, subfolders in structure.items():
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}/")
    for subfolder in subfolders:
        path = os.path.join(folder, subfolder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}/")

print("Folder structure created successfully!")