from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = "christophrus/rummikub"
local_folder = "./yolo_dataset" # Dein Ordner-Pfad

print(f"Starte Upload zu {repo_id}...")

# Wir prüfen manuell, ob die Library das Feature unterstützt
import inspect
sig = inspect.signature(api.upload_folder)

upload_kwargs = {
    "folder_path": local_folder,
    "repo_id": repo_id,
    "repo_type": "dataset",
}

# Nur hinzufügen, wenn die installierte Version es wirklich kennt
if 'multi_commits' in sig.parameters:
    upload_kwargs["multi_commits"] = True
    upload_kwargs["multi_commits_verbose"] = True
elif 'num_threads' in sig.parameters:
    upload_kwargs["num_threads"] = 16

try:
    api.upload_folder(**upload_kwargs)
    print("\n--- Upload erfolgreich! ---")
except Exception as e:
    print(f"\nFehler: {e}")
    print("Versuche Einzel-Datei-Upload (Salami-Taktik)...")
    # Backup: Falls upload_folder komplett streikt
    for root, _, files in os.walk(local_folder):
        for file in files:
            path_local = os.path.join(root, file)
            path_in_repo = os.path.relpath(path_local, local_folder)
            api.upload_file(path_or_fileobj=path_local, path_in_repo=path_in_repo, repo_id=repo_id, repo_type="dataset")