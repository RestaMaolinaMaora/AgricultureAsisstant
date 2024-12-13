import os

# Lokasi folder dataset tujuan
base_dir = "dataset"

# Membuat folder train, validation, dan test
for folder in ["train", "validation", "test"]:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

print("Folder train, validation, dan test berhasil dibuat!")
