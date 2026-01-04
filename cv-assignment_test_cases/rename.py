import os


file_name = ["avoy", "navin", "rakin", "yousha"]
# Folder containing images
for i in range(0, 4):
    folder_path = f"./test/{file_name[i]}"   # change this

    # New base name (same label)
    label = f"{file_name[i]}"

    # Supported image extensions
    image_exts = (".jpg")

    # Get and sort image files
    images = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_exts)
    )

    for idx, filename in enumerate(images, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{label}_{idx:03d}{ext}"

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print("✅ Renaming completed.")
