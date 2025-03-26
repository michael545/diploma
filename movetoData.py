import os
import shutil

source_dir = os.getcwd() 
destination_dir = os.path.join(source_dir, "data")


os.makedirs(destination_dir, exist_ok=True)


for file in os.listdir(source_dir):
    if file.endswith(".csv"):
        source = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)

        shutil.move(source, destination_path)
        print(f"Moved {file} to {destination_dir}")

print("all ok, moved")


