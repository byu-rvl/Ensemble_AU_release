import os
import gzip
import shutil
import time
from pathlib import Path
import tarfile

def decompress_gz_files(source_dir, dest_dir, extensions):
    """
    Decompress all .gz files in the source directory and save the decompressed files to the destination directory.
    
    :param source_dir: Directory to search for .gz files
    :param dest_dir: Directory to save the decompressed files
    :param extensions: Tuple of file extensions to process (e.g., '.tar.gz', '.pt.gz')
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    start_time = time.time()

    print("Decompressing .gz files...")
    print("Source directory:", source_dir)
    print("Destination directory:", dest_dir)

    # Loop through all files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extensions):
                compressed_path = os.path.join(root, file)
                decompressed_filename = file.rsplit('.gz', 1)[0]  # Remove the .gz extension
                decompressed_path = os.path.join(dest_dir, decompressed_filename)

                if Path(decompressed_path).exists():
                    print(f"Skipping: {compressed_path} -> {decompressed_path}")
                    continue
                
                # Decompress the .gz file
                with gzip.open(compressed_path, 'rb') as compressed_file:
                    with open(decompressed_path, 'wb') as decompressed_file:
                        shutil.copyfileobj(compressed_file, decompressed_file)
                
                print(f"Decompressed: {compressed_path} -> {decompressed_path}")
                print("Time elapsed: ", time.time() - start_time)
                start_time = time.time()

def expand_tar_files(source_dir, dest_dir):
    """
    Expand all .tar files in the source directory and extract their contents to the destination directory.

    :param source_dir: Directory to search for .tar files
    :param dest_dir: Directory to save the extracted contents
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    start_time = time.time()
    print("Expanding .tar files...")
    print("Source directory:", source_dir)
    print("Destination directory:", dest_dir)

    # Loop through all files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tar'):
                tar_path = os.path.join(root, file)
                extract_dir = os.path.join(dest_dir, Path(file).stem)  # Extract contents to a folder named after the tar file
                
                if Path(extract_dir).exists():
                    print(f"Skipping: {tar_path} -> {extract_dir}")
                    continue

                # Extract the tar file
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=extract_dir)
                
                print(f"Extracted: {tar_path} -> {extract_dir}")
                print("Time elapsed: ", time.time() - start_time)
                start_time = time.time()


file_extensions = ('.tar.gz', '.pt.gz', '.pth.gz')  # Specify the file extensions to decompress

# source_directory = "data/weights/stacking_head/BP4D_compressed/"
# destination_directory = "data/weights/stacking_head/BP4D"
# decompress_gz_files(source_directory, destination_directory, extensions=file_extensions)

# source_directory = "data/weights/stacking_head/DISFA_compressed/"
# destination_directory = "data/weights/stacking_head/DISFA"
# decompress_gz_files(source_directory, destination_directory, extensions=file_extensions)

# source_directory = "data/predictions/BP4D_compressed"
# destination_directory = "data/predictions/BP4D"
# decompress_gz_files(source_directory, destination_directory, extensions=file_extensions)

# source_directory = "data/predictions/DISFA_compressed"
# destination_directory = "data/predictions/DISFA"
# decompress_gz_files(source_directory, destination_directory, extensions=file_extensions)

# source_directory = "data/datasets/SynAU_compressed"
# destination_directory = "data/datasets/SynAU_tar"
# decompress_gz_files(source_directory, destination_directory, extensions=file_extensions)

source_directory = "data/datasets/SynAU_tar"
destination_directory = "data/datasets/SynAU_expanded"
expand_tar_files(source_directory, destination_directory)

print("Finished.")