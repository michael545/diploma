import os
import shutil

def organize_csv_by_frequency(base_directory, frequencies):
    for frequency in frequencies:
        freq_folder = os.path.join(base_directory, frequency)
        if not os.path.exists(freq_folder):
            os.makedirs(freq_folder)

    for file_name in os.listdir(base_directory):
        if file_name.endswith('.csv'):
            for frequency in frequencies:
                if f'_{frequency}.' in file_name:
                    src_file = os.path.join(base_directory, file_name)
                    dest_file = os.path.join(base_directory, frequency, file_name)
                    shutil.move(src_file, dest_file)
                    break

    print("finished, organized by freq.")

base_dir = "data"
frequency_types = ["1H", "5M", "15M"]  

organize_csv_by_frequency(base_dir, frequency_types)