import os
from PIL import Image

def convert_to_grayscale(input_folder, output_folder):
    """
    Converts all images in the input folder and its subfolders to grayscale and saves them in the output folder,
    maintaining the folder structure.

    :param input_folder: Path to the folder containing the original images.
    :param output_folder: Path to the folder where grayscale images will be saved.
    """
    for root, _, files in os.walk(input_folder):
        # Determine the relative path and corresponding output folder
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)
        
        # Ensure the current output folder exists
        os.makedirs(current_output_folder, exist_ok=True)

        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Convert to grayscale
                    grayscale_img = img.convert("L")

                    # Save the new image in the corresponding output folder
                    grayscale_img.save(os.path.join(current_output_folder, filename))

                    print(f"Converted {file_path} to grayscale.")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

# Example usage
input_folder = os.path.abspath(r'C:\data\25')  # Replace with the path to your input folder
output_folder = os.path.abspath(r'C:\Grayscale\25')  # Replace with the path to your output folder
convert_to_grayscale(input_folder, output_folder)