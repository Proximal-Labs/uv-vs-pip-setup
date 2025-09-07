import os
import pandas as pd
import csv

import os
import pandas as pd
import csv

def xlsx_to_csv(input_folder, output_folder):
    """
    Converts all .xlsx files in the input folder to .csv files in the output folder.
    
    Args:
    input_folder (str): The path to the folder containing the .xlsx files to be converted.
    output_folder (str): The path to the folder where the .csv files will be saved.
    
    Returns:
    None
    """
    # List all files in the input folder
    file_list = os.listdir(input_folder)
    
    # Filter the list to include only .xlsx files
    xlsx_files = [f for f in file_list if f.endswith('.xlsx')]
    
    for xlsx_file in xlsx_files:
        # Construct full path to the .xlsx file
        xlsx_file_path = os.path.join(input_folder, xlsx_file)
        
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(xlsx_file_path)
        
        # Construct the name of the output .csv file
        csv_file = os.path.splitext(xlsx_file)[0] + '.csv'
        csv_file_path = os.path.join(output_folder, csv_file)
        
        # Write DataFrame to .csv file, overwriting if it already exists,
        # preserving special Portuguese characters, and quoting all cells
        df.to_csv(csv_file_path, index=False, encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)

# Get the directory of the current script
current_directory = os.path.dirname(__file__)

# Define the folders for input and output
input_folder = os.path.join(current_directory, 'data', 'input')
output_folder = os.path.join(current_directory, 'data', 'output')

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Call the function
xlsx_to_csv(input_folder, output_folder)
