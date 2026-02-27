import pandas as pd
import os

def import_csv_file(file_path):
    """Import data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully imported CSV: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error importing CSV: {e}")
        return None

def import_excel_file(file_path, sheet_name=None):
    """Import data from an Excel file"""
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        print(f"Successfully imported Excel: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error importing Excel: {e}")
        return None

def import_json_file(file_path):
    """Import data from a JSON file"""
    try:
        df = pd.read_json(file_path)
        print(f"Successfully imported JSON: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error importing JSON: {e}")
        return None

def import_parquet_file(file_path):
    """Import data from a Parquet file"""
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully imported Parquet: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error importing Parquet: {e}")
        return None

def import_text_file(file_path, delimiter='\t'):
    """Import data from a text file with custom delimiter"""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Successfully imported text file: {file_path}")
        print(f"Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error importing text file: {e}")
        return None

def import_multiple_files(directory_path, file_extension='.csv'):
    """Import multiple files from a directory"""
    all_data = {}
    
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(file_extension):
                file_path = os.path.join(directory_path, filename)
                
                if file_extension == '.csv':
                    df = pd.read_csv(file_path)
                elif file_extension == '.xlsx':
                    df = pd.read_excel(file_path)
                elif file_extension == '.json':
                    df = pd.read_json(file_path)
                else:
                    print(f"Unsupported file extension: {file_extension}")
                    continue
                
                all_data[filename] = df
                print(f"Imported: {filename} (Shape: {df.shape})")
        
        return all_data
    except Exception as e:
        print(f"Error importing multiple files: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example file paths (replace with your actual file paths)
    csv_file = "data.csv"
    excel_file = "data.xlsx"
    json_file = "data.json"
    
    print("=== Pandas File Import Examples ===\n")
    
    # Import CSV
    print("1. Importing CSV file:")
    csv_data = import_csv_file(csv_file)
    
    # Import Excel
    print("\n2. Importing Excel file:")
    excel_data = import_excel_file(excel_file)
    
    # Import JSON
    print("\n3. Importing JSON file:")
    json_data = import_json_file(json_file)
    
    # Import multiple files from directory
    print("\n4. Importing multiple files from directory:")
    # multiple_data = import_multiple_files("./data_directory", ".csv")
    
    # Display sample data if imports were successful
    if csv_data is not None:
        print("\n--- CSV Data Sample ---")
        print(csv_data.head())
    
    if excel_data is not None:
        print("\n--- Excel Data Sample ---")
        print(excel_data.head())
    
    if json_data is not None:
        print("\n--- JSON Data Sample ---")
        print(json_data.head())
