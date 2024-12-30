import json

def read_and_print_json():
    try:
        # Open and read the JSON file
        with open('extracted_info/openai_extraction_20241226_202857.txt', 'r') as file:
            # Load JSON data
            data = json.load(file)
            
            # Pretty print the JSON data with indentation
            print(json.dumps(data, indent=2))
            
    except FileNotFoundError:
        print("Error: File not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    read_and_print_json()
