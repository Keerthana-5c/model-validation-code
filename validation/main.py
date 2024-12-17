import validate_models
import compare
import calculate_metrics
from utils.Models import utils as modelUtils

def main(model_path, image_folder, model_type, mapping_column):
    try:
        validate_models.main(model_path, image_folder, model_type, mapping_column)
    except Exception as e:
        print(f"Error in validate_models: {e}")
        return  # Stop execution if there's an error

    try:
        compare.main(mapping_column)
    except Exception as e:
        print(f"Error in compare: {e}")
        return  # Stop execution if there's an error

    try:
        calculate_metrics.main()
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return  # Stop execution if there's an error

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python main.py <model_path> <image_folder> <model_type> <mapping_column>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_folder = sys.argv[2]
    model_type = sys.argv[3]
    mapping_column = sys.argv[4]

    main(model_path, image_folder, model_type, mapping_column)
