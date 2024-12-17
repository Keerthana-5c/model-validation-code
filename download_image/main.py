from data import execute_query
from study_path import get_study_paths
from get_image import image_migration

def main():
    # Step 1: Execute SQL and save to CSV
    sql_file_path = 'query.sql'
    output_csv = "csv/query_result.csv"
    execute_query(sql_file_path, output_csv)

    # Step 2: Get study paths
    study_path_csv = 'csv/studypath.csv'
    get_study_paths(output_csv, study_path_csv)

    # Step 3: Manage image migration
    image_migration(study_path_csv)

if __name__ == "__main__":
    main()
