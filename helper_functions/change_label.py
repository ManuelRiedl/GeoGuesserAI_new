import os

import os


def merge_txt_files(dir_primary, dir_secondary):
    """
    For each .txt file in dir_primary, find a file with the same name in dir_secondary.
    If it exists, append its content to the one in dir_primary.
    """
    for filename in os.listdir(dir_primary):
        if filename.endswith(".txt"):
            primary_path = os.path.join(dir_primary, filename)
            secondary_path = os.path.join(dir_secondary, filename)

            if os.path.exists(secondary_path):
                with open(primary_path, "a") as f_primary, open(secondary_path, "r") as f_secondary:
                    content = f_secondary.read().strip()
                    if content:
                        f_primary.write("\n" + content + "\n")
                print(f"Merged: {filename}")
            else:
                print(f"Missing in secondary: {filename}")


# Example usage:
def update_class_ids_with_mapping(input_dir, original_ids, new_labels):
    """
    Update class IDs in all .txt files in input_dir based on a mapping.

    Parameters:
    - input_dir: Directory containing label files.
    - original_ids: List of original class IDs as strings, e.g., ['0', '1'].
    - new_labels: List of new class IDs as strings, e.g., ['3', '6'].
    """
    assert len(original_ids) == len(new_labels), "Mapping lists must be the same length."

    id_mapping = dict(zip(original_ids, new_labels))

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)

            with open(file_path, "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] in id_mapping:
                    parts[0] = id_mapping[parts[0]]
                updated_lines.append(" ".join(parts) + "\n")

            with open(file_path, "w") as f:
                f.writelines(updated_lines)
            print(f"Updated: {filename}")


# Example usage
update_class_ids_with_mapping("../../Geogusser_Learn_AI/data/images_unlabeled/slowenia/slowenia_bollards/labels", original_ids=['0', '1'], new_labels=['13', '14'])
#merge_txt_files("data/images_unlabeled/portugal/portugal_bollards/labels", "data/images_unlabeled/portugal/portugal_guardrails/labels")
