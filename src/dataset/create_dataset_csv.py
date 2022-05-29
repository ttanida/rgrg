import csv
import os

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized"
path_to_chest_imagenome_scene_graphs = "/u/home/tanida/datasets/chest-imagenome-dataset/silver_dataset/scene_graph"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"


def create_new_csv_file(dataset, path_csv_file):
    print(dataset)
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            subject_id = row[1]
            study_id = row[2]
            dicom_id = row[3]
            # file_path is of the form 'files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm'
            # so 'files/p../subject_id/study_id/dicom_id.dcm'
            file_path = row[4]



def create_new_csv_files(csv_files_dict):
    if os.path.exists(path_to_chest_imagenome_customized):
        print(f"Customized chest imagenome dataset already exists at {path_to_chest_imagenome_customized}.")
        print("Delete dataset folder before running script to create new folder!")
        return 1

    os.mkdir(path_to_chest_imagenome_customized)

    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file)


def get_train_val_test_csv_files():
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()
    create_new_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
