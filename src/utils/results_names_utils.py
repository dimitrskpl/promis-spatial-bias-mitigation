def combine_world_info(dataset_name, partioning_type_name, clf_name):
    """
    Generates formatted names for partitioning, prediction, and labels based on the given dataset,
    partitioning type, and classifier.

    Args:
        dataset_name (str): The name of the dataset.
        partioning_type_name (str): The type of partitioning applied to the dataset.
        clf_name (str): The name of the classifier used for predictions.

    Returns:
        tuple: A tuple containing:
            - A combined string with partitioning, prediction, and label names separated by '|'.
            - The generated partitioning name.
            - The generated prediction name.
            - The generated labels name.
    """

    partioning_name = f"regions_{partioning_type_name}_{dataset_name}"
    prediction_name = f"pred_{clf_name}_{dataset_name}"
    return (
        "|".join([partioning_name, prediction_name]),
        partioning_name,
        prediction_name,
    )


def get_train_val_test_paths(base_path, partioning_name, prediction_name, dataset_name):
    """
    Constructs file paths for train, validation and test datasets, including partitioning, predictions, and labels.

    Args:
        base_path (str): The base directory where the files are stored.
        partioning_name (str): The name of the partitioning file.
        prediction_name (str): The name of the prediction file.
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: A tuple containing:
            - train_path_info (dict): Paths for train data with keys:
                - "regions" (str): Path to the train partitioning file.
                - "predictions" (str): Path to the train predictions file.
                - "labels" (str): Path to the train labels file.
            - val_path_info (dict): Paths for validation data with keys:
                - "regions" (str): Path to the validation partitioning file.
                - "predictions" (str): Path to the validation predictions file.
                - "labels" (str): Path to the validation labels file.
            - test_path_info (dict): Paths for test data with keys:
                - "regions" (str): Path to the test partitioning file.
                - "predictions" (str): Path to the test predictions file.
                - "labels" (str): Path to the test labels file.
    """

    partionings_folder = "partitionings/"
    predictions_folder = "predictions/"
    labels_folder = "preprocess/"
    train_label = "train"
    val_label = "val"
    test_label = "test"

    train_regions_path = (
        f"{base_path}{partionings_folder}{train_label}_{partioning_name}.csv"
    )
    train_pred_df_path = (
        f"{base_path}{predictions_folder}{train_label}_{prediction_name}.csv"
    )
    train_labels_df_path = (
        f"{base_path}{labels_folder}y_{train_label}_{dataset_name}.csv"
    )
    val_regions_path = (
        f"{base_path}{partionings_folder}{val_label}_{partioning_name}.csv"
    )
    val_pred_df_path = (
        f"{base_path}{predictions_folder}{val_label}_{prediction_name}.csv"
    )
    val_labels_df_path = f"{base_path}{labels_folder}y_{val_label}_{dataset_name}.csv"

    test_regions_path = (
        f"{base_path}{partionings_folder}{test_label}_{partioning_name}.csv"
    )
    test_pred_df_path = (
        f"{base_path}{predictions_folder}{test_label}_{prediction_name}.csv"
    )
    test_labels_df_path = f"{base_path}{labels_folder}y_{test_label}_{dataset_name}.csv"

    train_path_info = {
        "regions": train_regions_path,
        "predictions": train_pred_df_path,
        "labels": train_labels_df_path,
    }
    val_path_info = {
        "regions": val_regions_path,
        "predictions": val_pred_df_path,
        "labels": val_labels_df_path,
    }
    test_path_info = {
        "regions": test_regions_path,
        "predictions": test_pred_df_path,
        "labels": test_labels_df_path,
    }
    return train_path_info, val_path_info, test_path_info
