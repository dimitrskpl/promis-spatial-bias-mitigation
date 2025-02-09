import pandas as pd
import xgboost as xgb
import joblib


def crime_xgb_train_predict(
    X_train,
    X_val,
    X_test,
    y_train,
    crime_train_predictions_filename,
    crime_val_predictions_filename,
    crime_test_predictions_filename,
    clf_filename,
):
    """
    Trains an XGBoost classifier on the crime dataset and saves predictions.

    Args:
        data_filename (str): Path to the crime dataset CSV file.
        crime_val_predictions_filename (str): Path to save validation predictions CSV.
        crime_test_predictions_filename (str): Path to save test predictions CSV.
        crime_val_labels_filename (str): Path to save validation labels CSV.
        crime_test_labels_filename (str): Path to save test labels CSV.
        clf_filename (str): Path to save the trained XGBoost model.

    Steps:
        - Loads and preprocesses the crime dataset.
        - Trains an XGBoost classifier on the training set.
        - Predicts labels and probabilities for test and validation sets.
        - Saves the predictions along with latitude and longitude.
        - Saves the trained model using `joblib`.
        - returns the predictions for the training, validation, and test sets.
    """

    clf = xgb.XGBClassifier(
        objective="binary:logistic",  # Logistic regression for binary classification
        eval_metric="logloss",  # Evaluation metric
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_val_pred = clf.predict(X_val)

    # find the column index of the positive labels
    pos_class_prob_idx = 1 if clf.classes_[1] == 1 else 0
    y_train_pred_proba = clf.predict_proba(X_train)[:, pos_class_prob_idx]
    y_test_pred_proba = clf.predict_proba(X_test)[:, pos_class_prob_idx]
    y_val_pred_proba = clf.predict_proba(X_val)[:, pos_class_prob_idx]

    # Predictions datasets
    train_pred_dataset = pd.DataFrame(
        {
            "pred": y_train_pred,
            "prob": y_train_pred_proba,
            "lat": X_train["LAT"].values,
            "lon": X_train["LON"].values,
        },
    )
    test_pred_dataset = pd.DataFrame(
        {
            "pred": y_test_pred,
            "prob": y_test_pred_proba,
            "lat": X_test["LAT"].values,
            "lon": X_test["LON"].values,
        },
    )

    val_pred_dataset = pd.DataFrame(
        {
            "pred": y_val_pred,
            "prob": y_val_pred_proba,
            "lat": X_val["LAT"].values,
            "lon": X_val["LON"].values,
        },
    )

    train_pred_dataset.to_csv(crime_train_predictions_filename, index=False)
    test_pred_dataset.to_csv(crime_test_predictions_filename, index=False)
    val_pred_dataset.to_csv(crime_val_predictions_filename, index=False)
    joblib.dump(clf, clf_filename)

    return y_train_pred, y_test_pred, y_val_pred
