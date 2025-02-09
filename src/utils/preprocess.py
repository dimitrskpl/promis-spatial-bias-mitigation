import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def proprocess_crime(dataset_path, preprocess_dir):
    """
    Preprocesses a crime dataset for machine learning tasks.

    This function loads a crime dataset, cleans and transforms the data,
    applies one-hot encoding to categorical features, and splits the data
    into training, validation, and test sets. The processed datasets are
    saved to CSV files in the specified directory.

    Args:
        dataset_path (str): Path to the raw crime dataset CSV file.
        preprocess_dir (str): Directory where the preprocessed datasets will be saved.

    Returns:
        tuple: A tuple containing six pandas DataFrames:
            - X_train (pd.DataFrame): Training feature set.
            - X_val (pd.DataFrame): Validation feature set.
            - X_test (pd.DataFrame): Test feature set.
            - y_train (pd.Series): Training labels.
            - y_val (pd.Series): Validation labels.
            - y_test (pd.Series): Test labels.

    Process:
        - Loads the dataset from `dataset_path`.
        - Filters out invalid location entries (LAT and LON).
        - Renames columns for consistency.
        - Labels serious crimes based on the top 10 most frequent crime codes.
        - Selects relevant features for training.
        - Handles missing values.
        - Converts categorical features to one-hot encoding.
        - Splits the dataset into train (60%), validation (20%), and test (20%) sets.
        - Saves the processed datasets as CSV files.

    """

    crime_df = pd.read_csv(dataset_path)
    crime_df = crime_df[(crime_df["LAT"] != 0) | (crime_df["LON"] != 0)]
    if "AREA " in crime_df.columns:
        crime_df.rename(columns={"AREA ": "AREA"}, inplace=True)
    serious_crime_codes = (
        crime_df["Crm Cd 1"].value_counts().sort_index().cumsum().index[:10].values
    )
    crime_df["label"] = crime_df["Crm Cd 1"].isin(serious_crime_codes).astype(int)

    columns = [
        "TIME OCC",
        "AREA",
        "Vict Age",
        "Vict Sex",
        "Vict Descent",
        "Premis Cd",
        "Weapon Used Cd",
        "LOCATION",
        "LAT",
        "LON",
        "label",
    ]

    crimes = crime_df[columns]
    crimes = crimes.dropna()

    crimes["Premis Cd"] = crimes["Premis Cd"].astype(int).astype(str)
    crimes["Weapon Used Cd"] = crimes["Weapon Used Cd"].astype(int).astype(str)

    crimes_areas = pd.get_dummies(crimes["AREA"], prefix="Area")
    crimes_sex = pd.get_dummies(crimes["Vict Sex"], prefix="Sex")
    crimes_descent = pd.get_dummies(crimes["Vict Descent"], prefix="Descent")

    crimes_onehot = pd.concat(
        [
            crimes[["label", "TIME OCC", "LAT", "LON"]],
            crimes_areas,
            crimes_sex,
            crimes_descent,
        ],
        axis=1,
    )

    X = crimes_onehot.iloc[:, 1:]
    y = crimes_onehot.iloc[:, 0]

    indices = X.index

    X_train, X_temp, y_train, y_temp, _, temp_indices = train_test_split(
        X, y, indices, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test, _, _ = train_test_split(
        X_temp,
        y_temp,
        temp_indices,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train size: {len(X_train)} ({len(X_train) / len(X):.2%})")
    print(f"Validation size: {len(X_val)} ({len(X_val) / len(X):.2%})")
    print(f"Test size: {len(X_test)} ({len(X_test) / len(X):.2%})")

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train.to_csv(f"{preprocess_dir}/X_train_crime.csv", index=False)
    X_val.to_csv(f"{preprocess_dir}/X_val_crime.csv", index=False)
    X_test.to_csv(f"{preprocess_dir}/X_test_crime.csv", index=False)
    y_train.to_csv(f"{preprocess_dir}/y_train_crime.csv", index=False)
    y_val.to_csv(f"{preprocess_dir}/y_val_crime.csv", index=False)
    y_test.to_csv(f"{preprocess_dir}/y_test_crime.csv", index=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_lar(lar_data_filename, census_gazetteer_data_filename, preprocess_dir):
    """
    The preprocess_lar function processes Loan/Application Register (LAR) data from the
    Consumer Financial Protection Bureau (CFPB) and merges it with census tract location data
    from the U.S. Census Bureau. The final dataset includes action labels and
    corresponding geographic coordinates for further analysis.

    Data Requirements: Download and store the required datasets in the specified directory.

    a. Modified LAR Data:

        Download from: CFPB Modified LAR Data 2021 <https://ffiec.cfpb.gov/data-publication/modified-lar/2021>

            Select the year 2021.
            Enter B4TYDEB6GKMZO031MB27 as the Legal Entity Identifier (LEI) for Bank of America.
            Opt to "Include File Header."
            Click on "Download Modified LAR with Header."
            File Name: Ensure the file is saved as B4TYDEB6GKMZO031MB27_header.csv in the specified datasets_path directory.

    b. Census Gazetteer Data:

        Download from: Census Gazetteer Files 2021 <https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2021.html>

        Locate the "Census Tracts" section.
        Click on "Download the National Census Tracts Gazetteer Files."
        File Name: Ensure the file is saved as 2021_Gaz_tracts_national.txt in the specified datasets_path directory.

    Parameters
    datasets_path (str): The directory path where the required CSV and TXT files are stored.

    Load Modified LAR Data:
    Reads the Bank of America LAR data file.
    Selects relevant columns: action_taken and census_tract.
    Removes rows with missing values.
    Filters rows where action_taken is either 1 or 3.
    Reads the national census tracts file.
    Creates a dictionary mapping GEOID to latitude and longitude.
    Maps census_tract in the LAR data to corresponding coordinates.
    Splits the location into separate lat and lon columns.
    Drops rows with missing location data.
    Removes unnecessary columns (census_tract, location).
    Renames action_taken to label.
    Saves the final dataframe to a CSV file named LAR.csv in the specified directory.

    Output
    A CSV file named LAR.csv containing the processed data with the following columns:
    label: Indicates the action taken (1 or 3).
    lat: Latitude of the census tract.
    lon: Longitude of the census tract.
    """

    df = pd.read_csv(lar_data_filename, delimiter="|")
    df.head()

    df = df[["action_taken", "census_tract"]]
    df = df.dropna()
    df["census_tract"] = df["census_tract"].astype(int)
    df["census_tract"] = df["census_tract"].astype(str)

    df = df[(df["action_taken"] == 1) | (df["action_taken"] == 3)]
    df.head()

    loc_df = pd.read_csv(census_gazetteer_data_filename, delimiter="\t")
    loc_df.rename(columns={loc_df.columns[-1]: "INTPTLONG"}, inplace=True)

    tract2loc = dict(zip(loc_df["GEOID"], zip(loc_df["INTPTLAT"], loc_df["INTPTLONG"])))
    df["location"] = df["census_tract"].astype(int).map(tract2loc)
    df[["lat", "lon"]] = pd.DataFrame(df["location"].tolist(), index=df.index)

    df = df.dropna()
    df.head()

    df.drop(columns=["census_tract", "location"], inplace=True, axis=1)

    df.rename(columns={"action_taken": "label"}, inplace=True)
    df["label"] = df["label"].apply(lambda x: 1 if x == 1 else 0)

    df.to_csv(f"{preprocess_dir}/lar.csv", index=False)
    return df
