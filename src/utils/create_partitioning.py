import pandas as pd
import numpy as np
import sys
import os
from rtree import index

sys.path.append(os.path.abspath(os.path.join("..")))
from utils.stats_utils import (
    get_points_not_covered,
)
from utils.geo_utils import (
    create_rtree,
    create_seeds,
    create_regions,
    create_non_over_regions,
)
from utils.data_utils import load_data
from shapely.geometry import Point
from utils.geo_utils import query_range_box
from tqdm import tqdm


def create_kmeans_partioning(
    train_with_loc_filename,
    partioning_base_path,
    dataset_name,
    val_with_loc_filename=None,
    test_with_loc_filename=None,
    overlapping=True,
    k=10,
    radii=None,
    with_partitioning_id=False,
):
    """
    Creates a k-means based partitioning of geographic data into regions
    based on validation and test prediction files.

    Args:
        train_with_loc_filename (str): Path to the train data file with columns lon, lat.
        val_with_loc_filename (str): Path to the validation data file with columns lon, lat.
        test_with_loc_filename (str): Path to the test data file with columns lon, lat.
        partioning_base_path (str): Prefix for the output partitioning files.
        dataset_name (str): Name of the dataset being processed.
        overlapping (bool, optional): If True, creates overlapping regions. Defaults to True.
        k (int, optional): Number of clusters (regions) to create. Defaults to 10.
        radii (list, optional): List of radii values for region creation. Defaults to None.
        with_partitioning_id (bool, optional): If True, saves partitioning IDs. Defaults to False.
    Returns:
        str: Description label used for saving partitioning files.

    The function:
    - Loads validation and test datasets.
    - Merges the datasets for region partitioning.
    - Creates non-overlapping or overlapping regions using spatial methods.
    - Ensures all points are assigned to a region by finding the closest neighbors.
    - Splits the regions into validation and test subsets.
    - Saves the partitioned regions to CSV files.
    - Returns a descriptive label indicating partitioning details.
    """

    train_df = load_data(train_with_loc_filename)

    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        val_df = load_data(val_with_loc_filename)
        test_df = load_data(test_with_loc_filename)

        all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    else:
        all_df = train_df

    # Create the partioning assigned the points to the regions

    if not overlapping:
        regions = create_non_over_regions(all_df, n_seeds=k, rand_seed=36)
    else:
        rtree = create_rtree(all_df)
        seeds = create_seeds(df=all_df, rtree=rtree, n_seeds=k, rand_seed=36)
        regions = create_regions(rtree, seeds, radii)

        print(f"Total seeds: {len(seeds)}")
        print(f"Total radii: {len(radii)}")

    print(f"Total regions created: {len(regions)}")

    # Check if all points are covered by the regions
    # If not, we assign each of the uncovered points
    # to its closest neighbor's region

    points_not_covered = get_points_not_covered(all_df, regions)
    print(
        f"There are {len(points_not_covered)} points that are not covered by the regions"
    )

    same_reg_as_point = []
    for point in tqdm(
        points_not_covered, desc="Search for closest regions to assign uncovered points"
    ):

        coords = (
            all_df.loc[[point]]["lat"].values[0],
            all_df.loc[[point]]["lon"].values[0],
        )
        target_point = Point(coords[1], coords[0])

        # Find the 2 nearest neighbors in the rtree
        nearest_neighbors = rtree.nearest(
            (coords[1], coords[0], coords[1], coords[0]),
            num_results=len(points_not_covered) + 1,
        )

        sorted_neighbors = sorted(
            nearest_neighbors,
            key=lambda idx: target_point.distance(
                Point(all_df.loc[idx, "lat"], all_df.loc[idx, "lon"])
            ),
        )

        found = False
        for nearest_idx in sorted_neighbors:
            if nearest_idx != point and nearest_idx not in points_not_covered:
                found = True
                same_reg_as_point.append(nearest_idx)
                break  # We found the closest point that is not the same
        if not found:
            print(f"Point {point} has no neighbors")
            break

    for idx, point in tqdm(
        enumerate(same_reg_as_point), desc="Adding uncovered points to regions"
    ):
        found = False
        for region in regions:
            if point in region["points"]:
                found = True
                region["points"].append(points_not_covered[idx])
                break

        if not found:
            print(f"Point {point} has no region, idx: {idx}")
            break

    points_not_covered = get_points_not_covered(all_df, regions)
    print(
        f"There are {len(points_not_covered)} points that are not covered by the regions"
    )

    # Split the regions into val and test regions
    # and assign the points to their respective regions

    max_train_idx = train_df.index.max()
    max_val_idx = -1
    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        max_val_idx = val_df.index.max() + max_train_idx + 1

    train_regions = []
    val_regions = []
    test_regions = []
    for region in regions:
        points = region["points"]
        train_points = []
        val_points = []
        test_points = []
        for p in points:
            if p <= max_train_idx:
                train_points.append(p)
            elif p <= max_val_idx:
                val_points.append(p - max_train_idx - 1)
            else:
                test_points.append(p - max_val_idx - 1)

        train_regions.append(
            {
                "points": train_points,
                "center_lat": region["center_lat"],
                "center_lon": region["center_lon"],
                "radius": region.get("radius", None),
            }
        )

        if val_with_loc_filename is not None and test_with_loc_filename is not None:

            val_regions.append(
                {
                    "points": val_points,
                    "center_lat": region["center_lat"],
                    "center_lon": region["center_lon"],
                    "radius": region.get("radius", None),
                }
            )
            test_regions.append(
                {
                    "points": test_points,
                    "center_lat": region["center_lat"],
                    "center_lon": region["center_lon"],
                    "radius": region.get("radius", None),
                }
            )

    points_not_covered = get_points_not_covered(all_df, regions)
    print(
        f"There are {len(points_not_covered)} points that are not covered by the regions"
    )
    points_not_covered = get_points_not_covered(train_df, train_regions)
    print(
        f"There are {len(points_not_covered)} points that are not covered by the regions of train set"
    )
    train_regions_df = pd.DataFrame(train_regions)

    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        points_not_covered = get_points_not_covered(val_df, val_regions)
        print(
            f"There are {len(points_not_covered)} points that are not covered by the regions of val set"
        )
        points_not_covered = get_points_not_covered(test_df, test_regions)
        print(
            f"There are {len(points_not_covered)} points that are not covered by the regions of test set"
        )

        val_regions_df = pd.DataFrame(val_regions)
        test_regions_df = pd.DataFrame(test_regions)

    overlap_label = "overlap" if overlapping else "non_overlap"
    radii_label = f"_radii_{len(radii)}" if overlapping else ""
    desc_label = f"{overlap_label}_k_{k}{radii_label}_{dataset_name}"

    partioning_file_name = f"{partioning_base_path}train_regions_{desc_label}.csv"
    train_regions_df.to_csv(partioning_file_name, index=False)

    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        val_partioning_file_name = f"{partioning_base_path}val_regions_{desc_label}.csv"
        val_regions_df.to_csv(val_partioning_file_name, index=False)

        test_partioning_file_name = (
            f"{partioning_base_path}test_regions_{desc_label}.csv"
        )
        test_regions_df.to_csv(test_partioning_file_name, index=False)

    if with_partitioning_id:
        train_regions = [reg["points"] for reg in train_regions]
        train_regions_id = pts_per_region_to_partioning_id(train_regions)
        train_regions_id_df = pd.DataFrame([train_regions_id])
        train_regions_id_df.to_csv(
            f"{partioning_base_path}train_regions_{desc_label}_partitioning_ids.csv",
            index=False,
        )
        if val_with_loc_filename is not None and test_with_loc_filename is not None:
            val_regions = [reg["points"] for reg in val_regions]
            val_regions_id = pts_per_region_to_partioning_id(val_regions)
            test_regions = [reg["points"] for reg in test_regions]
            test_regions_id = pts_per_region_to_partioning_id(test_regions)
            val_regions_id_df = pd.DataFrame([val_regions_id])
            test_regions_id_df = pd.DataFrame([test_regions_id])
            val_regions_id_df.to_csv(
                f"{partioning_base_path}val_regions_{desc_label}_partitioning_ids.csv",
                index=False,
            )
            test_regions_id_df.to_csv(
                f"{partioning_base_path}test_regions_{desc_label}_partitioning_ids.csv",
                index=False,
            )

    return desc_label


def create_partitioning(
    rtree,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    lon_n: float,
    lat_n: float,
):
    grid_info = {}
    grid_info["lon_min"] = lon_min
    grid_info["lon_max"] = lon_max
    grid_info["lat_min"] = lat_min
    grid_info["lat_max"] = lat_max
    grid_info["lat_n"] = lat_n
    grid_info["lon_n"] = lon_n

    grid_loc2_idx = (
        {}
    )  ## maps (x,y) grid_loc coords to an index in the partitions array

    partitions = []
    for i in range(lat_n):
        lat_start = lat_min + (i / lat_n) * (lat_max - lat_min)
        lat_end = lat_min + ((i + 1) / lat_n) * (lat_max - lat_min)
        for j in range(lon_n):
            lon_start = lon_min + (j / lon_n) * (lon_max - lon_min)
            lon_end = lon_min + ((j + 1) / lon_n) * (lon_max - lon_min)

            points = query_range_box(rtree, lon_start, lon_end, lat_start, lat_end)
            # print(len(points))
            partition = {
                "grid_loc": (j, i),
                "points": points,
            }
            grid_loc2_idx[(j, i)] = len(partitions)
            partitions.append(partition)

    return grid_info, grid_loc2_idx, partitions


def partionings_as_lists(ids, set_ids):
    list_partionings = []
    for id in ids:
        partioning_list = []
        for p in set_ids[id]:
            partioning_list.append(p[0].tolist())
        list_partionings.append(partioning_list)

    return list_partionings


def partioning_ids_to_pts_per_region(X_id):
    pts_per_region = []
    for partitions in X_id.values():
        for partition in partitions:
            pts_per_region.append(list(partition[0]))
    return pts_per_region


def pts_per_region_to_partioning_id(pts_per_region):
    X_id = {
        "id": (1, len(pts_per_region)),
        "partitioning": pts_per_region,
    }
    return X_id


def create_grid_partitioning(
    max_row_partition,
    max_col_partition,
    train_with_loc_filename,
    partitioning_dir,
    dataset_name,
    val_with_loc_filename=None,
    test_with_loc_filename=None,
):
    def create_rtree_(df):
        rtree = index.Index()

        for idx, row in df.iterrows():
            left, bottom, right, top = row["lon"], row["lat"], row["lon"], row["lat"]
            rtree.insert(idx, (left, bottom, right, top))

        return rtree

    desc_label = f"regions_{max_row_partition}_x_{max_col_partition}_{dataset_name}"
    X_train = pd.read_csv(train_with_loc_filename)
    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        X_val = pd.read_csv(val_with_loc_filename)
        X_test = pd.read_csv(test_with_loc_filename)

    X_train_loc = X_train[["lat", "lon"]]
    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        X_val_loc = X_val[["lat", "lon"]]
        X_test_loc = X_test[["lat", "lon"]]

    # RTree for Location fast Searching
    rtree_train = create_rtree_(X_train_loc)
    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        rtree_val = create_rtree_(X_val_loc)
        rtree_test = create_rtree_(X_test_loc)

    # Train Boundary and Test Boundary
    lat_max = X_train_loc["lat"].values.max()
    lat_min = X_train_loc["lat"].values.min()
    lon_max = X_train_loc["lon"].values.max()
    lon_min = X_train_loc["lon"].values.min()
    print("Train Boundary: ", lat_min, lat_max, lon_min, lon_max)

    global_lat_min = lat_min
    global_lat_max = lat_max
    global_lon_min = lon_min
    global_lon_max = lon_max
    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        lat_max_val = X_val_loc["lat"].values.max()
        lat_min_val = X_val_loc["lat"].values.min()
        lon_max_val = X_val_loc["lon"].values.max()
        lon_min_val = X_val_loc["lon"].values.min()
        print("Val Boundary: ", lat_min_val, lat_max_val, lon_min_val, lon_max_val)

        lat_max_test = X_test_loc["lat"].values.max()
        lat_min_test = X_test_loc["lat"].values.min()
        lon_max_test = X_test_loc["lon"].values.max()
        lon_min_test = X_test_loc["lon"].values.min()
        print("Test Boundary: ", lat_min_test, lat_max_test, lon_min_test, lon_max_test)

        global_lat_min = min(lat_min, lat_min_val, lat_min_test)
        global_lat_max = max(lat_max, lat_max_val, lat_max_test)
        global_lon_min = min(lon_min, lon_min_val, lon_min_test)
        global_lon_max = max(lon_max, lon_max_val, lon_max_test)

    print(
        "Global Boundary: ",
        global_lat_min,
        global_lat_max,
        global_lon_min,
        global_lon_max,
    )

    partition_dict_train = {}
    partition_dict_val = {}
    partition_dict_test = {}

    for i in range(1, max_row_partition + 1):
        for j in range(1, max_col_partition + 1):
            grid_info, grid_loc2_idx, regions = create_partitioning(
                rtree_train,
                global_lon_min,
                global_lon_max,
                global_lat_min,
                global_lat_max,
                i,
                j,
            )
            partition_dict_train[(i, j)] = [grid_info, grid_loc2_idx, regions]

            if val_with_loc_filename is not None and test_with_loc_filename is not None:
                grid_info, grid_loc2_idx, regions = create_partitioning(
                    rtree_val,
                    global_lon_min,
                    global_lon_max,
                    global_lat_min,
                    global_lat_max,
                    i,
                    j,
                )
                partition_dict_val[(i, j)] = [grid_info, grid_loc2_idx, regions]

                grid_info, grid_loc2_idx, regions = create_partitioning(
                    rtree_test,
                    global_lon_min,
                    global_lon_max,
                    global_lat_min,
                    global_lat_max,
                    i,
                    j,
                )
                partition_dict_test[(i, j)] = [grid_info, grid_loc2_idx, regions]

    # Construct the partitioning dictionaries
    X_train_id = {}
    X_val_id = {}
    X_test_id = {}

    for i in range(1, max_row_partition + 1):
        for j in range(1, max_col_partition + 1):
            X_train_id[(i, j)] = []
            X_val_id[(i, j)] = []
            X_test_id[(i, j)] = []
            for k in range(i * j):
                X_train_id[(i, j)].append(
                    (np.array(partition_dict_train[(i, j)][2][k]["points"]),)
                )
                if (
                    val_with_loc_filename is not None
                    and test_with_loc_filename is not None
                ):
                    X_val_id[(i, j)].append(
                        (np.array(partition_dict_val[(i, j)][2][k]["points"]),)
                    )
                    X_test_id[(i, j)].append(
                        (np.array(partition_dict_test[(i, j)][2][k]["points"]),)
                    )

    ids = []
    row_list = list(range(1, max_row_partition + 1))
    col_list = list(range(1, max_col_partition + 1))

    for r in row_list:
        for c in col_list:
            # Eliminate the possibility of drawing to whole set (1,1) partition
            if r == 1 and c == 1:
                continue
            ids.append((r, c))

    list_partitionings_train = partionings_as_lists(ids, X_train_id)
    train_id_df = pd.DataFrame({"id": ids, "partitioning": list_partitionings_train})
    train_id_df.to_csv(
        f"{partitioning_dir}train_{desc_label}_partitioning_ids.csv", index=False
    )
    X_train_id_cp = X_train_id.copy()
    X_train_id_cp.pop((1, 1))
    pts_per_region_train = partioning_ids_to_pts_per_region(X_train_id_cp)
    train_regions_df = pd.DataFrame([{"points": pts} for pts in pts_per_region_train])
    train_regions_df.to_csv(f"{partitioning_dir}train_{desc_label}.csv", index=False)

    if val_with_loc_filename is not None and test_with_loc_filename is not None:
        list_partitionings_val = partionings_as_lists(ids, X_val_id)
        list_partitionings_test = partionings_as_lists(ids, X_test_id)
        val_id_df = pd.DataFrame({"id": ids, "partitioning": list_partitionings_val})
        test_id_df = pd.DataFrame({"id": ids, "partitioning": list_partitionings_test})

        val_id_df.to_csv(
            f"{partitioning_dir}val_{desc_label}_partitioning_ids.csv", index=False
        )
        test_id_df.to_csv(
            f"{partitioning_dir}test_{desc_label}_partitioning_ids.csv",
            index=False,
        )

        X_val_id_cp = X_val_id.copy()
        X_test_id_cp = X_test_id.copy()
        X_val_id_cp.pop((1, 1))
        X_test_id_cp.pop((1, 1))

        pts_per_region_val = partioning_ids_to_pts_per_region(X_val_id_cp)
        pts_per_region_test = partioning_ids_to_pts_per_region(X_test_id_cp)

        val_regions_df = pd.DataFrame([{"points": pts} for pts in pts_per_region_val])
        test_regions_df = pd.DataFrame([{"points": pts} for pts in pts_per_region_test])

        val_regions_df.to_csv(f"{partitioning_dir}val_{desc_label}.csv", index=False)
        test_regions_df.to_csv(f"{partitioning_dir}test_{desc_label}.csv", index=False)
