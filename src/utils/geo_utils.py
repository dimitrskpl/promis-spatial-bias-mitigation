from sklearn.cluster import KMeans
from rtree import index
from shapely.geometry import MultiPoint


def create_rtree(df):
    """
    Creates an R-tree spatial index for efficient querying of spatial points.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns representing longitude and latitude.

    Returns:
        rtree.Index: An R-tree spatial index for the points in the DataFrame.
    """

    rtree = index.Index()

    for idx, row in df.iterrows():
        left, bottom, right, top = row["lon"], row["lat"], row["lon"], row["lat"]
        rtree.insert(idx, (left, bottom, right, top))

    return rtree


def filterbbox(df, min_lon, min_lat, max_lon, max_lat):
    """
    Filters the DataFrame to include only points within a specified bounding box.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns representing longitude and latitude.
        min_lon (float): Minimum longitude of the bounding box.
        min_lat (float): Minimum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.

    Returns:
        pd.DataFrame: A DataFrame containing points within the bounding box.
    """

    df = df.loc[df["lon"] >= min_lon]
    df = df.loc[df["lon"] <= max_lon]
    df = df.loc[df["lat"] >= min_lat]
    df = df.loc[df["lat"] <= max_lat]
    # df.reset_index(drop=True, inplace=True)

    return df


def create_seeds(df, rtree, n_seeds, rand_seed=None):
    """
    Uses K-means clustering to create seed points from clusters of geographical points.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        rtree (rtree.Index): An R-tree spatial index for efficient querying.
        n_seeds (int): Number of seeds (clusters) to generate.
        rand_seed (int, optional): Random seed for reproducibility.

    Returns:
        list: A list of indices representing seed points.
    """

    # Compute clusters with k-means
    X = df[["lon", "lat"]].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init="auto", random_state=rand_seed).fit(X)
    cluster_centers = kmeans.cluster_centers_

    # Pick seeds from cluster centroids
    seeds = []
    for c in cluster_centers:
        nearest_idx = list(rtree.nearest([c[0], c[1]], 1))[0]
        lat = df.loc[[nearest_idx]]["lat"].values[0]
        lon = df.loc[[nearest_idx]]["lon"].values[0]
        seeds.append((lat, lon))
    return seeds


def create_non_over_regions(df, n_seeds, rand_seed=None):
    """
    Uses K-means clustering to create n_seeds regions with non-overlapping points.
    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        n_seeds (int): Number of seeds (clusters) to generate.
        rand_seed (int, optional): Random seed for reproducibility.

    Returns:
        list: A list of indices representing seed points.
    """

    # Compute clusters with k-means
    X = df[["lon", "lat"]].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init="auto", random_state=rand_seed).fit(X)
    regions = []
    for seed in range(n_seeds):
        points = df.index[kmeans.labels_ == seed].tolist()
        region = {
            "points": points,
            "center": seed,
        }
        regions.append(region)

    return regions


def query_range(rtree, center_lat, center_lon, radius):
    """
    Queries points within a square bounding box centered at a specific point.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        rtree (rtree.Index): An R-tree spatial index for efficient querying.
        center (int): Index of the center point in the DataFrame.
        radius (float): Radius of the square bounding box to query.

    Returns:
        list: List of indices of points within the square bounding box.
    """

    ## for now returns points within square

    left, bottom, right, top = (
        center_lon - radius,
        center_lat - radius,
        center_lon + radius,
        center_lat + radius,
    )
    result = list(rtree.intersection((left, bottom, right, top)))
    # keep points that are in circle
    # tmp_result = []
    # for point in result:
    #     p_lat, p_lon = id2loc(df, point)
    #     dist = math.sqrt( (p_lon-lon)**2 + (p_lat-lat)**2 )
    #     if dist <= radius:
    #         tmp_result.append(point)
    # result = tmp_result

    return result


def create_regions(rtree, seeds, radii):
    """
    Creates regions around seed points with varying radii.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        rtree (rtree.Index): An R-tree spatial index for efficient querying.
        seeds (list): List of seed point indices.
        radii (list): List of radii for each region.

    Returns:
        list: List of dictionaries, each representing a region with 'points', 'center', and 'radius'.
    """

    regions = []
    for lat, lon in seeds:
        for radius in radii:
            points = query_range(rtree, lat, lon, radius)
            region = {
                "points": points,
                "center_lat": lat,
                "center_lon": lon,
                "radius": radius,
            }
            regions.append(region)

    return regions


def query_range_box(rtree, xmin, xmax, ymin, ymax):
    """
    Queries points within a rectangular bounding box.

    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        rtree (rtree.Index): An R-tree spatial index for efficient querying.
        xmin (float): Minimum longitude of the bounding box.
        xmax (float): Maximum longitude of the bounding box.
        ymin (float): Minimum latitude of the bounding box.
        ymax (float): Maximum latitude of the bounding box.

    Returns:
        list: List of indices of points within the bounding box.
    """

    left, bottom, right, top = xmin, ymin, xmax, ymax
    result = list(rtree.intersection((left, bottom, right, top)))

    return result


def create_non_over_regions(df, n_seeds, rand_seed=None):
    """
    Uses K-means clustering to create n_seeds regions with non-overlapping points.
    Args:
        df (pd.DataFrame): DataFrame containing 'lon' and 'lat' columns.
        n_seeds (int): Number of seeds (clusters) to generate.
        rand_seed (int, optional): Random seed for reproducibility.

    Returns:
        list: A list of indices representing seed points.
    """

    # Compute clusters with k-means
    X = df[["lon", "lat"]].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init="auto", random_state=rand_seed).fit(X)
    cluster_centers = kmeans.cluster_centers_

    regions = []
    for idx, center in enumerate(cluster_centers):
        points = df.index[kmeans.labels_ == idx].tolist()
        region = {
            "points": points,
            "center_lat": center[1],
            "center_lon": center[0],
        }
        regions.append(region)

    return regions


def compute_polygons(regions_df, df_points):
    """
    Computes convex hull polygons for each region using Shapely.

    Args:
        regions_df (pd.DataFrame): DataFrame containing 'center_lat', 'center_lon', and 'points' columns.
        df_points (pd.DataFrame): DataFrame containing 'lat' and 'lon' columns for points.

    Returns:
        pd.DataFrame: Updated regions_df with new 'polygon' column.
    """
    polygons = []

    for _, row in regions_df.iterrows():
        points = row["points"]

        if not points:
            polygons.append(None)
            continue

        # Get coordinates of points in the region
        region_points = df_points.iloc[points][["lon", "lat"]].values

        # Compute the convex hull (polygon boundary)
        hull = MultiPoint(region_points).convex_hull
        polygons.append(
            list(hull.exterior.coords) if hull.geom_type == "Polygon" else None
        )

    regions_df["polygon"] = polygons
    return regions_df
