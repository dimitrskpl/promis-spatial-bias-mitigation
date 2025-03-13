import folium
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import base64
import io
import matplotlib.colors as mcolors

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_fairness_map(
    regs_df_list=[],
    title="",
    score_label=None,
    center_loc=[34.067133814231646, -118.26273042624089],
):
    """
    Plots polygon-shaped regions on an interactive Folium map,
    color-coded by a fairness score.

    Args:
        center_loc (list, optional): Latitude and longitude of the map center.
        regs_df_list (list of pandas.DataFrame):
            List of DataFrames. Each DataFrame must have:
              - a 'polygon' column: polygon boundaries as a list of (longitude, latitude) tuples
              - a score column named `score_label`: numeric fairness scores between -1 and 1
        title (str, optional): Title for the Folium map.
        score_label (str, optional): Name of the column holding the fairness score (-1 to 1).

    Returns:
        folium.Map: Folium map object with polygons added.
    """
    # Define colormap from -1 (blue) to +1 (red) with 0 (white) in the middle.
    cmap = cm.get_cmap("coolwarm")
    norm = colors.Normalize(vmin=-1, vmax=1)

    mapit = folium.Map(
        location=center_loc,
        zoom_start=10,
    )

    for regs_df in regs_df_list:
        for _, row in regs_df.iterrows():
            polygon = row.get("polygon", None)
            score = row.get(score_label, None)
            if polygon is None or score is None:
                continue

            rgba_color = cmap(norm(score))
            hex_color = colors.to_hex(rgba_color)

            folium.Polygon(
                locations=[(lat, lon) for lon, lat in polygon],
                color="black",
                fill=True,
                fill_opacity=0.9,
                fill_color=hex_color,
                weight=2,
                tooltip=(f"Fairness metric: {score:.2f}"),
            ).add_to(mapit)

    if title:
        title_html = f"""
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        """
        mapit.get_root().html.add_child(folium.Element(title_html))

    fig, ax = plt.subplots(figsize=(5, 1.2))
    fig.subplots_adjust(bottom=0.4, top=0.9, left=0.05, right=0.95)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=ax, orientation="horizontal")

    cb.set_ticks([-1, 0, 1])
    cb.ax.set_xticklabels(["Unfavored (-1)", "Fair (0)", "Favored (1)"], fontsize=15)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight", transparent=True)
    img.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img.read()).decode("utf-8")

    legend_html = f"""
    <div style="
        position: fixed;
        top: 10px; right: 10px;
        z-index:9999;
        width: 300px;
        background-color: white;
        border-radius: 5px;
        padding: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        text-align: center;
        font-size:14px;
    ">
        <b>Fairness Score Legend</b>
        <br>
        <img src="data:image/png;base64,{img_base64}" style="width: 100%;">
    </div>
    """

    mapit.get_root().html.add_child(folium.Element(legend_html))

    css = f"""
    <style>
        .leaflet-tile {{
            filter: brightness({0.5:.2f});
        }}
    </style>
    """
    mapit.get_root().html.add_child(folium.Element(css))

    return mapit


def plot_map_with_polygons(
    df=None,
    y_pred=None,
    regs_df_list=[],
    regs_color_list=[],
    other_idxs=None,
    other_colors=None,
    title="",
):
    """
    Plots points and polygon-shaped regions on an interactive map using Folium.

    Args:
        df (pd.DataFrame): DataFrame containing latitude ('lat') and longitude ('lon') columns.
        y_pred (np.ndarray): Array of true types (binary labels) for the points.
        regs_df_list (list, optional): List of DataFrames, each representing regions to be plotted.
        regs_color_list (list, optional): List of colors corresponding to each region in `regs_df_list`.
        other_idxs (list, optional): List of point index lists for additional colored points. Defaults to None.
        other_colors (list, optional): List of colors corresponding to `other_idxs`. Defaults to None.
        title (str, optional): Title of the map.

    Returns:
        folium.Map: Folium map with points and polygons plotted.
    """
    center_loc = [34.067133814231646, -118.26273042624089]
    mapit = folium.Map(
        location=center_loc,
        zoom_start=10,
    )
    # Add points
    if y_pred is not None:
        indices = df.index
        shuffled_indices = np.random.permutation(indices)
        for index in shuffled_indices:
            color = "green" if y_pred[index] == 1 else "#FF0000"
            folium.CircleMarker(
                location=(df.at[index, "lat"], df.at[index, "lon"]),
                color=color,
                fill_color=color,
                fill=True,
                opacity=0.4,
                fill_opacity=0.4,
                radius=0.2 if y_pred[index] == 1 else 0.4,
                weight=0.1 if y_pred[index] == 1 else 0.2,
            ).add_to(mapit)

    if other_idxs != None and other_colors != None:
        if len(other_idxs) != len(other_colors):
            raise ValueError("len(other_idxs) != len(other_colors)")

        for idx_list, color in zip(other_idxs, other_colors):
            shuffled_idx_list = np.random.permutation(idx_list)
            for index in shuffled_idx_list:
                folium.CircleMarker(
                    location=(df.at[index, "lat"], df.at[index, "lon"]),
                    color=color,
                    fill_color=color,
                    fill=True,
                    opacity=0.4,
                    fill_opacity=0.4,
                    radius=0.2,
                    weight=0.1,
                ).add_to(mapit)

    if regs_df_list and regs_color_list:
        for regs_df, regs_color in zip(regs_df_list, regs_color_list):
            for _, row in regs_df.iterrows():
                polygon = row.get("polygon", None)
                if polygon:
                    folium.Polygon(
                        locations=[(lat, lon) for lon, lat in polygon],
                        color="black",
                        fill=True,
                        fill_color=regs_color,
                        fill_opacity=0,
                        weight=2,
                        tooltip=f'Region Center: ({row["center_lat"]}, {row["center_lon"]})',
                    ).add_to(mapit)

    if title:
        title_html = f"""
        <h3 align="center" style="font-size:20px"><b>{title}</b></h3>
        """
        mapit.get_root().html.add_child(folium.Element(title_html))

    css = f"""
    <style>
        .leaflet-tile {{
            filter: brightness({0.5:.2f});
        }}
    </style>
    """
    mapit.get_root().html.add_child(folium.Element(css))

    return mapit


def plot_graphs(
    values_lists,
    labels,
    title,
    xlabel,
    ylabel,
    colors,
    x_values=[],
    save_path="",
    figsize=(16, 8),
    linewidths=None,
    linestyles=None,
    markers=None,
    x_sticks_step=None,
    scatter_plot=False,
    rev_xaxis=False,
    scatter_markers=None,
    axhline=None,
    axhline_label=None,
    annotate_points=None,
    marker_sizes=None,
    line_plot=True,
    plot_legend=True,
    show_plot=True,
    ax=None,
    zorders=None,
):
    """
    Plots multiple data series with customizable line styles, widths, and markers

    Args:
        values_lists (list of lists): List of y-values for multiple data series.
        labels (list of str): Labels for the different data series (used in the legend).
        title (str): The title of the graph.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        colors (list of str): Colors for each data series.
        x_values (list of lists, optional): X-values for each data series. Defaults to an empty list.
        save_path (str, optional): Path to save the figure as an SVG file. Defaults to "" (not saved).
        figsize (tuple, optional): Figure size in inches. Defaults to (16, 8).
        linewidths (list, optional): Line widths for each series. Defaults to None (set to 3).
        linestyles (list, optional): Line styles for each series. Defaults to None (set to "-").
        markers (list, optional): Markers for each series. Defaults to None (set to "").
        bold_labels (list, optional): Labels to be displayed in bold in the legend. Defaults to an empty list.
        x_sticks_step (int, optional): Step size for x-axis ticks. Defaults to None (no modification).
        scatter_plot (bool, optional): If True, adds scatter points to the graph. Defaults to False.
        rev_xaxis (bool, optional): If True, reverses the x-axis. Defaults to False.
        scatter_markers (list, optional): Markers for scatter points. Defaults to None (set to "o").
        axhline (float, optional): Y-value for a horizontal reference line. Defaults to None (no line).
        axhline_label (str, optional): Label for the horizontal reference line. Defaults to None.
        annotate_points (list, optional): List of dictionaries containing annotations for specific points. Defaults to None.
        marker_sizes (list, optional): Sizes for scatter markers. Defaults to None (set to 150).
        line_plot (bool, optional): If True, plots lines. Defaults to True.
        plot_legend (bool, optional): If True, displays the legend. Defaults to True.
        show_plot (bool, optional): If True, displays the plot. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        zorders (list, optional): Z-orders for each series. Defaults to None (set to 2).

    Functionality:
        - Supports both line and scatter plots.
        - Allows setting x-axis tick steps and reversing the x-axis.
        - Provides an option to add a horizontal reference line.
        - Saves the figure if `save_path` is provided.
    """

    if len(values_lists) != len(colors) or len(values_lists) != len(labels):
        raise ValueError(
            "values_lists size should be same as colors size and labels size"
        )

    _, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)

    if linewidths is None:
        linewidths = [3] * len(colors)

    if linestyles is None:
        linestyles = ["-"] * len(colors)

    if markers is None:
        markers = [""] * len(colors)

    if marker_sizes is None:
        marker_sizes = [150] * len(colors)

    zorders = [2] * len(values_lists) if zorders is None else zorders

    if line_plot:
        if x_values:
            for i in range(len(values_lists)):
                ax.plot(
                    x_values[i],
                    values_lists[i],
                    label=labels[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                    linestyle=linestyles[i],
                    marker=markers[i],
                    zorder=zorders[i],
                )

            if x_sticks_step:
                min_x = min(min(x) for x in x_values)
                max_x = max(max(x) for x in x_values)
                ax.set_xticks(np.arange(min_x, max_x + 1, x_sticks_step))
        else:
            for i in range(len(values_lists)):
                ax.plot(
                    values_lists[i],
                    label=labels[i],
                    color=colors[i],
                    linewidth=linewidths[i],
                    linestyle=linestyles[i],
                    marker=markers[i],
                    zorder=zorders[i],
                )
    if scatter_plot:
        if scatter_markers == None:
            scatter_markers = ["o"] * len(values_lists)
        for i in range(len(values_lists)):
            ax.scatter(
                x_values[i],
                values_lists[i],
                color=colors[i],
                s=marker_sizes[i],
                marker=scatter_markers[i],
                label=labels[i],
                zorder=zorders[i],
            )

    if annotate_points:
        for annotation in annotate_points:
            series_idx = annotation["series_idx"]
            point_idx = annotation["point_idx"]

            x_coord = x_values[series_idx][point_idx] if x_values else point_idx
            y_coord = values_lists[series_idx][point_idx]

            text = annotation.get("text", f"({x_coord}, {y_coord})")

            ax.annotate(
                text,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(0, 50),
                ha="center",
                fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1),
            )

    if axhline is not None and axhline_label is not None:
        ax.axhline(
            y=axhline,
            color="red",
            linestyle="-",
            linewidth=linewidths[0],
            label=f"{axhline_label}",
        )

    if plot_legend:
        unique_label_color_marker = list(zip(labels, colors, scatter_markers))
        method_legend_handles = [
            plt.Line2D(
                [0], [0], marker=m, color="w", markerfacecolor=c, markersize=10, label=l
            )
            for (l, c, m) in sorted(unique_label_color_marker, key=lambda x: x[0])
        ]

        ax.legend(
            handles=method_legend_handles,
            loc="best",
            labelspacing=0.2,
            handletextpad=0.1,
            framealpha=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    if rev_xaxis:
        plt.gca().invert_xaxis()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    return ax


def get_linewidths_colors(labels, method_to_plot_info):
    """
    Retrieves line widths and colors for given labels based on a mapping.

    Args:
        labels (list): List of labels for the graphs.
        method_to_plot_info (dict): Dictionary containing line width and color information for each label.

    Returns:
        tuple: List of line widths and list of colors for the labels.
    """

    linewidths = [method_to_plot_info[label]["linewidth"] for label in labels]
    colors = [method_to_plot_info[label]["color"] for label in labels]
    return linewidths, colors


def plot_opt_methods_status(
    labels,
    budget_lists,
    status_lists,
    save_path="",
    figsize=(20, 8),
    title="Number of Flips where Methods Reached Time Limit",
    display_title=True,
):
    """
    Plots the number of flips where different optimization methods reached the time limit.

    Args:
        labels (list of str): Labels for different optimization methods.
        budget_lists (list of lists): Each sublist contains budget values (number of flips) for a method.
        status_lists (list of lists): Each sublist contains status values corresponding to budget values.
                                      A status of `3` indicates that the method reached the time limit.
        save_path (str, optional): Path to save the plot as an SVG file. Defaults to "" (not saved).
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        title (str, optional): Title of the plot. Defaults to "Number of Flips where Methods Reached Time Limit".
        exp_idx (int, optional): Experiment index to include in the title and filename. Defaults to None.
        display_title (bool, optional): Whether to display the plot title. Defaults to True.

    Raises:
        ValueError: If the number of labels does not match the number of status lists.

    Functionality:
        - Filters the budget values where the status is `3` (indicating time limit reached).
        - Plots a scatter plot showing where each method encountered the time limit.
        - Optionally includes an experiment index in the title and filename.
        - Saves the plot if `save_path` is specified.
    """

    if len(labels) != len(status_lists):
        raise ValueError("labels size should be equal to status_lists size")

    encoded_status_lists = []
    for i, (budget_list, status_list) in enumerate(zip(budget_lists, status_lists)):
        encoded_status_lists.append(
            (
                [
                    (flip_no, i)
                    for flip_no, x in zip(budget_list, status_list)
                    if x == 3
                ],
                labels[i],
            )
        )

    cnt = 0
    for encoded_status_list in encoded_status_lists:
        if encoded_status_list[0]:
            cnt += 1
            x_filtered, y_filtered = zip(*encoded_status_list[0])
            if cnt == 1:
                plt.figure(figsize=figsize)

            plt.scatter(x_filtered, y_filtered, label=f"{encoded_status_list[1]}")

    if cnt > 0:
        # plt.xlim(-1, len(status_lists[0]))

        if display_title:
            plt.title(title)

        plt.xlabel("Number of Flips Allowed (B)")
        plt.yticks([])

        plt.legend()

        if save_path:
            plt.savefig(f"{save_path}opt_status.pdf", format="pdf")

        plt.show()


def plot_scores(
    methods_to_res_info,
    init_sbi,
    method_to_plot_info,
    method_to_display_name,
    save_plots_path="",
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    append_to_title="",
    append_to_save="",
    score_label="sbi",
    display_title=True,
    other_sbi=None,
    other_sbi_method=None,
):
    """
    Plots the SBI score drop comparison across different methods.

    This function visualizes the performance of different optimization strategies by plotting
    their SBI scores over varying budgets. It allows customization of line styles, colors,
    and other graphical elements.

    Args:
        methods_to_res_info (dict): A dictionary mapping method names to their corresponding
            results DataFrame, which includes performance metrics.
        init_sbi (float): The initial SBI value before any optimization.
        method_to_plot_info (dict): A dictionary containing plot attributes for each method
            (e.g., color, linewidth, linestyle, marker size).
        method_to_display_name (dict): A mapping from method names to their display labels.
        save_plots_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of budget flips to include in the plot. Defaults to None.
        optim_sols_only (bool, optional): If True, filters only optimal solutions (status=1) from the results. Defaults to False.
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        score_label (str, optional): Column name in the DataFrame representing the score to plot. Defaults to "sbi".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        other_sbi (float, optional): A horizontal reference line at a specific SBI value. Defaults to None.
        other_sbi_method (str, optional): Label for the horizontal reference line. Defaults to None.

    Returns:
        None
    """

    scatter_sbis = []
    line_sbis = []

    labels = []
    line_budget_list = []
    scatter_budget_list = []
    colors = []
    linewidths = []
    linestyles = []
    markers_sz = []
    markers = []

    for method, method_res_info_df in methods_to_res_info.items():
        labels.append(method_to_display_name[method])
        method_res_info_df_cp = method_res_info_df.copy()
        if optim_sols_only and "status" in method_res_info_df_cp.columns:
            method_res_info_df_cp = method_res_info_df_cp[
                method_res_info_df_cp["status"] == 1
            ]
        if flips_limit:
            method_res_info_df_cp = method_res_info_df_cp[
                method_res_info_df_cp["budget"] <= flips_limit
            ]
        method_sbis = method_res_info_df_cp[score_label].to_list()
        line_sbis.append([init_sbi] + method_sbis)
        scatter_sbis.append(method_sbis)

        method_budget = method_res_info_df_cp["budget"].to_list()
        line_budget_list.append([0] + method_budget)
        scatter_budget_list.append(method_budget)

        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
        markers_sz.append(method_to_plot_info[method]["marker_size"])
        markers.append(method_to_plot_info[method]["scatter_marker"])

    if other_sbi is not None:
        labels.append(method_to_display_name[other_sbi_method])
        line_sbis.append([init_sbi, other_sbi])
        scatter_sbis.append([other_sbi])
        line_budget_list.append([0, max(line_budget_list[-1])])
        scatter_budget_list.append([max(line_budget_list[-1])])
        colors.append(method_to_plot_info[other_sbi_method]["color"])
        linewidths.append(method_to_plot_info[other_sbi_method]["linewidth"])
        linestyles.append(method_to_plot_info[other_sbi_method]["linestyle"])
        markers_sz.append(method_to_plot_info[other_sbi_method]["marker_size"])
        markers.append(method_to_plot_info[other_sbi_method]["scatter_marker"])

    title = f"Strategies SBI drop comparison{append_to_title}"
    xlabel = "Number of Flips Allowed (B)"
    ylabel = "SBI"
    sbi_save_path = ""
    if save_plots_path:
        sbi_save_path = f"{save_plots_path}methods_sbi{append_to_save}.pdf"

    ax = plot_graphs(
        values_lists=line_sbis,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        title=title if display_title else "",
        save_path=sbi_save_path,
        figsize=figsize,
        linewidths=linewidths,
        x_values=line_budget_list,
        linestyles=linestyles,
        scatter_plot=False,
        marker_sizes=markers_sz,
        scatter_markers=markers,
        line_plot=True,
        plot_legend=False,
        show_plot=False,
    )

    plot_graphs(
        values_lists=scatter_sbis + [init_sbi],
        labels=labels + [method_to_display_name["init"]],
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors + [method_to_plot_info["init"]["color"]],
        title=title if display_title else "",
        save_path=sbi_save_path,
        figsize=figsize,
        linewidths=linewidths + [method_to_plot_info["init"]["linewidth"]],
        x_values=scatter_budget_list + [0],
        linestyles=linestyles + [method_to_plot_info["init"]["linestyle"]],
        scatter_plot=True,
        marker_sizes=markers_sz + [method_to_plot_info["init"]["marker_size"]],
        scatter_markers=markers + [method_to_plot_info["init"]["scatter_marker"]],
        line_plot=False,
        ax=ax,
        zorders=[2] * len(labels) + [3],
    )


def plot_compare_methods_info(
    all_methods_to_results_info,
    init_p,
    init_rho,
    p_label,
    rho_label,
    actual_flips_label,
    method_to_plot_info,
    method_to_display_name,
    save_path="",
    figsize=(20, 8),
    append_to_title="",
    display_title=True,
    other_P=None,
    other_RHO=None,
    other_actual_flips=None,
    other_actual_pos_flips=None,
    other_method=None,
):
    """
    Compares and plots method-specific performance metrics across different strategies.

    This function visualizes multiple performance metrics, including positive predictions,
    positive prediction ratios, and actual flips, across various methods and budgets.

    Args:
        all_methods_to_results_info (dict): A dictionary mapping method names to their
            corresponding results DataFrame containing performance metrics.
        init_p (float): Initial value of the positive predictions metric.
        init_rho (float): Initial value of the positive prediction ratio metric.
        p_label (str): Column name representing the positive predictions metric.
        rho_label (str): Column name representing the positive ratio of predictions metric.
        actual_flips_label (str): Column name representing the actual number of flips metric.
        method_to_plot_info (dict): A dictionary containing plotting attributes for each method
            (e.g., color, linewidth, linestyle, marker size).
        method_to_display_name (dict): A mapping from method names to their display labels.
        save_path (str, optional): Path to save the generated plots. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plots. Defaults to (20, 8).
        append_to_title (str, optional): Additional text to append to the plot titles. Defaults to "".
        display_title (bool, optional): Whether to display plot titles. Defaults to True.
        other_P (float, optional): Additional value for the positive predictions plot. Defaults to None.
        other_RHO (float, optional): Additinal value for the positive ratio of predictions plot. Defaults to None.
        other_actual_flips (int, optional): Additional value for the actual flips plot. Defaults to None.
        other_actual_pos_flips (int, optional): Additional value for the actual positive flips plot. Defaults to None.
        other_method (str, optional): Additional method. Defaults to None.

    Returns:
        None
    """

    labels = []
    p_list = []
    rho_list = []
    actual_flips_list = []
    actual_flips_pos_list = []
    budgets_list = []
    all_linewidths = []
    all_colors = []
    all_linestyles = []
    markers_sz = []
    markers = []

    for method_name, res_info_df in all_methods_to_results_info.items():
        labels.append(method_to_display_name[method_name])
        p_list.append([init_p] + res_info_df[p_label].tolist())
        rho_list.append([init_rho] + res_info_df[rho_label].tolist())
        actual_flips_list.append([0] + res_info_df[actual_flips_label].tolist())
        if f"{actual_flips_label}_pos" in res_info_df.columns:
            actual_flips_pos_list.append(
                [0] + res_info_df[f"{actual_flips_label}_pos"].tolist()
            )
        budgets_list.append([0] + res_info_df["budget"].to_list())
        all_colors.append(method_to_plot_info[method_name]["color"])
        all_linewidths.append(method_to_plot_info[method_name]["linewidth"])
        all_linestyles.append(method_to_plot_info[method_name]["linestyle"])
        markers_sz.append(method_to_plot_info[method_name]["marker_size"])
        markers.append(method_to_plot_info[method_name]["scatter_marker"])

    if other_P and other_RHO and other_actual_flips and other_actual_pos_flips:
        labels.append(method_to_display_name[other_method])
        p_list.append([init_p, other_P])
        rho_list.append([init_rho, other_RHO])
        actual_flips_list.append([0, other_actual_flips])
        actual_flips_pos_list.append([0, other_actual_pos_flips])
        budgets_list.append([0, max(budgets_list[0])])
        all_colors.append(method_to_plot_info[other_method]["color"])
        all_linewidths.append(method_to_plot_info[other_method]["linewidth"])
        all_linestyles.append(method_to_plot_info[other_method]["linestyle"])
        markers_sz.append(method_to_plot_info[other_method]["marker_size"])
        markers.append(method_to_plot_info[other_method]["scatter_marker"])

    zorders = [2] * len(labels)

    x_values_scatter = [x[1:] for x in budgets_list] + [0]
    labels_scatter = labels + [method_to_display_name["init"]]
    all_colors_scatter = all_colors + [method_to_plot_info["init"]["color"]]
    all_linewidths_scatter = all_linewidths + [method_to_plot_info["init"]["linewidth"]]
    all_linestyles_scatter = all_linestyles + [method_to_plot_info["init"]["linestyle"]]
    markers_sz_scatter = markers_sz + [method_to_plot_info["init"]["marker_size"]]
    markers_scatter = markers + [method_to_plot_info["init"]["scatter_marker"]]
    zorders_scatter = zorders + [3]
    xlabel = "Number of Flips Allowed (B)"
    ax_p = plot_graphs(
        values_lists=p_list,
        labels=labels,
        xlabel=xlabel,
        ylabel="Positive Predictions",
        colors=all_colors,
        title="",
        linewidths=all_linewidths,
        figsize=figsize,
        linestyles=all_linestyles,
        x_values=budgets_list,
        scatter_plot=False,
        plot_legend=False,
        zorders=zorders,
        show_plot=False,
    )
    plot_graphs(
        values_lists=[x[1:] for x in p_list] + [init_p],
        labels=labels_scatter,
        xlabel=xlabel,
        ylabel="Positive Predictions",
        colors=all_colors_scatter,
        title=(
            f"{p_label} per Number of Flips{append_to_title}" if display_title else ""
        ),
        save_path=f"{save_path}methods_{p_label}.pdf" if save_path else "",
        linewidths=all_linewidths_scatter,
        figsize=figsize,
        linestyles=all_linestyles_scatter,
        x_values=x_values_scatter,
        scatter_plot=True,
        marker_sizes=markers_sz_scatter,
        scatter_markers=markers_scatter,
        ax=ax_p,
        line_plot=False,
        plot_legend=True,
        zorders=zorders_scatter,
        show_plot=True,
    )

    ax_rho = plot_graphs(
        values_lists=rho_list,
        labels=labels,
        xlabel=xlabel,
        ylabel="Positive Ratio of Predictions",
        colors=all_colors,
        title="",
        linewidths=all_linewidths,
        figsize=figsize,
        linestyles=all_linestyles,
        x_values=budgets_list,
        scatter_plot=False,
        plot_legend=False,
        zorders=zorders,
        show_plot=False,
    )
    plot_graphs(
        values_lists=[x[1:] for x in rho_list] + [init_rho],
        labels=labels_scatter,
        xlabel=xlabel,
        ylabel="Positive Ratio of Predictions",
        colors=all_colors_scatter,
        title=(
            f"{rho_label} per Number of Flips{append_to_title}" if display_title else ""
        ),
        save_path=f"{save_path}methods_{rho_label}.pdf" if save_path else "",
        linewidths=all_linewidths_scatter,
        figsize=figsize,
        linestyles=all_linestyles_scatter,
        x_values=x_values_scatter,
        scatter_plot=True,
        marker_sizes=markers_sz_scatter,
        scatter_markers=markers_scatter,
        ax=ax_rho,
        line_plot=False,
        plot_legend=True,
        zorders=zorders_scatter,
        show_plot=True,
    )

    amethod_res_df = all_methods_to_results_info[
        list(all_methods_to_results_info.keys())[0]
    ]
    n_flips_list = [[0] + amethod_res_df["budget"].tolist()]

    ax_act_flips = plot_graphs(
        values_lists=actual_flips_list + n_flips_list,
        labels=labels + ["Flips Constraint"],
        xlabel=xlabel,
        ylabel="Actual Number of Flips",
        colors=all_colors + ["orange"],
        title="",
        save_path="",
        linewidths=all_linewidths + [all_linewidths[0]],
        figsize=figsize,
        linestyles=all_linestyles + [all_linestyles[0]],
        x_values=budgets_list + n_flips_list,
        scatter_plot=False,
        plot_legend=False,
        zorders=zorders + [zorders[0]],
        show_plot=False,
    )
    plot_graphs(
        values_lists=[x[1:] for x in actual_flips_list] + [[0]] + [n_flips_list[0][1:]],
        labels=labels_scatter + ["Flips Constraint"],
        xlabel=xlabel,
        ylabel="Actual Number of Flips",
        colors=all_colors_scatter + ["orange"],
        title=(
            f"Actual Flips per Number of Flips{append_to_title}"
            if display_title
            else ""
        ),
        save_path=f"{save_path}methods_{actual_flips_label}.pdf" if save_path else "",
        linewidths=all_linewidths_scatter + [all_linewidths_scatter[0]],
        figsize=figsize,
        linestyles=all_linestyles_scatter + [all_linestyles_scatter[0]],
        x_values=x_values_scatter + [n_flips_list[0][1:]],
        scatter_plot=True,
        marker_sizes=markers_sz_scatter + [markers_sz_scatter[0]],
        scatter_markers=markers_scatter + ["v"],
        ax=ax_act_flips,
        line_plot=False,
        plot_legend=True,
        zorders=zorders_scatter + [zorders[0]],
        show_plot=True,
    )
    if len(actual_flips_pos_list):
        ax_act_pos_flips = plot_graphs(
            values_lists=actual_flips_pos_list + n_flips_list,
            labels=labels + ["Flips Constraint"],
            xlabel=xlabel,
            ylabel="Positive Actual Number of Flips",
            colors=all_colors + ["orange"],
            title="",
            save_path="",
            linewidths=all_linewidths + [all_linewidths[0]],
            figsize=figsize,
            linestyles=all_linestyles + [all_linestyles[0]],
            x_values=budgets_list + n_flips_list,
            scatter_plot=False,
            plot_legend=False,
            zorders=zorders + [zorders[0]],
            show_plot=False,
        )
        plot_graphs(
            values_lists=[x[1:] for x in actual_flips_pos_list]
            + [[0]]
            + [n_flips_list[0][1:]],
            labels=labels_scatter + ["Flips Constraint"],
            xlabel=xlabel,
            ylabel="Positive Actual Number of Flips",
            colors=all_colors_scatter + ["orange"],
            title=(
                f"Actual Flips Across Positive Labels per Number of Flips{append_to_title}"
                if display_title
                else ""
            ),
            save_path=(
                f"{save_path}methods_{actual_flips_label}_pos.pdf" if save_path else ""
            ),
            linewidths=all_linewidths_scatter + [all_linewidths_scatter[0]],
            figsize=figsize,
            linestyles=all_linestyles_scatter + [all_linestyles_scatter[0]],
            x_values=x_values_scatter + [n_flips_list[0][1:]],
            scatter_plot=True,
            marker_sizes=markers_sz_scatter + [markers_sz_scatter[0]],
            scatter_markers=markers_scatter + ["v"],
            ax=ax_act_pos_flips,
            line_plot=False,
            plot_legend=True,
            zorders=zorders_scatter + [zorders[0]],
            show_plot=True,
        )


def plot_regions_norm_stats(
    methods_stats,
    methods_labels,
    xlabel,
    save_path="",
    figsize=(16, 8),
    append_to_title="",
    append_to_save="",
    display_title=True,
    method_to_display_name={},
    method_to_plot_info={},
):
    """
    Plots normalized statistics across different regions for various methods.

    This function generates a bar chart where the statistics of different methods
    are displayed per region.

    Args:
        methods_stats (list of lists): A list where each element is a list of statistics
            for a method, with values corresponding to different regions.
        methods_labels (list of str): Labels for each method to be displayed in the legend.
        xlabel (str): Label for the x-axis.
        save_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plot. Defaults to (16, 8).
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        method_to_display_name (dict, optional): Mapping of method names to their display names. Defaults to {}.
        method_to_plot_info (dict, optional): Dictionary with plotting attributes (e.g., colors) for each method. Defaults to {}.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=figsize)

    num_regions = len(methods_stats[0])
    num_methods = len(methods_stats)
    bar_width = 0.8 / num_methods

    x_indices = np.arange(num_regions)

    for idx, (method_stats, method_label) in enumerate(
        zip(methods_stats, methods_labels)
    ):
        method_stats = (
            np.concatenate(method_stats)
            if isinstance(method_stats[0], list)
            else method_stats
        )

        plot_info = method_to_plot_info.get(method_label, None)
        if plot_info is not None:
            color = plot_info["color"]
        else:
            color = "red"

        method_label = method_to_display_name.get(method_label, method_label)

        ax.bar(
            x_indices + idx * bar_width,
            method_stats,
            width=bar_width,
            label=method_label,
            color=color,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SBIr")

    if display_title:
        ax.set_title(f"SBI per Region {append_to_title}")

    ax.legend(
        framealpha=0.2,
        loc="best",
        labelspacing=0.2,
    )
    plt.xticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(
            f"{save_path}final_norm_stats{append_to_save}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()


def plot_score1_vs_score2(
    methods_to_res_info,
    score_label1,
    score_label2,
    score_display_label1,
    score_display_label2,
    init_score1,
    init_score2,
    method_to_plot_info,
    method_to_display_name,
    score_label3=None,
    score_display_label3=None,
    init_score3=None,
    save_plots_path="",
    default_linestyle=False,
    figsize=(10, 6),
    flips_limit=None,
    append_to_title="",
    append_to_save="",
    display_title=True,
    other_score1=None,
    other_score2=None,
    other_score_3=None,
    other_method=None,
    score_2_min_axis=None,
    score_2_max_axis=None,
):
    """
    Plots the comparison of multiple methods based on two or three score metrics across different budgets.

    Parameters:
    - methods_to_res_info (dict): Mapping of method names to DataFrames containing scores and budgets.
    - score_label1 (str): Column name of the first score.
    - score_label2 (str): Column name of the second score.
    - score_display_label1 (str): Display name for the first score.
    - score_display_label2 (str): Display name for the second score.
    - init_score1 (float): Initial value for the first score.
    - init_score2 (float): Initial value for the second score.
    - method_to_plot_info (dict): Plotting information (color, marker, linestyle, etc.) for each method.
    - method_to_display_name (dict): Mapping of method names to their display names.
    - score_label3 (str, optional): Column name of the third score (if applicable). Defaults to None.
    - score_display_label3 (str, optional): Display name for the third score. Defaults to None.
    - init_score3 (float, optional): Initial value for the third score. Defaults to None.
    - save_plots_path (str, optional): Path to save the generated plots. Defaults to "".
    - default_linestyle (bool, optional): Whether to use default linestyle for all plots. Defaults to False.
    - figsize (tuple, optional): Figure size. Defaults to (10, 6).
    - flips_limit (int, optional): Number of Flips Allowed (B) allowed. Defaults to None.
    - append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
    - append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
    - display_title (bool, optional): Whether to display the plot title. Defaults to True.
    - other_score1 (float, optional): Additional comparison score for the first metric. Defaults to None.
    - other_score2 (float, optional): Additional comparison score for the second metric. Defaults to None.
    - other_score_3 (float, optional): Additional comparison score for the third metric (if applicable). Defaults to None.
    - other_method (str, optional): Additional comparison method. Defaults to None.
    - score_2_min_axis (float, optional): Minimum value for the second score axis. Defaults to None.
    - score_2_max_axis (float, optional): Maximum value for the second score axis. Defaults to None.

    Returns:
    - None
    """

    if not display_title:
        figsize = figsize
        for m_info in method_to_plot_info.values():
            m_info["linewidth"] = 2.5
            m_info["marker_size"] = 50
    hspace = 0.05
    if not score_label3:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"hspace": hspace},
        )
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"hspace": hspace},
        )

    methods_scores1 = []
    methods_scores2 = []
    methods_scores3 = []
    labels = []
    budget_list = []
    colors = []
    linewidths = []
    linestyles = []
    marker_sizes = []
    markers = []

    for method, method_res_info_df in methods_to_res_info.items():
        labels.append(method_to_display_name[method])
        df_cp = method_res_info_df.copy()

        if flips_limit:
            df_cp = df_cp[df_cp["budget"] <= flips_limit]

        method_scores1 = [init_score1] + df_cp[score_label1].tolist()
        method_scores2 = [init_score2] + df_cp[score_label2].tolist()
        if score_label3:
            method_scores3 = [init_score3] + df_cp[score_label3].tolist()
            methods_scores3.append(method_scores3)

        methods_scores1.append(method_scores1)
        methods_scores2.append(method_scores2)

        method_budget = [0] + df_cp["budget"].tolist()
        budget_list.append(method_budget)

        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
        marker_sizes.append(method_to_plot_info[method]["marker_size"])
        markers.append(method_to_plot_info[method]["scatter_marker"])

    if other_score1 is not None and other_score2 is not None:
        labels.append(method_to_display_name[other_method])
        methods_scores1.append([init_score1, other_score1])
        methods_scores2.append([init_score2, other_score2])
        budget_list.append([0, max(budget_list[0])])
        colors.append(method_to_plot_info[other_method]["color"])
        linewidths.append(method_to_plot_info[other_method]["linewidth"])
        linestyles.append(method_to_plot_info[other_method]["linestyle"])
        marker_sizes.append(method_to_plot_info[other_method]["marker_size"])
        markers.append(method_to_plot_info[other_method]["scatter_marker"])

        if other_score_3 is not None:
            methods_scores3.append([init_score3, other_score_3])

    labels.append(method_to_display_name["init"])
    methods_scores1.append([init_score1])
    methods_scores2.append([init_score2])
    budget_list.append([0])

    colors.append(method_to_plot_info["init"]["color"])
    linewidths.append(method_to_plot_info["init"]["linewidth"])
    linestyles.append(method_to_plot_info["init"]["linestyle"])
    marker_sizes.append(method_to_plot_info["init"]["marker_size"])
    markers.append(method_to_plot_info["init"]["scatter_marker"])
    if score_label3:
        methods_scores3.append([init_score3])

    if default_linestyle:
        linestyles = ["-"] * len(labels)

    for i, label in enumerate(labels):
        ax1.plot(
            budget_list[i],
            methods_scores1[i],
            label=label,
            color=colors[i],
            linewidth=linewidths[i],
            linestyle="-",
        )

        ax2.plot(
            budget_list[i],
            methods_scores2[i] if not score_label3 else methods_scores3[i],
            label=label,
            color=colors[i],
            linewidth=linewidths[i],
            linestyle="-",
        )
        if score_label3:
            ax3.plot(
                budget_list[i],
                methods_scores2[i],
                label=label,
                color=colors[i],
                linewidth=linewidths[i],
                linestyle="-",
            )

    for i, label in enumerate(labels):
        scatter_s_idx = 1 if i != len(labels) - 1 else 0
        ax1.scatter(
            budget_list[i][scatter_s_idx:],
            methods_scores1[i][scatter_s_idx:],
            color=colors[i],
            s=marker_sizes[i],
            marker=markers[i],
            zorder=3 if i == len(labels) - 1 else 2,
        )

        ax2.scatter(
            budget_list[i][scatter_s_idx:],
            (
                methods_scores2[i][scatter_s_idx:]
                if not score_label3
                else methods_scores3[i][scatter_s_idx:]
            ),
            color=colors[i],
            s=marker_sizes[i],
            marker=markers[i],
            zorder=3 if i == len(labels) - 1 else 2,
        )

        if score_label3:
            ax3.scatter(
                budget_list[i][scatter_s_idx:],
                methods_scores2[i][scatter_s_idx:],
                color=colors[i],
                s=marker_sizes[i],
                marker=markers[i],
                zorder=3 if i == len(labels) - 1 else 2,
            )

    ax1.set_ylabel(score_display_label1)
    (
        ax2.set_ylabel(score_display_label2)
        if not score_label3
        else ax2.set_ylabel(score_display_label3)
    )
    if score_label3:
        ax3.set_ylabel(score_display_label2)
        ax3.set_xlabel("Number of Flips Allowed (B)")
    else:
        ax2.set_xlabel("Number of Flips Allowed (B)")

    if (score_2_min_axis is not None) and (score_2_max_axis is not None):
        (
            ax2.set_ylim(score_2_min_axis, score_2_max_axis)
            if not score_label3
            else ax3.set_ylim(score_2_min_axis, score_2_max_axis)
        )

    if score_label3:
        title = f"Strategies {score_display_label1}, {score_display_label2}, {score_display_label3} comparison{append_to_title}"
    else:
        title = f"Strategies {score_display_label1} vs {score_display_label2} comparison{append_to_title}"

    if display_title:
        fig.suptitle(title)

    unique_label_color_marker = list(zip(labels, colors, markers))
    method_legend_handles = [
        plt.Line2D(
            [0], [0], marker=m, color="w", markerfacecolor=c, markersize=10, label=l
        )
        for (l, c, m) in sorted(unique_label_color_marker, key=lambda x: x[0])
    ]

    label_spacing = 0.2
    handle_marker_spacing = 0.1
    legend_opacity = 0.5

    if score_label3:
        ax3.legend(
            handles=method_legend_handles,
            loc="best",
            labelspacing=label_spacing,
            handletextpad=handle_marker_spacing,
            framealpha=legend_opacity,
        )
    else:
        ax2.legend(
            handles=method_legend_handles,
            loc="best",
            labelspacing=label_spacing,
            handletextpad=handle_marker_spacing,
            framealpha=legend_opacity,
        )

    if save_plots_path:
        if score_label3:
            save_path = f"{save_plots_path}methods_{score_label1}_vs_{score_label2}_vs_{score_label3}{append_to_save}.pdf"
        else:
            save_path = f"{save_plots_path}methods_{score_label1}_vs_{score_label2}{append_to_save}.pdf"
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_min_C_reach_limit(
    meths_min_C_reach_limit,
    method_to_plot_info,
    method_to_display_name,
    opt_methods_display_labels,
    figsize=(20, 8),
    save_path="",
    display_title=True,
):
    """
    Plots the minimum budget constraint required to reach a limit for different methods.

    This function generates a horizontal bar chart showing the minimum budget constraint
    needed for each method to reach a predefined limit. Methods are sorted in descending order
    based on their budget constraints.

    Args:
        meths_min_C_reach_limit (dict): Dictionary mapping method names to their corresponding
            minimum budget constraint needed to reach the limit.
        method_to_plot_info (dict): Dictionary containing plot attributes for each method
            (e.g., color, linewidth, linestyle).
        method_to_display_name (dict): Mapping of method names to display names.
        opt_methods_display_labels (list): List of method labels to be highlighted in bold.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        save_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        display_title (bool, optional): Whether to display the plot title. Defaults to True.

    Returns:
        None
    """

    labels = []
    colors = []
    linewidths = []
    linestyles = []
    min_C_reach_limit_list = []

    for method, min_C_reach_limit in meths_min_C_reach_limit.items():
        labels.append(method_to_display_name[method])
        min_C_reach_limit_list.append(min_C_reach_limit)
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])

    labels_min_C_reach_limit_colors = [
        (label, val, color)
        for label, val, color in zip(labels, min_C_reach_limit_list, colors)
    ]
    sorted_labels_min_C_reach_limit_colors = sorted(
        labels_min_C_reach_limit_colors, key=lambda x: x[1], reverse=True
    )
    sorted_min_C_reach_limit_labels = [
        x[0] for x in sorted_labels_min_C_reach_limit_colors
    ]
    sorted_min_C_reach_limit = [x[1] for x in sorted_labels_min_C_reach_limit_colors]
    sorted_min_C_reach_limit_colors = [
        x[2] for x in sorted_labels_min_C_reach_limit_colors
    ]

    fig, ax = plt.subplots(figsize=figsize)
    index = range(len(sorted_min_C_reach_limit_labels))
    for i, (method, cnt, color) in enumerate(sorted_labels_min_C_reach_limit_colors):
        ax.barh(index[i], cnt, color=color)
        ax.text(
            cnt * 1.01,
            index[i],
            f"{cnt}",
            va="center",
            ha="left",
            fontsize=10,
            color="black",
        )

    bars = ax.barh(
        index, sorted_min_C_reach_limit, color=sorted_min_C_reach_limit_colors
    )

    ax.set_yticks(index)
    ax.set_yticklabels(sorted_min_C_reach_limit_labels)

    for tick in ax.get_yticklabels():
        if tick.get_text() in opt_methods_display_labels:
            tick.set_fontweight("bold")

    xlabel = f"Min Budget Constraint"
    ylabel = "Method"
    title = f"Min Budget Constraint To Reach Limit per Method"

    if display_title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_yticks(index)
    ax.set_yticklabels(sorted_min_C_reach_limit_labels)

    if save_path:
        save_path = f"{save_path}min_budget_reach_limit.pdf"
        plt.savefig(save_path, format="pdf")
    plt.show()


def plot_thresholds_adjustments(
    thresholds,
    region_sizes,
    save_path="",
    figsize=(10, 6),
    display_title=True,
    format="svg",
    title="Classification Thresholds per Region",
):
    """
    Plots classification threshold adjustments for different regions.

    This function visualizes classification thresholds across various regions using
    a bar chart. The bars are color-coded based on region sizes using a colormap,
    helping to indicate relative fairness levels.

    Args:
        thresholds (list of float): A list of classification thresholds for each region.
        region_sizes (list of float): A list of region sizes used to determine the bar colors.
        save_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plot. Defaults to (10, 6).
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        format (str, optional): File format for saving the plot (e.g., "svg", "png"). Defaults to "svg".
        title (str, optional): Title of the plot. Defaults to "Classification Thresholds per Region".
    """

    fig, ax = plt.subplots(figsize=figsize)
    x_indices = np.arange(len(thresholds))
    bar_width = 0.6
    cmap = cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    for i, (thresh, size) in enumerate(zip(thresholds, region_sizes)):
        bar_color = cmap(norm(size))
        ax.bar(
            x_indices[i],
            thresh,
            width=bar_width,
            color=bar_color,
            edgecolor="black",
            alpha=0.85,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Classification Threshold", fontsize=14)
    ax.set_xlabel("Regions", fontsize=14)
    if display_title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm, ax=ax, aspect=60, pad=0.08, fraction=0.05, orientation="horizontal"
    )  # 👈 Increase `fraction`
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.set_xticklabels(["Unfavored (-1)", "Fair (0)", "Favored (1)"], fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.{format}", format=format, dpi=300)

    plt.show()


def plot_fairness_loss_per_partitioning(
    ids,
    init_fairness_loss_list,
    where_fairness_loss_list,
    init_fairness_loss_list_weighted,
    where_fairness_loss_list_weighted,
    save_plots_path,
    display_title=True,
    set_label="Test",
):
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))

    ids_str = [str(i) for i in ids]

    # Non-weighted Fairness Loss
    axes[0].plot(ids_str, init_fairness_loss_list, label="base")
    axes[0].scatter(ids_str, init_fairness_loss_list)
    axes[0].plot(
        ids_str,
        where_fairness_loss_list,
        label="FairWhere",
        linestyle="dashed",
    )
    axes[0].scatter(ids_str, where_fairness_loss_list)
    axes[0].set_xlabel("Partitioning Id")
    axes[0].set_ylabel("Fairness Loss")
    if display_title:
        axes[0].set_title(f"Fairness Loss per Partitioning ({set_label})")
    axes[0].legend()

    # Weighted Fairness Loss
    axes[1].plot(ids_str, init_fairness_loss_list_weighted, label="base")
    axes[1].scatter(ids_str, init_fairness_loss_list_weighted)
    axes[1].plot(
        ids_str,
        where_fairness_loss_list_weighted,
        label="FairWhere",
        linestyle="dashed",
    )
    axes[1].scatter(ids_str, where_fairness_loss_list_weighted)
    axes[1].set_xlabel("Partitioning Id")
    axes[1].set_ylabel("Weighted Fairness Loss")
    if display_title:
        axes[1].set_title(f"Weighted Fairness Loss per Partitioning ({set_label})")
    axes[1].legend()

    plt.tight_layout()

    if save_plots_path:
        plt.savefig(
            f"{save_plots_path}fairness_loss_per_partitioning.pdf", format="pdf"
        )
    plt.show()
