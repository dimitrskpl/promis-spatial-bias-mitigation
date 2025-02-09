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
        tiles="Cartodb Positron",  # "StamenToner",
        # attr="Stamen Toner",
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
                locations=[
                    (lat, lon) for lon, lat in polygon
                ],  # note order: (lat, lon)
                color="white",
                fill=True,
                fill_opacity=0.9,
                fill_color=hex_color,
                weight=1,
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
        tiles="Cartodb Positron",
    )
    # Add points
    if y_pred is not None:
        indices = df.index
        shuffled_indices = np.random.permutation(indices)
        for index in shuffled_indices:
            color = "#00FF00" if y_pred[index] == 1 else "#FF0000"
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
                        color=regs_color,
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
    bold_labels=[],
    x_sticks_step=None,
    scatter_plot=False,
    rev_xaxis=False,
    scatter_markers=None,
    axhline=None,
    axhline_label=None,
    annotate_points=None,
    marker_sizes=None,
):
    """
    Plots line or scatter graphs for multiple data series with customizable styles.

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

    Raises:
        ValueError: If the length of `values_lists`, `colors`, and `labels` are not the same.

    Functionality:
        - Plots multiple data series with customizable line styles, widths, and markers.
        - Supports both line and scatter plots.
        - Allows setting x-axis tick steps and reversing the x-axis.
        - Provides an option to add a horizontal reference line.
        - Saves the figure if `save_path` is provided.

    """

    if len(values_lists) != len(colors) or len(values_lists) != len(labels):
        raise ValueError(
            "values_lists size should be same as colors size and labels size"
        )

    fig, ax = plt.subplots(figsize=figsize)

    if linewidths is None:
        linewidths = [3] * len(colors)

    if linestyles is None:
        linestyles = ["-"] * len(colors)

    if markers is None:
        markers = [""] * len(colors)

    if marker_sizes is None:
        marker_sizes = [150] * len(colors)

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
            )

        if x_sticks_step:
            min_x = min(min(x) for x in x_values)  # Get the smallest x-value
            max_x = max(max(x) for x in x_values)  # Get the largest x-value
            ax.set_xticks(np.arange(min_x, max_x + 1, x_sticks_step))

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
                )
    else:
        for i in range(len(values_lists)):
            ax.plot(
                values_lists[i],
                label=labels[i],
                color=colors[i],
                linewidth=linewidths[i],
                linestyle=linestyles[i],
                marker=markers[i],
            )

    if annotate_points:
        for annotation in annotate_points:
            series_idx = annotation["series_idx"]
            point_idx = annotation["point_idx"]

            # Retrieve coordinates
            x_coord = x_values[series_idx][point_idx] if x_values else point_idx
            y_coord = values_lists[series_idx][point_idx]

            # Custom text or coordinates as default
            text = annotation.get("text", f"({x_coord}, {y_coord})")

            # Annotate the point
            ax.annotate(
                text,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(0, 50),  # Offset the annotation slightly above the point
                ha="center",
                fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1),
            )

    if axhline is not None and axhline_label is not None:
        ax.axhline(
            y=axhline,
            color="red",
            linestyle="--",
            linewidth=linewidths[0],
            label=f"{axhline_label}",
        )

    # legend = ax.legend(loc="best", handlelength=1.2, handletextpad=1.5)
    # legend = ax.legend(loc="best", handlelength=3, handletextpad=2)
    legend = ax.legend(loc="best", markerscale=1.2, handlelength=2.5, handletextpad=1.5)

    for text in legend.get_texts():
        if text.get_text() in bold_labels:
            text.set_fontweight("bold")  # Set the font weight to bold

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    if rev_xaxis:
        plt.gca().invert_xaxis()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    plt.show()


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

        plt.xlabel("Number of Flips")
        plt.yticks([])

        plt.legend()

        if save_path:
            plt.savefig(f"{save_path}opt_status.pdf", format="pdf")

        plt.show()


def plot_scores(
    methods_to_res_info,
    init_mlr,
    method_to_plot_info,
    method_to_display_name,
    opt_methods_display_labels,
    save_plots_path="",
    exp_idx=None,
    default_linestyle=False,
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    predefined_colors=[],
    append_to_title="",
    append_to_save="",
    score_label="mlr",
    display_title=True,
    axhline_mlr=None,
    axhline_mlr_label=None,
):
    """
    Plots the M.L.R. (Modified Likelihood Ratio) drop comparison across different methods.

    Args:
        methods_to_res_info (dict): Dictionary where keys are method names and values are Pandas DataFrames
            containing budgeted solutions and their respective scores.
        init_mlr (float): The initial M.L.R. score before any flips.
        method_to_plot_info (dict): Dictionary where keys are method names and values contain plotting information
            such as color, linewidth, and linestyle.
        method_to_display_name (dict): Dictionary mapping method names to their display names in the legend.
        opt_methods_display_labels (list of str): List of method labels that should be bolded in the legend.
        save_plots_path (str, optional): Directory path to save the plot. Defaults to "" (not saved).
        exp_idx (int, optional): Experiment index to include in the title and filename. Defaults to None.
        default_linestyle (bool, optional): If True, sets all line styles to solid (`"-"`). Defaults to False.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of flips to consider when plotting. Defaults to None (no limit).
        optim_sols_only (bool, optional): If True, only considers solutions where the optimization status is 1
            (indicating an optimal solution). Defaults to False.
        predefined_colors (list, optional): List of predefined colors to use for the methods. Defaults to an empty list.
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        score_label (str, optional): Column name in the DataFrame that represents the score to plot. Defaults to "mlr".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        axhline_mlr (float, optional): A horizontal reference line at this M.L.R. value. Defaults to None.
        axhline_mlr_label (str, optional): Label for the horizontal reference line. Defaults to None.

    Raises:
        ValueError: If the number of predefined colors is less than the number of labels.

    Functionality:
        - Extracts M.L.R. values for each method from `methods_to_res_info`.
        - Filters based on optimization status and flip budget constraints.
        - Generates a line plot comparing M.L.R. drop across different methods.
        - Allows for bold labeling of specific optimization methods.
        - Saves the plot if `save_plots_path` is provided.

    """

    methods_mlrs = []

    labels = []
    budget_list = []
    colors = []
    linewidths = []
    linestyles = []

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
        method_mlrs = [init_mlr] + method_res_info_df_cp[score_label].to_list()
        methods_mlrs.append(method_mlrs)
        method_budget = [0] + method_res_info_df_cp["budget"].to_list()
        budget_list.append(method_budget)
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
    if default_linestyle:
        linestyles = ["-"] * (len(labels))

    if predefined_colors:
        if len(labels) > len(predefined_colors):
            raise ValueError("len(labels) > len(predefined_colors)")
        colors = predefined_colors[: len(labels)]

    values_lists = methods_mlrs

    title = f"Strategies M.L.R drop comparison{append_to_title}"
    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"
    xlabel = "Number of flips"
    ylabel = "M.L.R."
    mlr_save_path = ""
    if save_plots_path:
        mlr_save_path = f"{save_plots_path}methods_mlr{append_to_save}.pdf"

        if exp_idx is not None:
            mlr_save_path = (
                f"{save_plots_path}methods_mlr{append_to_save}_exp_idx_{exp_idx}.pdf"
            )

    bold_labels = [label for label in labels if label in opt_methods_display_labels]
    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        title=title if display_title else "",
        save_path=mlr_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        x_values=budget_list,
        linestyles=linestyles,
        scatter_plot=True,
        axhline=axhline_mlr,
        axhline_label=axhline_mlr_label,
    )


def plot_multi_scores(
    method_name,
    res_df,
    init_mlrs,
    score_labels,
    opt_methods_display_labels,
    method_to_display_name,
    colors_list,
    save_plots_path="",
    exp_idx=None,
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    predefined_colors=[],
    display_title=True,
    append_to_save="",
):
    """
    Plots multiple M.L.R. scores for a given method across different sets.

    Args:
        method_name (str): The name of the method being plotted.
        res_df (pd.DataFrame): A DataFrame containing budgeted solutions and their respective scores.
        init_mlrs (list of float): Initial M.L.R. values before any flips for each score label.
        score_labels (list of str): List of score labels (columns in `res_df`) to plot.
        opt_methods_display_labels (list of str): List of method labels that should be bolded in the legend.
        method_to_display_name (dict): Dictionary mapping method names to their display names in the legend.
        colors_list (list of str): List of colors for each score label.
        save_plots_path (str, optional): Directory path to save the plot. Defaults to "" (not saved).
        exp_idx (int, optional): Experiment index to include in the title and filename. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of flips to consider when plotting. Defaults to None (no limit).
        optim_sols_only (bool, optional): If True, only considers solutions where the optimization status is 1
            (indicating an optimal solution). Defaults to False.
        predefined_colors (list, optional): List of predefined colors to use for the methods. Defaults to an empty list.
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".

    Raises:
        ValueError: If the number of predefined colors is less than the number of labels.

    Functionality:
        - Extracts M.L.R. values for each score label from `res_df`.
        - Filters based on optimization status and flip budget constraints.
        - Generates a line plot comparing M.L.R. drop across different datasets for the specified method.
        - Allows for bold labeling of specific optimization methods.
        - Saves the plot if `save_plots_path` is provided.
    """

    assert len(init_mlrs) == len(score_labels)
    methods_mlrs = []

    score_label_2_disp_name = {
        "mlr_eq_opp_sol": "Solution",
        "mlr_eq_opp_val": "Val Set",
        "mlr_eq_opp_test": "Test Set",
        "mlr_st_par_sol": "Solution",
        "mlr_st_par_val": "Val Set",
        "mlr_st_par_test": "Test Set",
    }

    labels = [
        f"{method_to_display_name[method_name]} - {score_label_2_disp_name[score_label]}"
        for score_label in score_labels
    ]
    budget_list = []
    colors = []

    method_res_info_df_cp = res_df.copy()
    if optim_sols_only and "status" in method_res_info_df_cp.columns:
        method_res_info_df_cp = method_res_info_df_cp[
            method_res_info_df_cp["status"] == 1
        ]
    if flips_limit:
        method_res_info_df_cp = method_res_info_df_cp[
            method_res_info_df_cp["budget"] <= flips_limit
        ]

    methods_mlrs = [
        [init_mlrs[i]] + method_res_info_df_cp[score_labels[i]].to_list()
        for i in range(len(score_labels))
    ]
    method_budget = [0] + method_res_info_df_cp["budget"].to_list()

    budget_list = [method_budget] * len(score_labels)
    colors = [colors_list[i] for i in range(len(score_labels))]

    if predefined_colors:
        if len(labels) > len(predefined_colors):
            raise ValueError("len(labels) > len(predefined_colors)")
        colors = predefined_colors[: len(labels)]

    values_lists = methods_mlrs

    title = f"Strategies M.L.R drop comparison For {method_name}"
    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"
    xlabel = "Number of flips"
    ylabel = "M.L.R."
    mlr_save_path = ""
    if save_plots_path:
        mlr_save_path = f"{save_plots_path}methods_mlr_{method_name}.pdf"

        if exp_idx is not None:
            mlr_save_path = f"{save_plots_path}methods_mlr{append_to_save}_exp_idx_{exp_idx}_{method_name}.pdf"

    bold_labels = [label for label in labels if label in opt_methods_display_labels]
    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        title=title if display_title else "",
        save_path=mlr_save_path,
        figsize=figsize,
        bold_labels=bold_labels,
        x_values=budget_list,
        scatter_plot=True,
        linewidths=[8] * len(labels),
    )


def plot_all_methods_multi_scores(
    all_methods_to_results_info,
    init_mlrs,
    score_labels,
    opt_methods_display_labels,
    method_to_display_name,
    colors_list,
    save_plots_path="",
    exp_idx=None,
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    predefined_colors=[],
    display_title=True,
):
    """
    Plots multiple M.L.R. scores for all methods present in the results dictionary.

    Args:
        all_methods_to_results_info (dict): A dictionary where keys are method names and values are DataFrames
            containing budgeted solutions and their respective scores.
        init_mlrs (list of float): Initial M.L.R. values before any flips for each score label.
        score_labels (list of str): List of score labels (columns in the DataFrame) to plot.
        opt_methods_display_labels (list of str): List of method labels that should be bolded in the legend.
        method_to_display_name (dict): Dictionary mapping method names to their display names in the legend.
        colors_list (list of str): List of colors to use for each score label.
        save_plots_path (str, optional): Directory path to save the plots. Defaults to "" (not saved).
        exp_idx (int, optional): Experiment index to include in the title and filename. Defaults to None.
        figsize (tuple, optional): Figure size for the plots. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of flips to consider when plotting. Defaults to None (no limit).
        optim_sols_only (bool, optional): If True, only considers solutions where the optimization status is 1
            (indicating an optimal solution). Defaults to False.
        predefined_colors (list, optional): List of predefined colors to use for the methods. Defaults to an empty list.
        display_title (bool, optional): Whether to display the plot title. Defaults to True.

    Functionality:
        - Iterates over all methods in `all_methods_to_results_info`.
        - Calls `plot_multi_scores` for each method, passing the relevant parameters.
        - Generates plots for each method comparing M.L.R. drop across different sets.
        - Allows for filtering based on optimization status and flip budget constraints.
        - Supports saving plots to a specified directory.

    """

    for method, res_df in all_methods_to_results_info.items():
        plot_multi_scores(
            method,
            res_df,
            init_mlrs,
            score_labels=score_labels,
            opt_methods_display_labels=opt_methods_display_labels,
            method_to_display_name=method_to_display_name,
            colors_list=colors_list,
            save_plots_path=save_plots_path,
            exp_idx=exp_idx,
            figsize=figsize,
            flips_limit=flips_limit,
            optim_sols_only=optim_sols_only,
            predefined_colors=predefined_colors,
            display_title=display_title,
        )


def plot_flips_time(
    exp_methods_to_res_info,
    method_to_display_name,
    method_to_plot_info,
    opt_methods_display_labels,
    title_append="",
    save_append="",
    save_plots_path="",
    log_time=False,
    exp_idx=None,
    figsize=(20, 8),
    display_title=True,
    axhline_time=None,
    axhline_time_label=None,
):
    """
    Plots the execution time for different methods as a function of the number of flips.

    Args:
        exp_methods_to_res_info (dict): A dictionary where keys are method names and values are DataFrames
            containing execution times and budgets.
        method_to_display_name (dict): Mapping of method names to display names for the legend.
        method_to_plot_info (dict): Dictionary containing plot attributes (color, linewidth, linestyle) for each method.
        opt_methods_display_labels (list of str): List of method labels to highlight in bold in the legend.
        title_append (str, optional): Additional text to append to the plot title. Defaults to "".
        save_append (str, optional): Additional text to append to the save file name. Defaults to "".
        save_plots_path (str, optional): Path to save the generated plot. Defaults to "" (not saved).
        log_time (bool, optional): If True, plots the logarithm of execution time. Defaults to False.
        exp_idx (int, optional): Experiment index for display and saving purposes. Defaults to None.
        figsize (tuple, optional): Figure size of the plot. Defaults to (20, 8).
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        axhline_time (float, optional): Value for drawing a horizontal reference line in the plot. Defaults to None.
        axhline_time_label (str, optional): Label for the horizontal reference line. Defaults to None.

    Functionality:
        - Iterates through methods and extracts execution times and flip budgets.
        - Optionally applies a logarithmic scale to execution times.
        - Generates a plot where execution time is shown against the number of flips.
        - Highlights specific methods in bold.
        - Supports saving the plot in `.pdf` format if `save_plots_path` is provided.
        - Allows adding a reference line for execution time.
    """

    if axhline_time and log_time:
        axhline_time = np.log(axhline_time)
    labels = []
    exec_times = []
    n_flips_list = []
    colors = []
    linewidths = []
    linestyles = []
    markers_sizes = []
    for method, res_df in exp_methods_to_res_info.items():
        labels.append(method_to_display_name[method])
        if log_time:
            exec_times.append([np.log(t) for t in res_df["time"].to_list()])
        else:
            exec_times.append(res_df["time"].to_list())
        n_flips_list.append(res_df["budget"].tolist())
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
        markers_sizes.append(method_to_plot_info[method]["marker_size"])

    time_label = "Execution Time"
    time_save_label = "exec_time"
    if log_time:
        time_label = "Log Execution Time"
        time_save_label = "log_exec_time"

    if display_title:
        title = f"{time_label} per Number of Flips"
        if title_append:
            title = f"{title} {title_append}"
        if exp_idx is not None:
            title = f"{title} exp_idx: {exp_idx}"
    else:
        title = ""

    save_flips_time_path = ""
    if save_plots_path:
        save_flips_time_path = (
            f"{save_plots_path}{time_save_label}_per_flips{save_append}"
        )
        if exp_idx is not None:
            save_flips_time_path = f"{save_flips_time_path}_exp_idx_{exp_idx}"

        save_flips_time_path = f"{save_flips_time_path}.pdf"

    xlabel = "Number Of Flips"
    ylabel = f"{time_label} (seconds)"

    bold_labels = [label for label in labels if label in opt_methods_display_labels]
    plot_graphs(
        values_lists=exec_times,
        x_values=n_flips_list,
        labels=labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        save_path=save_flips_time_path,
        figsize=figsize,
        linewidths=linewidths,
        linestyles=linestyles,
        bold_labels=bold_labels,
        scatter_plot=True,
        axhline=axhline_time,
        axhline_label=axhline_time_label,
        marker_sizes=markers_sizes,
    )


def plot_mlr_vs_time(
    methods_to_res_info,
    mlr_label,
    method_to_display_name,
    method_to_plot_info,
    opt_methods_display_labels,
    save_plots_path="",
    exp_idx=None,
    figsize=(20, 8),
    log_time=False,
    append_to_title="",
    append_to_save="",
    display_title=True,
):
    """
    Plots the relationship between MLR (Mean Log Ratio) and execution time for multiple methods.

    This function generates two scatter plots:
    1. MLR vs Time (Execution time on the X-axis and MLR on the Y-axis)
    2. Time vs MLR (MLR on the X-axis and Execution time on the Y-axis, with reversed X-axis)

    Args:
        methods_to_res_info (dict): A dictionary mapping method names to DataFrames containing
            results including MLR and execution time.
        mlr_label (str): The column name representing MLR in the DataFrames.
        method_to_display_name (dict): A mapping of method names to their display names for legends.
        method_to_plot_info (dict): Dictionary containing visualization attributes (color, linewidth, scatter markers)
            for each method.
        opt_methods_display_labels (list of str): Labels of methods to be highlighted in bold in the legend.
        save_plots_path (str, optional): Path to save the generated plots. Defaults to "" (no saving).
        exp_idx (int, optional): Experiment index for labeling the plots. Defaults to None.
        figsize (tuple, optional): Figure size for the plots. Defaults to (20, 8).
        log_time (bool, optional): If True, the execution time will be plotted on a logarithmic scale. Defaults to False.
        append_to_title (str, optional): Additional text to append to the plot titles. Defaults to "".
        append_to_save (str, optional): Additional text to append to the filenames when saving plots. Defaults to "".
        display_title (bool, optional): Whether to display the title on the plots. Defaults to True.

    Functionality:
        - Extracts MLR and execution time for each method.
        - Optionally applies logarithm transformation to execution time.
        - Generates two scatter plots: one for MLR vs Time and another for Time vs MLR.
        - Saves plots to the specified path if provided.

    """

    methods_mlr = []
    methods_times = []
    labels = []
    scatter_markers = []
    colors = []
    linewidths = []

    time_label = ""
    if log_time:
        time_label = "log Time"

    for method, method_res_info_df in methods_to_res_info.items():
        labels.append(method)
        methods_mlr.append(method_res_info_df[mlr_label].to_list())
        method_times = method_res_info_df["time"].tolist()
        if log_time:
            method_times = [np.log(tm) for tm in method_times]
        methods_times.append(method_times)
        scatter_markers.append(method_to_plot_info[method]["scatter_marker"])
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])

    title = f"MLR vs {time_label}{append_to_title} "

    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"

    xlabel = f"{time_label} (s)"
    ylabel = "M.L.R."

    mlr_time_save_path = ""
    if save_plots_path:
        mlr_time_save_path = f"{save_plots_path}methods_mlr_vs_time{append_to_save}.pdf"
        if exp_idx is not None:
            mlr_time_save_path = f"{save_plots_path}methods_mlr_vs_time{append_to_save}_exp_idx_{exp_idx}.pdf"

    labels = [method_to_display_name[label] for label in labels]
    bold_labels = [label for label in labels if label in opt_methods_display_labels]

    plot_graphs(
        values_lists=methods_mlr,
        labels=labels,
        title=title,
        x_values=methods_times,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        save_path=mlr_time_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        scatter_plot=True,
        scatter_markers=scatter_markers,
    )

    xlabel = "MLR"
    ylabel = f"{time_label} (s)"
    title = f"{time_label} vs MLR{append_to_title} "

    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"

    time_mlr_save_path = ""
    if save_plots_path:
        time_mlr_save_path = f"{save_plots_path}methods_time_vs_mlr{append_to_save}.pdf"
        if exp_idx is not None:
            time_mlr_save_path = f"{save_plots_path}methods_time_vs_mlr{append_to_save}_exp_idx_{exp_idx}.pdf"
    title = title if display_title else ""
    plot_graphs(
        values_lists=methods_times,
        labels=labels,
        title=title,
        x_values=methods_mlr,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        save_path=time_mlr_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        scatter_plot=True,
        rev_xaxis=True,
        scatter_markers=scatter_markers,
    )


def plot_metrics(
    methods_to_res_info,
    score_label,
    ylabel,
    method_to_plot_info,
    method_to_display_name,
    opt_methods_display_labels,
    save_plots_path="",
    exp_idx=None,
    default_linestyle=False,
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    predefined_colors=[],
    append_to_title="",
    append_to_save="",
    display_title=True,
    axhline_score=None,
    axhline_score_label=None,
    init_score=None,
):
    """
    Plots a comparison of different methods' performance metrics (e.g., accuracy, fairness) over the number of flips.

    This function visualizes how different methods perform over an increasing number of label flips, using a line plot.
    It supports optional initial values, filtering based on solution optimality, and saving the plots.

    Args:
        methods_to_res_info (dict): A dictionary mapping method names to their result DataFrames.
        score_label (str): The column name in the DataFrame representing the metric to plot.
        ylabel (str): The label for the y-axis.
        method_to_plot_info (dict): A dictionary containing visualization attributes (color, linewidth, linestyle)
            for each method.
        method_to_display_name (dict): A mapping of method names to their display names.
        opt_methods_display_labels (list of str): Labels of methods to be highlighted in bold in the legend.
        save_plots_path (str, optional): Path to save the generated plot. Defaults to "" (no saving).
        exp_idx (int, optional): Experiment index for labeling the plots. Defaults to None.
        default_linestyle (bool, optional): If True, all lines will have a solid linestyle. Defaults to False.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): If set, filters the data to include only up to this number of flips. Defaults to None.
        optim_sols_only (bool, optional): If True, only solutions with status 1 (optimal) are considered. Defaults to False.
        predefined_colors (list, optional): List of predefined colors for the lines. Defaults to an empty list.
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the filename when saving the plot. Defaults to "".
        display_title (bool, optional): Whether to display the title on the plot. Defaults to True.
        axhline_score (float, optional): Horizontal line value for reference (e.g., initial accuracy). Defaults to None.
        axhline_score_label (str, optional): Label for the horizontal reference line. Defaults to None.
        init_score (float, optional): Initial value of the metric before any flips. Defaults to None.

    Functionality:
        - Extracts metric values from the result DataFrames.
        - Filters and processes the data based on constraints (e.g., `optim_sols_only`, `flips_limit`).
        - Plots metric evolution over the number of flips for different methods.
        - Allows optional horizontal reference lines and bold labels for optimization methods.
        - Saves the plot if `save_plots_path` is provided.
    """

    methods_accs = []

    labels = []
    budget_list = []
    colors = []
    linewidths = []
    linestyles = []

    init_budget = [0] if init_score is not None else []
    init_score = [init_score] if init_score is not None else []

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
        method_acc = method_res_info_df_cp[score_label].to_list()
        methods_accs.append(init_score + method_acc)
        method_budget = method_res_info_df_cp["budget"].to_list()
        budget_list.append(init_budget + method_budget)
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])

    if default_linestyle:
        linestyles = ["-"] * (len(labels))

    if predefined_colors:
        if len(labels) > len(predefined_colors):
            raise ValueError("len(labels) > len(predefined_colors)")
        colors = predefined_colors[: len(labels)]
    if display_title:
        title = f"Strategies {ylabel} per Number of Flips{append_to_title}"
        if exp_idx is not None:
            title = f"{title} exp_idx: {exp_idx}"
    else:
        title = ""

    xlabel = "Number of flips"
    acc_save_path = ""
    if save_plots_path:
        acc_save_path = f"{save_plots_path}methods_{score_label}{append_to_save}.pdf"

        if exp_idx is not None:
            acc_save_path = f"{save_plots_path}methods_{score_label}_{append_to_save}_exp_idx_{exp_idx}.pdf"

    bold_labels = [label for label in labels if label in opt_methods_display_labels]
    plot_graphs(
        methods_accs,
        labels,
        title,
        xlabel,
        ylabel,
        colors,
        save_path=acc_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        x_values=budget_list,
        linestyles=linestyles,
        scatter_plot=True,
        axhline=axhline_score,
        axhline_label=axhline_score_label,
    )


def plot_scores(
    methods_to_res_info,
    init_mlr,
    method_to_plot_info,
    method_to_display_name,
    opt_methods_display_labels,
    save_plots_path="",
    exp_idx=None,
    default_linestyle=False,
    figsize=(20, 8),
    flips_limit=None,
    optim_sols_only=False,
    predefined_colors=[],
    append_to_title="",
    append_to_save="",
    score_label="mlr",
    display_title=True,
    axhline_mlr=None,
    axhline_mlr_label=None,
):
    """
    Plots the MLR score drop comparison across different methods.

    This function visualizes the performance of different optimization strategies by plotting
    their MLR scores over varying budgets. It allows customization of line styles, colors,
    and other graphical elements.

    Args:
        methods_to_res_info (dict): A dictionary mapping method names to their corresponding
            results DataFrame, which includes performance metrics.
        init_mlr (float): The initial MLR value before any optimization.
        method_to_plot_info (dict): A dictionary containing plot attributes for each method
            (e.g., color, linewidth, linestyle, marker size).
        method_to_display_name (dict): A mapping from method names to their display labels.
        opt_methods_display_labels (list): A list of method labels that should be highlighted
            in bold on the plot.
        save_plots_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        exp_idx (int, optional): Experiment index to be appended to the title and filename. Defaults to None.
        default_linestyle (bool, optional): If True, overrides all linestyles with a default solid line. Defaults to False.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of budget flips to include in the plot. Defaults to None.
        optim_sols_only (bool, optional): If True, filters only optimal solutions (status=1) from the results. Defaults to False.
        predefined_colors (list, optional): A predefined list of colors for the plot lines. Defaults to an empty list.
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        score_label (str, optional): Column name in the DataFrame representing the score to plot. Defaults to "mlr".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        axhline_mlr (float, optional): A horizontal reference line at a specific MLR value. Defaults to None.
        axhline_mlr_label (str, optional): Label for the horizontal reference line. Defaults to None.

    Returns:
        None: The function generates and displays a plot but does not return a value.
    """

    methods_mlrs = []

    labels = []
    budget_list = []
    colors = []
    linewidths = []
    linestyles = []
    markers_sz = []

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
        method_mlrs = [init_mlr] + method_res_info_df_cp[score_label].to_list()
        methods_mlrs.append(method_mlrs)
        method_budget = [0] + method_res_info_df_cp["budget"].to_list()
        budget_list.append(method_budget)
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
        markers_sz.append(method_to_plot_info[method]["marker_size"])
    if default_linestyle:
        linestyles = ["-"] * (len(labels))

    if predefined_colors:
        if len(labels) > len(predefined_colors):
            raise ValueError("len(labels) > len(predefined_colors)")
        colors = predefined_colors[: len(labels)]

    values_lists = methods_mlrs

    title = f"Strategies MLR drop comparison{append_to_title}"
    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"
    xlabel = "Number of Flips"
    ylabel = "MLR"
    mlr_save_path = ""
    if save_plots_path:
        mlr_save_path = f"{save_plots_path}methods_mlr{append_to_save}.pdf"

        if exp_idx is not None:
            mlr_save_path = (
                f"{save_plots_path}methods_mlr{append_to_save}_exp_idx_{exp_idx}.pdf"
            )

    bold_labels = [label for label in labels if label in opt_methods_display_labels]
    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        title=title if display_title else "",
        save_path=mlr_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        x_values=budget_list,
        linestyles=linestyles,
        scatter_plot=True,
        axhline=axhline_mlr,
        axhline_label=axhline_mlr_label,
        marker_sizes=markers_sz,
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
    opt_methods_display_labels,
    save_path="",
    figsize=(20, 8),
    append_to_title="",
    exp_idx=None,
    display_title=True,
    axhline_P=None,
    axhline_RHO=None,
    axhline_label=None,
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
        opt_methods_display_labels (list): List of method labels that should be highlighted in bold.
        save_path (str, optional): Path to save the generated plots. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plots. Defaults to (20, 8).
        append_to_title (str, optional): Additional text to append to the plot titles. Defaults to "".
        exp_idx (int, optional): Experiment index to be included in the plot titles and filenames. Defaults to None.
        display_title (bool, optional): Whether to display plot titles. Defaults to True.
        axhline_P (float, optional): Horizontal reference line for the positive predictions plot. Defaults to None.
        axhline_RHO (float, optional): Horizontal reference line for the positive ratio of predictions plot. Defaults to None.
        axhline_label (str, optional): Label for the horizontal reference lines. Defaults to None.

    Returns:
        None: The function generates and displays multiple plots but does not return a value.
    """

    all_methods_labels = []
    p_list = []
    p_signif_list = []
    rho_list = []
    rho_signif_list = []
    actual_flips_list = []
    actual_flips_pos_list = []
    actual_flips_signif_list = []
    num_flips = []
    signif_num_flips = []
    all_linewidths = []
    all_colors = []
    all_linestyles = []
    markers_sz = []

    for method_name, res_info_df in all_methods_to_results_info.items():
        all_methods_labels.append(method_to_display_name[method_name])
        p_list.append([init_p] + res_info_df[p_label].tolist())
        rho_list.append([init_rho] + res_info_df[rho_label].tolist())
        actual_flips_list.append([0] + res_info_df[actual_flips_label].tolist())
        if f"{actual_flips_label}_pos" in res_info_df.columns:
            actual_flips_pos_list.append(
                [0] + res_info_df[f"{actual_flips_label}_pos"].tolist()
            )
        num_flips.append([0] + res_info_df["budget"].to_list())
        all_colors.append(method_to_plot_info[method_name]["color"])
        all_linewidths.append(method_to_plot_info[method_name]["linewidth"])
        all_linestyles.append(method_to_plot_info[method_name]["linestyle"])
        markers_sz.append(method_to_plot_info[method_name]["marker_size"])

    all_p_list = p_list + p_signif_list
    all_rho_list = rho_list + rho_signif_list
    all_actual_flips_list = actual_flips_list + actual_flips_signif_list
    flips_lists = num_flips + signif_num_flips

    values_lists = all_p_list
    labels = []
    bold_labels = []
    for i in range(len(all_methods_labels)):
        new_label = f"{all_methods_labels[i]}"
        labels.append(new_label)
        if all_methods_labels[i] in opt_methods_display_labels:
            bold_labels.append(new_label)

    if display_title:
        title = f"{p_label} per Number of Flips{append_to_title}"
        if exp_idx is not None:
            title = f"{title} exp_idx: {exp_idx}"
    else:
        title = ""

    xlabel = "Number of Flips"
    ylabel = p_label
    if save_path:
        p_save_path = f"{save_path}methods_{p_label}.pdf"
    else:
        p_save_path = ""

    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel="Positive Predictions",
        colors=all_colors,
        title=title,
        save_path=p_save_path,
        linewidths=all_linewidths,
        figsize=figsize,
        linestyles=all_linestyles,
        bold_labels=bold_labels,
        x_values=flips_lists,
        scatter_plot=True,
        axhline=axhline_P,
        axhline_label=f"{axhline_label}",
        marker_sizes=markers_sz,
    )

    values_lists = all_rho_list
    labels = []
    bold_labels = []
    for i in range(len(all_methods_labels)):
        new_label = f"{all_methods_labels[i]}"
        labels.append(new_label)
        if all_methods_labels[i] in opt_methods_display_labels:
            bold_labels.append(new_label)

    if display_title:
        title = f"{rho_label} per Number of Flips{append_to_title}"
        if exp_idx is not None:
            title = f"{title} exp_idx: {exp_idx}"
    else:
        title = ""

    xlabel = "Number of Flips"
    ylabel = "Positive Ratio of Predictions"
    if save_path:
        rho_save_path = f"{save_path}methods_{rho_label}.pdf"
    else:
        rho_save_path = ""

    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=all_colors,
        title=title,
        save_path=rho_save_path,
        linewidths=all_linewidths,
        figsize=figsize,
        linestyles=all_linestyles,
        bold_labels=bold_labels,
        x_values=flips_lists,
        scatter_plot=True,
        axhline=axhline_RHO,
        axhline_label=f"{axhline_label}",
        marker_sizes=markers_sz,
    )

    amethod_res_df = all_methods_to_results_info[
        list(all_methods_to_results_info.keys())[0]
    ]
    n_flips_list = [[0] + amethod_res_df["budget"].tolist()]

    values_lists = n_flips_list + all_actual_flips_list
    labels = ["Flips Constraint"]
    for i in range(len(all_methods_labels)):
        new_label = f"{all_methods_labels[i]}"
        labels.append(new_label)
        if all_methods_labels[i] in opt_methods_display_labels:
            bold_labels.append(new_label)

    if display_title:
        title = f"Actual Flips per Number of Flips{append_to_title}"
        if exp_idx is not None:
            title = f"{title} exp_idx: {exp_idx}"
    else:
        title = ""

    xlabel = "Number of Flips"
    ylabel = "Actual Number of Flips"
    if save_path:
        rho_save_path = f"{save_path}methods_{actual_flips_label}.pdf"
    else:
        rho_save_path = ""

    plot_graphs(
        values_lists=values_lists,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=["red"] + all_colors,
        title=title,
        save_path=rho_save_path,
        linewidths=[2] + all_linewidths,
        figsize=figsize,
        linestyles=["-"] + all_linestyles,
        bold_labels=bold_labels,
        x_values=n_flips_list + flips_lists,
        scatter_plot=True,
        marker_sizes=markers_sz + [markers_sz[0]],
    )

    ### Actual Flips Across Positive True Labels
    if actual_flips_pos_list:
        values_lists = n_flips_list + actual_flips_pos_list
        labels = ["Flips Constraint"]
        for i in range(len(all_methods_labels)):
            new_label = f"{all_methods_labels[i]}"
            labels.append(new_label)
            if all_methods_labels[i] in opt_methods_display_labels:
                bold_labels.append(new_label)

        if display_title:
            title = f"Actual Flips Across Positive Labels per Number of Flips{append_to_title}"
            if exp_idx is not None:
                title = f"{title} exp_idx: {exp_idx}"
        else:
            title = ""

        xlabel = "Number of Flips"
        ylabel = "Positive Actual Number of Flips"
        if save_path:
            rho_save_path = f"{save_path}methods_{actual_flips_label}_pos.pdf"
        else:
            rho_save_path = ""

        plot_graphs(
            values_lists=values_lists,
            labels=labels,
            xlabel=xlabel,
            ylabel=ylabel,
            colors=["red"] + all_colors,
            title=title,
            save_path=rho_save_path,
            linewidths=[2] + all_linewidths,
            figsize=figsize,
            linestyles=["-"] + all_linestyles,
            bold_labels=bold_labels,
            x_values=n_flips_list + flips_lists,
            scatter_plot=True,
            marker_sizes=markers_sz + [markers_sz[0]],
        )


def plot_regions_norm_stats(
    methods_stats,
    methods_labels,
    xlabel,
    ylabel,
    max_stat,
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
    are normalized by the maximum statistic value and displayed per region.

    Args:
        methods_stats (list of lists): A list where each element is a list of statistics
            for a method, with values corresponding to different regions.
        methods_labels (list of str): Labels for each method to be displayed in the legend.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        max_stat (float): The maximum statistic value used for normalization.
        save_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        figsize (tuple, optional): Figure size for the plot. Defaults to (16, 8).
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        method_to_display_name (dict, optional): Mapping of method names to their display names. Defaults to {}.
        method_to_plot_info (dict, optional): Dictionary with plotting attributes (e.g., colors) for each method. Defaults to {}.

    Returns:
        None: The function generates and displays a bar chart but does not return a value.
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

        normalized_stats = [stat / max_stat for stat in method_stats]
        plot_info = method_to_plot_info.get(method_label, None)
        if plot_info is not None:
            color = plot_info["color"]
        else:
            color = "red"

        method_label = method_to_display_name.get(method_label, method_label)

        ax.bar(
            x_indices + idx * bar_width,
            normalized_stats,
            width=bar_width,
            label=method_label,
            color=color,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if display_title:
        ax.set_title(f"Normalized {ylabel} Per Region {append_to_title}")

    ax.legend()
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
    opt_methods_display_labels,
    save_plots_path="",
    exp_idx=None,
    default_linestyle=False,
    figsize=(20, 8),
    flips_limit=None,
    predefined_colors=[],
    append_to_title="",
    append_to_save="",
    display_title=True,
    other_score1=None,
    other_score2=None,
    other_method_label=None,
):
    """
    Plots the comparison of two scores across different methods.

    This function visualizes the relationship between two scores (e.g., performance metrics)
    for multiple methods. The scores are plotted against each other, allowing for comparison.

    Args:
        methods_to_res_info (dict): Dictionary mapping method names to their results DataFrame
            containing performance metrics.
        score_label1 (str): Column name for the first score to be plotted on the x-axis.
        score_label2 (str): Column name for the second score to be plotted on the y-axis.
        score_display_label1 (str): Display label for the first score.
        score_display_label2 (str): Display label for the second score.
        init_score1 (float): Initial value of the first score before any optimization.
        init_score2 (float): Initial value of the second score before any optimization.
        method_to_plot_info (dict): Dictionary containing plot attributes for each method
            (e.g., color, linewidth, linestyle, marker size).
        method_to_display_name (dict): Mapping of method names to display names.
        opt_methods_display_labels (list): List of method labels to be highlighted in bold.
        save_plots_path (str, optional): Path to save the generated plot. Defaults to "" (no save).
        exp_idx (int, optional): Experiment index to be included in the plot title and filename. Defaults to None.
        default_linestyle (bool, optional): If True, overrides all linestyles with a default solid line. Defaults to False.
        figsize (tuple, optional): Figure size for the plot. Defaults to (20, 8).
        flips_limit (int, optional): Maximum number of budget flips to include in the plot. Defaults to None.
        predefined_colors (list, optional): A predefined list of colors for the plot lines. Defaults to an empty list.
        append_to_title (str, optional): Additional text to append to the plot title. Defaults to "".
        append_to_save (str, optional): Additional text to append to the save filename. Defaults to "".
        display_title (bool, optional): Whether to display the plot title. Defaults to True.
        other_score1 (float, optional): Alternative score value to be plotted for comparison. Defaults to None.
        other_score2 (float, optional): Alternative score value to be plotted for comparison. Defaults to None.
        other_method_label (str, optional): Label for the alternative method if `other_score1` and `other_score2` are provided. Defaults to None.
    """

    methods_scores1 = []
    methods_scores2 = []

    labels = []
    budget_list = []
    colors = []
    linewidths = []
    linestyles = []
    marker_sizes = []

    for method, method_res_info_df in methods_to_res_info.items():
        labels.append(method_to_display_name[method])
        method_res_info_df_cp = method_res_info_df.copy()
        if flips_limit:
            method_res_info_df_cp = method_res_info_df_cp[
                method_res_info_df_cp["budget"] <= flips_limit
            ]
        method_scores1 = [init_score1] + method_res_info_df_cp[score_label1].to_list()
        methods_scores1.append(method_scores1)
        method_scores2 = [init_score2] + method_res_info_df_cp[score_label2].to_list()
        methods_scores2.append(method_scores2)
        method_budget = [0] + method_res_info_df_cp["budget"].to_list()
        budget_list.append(method_budget)
        colors.append(method_to_plot_info[method]["color"])
        linewidths.append(method_to_plot_info[method]["linewidth"])
        linestyles.append(method_to_plot_info[method]["linestyle"])
        marker_sizes.append(method_to_plot_info[method]["marker_size"])

    if default_linestyle:
        linestyles = ["-"] * (len(labels))

    if other_score1 is not None and other_score2 is not None:
        labels.append(other_method_label)
        methods_scores1.append([init_score1, other_score1])
        methods_scores2.append([init_score2, other_score2])
        budget_list.append([0, max(budget_list[0])])
        colors.append("red")
        linewidths.append(linewidths[0])
        linestyles.append("--")
        marker_sizes.append(marker_sizes[0])

    if predefined_colors:
        if len(labels) > len(predefined_colors):
            raise ValueError("len(labels) > len(predefined_colors)")
        colors = predefined_colors[: len(labels)]

    title = f"Strategies {score_display_label1} vs {score_display_label2} comparison{append_to_title}"
    if exp_idx is not None:
        title = f"{title} exp_idx: {exp_idx}"

    bold_labels = [label for label in labels if label in opt_methods_display_labels]

    mlr_save_path = ""
    if save_plots_path:
        mlr_save_path = f"{save_plots_path}methods_{score_label1}_vs_{score_label2}{append_to_save}.pdf"

    plot_graphs(
        values_lists=methods_scores2,
        labels=labels,
        xlabel=score_display_label1,
        ylabel=score_display_label2,
        colors=colors,
        title=title if display_title else "",
        save_path=mlr_save_path,
        figsize=figsize,
        linewidths=linewidths,
        bold_labels=bold_labels,
        x_values=methods_scores1,
        linestyles=linestyles,
        scatter_plot=True,
        rev_xaxis=True,
        marker_sizes=marker_sizes,
    )


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
        None: The function generates and displays a horizontal bar chart but does not return a value.
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
