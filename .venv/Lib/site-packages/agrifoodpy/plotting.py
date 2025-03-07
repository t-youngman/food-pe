import matplotlib.pyplot as plt
import altair as alt

def plot_bars(food, show="Item", ax=None, colors=None, labels=None, **kwargs):

    if ax is None:
        f, ax = plt.subplots(**kwargs)

    # Make sure only FAOSTAT food elements are present
    input_elements = list(food.keys())
    plot_elements = ["production", "imports", "exports", "food"]

    for element in plot_elements:
        if element not in input_elements:
            elements.remove(element)

    for element in input_elements:
        if element not in plot_elements and element in FAOSTAT_elements:
            plot_elements.insert(-1, element)

    len_elements = len(plot_elements)

    # Define dimensions to sum over
    bar_dims = list(food.dims)
    if show in bar_dims:
        bar_dims.remove(show)
        size_show = food.sizes[show]
    else:
        size_show = 1

    # Make sure NaN and inf do not interfere
    food = food.fillna(0)
    food = food.where(np.isfinite(food), other=0)

    food_sum = food.sum(dim=bar_dims)

    # If colors are not defined, generate a list from the standard cycling
    if colors is None:
        colors = [f"C{ic}" for ic in range(size_show)]

    # If labels are not defined, generate a list from the dimensions
    if labels is None:
        labels = np.repeat("", len(colors))

    # Plot the production and imports first
    cumul = 0
    for ie, element in enumerate(["production", "imports"]):
        ax.hlines(ie, 0, cumul, color="k", alpha=0.2, linestyle="dashed", linewidth=0.5)
        if size_show == 1:
            ax.barh(ie, left = cumul, width=food_sum[element], color=colors[0])
            cumul +=food_sum[element]
        else:
            for ii, val in enumerate(food_sum[element]):
                ax.barh(ie, left = cumul, width=val, color=colors[ii], label=labels[ii])
                cumul += val

    # Then the rest of elements in reverse to keep dimension ordering
    cumul = 0
    for ie, element in enumerate(reversed(plot_elements[2:])):
        ax.hlines(len_elements-1 - ie, 0, cumul, color="k", alpha=0.2, linestyle="dashed", linewidth=0.5)
        if size_show == 1:
            ax.barh(len_elements-1 - ie, left = cumul, width=food_sum[element], color=colors[0])
            cumul +=food_sum[element]
        else:
            for ii, val in enumerate(food_sum[element]):
                ax.barh(len_elements-1 - ie, left = cumul, width=val, color=colors[ii], label=labels[ii])
                cumul += val

    # Plot decorations
    ax.set_yticks(np.arange(len_elements), labels=plot_elements)
    ax.tick_params(axis="x",direction="in", pad=-12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_ylim(len_elements+1,-1)

    # Unique labels
    if labels[0] != "":
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6)

    return ax

def plot_years(food, show="Item", ax=None, colors=None, label=None, **kwargs):

    # If no years are found in the dimensions, raise an exception
    sum_dims = list(food.dims)
    if "Year" not in sum_dims:
        raise TypeError("'Year' dimension not found in array data")

    # Define the cumsum and sum dimentions and check for one element dimensions
    sum_dims.remove("Year")
    if ax is None:
        f, ax = plt.subplots(1, **kwargs)

    if show in sum_dims:
        sum_dims.remove(show)
        size_cumsum = food.sizes[show]
        cumsum = food.cumsum(dim=show).transpose(show, ...)
    else:
        size_cumsum = 1
        cumsum = food

    # Collapse remaining dimensions
    cumsum = cumsum.sum(dim=sum_dims)
    years = food.Year.values

    # If colors are not defined, generate a list from the standard cycling
    if colors is None:
        colors = [f"C{ic}" for ic in range(size_cumsum)]

    # Plot
    if size_cumsum == 1:
        ax.fill_between(years, cumsum, color=colors[0], alpha=0.5)
        ax.plot(years, cumsum, color=colors[0], linewidth=0.5, label=label)
    else:
        for id in reversed(range(size_cumsum)):
            ax.fill_between(years, cumsum[id], color=colors[id], alpha=0.5)
            ax.plot(years, cumsum[id], color=colors[id], linewidth=0.5, label=label)

    ax.set_xlim(years.min(), years.max())

    return ax

def plot_years_altair(food, show="Item"):

    # If no years are found in the dimensions, raise an exception
    sum_dims = list(food.coords)
    if "Year" not in sum_dims:
        raise TypeError("'Year' dimension not found in array data")

    df = food.to_dataframe().reset_index().fillna(0)
    df = df.melt(id_vars=sum_dims, value_vars=food.name)

    selection = alt.selection_multi(fields=[show], bind='legend')

    c = alt.Chart(df).mark_area().encode(
            alt.X('Year:T'),
            alt.Y('sum(value):Q'),
            alt.Color(f'{show}:N', scale=alt.Scale(scheme='category20b')),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=f'{show}:N'
        ).add_selection(selection)

    return c
