from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "processed" / "CGWB_Groundwater_Quality_2024_Cleaned.csv"
DATA = pd.read_csv(DATA_PATH)

LIMITS = {
    "Nitrate (NO3)": ("NO3_mgL", 45, "mg/L"),
    "Fluoride (F)": ("F_mgL", 1.5, "mg/L"),
    "TDS": ("TDS_mgL", 2000, "mg/L"),
    "Iron (Fe)": ("Fe_mgL", 1.0, "mg/L"),
    "Uranium (U)": ("U_ppb", 30, "ppb"),
    "Arsenic (As)": ("As_ppb", 10, "ppb"),
}
PARAMETERS = list(LIMITS)
PARAMETER_COLUMNS = {
    "Nitrate (NO3)": "NO3",
    "Fluoride (F)": "F",
    "TDS": "TDS",
    "Iron (Fe)": "Fe",
    "Uranium (U)": "U",
    "Arsenic (As)": "As",
}
COLORS = {"Within limit": "#AAB7C4", "Above limit": "#C0392B"}
TEAL, NAVY, AMBER, RED = "#147483", "#182F4B", "#E0A030", "#C0392B"
CHART_NAMES = [
    "Exceedance rates",
    "State heatmap",
    "Measurement distribution",
    "State distributions",
    "Parameter vs EC",
    "Season comparison",
    "Station map small multiples",
    "Paired-station comparison",
    "Multiple exceedances",
    "Parameter correlations",
    "Piper water types",
    "Selected-parameter map",
    "State ranking",
]


def filtered_data(state, season):
    result = DATA
    if state != "All states":
        result = result[result["State"] == state]
    if season != "All seasons":
        result = result[result["Season"] == season]
    return result


def selected_parameters(parameter):
    return list(LIMITS.items()) if parameter == "All parameters" else [(parameter, LIMITS[parameter])]


def style_figure(figure, height=520):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 55, "r": 35, "t": 75, "b": 55},
        font={"family": "Arial, sans-serif", "size": 12},
        hoverlabel={"namelength": -1},
    )
    return figure


def build_figures(state, season, parameter):
    data = filtered_data(state, season)
    season_data = filtered_data(state, "All seasons")
    chosen = selected_parameters(parameter)
    figures = []

    rates = []
    for name, (column, limit, unit) in chosen:
        values = data[column].dropna()
        rates.append({
            "Parameter": name,
            "Above limit (%)": (values > limit).mean() * 100 if len(values) else 0,
            "Samples": len(values),
            "Limit": limit,
            "Unit": unit,
        })
    rate_frame = pd.DataFrame(rates).sort_values("Above limit (%)")
    figure = px.bar(
        rate_frame,
        x="Above limit (%)",
        y="Parameter",
        orientation="h",
        text="Above limit (%)",
        color="Above limit (%)",
        color_continuous_scale="YlOrRd",
        hover_data=["Samples", "Limit", "Unit"],
        title="1. Samples above drinking-water limits",
    )
    figure.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    figure.update_layout(coloraxis_showscale=False, xaxis_title="Samples above limit (%)")
    figures.append(style_figure(figure, 440))

    heat_columns = {}
    for name, (column, limit, _) in chosen:
        grouped = data.dropna(subset=[column]).groupby("State")[column]
        heat_columns[name] = grouped.apply(lambda values: (values > limit).mean() * 100)
    heat = pd.DataFrame(heat_columns).sort_index()
    figure = go.Figure(go.Heatmap(
        z=heat.values,
        x=heat.columns,
        y=heat.index,
        colorscale="YlOrRd",
        zmin=0,
        colorbar={"title": "Above limit (%)"},
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    figure.update_layout(title="2. Exceedance by state and parameter", xaxis_title="Parameter", yaxis_title="State")
    figures.append(style_figure(figure, max(430, len(heat) * 24 + 150)))

    hist_name, (hist_column, hist_limit, hist_unit) = (
        chosen[0] if len(chosen) == 1 else ("Nitrate (NO3)", LIMITS["Nitrate (NO3)"])
    )
    histogram_values = data[hist_column].dropna()
    if hist_name == "Nitrate (NO3)":
        histogram_values = histogram_values[histogram_values <= 300]
    figure = px.histogram(
        x=histogram_values,
        nbins=40,
        labels={"x": f"{hist_name} ({hist_unit})", "y": "Samples"},
        title=f"3. Distribution of {hist_name}",
    )
    figure.add_vline(x=hist_limit, line_dash="dash", line_color=RED,
                     annotation_text=f"Limit {hist_limit:g} {hist_unit}")
    figures.append(style_figure(figure, 440))

    box_name, (box_column, box_limit, box_unit) = (
        chosen[0] if len(chosen) == 1 else ("Fluoride (F)", LIMITS["Fluoride (F)"])
    )
    box_data = data.dropna(subset=[box_column]).copy()
    state_rates = box_data.groupby("State")[box_column].apply(lambda values: (values > box_limit).mean())
    top_states = state_rates.sort_values(ascending=False).head(8).index
    box_data = box_data[box_data["State"].isin(top_states) & (box_data[box_column] > 0)]
    figure = px.box(box_data, x="State", y=box_column, points="outliers", color_discrete_sequence=[TEAL],
                    title=f"4. {box_name} distribution in highest-exceedance states")
    figure.add_hline(y=box_limit, line_dash="dash", line_color=RED,
                     annotation_text=f"Limit {box_limit:g} {box_unit}")
    figure.update_yaxes(type="log", title=f"{box_name} ({box_unit}, log scale)")
    figure.update_xaxes(title="", tickangle=-25)
    figures.append(style_figure(figure, 500))

    scatter_name, (scatter_column, scatter_limit, scatter_unit) = (
        chosen[0] if len(chosen) == 1 else ("Nitrate (NO3)", LIMITS["Nitrate (NO3)"])
    )
    scatter_data = data.dropna(subset=["EC_uS_cm", scatter_column, "Station", "District", "State"])
    scatter_data = scatter_data[(scatter_data["EC_uS_cm"] > 0) & (scatter_data[scatter_column] > 0)]
    if len(scatter_data) > 5000:
        scatter_data = scatter_data.sample(5000, random_state=42)
    scatter_data = scatter_data.copy()
    scatter_data["Reading"] = np.where(scatter_data[scatter_column] > scatter_limit, "Above limit", "Within limit")
    rho = scatter_data["EC_uS_cm"].corr(scatter_data[scatter_column], method="spearman") if len(scatter_data) > 1 else np.nan
    figure = px.scatter(
        scatter_data,
        x="EC_uS_cm",
        y=scatter_column,
        color="Reading",
        color_discrete_map=COLORS,
        hover_data=["Station", "District", "State", "Season"],
        opacity=0.55,
        title=f"5. {scatter_name} vs electrical conductivity (Spearman rho = {rho:.2f})",
    )
    figure.update_xaxes(type="log", title="Electrical conductivity (µS/cm)")
    figure.update_yaxes(type="log", title=f"{scatter_name} ({scatter_unit})")
    figure.add_hline(y=scatter_limit, line_dash="dash", line_color=RED)
    figures.append(style_figure(figure))

    season_rows = []
    for name, (column, limit, _) in chosen:
        for season_name, group in season_data.groupby("Season"):
            values = group[column].dropna()
            season_rows.append({
                "Season": season_name,
                "Parameter": name,
                "Above limit (%)": (values > limit).mean() * 100 if len(values) else 0,
                "Samples": len(values),
            })
    season_frame = pd.DataFrame(season_rows)
    figure = px.line(season_frame, x="Season", y="Above limit (%)", color="Parameter", markers=True,
                     custom_data=["Samples"], title="6. Exceedance rates by season (both seasons retained)")
    figure.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.1f}%<br>Samples: %{customdata[0]}<extra></extra>")
    figures.append(style_figure(figure, 480))

    map_figure = make_subplots(rows=2, cols=3, subplot_titles=[name for name, _ in chosen],
                               horizontal_spacing=0.05, vertical_spacing=0.12)
    map_data = data.dropna(subset=["Latitude", "Longitude"])
    for index, (name, (column, limit, unit)) in enumerate(chosen):
        row, col = index // 3 + 1, index % 3 + 1
        measured = map_data.dropna(subset=[column]).copy()
        measured["Reading"] = np.where(measured[column] > limit, "Above limit", "Within limit")
        for reading in ["Within limit", "Above limit"]:
            points = measured[measured["Reading"] == reading]
            map_figure.add_trace(go.Scattergl(
                x=points["Longitude"], y=points["Latitude"], mode="markers", name=reading,
                legendgroup=reading, showlegend=index == 0,
                marker={"size": 4, "color": COLORS[reading], "opacity": 0.75},
                customdata=points[["Station", "District", "State", "Season", column]].values,
                hovertemplate=f"%{{customdata[0]}}<br>%{{customdata[1]}}, %{{customdata[2]}}<br>%{{customdata[3]}}<br>{name}: %{{customdata[4]:.2f}} {unit}<extra></extra>",
            ), row=row, col=col)
        map_figure.update_xaxes(range=[67, 98], visible=False, row=row, col=col)
        map_figure.update_yaxes(range=[5, 38], visible=False, row=row, col=col)
    map_figure.update_layout(title="7. Station exceedance maps by parameter", height=760,
                             template="plotly_white", margin={"l": 20, "r": 20, "t": 90, "b": 35},
                             legend={"orientation": "h", "y": -0.03})
    figures.append(map_figure)

    pair_source = season_data.dropna(subset=[box_column]).copy()
    station_key = ["State", "District", "Block", "Station"]
    if not pair_source.empty:
        paired_rows = pair_source.groupby(station_key, dropna=False)["Season"].transform("nunique") == 2
        pair_source = pair_source[paired_rows]
    if not pair_source.empty:
        paired_values = pair_source.groupby(station_key + ["Season"], dropna=False)[box_column].mean().unstack("Season").dropna()
    else:
        paired_values = pd.DataFrame(columns=["Pre-Monsoon", "Post-Monsoon"])
    pair_figure = go.Figure()
    paired_rates = []
    for season_name in ["Pre-Monsoon", "Post-Monsoon"]:
        if season_name in paired_values:
            paired_rates.append((season_name, (paired_values[season_name] > box_limit).mean() * 100))
    if len(paired_rates) == 2:
        pair_figure.add_trace(go.Scatter(x=[rate for _, rate in paired_rates], y=[box_name] * 2,
                                         mode="lines", line={"color": "#74808A", "width": 3},
                                         showlegend=False, hoverinfo="skip"))
        for season_name, rate in paired_rates:
            pair_figure.add_trace(go.Scatter(x=[rate], y=[box_name], mode="markers+text", name=season_name,
                                             text=[f"{rate:.1f}%"], textposition="middle right",
                                             hovertemplate=f"{season_name}: %{{x:.1f}}%<extra></extra>"))
    else:
        pair_figure.add_annotation(text="No stations have measurements in both seasons for these filters.",
                                   showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    pair_figure.update_layout(title=f"8. Same-station seasonal comparison: {box_name}",
                              xaxis_title="Stations above limit (%)", yaxis_title="")
    figures.append(style_figure(pair_figure, 440))

    flag_columns = []
    for name, (column, limit, _) in LIMITS.items():
        flag_column = f"Exceeds_{column}"
        if flag_column not in data:
            data = data.copy()
            data[flag_column] = np.where(data[column].isna(), np.nan, (data[column] > limit).astype(float))
        flag_columns.append(flag_column)
    complete = data[data[flag_columns].notna().all(axis=1)]
    counts = complete[flag_columns].sum(axis=1).astype(int).value_counts().sort_index()
    count_labels = [str(index) if index < 3 else "3+" for index in counts.index]
    count_frame = pd.DataFrame({"Exceedances": count_labels, "Samples": counts.values})
    count_frame = count_frame.groupby("Exceedances", sort=False, as_index=False)["Samples"].sum()
    pair_counts = {}
    for first, second in combinations(flag_columns, 2):
        pair_counts[f"{first.removeprefix('Exceeds_')} + {second.removeprefix('Exceeds_')}"] = int(
            ((complete[first] == 1) & (complete[second] == 1)).sum()
        )
    common_pairs = pd.Series(pair_counts).sort_values().tail(6)
    multi_figure = make_subplots(rows=1, cols=2, subplot_titles=["Limits exceeded per sample", "Most common pairs"])
    multi_figure.add_trace(go.Bar(x=count_frame["Exceedances"], y=count_frame["Samples"], marker_color=AMBER,
                                  name="Samples"), row=1, col=1)
    multi_figure.add_trace(go.Bar(x=common_pairs.values, y=common_pairs.index, orientation="h",
                                  marker_color=TEAL, name="Samples above both"), row=1, col=2)
    multi_figure.update_layout(title="9. Multiple exceedances per fully measured sample", showlegend=False)
    figures.append(style_figure(multi_figure, 520))

    correlation_columns = {
        "pH": "pH", "EC_uS_cm": "EC", "TDS_mgL": "TDS", "Cl_mgL": "Cl", "SO4_mgL": "SO4",
        "NO3_mgL": "NO3", "HCO3_mgL": "HCO3", "Ca_mgL": "Ca", "Mg_mgL": "Mg",
        "Na_mgL": "Na", "F_mgL": "F", "Fe_mgL": "Fe", "As_ppb": "As", "U_ppb": "U",
    }
    correlations = data[list(correlation_columns)].rename(columns=correlation_columns).corr(method="spearman")
    if parameter != "All parameters":
        correlations = correlations.loc[[PARAMETER_COLUMNS[parameter]], :]
    correlation_figure = go.Figure(go.Heatmap(
        z=correlations.values, x=correlations.columns, y=correlations.index, zmin=-1, zmax=1, zmid=0,
        colorscale="RdBu", text=np.round(correlations.values, 2), texttemplate="%{text}",
        colorbar={"title": "Spearman rho"}, hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    correlation_figure.update_layout(title="10. Spearman correlation between water-quality parameters")
    figures.append(style_figure(correlation_figure, 520 if parameter == "All parameters" else 300))

    ion_factors = {"Ca_mgL": 20.04, "Mg_mgL": 12.15, "Na_mgL": 22.99, "K_mgL": 39.10,
                   "HCO3_mgL": 61.02, "CO3_mgL": 30.00, "Cl_mgL": 35.45, "SO4_mgL": 48.03}
    piper = data.dropna(subset=list(ion_factors)).copy()
    for column, factor in ion_factors.items():
        piper[column] = piper[column] / factor
    piper = piper[(piper[list(ion_factors)] >= 0).all(axis=1)]
    cations = piper[["Ca_mgL", "Mg_mgL", "Na_mgL", "K_mgL"]].sum(axis=1)
    anions = piper[["HCO3_mgL", "CO3_mgL", "Cl_mgL", "SO4_mgL"]].sum(axis=1)
    valid = (cations > 0) & (anions > 0)
    piper, cations, anions = piper.loc[valid], cations.loc[valid], anions.loc[valid]
    calcium, magnesium = piper["Ca_mgL"] / cations, piper["Mg_mgL"] / cations
    sodium = (piper["Na_mgL"] + piper["K_mgL"]) / cations
    bicarbonate = (piper["HCO3_mgL"] + piper["CO3_mgL"]) / anions
    chloride, sulphate = piper["Cl_mgL"] / anions, piper["SO4_mgL"] / anions
    water_types = np.select(
        [(calcium + magnesium >= 0.5) & (bicarbonate >= 0.5),
         (calcium + magnesium >= 0.5) & (bicarbonate < 0.5),
         (calcium + magnesium < 0.5) & (bicarbonate < 0.5)],
        ["Ca-Mg-HCO3", "Ca-Mg-Cl-SO4", "Na-Cl-SO4"], default="Na-HCO3",
    )
    piper_colors = {"Ca-Mg-HCO3": TEAL, "Ca-Mg-Cl-SO4": AMBER, "Na-Cl-SO4": RED, "Na-HCO3": NAVY}
    root_three, gap = np.sqrt(3) / 2, 0.2
    cation_x, cation_y = sodium + 0.5 * magnesium, root_three * magnesium
    anion_x, anion_y = 1 + gap + chloride + 0.5 * sulphate, root_three * sulphate
    diamond_x = 1 + gap / 2 + 0.5 * (chloride + sulphate) - 0.5 * (calcium + magnesium)
    diamond_y = root_three * gap + root_three * (chloride + sulphate) + root_three * (calcium + magnesium)
    piper_figure = go.Figure()
    for water_type, color in piper_colors.items():
        selected = water_types == water_type
        for x_values, y_values, group_index in [
            (cation_x, cation_y, 0), (anion_x, anion_y, 1), (diamond_x, diamond_y, 2)
        ]:
            piper_figure.add_trace(go.Scatter(
                x=x_values[selected], y=y_values[selected], mode="markers", name=water_type,
                legendgroup=water_type, showlegend=group_index == 0,
                marker={"size": 4, "color": color, "opacity": 0.45},
                hovertemplate=f"{water_type}<extra></extra>",
            ))
    boundaries = [
        ([0, 1, 0.5, 0], [0, 0, root_three, 0]),
        ([1 + gap, 2 + gap, 1.5 + gap, 1 + gap], [0, 0, root_three, 0]),
        ([1 + gap / 2, 1 + gap, 1 + gap / 2, gap / 2, 1 + gap / 2],
         [root_three * gap, root_three * gap + root_three, root_three * gap + 2 * root_three,
          root_three * gap + root_three, root_three * gap]),
    ]
    for x_values, y_values in boundaries:
        piper_figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", line={"color": NAVY},
                                         showlegend=False, hoverinfo="skip"))
    piper_figure.update_layout(title=f"11. Piper diagram: major-ion water types (n = {len(piper):,})",
                               xaxis={"visible": False, "range": [-0.08, 2.35]},
                               yaxis={"visible": False, "range": [-0.1, 1.9], "scaleanchor": "x", "scaleratio": 1},
                               legend_title="Water type")
    figures.append(style_figure(piper_figure, 620))

    map_name, (map_column, map_limit, map_unit) = (
        chosen[0] if len(chosen) == 1 else ("Nitrate (NO3)", LIMITS["Nitrate (NO3)"])
    )
    station_data = data.dropna(subset=["Latitude", "Longitude", map_column, "Station", "State"]).copy()
    station_data["Reading"] = np.where(station_data[map_column] > map_limit, "Above limit", "Within limit")
    figure = px.scatter(station_data, x="Longitude", y="Latitude", color="Reading", color_discrete_map=COLORS,
                        hover_data=["Station", "District", "State", "Season", map_column], opacity=0.75,
                        title=f"12. Station map: {map_name} (limit {map_limit:g} {map_unit})")
    figure.update_traces(marker={"size": 6})
    figure.update_xaxes(range=[67, 98], title="Longitude")
    figure.update_yaxes(range=[5, 38], title="Latitude", scaleanchor="x", scaleratio=1)
    figures.append(style_figure(figure, 600))

    rank_name, (rank_column, rank_limit, _) = (
        chosen[0] if len(chosen) == 1 else ("Nitrate (NO3)", LIMITS["Nitrate (NO3)"])
    )
    ranking = data.dropna(subset=[rank_column]).groupby("State")[rank_column].agg(
        Samples="count", Exceeding=lambda values: (values > rank_limit).sum()
    ).reset_index()
    ranking = ranking[ranking["Samples"] >= 30]
    ranking["Above limit (%)"] = ranking["Exceeding"] / ranking["Samples"] * 100
    ranking = ranking.sort_values("Above limit (%)")
    figure = px.bar(ranking, x="Above limit (%)", y="State", orientation="h", text="Above limit (%)",
                    custom_data=["Samples", "Exceeding"], title=f"13. State ranking: {rank_name} above limit")
    figure.update_traces(marker_color=TEAL, texttemplate="%{x:.1f}%", textposition="outside",
                         hovertemplate="<b>%{y}</b><br>%{x:.1f}% above limit<br>%{customdata[1]} of %{customdata[0]} samples<extra></extra>")
    figures.append(style_figure(figure, max(460, len(ranking) * 23 + 140)))

    return figures


app = Dash(__name__, title="Groundwater Quality in India, 2024")
server = app.server

app.layout = html.Main([
    html.H1("Groundwater Quality in India, 2024"),
    html.P("Interactive groundwater monitoring results from CGWB samples."),
    html.Div([
        html.Label(["State", dcc.Dropdown(
            id="state-filter",
            options=[{"label": "All states", "value": "All states"}]
            + [{"label": value, "value": value} for value in sorted(DATA["State"].dropna().unique())],
            value="All states",
            clearable=False,
        )]),
        html.Label(["Season", dcc.Dropdown(
            id="season-filter",
            options=[{"label": "All seasons", "value": "All seasons"}]
            + [{"label": value, "value": value} for value in sorted(DATA["Season"].dropna().unique())],
            value="All seasons",
            clearable=False,
        )]),
        html.Label(["Parameter", dcc.Dropdown(
            id="parameter-filter",
            options=[{"label": "All parameters", "value": "All parameters"}]
            + [{"label": value, "value": value} for value in PARAMETERS],
            value="All parameters",
            clearable=False,
        )]),
    ], className="filters"),
    html.P("Season-comparison charts retain both seasons so the comparison remains meaningful.", className="note"),
    dcc.Tabs(id="visualization-tabs", value="chart-1", children=[
        dcc.Tab(label=f"{index + 1:02d} {name}", value=f"chart-{index + 1}", children=[
            dcc.Graph(id=f"chart-{index + 1}", config={"displaylogo": False, "responsive": True})
        ])
        for index, name in enumerate(CHART_NAMES)
    ]),
], className="page")


@app.callback(
    [Output(f"chart-{index}", "figure") for index in range(1, 14)],
    Input("state-filter", "value"),
    Input("season-filter", "value"),
    Input("parameter-filter", "value"),
)
def update_visualizations(state, season, parameter):
    return build_figures(state, season, parameter)


if __name__ == "__main__":
    app.run(debug=True)