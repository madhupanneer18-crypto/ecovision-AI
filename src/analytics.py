"""
EcoVision AI - Interactive Analytics Dashboard Visualizations
Generates modern Plotly charts, KPI metrics, and eco-impact milestone calculations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import CLASS_COLORS


def generate_kpis(df: pd.DataFrame) -> dict:
    """Calculates summary KPIs from the history dataframe."""
    if df.empty:
        return {
            "total_scans": 0,
            "total_items": 0,
            "recyclable_items": 0,
            "diversion_rate": 0.0,
            "total_co2_kg": 0.0,
            "total_energy_kwh": 0.0,
            "total_water_l": 0.0,
            "avg_confidence": 0.0,
            "trees_saved": 0.0,
            "phones_charged": 0
        }

    total_scans = len(df)
    total_items = int(df["item_count"].sum()) if "item_count" in df.columns else total_scans
    
    # Recyclable items count
    rec_mask = df["recyclable"].str.lower() == "yes"
    recyclable_items = int(df.loc[rec_mask, "item_count"].sum()) if "item_count" in df.columns else int(rec_mask.sum())
    
    diversion_rate = round((recyclable_items / max(1, total_items)) * 100, 1)

    total_co2 = round(float(df["co2_saved_kg"].sum()), 2) if "co2_saved_kg" in df.columns else 0.0
    total_energy = round(float(df["energy_saved_kwh"].sum()), 2) if "energy_saved_kwh" in df.columns else 0.0
    total_water = round(float(df["water_saved_l"].sum()), 1) if "water_saved_l" in df.columns else 0.0
    avg_conf = round(float(df["confidence"].mean()) * 100, 1) if "confidence" in df.columns else 0.0

    # Eco equivalencies
    # 1 kg CO2 avoided is roughly equivalent to 4 km driven in an average gasoline car
    # 1 mature tree absorbs ~22 kg CO2 per year (so total_co2 / 22 tree-years)
    # 1 kWh charges an iPhone 15 (~12 Wh) approx 83 times
    trees_saved = round(total_co2 / 22.0, 2)
    phones_charged = int(total_energy * 83)

    return {
        "total_scans": total_scans,
        "total_items": total_items,
        "recyclable_items": recyclable_items,
        "diversion_rate": diversion_rate,
        "total_co2_kg": total_co2,
        "total_energy_kwh": total_energy,
        "total_water_l": total_water,
        "avg_confidence": avg_conf,
        "trees_saved": trees_saved,
        "phones_charged": phones_charged
    }


def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Creates a donut chart of waste categories distribution."""
    if df.empty or "primary_class" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    # Group classes (clean multi-item labels)
    clean_classes = []
    for c in df["primary_class"]:
        if "plastic" in str(c).lower():
            clean_classes.append("Plastic")
        elif "glass" in str(c).lower():
            clean_classes.append("Glass")
        elif "metal" in str(c).lower():
            clean_classes.append("Metal")
        elif "cardboard" in str(c).lower():
            clean_classes.append("Cardboard")
        elif "paper" in str(c).lower():
            clean_classes.append("Paper")
        else:
            clean_classes.append("Trash / Residual")

    counts = pd.Series(clean_classes).value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    color_map = {
        "Plastic": CLASS_COLORS["Plastic"]["hex"],
        "Glass": CLASS_COLORS["Glass"]["hex"],
        "Metal": CLASS_COLORS["Metal"]["hex"],
        "Paper": CLASS_COLORS["Paper"]["hex"],
        "Cardboard": CLASS_COLORS["Cardboard"]["hex"],
        "Trash / Residual": CLASS_COLORS["Trash"]["hex"]
    }
    pie_colors = [color_map.get(cat, "#78909C") for cat in counts["Category"]]

    fig = go.Figure(data=[go.Pie(
        labels=counts["Category"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=pie_colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Scans: %{value}<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="<b>Waste Category Breakdown</b>", font=dict(size=18, color="#1b4332")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_recyclability_ratio(df: pd.DataFrame) -> go.Figure:
    """Creates a pie chart showing Recyclable vs Non-Recyclable items."""
    if df.empty or "recyclable" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    rec_counts = df["recyclable"].value_counts().reset_index()
    rec_counts.columns = ["Status", "Count"]

    color_map = {
        "Yes": "#2e7d32",     # Forest Green
        "No": "#c62828",      # Crimson Red
        "Mixed": "#f57f17",   # Amber
        "Special": "#0277bd"  # Blue
    }
    slice_colors = [color_map.get(s, "#78909C") for s in rec_counts["Status"]]

    fig = go.Figure(data=[go.Pie(
        labels=rec_counts["Status"],
        values=rec_counts["Count"],
        hole=0.50,
        marker=dict(colors=slice_colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        title=dict(text="<b>Recyclability Ratio</b>", font=dict(size=18, color="#1b4332")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_confidence_distribution(df: pd.DataFrame) -> go.Figure:
    """Creates a box plot of model confidence distributions across categories."""
    if df.empty or "confidence" not in df.columns:
        fig = go.Figure()
        return fig

    # Map primary_class
    df_plot = df.copy()
    df_plot["conf_pct"] = df_plot["confidence"] * 100

    fig = px.box(
        df_plot,
        x="primary_class",
        y="conf_pct",
        color="primary_class",
        points="all",
        labels={"primary_class": "Class / Event", "conf_pct": "Confidence (%)"},
        title="<b>Model Confidence Distribution by Class</b>"
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[50, 105], title="Confidence (%)"),
        xaxis=dict(title=""),
        margin=dict(l=20, r=20, t=50, b=20),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_activity_timeline(df: pd.DataFrame) -> go.Figure:
    """Plots prediction scan count over time."""
    if df.empty or "timestamp" not in df.columns:
        fig = go.Figure()
        return fig

    df_time = df.copy()
    df_time["dt"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
    df_time = df_time.dropna(subset=["dt"])

    if df_time.empty:
        return go.Figure()

    # Resample by date or order
    df_time["date"] = df_time["dt"].dt.strftime("%Y-%m-%d")
    daily = df_time.groupby(["date", "mode"]).size().reset_index(name="scans")

    fig = px.bar(
        daily,
        x="date",
        y="scans",
        color="mode",
        barmode="group",
        labels={"date": "Date", "scans": "Number of Scans", "mode": "Inference Mode"},
        title="<b>Daily Scan Activity by Mode</b>",
        color_discrete_map={
            "Classification": "#2e7d32",
            "Detection": "#0288d1",
            "Camera Snap": "#f57c00"
        }
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_probability_bar(probabilities: dict) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of probabilities across all 6 classes.
    """
    classes = list(probabilities.keys())
    scores = [probabilities[c] * 100 for c in classes]
    
    # Sort descending
    sorted_items = sorted(zip(classes, scores), key=lambda x: x[1], reverse=False)
    sorted_classes = [item[0] for item in sorted_items]
    sorted_scores = [item[1] for item in sorted_items]

    colors = [CLASS_COLORS.get(c, {}).get("hex", "#2e7d32") for c in sorted_classes]

    fig = go.Figure(go.Bar(
        x=sorted_scores,
        y=sorted_classes,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255, 255, 255, 0.6)", width=1.5)
        ),
        text=[f"<b>{s:.1f}%</b>" for s in sorted_scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Class Probability Distribution</b>", font=dict(size=16, color="#1b4332")),
        xaxis=dict(range=[0, 115], title="Confidence Score (%)", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="", showgrid=False),
        margin=dict(l=10, r=40, t=40, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
