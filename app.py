# app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import random
import pickle

# Set page config
st.set_page_config(layout="wide", page_title="Firearm Injury Mentorship Lineage")

# Add title
st.title("Firearm Injury Mentorship Lineage")

# Load data from pickle files
@st.cache_resource
def load_data():
    with open(f'source_files/G_v4.pkl', 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    with open(f'source_files/df_track_record.pkl', 'rb') as pickle_file:
        df_track_record = pickle.load(pickle_file)
    return G, df_track_record

G, df_track_record = load_data()

def compute_positions(G, selected_authors=None):
    """
    Compute node positions based on their generations and publication years.
    Uses a combination of level_y_offsets and dynamic spacing.
    """
    random.seed(422)

    # If selected_authors is provided, create a subgraph with those authors and their descendants
    if selected_authors is not None:
        # Get all selected authors that exist in the graph
        valid_authors = [author for author in selected_authors if author in G.nodes()]

        # Get all descendants for each selected author
        all_nodes = set(valid_authors)
        for author in valid_authors:
            descendants = nx.descendants(G, author)
            all_nodes.update(descendants)

        # Create subgraph with selected nodes
        G = G.subgraph(all_nodes).copy()

    level_y_offsets = {
        'First Gen': 0,
        'Second Gen': 20,
        'Third Gen': 50,
        'Fourth Gen': 80,
        'Fifth Gen': 110,
        'Sixth Gen': 140,
        'Seventh Gen': 170,
        'Other': 200
    }

    all_levels = ['First Gen', 'Second Gen', 'Third Gen', 'Fourth Gen', 'Fifth Gen', 'Sixth Gen', 'Seventh Gen', 'Other']
    nodes_by_level = {
        level: [n for n, attr in G.nodes(data=True) if attr.get('level') == level]
        for level in all_levels
    }

    # Sort 'First Gen' nodes by publication year (reverse order)
    first_gen_sorted = sorted(
        nodes_by_level['First Gen'],
        key=lambda n: G.nodes[n]['first_publication_year'],
        reverse=True
    )

    first_gen_y_offsets = {node: i * 10 for i, node in enumerate(first_gen_sorted)}
    node_positions = {}

    # Assign positions for 'First Gen'
    for node in first_gen_sorted:
        year = G.nodes[node].get('first_publication_year', 0)
        node_positions[node] = (year, first_gen_y_offsets[node])

    # Dynamic spacing
    years = [attr.get('first_publication_year', 0) for _, attr in G.nodes(data=True)]
    year_counts = {year: years.count(year) for year in set(years)}
    max_spacing = 30
    min_spacing = 20
    dynamic_y_spacing = {year: max(min_spacing, max_spacing / max(1, count)) for year, count in year_counts.items()}

    levels_to_place = ['Second Gen', 'Third Gen', 'Fourth Gen', 'Fifth Gen', 'Sixth Gen', 'Seventh Gen']
    for level in levels_to_place:
        for node in nodes_by_level[level]:
            if node in node_positions:
                continue
            attributes = G.nodes[node]
            x_pos = attributes.get('first_publication_year', 0)
            base_y_offset = level_y_offsets.get(level, 20)
            y_spacing = dynamic_y_spacing.get(x_pos, min_spacing)
            predecessors = list(G.predecessors(node))
            if predecessors:
                parent_pos = node_positions.get(predecessors[0], (x_pos, base_y_offset))
                parent_y = parent_pos[1]
                y_pos = parent_y + random.uniform(y_spacing - 10, y_spacing + 10)
            else:
                y_pos = base_y_offset + random.uniform(y_spacing - 10, y_spacing + 10)
            node_positions[node] = (x_pos, y_pos)

    return node_positions, nodes_by_level

# Execute compute positions
node_positions, nodes_by_level = compute_positions(G)

def create_figure(G, node_positions, nodes_by_level):
    """
    Create a Plotly figure for the world graph.
    """
    level_colors = {
        'First Gen': 'red',
        'Second Gen': 'blue',
        'Third Gen': 'green',
        'Fourth Gen': 'purple',
        'Fifth Gen': 'orange',
        'Sixth Gen': 'pink',
        'Seventh Gen': 'brown',
        'Other': 'grey'
    }

    plot_nodes = [n for n in G.nodes()]

    node_attrs = [
        (n, node_positions[n][0], node_positions[n][1], G.nodes[n].get('level', 'Seventh Gen'))
        for n in plot_nodes if n in node_positions
    ]

    # Edges
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        if u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    all_levels = ['First Gen', 'Second Gen', 'Third Gen', 'Fourth Gen', 'Fifth Gen', 'Sixth Gen', 'Seventh Gen', 'Other']
    node_traces = []
    for level in all_levels:
        level_nodes = [(n, x, y) for (n, x, y, lvl) in node_attrs if lvl == level]
        if not level_nodes:
            continue

        level_node_x = [x for (_, x, _) in level_nodes]
        level_node_y = [y for (_, _, y) in level_nodes]
        level_node_text = [
            f"Node: {n}<br>"
            f"Level: {level}<br>"
            f"First Publication Year: {G.nodes[n].get('first_publication_year', 'Unknown')}<br>"
            f"First Title: {G.nodes[n].get('first_title', 'Unknown')}<br>"
            f"Predecessors: {', '.join(list(G.predecessors(n)))}<br>"
            f"Clusters: {G.nodes[n].get('cluster_keywords', 'Unknown')}<br>"
            f"Gender: {G.nodes[n].get('gender', 'Unknown')}"
            for (n, _, _) in level_nodes
        ]

        trace = go.Scatter(
            x=level_node_x,
            y=level_node_y,
            mode='markers',
            marker=dict(size=7, color=level_colors[level], line_width=1),
            text=level_node_text,
            textposition='top center',
            hoverinfo='text',
            name=level,
            legendgroup=level
        )
        node_traces.append(trace)

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            # title=dict(
            #     text='Chronologically Ordered Mentorship Lineage',
            #     x=0.5,
            #     y=1.0,
            #     xanchor='center',
            #     yanchor='top',
            #     font=dict(size=20)
            # ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=30),
            xaxis=dict(
                title='Year',
                showgrid=True,
                gridcolor='white',
                gridwidth=0.5,
                zeroline=False,
                tickmode='linear',
                dtick=10,
                tickangle=0,
                tickfont=dict(size=10),
            ),
            yaxis=dict(title=None, showticklabels=False, showgrid=True, zeroline=False),
            height=2000,
            legend=dict(
                title="Node Level",
                orientation="h",
                yanchor="top",
                y=1.0001,
                xanchor="right",
                x=1
            )
        )
    )
    return fig, edge_trace, node_traces

# Sidebar
st.sidebar.header("Network Information")
st.sidebar.write(f"Total Mentors: {G.number_of_nodes()}")
st.sidebar.write(f"Total Connections: {G.number_of_edges()}")

# Create visualization
fig, edge_trace, node_traces = create_figure(G, node_positions, nodes_by_level)

# Display the graph
st.plotly_chart(fig, use_container_width=True)

# Add filters
st.sidebar.subheader("Filters")
selected_level = st.sidebar.selectbox(
    "Filter by Generation",
    ["All"] + list(set(nx.get_node_attributes(G, 'level').values()))
)

if selected_level != "All":
    filtered_nodes = [node for node, attr in G.nodes(data=True) 
                     if attr.get('level') == selected_level]
    subgraph = G.subgraph(filtered_nodes)
    filtered_fig, _, _ = create_figure(subgraph)
    st.plotly_chart(filtered_fig, use_container_width=True)

# Display node details on click
selected_node = st.sidebar.selectbox(
    "Select a Mentor to View Details",
    [""] + sorted(G.nodes())
)

if selected_node:
    st.sidebar.subheader("Mentor Details")
    node_data = G.nodes[selected_node]
    st.sidebar.write(f"**Level:** {node_data.get('level', 'Unknown')}")
    st.sidebar.write(f"**First Publication Year:** {node_data.get('first_publication_year', 'Unknown')}")
    st.sidebar.write(f"**Gender:** {node_data.get('gender', 'Unknown')}")
    st.sidebar.write(f"**Clusters:** {node_data.get('cluster_keywords', 'Unknown')}")

