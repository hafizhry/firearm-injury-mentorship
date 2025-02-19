# app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import random
import pickle

# Set page config
st.set_page_config(layout="wide", page_title="Shirtsleeves to Shirtsleeves")

# Add title
st.title("Shirtsleeves to Shirtsleeves")

st.markdown("""
### A University of Michigan Project on Research Legacy in Firearm Injury Prevention

This visualization explores the mentorship lineages in firearm injury research, inspired by the economic principle 
"shirtsleeves to shirtsleeves in three generations" - where wealth typically diminishes through generations.

In firearm injury research, we examine how research expertise and funding, like wealth, can be sustained across 
generations of mentors and mentees. Since the field's inception fifty years ago, we've seen pioneering investigators, 
faced funding challenges, and now witness renewed support for research.

**Our Goal**: To understand and strengthen the mentorship patterns that create lasting research legacies, ensuring 
future generations of researchers can build upon their predecessors' work effectively.

---
""")

# Add legend header
st.markdown("<h3 style='text-align: left; font-size: 18px; margin-bottom: 10px;'>Generation Level Legend:</h3>", unsafe_allow_html=True)

# Add legend below title
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

# Create a horizontal layout for the legend using columns
legend_cols = st.columns(8)  # Create 8 columns for the legend
for i, (level, color) in enumerate(level_colors.items()):
    col_index = i % 8  # Determine which column to put the legend item in
    with legend_cols[col_index]:
        st.markdown(
            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
            f'<div style="width: 15px; height: 15px; background-color: {color}; border-radius: 50%; margin-right: 8px;"></div>'
            f'<div style="font-size: 14px;">{level}</div>'
            '</div>',
            unsafe_allow_html=True
        )

# Add a separator
st.markdown("---")

# Load data from pickle files
@st.cache_resource
def load_data():
    with open(f'source_files/G_v4.pkl', 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    with open(f'source_files/df_track_record.pkl', 'rb') as pickle_file:
        df_track_record = pickle.load(pickle_file)
    return G, df_track_record

G, df_track_record = load_data()

# Add search functionality
st.subheader("Search and Select Mentor")

# Create single column for dropdown
# Get all author names and sort them
all_authors = sorted(list(G.nodes()))
# Create the dropdown
selected_mentor = st.selectbox("Select a mentor to zoom to or 'World View' to see the entire network", ["World View"] + all_authors, help="Select a mentor from the dropdown list or 'World View' to see the entire network")

# Use the dropdown selection
final_search_term = "" if selected_mentor == "World View" else selected_mentor

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

# Compute node positions
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
            # f"Gender: {G.nodes[n].get('gender', 'Unknown')}"
            for (n, _, _) in level_nodes
        ]

        # Extract just the names for the node labels
        node_names = [n for (n, _, _) in level_nodes]

        trace = go.Scatter(
            x=level_node_x,
            y=level_node_y,
            mode='markers+text',  # Add text mode to support showing names
            marker=dict(size=7, color=level_colors[level], line_width=1),
            text=node_names,  # Use node names for text
            textposition='top center',
            textfont=dict(size=10, color='rgba(0,0,0,0)'),  # Start with transparent text
            hovertext=level_node_text,  # Use full info for hover
            hoverinfo='text',
            name=level,
            legendgroup=level
        )
        node_traces.append(trace)

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=0, r=0, t=45),
            paper_bgcolor='#F0F8FF',
            plot_bgcolor='#F0F8FF',
            xaxis=dict(
                title=dict(
                    text='Year',
                    font=dict(color='black')
                ),
                showgrid=True,
                gridcolor='lightgrey',
                gridwidth=0.5,
                zeroline=False,
                tickmode='linear',
                dtick=5,
                tickangle=0,
                tickfont=dict(size=10, color='black'),
                ticks='outside',
                tickwidth=2,
                tickcolor='grey',
                ticklen=8,
                showline=True,
                linecolor='grey',
                mirror=True,
                showticklabels=True,
                side='top'
            ),
            yaxis=dict(title=None, showticklabels=False, showgrid=True, zeroline=False),
            height=2000  # Full view height
        )
    )
    return fig, edge_trace, node_traces

def highlight_and_zoom_to_mentor(fig, G, node_positions, search_term):
    """
    Highlight a mentor and their lineage, then zoom to their position
    """
    if not search_term:
        return fig
    
    # Find matching mentors (case-insensitive partial match)
    matching_mentors = [n for n in G.nodes() if search_term.lower() in n.lower()]
    
    if not matching_mentors:
        st.warning(f"No mentor found matching '{search_term}'")
        return fig
    
    # If multiple matches found, let user select one
    if len(matching_mentors) > 1:
        selected_mentor = st.selectbox(
            "Multiple matches found. Select a mentor:",
            matching_mentors
        )
    else:
        selected_mentor = matching_mentors[0]
    
    # Get ancestors and descendants
    ancestors = list(nx.ancestors(G, selected_mentor))
    descendants = list(nx.descendants(G, selected_mentor))
    lineage = ancestors + descendants + [selected_mentor]
    
    # Get position of the selected mentor
    if selected_mentor in node_positions:
        center_x, center_y = node_positions[selected_mentor]
        
        # Calculate the bounds of the lineage
        x_positions = [node_positions[n][0] for n in lineage if n in node_positions]
        y_positions = [node_positions[n][1] for n in lineage if n in node_positions]
        
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        
        # Add padding
        x_padding = max(10, (x_max - x_min) * 0.4)  # At least 40 years padding or 20% of range
        y_padding = 20
        
        # Ensure minimum x-axis range of 20 years for longitudinal view
        x_range_size = x_max - x_min
        if x_range_size < 100:
            x_center = (x_max + x_min) / 2
            x_min = x_center - 10  
            x_max = x_center + 10 
        
        x_range = [x_min - x_padding, x_max + x_padding]
        y_range = [y_min - y_padding, y_max + y_padding]
        
        # Update figure layout for zooming
        fig.update_layout(
            xaxis=dict(
                range=x_range,
                dtick=2,  
                tickmode='linear',
                gridcolor='lightgrey',
                side='top'
            ),
            yaxis=dict(
                range=y_range,
            ),
            height=700  
        )
        
        # Highlight the lineage
        for trace in fig.data:
            if hasattr(trace, 'marker'):
                # Get the node positions for this trace
                node_x = trace.x
                node_y = trace.y
                node_text = trace.text
                
                # Create lists for opacity
                opacity_list = []
                size_list = []
                text_list = []
                
                for i in range(len(node_x)):
                    # Safely handle None values in node_text
                    try:
                        node_text_value = node_text[i] if node_text and i < len(node_text) else None
                        node_name = node_text_value.split('<br>')[0].replace('Node: ', '') if node_text_value else ''
                    except (AttributeError, IndexError):
                        node_name = ''
                        
                    if node_name in lineage:
                        opacity_list.append(1.0)
                        size_list.append(10 if node_name == selected_mentor else 7)  # Make selected mentor larger
                        text_list.append(node_name)  # Show name for highlighted nodes
                    else:
                        opacity_list.append(0.2)
                        size_list.append(7)
                        text_list.append('')  # No text for non-highlighted nodes
                
                # Update marker properties
                trace.marker.opacity = opacity_list
                trace.marker.size = size_list
                
                # Update text visibility based on lineage
                text_colors = []
                for i in range(len(node_x)):
                    node_name = trace.text[i] if trace.text and i < len(trace.text) else ''
                    if node_name in lineage:
                        text_colors.append('black')  # Show text for lineage
                    else:
                        text_colors.append('rgba(0,0,0,0)')  # Hide text for others
                
                trace.textfont.color = text_colors
        
        st.success(f"Showing lineage for {selected_mentor}")
    
    return fig

# Create visualization
fig, edge_trace, node_traces = create_figure(G, node_positions, nodes_by_level)

# Apply search and zoom if search term is provided and not World View
if final_search_term:
    fig = highlight_and_zoom_to_mentor(fig, G, node_positions, final_search_term)
else:
    # Reset to world view settings
    fig.update_layout(
        xaxis=dict(
            autorange=True,
            dtick=5,
            tickmode='linear',
            side='top'
        ),
        yaxis=dict(autorange=True),
        height=2000  # Full view height for world view
    )
    # Reset all nodes to full opacity and original size
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            trace.marker.opacity = 1.0
            trace.marker.size = 7
            trace.textfont.color = 'rgba(0,0,0,0)'  # Hide all text in world view

# Display the graph
st.plotly_chart(fig, theme=None, use_container_width=True)
