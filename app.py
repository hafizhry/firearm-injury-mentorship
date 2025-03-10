# app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pickle
import base64
from pathlib import Path

# Set page config
st.set_page_config(layout="wide", page_title="Shirtsleeves to Shirtsleeves")

# Load and encode the background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
        .title-container {
            background-image: url("data:image/png;base64,%s");
        }
    </style>
    ''' % bin_str
    return page_bg_img

# Set the background image
background_image_path = "source_files/Shield-Pattern-Hero.png"
st.markdown(set_background_image(background_image_path), unsafe_allow_html=True)

# Add custom CSS for the title section
st.markdown("""
    <style>
        .title-container {
            position: relative;
            text-align: center;
            color: white;
            padding: 4rem 0;
            margin-bottom: 2rem;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .title-text {
            font-size: 2.5rem;
            font-weight: bold;
            text-decoration: none;
            border-bottom: 3px solid #FFD43B;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin: 0;
            padding: 1rem 0;
            display: inline-block;
        }
        .subtitle-text {
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            margin: 0;
            padding-bottom: 1rem;
        }
    </style>
    <div class="title-container">
        <h1 class="title-text">Shirtsleeves to Shirtsleeves in Three Generations</h1>
    </div>
""", unsafe_allow_html=True)

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

# Load data from pickle files
@st.cache_resource
def load_data():
    with open(f'source_files/G_v4.pkl', 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    with open(f'source_files/df_track_record.pkl', 'rb') as pickle_file:
        df_track_record = pickle.load(pickle_file)
    return G, df_track_record

G, df_track_record = load_data()

st.markdown("""
### Search and Select Author

##### How to use:
1. Use the dropdown to select an author and zoom into their mentorship network.
2. Select 'World View' to see the entire mentorship network.
3. Drag to zoom into a specific section and hover the mouse over each dot to see detailed information on the author.
4. Explore the different generation levels and mentorship direction using the color legend below.

""")

# Define color scheme for generation levels and edges
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

edge_colors = {
    'Forward in time': '#1f77b4',
    'Backward in time': 'orange'
}

# # Add legend headers
# st.markdown("<h3 style='text-align: left; font-size: 18px; margin-bottom: 0px; color: #666;'>Legend:</h3>", unsafe_allow_html=True)

# Create columns for both legends
st.markdown("<div style='display: flex;'>", unsafe_allow_html=True)

# Left column for Generation Levels (reduced margin-bottom to 2px)
st.markdown("<div style='flex: 1;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 14px; margin-bottom: 2px; color: #666;'>Generation Levels:</p>", unsafe_allow_html=True)

# Create a horizontal layout for the generation legend
legend_cols = st.columns(8)
for i, (level, color) in enumerate(level_colors.items()):
    col_index = i % 8
    with legend_cols[col_index]:
        st.markdown(
            f'<div style="display: flex; align-items: center; margin: 2px 0;">'
            f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin-right: 6px;"></div>'
            f'<div style="font-size: 12px; color: #444;">{level}</div>'
            '</div>',
            unsafe_allow_html=True
        )

# Right column for Lineage Direction
st.markdown("<div style='flex: 1;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 14px; margin-bottom: 5px; color: #666;'>Lineage Direction:</p>", unsafe_allow_html=True)

# Create a horizontal layout for the edge legend
edge_cols = st.columns(2)
for i, (edge_type, color) in enumerate(edge_colors.items()):
    with edge_cols[i]:
        st.markdown(
            f'<div style="display: flex; align-items: center; margin: 2px 0;">'
            f'<div style="width: 20px; height: 2px; background-color: {color}; margin-right: 6px;"></div>'
            f'<div style="font-size: 12px; color: #444;">{edge_type}</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown("</div></div>", unsafe_allow_html=True)

# Add a small space after the legend
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Create single column for dropdown
# Get all author names and sort them
all_authors = sorted(list(G.nodes()))

# Create the dropdown
selected_mentor = st.selectbox("Select an author to zoom to or 'World View' to see the entire network", ["World View"] + all_authors, help="Select an author from the dropdown list or 'World View' to see the entire network")

# Use the dropdown selection
final_search_term = "" if selected_mentor == "World View" else selected_mentor

def compute_sequential_grid_positions(G, grid_spacing=30):
    """
    Compute node positions by processing complete lineages sequentially.
    Each lineage tree gets its own grid space.
    """
    node_positions = {}
    nodes_by_level = {
        'First Gen': [], 'Second Gen': [], 'Third Gen': [], 
        'Fourth Gen': [], 'Fifth Gen': [], 'Sixth Gen': [], 
        'Seventh Gen': [], 'Other': []
    }
    
    # Find all first gen authors (roots)
    first_gen_authors = [
        node for node, attr in G.nodes(data=True) 
        if attr.get('level') == 'First Gen'
    ]
    
    # Sort first gen authors by publication year
    first_gen_authors.sort(
        key=lambda x: G.nodes[x]['first_publication_year'],
        reverse=True
    )
    
    def process_lineage(root_node, current_base_row):
        """
        Process a complete lineage tree starting from a root node.
        Returns the maximum row used in this lineage.
        """
        # Dictionary to track occupied grid points in this lineage
        occupied_grid = set()
        lineage_positions = {}
        
        def place_node_and_descendants(node, base_row, visited=None):
            if visited is None:
                visited = set()
            
            if node in visited:
                return base_row
            
            visited.add(node)
            year = G.nodes[node]['first_publication_year']
            level = G.nodes[node]['level']
            
            # Find first available row for this node
            current_row = base_row
            while (year, current_row) in occupied_grid:
                current_row += 1
            
            # Place the node
            lineage_positions[node] = (year, current_row * grid_spacing)
            occupied_grid.add((year, current_row))
            nodes_by_level[level].append(node)
            
            # Process all descendants
            max_descendant_row = current_row
            descendants = list(G.successors(node))
            
            # Sort descendants by year
            descendants.sort(
                key=lambda x: G.nodes[x]['first_publication_year']
            )
            
            for descendant in descendants:
                if descendant not in visited:
                    descendant_row = place_node_and_descendants(
                        descendant,
                        max_descendant_row + 1,
                        visited
                    )
                    max_descendant_row = max(max_descendant_row, descendant_row)
            
            return max_descendant_row
        
        # Process the entire lineage
        max_row_used = place_node_and_descendants(root_node, current_base_row)
        
        # Update global positions
        node_positions.update(lineage_positions)
        
        return max_row_used
    
    # Process each first gen author and their complete lineage
    current_base_row = 0
    for first_gen in first_gen_authors:
        # Process this lineage tree
        max_row = process_lineage(first_gen, current_base_row)
        
        # Start next lineage at new base row with some padding
        current_base_row = max_row + 2  # Add padding between lineages
    
    return node_positions, nodes_by_level

# Execute compute positions
node_positions, nodes_by_level = compute_sequential_grid_positions(G)

def create_figure(G, node_positions, nodes_by_level):
    """
    Create a Plotly figure with colored edges based on temporal direction.
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

    # Separate edges by direction
    forward_edge_x = []
    forward_edge_y = []
    backward_edge_x = []
    backward_edge_y = []
    
    for u, v in G.edges():
        if u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            year_u = G.nodes[u]['first_publication_year']
            year_v = G.nodes[v]['first_publication_year']
            
            # If target year is greater than source year, it's forward
            if year_v >= year_u:
                forward_edge_x.extend([x0, x1, None])
                forward_edge_y.extend([y0, y1, None])
            else:
                backward_edge_x.extend([x0, x1, None])
                backward_edge_y.extend([y0, y1, None])

    # Create separate traces for forward and backward edges
    forward_edge_trace = go.Scatter(
        x=forward_edge_x,
        y=forward_edge_y,
        line=dict(width=1, color='#1f77b4'),
        hoverinfo='none',
        mode='lines',
        name='Forward in time',
        showlegend=False,
        opacity=0.6
    )

    backward_edge_trace = go.Scatter(
        x=backward_edge_x,
        y=backward_edge_y,
        line=dict(width=1, color='orange'),
        hoverinfo='none',
        mode='lines',
        name='Backward in time',
        showlegend=False,
        opacity=0.6
    )

    # Create node traces
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
            + (f"Predecessors: {', '.join(list(G.predecessors(n)))}<br>" if list(G.predecessors(n)) else "")
            # + f"Clusters: {G.nodes[n].get('cluster_keywords', 'Unknown')}<br>"
            # f"Coordinates: ({x:.2f}, {y:.2f})" # Used for debugging
            for (n, x, y) in level_nodes
        ]

        node_names = [n for (n, _, _) in level_nodes]

        trace = go.Scatter(
            x=level_node_x,
            y=level_node_y,
            mode='markers+text',
            marker=dict(size=10, color=level_colors[level], line_width=1),
            text=node_names,
            textposition='top center',
            textfont=dict(size=12, color='rgba(0,0,0,0)'),
            hovertext=level_node_text,
            hoverinfo='text',
            name=level,
            legendgroup=level
        )
        node_traces.append(trace)

    fig = go.Figure(
        data=[forward_edge_trace, backward_edge_trace] + node_traces,
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
                tickfont=dict(size=12, color='black'),
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
            height=2000
        )
    )
    return fig, [forward_edge_trace, backward_edge_trace], node_traces

def highlight_and_zoom_to_mentor(fig, G, node_positions, search_term, df_track_record=None):
    """
    Highlight an author and their lineage, then zoom to their position
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
            "Multiple matches found. Select an author:",
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
        # center_x, center_y = node_positions[selected_mentor]
        
        # Calculate the bounds of the lineage
        x_positions = [node_positions[n][0] for n in lineage if n in node_positions]
        y_positions = [node_positions[n][1] for n in lineage if n in node_positions]
        
        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)
        
        # Add padding
        x_padding = max(10, (x_max - x_min) * 0.4) 
        y_padding = 50
        
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
            height=700,
            margin=dict(b=0, t=30, l=0, r=0)
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
                        size_list.append(12 if node_name == selected_mentor else 9) 
                        text_list.append(node_name)  
                    else:
                        opacity_list.append(0.2)
                        size_list.append(7)
                        text_list.append('')
                
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

        # Add career milestones if df_track_record is provided
        if df_track_record is not None:
            # Filter milestones for the selected mentor
            df_mentor_milestones = df_track_record[df_track_record['author_name'] == selected_mentor]

            # Get related nodes (lineage)
            related_nodes = lineage

            # Determine the y-position of the 'First Gen' nodes
            first_gen_y_position = [
                node_positions[n][1] for n in related_nodes
                if G.nodes[n].get('level') == 'First Gen' and n in node_positions
            ]
            first_gen_y_start = min(first_gen_y_position) if first_gen_y_position else 0

            # Determine the y-position of the last node
            last_gen_y_position = [
                node_positions[n][1] for n in related_nodes
                if G.nodes[n].get('level') in ['Seventh Gen', 'Sixth Gen', 'Fifth Gen', 'Fourth Gen', 'Third Gen', 'Second Gen'] 
                and n in node_positions
            ]
            last_gen_y_end = max(last_gen_y_position) if last_gen_y_position else max(node_positions.values(), key=lambda v: v[1])[1]

            # Add vertical lines for each milestone
            for _, milestone in df_mentor_milestones.iterrows():
                start_year = milestone['start_year'] if pd.notna(milestone['start_year']) else milestone['end_year']
                position_grant = milestone['position_grant']
                institution_source = milestone['institution_source']

                # Create hover points at intervals along the line
                num_points = 50  
                y_start = first_gen_y_start - 50  
                y_end = last_gen_y_end + 50      
                y_points = np.linspace(y_start, y_end, num_points)
                x_points = [start_year] * num_points

                fig.add_trace(go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode="lines+markers",
                    line=dict(
                        color="gray",
                        dash="dash",
                        width=2  
                    ),
                    marker=dict(
                        size=0,
                        opacity=0
                    ),
                    hoverinfo="text",
                    hovertext=[f"Position/Grant: {position_grant}<br>Institution: {institution_source}"] * num_points,
                    showlegend=False,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                ))

            # Update layout to ensure hover works
            fig.update_layout(
                hovermode='closest',
                hoverdistance=10,
                # Increase the plot margins to accommodate the extended lines
                margin=dict(t=60, b=10)  # Added top and bottom margins
            )
        
        st.success(f"Showing lineage for {selected_mentor}")
    
    return fig

# Create visualization
fig, edge_trace, node_traces = create_figure(G, node_positions, nodes_by_level)

# Apply search and zoom if search term is provided and not World View
if final_search_term:
    fig = highlight_and_zoom_to_mentor(fig, G, node_positions, final_search_term, df_track_record)
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

# Add custom CSS to remove plot container margins
st.markdown("""
    <style>
    .element-container:has(div.stPlotlyChart) {
        margin-bottom: -10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Display the graph
st.plotly_chart(fig, theme=None, use_container_width=True)

def calculate_mentorship_stats(G, selected_mentor):
    """Calculate mentorship statistics for the selected mentor."""
    # Get all descendants
    descendants = list(nx.descendants(G, selected_mentor))
    total_descendants = len(descendants)
    
    # Calculate branching factor
    direct_mentees = list(G.successors(selected_mentor))
    branching_factor = len(direct_mentees)
    
    # Calculate lineage depth
    def get_max_depth(node, current_depth=0, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return current_depth
        visited.add(node)
        successors = list(G.successors(node))
        if not successors:
            return current_depth
        return max(get_max_depth(successor, current_depth + 1, visited) for successor in successors)
    
    lineage_depth = get_max_depth(selected_mentor)
    
    # Calculate average time gap between generations
    time_gaps = []
    mentor_year = G.nodes[selected_mentor].get('first_publication_year')
    
    # Collect all mentee years for mentoring span calculation
    mentee_years = []
    
    # First check direct mentees
    for mentee in direct_mentees:
        mentee_year = G.nodes[mentee].get('first_publication_year')
        if mentee_year and isinstance(mentee_year, (int, float)):
            mentee_years.append(mentee_year)
        if mentee_year and mentor_year and isinstance(mentee_year, (int, float)) and isinstance(mentor_year, (int, float)):
            gap = mentee_year - mentor_year
            time_gaps.append(gap)
    
    # Then check all other mentor-mentee relationships in the lineage
    for node in descendants:
        predecessors = list(G.predecessors(node))
        for pred in predecessors:
            if pred != selected_mentor:  # Skip if already counted above
                mentee_year = G.nodes[node].get('first_publication_year')
                pred_year = G.nodes[pred].get('first_publication_year')
                if mentee_year and pred_year and isinstance(mentee_year, (int, float)) and isinstance(pred_year, (int, float)):
                    gap = mentee_year - pred_year
                    time_gaps.append(gap)
                if mentee_year and isinstance(mentee_year, (int, float)):
                    mentee_years.append(mentee_year)
    
    # Calculate mentoring span
    if mentee_years:
        mentoring_span = max(mentee_years) - min(mentee_years)
    else:
        mentoring_span = None
    
    # Calculate average time gap
    if time_gaps:
        avg_time_gap = round(sum(time_gaps) / len(time_gaps), 1)
    else:
        avg_time_gap = None
    
    return {
        'total_descendants': total_descendants,
        'branching_factor': branching_factor,
        'lineage_depth': lineage_depth,
        'avg_time_gap': avg_time_gap if avg_time_gap is not None else "N/A",
        'mentoring_span': mentoring_span if mentoring_span is not None else "N/A"
    }

# After the plotly chart display, add the statistics section
if final_search_term and final_search_term != "World View":
    st.markdown('<h3 style="margin: 0 0 10px 0; text-align: center;">Mentorship Tree Statistics</h3>', unsafe_allow_html=True)
    
    # Calculate statistics
    stats = calculate_mentorship_stats(G, final_search_term)
    
    # Create five columns for the statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Custom CSS for the metric cards
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 12px 8px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            height: 100%;
            margin: 0 4px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #1f77b4;
            margin: 3px 0;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            line-height: 1.2;
            margin-top: 2px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display statistics in cards
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_descendants']}</div>
                <div class="metric-label">Total Academic Descendants<br>(Direct + Indirect Mentees)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['branching_factor']}</div>
                <div class="metric-label">Direct Mentees<br>(First-degree Connections)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['lineage_depth']}</div>
                <div class="metric-label">Lineage Depth<br>(Maximum Generations)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['avg_time_gap']}</div>
                <div class="metric-label">Average Time Gap<br>(Years Between Generations)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['mentoring_span']}</div>
                <div class="metric-label">Mentoring Span<br>(Years First to Last Mentee)</div>
            </div>
        """, unsafe_allow_html=True)
