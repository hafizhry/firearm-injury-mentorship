# app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pickle
import base64
from pathlib import Path
from visualization_utils import compute_sequential_grid_positions, create_figure, highlight_and_zoom_to_mentor, calculate_mentorship_stats

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
    with open(f'source_files/G_v5.2.pkl', 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    with open(f'source_files/df_track_record.pkl', 'rb') as pickle_file:
        df_track_record = pickle.load(pickle_file)
    return G, df_track_record

G, df_track_record = load_data()

st.markdown(""" 
### Search and Select Author

In this visualization, each dot represents a publication by an author in the mentorship network. Authors can have two types of publications: solo publications (hollow circles) and mentored publications (filled circles). The lines connecting dots show mentorship relationships:

- **Blue lines** indicate forward mentorship connections (mentor published earlier than mentee)
- **Orange lines** indicate backward connections (mentee published earlier than mentor)
- **Green dotted lines** connect an author's own solo and mentored publications, showing their career progression

Authors are color-coded by their generation level, from first generation pioneers (red) to seventh generation researchers (brown).

##### How to use:
1. Select 'World View' to see the entire mentorship network.
2. Use the dropdown to select an author and zoom into their complete mentorship network.
3. When viewing an author's lineage, you'll see:
   - All publications by the selected author (both solo and mentored)
   - All their mentors (academic ancestors)
   - All their mentees (academic descendants)
4. Hover over any dot to see detailed information about that publication and its author.
5. The visualization maintains the same author name across all their publications, making it easier to track an individual's research lineage.

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

# Create a horizontal layout for the edge legend with 4 types of edges
edge_types = {
    'Forward Mentorship': {
        'color': '#1f77b4', 
        'dash': 'solid',
        'description': 'Mentor published first, traditional mentorship'
    },
    'Backward Mentorship': {
        'color': 'orange', 
        'dash': 'solid',
        'description': 'Mentee published before mentor'
    },
    'Author Progression': {
        'color': 'green',
        'dash': 'dot',
        'description': 'Connects an author\'s first solo and first mentored works'
    }
}

# Create three exactly equal columns
edge_cols = st.columns(3)

# Get the list of items
edge_items = list(edge_types.items())

# Display each edge type in its own column
for i in range(3):
    if i < len(edge_items):
        edge_type, style = edge_items[i]
        with edge_cols[i]:
            # Create the line style based on dash type
            if style['dash'] == 'solid':
                line_style = f"border-top: 2px solid {style['color']};"
            elif style['dash'] == 'dash':
                line_style = f"border-top: 2px dashed {style['color']};"
            elif style['dash'] == 'dot':
                line_style = f"border-top: 2px dotted {style['color']};"
                
            st.markdown(
                f'<div style="display: flex; align-items: center; margin: 2px 0;">'
                f'<div style="width: 20px; height: 0px; {line_style} margin-right: 6px;"></div>'
                f'<div style="font-size: 12px; color: #444;">{edge_type}</div>'
                '</div>'
                f'<div style="font-size: 11px; color: #666; margin-left: 26px;">{style["description"]}</div>',
                unsafe_allow_html=True
            )

st.markdown("</div></div>", unsafe_allow_html=True)

# Add a small space after the legend
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Create single column for dropdown
# Modify the part that creates the author list for the dropdown
# Replace this code in the app.py file

# Get all author names and sort them
author_lookup = {}
for n in G.nodes():
    # Extract the author name without suffixes - use the 'author' attribute
    author = G.nodes[n].get('author', '')
    if not author:
        continue
        
    # Get earliest publication year for this author
    pub_year = G.nodes[n].get('first_publication_year', '')
    
    # Only add each author once to the lookup
    if author not in author_lookup:
        author_lookup[author] = {
            'nodes': [],
            'pub_year': pub_year
        }
    else:
        # Keep track of earliest publication year
        if pub_year and (not author_lookup[author]['pub_year'] or pub_year < author_lookup[author]['pub_year']):
            author_lookup[author]['pub_year'] = pub_year
            
    # Add this node to the author's node list
    author_lookup[author]['nodes'].append(n)

# Create author options with their earliest publication year
author_options = [f"{author}" for author, info in author_lookup.items()]
author_options.sort()  # Sort alphabetically

# Create the dropdown - use the new author_options list
selected_mentor = st.selectbox(
    "Select an author to zoom to or 'World View' to see the entire network", 
    ["World View"] + author_options, 
    help="Select an author from the dropdown list or 'World View' to see the entire network"
)

# Extract the author name without the publication year
final_search_term = ""
if selected_mentor != "World View":
    author_name = selected_mentor.split(' (')[0]
    if author_name in author_lookup:
        # Use the first node for this author to search
        final_search_term = author_lookup[author_name]['nodes'][0]

# Execute compute positions
node_positions, nodes_by_level = compute_sequential_grid_positions(G)

# Create visualization
fig, edge_trace, node_traces = create_figure(G, node_positions, nodes_by_level, final_search_term)

# Apply search and zoom if search term is provided and not World View
# Modify the application of search and zoom
if final_search_term:
    # First identify the author name for this node
    author_name = G.nodes[final_search_term].get('author', '')
    
    # Find all nodes for this author
    author_nodes = [n for n in G.nodes() if G.nodes[n].get('author') == author_name]
    
    # Highlight and zoom to all nodes for this author
    fig = highlight_and_zoom_to_mentor(fig, G, node_positions, final_search_term, df_track_record, author_nodes)
else:
    # Calculate the min and max y-coordinates from node positions
    y_values = [pos[1] for pos in node_positions.values()]
    min_y = min(y_values) - 3000  # Add padding below
    max_y = max(y_values) + 3000  # Add padding above
    
    # Reset to world view settings
    fig.update_layout(
        xaxis=dict(
            range=[1969, 2026],
            dtick=5,
            tickmode='linear',
            side='top'
        ),
        yaxis=dict(
            # Set range based on actual data with padding
            range=[min_y, max_y],
            # showgrid=True,
            # showticklabels=True
        ),
        height=20000,
        # margin=dict(t=45, l=50)
    )
    
    # Add repeating x-axis grid lines at regular intervals
    # Set fixed x-axis range starting at 1965
    x_min = 1965
    x_max = 2030  # Adjust end year as needed
    
    # Create repeating x-axis grids at large intervals
    y_interval = 40000  # Large interval for repeating x-axis
    y_start = min_y + 5000  # Start a bit below the top axis
    
    # Calculate how many repeating axes we need
    num_repeats = int((max_y - y_start) / y_interval) + 1
    
    # Add subtle horizontal grid lines at each repeat position
    for i in range(num_repeats):
        y_pos = y_start + (i * y_interval)
        
        # Skip if we're at the very top (original axis)
        if y_pos < y_start + 1000:
            continue
            
        # Add a horizontal line to represent the x-axis
        fig.add_shape(
            type="line",
            x0=x_min,
            y0=y_pos,
            x1=x_max,
            y1=y_pos,
            line=dict(
                color="rgba(150, 150, 150, 0.3)",  # Subtle color
                width=1,
                dash="solid",
            )
        )
        
        # Add year labels at 5-year intervals
        for year in range(x_min, x_max + 1, 5):  # Every 5 years
            fig.add_annotation(
                x=year,
                y=y_pos,
                text=str(year),
                showarrow=False,
                font=dict(
                    size=9,  # Smaller font
                    color="rgba(100, 100, 100, 0.5)"  # Subtle color
                ),
                yshift=10
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
