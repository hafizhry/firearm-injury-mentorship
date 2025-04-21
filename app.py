# app.py

import streamlit as st
import pickle
import base64
import time
from visualization_utils import compute_positions_for_graph, calculate_mentorship_stats, create_world_view, create_author_view

# Load and encode the background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load data from pickle files - use cache_resource
@st.cache_resource
def load_data():
    """Load graph data as a resource (loaded only once)"""
    start_time = time.time()
    with open(f'source_files/G.pkl', 'rb') as pickle_file:
        G = pickle.load(pickle_file)

    with open(f'source_files/df_track_record.pkl', 'rb') as pickle_file:
        df_track_record = pickle.load(pickle_file)
    
    load_time = time.time() - start_time
    print(f"Graph data loaded in {load_time:.2f} seconds")
    return G, df_track_record

# Cache the author lookup creation
@st.cache_resource
def prepare_author_lookup(graph_id):
    """Create a lookup of authors to their nodes"""
    G, _ = load_data()  # Use the cached graph
    
    author_lookup = {}
    for n in G.nodes():
        # Extract the author name without suffixes
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
    
    return author_lookup

def main():
    # Set page config
    st.set_page_config(layout="wide", page_title="Shirtsleeves to Shirtsleeves")

    # Load data with timing information
    with st.spinner("Loading graph data..."):
        start_time = time.time()
        G, df_track_record = load_data()
        load_time = time.time() - start_time
        st.toast(f"Graph loaded in {load_time:.2f} seconds", icon="✅")

    # Precompute author relationships for faster queries
    with st.spinner("Precomputing author relationships..."):
        start_time = time.time()
        from visualization_utils import precompute_author_relationships
        precompute_author_relationships("graph_cache")
        precompute_time = time.time() - start_time
        st.toast(f"Author relationships precomputed in {precompute_time:.2f} seconds", icon="✅")

    st.markdown("""
    ### Search and Select Author

    In this visualization, each dot represents a publication by an author in the mentorship network. Authors can have two types of publications: solo publications (hollow circles) and mentored publications (filled circles). The lines connecting dots show mentorship relationships:

    - **Blue lines** indicate forward mentorship connections (mentor published earlier than mentee)
    - **Orange lines** indicate backward connections (mentee published earlier than mentor)
    - **Green dotted lines** connect an author's first solo publication to the first mentored publication, showing their career progression

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

    # Create three columns
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

    # Use the cached function to get author lookup
    with st.spinner("Preparing author data..."):
        start_time = time.time()
        author_lookup = prepare_author_lookup("graph_cache")
        prep_time = time.time() - start_time
        st.toast(f"Author data prepared in {prep_time:.2f} seconds", icon="✅")

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

    # Execute compute positions with caching
    with st.spinner("Computing node positions..."):
        start_time = time.time()
        # Use the graph_id "graph_cache" as a stable identifier for caching
        node_positions = compute_positions_for_graph("graph_cache")
        compute_time = time.time() - start_time
        st.toast(f"Positions computed in {compute_time:.2f} seconds", icon="✅")

    # Create visualization
    with st.spinner("Rendering visualization..."):
        start_time = time.time()

        if selected_mentor == "World View":
            # Use the cached world view
            fig = create_world_view("graph_cache")
        else:
            # For author-specific view, use the cached function with the mentor
            fig = create_author_view("graph_cache", final_search_term)

        render_time = time.time() - start_time
        st.toast(f"Visualization rendered in {render_time:.2f} seconds", icon="✅")

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

        # Get author name
        author_name = G.nodes[final_search_term].get('author', '')
        
        # Get precomputed relationships
        precomputed = precompute_author_relationships("graph_cache")
        
        # Get statistics (either from precomputed or calculate)
        if author_name in precomputed and 'stats' in precomputed[author_name]:
            stats = precomputed[author_name]['stats']
        else:
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

if __name__ == "__main__":
    main()