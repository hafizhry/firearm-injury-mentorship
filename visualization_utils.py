import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import time

@st.cache_data(ttl=3600) 
def compute_positions_for_graph(graph_id, grid_spacing=40):
    """
    Compute node positions using a string ID instead of G.
    This is a cache-friendly wrapper around compute_sequential_grid_positions.
    """
    # Import here to avoid circular imports
    from app import load_data
    
    # Get the graph from the cached function
    start_time = time.time()
    G, _ = load_data()
    print(f"Graph loaded for positions in {time.time() - start_time:.2f} seconds")
    
    # Call the uncached function
    start_time = time.time()
    node_positions, nodes_by_level = compute_sequential_grid_positions(G, grid_spacing)
    print(f"Positions computed in {time.time() - start_time:.2f} seconds")
    
    return node_positions, nodes_by_level

def compute_sequential_grid_positions(G, grid_spacing=40):
    """
    Compute node positions by processing complete lineages sequentially.
    Each lineage tree gets its own grid space.
    Places *_solo and *_mentored nodes for the same author on the same y-axis.
    """
    node_positions = {}
    nodes_by_level = {
        'First Gen': [], 'Second Gen': [], 'Third Gen': [],
        'Fourth Gen': [], 'Fifth Gen': [], 'Sixth Gen': [],
        'Seventh Gen': [], 'Other': []
    }

    # Create a mapping of authors to their nodes
    author_to_nodes = {}
    for node in G.nodes():
        author = G.nodes[node].get('author')
        if author:
            if author not in author_to_nodes:
                author_to_nodes[author] = []
            author_to_nodes[author].append(node)

    # Find all first gen authors (roots)
    first_gen_authors = set()
    for node, attr in G.nodes(data=True):
        if attr.get('level') == 'First Gen':
            author = attr.get('author')
            if author:
                first_gen_authors.add(author)

    # Convert to list and sort by publication year (use the earliest publication of author's nodes)
    first_gen_list = list(first_gen_authors)
    first_gen_list.sort(
        key=lambda author: min(G.nodes[node]['first_publication_year']
                               for node in author_to_nodes[author]),
        reverse=True
    )

    def process_lineage(root_author, current_base_row):
        """
        Process a complete lineage tree starting from a root author.
        Returns the maximum row used in this lineage.
        """
        # Dictionary to track occupied grid points in this lineage
        occupied_grid = set()
        lineage_positions = {}
        author_rows = {}  # Maps authors to their assigned rows

        def place_author_and_descendants(author, base_row, visited=None):
            if visited is None:
                visited = set()

            if author in visited:
                return base_row

            visited.add(author)

            # Find all nodes for this author
            author_nodes = author_to_nodes.get(author, [])
            if not author_nodes:
                return base_row

            # Find first available row for this author
            author_min_year = min(G.nodes[node]['first_publication_year'] for node in author_nodes)
            current_row = base_row
            while any((author_min_year, current_row) in occupied_grid for node in author_nodes):
                current_row += 1

            # Assign row to this author
            author_rows[author] = current_row

            # Place all nodes for this author on the same row (y-axis)
            for node in author_nodes:
                year = G.nodes[node]['first_publication_year']
                level = G.nodes[node]['level']

                lineage_positions[node] = (year, current_row * grid_spacing)
                occupied_grid.add((year, current_row))
                nodes_by_level[level].append(node)

            # Process all descendants (by author)
            max_descendant_row = current_row

            # Get all unique descendant authors
            descendant_authors = set()
            for node in author_nodes:
                for descendant_node in G.successors(node):
                    descendant_author = G.nodes[descendant_node].get('author')
                    if descendant_author and descendant_author != author and descendant_author not in visited:
                        descendant_authors.add(descendant_author)

            # Sort descendant authors by their earliest publication year
            descendant_authors = list(descendant_authors)
            descendant_authors.sort(
                key=lambda a: min(G.nodes[n]['first_publication_year']
                                 for n in author_to_nodes.get(a, []))
            )

            for descendant_author in descendant_authors:
                if descendant_author not in visited:
                    descendant_row = place_author_and_descendants(
                        descendant_author,
                        max_descendant_row + 1,
                        visited
                    )
                    max_descendant_row = max(max_descendant_row, descendant_row)

            return max_descendant_row

        # Process the entire lineage by author
        max_row_used = place_author_and_descendants(root_author, current_base_row)

        # Update global positions
        node_positions.update(lineage_positions)

        return max_row_used

    # Process each first gen author and their complete lineage
    current_base_row = 0
    for first_gen_author in first_gen_list:
        # Process this lineage tree
        max_row = process_lineage(first_gen_author, current_base_row)

        # Start next lineage at new base row with some padding
        current_base_row = max_row + 2  # Add padding between lineages

    return node_positions, nodes_by_level

@st.cache_data
def create_figure_cached(graph_id, selected_mentor=None):
    """Cache-friendly wrapper around create_figure"""
    from app import load_data
    G, _ = load_data()
    
    # Get positions
    node_positions, _ = compute_positions_for_graph(graph_id)
    
    # Create the figure
    return create_figure(G, node_positions, selected_mentor)

def create_figure(G, node_positions, selected_mentor=None):
    """
    Create a Plotly figure with colored edges based on temporal direction and relationship type.
    Supports both solo and mentored nodes for the same author on the same y-axis.
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

    # Filter out nodes without positions
    plot_nodes = [n for n in G.nodes() if n in node_positions]

    # If we have a selected mentor, get their lineage
    lineage = set()
    if selected_mentor:
        lineage = set(nx.ancestors(G, selected_mentor) | nx.descendants(G, selected_mentor) | {selected_mentor})

    node_attrs = [
        (n, node_positions[n][0], node_positions[n][1], G.nodes[n].get('level', 'Seventh Gen'))
        for n in plot_nodes
    ]

    # Separate edges by relationship type and direction
    forward_direct_x = []
    forward_direct_y = []
    backward_direct_x = []
    backward_direct_y = []

    # Keep separate lists for self-connections
    self_connection_x = []
    self_connection_y = []
    self_connection_hover = []

    for u, v, data in G.edges(data=True):
        if u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            year_u = G.nodes[u]['first_publication_year']
            year_v = G.nodes[v]['first_publication_year']

            # Determine if this is a self-connection (solo to mentored for same author)
            u_author = G.nodes[u].get('author')
            v_author = G.nodes[v].get('author')
            is_self_connection = u_author and v_author and u_author == v_author

            if is_self_connection:
                # Determine which node is solo and which is mentored
                if G.nodes[u].get('type') == 'solo_publication' and G.nodes[v].get('type') == 'mentored_publication':
                    solo_node = u
                    mentored_node = v
                elif G.nodes[v].get('type') == 'solo_publication' and G.nodes[u].get('type') == 'mentored_publication':
                    solo_node = v
                    mentored_node = u
                else:
                    # If types can't be determined, just continue
                    continue

                author_name = u_author  # The author's name

                # Get the mentors from the mentored node's predecessors
                mentors = []
                for pred in G.predecessors(mentored_node):
                    # Skip if it's the author's own solo node or self
                    if pred != solo_node and G.nodes[pred].get('author') != author_name:
                        mentor_name = G.nodes[pred].get('author', 'Unknown')
                        mentors.append(mentor_name)

                # Create hover text
                solo_year = G.nodes[solo_node].get('first_publication_year', 'Unknown')
                mentored_year = G.nodes[mentored_node].get('first_publication_year', 'Unknown')

                if mentors:
                    hover_text = f"{author_name} made self progression from solo authoring ({solo_year}) to be mentored by {', '.join(mentors)} ({mentored_year})"
                else:
                    hover_text = f"{author_name} made self progression from solo authoring ({solo_year}) to be mentored ({mentored_year})"

                # Store the self-connection for later
                self_connection_x.extend([x0, x1, None])
                self_connection_y.extend([y0, y1, None])
                self_connection_hover.extend([hover_text, hover_text, None])

                continue

            # If target year is greater than or equal to source year, it's forward
            if year_v >= year_u:
                forward_direct_x.extend([x0, x1, None])
                forward_direct_y.extend([y0, y1, None])
            else:
                backward_direct_x.extend([x0, x1, None])
                backward_direct_y.extend([y0, y1, None])

    # Create separate traces for each edge type
    edge_traces = []

    # Direct mentorship (forward)
    if forward_direct_x:
        edge_traces.append(go.Scatter(
            x=forward_direct_x,
            y=forward_direct_y,
            line=dict(width=1.5, color='#1f77b4'),
            marker=dict(
                size=9,
                symbol="arrow",
                angleref="previous",
                color='#1f77b4',
                opacity=0.85
            ),
            hoverinfo='none',
            mode='lines+markers',
            name='Direct Mentorship (Forward)',
            showlegend=False,
            opacity=0.7
        ))

    # Direct mentorship (backward)
    if backward_direct_x:
        edge_traces.append(go.Scatter(
            x=backward_direct_x,
            y=backward_direct_y,
            line=dict(width=1.5, color='orange'),
            marker=dict(
                size=9,
                symbol="arrow",
                angleref="previous",
                color='orange',
                opacity=0.85
            ),
            hoverinfo='none',
            mode='lines+markers',
            name='Direct Mentorship (Backward)',
            showlegend=False,
            opacity=0.7
        ))

    # Create node traces
    all_levels = ['First Gen', 'Second Gen', 'Third Gen', 'Fourth Gen', 'Fifth Gen', 'Sixth Gen', 'Seventh Gen', 'Other']
    node_traces = []

    # Create a mapping to show node types differently
    node_shapes = {
        'solo_publication': 'circle-open',
        'mentored_publication': 'circle'
    }

    for level in all_levels:
        level_nodes = [(n, x, y) for (n, x, y, lvl) in node_attrs if lvl == level]
        if not level_nodes:
            continue

        # Group nodes by type
        solo_nodes = [(n, x, y) for (n, x, y) in level_nodes if G.nodes[n].get('type') == 'solo_publication']
        mentored_nodes = [(n, x, y) for (n, x, y) in level_nodes if G.nodes[n].get('type') == 'mentored_publication']
        other_nodes = [(n, x, y) for (n, x, y) in level_nodes if G.nodes[n].get('type') not in ['solo_publication', 'mentored_publication']]

        # Create traces for each node type
        for node_type, nodes in [
            ('solo_publication', solo_nodes),
            ('mentored_publication', mentored_nodes),
            (None, other_nodes)
        ]:
            if not nodes:
                continue

            node_x = [x for (_, x, _) in nodes]
            node_y = [y for (_, _, y) in nodes]

            # Create hover text with relevant node info
            node_text = [
                f"Author: {G.nodes[n].get('author', 'Unknown')}<br>"
                f"Publication Type: {G.nodes[n].get('type', 'Unknown').replace('_', ' ').title()}<br>"
                f"Generation: {level}<br>"
                f"Publication Year: {G.nodes[n].get('first_publication_year', 'Unknown')}<br>"
                f"Title: {G.nodes[n].get('first_title', 'Unknown')}"
                for (n, _, _) in nodes
            ]

            marker_symbol = node_shapes.get(node_type, 'circle')
            display_name = f"{level} - {node_type.replace('_', ' ').title()}" if node_type else level

            # Set opacity based on whether node is in lineage
            opacity = [1.0 if n in lineage else 0.2 for n, _, _ in nodes] if selected_mentor else [1.0] * len(nodes)
            size = [12 if n == selected_mentor else 9 for n, _, _ in nodes] if selected_mentor else [8] * len(nodes)

            trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=size,
                    color=level_colors[level],
                    symbol=marker_symbol,
                    line_width=1,
                    opacity=opacity
                ),
                text=[G.nodes[n].get('author', '') for (n, _, _) in nodes],  # Use author names for text
                textposition='top center',
                textfont=dict(size=10, color='rgba(0,0,0,0)'),
                hovertext=node_text,
                hoverinfo='text',
                name=display_name,
                legendgroup=level
            )
            node_traces.append(trace)

    # Create a flat list of all traces
    all_traces = edge_traces + node_traces

    # Create the figure
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            hovermode='closest',
            margin=dict(b=10, l=0, r=0, t=45),
            paper_bgcolor='#F0F8FF',
            plot_bgcolor='#F0F8FF',
            xaxis=dict(
                title=dict(
                    text='Publication Year',
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
            yaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
            height=900,
            showlegend=False
        )
    )

    # Replace the self-connection trace code with this improved version
    if self_connection_x:
        # Process self-connection edges to avoid duplicated points
        processed_edges = set()
        unique_self_connection_x = []
        unique_self_connection_y = []
        unique_self_connection_hover = []

        i = 0
        while i < len(self_connection_x) - 2:  # Process line segments (skipping None values)
            x0, x1 = self_connection_x[i], self_connection_x[i+1]
            y0, y1 = self_connection_y[i], self_connection_y[i+1]
            hover = self_connection_hover[i]

            # Create an edge key for deduplication
            edge_key = ((x0, y0), (x1, y1))
            rev_edge_key = ((x1, y1), (x0, y0))

            # Only process if this edge hasn't been seen before
            if edge_key not in processed_edges and rev_edge_key not in processed_edges:
                processed_edges.add(edge_key)
                unique_self_connection_x.extend([x0, x1, None])
                unique_self_connection_y.extend([y0, y1, None])
                unique_self_connection_hover.extend([hover, hover, None])

            i += 3  # Skip to next line segment (after the None)

        # Create denser points for better hover experience
        dense_x = []
        dense_y = []
        dense_hover = []

        i = 0
        while i < len(unique_self_connection_x) - 2:  # Process line segments (skipping None values)
            x0, x1 = unique_self_connection_x[i], unique_self_connection_x[i+1]
            y0, y1 = unique_self_connection_y[i], unique_self_connection_y[i+1]
            hover = unique_self_connection_hover[i]

            # Create intermediate points along this segment
            for j in range(10):  # Add 10 points along each segment
                t = j / 10
                dense_x.append(x0 + t * (x1 - x0))
                dense_y.append(y0 + t * (y1 - y0))
                dense_hover.append(hover)

            i += 3  # Skip to next line segment (after the None)

        # Add the visible dotted line
        fig.add_trace(go.Scatter(
            x=unique_self_connection_x,
            y=unique_self_connection_y,
            line=dict(width=2.0, color='green', dash='dot'),
            hoverinfo='none',  # No hover on the line itself
            mode='lines',
            name='Author Development (Solo→Mentored)',
            showlegend=False,
            opacity=0.8
        ))

        # Add truly invisible hover points with larger click area
        fig.add_trace(go.Scatter(
            x=dense_x,
            y=dense_y,
            mode="none",
            marker=dict(
                size=9,
                opacity=0,
                color="rgba(0,0,0,0)",  # Completely transparent color
                line=dict(width=0)  # No border
            ),
            hoverinfo="text",
            hovertext=dense_hover,
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#4caf50",
                font_size=12,
                font_family="Arial",
                font_color="white"
            )
        ))

        # Update layout for better hover behavior
        fig.update_layout(
            hovermode='closest',
            hoverdistance=20  # Increase hover detection distance
        )

    return fig, edge_traces, node_traces

def highlight_and_zoom_to_mentor(fig, G, node_positions, selected_mentor, df_track_record=None, author_nodes=None):
    """
    Highlight an author and their lineage, then zoom to their position
    """
    if not selected_mentor:
        # If in world view, remove all annotations
        fig.update_layout(annotations=[])
        return fig

    # Get the author name for the selected mentor
    author_name = G.nodes[selected_mentor].get('author', '')

    # If author_nodes not provided, get all nodes for this author
    if author_nodes is None:
        author_nodes = [n for n in G.nodes() if G.nodes[n].get('author') == author_name]

    # Get ancestors and descendants for all nodes of this author
    ancestors = set()
    descendants = set()
    for node in author_nodes:
        ancestors.update(nx.ancestors(G, node))
        descendants.update(nx.descendants(G, node))

    lineage = list(ancestors.union(descendants).union(set(author_nodes)))

    # Get position of the selected mentor
    if selected_mentor in node_positions:
        # Calculate the bounds of the lineage
        x_positions = [node_positions[n][0] for n in lineage if n in node_positions]
        y_positions = [node_positions[n][1] for n in lineage if n in node_positions]

        x_min, x_max = min(x_positions), max(x_positions)
        y_min, y_max = min(y_positions), max(y_positions)

        # Add padding
        x_padding = max(10, (x_max - x_min) * 0.4)
        y_padding = 70

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

        # Process each trace type correctly
        for i, trace in enumerate(fig.data):
            # Handle node traces
            if hasattr(trace, 'marker') and (trace.mode == 'markers' or trace.mode == 'markers+text'):
                # Get the node positions for this trace
                node_x = trace.x
                node_y = trace.y

                # We need to map these coordinates back to nodes
                coords_to_nodes = {}
                for node in G.nodes():
                    if node in node_positions:
                        coords_to_nodes[node_positions[node]] = node

                # Create lists for opacity
                opacity_list = []
                size_list = []
                text_list = []

                for j in range(len(node_x)):
                    # Try to identify which node this is from its coordinates
                    node_pos = (node_x[j], node_y[j])
                    node_name = None

                    # Look for a node with matching coordinates
                    for pos, node in coords_to_nodes.items():
                        if abs(pos[0] - node_pos[0]) < 0.001 and abs(pos[1] - node_pos[1]) < 0.001:
                            node_name = node
                            break

                    if node_name in lineage:
                        opacity_list.append(1.0)
                        # Highlight author's own nodes more
                        if node_name in author_nodes:
                            size_list.append(12)
                        else:
                            size_list.append(9)
                        # Use author attribute for text, not node name
                        if node_name:
                            text_list.append(G.nodes[node_name].get('author', ''))
                        else:
                            text_list.append('')
                    else:
                        opacity_list.append(0.2)
                        size_list.append(7)
                        text_list.append('')

                # Update marker properties
                trace.marker.opacity = opacity_list
                trace.marker.size = size_list

                # Update text visibility
                text_colors = []
                for j in range(len(node_x)):
                    node_pos = (node_x[j], node_y[j])
                    node_name = None

                    # Look for a node with matching coordinates
                    for pos, node in coords_to_nodes.items():
                        if abs(pos[0] - node_pos[0]) < 0.001 and abs(pos[1] - node_pos[1]) < 0.001:
                            node_name = node
                            break

                    if node_name in lineage:
                        text_colors.append('black')  # Show text for lineage
                    else:
                        text_colors.append('rgba(0,0,0,0)')  # Hide text for others

                # Set text values and style
                trace.text = text_list
                trace.mode = 'markers+text'
                trace.textposition = 'top center'
                trace.textfont.color = text_colors

        # Add career milestones if df_track_record is provided
        if df_track_record is not None:
            # Filter milestones for the selected author (using author name)
            df_mentor_milestones = df_track_record[df_track_record['author_name'] == author_name]

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
                margin=dict(t=60, b=10)
            )

        import streamlit as st
        st.success(f"Showing lineage for {author_name}")

    return fig

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