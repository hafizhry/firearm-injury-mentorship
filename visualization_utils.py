import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

def compute_sequential_grid_positions(G, grid_spacing=40):
    """
    Compute node positions by processing complete lineages sequentially.
    Each lineage tree gets its own grid space.
    Places *solo and *_mentored nodes for the same author on the same y-axis.
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

def create_figure(G, node_positions, nodes_by_level, selected_mentor=None):
    """
    Create a Plotly figure with colored edges based on temporal direction and relationship type.
    Supports both solo and mentored nodes for the same author on the same y-axis.
    
    Args:
        G: NetworkX graph
        node_positions: Dictionary of node positions
        nodes_by_level: Dictionary of nodes grouped by level
        selected_mentor: Optional selected mentor node to highlight
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
    # For solo/mentored connections between the same author, we'll use a special edge type
    forward_direct_x = []
    forward_direct_y = []
    forward_adopted_x = []
    forward_adopted_y = []
    backward_direct_x = []
    backward_direct_y = []
    backward_adopted_x = []
    backward_adopted_y = []
    self_connection_x = []
    self_connection_y = []
    annotations = []

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
                # Special styling for connections between an author's own nodes
                self_connection_x.extend([x0, x1, None])
                self_connection_y.extend([y0, y1, None])
                continue

            # Get relationship type (default to direct if not specified)
            relationship_type = data.get('relationship_type', 'direct_mentorship')

            # Determine if relationship is adopted or direct
            is_adopted = relationship_type in ['secondary_mentor', 'adopted_mentorship', 'self_development']

            # If target year is greater than or equal to source year, it's forward
            if year_v >= year_u:
                if not is_adopted:  # Direct mentorship
                    forward_direct_x.extend([x0, x1, None])
                    forward_direct_y.extend([y0, y1, None])
                else:  # Adopted mentorship
                    forward_adopted_x.extend([x0, x1, None])
                    forward_adopted_y.extend([y0, y1, None])
                    # Add first mentored publication info for forward adopted edges
                    if selected_mentor and (u in lineage or v in lineage):
                        first_mentored_pub = G.nodes[v].get('first_mentored_pub', {})
                        if first_mentored_pub:
                            pub_year = first_mentored_pub.get('pub_year', '')
                            title = first_mentored_pub.get('title', '')
                            text = f"{u} mentored {v} in {pub_year} after solo pub"
                            annotations.append({
                                'x': (x0 + x1) / 2,
                                'y': (y0 + y1) / 2,
                                'text': text,
                                'showarrow': False,
                                'textangle': 0,
                                'font': {'color': '#1f77b4', 'size': 9},
                                'bgcolor': 'white',
                                'bordercolor': '#1f77b4',
                                'borderwidth': 1,
                                'borderpad': 2,
                                'opacity': 0.9
                            })
            else:
                if not is_adopted:  # Direct mentorship (backward in time)
                    backward_direct_x.extend([x0, x1, None])
                    backward_direct_y.extend([y0, y1, None])
                else:  # Adopted mentorship (backward in time)
                    backward_adopted_x.extend([x0, x1, None])
                    backward_adopted_y.extend([y0, y1, None])
                    # Add first mentored publication info for backward adopted edges
                    if selected_mentor and (u in lineage or v in lineage):
                        first_mentored_pub = G.nodes[v].get('first_mentored_pub', {})
                        if first_mentored_pub:
                            pub_year = first_mentored_pub.get('pub_year', '')
                            title = first_mentored_pub.get('title', '')
                            text = f"{u} mentored {v} in {pub_year} after solo pub"
                            annotations.append({
                                'x': (x0 + x1) / 2,
                                'y': (y0 + y1) / 2,
                                'text': text,
                                'showarrow': False,
                                'textangle': 0,
                                'font': {'color': 'orange', 'size': 9},
                                'bgcolor': 'white',
                                'bordercolor': 'orange',
                                'borderwidth': 1,
                                'borderpad': 2,
                                'opacity': 0.9
                            })

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

    # Adopted mentorship (forward)
    if forward_adopted_x:
        edge_traces.append(go.Scatter(
            x=forward_adopted_x,
            y=forward_adopted_y,
            line=dict(width=1.5, color='#1f77b4', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Adopted Mentorship (Forward)',
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

    # Adopted mentorship (backward)
    if backward_adopted_x:
        edge_traces.append(go.Scatter(
            x=backward_adopted_x,
            y=backward_adopted_y,
            line=dict(width=1.5, color='orange', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Adopted Mentorship (Backward)',
            showlegend=False,
            opacity=0.7
        ))

    # Self connections (solo to mentored)
    if self_connection_x:
        edge_traces.append(go.Scatter(
            x=self_connection_x,
            y=self_connection_y,
            line=dict(width=1.5, color='green', dash='dot'),
            hoverinfo='none',
            mode='lines',
            name='Author Development (Solo→Mentored)',
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
                f"Node Type: {G.nodes[n].get('type', 'Unknown')}<br>"
                f"Level: {level}<br>"
                f"First Publication Year: {G.nodes[n].get('first_publication_year', 'Unknown')}<br>"
                f"First Title: {G.nodes[n].get('first_title', 'Unknown')}<br>"
                # + (f"Predecessors: {', '.join(list(G.predecessors(n)))}<br>" if list(G.predecessors(n)) else "")
                # + (f"Clusters: {G.nodes[n].get('cluster_keywords', 'Unknown')}<br>" if G.nodes[n].get('cluster_keywords') else "")
                # + (f"Gender: {G.nodes[n].get('gender', 'Unknown')}" if G.nodes[n].get('gender') else "")
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
                text=[n for (n, _, _) in nodes],
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

    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            hovermode='closest',
            margin=dict(b=10, l=0, r=0, t=45),
            paper_bgcolor='#F0F8FF',
            plot_bgcolor='#F0F8FF',
            xaxis=dict(
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
            # legend=dict(
            #     title="Node Level & Type",
            #     orientation="h",
            #     yanchor="top",
            #     y=1.2,
            #     xanchor="right",
            #     x=1
            # )
        )
    )
    
    # Add the collected annotations to the figure
    if annotations:
        fig.update_layout(annotations=annotations)
    
    return fig, edge_traces, node_traces

def get_lineage_nodes(G, node_name):
    """
    Get the ancestors and descendants of a node in the graph.
    Also include nodes from the same author.
    """
    ancestors = list(nx.ancestors(G, node_name))
    descendants = list(nx.descendants(G, node_name))

    # Include all nodes from the same author
    same_author_nodes = []
    selected_author = G.nodes[node_name].get('author')
    if selected_author:
        for node in G.nodes():
            if node != node_name and G.nodes[node].get('author') == selected_author:
                same_author_nodes.append(node)

    lineage_nodes = ancestors + descendants + [node_name] + same_author_nodes
    return ancestors, descendants, lineage_nodes

def highlight_lineage(fig, G, node_name, node_positions, edge_traces, node_traces):
    """
    Highlight the lineage of a node in the graph.
    Ensures that both solo and mentored nodes for the same author are highlighted.
    """
    ancestors, descendants, lineage = get_lineage_nodes(G, node_name)

    if node_name not in node_positions:
        print(f"Node '{node_name}' position not found.")
        return fig

    # Create a new figure based on the current one
    fig_copy = go.Figure(fig)

    # Calculate the bounds of the lineage
    x_positions = [node_positions[n][0] for n in lineage if n in node_positions]
    y_positions = [node_positions[n][1] for n in lineage if n in node_positions]

    if not x_positions or not y_positions:
        print(f"No position data for lineage of {node_name}")
        return fig

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
    fig_copy.update_layout(
        title=dict(
            text=f"Lineage of '{G.nodes[node_name].get('author', node_name)}'",
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
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
        height=900
    )

    # Separate lineage edges by relationship type
    direct_edge_x = []
    direct_edge_y = []
    adopted_edge_x = []
    adopted_edge_y = []
    self_edge_x = []
    self_edge_y = []

    for u, v, data in G.edges(data=True):
        if u in lineage and v in lineage and u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]

            # Check if this is a self-connection (same author)
            u_author = G.nodes[u].get('author')
            v_author = G.nodes[v].get('author')
            is_self_connection = u_author and v_author and u_author == v_author

            if is_self_connection:
                self_edge_x.extend([x0, x1, None])
                self_edge_y.extend([y0, y1, None])
                continue

            # Get relationship type
            relationship_type = data.get('relationship_type', 'direct_mentorship')

            # Determine if relationship is adopted or direct
            is_adopted = relationship_type in ['secondary_mentor', 'adopted_mentorship', 'self_development']

            if is_adopted:
                adopted_edge_x.extend([x0, x1, None])
                adopted_edge_y.extend([y0, y1, None])
            else:
                direct_edge_x.extend([x0, x1, None])
                direct_edge_y.extend([y0, y1, None])

    edge_highlight_traces = []

    # Direct mentorship connections (solid line)
    if direct_edge_x:
        edge_highlight_traces.append(go.Scatter(
            x=direct_edge_x,
            y=direct_edge_y,
            line=dict(width=2.5, color='rgba(50, 50, 50, 0.9)'),
            hoverinfo='none',
            mode='lines',
            name='Direct Mentorship',
            showlegend=False
        ))

    # Adopted mentorship connections (dashed line)
    if adopted_edge_x:
        edge_highlight_traces.append(go.Scatter(
            x=adopted_edge_x,
            y=adopted_edge_y,
            line=dict(width=2.5, color='rgba(50, 50, 50, 0.8)', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Adopted Mentorship',
            showlegend=False
        ))

    # Self connections (solo to mentored)
    if self_edge_x:
        edge_highlight_traces.append(go.Scatter(
            x=self_edge_x,
            y=self_edge_y,
            line=dict(width=2.5, color='green', dash='dot'),
            hoverinfo='none',
            mode='lines',
            name='Author Development',
            showlegend=False
        ))

    # Create node hover text
    node_shapes = {
        'solo_publication': 'circle-open',
        'mentored_publication': 'circle'
    }

    # Highlighted nodes
    level_node_x = [node_positions[n][0] for n in lineage if n in node_positions]
    level_node_y = [node_positions[n][1] for n in lineage if n in node_positions]

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

    # Group lineage nodes by type
    for node_type in ['solo_publication', 'mentored_publication', None]:
        type_nodes = [n for n in lineage if n in node_positions and
                    G.nodes[n].get('type') == node_type or
                    (node_type is None and G.nodes[n].get('type') not in ['solo_publication', 'mentored_publication'])]

        if not type_nodes:
            continue

        node_x = [node_positions[n][0] for n in type_nodes]
        node_y = [node_positions[n][1] for n in type_nodes]

        marker_symbol = node_shapes.get(node_type, 'circle')
        type_name = f"{node_type.replace('_', ' ').title()} Nodes" if node_type else "Other Nodes"

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=10,
                color=[level_colors.get(G.nodes[n].get('level', 'Other'), 'grey') for n in type_nodes],
                symbol=marker_symbol,
                line=dict(width=2, color='black')
            ),
            text=[G.nodes[n].get('author', '') for n in type_nodes],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            hovertext=[
                f"Author: {G.nodes[n].get('author', 'Unknown')}<br>"
                f"Node Type: {G.nodes[n].get('type', 'Unknown')}<br>"
                f"Level: {G.nodes[n].get('level', 'Unknown')}<br>"
                f"First Publication Year: {G.nodes[n].get('first_publication_year', 'Unknown')}<br>"
                + (f"Predecessors: {', '.join(list(G.predecessors(n)))}<br>" if list(G.predecessors(n)) else "")
                for n in type_nodes
            ],
            name=type_name,
            showlegend=False
        )
        fig_copy.add_trace(trace)

    # Highlight the selected node specially
    selected_node_trace = go.Scatter(
        x=[node_positions[node_name][0]],
        y=[node_positions[node_name][1]],
        mode='markers',
        marker=dict(
            size=15,
            color='yellow',
            symbol=node_shapes.get(G.nodes[node_name].get('type'), 'circle'),
            line=dict(width=3, color='black')
        ),
        hoverinfo='text',
        hovertext=[
            f"SELECTED: {G.nodes[node_name].get('author', node_name)}<br>"
            f"Node Type: {G.nodes[node_name].get('type', 'Unknown')}<br>"
            f"Level: {G.nodes[node_name].get('level', 'Unknown')}<br>"
            f"First Publication Year: {G.nodes[node_name].get('first_publication_year', 'Unknown')}"
        ],
        name="Selected Node",
        showlegend=False
    )

    # Add all the traces to the figure
    for trace in edge_highlight_traces:
        fig_copy.add_trace(trace)

    fig_copy.add_trace(selected_node_trace)

    print(f"Showing lineage for {node_name} (Author: {G.nodes[node_name].get('author', 'Unknown')})")
    print(f"Ancestors: {len(ancestors)}, Descendants: {len(descendants)}, Total lineage: {len(lineage)}")

    return fig_copy

def show_lineage(node_name, G=None, node_positions=None, nodes_by_level=None, df_track_record=None):
    """
    Show the lineage visualization for a selected node
    Ensures that both solo and mentored nodes for the same author are included
    """
    # Handle the case where globals are not provided
    if G is None or node_positions is None:
        print("Error: Graph or node positions not provided.")
        return None

    if not node_name:
        # Create a new default figure
        fig, edge_traces, node_traces = create_figure(G, node_positions, nodes_by_level)
        return fig

    if node_name not in G.nodes():
        print(f"Node '{node_name}' not found in the graph.")
        return None

    # Create a base figure first
    fig, edge_traces, node_traces = create_figure(G, node_positions, nodes_by_level)

    # Highlight the lineage
    fig = highlight_lineage(fig, G, node_name, node_positions, edge_traces, node_traces)

    # Display
    fig.update_layout(
        height=900,
        # autosize=True
    )

    return fig

def highlight_author_lineage(G, author_name, author_nodes, node_positions):
    """
    Highlight all nodes and connections for an author in the graph, maintaining original colors 
    and showing author names for ALL related nodes.
    """
    # First, create base figure
    fig, edge_traces, node_traces = create_figure(G, node_positions, nodes_by_level)
    
    # Get all ancestors and descendants (complete lineage)
    combined_ancestors = set()
    combined_descendants = set()
    combined_lineage = set(author_nodes)  # Start with the author's own nodes
    
    for node in author_nodes:
        # Get complete lineage for this specific node
        ancestors = set(nx.ancestors(G, node))
        descendants = set(nx.descendants(G, node))
        
        combined_ancestors.update(ancestors)
        combined_descendants.update(descendants)
        combined_lineage.update(ancestors)
        combined_lineage.update(descendants)
    
    # Update figure title and axes
    x_positions = [node_positions[n][0] for n in combined_lineage if n in node_positions]
    y_positions = [node_positions[n][1] for n in combined_lineage if n in node_positions]
    
    if not x_positions or not y_positions:
        print(f"No position data for lineage of {author_name}")
        return fig
    
    # Calculate view bounds with padding
    x_min, x_max = min(x_positions), max(x_positions)
    y_min, y_max = min(y_positions), max(y_positions)
    
    # Add padding
    x_padding = max(10, (x_max - x_min) * 0.2)  # Reduced padding
    y_padding = 50
    
    x_range_size = x_max - x_min
    if x_range_size < 100:
        x_center = (x_max + x_min) / 2
        x_min = x_center - 10
        x_max = x_center + 10
    
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]
    
    # Update figure layout
    fig.update_layout(
        title=dict(
            text=f"Lineage of '{author_name}'",
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
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
        height=800,
        margin=dict(l=20, r=20, t=100, b=20)  # Reduced margins
    )
    
    # Remove all existing traces
    fig.data = []
    
    # Add edges with original colors - just filter to show only lineage edges
    forward_direct_x = []
    forward_direct_y = []
    forward_adopted_x = []
    forward_adopted_y = []
    backward_direct_x = []
    backward_direct_y = []
    backward_adopted_x = []
    backward_adopted_y = []
    self_connection_x = []
    self_connection_y = []
    
    for u, v, data in G.edges(data=True):
        if u in combined_lineage and v in combined_lineage and u in node_positions and v in node_positions:
            x0, y0 = node_positions[u]
            x1, y1 = node_positions[v]
            year_u = G.nodes[u].get('first_publication_year', 0)
            year_v = G.nodes[v].get('first_publication_year', 0)

            # Determine if this is a self-connection (solo to mentored for same author)
            u_author = G.nodes[u].get('author')
            v_author = G.nodes[v].get('author')
            is_self_connection = u_author and v_author and u_author == v_author

            if is_self_connection:
                # Special styling for connections between an author's own nodes
                self_connection_x.extend([x0, x1, None])
                self_connection_y.extend([y0, y1, None])
                continue

            # Get relationship type (default to direct if not specified)
            relationship_type = data.get('relationship_type', 'direct_mentorship')

            # Determine if relationship is adopted or direct
            is_adopted = relationship_type in ['secondary_mentor', 'adopted_mentorship', 'self_development']

            # If target year is greater than or equal to source year, it's forward
            if year_v >= year_u:
                if not is_adopted:  # Direct mentorship
                    forward_direct_x.extend([x0, x1, None])
                    forward_direct_y.extend([y0, y1, None])
                else:  # Adopted mentorship
                    forward_adopted_x.extend([x0, x1, None])
                    forward_adopted_y.extend([y0, y1, None])
            else:
                if not is_adopted:  # Direct mentorship (backward in time)
                    backward_direct_x.extend([x0, x1, None])
                    backward_direct_y.extend([y0, y1, None])
                else:  # Adopted mentorship (backward in time)
                    backward_adopted_x.extend([x0, x1, None])
                    backward_adopted_y.extend([y0, y1, None])
    
    # Add edge traces with original colors
    # Direct mentorship (forward)
    if forward_direct_x:
        fig.add_trace(go.Scatter(
            x=forward_direct_x,
            y=forward_direct_y,
            line=dict(width=2, color='#1f77b4'),
            marker=dict(
                size=9,
                symbol="arrow",
                angleref="previous",
                color='#1f77b4',
                opacity=0.85
            ),
            hoverinfo='none',
            mode='lines',
            name='Direct Mentorship (Forward)',
            showlegend=False,
            opacity=0.7
        ))

    # Adopted mentorship (forward)
    if forward_adopted_x:
        fig.add_trace(go.Scatter(
            x=forward_adopted_x,
            y=forward_adopted_y,
            line=dict(width=2, color='#1f77b4', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Adopted Mentorship (Forward)',
            showlegend=False,
            opacity=0.7
        ))

    # Direct mentorship (backward)
    if backward_direct_x:
        fig.add_trace(go.Scatter(
            x=backward_direct_x,
            y=backward_direct_y,
            line=dict(width=2, color='orange'),
                marker=dict(
                size=9,
                symbol="arrow",
                angleref="previous",
                color='orange',
                opacity=0.85
            ),
            hoverinfo='none',
            mode='lines',
            name='Direct Mentorship (Backward)',
            showlegend=False,
            opacity=0.7
        ))

    # Adopted mentorship (backward)
    if backward_adopted_x:
        fig.add_trace(go.Scatter(
            x=backward_adopted_x,
            y=backward_adopted_y,
            line=dict(width=2, color='orange', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Adopted Mentorship (Backward)',
            showlegend=False,
            opacity=0.7
        ))

    # Self connections (solo to mentored)
    if self_connection_x:
        fig.add_trace(go.Scatter(
            x=self_connection_x,
            y=self_connection_y,
            line=dict(width=2, color='green', dash='dot'),
            hoverinfo='none',
            mode='lines',
            name='Author Development (Solo→Mentored)',
            showlegend=False,
            opacity=0.7
        ))
    
    # Create node hover text
    node_shapes = {
        'solo_publication': 'circle-open',
        'mentored_publication': 'circle'
    }
    
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
    
    # Create list of all nodes to display
    all_display_nodes = list(combined_lineage)
    
    # Get a mapping of nodes to their authors
    node_to_author = {}
    for node in all_display_nodes:
        node_to_author[node] = G.nodes[node].get('author', '')
    
    # Group authors by their generation for more organized display
    authors_by_generation = {}
    for node in all_display_nodes:
        author = node_to_author[node]
        level = G.nodes[node].get('level', 'Unknown')
        if level not in authors_by_generation:
            authors_by_generation[level] = set()
        authors_by_generation[level].add(author)
    
    # Group all lineage nodes by type
    for node_type in ['solo_publication', 'mentored_publication', None]:
        type_nodes = [n for n in all_display_nodes if n in node_positions and 
                    (G.nodes[n].get('type') == node_type or 
                     (node_type is None and G.nodes[n].get('type') not in ['solo_publication', 'mentored_publication']))]
        
        if not type_nodes:
            continue
        
        node_x = [node_positions[n][0] for n in type_nodes]
        node_y = [node_positions[n][1] for n in type_nodes]
        
        # Get display information
        marker_symbol = node_shapes.get(node_type, 'circle')
        type_name = f"{node_type.replace('_', ' ').title()} Nodes" if node_type else "Other Nodes"
        
        # Identify the main author's nodes
        is_selected_author = [G.nodes[n].get('author') == author_name for n in type_nodes]
        
        # Prepare text labels - show names for ALL nodes in lineage
        node_texts = [G.nodes[n].get('author', '') for n in type_nodes]
        
        # Size and styling based on node role
        node_sizes = []
        line_widths = []
        for node in type_nodes:
            if G.nodes[node].get('author') == author_name:
                node_sizes.append(12)  # Larger for selected author
                line_widths.append(2)  # Thicker border
            elif node in combined_ancestors:
                node_sizes.append(10)  # Medium for ancestors
                line_widths.append(1.5)
            elif node in combined_descendants:
                node_sizes.append(10)  # Medium for descendants
                line_widths.append(1.5)
            else:
                node_sizes.append(8)  # Standard for other lineage nodes
                line_widths.append(1)
        
        # Get colors for nodes based on their generation
        node_colors = [level_colors.get(G.nodes[n].get('level', 'Other'), 'grey') for n in type_nodes]
        
        # Add node trace
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=marker_symbol,
                line=dict(width=line_widths, color='black')
            ),
            text=node_texts,
            textposition='top center',
            textfont=dict(size=11, color='black'),
            hoverinfo='text',
            hovertext=[
                f"Author: {G.nodes[n].get('author', 'Unknown')}<br>"
                f"Node Type: {G.nodes[n].get('type', 'Unknown')}<br>"
                f"Level: {G.nodes[n].get('level', 'Unknown')}<br>"
                f"First Publication Year: {G.nodes[n].get('first_publication_year', 'Unknown')}<br>"
                + (f"Predecessors: {', '.join([G.nodes[p].get('author', p) for p in G.predecessors(n)])}<br>" 
                   if list(G.predecessors(n)) else "")
                + (f"Successors: {', '.join([G.nodes[s].get('author', s) for s in G.successors(n)])}" 
                   if list(G.successors(n)) else "")
                for n in type_nodes
            ],
            name=type_name,
            showlegend=False
        ))
    
    return fig

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
            
            # Keep all other traces visible
        
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