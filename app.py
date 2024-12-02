# app.py

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import random
import pickle

# Load data from pickle files
with open('df_track_record.pkl', 'rb') as pickle_file:
    df_track_record = pickle.load(pickle_file)

with open('df_lineage.pkl', 'rb') as pickle_file:
    df_lineage = pickle.load(pickle_file)

# Create a network graph
def plot_mentorship_3(selected_mentor, max_level):
    # Filter the DataFrame to select the specific center mentor row
    df_lineage_sel = df_lineage[df_lineage['last_author'] == selected_mentor]

    # Initialize a directed graph
    G = nx.DiGraph()

    # First, add the center mentor node
    if not df_lineage_sel.empty:
        center_mentor = df_lineage_sel.iloc[0]['last_author']
        center_mentor_first_year = df_lineage_sel.iloc[0]['last_author_first_publication_year']
        center_num_of_first_author = df_lineage_sel.iloc[0]['last_author_num_of_first_author']
        center_num_of_last_author = df_lineage_sel.iloc[0]['last_author_num_of_last_author']
        # Add center mentor node with their publication year and publication counts
        G.add_node(center_mentor,
                   year=center_mentor_first_year,
                   type="center_mentor",
                   num_of_first_author=center_num_of_first_author,
                   num_of_last_author=center_num_of_last_author)

    # Sets of mentees
    first_level_mentees = set()
    second_level_mentees = set()
    third_level_mentees = set()

    # First Level: Center mentor and their direct mentees (after center_mentor's first year)
    for _, row in df_lineage_sel.iterrows():
        center_mentor = row['last_author']
        center_mentor_first_year = row['last_author_first_publication_year']

        # Iterate over each first_level mentee in the first_authors list
        for entry in row['first_authors']:
            first_level_mentee = entry['first_author']
            first_level_year = entry['first_publication_year']
            num_of_first_author = entry['num_of_first_author']
            num_of_last_author = entry['num_of_last_author']

            if first_level_mentee in first_level_mentees:
                continue  # Skip if already added
            first_level_mentees.add(first_level_mentee)

            # Add each first_level mentee node with their first publication year and publication counts
            if first_level_mentee != center_mentor:
                G.add_node(first_level_mentee,
                          year=first_level_year,
                          type="first_level",
                          num_of_first_author=num_of_first_author,
                          num_of_last_author=num_of_last_author)

                # Create an edge from center mentor to first_level mentee
                G.add_edge(center_mentor, first_level_mentee, year=center_mentor_first_year)

                # Stop here if max_level is 1
                if max_level == 1:
                    continue

                # Second Level: Check if this first level mentee is also a mentor for someone else
                df_second_level = df_lineage[(df_lineage['last_author'] == first_level_mentee)]
                for _, second_row in df_second_level.iterrows():
                    for second_entry in second_row['first_authors']:
                        second_level_mentee = second_entry['first_author']
                        second_level_year = second_entry['first_publication_year']
                        num_of_first_author = second_entry['num_of_first_author']
                        num_of_last_author = second_entry['num_of_last_author']

                        if (second_level_mentee in first_level_mentees or second_level_mentee in second_level_mentees):
                            continue  # Skip if already added in previous levels
                        second_level_mentees.add(second_level_mentee)

                        # Skip if second-level mentee is the center mentor
                        if second_level_mentee != center_mentor and second_level_mentee != first_level_mentee:
                            # Add the second level mentee node
                            G.add_node(second_level_mentee,
                                      year=second_level_year,
                                      type="second_level",
                                      num_of_first_author=num_of_first_author,
                                      num_of_last_author=num_of_last_author)

                            # Create an edge from first_level mentee to second_level mentee
                            G.add_edge(first_level_mentee, second_level_mentee, year=first_level_year)

                            # Stop here if max_level is 2
                            if max_level == 2:
                                continue

                            # Third Level: Check if this second level mentee is also a mentor for someone else
                            df_third_level = df_lineage[(df_lineage['last_author'] == second_level_mentee)]
                            for _, third_row in df_third_level.iterrows():
                                for third_entry in third_row['first_authors']:
                                    third_level_mentee = third_entry['first_author']
                                    third_level_year = third_entry['first_publication_year']
                                    num_of_first_author = third_entry['num_of_first_author']
                                    num_of_last_author = third_entry['num_of_last_author']

                                    if (third_level_mentee in first_level_mentees or
                                        third_level_mentee in second_level_mentees or
                                        third_level_mentee in third_level_mentees):
                                        continue  # Skip if already added in previous levels
                                    third_level_mentees.add(third_level_mentee)

                                    if third_level_mentee != center_mentor and third_level_mentee != first_level_mentee and third_level_mentee != second_level_mentee:
                                        # Add the third level mentee node
                                        G.add_node(third_level_mentee,
                                                  year=third_level_year,
                                                  type="third_level",
                                                  num_of_first_author=num_of_first_author,
                                                  num_of_last_author=num_of_last_author)

                                        # Create an edge from second level mentee to third level mentee
                                        G.add_edge(second_level_mentee, third_level_mentee, year=second_level_year)

    # Calculate dynamic y_spacing based on node density in each year
    year_counts = {year: sum(1 for _, data in G.nodes(data=True) if data['year'] == year) for year in {data['year'] for _, data in G.nodes(data=True)}}
    max_spacing = 8  # Maximum y_spacing for dense years
    min_spacing = 1  # Minimum y_spacing for sparse years
    dynamic_y_spacing = {year: max(min_spacing, max_spacing / max(1, count)) for year, count in year_counts.items()}

    # 2D Positioning: Use publication year as the x-coordinate and separate y-positions by mentorship level
    pos = {}
    level_y_offset = {"center_mentor": 0, "first_level": 6, "second_level": 12, "third_level": 18}

    # To ensure consistent results, set a random seed
    random.seed(42)

    # Assign positions for each node based on mentorship level and publication year
    for year in sorted(year_counts.keys()):
        mentors_in_year = [node for node, data in G.nodes(data=True) if data['year'] == year]
        level_positions = {level: 0 for level in level_y_offset}  # Track positions within each level

        for mentor in mentors_in_year:
            node_type = G.nodes[mentor]['type']

            # Add a random jitter to the Y position within a small range
            jitter = random.uniform(-1, 1)  # Adjust jitter range as needed
            pos[mentor] = (
                year,
                level_y_offset[node_type] + level_positions[node_type] * dynamic_y_spacing[year] + jitter
            )

            level_positions[node_type] += 1  # Increment position within the level

    # Prepare edge traces for the 2D plot with arrows and color-coded by direction
    edge_x_forward, edge_y_forward = [], []
    edge_x_backward, edge_y_backward = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Determine edge direction based on publication year
        if x1 >= x0:  # Forward or same year
            edge_x_forward.extend([x0, x1, None])
            edge_y_forward.extend([y0, y1, None])
        else:  # Backward
            edge_x_backward.extend([x0, x1, None])
            edge_y_backward.extend([y0, y1, None])

    # Create forward edge trace with blue arrows
    edge_trace_forward = go.Scatter(
        x=edge_x_forward, y=edge_y_forward,
        line=dict(width=1.5, color='lightblue'),
        mode='lines',
        name="Forward Lineage",
        hoverinfo="skip"
    )

    # Create backward edge trace with orange arrows
    edge_trace_backward = go.Scatter(
        x=edge_x_backward, y=edge_y_backward,
        line=dict(width=1.5, color='orange'),
        mode='lines',
        name="Backward Lineage",
        hoverinfo="skip"
    )

    # Separate nodes by mentorship level for different colors and prepare node traces
    center_mentor_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'center_mentor']
    first_level_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'first_level']
    second_level_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'second_level']
    third_level_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'third_level']

    # Generate hover text with the added fields
    center_mentor_hover_text = [
        f"Mentor: {node}<br>"
        f"Year: {G.nodes[node]['year']}<br>"
        f"Number of First Author Publications: {G.nodes[node]['num_of_first_author']}<br>"
        f"Number of Last Author Publications: {G.nodes[node]['num_of_last_author']}"
        for node in center_mentor_nodes
    ]

    first_level_hover_text = [
        f"Center Mentor: {selected_mentor}<br>"
        f"First Lineage: {node}<br>"
        f"Year: {G.nodes[node]['year']}<br>"
        f"Number of First Author Publications: {G.nodes[node]['num_of_first_author']}<br>"
        f"Number of Last Author Publications: {G.nodes[node]['num_of_last_author']}"
        for node in first_level_nodes
    ]

    second_level_hover_text = [
        f"Center Mentor: {selected_mentor}<br>"
        f"First Lineage: {list(G.predecessors(node))[0]}<br>"
        f"Second Lineage: {node}<br>"
        f"Year: {G.nodes[node]['year']}<br>"
        f"Number of First Author Publications: {G.nodes[node]['num_of_first_author']}<br>"
        f"Number of Last Author Publications: {G.nodes[node]['num_of_last_author']}"
        for node in second_level_nodes
    ]

    third_level_hover_text = [
        f"Center Mentor: {selected_mentor}<br>"
        f"First Lineage: {list(G.predecessors(list(G.predecessors(node))[0]))[0] if list(G.predecessors(list(G.predecessors(node))[0])) else 'N/A'}<br>"
        f"Second Lineage: {list(G.predecessors(node))[0]}<br>"
        f"Third Lineage: {node}<br>"
        f"Year: {G.nodes[node]['year']}<br>"
        f"Number of First Author Publications: {G.nodes[node]['num_of_first_author']}<br>"
        f"Number of Last Author Publications: {G.nodes[node]['num_of_last_author']}"
        for node in third_level_nodes
    ]

    # Define alternate text position function
    def alternate_text_position(index):
        """Returns an alternating text position to avoid overlap."""
        positions = ['top center', 'bottom center', 'middle left', 'middle right']
        return positions[index % len(positions)]

    # Generate trace data
    center_mentor_trace = go.Scatter(
        x=[pos[node][0] for node in center_mentor_nodes],
        y=[pos[node][1] for node in center_mentor_nodes],
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=[node for node in center_mentor_nodes],
        textposition=[alternate_text_position(i) for i in range(len(center_mentor_nodes))],
        textfont=dict(size=12),
        name="Center Mentor",
        hovertext=center_mentor_hover_text,
        hoverinfo="text"
    )

    first_level_trace = go.Scatter(
        x=[pos[node][0] for node in first_level_nodes],
        y=[pos[node][1] for node in first_level_nodes],
        mode='markers+text',
        marker=dict(size=12, color='darkblue'),
        text=[node for node in first_level_nodes],
        textposition=[alternate_text_position(i) for i in range(len(first_level_nodes))],
        textfont=dict(size=12),
        name="First Generation Mentorship",
        hovertext=first_level_hover_text,
        hoverinfo="text"
    )

    second_level_trace = go.Scatter(
        x=[pos[node][0] for node in second_level_nodes],
        y=[pos[node][1] for node in second_level_nodes],
        mode='markers+text',
        marker=dict(size=10, color='purple'),
        text=[node for node in second_level_nodes],
        textposition=[alternate_text_position(i) for i in range(len(second_level_nodes))],
        textfont=dict(size=10),
        name="Second Generation Mentorship",
        hovertext=second_level_hover_text,
        hoverinfo="text"
    )

    third_level_trace = go.Scatter(
        x=[pos[node][0] for node in third_level_nodes],
        y=[pos[node][1] for node in third_level_nodes],
        mode='markers+text',
        marker=dict(size=8, color='green'),
        text=[node for node in third_level_nodes],
        textposition=[alternate_text_position(i) for i in range(len(third_level_nodes))],
        textfont=dict(size=8),
        name="Third Generation Mentorship",
        hovertext=third_level_hover_text,
        hoverinfo="text"
    )

    # Career Milestones: Filter milestones for the selected mentor
    df_mentor_milestones = df_track_record[df_track_record['author_name'] == selected_mentor]

    # Add vertical lines at each milestone's start year on the plot
    milestone_lines = []
    for _, milestone in df_mentor_milestones.iterrows():
        start_year = milestone['start_year'] if pd.notna(milestone['start_year']) else milestone['end_year']
        position_grant = milestone['position_grant']
        institution_source = milestone['institution_source']

        # Add a vertical line at the start year with hover text showing the milestone
        milestone_lines.append(go.Scatter(
            x=[start_year, start_year],
            y=[0, max(pos.values(), key=lambda v: v[1])[1] + 5],  # Extend the line to fit the plot vertically
            mode="lines",
            line=dict(color="gray", dash="dash"),
            hovertext=f"{position_grant}<br>{institution_source}",
            name="Career Milestone",
            hoverinfo="text",
            showlegend=False
        ))

    # Plot using Plotly
    fig = go.Figure(data=[
        edge_trace_forward,
        edge_trace_backward,
        center_mentor_trace,
        first_level_trace,
        second_level_trace,
        third_level_trace,
        *milestone_lines  # Add all milestone lines to the plot
    ])

    # Set plot layout
    fig.update_layout(
        title=dict(
            text=f"{selected_mentor}'s Mentorship Network and Career Milestones",
            x=0.5,  # Title position (0: left, 1: right)
            y=0.98,  # Title position (0: bottom, 1: top)
            xanchor='center',  # Horizontal alignment
            yanchor='top',  # Vertical alignment
            font=dict(
                family="Arial",
                size=20,
                color="white")
        ),
        xaxis=dict(
            title="Publication Year",
            tickmode='linear',
            dtick=1
        ),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5
        ),
        height=1000,
        width=2000,
        margin=dict(l=10, r=10, t=100, b=60)
    )

    # Return the figure
    return fig

# Set wide config
st.set_page_config(layout="wide")

# Title of the app
st.title('Mentorship Network and Career Milestones Visualization')

with st.container():
    # Mentor selection
    # mentor_options = df_lineage['last_author'].unique().tolist()
    mentor_options = ['Douglas J. Wiebe', 'Stephen Hargarten', 'Andrew V. Papachristos', 'Susan B. Sorenson'] # Whitelisted researchers
    selected_mentor = st.selectbox('Select Mentor Name:', mentor_options)

    # Lineage Level selection
    max_level = st.selectbox('Select Lineage Level:', [1, 2, 3], index=2)

# Generate the plot based on user selections
fig = plot_mentorship_3(selected_mentor, max_level)

# Display the plot
st.plotly_chart(fig, theme=None, use_container_width=True)

