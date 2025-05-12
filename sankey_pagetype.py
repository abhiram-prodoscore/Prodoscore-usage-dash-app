import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np
from shiny import App, reactive, render, ui
from shiny.types import FileInfo # Required for potential file upload later
import io # Required for potential file upload later
from google.cloud import bigquery
from google.auth import default
from shiny.ui import HTML, input_select


# --- Configuration ---
SESSION_TIMEOUT_MINUTES = 30
MAX_SANKEY_STEPS = 5 # Keep this relatively low for performance

# --- Data Loading and Initial Processing ---
# This part runs once when the Shiny app starts.
# In a real app, you might load from a file here.

print("--- Initial Data Loading and Processing ---")
credentials, _ = default()
project_id = "prodoscore-prodolab-live"
client = bigquery.Client(credentials=credentials, project=project_id)

query = f"""SELECT 
                id,
                domain_id ,
                employee_id,
                cast(date as DATE) as date,
                page            
            FROM 
                `prodoapp_analytics_dataset.click` 
            WHERE domain_id = 9 AND cast(date as DATE) >= DATE('2025-03-01') AND cast(date as DATE) <= DATE('2025-03-31')"""

df = client.query(query).to_dataframe()
df_initial = pd.DataFrame(df)

# Preprocessing
print("Preprocessing data...")
df_initial['date'] = pd.to_datetime(df_initial['date'], errors='coerce')
df_initial.dropna(subset=['date'], inplace=True)
df_initial = df_initial.sort_values(by=['employee_id', 'date'])

# Sessionization
print(f"Identifying sessions with timeout: {SESSION_TIMEOUT_MINUTES} minutes...")
session_timeout = pd.Timedelta(minutes=SESSION_TIMEOUT_MINUTES)
df_initial['time_diff'] = df_initial.groupby('employee_id')['date'].diff()
df_initial['new_session'] = (df_initial['time_diff'].isnull()) | (df_initial['time_diff'] > session_timeout)
df_initial['session_id'] = df_initial.groupby('employee_id')['new_session'].cumsum()
df_initial['global_session_id'] = df_initial['employee_id'].astype(str) + '_session_' + df_initial['session_id'].astype(str)

# Path Extraction
print("Extracting unique page navigation paths per session...")
df_paths_global = df_initial.loc[df_initial['page'] != df_initial.groupby(['global_session_id'])['page'].shift()]
print("--- Initial processing complete ---")

# Get unique pages for dropdown choices
unique_pages = sorted(df_paths_global['page'].unique().tolist())
filter_choices = {"ALL": "Show All Paths"} # Use a dictionary for better labels
filter_choices.update({page: page for page in unique_pages})

# --- Helper function to get step from label ---
# (Same as before)
def get_step_from_label(node_label):
    try:
        return int(node_label.split('(Step ')[1].replace(')', ''))
    except (IndexError, ValueError):
        print(f"Warning: Could not parse step number from label: {node_label}")
        return 0

# --- Shiny UI Definition ---
app_ui = ui.page_fluid(
    ui.panel_title(
        ui.row(
            ui.column(6,
                 ui.h4("Sankey Filter"),
            ),
            ui.column(6,
                ui.input_select(
                id="page_filter",
                label="Filter by Page:",
                choices=filter_choices, # Use the dictionary here
                selected="ALL"
                ),
            ),            
        )        
    ),
        ui.p(f"Max steps shown: {MAX_SANKEY_STEPS}"),
        # Add other controls here if needed (e.g., slider for max_steps)
        ui.output_ui("sankey_plot") # Placeholder for the plot
    
)

# --- Shiny Server Logic ---
def server(input, output, session):

    @reactive.Calc
    def filtered_data_paths():
        """
        Reactively filters the global df_paths based on the dropdown selection.
        """
        selected_page = input.page_filter()
        print(f"Input filter changed to: {selected_page}") # Debug print

        if selected_page != 'ALL':
            # Find sessions containing the selected page
            sessions_with_page = df_paths_global[df_paths_global['page'] == selected_page]['global_session_id'].unique()
            if len(sessions_with_page) > 0:
                # Return paths belonging to those sessions
                return df_paths_global[df_paths_global['global_session_id'].isin(sessions_with_page)].copy()
            else:
                # Return an empty DataFrame if no sessions match
                return pd.DataFrame(columns=df_paths_global.columns)
        else:
            # Return all paths if 'ALL' is selected
            return df_paths_global.copy()

    @output
    @render.ui
    def sankey_plot():
        """
        Generates and renders the Sankey Plotly figure based on filtered data.
        """
        data_paths = filtered_data_paths() # Get the reactively filtered data
        selected_page_filter = input.page_filter() # Get current filter value for title

        if data_paths.empty and selected_page_filter != 'ALL':
            print(f"No sessions contain the page: {selected_page_filter}")
            fig = go.Figure()
            fig.update_layout(title_text=f"No paths found containing '{selected_page_filter}'")
            return fig

        # --- Generate Transitions ---
        transitions = defaultdict(int)
        session_groups = data_paths.groupby('global_session_id')
        for session_id, group in session_groups:
            pages = group['page'].tolist()
            for i in range(len(pages) - 1):
                current_step_number = i + 1
                if current_step_number >= MAX_SANKEY_STEPS:
                    break
                source_page = pages[i]
                target_page = pages[i+1]
                source_label = f"{source_page} (Step {current_step_number})"
                target_label = f"{target_page} (Step {current_step_number + 1})"
                transitions[(source_label, target_label)] += 1

        if not transitions:
            print("No transitions found for the selected filter.")
            fig = go.Figure()
            fig.update_layout(title_text=f"No transitions found for filter '{selected_page_filter}'")
            return fig

        # --- Prepare Data for Plotly (Nodes/Links/Positioning) ---
        # (This logic is the same as in the create_sankey function before)
        all_node_labels = set()
        for source, target in transitions.keys():
            all_node_labels.add(source)
            all_node_labels.add(target)

        sorted_node_labels = sorted(list(all_node_labels), key=lambda x: (get_step_from_label(x), x))
        node_map = {label: i for i, label in enumerate(sorted_node_labels)}

        link_sources, link_targets, link_values = [], [], []
        for (source_label, target_label), count in transitions.items():
            if source_label in node_map and target_label in node_map:
                link_sources.append(node_map[source_label])
                link_targets.append(node_map[target_label])
                link_values.append(count)

        node_labels_for_plotly = sorted_node_labels
        node_x_positions, node_y_positions = [], []
        max_actual_step = max(get_step_from_label(label) for label in node_labels_for_plotly) if node_labels_for_plotly else 1
        x_divisor = max(1, max_actual_step -1) if max_actual_step > 1 else 1
        x_padding, x_range = 0.01, 1.0 - 2 * 0.01
        node_step_map = {label: get_step_from_label(label) for label in node_labels_for_plotly}
        for label in node_labels_for_plotly:
            step = node_step_map[label]
            x_pos = x_padding + ( (step - 1) / x_divisor ) * x_range if x_divisor > 0 else 0.5
            node_x_positions.append(x_pos)

        nodes_at_step = defaultdict(list)
        for i, label in enumerate(node_labels_for_plotly):
            nodes_at_step[get_step_from_label(label)].append(i)
        node_y_positions = [0.5] * len(node_labels_for_plotly)
        y_padding, y_range = 0.05, 1.0 - 2 * 0.05
        for step, indices_in_step in nodes_at_step.items():
            count_at_step = len(indices_in_step)
            if count_at_step > 1:
                for j, node_index in enumerate(indices_in_step):
                     node_y_positions[node_index] = y_padding + (j / (count_at_step - 1)) * y_range
            elif count_at_step == 1:
                 node_y_positions[indices_in_step[0]] = 0.5

        # --- Create Figure ---
        print(f"Rendering plot for filter: {selected_page_filter}") # Debug print
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25, thickness=15, line=dict(color="black", width=0.5),
                label=node_labels_for_plotly, color="lightcoral", # Changed color slightly
                x=node_x_positions, y=node_y_positions
            ),
            link=dict(
                source=link_sources, target=link_targets, value=link_values,
                hovertemplate='Path: %{source.label} -> %{target.label}<br>Count: %{value} sessions<extra></extra>'
           )
        )])

        fig.update_layout(
            title_text=f"User Navigation Path ({MAX_SANKEY_STEPS}-Step Sankey) - Filter: {selected_page_filter}",
            font=dict(size=10, color="black"), paper_bgcolor='white',
            margin=dict(l=50, r=50, t=60, b=40)
        )
        return HTML(fig.to_html(full_html=False))

# --- Create and Run the App ---
app = App(app_ui, server)

# If running directly as a script, you might need:
if __name__ == "__main__":
    app.run()
# However, Shiny apps are often run using the command line: `shiny run your_app_file.py --reload`

