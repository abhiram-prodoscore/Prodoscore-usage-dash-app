from shiny import App,render, ui, reactive
from google.cloud import bigquery
import pandas as pd
from google.auth import default
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from shiny.ui import HTML, input_select
import plotly.io as pio      
from plotly.graph_objs import Figure
from collections import defaultdict
import datetime
import plotly.express as px


credentials, _ = default()
project_id = "prodoscore-prodolab-live"
client = bigquery.Client(credentials=credentials, project=project_id)

domain_query = "select DISTINCT(c.domain_id) as domain_id, d.title as title from `prodoapp_analytics_dataset.click` as c left join `prodoapp_analytics_dataset.domain` as d on c.domain_id = d.id "
domain_table = client.query(domain_query).to_dataframe() #["domain_id"].tolist()
domain_dict = domain_table.set_index("domain_id")["title"].to_dict()
dropdown = ["Select Domain"] + list(domain_dict.values())

today = datetime.date.today()
day7 = today - datetime.timedelta(days=7)

def get_click_data(domain_filter,start_date, end_date):
    query = f"""SELECT 
                id,
                domain_id ,
                employee_id,
                cast(date as DATE) as date,
                page            
            FROM 
                `prodoapp_analytics_dataset.click` 
            WHERE domain_id = {domain_filter} AND cast(date as DATE) >= DATE('{start_date}') AND cast(date as DATE) <= DATE('{end_date}')"""

    df = client.query(query).to_dataframe()
    return df


ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            body {
                background-color: #f4f4f4;
            }
            .custom-class {
                color: blue;
                font-size: 18px;
            }
            #domain_selector {
                width: 300px;
                height: 40px;
                border-radius: 20px;
                border: 2px solid #007BFF;
                padding: 5px;
                font-size: 16px;
            }
        """)
    ),

    ui.panel_title(
        ui.row(
            ui.column(3, 
                      ui.input_select(id = "domain_selector", label="Select Domain", choices=dropdown, selected="Select Domain" ),
            ),
            ui.column(3,
                ui.output_ui("employee_selector"),
            ),
            ui.column(3,
                      ui.input_date_range("date_filter", "Select Date Range", start=day7, end=today,),
            ),
        ),
    ),
    ui.navset_pill(
        ui.nav_panel("Table view", ui.output_table("clicks")),
        ui.nav_panel("Time Series Plot", ui.output_ui("time_series_plot")),
        ui.nav_panel("Sankey Plot",
                    ui.row(
                        ui.column(6, 
                            ui.output_ui("page_selector")
                        ),
                        ui.column(6,
                                ui.input_slider("max_steps_slider", "Maximum Steps", min=1, max=20, value=10, step=1)
                        ),
                        ui.column(12, 
                            ui.output_ui("sankey_plot"),
                        )
                    ),
                    ui.row(
                        ui.column(12,
                            ui.output_ui("bar_plot")
                        )
                    )
            ) ,
    ) 
)


def server(input, output, session):

    # Initialize a reactive value to store the filtered data
    filtered_data = reactive.Value(pd.DataFrame())
    # selected_key = None
    employee_choice = reactive.Value(["Select Employee"])
    employee_filtered_data = reactive.Value(pd.DataFrame())


    @reactive.Effect
    @reactive.event(input.domain_selector, input.date_filter)
    def update_data():
        domain_filter = input.domain_selector()
        start_date, end_date = input.date_filter()
        #print(start_date,end_date)

        global empl_dict

        if domain_filter == "Select Domain" or not start_date or not end_date:
            filtered_data.set(pd.DataFrame())
            employee_choice.set(["Select Employee"])
            empl_dict = {}
        else:
            for key, value in domain_dict.items():
                if value == domain_filter:
                    selected_key = key
                    df = get_click_data(selected_key,start_date,end_date)
                    filtered_data.set(df)
                    empl_query = f"select DISTINCT(c.employee_id) as employee_id, e.fullname as fullname from `prodoapp_analytics_dataset.click` as c left join `prodoapp_analytics_dataset.employee` as e  on c.employee_id = e.id where c.domain_id = {selected_key} AND cast(c.date as DATE) >= DATE('2025-03-01') AND cast(c.date as DATE) <= DATE('2025-03-31') "
                    empl_table = client.query(empl_query).to_dataframe() 
                    empl_dict = empl_table.set_index("employee_id")["fullname"].to_dict()
                    dropdown_empl = ["Select Employee"] + list(empl_dict.values())
                    employee_choice.set(dropdown_empl)
                    break
            else:
                print("domain selected has problems")
            
    
    @output
    @render.ui
    def employee_selector():
        domain_filter = input.domain_selector()
        if domain_filter != "Select Domain" and not filtered_data.get().empty:
            return input_select("employee_selector", "Filter by Employee", choices=["Select Employee"] + list(employee_choice.get()), selected="Select Employee")
        return None
        
    @reactive.Effect
    @reactive.event(input.employee_selector)
    def employee_filter_selector():
        global empl_dict
        employee_filter = input.employee_selector()
        if employee_filter != "Select Employee":
            df = filtered_data.get()
            for key, value in empl_dict.items():
                if value == employee_filter:
                    employee_key = key
                    employee_df = df[df['employee_id'] == employee_key].copy()
                    employee_filtered_data.set(employee_df)
                    break
        else:
            print("employee_filter not selected")
    

    @output
    @render.table
    def clicks():
        if input.employee_selector() == "Select Employee":
            # If no employee is selected, show the full data
            return filtered_data.get() if not filtered_data.get().empty else pd.DataFrame({"Message":["No data available"]})
        return employee_filtered_data.get()
    
    @output
    # @render.plot
    @render.ui
    def time_series_plot():
        if input.employee_selector() != "Select Employee":
            click_data = employee_filtered_data.get().copy()
        else:
            click_data = filtered_data.get().copy()
        
        if click_data.empty:
            return None

        click_data['date'] = pd.to_datetime(click_data['date'])
       

        plot_data_grouped = click_data.groupby(['date', 'page']).size().reset_index(name='count')
        
        # Create a Plotly figure
        fig = go.Figure()

        # Loop through unique pages to add traces
        for page in plot_data_grouped['page'].unique():
            page_data = plot_data_grouped[plot_data_grouped['page'] == page]
            fig.add_trace(
                go.Scatter(
                    x=page_data['date'],
                    y=page_data['count'],
                    mode='lines+markers',
                    name=f"Page: {page}"
                )
            )

        # Set figure layout
        fig.update_layout(
            title=f"Time Series Chart for Domain: {input.domain_selector()}",
            xaxis_title=dict(text="Time"),
            yaxis_title=dict(text="Value"),
            legend_title="Pages",
        )

        # Display the figure
        return HTML(fig.to_html(full_html=False))
    
    @output
    @render.ui
    def page_selector():
        domain_filter = input.domain_selector()
        if input.employee_selector() != "Select Employee":
            df = employee_filtered_data.get()
        else:
            df = filtered_data.get()

        if domain_filter != "Select Domain" and not filtered_data.get().empty:
            unique_pages = sorted(df['page'].unique().tolist())
            page_choices = {"ALL": "Show All Paths"} # Use a dictionary for better labels
            page_choices.update({page: page for page in unique_pages})
            return input_select("page_selector", "Filter by Page", choices=page_choices, selected="Select Page")
        return None
    
    
    @reactive.Calc
    def filtered_data_paths():
        """
        Reactively filters the global df_paths based on the dropdown selection.
        """
        # Sankey plot data transformation steps
        if input.employee_selector() != "Select Employee":
            df = employee_filtered_data.get()
        else:
            df = filtered_data.get()
        
        # 3. Sessionization
        # Define session timeout (e.g., 30 minutes)

        session_timeout = pd.Timedelta(minutes=30)

        # Calculate time difference between consecutive events for the same user
        df['time_diff'] = df.groupby('employee_id')['date'].diff()

        # Identify session starts: first event for user OR time diff > timeout
        df['new_session'] = (df['time_diff'].isnull()) | (df['time_diff'] > session_timeout)

        # Assign unique session ID to each session
        df['session_id'] = df.groupby('employee_id')['new_session'].cumsum()

        # Create a unique global session ID if needed (optional, good practice)
        df['global_session_id'] = df['employee_id'].astype(str) + '_' + df['session_id'].astype(str)
        
        # 4. Path Extraction (Remove Consecutive Duplicates)
        # Keep only the first occurrence of consecutive identical pages within a session
        df_paths_global = df.loc[df['page'] != df.groupby(['global_session_id'])['page'].shift()]

        selected_page = input.page_selector()
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

        data_paths = filtered_data_paths()
        selected_page_filter = input.page_selector()
        max_steps = input.max_steps_slider()

        if data_paths.empty and selected_page_filter != 'ALL':
            print(f"No sessions contain the page: {selected_page_filter}")
            fig = go.Figure()
            fig.update_layout(title_text=f"No paths found containing '{selected_page_filter}'")
            return HTML(fig.to_html(full_html=False))

        # 5. Generate Transitions 
        transitions = defaultdict(int) # Dictionary to store counts of transitions { (source_step_label, target_step_label): count }
        # max_steps = 10 # Define how many steps/levels you want to visualize

        session_groups = data_paths.groupby('global_session_id')

        for name, group in session_groups:
            pages = group['page'].tolist()
            for i in range(len(pages) - 1):
                if i >= max_steps - 1: # Stop if we exceed the desired number of steps
                    break
                source_page = pages[i]
                target_page = pages[i+1]
                step = i + 1 # Step number (level) of the source node

                # Create labels that include the step number
                source_label = f"{source_page} (Step {step})"
                target_label = f"{target_page} (Step {step + 1})"

                transitions[(source_label, target_label)] += 1

        if not transitions:
            print("No transitions found. Check session logic or data.")
            fig = go.Figure()
            fig.update_layout(title_text=f"No transitions found for filter '{selected_page_filter}'")
            return HTML(fig.to_html(full_html=False))
        else:
            # 6. Aggregate Transitions (already done in step 5) & Prepare for Plotly 

            # Create list of unique node labels
            all_nodes = set()
            for source, target in transitions.keys():
                all_nodes.add(source)
                all_nodes.add(target)

            # Sort nodes for consistent ordering (optional but helpful)
            # Sort primarily by step number, then alphabetically
            def get_step(node_label):
                try:
                    return int(node_label.split('(Step ')[1].replace(')', ''))
                except:
                    print(f"Warning: Could not parse step number from label: {node_label}")
                    return 0 # Should not happen with correct labels

            sorted_nodes = sorted(list(all_nodes), key=lambda x: (get_step(x), x))

            # Create mapping from node label to index
            node_map = {node: i for i, node in enumerate(sorted_nodes)}

            # Prepare links data structure for Plotly
            link_sources = []
            link_targets = []
            link_values = []
            link_colors = [] # Optional: for coloring links

            # Define a color map or sequence if desired
            # E.g., colors = px.colors.qualitative.Plotly

            for (source_label, target_label), count in transitions.items():
                if source_label in node_map and target_label in node_map: # Ensure nodes exist
                    link_sources.append(node_map[source_label])
                    link_targets.append(node_map[target_label])
                    link_values.append(count)
                    # Add color logic here if needed, e.g., based on source/target node
                    # link_colors.append(colors[node_map[source_label] % len(colors)])


            # Prepare nodes data structure for Plotly
            node_labels = sorted_nodes
            node_x = [] # To control horizontal positioning for levels
            node_y = [] # To control vertical positioning (can be tricky, often let Plotly decide)
            default_x_spacing = 1.0 / max(max_steps, 1) # Normalized spacing

            # Assign x positions based on step number for clear levels
            node_step_map = {label: get_step(label) for label in node_labels}
            node_x = [ (node_step_map[label] -1 ) * default_x_spacing for label in node_labels]

            # Crude y positioning: distribute nodes evenly within each step
            nodes_at_step = defaultdict(list)
            for i, label in enumerate(node_labels):
                nodes_at_step[get_step(label)].append(i)

            node_y = [0] * len(node_labels) # Initialize y positions
            for step, indices in nodes_at_step.items():
                count_at_step = len(indices)
                for j, index in enumerate(indices):
                    # Normalize y position (0.0 to 1.0, with small padding)
                    node_y[index] = (j + 1) / (count_at_step + 1)


            # 8. Generate Sankey Diagram
            fig = go.Figure(data=[go.Sankey(
                # Define nodes
                node=dict(
                    pad=15,             # Padding between nodes
                    thickness=20,       # Thickness of nodes
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color="orange",    # Color of nodes (can be a list for individual colors)
                    x=node_x,          # Assign x coordinates for vertical levels
                    y=node_y           # Assign y coordinates for vertical positioning
                    # hovertemplate='%{label} had %{value} visits<extra></extra>' # Customize hover info if needed
                ),
                # Define links
                link=dict(
                    source=link_sources,
                    target=link_targets,
                    value=link_values,
                    # color=link_colors, # Optional link colors
                    hovertemplate='From %{source.label} to %{target.label}:<br />%{value} sessions<extra></extra>' # Custom hover info
            )
            )])

            fig.update_layout(
                title_text=f"User Navigation Path ({max_steps}-Step Sankey)",
                font_size=10,
                # Adjust margins if labels are cut off
                margin=dict(l=50, r=50, t=50, b=50),
                height= 600 # Adjust height if needed
                # width = 1000 # Adjust width if needed
            )

            return HTML(fig.to_html(full_html=False))

    @output
    @render.ui    
    def bar_plot():
        data_paths = filtered_data_paths()
        selected_page_filter = input.page_selector()
        max_steps = input.max_steps_slider()

        page_frequency = defaultdict(lambda: defaultdict(int))  # {page: {step: count}}

        session_groups = data_paths.groupby('global_session_id')

        for name, group in session_groups:
            pages = group['page'].tolist()
            for i in range(len(pages)):
                if i >= max_steps:  # Stop if we exceed max steps
                    break
                page = pages[i]
                step = i + 1  # Step number (1-based index)
                page_frequency[page][step] += 1

        # Flatten the nested dictionary
        page_data = [{"page": page, "step": step, "frequency": count}
                    for page, steps in page_frequency.items()
                    for step, count in steps.items()]
        page_df = pd.DataFrame(page_data)

        # Sort DataFrame by frequency and step (optional)
        page_df = page_df.sort_values(by=["step", "frequency"], ascending=[True, False])



        fig = px.bar(page_df, x="step", y="frequency", color="page",
                    title="Page Frequency by Step",
                    labels={"page": "Pages", "frequency": "Frequency", "step": "Step"},
                    height=400)
        fig.update_layout(barmode="stack")  # Group bars by step
      

        # Ensure all pages are included
        #fig.update_xaxes(categoryarray=list(page_df["page"].unique()))

        # Prevent Plotly from omitting low-frequency pages
        fig.update_traces(marker=dict(line=dict(width=0.5)))

        # Ensure order of categories is based on total frequency
        #fig.update_layout(xaxis={'categoryorder':'total descending'}, barmode="group")

        return HTML(fig.to_html(full_html=False))
        #return HTML(fig.to_html(full_html=False))

app = App(ui, server)

if __name__ == "__main__":
    app.run()
                                                            