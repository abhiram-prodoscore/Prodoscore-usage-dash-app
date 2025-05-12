from shiny import App, ui, render, reactive, event
import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
import calendar

# ---- Data ----

def make_data(month, year):
    # Only weekdays in month
    first = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = date(year, month, last_day)
    dates = pd.bdate_range(first, last)  # Only Mon-Fri dates
    vals = np.random.randint(1, 100, len(dates))
    df = pd.DataFrame({'date': dates, 'value': vals})
    return df

# ---- COLOR Mapping ----

def value_to_color(val, vmin, vmax):
    """
    Maps val in [vmin,vmax] to color.
    Returns HSL string in blue-green gradient
    """
    if vmax == vmin:
        norm = 0.5
    else:
        norm = (val - vmin) / (vmax - vmin)
    # 160 (green) to 200 (blue)
    h = int(160 + norm * (200-160))
    return f"hsl({h},65%,70%)"

# ---- Calendar Box Generation ----

def generate_calendar_box(df, selected_start=None, selected_end=None):
    """HTML calendar grid with weekday columns, month days, color-coding, and JS for range selection."""
    df = df.copy()
    df['day'] = df['date'].dt.day
    num_weeks = (len(df)+4)//5
    vmin, vmax = df['value'].min(), df['value'].max()
    # Range as pd.Timestamp for comparison
    if selected_start: selected_start = pd.to_datetime(selected_start)
    if selected_end:   selected_end = pd.to_datetime(selected_end)
    # Calendar table
    calendar_html = '''
    <style>
    .calendar-day.selected {border:2px solid #000;}
    .calendar-day.in-range {border:2px solid #008CBA; background-color: #CDEEFF !important;}
    .calendar-day {transition: background 0.2s;}
    </style>
    <table style="border-collapse:collapse; text-align:center;">
    <tr>
    ''' + ''.join(f'<th style="width:40px;">{d}</th>' for d in ['Mon','Tue','Wed','Thu','Fri']) + '</tr>'

    for week in range(num_weeks):
        calendar_html += '<tr>'
        for i in range(5):
            idx = week*5 + i
            if idx < len(df):
                row = df.iloc[idx]
                day = row['day']
                date_str = row['date'].strftime('%Y-%m-%d')
                color = value_to_color(row['value'], vmin, vmax)
                # Selection styling:
                selected = ""
                in_range = ""
                if selected_start and selected_end:
                    if selected_start > selected_end:
                        s1, s2 = selected_end, selected_start
                    else:
                        s1, s2 = selected_start, selected_end
                    if s1 <= row['date'] <= s2:
                        in_range = "in-range"
                if selected_start and row['date'] == selected_start:
                    selected = "selected"
                if selected_end and row['date'] == selected_end and selected_end!=selected_start:
                    selected = "selected"
                calendar_html += (
                    f'<td><button '
                    f'class="calendar-day {selected} {in_range}" '
                    f'style="width:40px;height:40px;margin:2px;background:{color};border-radius:6px;cursor:pointer;" '
                    f'onclick="calendar_select_date(\'{date_str}\', this)">{day}</button></td>'
                )
            else:
                calendar_html += '<td></td>'
        calendar_html += '</tr>'
    calendar_html += '</table>'
    # JavaScript for range selection and communication with Shiny
    calendar_html += '''
    <input type="hidden" id="range_start" name="range_start" />
    <input type="hidden" id="range_end" name="range_end" />
    <script>
    window.calendar_selected = [];
    function calendar_select_date(date_str, btn) {
        // update selections: range logic
        if(window.calendar_selected.length==0 || window.calendar_selected.length==2){
            window.calendar_selected = [date_str];
        } else {
            window.calendar_selected.push(date_str);
        }
        // assign in input for Shiny
        document.getElementById('range_start').value = window.calendar_selected[0];
        document.getElementById('range_end').value = window.calendar_selected.length>1 ? window.calendar_selected[1] : window.calendar_selected[0];
        // Send to Shiny
        if(window.Shiny){
            Shiny.setInputValue('range_start', window.calendar_selected[0], {priority: "event"});
            Shiny.setInputValue('range_end', window.calendar_selected.length>1 ? window.calendar_selected[1] : window.calendar_selected[0], {priority: "event"});
        }
    }
    </script>
    '''
    return calendar_html

# ---- UI ----

app_ui = ui.page_fluid(
    ui.h3("Interactive Weekday Heatmap Calendar"),
    ui.page_sidebar(
        ui.sidebar(
            ui.input_action_button("prev_month", "Prev Month"),
            ui.input_action_button("next_month", "Next Month"),
            ui.hr(),
            ui.markdown("**Click two dates to select a range.**"),
            ui.output_text("current_month"),
            height="200px"
        ),
        ui.output_ui("calendar_box"),
        ui.hr(),
        ui.output_text("range_summary"),
    )
)

# ---- Server ----

def server(input, output, session):
    # --- Month/Year reactive values ---
    current_month = reactive.value(datetime.today().month)
    current_year = reactive.value(datetime.today().year)

    @event(input.prev_month)
    def _( ):
        # Move to previous month
        m = current_month.get()
        y = current_year.get()
        if m==1:
            m=12
            y-=1
        else:
            m-=1
        current_month.set(m)
        current_year.set(y)

    @event(input.next_month)
    def _( ):
        # Move to next month
        m = current_month.get()
        y = current_year.get()
        if m==12:
            m=1
            y+=1
        else:
            m+=1
        current_month.set(m)
        current_year.set(y)

    # ---- Data and calendar box ----

    @output
    @render.text
    def current_month():
        m = current_month.get()
        y = current_year.get()
        return f"**{calendar.month_name[m]} {y}**"

    @output
    @render.ui
    def calendar_box():
        m = current_month.get()
        y = current_year.get()
        df = make_data(m, y)
        return ui.HTML(generate_calendar_box(df, input.range_start(), input.range_end() ))

    # ---- Range summary ----

    @output
    @render.text
    def range_summary():
        range_start = input.range_start()
        range_end = input.range_end()
        m = current_month.get()
        y = current_year.get()
        df = make_data(m, y)
        if not range_start or not range_end:
            return "Select a date range"
        d1 = pd.to_datetime(range_start)
        d2 = pd.to_datetime(range_end)
        start, end = min(d1, d2), max(d1, d2)
        mask = (df['date'] >= start) & (df['date'] <= end)
        selected_rows = df[mask]
        if selected_rows.empty:
            return f"Range {start.date()} to {end.date()}: No data"
        # e.g. Summarize values
        return (f"Selected Range: {start.date()} to {end.date()}  \n"
                f"- {len(selected_rows)} weekdays selected  \n"
                f"- Value sum: {selected_rows['value'].sum()}  \n"
                f"- Mean: {selected_rows['value'].mean():.2f}")

app = App(app_ui, server)