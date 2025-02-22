####################################################################
#
#   This code fetchs backend data to display a dashboard with the most
#   recent data on applications in the NHPS system.
#
#   Author: Ignacio Lepe
#   Created: 12/02/2025
#   Modified: 14/02/2025 
####################################################################

# Pedir fonts
# Arreglar grafico de calor
# colores explorador en mapa

################# Packages
import dash
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from io import BytesIO
import base64
from datetime import datetime

from db_connection import conect_bd

# Select input data
simulated_data = 0 # if equal to 1 => Create simulated data to test dashboard, 0 => Fetch most recent backend data and upload to dashboard.
year = 2025
day_zero_current_year = pd.to_datetime("2025-02-02")  # Adjust the format
day_zero_previous_year = pd.to_datetime("2024-02-02")  # Adjust the format

# Bring private crosswalk (list of users in intervetion)
# registration_crosswalk = pd.read_csv('/Users/ignaciolepe/Documents/GitHub/nhps-schoolmint-pipeline/1_data/intermediates/registration_crosswalk.csv')

############

if simulated_data == 0:
    
    ################# Fetch backend data 

    # Define environment and database schema
    environment = 'staging'  # Change to 'production' if needed
    tenant = 'newhaven'
    
    # Establish connection
    conn = conect_bd('core', environment) 

    # Fetch applications per day for the current and previous year
    query = f"""
        SELECT 
            application_id,
            DATE(created) AS Date, 
            COUNT(DISTINCT application_id) AS Applications
        FROM {tenant}.institutions_application_ranking
        WHERE EXTRACT(YEAR FROM created) = {year}
        GROUP BY application_id, Date
        ORDER BY Date;
    """
    applications_current_year_df = pd.read_sql_query(query, conn)

    # Ensure 'Date' and 'Applications' are correctly capitalized
    applications_current_year_df.rename(
        columns={'date': 'Date', 'applications': 'Applications'}, inplace=True
    )
    applications_current_year_df['Date'] = pd.to_datetime(applications_current_year_df['Date'])

    ########## SIMULATION ON PREVIOUS YEAR DATA

    # Simulate the previous year's data by selecting 80% of the applications per day
    applications_previous_year_list = []

    for date, group in applications_current_year_df.groupby("Date"):
        sample_size = int(len(group) * 0.85)  # 80% sample per day
        sampled_group = group.sample(n=sample_size, random_state=42)  # Ensure reproducibility
        sampled_group["Date"] = sampled_group["Date"] - pd.DateOffset(years=1)  # Shift back by 1 year
        applications_previous_year_list.append(sampled_group)

    # Concatenate all sampled groups
    applications_previous_year_df = pd.concat(applications_previous_year_list, ignore_index=True)

    # Aggregate counts by date
    applications_previous_year_df = (
        applications_previous_year_df.groupby("Date")
        .size()
        .reset_index(name="Applications (Previous Year)")
    )

    # Aggregate counts for the current year
    applications_current_year_df = (
        applications_current_year_df.groupby("Date")
        .size()
        .reset_index(name="Applications")
    )

    # Merge current and previous year data
    applications_df = applications_current_year_df.merge(
        applications_previous_year_df, on="Date", how="outer"
    ).fillna(0)

    # Compute cumulative applications
    applications_df["Cumulative Applications (Current Year)"] = applications_df["Applications"].cumsum()
    applications_df["Cumulative Applications (Previous Year)"] = applications_df["Applications (Previous Year)"].cumsum()

    # ✅ Ensure 'Date' is in datetime format before filtering for Mondays
    applications_df["Date"] = pd.to_datetime(applications_df["Date"])

    # Select only weekly (Monday) dates for x-axis ticks
    weekly_dates = applications_df["Date"][applications_df["Date"].dt.weekday == 0]  # ✅ Now this works
    
    # Fetch applications by grade
    query = f"""
        SELECT l.grade_name AS Grade, COUNT(DISTINCT g.applicant_id) AS Applications
        FROM {tenant}.registration_applicant_interested_grade g
        JOIN {tenant}.institutions_grade_label l ON g.grade_label_id = l.id
        WHERE EXTRACT(YEAR FROM created) = {year}
        GROUP BY l.grade_name
        ORDER BY l.grade_name;
    """
    applications_grade_df = pd.read_sql_query(query, conn)

    # Fetch school demand data
    query = f"""
        SELECT 
            ins.institution_name AS School, 
            p.regular_vacancies AS Total_Vacants, 
            g.grade_label_id AS Grade
        FROM {tenant}.institutions_program p
        JOIN {tenant}.institutions_institutions ins ON p.institution_code::BIGINT = ins.id
        JOIN {tenant}.registration_applicant_interested_grade g ON p.id = g.grade_label_id
        ORDER BY ins.institution_name, g.grade_label_id;
    """
    school_demand_df = pd.read_sql_query(query, conn)

    # Simulate 'Used_Vacants' as a random number between 0 and Total_Vacants ##### CHANGE THIS LATER WITH SIMULATION RESULTS!!!!
    school_demand_df["Used_Vacants"] = np.random.randint(0, school_demand_df["total_vacants"] + 1, size=len(school_demand_df))

    # Calculate % Used
    school_demand_df["% Used"] = (school_demand_df["Used_Vacants"] / school_demand_df["total_vacants"]) * 100

    # Get most and least demanded schools
    most_demanded_df = school_demand_df.sort_values("% Used", ascending=False).head(10)
    least_demanded_df = school_demand_df.sort_values("% Used", ascending=True).head(10)

    # Fetch applicant demand heatmap data
    query_applicant_demand = f"""
        SELECT 
            ad.address_lat AS lat, 
            ad.address_lon AS lon, 
            COUNT(DISTINCT a.id) AS Demand
        FROM {tenant}.registration_applicant a
        JOIN {tenant}.registration_applicant_address ad 
            ON a.id = ad.applicant_id
        WHERE a.id IN (SELECT applicant_id FROM {tenant}.institutions_application WHERE year = {year})
        GROUP BY ad.address_lat, ad.address_lon
        ORDER BY Demand DESC;
    """

    heatmap_df = pd.read_sql_query(query_applicant_demand, conn)
    # Fetch school demand heatmap data
    query_school_demand = f"""
        SELECT 
            loc.latitud AS lat, 
            loc.longitud AS lon, 
            COUNT(DISTINCT ia.applicant_id) AS Demand
        FROM {tenant}.institutions_application ia
        JOIN {tenant}.institutions_location loc 
            ON loc.institution_code::BIGINT = loc.institution_code::BIGINT  
        WHERE loc.latitud IS NOT NULL 
        AND loc.longitud IS NOT NULL 
        AND ia.year = {year}
        GROUP BY loc.latitud, loc.longitud
        ORDER BY Demand DESC;
    """

    school_heatmap_df = pd.read_sql_query(query_school_demand, conn)

    # Convert lat and lon to numeric if necessary 
    heatmap_df["lat"] = pd.to_numeric(heatmap_df["lat"], errors="coerce")
    heatmap_df["lon"] = pd.to_numeric(heatmap_df["lon"], errors="coerce")
    school_heatmap_df["lat"] = pd.to_numeric(school_heatmap_df["lat"], errors="coerce")
    school_heatmap_df["lon"] = pd.to_numeric(school_heatmap_df["lon"], errors="coerce")

    # Remove rows where lat or lon are NaN
    school_heatmap_df = school_heatmap_df.dropna(subset=['lat', 'lon'])
    heatmap_df = heatmap_df.dropna(subset=['lat', 'lon'])

    # Explicitly rename the demand column to ensure consistency
    heatmap_df.rename(columns={"demand": "Demand"}, inplace=True)
    school_heatmap_df.rename(columns={"demand": "Demand"}, inplace=True)

    # Ensure 'Date' is correctly formatted as a datetime object
    applications_current_year_df["Date"] = pd.to_datetime(applications_current_year_df["Date"])
    applications_previous_year_df["Date"] = pd.to_datetime(applications_previous_year_df["Date"])

    # Calculate days since "Day 0"
    applications_current_year_df["Day"] = (applications_current_year_df["Date"] - day_zero_current_year).dt.days
    applications_previous_year_df["Day"] = (applications_previous_year_df["Date"] - day_zero_previous_year).dt.days

    # **Filter out values before Day 0**
    applications_current_year_df = applications_current_year_df[applications_current_year_df["Day"] >= 0]
    applications_previous_year_df = applications_previous_year_df[applications_previous_year_df["Day"] >= 0]

    # Merge current and previous year data based on 'Day'
    applications_df = applications_current_year_df.merge(
        applications_previous_year_df, on="Day", how="outer", suffixes=("", " (Previous Year)")
    ).fillna(0)

    ############### SIMULATION SECTION. PLEASE UPDATE LATER WHEN SIMULAION API IS DONE

    # Simulate data for the simulation section
    num_applicants = 1000
    assigned_fraction = np.random.uniform(0.6, 0.9, num_applicants)
    assignment_counts = np.random.dirichlet([4, 3, 2, 1, 1, 2, 2], size=1)[0] * num_applicants
    assignment_labels = [
        "1st Preference", "2nd Preference", "3rd Preference",
        "4th Preference", "+5th Preference", "Non-Assigned"
    ]

    # Ensure assignment_counts has the same length as assignment_labels
    num_preferences = len(assignment_labels)  # Ensure matching lengths
    assignment_counts = np.random.dirichlet([4, 3, 2, 1, 1, 2], size=1)[0] * num_applicants

    # Ensure the generated counts have the correct length
    if len(assignment_counts) != num_preferences:
        assignment_counts = np.append(assignment_counts, num_applicants - sum(assignment_counts))  # Adjust to match total

    assignment_df = pd.DataFrame({"Assignment": assignment_labels, "Fraction": (assignment_counts / 10).round(1)})

    # Generate simulated data
    num_applicants = 1000
    non_assigned_prob = np.random.rand(num_applicants)  # Probabilities of non-assignment

    # Close the connection
    conn.close()

elif simulated_data == 1:
    ################# Generate simulated data
    np.random.seed(42)
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(60)]  # 60 days of application period
    applications_per_day = np.random.randint(5, 35, size=len(dates))
    cumulative_applications = np.cumsum(applications_per_day)

    # Generate previous year's data for comparison
    applications_per_day_prev = np.random.randint(5, 50, size=len(dates))
    cumulative_applications_prev = np.cumsum(applications_per_day_prev)

    # Simulate applications by grade
    grades = ["K", "1", "2", "3", "4", "5"]
    applications_by_grade = np.random.randint(100, 500, size=len(grades))
    applications_grade_df = pd.DataFrame({"Grade": grades, "Applications": applications_by_grade})

    # Create DataFrame for applications
    applications_df = pd.DataFrame({
        "Date": dates,
        "Applications": applications_per_day,
        "Cumulative Applications (Current Year)": cumulative_applications,
        "Applications (Previous Year)": applications_per_day_prev,
        "Cumulative Applications (Previous Year)": cumulative_applications_prev
    })

    # Define day_zero_current_year dynamically for simulated data
    day_zero_current_year_simulated = dates[0]  # First date in simulated data

    # Calculate "Day" as days since "Day 0"
    applications_df["Day"] = (applications_df["Date"] - day_zero_current_year_simulated).dt.days

    # **Filter out values before Day 0**
    applications_df = applications_df[applications_df["Day"] >= 0]

    # ✅ Ensure "Day 0" exists explicitly
    if 0 not in applications_df["Day"].values:
        day_zero_entry = pd.DataFrame({
            "Date": [day_zero_current_year_simulated],
            "Applications": [0],  # No applications at Day 0
            "Cumulative Applications (Current Year)": [0],
            "Applications (Previous Year)": [0],
            "Cumulative Applications (Previous Year)": [0],
            "Day": [0]
        })
        applications_df = pd.concat([day_zero_entry, applications_df]).sort_values("Day").reset_index(drop=True)

    # Simulate demand heatmap data for New Haven
    latitudes = np.random.uniform(41.26, 41.34, 100)
    longitudes = np.random.uniform(-72.95, -72.90, 100)
    demand_values = np.random.randint(1, 100, 100)
    heatmap_df = pd.DataFrame({"lat": latitudes, "lon": longitudes, "Demand": demand_values})

    # Simulate school demand data
    schools = [f"School {i}" for i in range(1, 21)]
    school_data = []
    for school in schools:
        for grade in grades:
            total_vacants = np.random.randint(20, 100)
            used_vacants = np.random.randint(0, total_vacants + 1)
            percent_used = round((used_vacants / total_vacants) * 100, 1)  # Round to 1 decimal place
            school_data.append([school, grade, total_vacants, used_vacants, percent_used])

    school_demand_df = pd.DataFrame(school_data, columns=["School", "Grade", "Total Vacants", "Used Vacants", "% Used"])

    # Simulate demand heatmap data for Applicants
    latitudes = np.random.uniform(41.26, 41.34, 100)
    longitudes = np.random.uniform(-72.95, -72.90, 100)
    demand_values = np.random.randint(1, 100, 100)
    heatmap_df = pd.DataFrame({"lat": latitudes, "lon": longitudes, "Demand": demand_values})

    # Simulate demand heatmap data for Schools (Random locations for now)
    school_latitudes = np.random.uniform(41.26, 41.34, len(schools))
    school_longitudes = np.random.uniform(-72.95, -72.90, len(schools))
    school_demand = np.random.randint(20, 100, len(schools))

    school_heatmap_df = pd.DataFrame({"lat": school_latitudes, "lon": school_longitudes, "Demand": school_demand})

    # Get top 10 most demanded schools
    most_demanded_df = school_demand_df.sort_values("% Used", ascending=False).head(10)
    # Get top 10 least demanded schools
    least_demanded_df = school_demand_df.sort_values("% Used", ascending=True).head(10)

    # Simulate data for the simulation section
    num_applicants = 1000
    assigned_fraction = np.random.uniform(0.6, 0.9, num_applicants)
    assignment_counts = np.random.dirichlet([4, 3, 2, 1, 1, 2, 2], size=1)[0] * num_applicants
    assignment_labels = [
        "1st Preference", "2nd Preference", "3rd Preference",
        "4th Preference", "+5th Preference", "Non-Assigned"
    ]

    # Ensure assignment_counts has the same length as assignment_labels
    num_preferences = len(assignment_labels)  # Ensure matching lengths
    assignment_counts = np.random.dirichlet([4, 3, 2, 1, 1, 2], size=1)[0] * num_applicants

    # Ensure the generated counts have the correct length
    if len(assignment_counts) != num_preferences:
        assignment_counts = np.append(assignment_counts, num_applicants - sum(assignment_counts))  # Adjust to match total

    assignment_df = pd.DataFrame({"Assignment": assignment_labels, "Fraction": (assignment_counts / 10).round(1)})

    # Generate simulated data
    num_applicants = 1000
    non_assigned_prob = np.random.rand(num_applicants)  # Probabilities of non-assignment


################# Auxiliary graphs

# Ensure the column names match what px.pie expects
applications_grade_df.rename(columns={'grade': 'Grade', 'applications': 'Applications'}, inplace=True)

# Create a donut chart for applications by grade
fig_donut = px.pie(
    applications_grade_df, 
    values="Applications", 
    names="Grade", 
    title="Applications by Grade",
    hole=0.4,  # Makes it a donut chart
    color_discrete_sequence=["#5AE0D3", "#5DDBDB", "#5627FF", "#713BF4", "#22114F"]  # Tether palette
)

# Kernel Density Estimation (KDE) for smooth density curve
kde = gaussian_kde(non_assigned_prob)
x_vals = np.linspace(0, 1, 100)
y_vals = kde(x_vals)

# Create the figure
fig1 = go.Figure()

# Add the histogram with cumulative probability (Change color here)
fig1.add_trace(go.Histogram(
    x=non_assigned_prob,
    nbinsx=30,
    histnorm='probability density',
    cumulative=dict(enabled=True),  # Enables cumulative probability display
    marker=dict(color='#22114F', opacity=1),  # Blue color for histogram
    name="Cumulative Probability"
))

# Add the density curve (Change color here)
fig1.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    line=dict(color='#5DDBDB', width=2),  # Red color for density curve
    name="Density Curve"
))

# Update layout for interactivity
fig1.update_layout(
    title="Probability Distribution of Non-Assignment",
    xaxis_title="Probability",
    yaxis_title="Cumulative Fraction",
    template="plotly_white",
    hovermode="x"  # Enables hover interaction along the x-axis
)

# Calculate differences and determine arrow direction for simulated data
# Calculate differences and determine arrow direction for simulated data
arrow_annotations = []
num_positive_arrows = 0  # Counter for positive arrows
num_negative_arrows = 0  # Counter for negative arrows

for i in range(len(applications_df)):
    day = applications_df["Day"].iloc[i]  # ✅ Use 'Day' instead of 'Date'
    current = applications_df["Applications"].iloc[i]
    previous = applications_df["Applications (Previous Year)"].iloc[i]
    difference = current - previous

    # Determine arrow properties
    if difference > 0:
        arrow_color = "#27AE60"  # Green for positive
        num_positive_arrows += 1  # Increase positive counter
        arrow_text = f"▲ {difference}"
    elif difference < 0:
        arrow_color = "#E74C3C"  # Red for negative
        num_negative_arrows += 1  # Increase negative counter
        arrow_text = f"▼ {abs(difference)}"
    else:
        continue  # Skip if no difference

    arrow_y_offset = max(current, previous) + 2  # Place arrow slightly above the highest bar

    # Create annotation
    arrow_annotations.append(
        dict(
            x=day,  # ✅ Use 'Day' (numeric) instead of 'Date' (datetime)
            y=arrow_y_offset,
            text=arrow_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor=arrow_color,
            ax=0, ay=-20,
            font=dict(size=12, color=arrow_color)
        )
    )


# Compute cumulative applications
applications_df["Cumulative Applications (Current Year)"] = applications_df["Applications"].cumsum()
applications_df["Cumulative Applications (Previous Year)"] = applications_df["Applications (Previous Year)"].cumsum()

# Select only weekly intervals for x-axis
weekly_days = applications_df["Day"][applications_df["Day"] % 7 == 0]  # Every 7 days

# Create figure for "Applications per Day"
fig = go.Figure()

fig.add_trace(go.Bar(
    x=applications_df["Day"],
    y=applications_df["Applications"],
    name="Applications (Current Year)",
    marker=dict(color="rgba(93, 219, 219, 0.7)"),
    hoverinfo="x+y"
))

fig.add_trace(go.Bar(
    x=applications_df["Day"],
    y=applications_df["Applications (Previous Year)"],
    name="Applications (Previous Year)",
    marker=dict(color="rgba(34, 17, 79, 0.7)"),
    hoverinfo="x+y"
))

fig.update_layout(
    title="Applications per Day (Current vs Previous Year)",
    template="plotly_white",
    barmode="overlay",
    xaxis_title="Day (relative to Day 0)",
    yaxis_title="Number of Applications",
    xaxis=dict(
        tickmode="array",
        tickvals=weekly_days,  # Show only weekly ticks
        range=[0, applications_df["Day"].max()]  # ✅ Ensures all annotations are visible
    ),
    hovermode="x",
    annotations=arrow_annotations  # ✅ Ensure annotations are added
)
# Create cumulative applications graph
fig_cumulative = go.Figure()

fig_cumulative.add_trace(go.Scatter(
    x=applications_df["Day"],
    y=applications_df["Cumulative Applications (Current Year)"],
    name="Cumulative Applications (Current Year)",
    mode="lines+markers",
    marker=dict(size=6, color="#5DDBDB"),
    line=dict(color="#5DDBDB", width=2),
    hoverinfo="x+y"
))

fig_cumulative.add_trace(go.Scatter(
    x=applications_df["Day"],
    y=applications_df["Cumulative Applications (Previous Year)"],
    name="Cumulative Applications (Previous Year)",
    mode="lines+markers",
    marker=dict(size=6, color="#22114F"),
    line=dict(color="#22114F", width=2),
    hoverinfo="x+y"
))

fig_cumulative.update_layout(
    title="Cumulative Applications (Current vs Previous Year)",
    template="plotly_white",
    xaxis_title="Day (relative to Day 0)",
    yaxis_title="Cumulative Applications",
    xaxis=dict(
        tickmode="array",
        tickvals=weekly_days,  # Show only weekly ticks
    ),
    hovermode="x"
)

tether_colors = ["#5AE0D3", "#5DDBDB", "#5627FF", "#713BF4", "#22114F"]  # Tether palette

################# Auxiliary stats

# Get total applicants (current & last year)
total_applicants = applications_df["Cumulative Applications (Current Year)"].iloc[-1]
total_applicants_last_year = applications_df["Cumulative Applications (Previous Year)"].iloc[-1]

# Calculate progress as a percentage
progress_percentage = (total_applicants / total_applicants_last_year) * 100 if total_applicants_last_year > 0 else 0

# Format date
progress_date = datetime.today().strftime("%B %d, %Y")  # Example: "February 12, 2025"

################# Create Dash
app = dash.Dash(__name__)
app.title = "New Haven Public Schools Dashboard - 2025"

# Layout
app.layout = html.Div([
    html.Div([
        # Left-side Logo (NHPS)
        html.Img(src='/assets/nhps.png',  
                 style={'height': '60px', 'marginRight': '15px'}),  

        # Title
        html.H1("School Choice Dashboard", 
                style={'textAlign': 'center', 'color': '#FFFFFF', 'backgroundColor': '#22114F', 
                       'padding': '20px', 'borderRadius': '10px', 'flexGrow': '1', 'margin': '0'}),

        # Right-side Logo (TETHER)
        html.Img(src='/assets/TETHER.png',  
                 style={'height': '60px', 'marginLeft': '15px'})  

    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'width': '100%'}),

    # Subtitle for the admission period
    html.H3("2025 Admission Period",  
            style={'textAlign': 'center', 'color': '#FFFFFF', 'backgroundColor': '#713BF4',  
                   'padding': '10px', 'borderRadius': '10px', 'width': '100%', 'fontWeight': 'normal'}),               
    
html.Div([
    html.H2("Summary", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.H3("Total Applicants", style={'color': '#22114F', 'textAlign': 'center'}),
            html.P(f"{total_applicants:,}", style={'fontSize': '24px', 'textAlign': 'center', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'padding': '10px'}),

        html.Div([
            html.H3("Total Applicants Last Year", style={'color': '#22114F', 'textAlign': 'center'}),
            html.P(f"{total_applicants_last_year:,}", style={'fontSize': '24px', 'textAlign': 'center', 'fontWeight': 'bold'})
        ], style={'flex': '1', 'padding': '10px'}),

        html.Div([
            html.H3(f"Progress as of {progress_date}", style={'color': '#22114F', 'textAlign': 'center'}),
            html.P(f"{progress_percentage:.1f}%", 
                   style={'fontSize': '24px', 'textAlign': 'center', 'fontWeight': 'bold', 
                          'color': '#E67E22' if progress_percentage < 100 else '#27AE60'})
        ], style={'flex': '1', 'padding': '10px'}),

        # New fourth statistic for positive vs negative arrow annotations
        html.Div([
            html.H3("Application Trends", style={'color': '#22114F', 'textAlign': 'center'}),
            html.P([
                html.Span(f"▲ {num_positive_arrows}  ", style={'color': '#27AE60', 'fontWeight': 'bold'}),  # Green for positive arrows
                html.Span(f"▼ {num_negative_arrows}", style={'color': '#E74C3C', 'fontWeight': 'bold'})  # Red for negative arrows
            ], style={'fontSize': '24px', 'textAlign': 'center'})
        ], style={'flex': '1', 'padding': '10px'}),

    ], style={'display': 'flex', 'justifyContent': 'space-around', 'backgroundColor': 'white', 
              'borderRadius': '10px', 'padding': '20px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'})
], style={'padding': '20px', 'backgroundColor': '#F4F6F6', 'borderRadius': '10px', 
          'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),
    

    html.Div([
        html.H2("Applications", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig_cumulative), 
        dcc.Graph(figure=fig_donut, style={'flex': '1'})  # Donut chart for grade distribution
        ], style={'padding': '20px', 'backgroundColor': '#ECF0F1', 'borderRadius': '10px', 'margin-bottom': '20px'}),
    
    # Layout update with dropdown for heatmap selection
    html.Div([
        html.H2("Map", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),

        html.Div([
            dcc.Graph(id="dynamic-heatmap", style={'flex': '1'}),  # 📌 Map goes first

            dcc.Dropdown(
                id="heatmap-selector",
                options=[
                    {"label": "Applicants", "value": "Applicants"},
                    {"label": "Schools", "value": "Schools"}
                ],
                value="Applicants",  # Default selection
                clearable=False,
                style={'width': '50%', 'margin': 'auto'}
            )  # 📌 Dropdown is placed BELOW the map
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'})  # Ensures vertical layout

    ], style={'padding': '20px', 'backgroundColor': '#ECF0F1', 'borderRadius': '10px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)', 'margin-bottom': '20px'}),
        
html.Div([
    html.H2("Simulation", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),

    # White Box that contains everything
    html.Div([
        # Row for Distribution Graph (Left) & Pie Chart (Right)
        html.Div([
            dcc.Graph(figure=fig1, style={'flex': '1'}),  # Interactive probability distribution graph
            dcc.Graph(
                figure=px.pie(
                    values=[sum(assignment_counts[:-1]), assignment_counts[-1]],
                    names=["Assigned", "Not Assigned"],
                    title="Assignment Fraction",
                    color_discrete_sequence=["#22114F", "#5DDBDB"]  # Green for Assigned, Red for Not Assigned
                ), 
                style={'flex': '1'}
            ),
        ], style={'display': 'flex', 'gap': '20px'}),  # Displays both graphs side by side

        # Assignment Preferences Table
        html.H3("Assignment Preferences", style={'color': '#22114F', 'textAlign': 'center'}),
        dash_table.DataTable(
            columns=[{"name": "Assignment", "id": "Assignment"}, {"name": "Percentage", "id": "Fraction"}],
            data=assignment_df.to_dict("records"),
            style_table={'overflowX': 'auto', 'margin': 'auto', 'width': '90%', 
                         'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '10px'},
            style_header={'backgroundColor': '#713BF4', 'color': 'white', 
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_data={'backgroundColor': '#ECF0F1', 'color': '#2C3E50', 'textAlign': 'center'}
        )
    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px',
              'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'})  # White background applied to everything inside

], style={'padding': '20px', 'backgroundColor': '#F4F6F6', 'borderRadius': '10px', 
          'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),

    html.Div([
        html.H2("Vacancies", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),
        html.Div([
            html.H3("Schools with Most Demand", style={'color': '#22114F', 'textAlign': 'center'}),
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in most_demanded_df.columns],
                data=most_demanded_df.to_dict("records"),
                style_table={'overflowX': 'auto', 'margin': 'auto', 'width': '90%', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '10px'},
                style_header={'backgroundColor': '#713BF4', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                style_data={'backgroundColor': '#ECF0F1', 'color': '#2C3E50', 'textAlign': 'center'}
            ),
            html.H3("Schools with Least Demand", style={'color': '#22114F', 'textAlign': 'center'}),
            dash_table.DataTable(
                columns=[{"name": col, "id": col} for col in least_demanded_df.columns],
                data=least_demanded_df.to_dict("records"),
                style_table={'overflowX': 'auto', 'margin': 'auto', 'width': '90%', 'backgroundColor': 'white', 'borderRadius': '10px', 'padding': '10px'},
                style_header={'backgroundColor': '#713BF4', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                style_data={'backgroundColor': '#ECF0F1', 'color': '#2C3E50', 'textAlign': 'center'}
            )
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'})
    ], style={'padding': '20px', 'backgroundColor': '#F4F6F6', 'borderRadius': '10px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),
    
    html.Footer("TetherEd, 2025", 
                style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#22114F', 'color': 'white'})

])

################# Callback to update heatmap dynamically
@app.callback(
    Output("dynamic-heatmap", "figure"),
    Input("heatmap-selector", "value")
)
def update_heatmap(selection):
    if selection == "Applicants":
        df = heatmap_df
        title = "Applicants Demand Heatmap"
    else:
        df = school_heatmap_df
        title = "School Demand Heatmap"

    fig = px.density_mapbox(
        df, lat='lat', lon='lon', z='Demand', radius=10,
        center={'lat': 41.30, 'lon': -72.92}, zoom=12,
        mapbox_style="open-street-map",
        title=title
    )

    return fig

# Run app locally
if __name__ == '__main__':
    app.run_server(debug=True)