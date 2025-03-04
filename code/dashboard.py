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
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from io import BytesIO
import base64
from datetime import datetime
import os
import glob

from db_connection import conect_bd

# Select input data
data_source = "schoolmint" # select source of data between: "backend", "schoolmint", "simulated"
year = 2025
day_zero_current_year = pd.to_datetime("2025-02-03")  # Adjust the format
day_zero_previous_year = pd.to_datetime("2024-01-30")  # Adjust the format

last_day_current_year = pd.to_datetime("2025-03-07")
last_day_previous_year = pd.to_datetime("2025-03-02")

############

if data_source == "backend":
    
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
        SELECT l.grade_name_en AS Grade, COUNT(DISTINCT g.applicant_id) AS Applications
        FROM {tenant}.registration_applicant_interested_grade g
        JOIN {tenant}.institutions_grade_label l ON g.grade_label_id = l.id
        WHERE EXTRACT(YEAR FROM created) = {year}
        GROUP BY l.grade_name_en
        ORDER BY l.grade_name_en;
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
    
    # Ensure the column names match what px.pie expects
    applications_grade_df.rename(columns={'grade': 'Grade', 'applications': 'Applications'}, inplace=True)

    # Define the strict order for grades
    grade_order = ["PreK-3", "PreK-4", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade",
                "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", "11th Grade", "12th Grade"]    

elif data_source == "schoolmint":
    import os

    # Get the current script directory dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define base path relative to the script directory
    base_path = os.path.join(script_dir, "../data/inputs")  # Move up one level and into `data/inputs`
    base_path = os.path.abspath(base_path)  # Convert to absolute path

    print(base_path)  # Debugging: Ensure it's correct

    def load_csvs(year):
        """Load all CSVs for a given year into a dictionary of DataFrames."""
        folder_path = os.path.join(base_path, str(year))
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        dataframes = {}

        for file in csv_files:
            file_name = os.path.basename(file).replace(".csv", "")
            df = pd.read_csv(file)
            dataframes[file_name] = df
            print(f"Loaded: {file_name} | Variables: {list(df.columns)}")
        
        return dataframes

    # ✅ Load CSV files for both years
    data_2024 = load_csvs(2024)
    data_2025 = load_csvs(2025)

    # ✅ Load datasets
    applications1_2024 = data_2024['applications1']
    applications2_2024 = data_2024['applications2']
    applications1_2025 = data_2025['applications1']
    applications2_2025 = data_2025['applications2']
    schools_df = data_2025['schools']
    students_df = data_2025['students']
    students_annual_df = data_2025['students_annual']
    programs_df = data_2025['programs']
    siblings_df = data_2025['siblings']
    guardian_account_df = data_2025['guardian_account']

    # ✅ Rename 'id' to 'student_id' in students_df to match students_annual_df
    students_df.rename(columns={'id': 'student_id'}, inplace=True)

    # ✅ Convert 'student_id' and 'id' to integer before merging
    for df in [applications2_2025, applications2_2024]:
        df['id'] = df['id'].astype('Int64')  # Preserve NaNs
        df['student_id'] = df['student_id'].astype('Int64')

    # ✅ Merge applications1 with applications2 to retrieve `student_id`
    def merge_applications(app1, app2):
        return (
            app1.merge(app2[['id', 'student_id', 'created_at', 'program_id']], left_on='App#', right_on='id', how='left')
            .drop(columns=['id'])  # Drop duplicate ID column
        )

    applications_current_year_df = merge_applications(applications1_2025, applications2_2025)
    applications_previous_year_df = merge_applications(applications1_2024, applications2_2024)

    # ✅ Ensure 'Date' is correctly formatted as a datetime object
    applications_current_year_df["Date"] = pd.to_datetime(applications_current_year_df["created_at"]).dt.date
    applications_previous_year_df["Date"] = pd.to_datetime(applications_previous_year_df["created_at"]).dt.date

    # ✅ Convert 'Date' column to datetime, ensuring proper format and handling NaNs
    applications_current_year_df["Date"] = pd.to_datetime(applications_current_year_df["Date"], errors='coerce')
    applications_previous_year_df["Date"] = pd.to_datetime(applications_previous_year_df["Date"], errors='coerce')

    # ✅ Drop NaN values in 'Date' before computing the minimum
    applications_current_year_df = applications_current_year_df.dropna(subset=["Date"])
    applications_previous_year_df = applications_previous_year_df.dropna(subset=["Date"])

    # ✅ Drop duplicates at the student-date level
    applications_current_year_df = applications_current_year_df.drop_duplicates(subset=["Date", "student_id"])
    applications_previous_year_df = applications_previous_year_df.drop_duplicates(subset=["Date", "student_id"])

    # ✅ Aggregate daily applications by unique students
    applications_daily_counts_df = (
        applications_current_year_df.groupby("Date")["student_id"]
        .nunique()
        .reset_index(name="Applications")
    )

    applications_previous_year_counts_df = (
        applications_previous_year_df.groupby("Date")["student_id"]
        .nunique()
        .reset_index(name="Applications (Previous Year)")
    )

    # ✅ Compute cumulative applications
    applications_daily_counts_df["Cumulative Applications (Current Year)"] = applications_daily_counts_df["Applications"].cumsum()
    applications_previous_year_counts_df["Cumulative Applications (Previous Year)"] = applications_previous_year_counts_df["Applications (Previous Year)"].cumsum()

    # ✅ Compute "Day" since "Day 0"
    applications_daily_counts_df["Date"] = pd.to_datetime(applications_daily_counts_df["Date"])
    applications_previous_year_counts_df["Date"] = pd.to_datetime(applications_previous_year_counts_df["Date"])

    day_zero_current_year = pd.Timestamp(day_zero_current_year)
    day_zero_previous_year = pd.Timestamp(day_zero_previous_year)

    applications_daily_counts_df["Day"] = (applications_daily_counts_df["Date"] - day_zero_current_year).dt.days
    applications_previous_year_counts_df["Day"] = (applications_previous_year_counts_df["Date"] - day_zero_previous_year).dt.days

    # ✅ Filter applications up to the given deadlines
    applications_daily_counts_df = applications_daily_counts_df[applications_daily_counts_df["Date"] <= last_day_current_year]
    applications_previous_year_counts_df = applications_previous_year_counts_df[applications_previous_year_counts_df["Date"] <= last_day_previous_year]

    # **Filter out values before Day 0**
    applications_daily_counts_df = applications_daily_counts_df[applications_daily_counts_df["Day"] >= 0]
    applications_previous_year_counts_df = applications_previous_year_counts_df[applications_previous_year_counts_df["Day"] >= 0]

    # ✅ Get today's date
    today_date = pd.Timestamp(datetime.today().date())

    # ✅ Find the relative "Day" in 2025
    latest_day_2025 = (today_date - day_zero_current_year).days

    # ✅ Find the equivalent "Day" in 2024 based on the relative position
    equivalent_day_2024 = (day_zero_previous_year + pd.Timedelta(days=latest_day_2025)).date()

    # ✅ Trim 2025 applications **up to today**
    applications_per_day_filtered_df = applications_daily_counts_df[
        applications_daily_counts_df["Day"] <= latest_day_2025
    ]

    # ✅ Trim 2024 applications **up to the equivalent day in 2024**
    applications_previous_year_filtered_df = applications_previous_year_counts_df[
        applications_previous_year_counts_df["Date"] <= pd.Timestamp(equivalent_day_2024)
    ]

    # ✅ Merge **filtered 2024** (up to equivalent day) with **filtered 2025** (up to today)
    applications_df_filtered = applications_previous_year_filtered_df.merge(
        applications_per_day_filtered_df, on="Day", how="left", suffixes=(" (Previous Year)", "")
    ).fillna(0)

    # ✅ Merge **full 2024** with **full 2025** (unfiltered version for cumulative applications)
    applications_df = applications_previous_year_counts_df.merge(
        applications_daily_counts_df, on="Day", how="left", suffixes=(" (Previous Year)", "")
    ).fillna(0)

    # ✅ Use `applications_df_filtered` only for the "Applicants per Day" graph
    applications_per_day_df = applications_df_filtered.copy()

    # ✅ Count unique student applications per grade
    applications_grade_df = (
        applications_current_year_df.groupby("Grade")["student_id"]
        .nunique()
        .reset_index(name="Applications")
    )

    # ✅ Ensure both columns have the same type (string) before merging
    applications_current_year_df["Program"] = applications_current_year_df["Program"].astype(str)
    schools_df["id"] = schools_df["id"].astype(str)

    # ✅ Merge applications with students_annual_df and schools_df
    school_demand_df = (
        applications_current_year_df
        .merge(students_annual_df[['student_id', 'grade']], on='student_id', how='left')
        .merge(schools_df[['id', 'school_name']], left_on='Program', right_on='id', how='left')
        .groupby(['school_name', 'grade'])
        .size()
        .reset_index(name="Total Applications")
    )

    # Convert float columns to integers where possible
    columns_to_fix = [
        "Applications", 
        "Cumulative Applications (Current Year)"
    ]

    for col in columns_to_fix:
        applications_per_day_df[col] = applications_per_day_df[col].fillna(0).astype(int)

    # ✅ Handle missing school names
    school_demand_df = school_demand_df.dropna(subset=['school_name'])

    # ✅ Simulate "Used Vacants" (Replace with actual data if available)
    school_demand_df["Total Vacants"] = np.random.randint(20, 100, size=len(school_demand_df))
    school_demand_df["Used Vacants"] = np.minimum(school_demand_df["Total Applications"], school_demand_df["Total Vacants"])
    school_demand_df["% Used"] = (school_demand_df["Used Vacants"] / school_demand_df["Total Vacants"]) * 100

    # ✅ Create demand heatmap data
    heatmap_df = (
        students_df[['student_id', 'birth_city']]
        .merge(students_annual_df[['student_id', 'address_lat', 'address_lng']], on='student_id', how='left')
        .groupby(['address_lat', 'address_lng']).size().reset_index(name="Demand")
    )

    # ✅ Ensure both columns are the same type before merging
    schools_df["id"] = schools_df["id"].astype(str)
    applications_current_year_df["student_id"] = applications_current_year_df["student_id"].astype(str)

    # ✅ Rename `address_lat` and `address_lng` in applicant_heatmap_df
    schools_df.rename(columns={'lng': 'lon'}, inplace=True)

    # ✅ Ensure ID types match before merging
    applications_current_year_df["program_id"] = applications_current_year_df["program_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    programs_df["id"] = programs_df["id"].astype(str)
    programs_df["school_id"] = programs_df["school_id"].astype(str)
    schools_df["id"] = schools_df["id"].astype(str)
    
    # ✅ Merge applications → programs → schools
    school_heatmap_df = (
        applications_current_year_df
        .merge(programs_df[['id', 'school_id']], left_on='program_id', right_on='id', how='left')
        .drop(columns=['id'])  # Drop duplicate 'id' from programs_df
        .merge(schools_df[['id', 'lat', 'lon', 'school_name']], left_on='school_id', right_on='id', how='left')
        .drop(columns=['id'])  # Drop duplicate 'id' from schools_df
        .groupby(['lat', 'lon', 'school_name'])['student_id']
        .nunique()
        .reset_index(name="Total Applicants")  # Count unique applicants per school
    )

    #Rename Total Applicants to Demand
    school_heatmap_df.rename(columns={'Total Applicants': 'Demand'}, inplace=True)

    # ✅ Rename `address_lat` and `address_lng` in applicant_heatmap_df
    heatmap_df.rename(columns={'address_lat': 'lat', 'address_lng': 'lon'}, inplace=True)

    # ✅ Convert lat/lon to numeric & drop NaN
    for df in [heatmap_df, school_heatmap_df]:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df.dropna(subset=['lat', 'lon'], inplace=True)

    # ✅ Standardize Grade Names
    grade_mapping = {
        "PreK3": "PreK-3", "PreK4": "PreK-4", "K": "Kindergarten",
        "1": "1st Grade", "2": "2nd Grade", "3": "3rd Grade", "4": "4th Grade",
        "5": "5th Grade", "6": "6th Grade", "7": "7th Grade", "8": "8th Grade",
        "9": "9th Grade", "10": "10th Grade", "11": "11th Grade", "12": "12th Grade"
    }

    applications_grade_df["Grade"] = applications_grade_df["Grade"].map(grade_mapping)

    grade_order = ["PreK-3", "PreK-4", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade",
                "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", "11th Grade", "12th Grade"]
    

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

    most_demanded_df = school_demand_df.sort_values("Total Applications", ascending=False).head(10)
    least_demanded_df = school_demand_df.sort_values("Total Applications", ascending=True).head(10)


elif data_source == "simulated":
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

# Ensure the DataFrame follows the strict order
applications_grade_df["Grade"] = pd.Categorical(applications_grade_df["Grade"], categories=grade_order, ordered=True)
applications_grade_df = applications_grade_df.sort_values("Grade", ascending=True)  # Enforce the specified order

# Calculate relative percentage
applications_grade_df["Percentage"] = (applications_grade_df["Applications"] / applications_grade_df["Applications"].sum()) * 100

# Define a custom color scale based on the given color palette
color_palette = ["#0C1461", "#313F89", "#5627FF", "#5DDBDB", "#5AE0D3"]  # Darker to lighter
num_colors = len(color_palette)

# Normalize application counts to distribute colors
applications_grade_df["Color Index"] = applications_grade_df["Applications"].rank(pct=True)  # Scale between 0-1
applications_grade_df["Color Index"] = (applications_grade_df["Color Index"] * (num_colors - 1)).astype(int)  # Convert to index
applications_grade_df["Color"] = applications_grade_df["Color Index"].apply(lambda x: color_palette[x])  # Assign colors

# Create a horizontal bar chart with correct order
fig_bar = px.bar(
    applications_grade_df, 
    x="Applications", 
    y="Grade", 
    orientation="h",  # Horizontal bars
    text="Applications",  # Show applicant count inside bars
    title="Applicants by Interested Grade",
    color=applications_grade_df["Color"],  # Use dynamically assigned colors
    color_discrete_map="identity",  # Uses the manually mapped colors instead of a default scale
    category_orders={"Grade": grade_order}  # ✅ Forces the correct order!
)

# Update layout and fonts
fig_bar.update_layout(
    title=dict(text="Applicants by Grade", font=dict(family="Inter, sans-serif", size=18)),
    xaxis=dict(
        title=dict(text="Number of Applicants", font=dict(family="Inter, sans-serif", size=14))
    ),
    yaxis=dict(
        title=dict(text="Grade", font=dict(family="Inter, sans-serif", size=14)),
        tickfont=dict(family="Inter, sans-serif", size=12)
    ),
    template="plotly_white",
    font=dict(family="Inter, sans-serif")
)

# Update text labels to be inside the bars
fig_bar.update_traces(textposition="inside")

# Add relative percentage labels at the top of each bar
for i, row in applications_grade_df.iterrows():
    fig_bar.add_annotation(
        x=row["Applications"] + max(applications_grade_df["Applications"]) * 0.05,  # Offset text slightly to the right
        y=row["Grade"],
        text=f"{row['Percentage']:.1f}%",  # Show percentage above bar
        showarrow=False,
        font=dict(family="Inter, sans-serif", size=12, color="black")
    )

fig_bar.update_traces(
    textposition="inside",
    textangle=0  # ✅ Ensures all text inside bars is horizontal
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

# Update layout for interactivity with Inter font
fig1.update_layout(
    title={
        "text": "Probability Distribution of Non-Assignment",
        "font": {"family": "Inter, sans-serif", "size": 18}  # ✅ Title in Inter
    },
    xaxis_title="Probability",
    yaxis_title="Cumulative Fraction",
    xaxis={
        "titlefont": {"family": "Inter, sans-serif", "size": 14},  # ✅ X-axis title in Inter
        "tickfont": {"family": "Inter, sans-serif", "size": 12}  # ✅ X-axis tick labels in Inter
    },
    yaxis={
        "titlefont": {"family": "Inter, sans-serif", "size": 14},  # ✅ Y-axis title in Inter
        "tickfont": {"family": "Inter, sans-serif", "size": 12}  # ✅ Y-axis tick labels in Inter
    },
    template="plotly_white",
    hovermode="x",  # Enables hover interaction along the x-axis
    font={"family": "Inter, sans-serif"}  # ✅ Ensures all text elements (legend, annotations) use Inter
)

# Calculate differences and determine arrow direction for simulated data
arrow_annotations = []
num_positive_arrows = 0  # Counter for positive arrows
num_negative_arrows = 0  # Counter for negative arrows

for i in range(len(applications_per_day_df)):
    day = applications_per_day_df["Day"].iloc[i]  # ✅ Use 'Day' instead of 'Date'
    current = applications_per_day_df["Applications"].iloc[i]
    previous = applications_per_day_df["Applications (Previous Year)"].iloc[i]
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
    x=applications_per_day_df["Day"],
    y=applications_per_day_df["Applications"],
    name="Applicants (Current Year)",
    marker=dict(color="rgba(93, 219, 219, 0.7)"),
    hoverinfo="x+y"
))

fig.add_trace(go.Bar(
    x=applications_per_day_df["Day"],
    y=applications_per_day_df["Applications (Previous Year)"],
    name="Applicants (Previous Year)",
    marker=dict(color="rgba(34, 17, 79, 0.7)"),
    hoverinfo="x+y"
))

fig.update_layout(
    title={
        "text": "Applicants per Day (Current vs Previous Year)",
        "font": {"family": "Inter, sans-serif", "size": 18}  # ✅ Apply Inter to title
    },
    template="plotly_white",
    barmode="overlay",
    xaxis=dict(
        title={"text": "Day (relative to Day 0)", "font": {"family": "Inter, sans-serif", "size": 14}},
        tickmode="array",
        tickvals=weekly_days,  # Show only weekly ticks
        range=[0, applications_per_day_df["Day"].max()]  # ✅ Ensures all annotations are visible
    ),
    yaxis=dict(
        title={"text": "Number of Applicants", "font": {"family": "Inter, sans-serif", "size": 14}}
    ),
    font={"family": "Inter, sans-serif"},  # ✅ Apply Inter globally to the entire figure
    hovermode="x",
    annotations=arrow_annotations  # ✅ Ensure annotations are added
)
# Create cumulative applications graph
fig_cumulative = go.Figure()

fig_cumulative.add_trace(go.Scatter(
    x=applications_df["Day"],
    y=applications_df["Cumulative Applications (Current Year)"],
    name="Cumulative Applicants (Current Year)",
    mode="lines+markers",
    marker=dict(size=6, color="#5DDBDB"),
    line=dict(color="#5DDBDB", width=2),
    hoverinfo="x+y"
))

fig_cumulative.add_trace(go.Scatter(
    x=applications_df["Day"],
    y=applications_df["Cumulative Applications (Previous Year)"],
    name="Cumulative Applicants (Previous Year)",
    mode="lines+markers",
    marker=dict(size=6, color="#22114F"),
    line=dict(color="#22114F", width=2),
    hoverinfo="x+y"
))

fig_cumulative.update_layout(
    title={
        "text": "Cumulative Applicants (Current vs Previous Year)",
        "font": {"family": "Inter, sans-serif", "size": 18}  # ✅ Apply Inter to title
    },
    template="plotly_white",
    xaxis=dict(
        title={"text": "Day (relative to Day 0)", "font": {"family": "Inter, sans-serif", "size": 14}},
        tickmode="array",
        tickvals=weekly_days,  # Show only weekly ticks
    ),
    yaxis=dict(
        title={"text": "Cumulative Applicants", "font": {"family": "Inter, sans-serif", "size": 14}}
    ),
    font={"family": "Inter, sans-serif"},  # ✅ Apply Inter globally to the entire figure
    hovermode="x"
)

tether_colors = ["#5AE0D3", "#5DDBDB", "#5627FF", "#713BF4", "#22114F"]  # Tether palette

################# Auxiliary stats

# Get total applicants (current & last year)
total_applicants = applications_df["Cumulative Applications (Current Year)"].iloc[-1]
total_applicants_last_year = applications_df["Cumulative Applications (Previous Year)"].iloc[-1]

# Convert to integer if it's a whole number, otherwise keep as float
if total_applicants % 1 == 0:
    total_applicants = int(total_applicants)
else:
    total_applicants = float(total_applicants)

# Calculate progress as a percentage
progress_percentage = (total_applicants / total_applicants_last_year) * 100 if total_applicants_last_year > 0 else 0

# Format date
progress_date = datetime.today().strftime("%B %d, %Y")  # Example: "February 12, 2025"

################# Create Dash
app = dash.Dash(__name__, external_stylesheets=['/assets/styles.css'])
app.title = "New Haven Public Schools Dashboard - 2025"

# Layout
app.layout = html.Div([
    html.Div([
        # Left-side Logo (NHPS)
        html.Img(src='/assets/nhps.png',  
                 style={'height': '60px', 'marginRight': '15px'}),  

        # Title
        html.H1("School Choice Dashboard", 
                style={'textAlign': 'center', 'color': '#FFFFFF', 'backgroundColor': '#0C1461', 
                       'padding': '20px', 'borderRadius': '10px', 'flexGrow': '1', 'margin': '0'}),

        # Right-side Logo (TETHER)
        html.Img(src='/assets/TETHER.png',  
                 style={'height': '60px', 'marginLeft': '15px'})  

    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'width': '100%'}),

    # Subtitle for the admission period
    html.H3("2025 Admission Period",  
            style={'textAlign': 'center', 'color': '#FFFFFF', 'backgroundColor': '#0B114D',  
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
                          'color': '#D7005A' if progress_percentage < 100 else '#28B911'})
        ], style={'flex': '1', 'padding': '10px'}),

        # New fourth statistic for positive vs negative arrow annotations
        html.Div([
            html.H3("Application Trends", style={'color': '#22114F', 'textAlign': 'center'}),
            html.P([
                html.Span(f"▲ {num_positive_arrows}  ", style={'color': '#28B911', 'fontWeight': 'bold'}),  # Green for positive arrows
                html.Span(f"▼ {num_negative_arrows}", style={'color': '#D7005A', 'fontWeight': 'bold'})  # Red for negative arrows
            ], style={'fontSize': '24px', 'textAlign': 'center'})
        ], style={'flex': '1', 'padding': '10px'}),

    ], style={'display': 'flex', 'justifyContent': 'space-around', 'backgroundColor': 'white', 
              'borderRadius': '10px', 'padding': '20px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'})
], style={'padding': '20px', 'backgroundColor': '#F4F7FF', 'borderRadius': '10px', 
          'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),
    

html.Div([
    html.H2("Applications", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),

    # 🔹 White Background Applied Here
    html.Div([
        # Row for main application charts
        html.Div([
            dcc.Graph(figure=fig, style={'width': '50%'}),  # Applications per Day
            dcc.Graph(figure=fig_cumulative, style={'width': '50%'}),  # Cumulative Applications
        ], style={'display': 'flex', 'gap': '20px'}),

        # Row for Applications by Grade (Donut Chart) + Heatmap with Dropdown
        html.Div([
            # Donut Chart on the Left
            html.Div([
                dcc.Graph(figure=fig_bar)
            ], style={'width': '50%'}),  # Donut Chart Container

            # Heatmap + Dropdown on the Right
            html.Div([
                dcc.Graph(id="dynamic-heatmap", style={'height': '450px', 'width': '100%', 'padding': '10px'}),

                # Heatmap Selection Dropdown
                dcc.Dropdown(
                    id="heatmap-selector",
                    options=[
                        {"label": "Applicants", "value": "Applicants"},
                        {"label": "Schools", "value": "Schools"}
                    ],
                    value="Applicants",  # Default selection
                    clearable=False,
                    style={'width': '80%', 'margin': 'auto'}
                )
            ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})  # Centered layout
        ], style={'display': 'flex', 'gap': '20px'}),  # Ensures side-by-side layout
    ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px'}),  # 🔹 Applied White Background to Whole Section

], style={'padding': '20px', 'backgroundColor': '#F4F7FF', 'borderRadius': '10px', 'margin-bottom': '20px'}),
        
html.Div([
    html.H2("Simulation (UNDER CONSTRUCTION)", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),

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

], style={'padding': '20px', 'backgroundColor': '#F4F7FF', 'borderRadius': '10px', 
          'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),

    html.Div([
        html.H2("Vacancies (UNDER CONSTRUCTION)", style={'color': '#2C3E50', 'textAlign': 'center', 'margin-bottom': '20px'}),
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
    ], style={'padding': '20px', 'backgroundColor': '#F4F7FF', 'borderRadius': '10px', 'boxShadow': '2px 2px 12px rgba(0,0,0,0.1)'}),
    
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

    # Create heatmap figure
    fig = px.density_mapbox(
        df, lat='lat', lon='lon', z='Demand', radius=10,
        center={'lat': 41.30, 'lon': -72.92}, zoom=12,
        mapbox_style="open-street-map",
        title=title
    )

    # ✅ Move the color bar to the bottom
    fig.update_layout(
        title={"text": title, "font": {"family": "Inter, sans-serif", "size": 18}},
        font={"family": "Inter, sans-serif"},
        coloraxis_colorbar=dict(
            title="Demand",
            titlefont={"family": "Inter, sans-serif", "size": 14},
            orientation="h",  # ✅ Horizontal orientation
            x=0.5,  # ✅ Center align
            xanchor="center",
            y=-0.2,  # ✅ Move below the graph
            thicknessmode="pixels", thickness=10,  # ✅ Adjust thickness
            lenmode="fraction", len=0.7  # ✅ Adjust length (70% of width)
        )
    )

    return fig

# Run app locally
if __name__ == '__main__':
    app.run_server(debug=True)