import datetime
import random
import warnings
from datetime import *

import dash
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pmdarima as pm
import statsmodels.api as sm
from dash import dash_table, dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")


# Function to generate patient data
def generate_patient_data(start_date, end_date, frequency, num_patients):
    data = []
    current_date = start_date  # Initialize current_date as a datetime object

    while current_date <= end_date:
        for patient_id in range(1, num_patients + 1):
            blood_sugar = np.random.uniform(70, 200)
            blood_pressure = np.random.uniform(90, 140)
            cholesterol = np.random.uniform(150, 250)
            heart_rate = np.random.uniform(60, 100)

            # Additional columns you want to include
            weight = np.random.uniform(50, 100)
            height = np.random.uniform(150, 190)
            bmi = weight / ((height / 100) ** 2)

            # Introduce anomalies for patient ID 5
            if patient_id == 5 and current_date >= datetime(2023, 6, 1):
                blood_sugar *= 1.5  # Increase blood sugar for anomalies
                blood_pressure *= 1.2  # Increase blood pressure for anomalies
                cholesterol *= 1.2

            data.append({
                'PatientID': patient_id,
                'Timestamp': current_date,
                'BloodSugar': blood_sugar,
                'BloodPressure': blood_pressure,
                'Cholesterol': cholesterol,
                'HeartRate': heart_rate,
                'Weight': weight,
                'Height': height,
                'BMI': bmi
            })

        current_date += timedelta(days=1)

    return pd.DataFrame(data)


# Set parameters for data generation
start_date = datetime(2023, 2, 1)
end_date = datetime.now()
frequency = 'D'
num_patients = 10


# Generate patient data
patient_data = generate_patient_data(
    start_date, end_date, frequency, num_patients)

# Save patient data to a CSV file
patient_data.to_csv('patient_data.csv', index=False)

# Function to generate general health conditions data


def generate_conditions_data(num_patients):
    conditions = ['Healthy', 'Diabetes', 'Hypertension',
                  'High Cholesterol', 'Heart Disease']
    data = []

    for patient_id in range(1, num_patients + 1):
        patient_conditions = random.sample(
            conditions, random.randint(1, len(conditions)))
        data.append({
            'PatientID': patient_id,
            'Conditions': patient_conditions
        })
    return pd.DataFrame(data)


# Set the number of patients
num_patients = 10

# Generate general health conditions data
conditions_data = generate_conditions_data(num_patients)


# Save conditions data to a CSV file
conditions_data.to_csv('conditions_data.csv', index=False)

# Perform clustering using K-means algorithm


def perform_clustering(patient_data, num_clusters):
    features = ['BloodSugar', 'BloodPressure', 'Cholesterol', 'HeartRate']
    data_for_clustering = patient_data[features].values

    # Apply feature scaling using Standardization (Z-score scaling)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)  # You can change this to 3 for a 3D scatter plot
    reduced_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    patient_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Add PCA components to patient_data for visualization
    patient_data['PCA1'] = reduced_data[:, 0]
    patient_data['PCA2'] = reduced_data[:, 1]

    return patient_data


# Set parameters
start_date = datetime(2023, 2, 1)
end_date = datetime.now()
frequency = 'D'
num_patients = 10
num_clusters = 3  # Set the number of clusters


# Generate patient health data
patient_data = generate_patient_data(
    start_date, end_date, frequency, num_patients)
conditions_data = generate_conditions_data(num_patients)
patient_data = patient_data.drop_duplicates()
# Example: Group by 'Timestamp' and 'PatientID', and then aggregate the data
patient_data_grouped = patient_data.groupby(
    ['Timestamp', 'PatientID']).mean().reset_index()

# Perform clustering
patient_data = perform_clustering(patient_data, num_clusters)

app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("HealthCare Advance Data Analytics Dashboard"),

    # Interactive widgets
    html.H2("Patient ID: "),
    dcc.Input(

        id='patient-input',
        type='number',
        placeholder='Enter Patient ID',
        value=5
    ),
    html.H2("Start to End: "),
    dcc.DatePickerRange(
        id='date-range-input',
        start_date=start_date,
        end_date=end_date,
        display_format='YYYY-MM-DD',
        initial_visible_month=start_date
    ),

    # Div for test results
    html.Div(id='test-results'),


    # Placeholder for various graphs
    dcc.Graph(id='sugar-chart'),
    dcc.Graph(id='bp-chart'),
    dcc.Graph(id='cholesterol-chart'),
    dcc.Graph(id='cluster-chart'),
    dcc.Graph(id='correlation-heatmap'),
    dcc.Graph(id='violin-plot'),
    dcc.Graph(id='decomposition-plot'),
    dcc.Graph(id='parallel-coordinates-plot'),
    dcc.Graph(id='health-parameters-scatter'),
    dcc.Graph(id='heart-rate-chart'),
    dcc.Graph(id='scatter-3d-plot'),
    dcc.Graph(id='arima-forecast-sugar-chart'),
    dcc.Graph(id='arima-forecast-bp-chart'),
    dcc.Graph(id='arima-forecast-heart-rate-chart'),
    dcc.Graph(id='condition-network')


])


@app.callback(

    [Output('sugar-chart', 'figure'),
     Output('bp-chart', 'figure'),
     Output('cholesterol-chart', 'figure'),
     Output('cluster-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('violin-plot', 'figure'),
     Output('decomposition-plot', 'figure'),
     Output('parallel-coordinates-plot', 'figure'),
     Output('health-parameters-scatter', 'figure'),
     Output('heart-rate-chart', 'figure'),
     Output('scatter-3d-plot', 'figure'),
     Output('test-results', 'children')],
    [Input('patient-input', 'value'),
     Input('date-range-input', 'start_date'),
     Input('date-range-input', 'end_date')],
)
def update_graphs(patient_id, start_date, end_date):
    if patient_id is None:
        return no_update

    # Remove the "T00:00:00" time part from the date strings
    start_date = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
    end_date = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')

    patient_df = patient_data[patient_data['PatientID'] == patient_id]
    conditions_df = conditions_data[conditions_data['PatientID'] == patient_id]

    # Filter patient data based on the selected date range
    filtered_patient_data = patient_data[
        (patient_data['PatientID'] == patient_id) &
        (patient_data['Timestamp'] >= start_date) &
        (patient_data['Timestamp'] <= end_date)
    ]

    # Generate charts using filtered_patient_data
    sugar_fig = px.line(filtered_patient_data, x='Timestamp',
                        y='BloodSugar', title='Blood Sugar Trend')
    bp_fig = px.line(filtered_patient_data, x='Timestamp',
                     y='BloodPressure', title='Blood Pressure Trend')
    cholesterol_fig = px.line(
        filtered_patient_data, x='Timestamp', y='Cholesterol', title='Cholesterol Trend')
    heart_rate_fig = px.line(filtered_patient_data, x='Timestamp',
                             y='HeartRate', title='Heart Rate Trend')

    # Add anomaly lines for all patients

    for patient_id in range(1, num_patients + 1):
        if patient_id == 5:  # Skip patient 5 to avoid duplicating anomaly markers
            continue

        patient_anomalies = patient_data[patient_data['PatientID'] == patient_id]
        anomaly_dates = patient_anomalies['Timestamp']
        anomaly_low_threshold = 50  # Set the low threshold for low-value anomalies

        sugar_fig.add_trace(go.Scatter(x=anomaly_dates, y=[anomaly_low_threshold] * len(
            anomaly_dates), mode='lines', line=dict(color='orange', dash='dot'), name=f'Low Anomalies (Patient {patient_id})'))
        sugar_fig.add_trace(go.Scatter(x=anomaly_dates, y=[200] * len(anomaly_dates), mode='lines', line=dict(
            color='red', dash='dot'), name=f'High Anomalies (Patient {patient_id})'))

        bp_fig.add_trace(go.Scatter(x=anomaly_dates, y=[80] * len(anomaly_dates), mode='lines', line=dict(
            color='orange', dash='dot'), name=f'Low Anomalies (Patient {patient_id})'))
        bp_fig.add_trace(go.Scatter(x=anomaly_dates, y=[140] * len(anomaly_dates), mode='lines', line=dict(
            color='red', dash='dot'), name=f'High Anomalies (Patient {patient_id})'))

        cholesterol_fig.add_trace(go.Scatter(x=anomaly_dates, y=[125] * len(anomaly_dates), mode='lines', line=dict(
            color='orange', dash='dot'), name=f'Low Anomalies (Patient {patient_id})'))
        cholesterol_fig.add_trace(go.Scatter(x=anomaly_dates, y=[250] * len(anomaly_dates), mode='lines', line=dict(
            color='red', dash='dot'), name=f'High Anomalies (Patient {patient_id})'))

    # Generate cluster visualization
    cluster_fig = px.scatter(patient_data, x='PCA1', y='PCA2',
                             color='Cluster', title='Cluster Visualization')

    # Update axis labels
    cluster_fig.update_xaxes(title_text='PCA Component 1')
    cluster_fig.update_yaxes(title_text='PCA Component 2')

    # Calculate correlation matrix
    correlation_matrix = patient_df[[
        'BloodSugar', 'BloodPressure', 'Cholesterol', 'HeartRate']].corr()

    # Create heatmap for correlation analysis
    heatmap_fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                            x=correlation_matrix.columns,
                                            y=correlation_matrix.columns,
                                            colorscale='Viridis'))
    heatmap_fig.update_layout(title='Correlation Heatmap')

    # Create violin plots for distribution comparison
    violin_fig = px.violin(patient_data, y='BloodSugar', x='Cluster',
                           box=True, title='Blood Sugar Distribution by Cluster')

    # Generate heart rate chart using filtered_patient_data
    heart_rate_fig = px.line(filtered_patient_data, x='Timestamp',
                             y='HeartRate', title='Heart Rate Trend')

    # Interactive Time Series Decomposition Plot
    decomposition = seasonal_decompose(
        patient_df['BloodSugar'], model='additive', period=7)
    decomposition_fig = go.Figure()
    decomposition_fig.add_trace(go.Scatter(
        x=patient_df['Timestamp'], y=decomposition.trend, mode='lines', name='Trend'))
    decomposition_fig.add_trace(go.Scatter(
        x=patient_df['Timestamp'], y=decomposition.seasonal, mode='lines', name='Seasonal'))
    decomposition_fig.add_trace(go.Scatter(
        x=patient_df['Timestamp'], y=decomposition.resid, mode='lines', name='Residual'))
    decomposition_fig.update_layout(title='Time Series Decomposition')

    # Parallel Coordinates Plot for Multivariate Analysis
    patient_df_scaled = (patient_df[['BloodSugar', 'BloodPressure', 'Cholesterol', 'HeartRate']] - patient_df[['BloodSugar', 'BloodPressure',
                         'Cholesterol', 'HeartRate']].mean()) / patient_df[['BloodSugar', 'BloodPressure', 'Cholesterol', 'HeartRate']].std()
    parallel_coordinates_fig = px.parallel_coordinates(patient_data, color='Cluster', labels={
                                                       'BloodSugar': 'Blood Sugar', 'BloodPressure': 'Blood Pressure', 'Cholesterol': 'Cholesterol', 'HeartRate': 'Heart Rate'}, title='Parallel Coordinates Plot')

    # Update the title of the size axis
    parallel_coordinates_fig.update_layout(
        title={'text': 'Parallel Coordinates Plot'},
        xaxis_title='Health Parameters',
        yaxis_title='Standardized Values',
        coloraxis_showscale=False)

    # Interactive Scatter Plot for Health Parameters
    scatter_health_fig = px.scatter(patient_df, x='BloodSugar', y='BloodPressure', color='Cholesterol', size='HeartRate', title='Health Parameters Scatter Plot', labels={
                                    'BloodSugar': 'Blood Sugar', 'BloodPressure': 'Blood Pressure', 'Cholesterol': 'Cholesterol', 'HeartRate': 'Heart Rate'})
    scatter_health_fig.update_layout(
        xaxis_title='Blood Sugar',
        yaxis_title='Blood Pressure',
        coloraxis_colorbar_title='Cholesterol',
        coloraxis_colorbar_x=-0.15,
        coloraxis_colorbar_xanchor='left',
    )

    # Generate 3D Scatter Plot
    scatter_3d_fig = go.Figure(data=[go.Scatter3d(
        x=patient_df['BloodSugar'],
        y=patient_df['BloodPressure'],
        z=patient_df['Cholesterol'],
        mode='markers',
        marker=dict(
            size=8,
            color=patient_df['HeartRate'],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=patient_df['Timestamp'],
        hoverinfo='text'
    )])
    scatter_3d_fig.update_layout(
        scene=dict(
            xaxis_title='Blood Sugar',
            yaxis_title='Blood Pressure',
            zaxis_title='Cholesterol'
        ),
        title='3D Scatter Plot: Blood Sugar vs. Blood Pressure vs. Cholesterol'
    )

    # Calculate summary statistics for health parameters
    health_parameters = ['BloodSugar',
                         'BloodPressure', 'Cholesterol', 'HeartRate']
    summary_stats = {}
    for parameter in health_parameters:
        mean = patient_df[parameter].mean()
        median = patient_df[parameter].median()
        std_dev = patient_df[parameter].std()
        summary_stats[parameter] = {'Mean': mean,
                                    'Median': median, 'Std Dev': std_dev}

    # Conduct t-test and get results
    t_stat, p_value = conduct_t_test(
        cluster_0_data, cluster_1_data, 'BloodSugar')

    # Create a DataFrame from the summary_stats dictionary
    summary_stats_df = pd.DataFrame.from_dict(summary_stats, orient='index')

    # Save the summary statistics as a CSV file
    summary_stats_df.to_csv('summary_statistics.csv')

    t_test_results = html.P(
        f"T-Test Results for Blood Sugar between Cluster 0 and Cluster 1: T-Statistic: {t_stat}, P-Value: {p_value}"
    )

    return (
        sugar_fig,
        bp_fig,
        cholesterol_fig,
        cluster_fig,
        heatmap_fig,
        violin_fig,
        decomposition_fig,
        parallel_coordinates_fig,
        scatter_health_fig,
        heart_rate_fig,
        scatter_3d_fig,
        t_test_results
    )


@app.callback(
    [Output('arima-forecast-sugar-chart', 'figure'),
     Output('arima-forecast-bp-chart', 'figure'),
     Output('arima-forecast-heart-rate-chart', 'figure')],
    [Input('patient-input', 'value')]
)
def update_arima_forecast_graph(patient_id):
    if patient_id is None:
        return no_update  # Return no_update if patient_id is None

    patient_df = patient_data[patient_data['PatientID'] == patient_id]
    columns_to_forecast = ['BloodSugar', 'BloodPressure', 'HeartRate']
    arima_forecast_figs = []

    for column in columns_to_forecast:
        data_to_forecast = patient_df[column]

        # Ensure that data_to_forecast is not empty
        if data_to_forecast.empty:
            continue

        # Fit ARIMA model using pmdarima
        model = pm.auto_arima(
            data_to_forecast, seasonal=False, suppress_warnings=True)

        # Forecast next 7 days
        forecast_steps = 7
        forecast, conf_int = model.predict(
            n_periods=forecast_steps, return_conf_int=True)

        # Create a date range for the forecasted period
        forecast_dates = pd.date_range(
            start=patient_df['Timestamp'].iloc[-1], periods=forecast_steps+1, freq='D')[1:]

        # Create figure with actual and forecasted data
        arima_forecast_fig = go.Figure()
        arima_forecast_fig.add_trace(go.Scatter(
            x=patient_df['Timestamp'], y=data_to_forecast, mode='lines', name='Actual'))
        arima_forecast_fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast, mode='lines', name='Forecast'))
        arima_forecast_fig.update_layout(title=f'ARIMA {column} Forecast')

        arima_forecast_figs.append(arima_forecast_fig)

    return tuple(arima_forecast_figs)


def conduct_t_test(cluster_data1, cluster_data2, parameter):
    t_stat, p_value = stats.ttest_ind(
        cluster_data1[parameter], cluster_data2[parameter])
    return t_stat, p_value


# Example: Compare Blood Sugar between Cluster 0 and Cluster 1
cluster_0_data = patient_data[patient_data['Cluster'] == 0]
cluster_1_data = patient_data[patient_data['Cluster'] == 1]

t_stat, p_value = conduct_t_test(cluster_0_data, cluster_1_data, 'BloodSugar')


def create_condition_network(patient_conditions_data):
    G = nx.Graph()

    for _, row in patient_conditions_data.iterrows():
        patient_id = row['PatientID']
        conditions = row['Conditions']

        for condition in conditions:
            G.add_node(condition)  # Add condition as a node
            # Connect patient ID to condition
            G.add_edge(patient_id, condition)

    return G


@app.callback(
    Output('condition-network', 'figure'),
    [Input('patient-input', 'value')]
)
def update_condition_network(patient_id):
    if patient_id is None:
        return no_update

    patient_conditions = conditions_data[conditions_data['PatientID']
                                         == patient_id]['Conditions'].iloc[0]

    condition_network = create_condition_network(conditions_data)

    # Layout for network visualization
    pos = nx.spring_layout(condition_network)
    node_trace = go.Scatter(x=[], y=[], mode='markers+text',
                            text=[], textposition='bottom center')
    edge_trace = go.Scatter(x=[], y=[], mode='lines')

    for node in condition_network.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])

    for edge in condition_network.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title=f'Patients Condition')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
