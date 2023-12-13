# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:48:30 2023

@author: 91905
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        df (pandas.DataFrame): DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df


def describe_data(df):
    """
    Display summary statistics of the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to describe.

    Returns:
        None (prints the summary statistics).
    """
    print(df.describe())


def display_dataframe(df, head_rows=5):
    """
    Display the first few rows of the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to display.
        head_rows (int): Number of rows to display. Default is 5.

    Returns:
        None (prints the DataFrame).
    """
    print(df.head(head_rows))


# Load the data into a DataFrame
df = load_data(
    'D:\\API_19_DS2_en_csv_v2_6183479\\API_19_DS2_en_csv_v2_6183479.csv')

# Display summary statistics of the DataFrame
describe_data(df)

# Display the first few rows of the DataFrame with years as columns
df_years = df.set_index(
    ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']).T
df_years = df_years.iloc[4:]
display_dataframe(df_years)

# Display the first few rows of the DataFrame with countries as columns
df_countries = df.set_index(['Indicator Name', 'Indicator Code']).T
df_countries = df_countries.iloc[:-1]
display_dataframe(df_countries)

# Define the country and indicators
countries = ['United States', 'United Kingdom']
indicators = [
    'Electricity production from oil sources (% of total)',
    'Electricity production from nuclear sources (% of total)',
    'Electricity production from natural gas sources (% of total)',
    'Electricity production from hydroelectric sources (% of total)',
    'Electricity production from coal sources (% of total)',
    'Urban population (% of total population)',
    'Urban population',
    'Urban population growth (annual %)',
    'Population, total',
    'Population growth (annual %)'
]

# Define alias names for the indicators
alias_names = {
    'Electricity production from oil sources (% of total)': 'Oil',
    'Electricity production from nuclear sources (% of total)': 'Nuclear',
    'Electricity production from natural gas sources (% of total)': 'Natural Gas',
    'Electricity production from hydroelectric sources (% of total)': 'Hydroelectric',
    'Electricity production from coal sources (% of total)': 'Coal',
    'Urban population (% of total population)': 'Urban Pop. (%)',
    'Urban population': 'Urban Pop.',
    'Urban population growth (annual %)': 'Urban Pop. Growth (%)',
    'Population, total': 'Total Population',
    'Population growth (annual %)': 'Pop. Growth (%)'}

# Create subplots
fig, axes = plt.subplots(nrows=len(countries), figsize=(8, 10), sharex=True)

for ax, country in zip(axes, countries):
    # Filter the DataFrame for the selected country and indicators
    df_selected = df[(df['Country Name'] == country) &
                     (df['Indicator Name'].isin(indicators))]

    # Select columns based on the specified year range
    selected_years = [str(year) for year in range(1990, 2021)]
    selected_years = [
        year for year in selected_years if year in df_selected.columns]

    # Transpose the DataFrame for better visualization
    df_selected_T = df_selected.set_index('Indicator Name')[selected_years].T

    # Calculate the correlation matrix
    corr_matrix = df_selected_T.corr()

    # Create the correlation heatmap with decreased font size for values
    cmap = 'viridis' if country == 'United Kingdom' else 'coolwarm'
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='YlGnBu',
        fmt=".3f",
        cbar=True,
        ax=ax,
        annot_kws={
            'size': 6})
    ax.set_title(
        f'Correlation Heatmap for {country} (1990-2020)',
        fontsize=8)  # Increase title font size

    # Rename x-axis and y-axis tick labels using alias names
    ax.set_xticklabels([alias_names.get(label.get_text(), label.get_text())
                       for label in ax.get_xticklabels()], fontsize=8)
    ax.set_yticklabels([alias_names.get(label.get_text(), label.get_text())
                       for label in ax.get_yticklabels()], fontsize=8)
    ax.tick_params(axis='both', labelsize=8)  # Adjust tick label font size

# Adjust layout
plt.tight_layout()
plt.show()


# Define the country and indicators
countries = ['United States', 'United Kingdom']
indicators = [
    'Electricity production from oil sources (% of total)',
    'Electricity production from nuclear sources (% of total)',
    'Electricity production from natural gas sources (% of total)',
    'Electricity production from hydroelectric sources (% of total)',
    'Electricity production from coal sources (% of total)'
]

# Create subplots with multiple columns
fig, axes = plt.subplots(
    nrows=len(countries), ncols=2, figsize=(
        15, 10), sharex=True)

# Define a color palette for lines
line_colors = sns.color_palette('husl', n_colors=len(indicators))

for ax_row, country in zip(axes, countries):
    for ax, indicator, color in zip(ax_row, indicators, line_colors):
        # Filter the DataFrame for the selected country and indicator
        df_selected = df[(df['Country Name'] == country) &
                         (df['Indicator Name'] == indicator)]

        # Select columns based on the specified year range
        selected_years = [str(year) for year in range(1990, 2021)]
        selected_years = [
            year for year in selected_years if year in df_selected.columns]

        # Transpose the DataFrame for better visualization
        df_selected_T = df_selected[selected_years].T

        # Plot the data with specified color and add grid
        df_selected_T.plot(ax=ax, legend=False, marker='o', color=color)
        ax.set_title(f'{indicator} in {country} (1990-2020)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage')
        ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# Define the countries and indicators
countries = [
    'United States',
    'United Kingdom',
    'China',
    'India',
    'Saudi Arabia',
    'Brazil',
    'Russia',
    'Germany',
    'Japan',
    'Australia']

indicators = [
    'Forest area (% of land area)',
    'Agricultural land (% of land area)'
]

# Filter the DataFrame for the selected indicators and countries
df_selected = df[(df['Country Name'].isin(countries)) &
                 (df['Indicator Name'].isin(indicators))]

# Pivot the DataFrame for better visualization
df_pivoted = df_selected.pivot(
    index='Country Name',
    columns='Indicator Name',
    values='2018')

# Set up colors for the bars
colors = sns.color_palette("pastel")

# Create subplots with multiple bar charts
fig, axes = plt.subplots(nrows=len(indicators), figsize=(12, 12))

for ax, indicator, color in zip(axes, indicators, colors):
    # Plot the data with specified colors
    bars = ax.bar(
        df_pivoted.index,  # x-axis values
        df_pivoted[indicator],  # y-axis values
        color=color
    )

    # Display values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() +
            bar.get_width() /
            2,
            yval +
            0.01,
            round(
                yval,
                2),
            ha='center',
            va='bottom',
            fontsize=8,
            color='black')

    # Set title and labels with reduced font size
    ax.set_title(f"{indicator}", fontsize=12, color='navy')
    ax.set_xlabel("Country Name", fontsize=12, color='navy')
    ax.set_ylabel(f"% of {indicator}", fontsize=12, color='navy')

    # Set x-axis tick positions and labels with 90-degree rotation and reduced
    # font size
    ax.set_xticks(range(len(df_pivoted.index)))
    ax.set_xticklabels(
        df_pivoted.index,
        rotation=45,
        fontsize=10,
        ha='right',
        color='darkslategray')

    # Set grid lines
    ax.grid(axis='y', alpha=0.4)
    ax.grid(axis='x', alpha=0.2, linestyle='--')

# Adjust layout
plt.tight_layout()
plt.show()
