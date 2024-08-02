import streamlit as st
from streamlit import set_page_config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
import os
import argparse
import sys

# Add streamlit terminal parameters to specify the path of the data file
# streamlit run EDA.py -- --data_path data.csv

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def additional_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns='Unnamed: 0')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['create_at_date'] = df['created_at'].dt.date
    df['create_at_hour'] = df['created_at'].dt.hour
    df['month'] = df['created_at'].dt.month
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['fx_rate_difference'] = df['fx_rate'] - df['mid_market_fx_rate']
    df['fx_rate_markup'] = (df['fx_rate'] - df['mid_market_fx_rate']) / df['mid_market_fx_rate'] * 100
    df['user_first_visit'] = pd.to_datetime(df['user_first_visit'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['days_since_first_visit'] = (df['created_at'] - df['user_first_visit']).dt.days
    df['weeks_since_first_visit'] = (df['created_at'] - df['user_first_visit']).dt.days // 7

    return df

def display_basic_stats(df: pd.DataFrame) -> None:
    st.write('Dataframe info:', df.info())
    st.write("Missing values:", df.isnull().sum())
    st.write("Basic statistics:", df.describe())

def analyze_order_creation(df: pd.DataFrame) -> None:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
    df['create_at_date'].value_counts().sort_index().plot(kind='line', ax=ax1)
    ax1.set_title('Orders over time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of orders')

    df['create_at_hour'].value_counts().sort_index().plot(kind='bar', ax=ax2)
    ax2.set_title('Orders by hour of day')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Number of orders')

    st.pyplot(fig)

def analyze_payment_amount(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.histplot(df['payment_amount'], kde=True, ax=ax)
    ax.set_title('Distribution of Payment Amounts')
    ax.set_xlabel('Payment Amount')
    st.pyplot(fig)

    st.write("Payment amount statistics:")
    st.write(df['payment_amount'].describe())

    st.write("Payment amount percentiles:")
    st.write(df['payment_amount'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    fig, ax = plt.subplots(figsize=(25, 15))
    sns.boxplot(x='from_currency', y='payment_amount', data=df.nlargest(1000, 'payment_amount'), ax=ax)
    ax.set_title('Payment Amount Distribution by From Currency (Top 1000)')
    plt.xticks(rotation=90)
    st.pyplot(fig)

def analyze_user_id(df: pd.DataFrame) -> None:
    user_order_counts = df['user_id'].value_counts()
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.histplot(user_order_counts, kde=True, ax=ax)
    ax.set_title('Distribution of Orders per User')
    ax.set_xlabel('Number of Orders')
    ax.set_ylabel('Number of Users')
    st.pyplot(fig)

    st.write("Top 10 users by number of orders:")
    st.write(user_order_counts.head(10))

def analyze_fx_rates(df: pd.DataFrame) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(25, 15))
    sns.histplot(df['fx_rate'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of FX Rates')
    sns.histplot(df['mid_market_fx_rate'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Mid-Market FX Rates')
    sns.histplot(df['fx_rate_difference'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of FX Rate Differences')
    sns.histplot(df['fx_rate_markup'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of FX Rate Markup (%)')
    st.pyplot(fig)

    st.write("FX Rate Markup Statistics:")
    st.write(df['fx_rate_markup'].describe())

def analyze_delivery_option(df: pd.DataFrame) -> None:
    delivery_counts = df['delivery_option'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
    delivery_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Delivery Option Counts')
    ax1.set_xlabel('Delivery Option')
    ax1.set_ylabel('Count')
    sns.histplot(df['delivery_option_fee'], kde=True, ax=ax2)
    ax2.set_title('Distribution of Delivery Option Fees')
    ax2.set_xlabel('Fee')
    st.pyplot(fig)

def analyze_countries_currencies(df: pd.DataFrame) -> None:
    st.write("Top 10 from_countries:")
    st.write(df['from_country'].value_counts().head(10))
    st.write("Top 10 from_currencies:")
    st.write(df['from_currency'].value_counts().head(10))
    st.write("Top 10 to_currencies:")
    st.write(df['to_currency'].value_counts().head(10))

    fig, ax = plt.subplots(figsize=(25, 15))
    df.groupby(['from_currency', 'to_currency']).size().nlargest(20).plot(kind='bar', ax=ax)
    ax.set_title('Top 20 Currency Pairs')
    ax.set_xlabel('Currency Pair')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(fig)

def analyze_user_demographics(df: pd.DataFrame) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 15))
    df['user_nationality'].value_counts().head(10).plot(kind='bar', ax=ax1)
    ax1.set_title('Top 10 User Nationalities')
    ax1.set_xlabel('Nationality')
    ax1.set_ylabel('Count')

    df['user_language'].value_counts().head(10).plot(kind='bar', ax=ax2)
    ax2.set_title('Top 10 User Languages')
    ax2.set_xlabel('Language')
    ax2.set_ylabel('Count')

    sns.histplot(df['user_birthyear'], kde=True, bins=50, ax=ax3)
    ax3.set_title('Distribution of User Birth Years')
    ax3.set_xlabel('Birth Year')
    st.pyplot(fig)

def analyze_user_first_visit(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(25, 15))

    sns.histplot(df['days_since_first_visit'], kde=True, bins=50, ax=ax)
    ax.set_title('Distribution of Days Since First Visit')
    ax.set_xlabel('Days')
    st.pyplot(fig)

    st.write("Days since first visit statistics:")
    st.write(df['days_since_first_visit'].describe())

def analyze_user_behavior(df: pd.DataFrame) -> None:
    df['order_count'] = df.groupby('user_id')['user_id'].transform('count')
    df['total_spent'] = df.groupby('user_id')['payment_amount'].transform('sum')

    fig, ax = plt.subplots(figsize=(25, 15))

    ax.scatter(df['order_count'], df['total_spent'])
    ax.set_title('User Order Count vs Total Spent')
    ax.set_xlabel('Order Count')
    ax.set_ylabel('Total Spent')
    st.pyplot(fig)

    st.write("Correlation between order count and total spent:")
    st.write(df[['order_count', 'total_spent']].corr())


def analyze_seasonality(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(25, 15))
    df.groupby('month')['payment_amount'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average Payment Amount by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Payment Amount')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(25, 15))
    df.groupby('day_of_week')['payment_amount'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average Payment Amount by Day of Week')
    ax.set_xlabel('Day of Week (0 = Monday, 6 = Sunday)')
    ax.set_ylabel('Average Payment Amount')
    st.pyplot(fig)

def analyze_user_retention(df: pd.DataFrame) -> None:
    retention_data = df.groupby('weeks_since_first_visit')['user_id'].nunique()

    fig, ax = plt.subplots(figsize=(25, 15))
    retention_data.plot(kind='line', ax=ax)
    ax.set_title('User Retention Over Time')
    ax.set_xlabel('Weeks Since First Visit')
    ax.set_ylabel('Number of Active Users')
    st.pyplot(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="EDA Dashboard")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument('--preprocess', action='store_true', help="Preprocess the data")
    parser.add_argument("--output_path", type=str, default="updated_data.csv", help="Path to save the output file")
    return parser.parse_args()

def main():

    args = parse_args()
    set_page_config(layout="wide", page_title="EDA Dashboard")
    df = load_data(args.data_path)
    if args.preprocess:
        df = additional_cols(df)
        df.to_csv(args.output_path, index=False)

    analysis_functions = {
        'Basic Stats': display_basic_stats,
        'Order Creation': analyze_order_creation,
        'Payment Amount': analyze_payment_amount,
        'Users': analyze_user_id,
        'FX Rates': analyze_fx_rates,
        'Delivery Option': analyze_delivery_option,
        'Countries & Currencies': analyze_countries_currencies,
        'User Demographics': analyze_user_demographics,
        'User First Visit': analyze_user_first_visit,
        'User Behavior': analyze_user_behavior,
        'Seasonality': analyze_seasonality,
        'User Retention': analyze_user_retention
    }

    st.sidebar.title("EDA Options")
    analysis = st.sidebar.radio('Choose analysis:', list(analysis_functions.keys()))

    st.header(f'{analysis} Analysis')
    analysis_functions[analysis](df)




if __name__ == '__main__':

    main()