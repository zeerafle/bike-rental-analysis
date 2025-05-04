import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


MIN_SAMPLE_SIZE = 10

hour_df = pd.read_csv('data/hour_cleaned.csv')
seasons = ['Gugur', 'Semi', 'Panas', 'Dingin']

def create_relative_optimal(df):
    # hitung rata-rata penyewaan berdasarkan jam
    hourly_baseline = df.groupby('hr', observed=True)['cnt'].mean().reset_index()
    hourly_baseline.rename(columns={'cnt': 'baseline_cnt'}, inplace=True)
    df_with_baseline = df.merge(hourly_baseline, on='hr')

    # hitung kinerja relatif dibandingkan dengan baseline
    # kinerja relatif = penyewaan saat ini / penyewaan rata-rata jam
    df_with_baseline['relative_performance'] = df_with_baseline['cnt'] / df_with_baseline['baseline_cnt']

    # kondisi yang baik terlepas dari waktu
    relative_optimal = df_with_baseline.groupby(
        ['season', 'temp_bin', 'humidity_bin', 'weathersit'],
        observed=True
    ).agg({'relative_performance': ['mean', 'count']}).reset_index()

    # filter data yang memiliki lebih dari minimum sampel
    relative_optimal = relative_optimal[relative_optimal[('relative_performance', 'count')] >= MIN_SAMPLE_SIZE]

    return relative_optimal


def create_hourly_usage(df):
    # buat kolom baru untuk menandakan akhir pekan atau tidak
    df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])

    # kelompokkan menjadi berdasarkan jam dan akhir pekan
    hourly_usage = df.groupby(['hr', 'is_weekend'], observed=True).agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()

    # hitung rasio pengguna kasual terhadap pengguna terdaftar
    hourly_usage['casual_to_registered_ratio'] = hourly_usage['casual'] / hourly_usage['registered']
    weekday_data = hourly_usage[~hourly_usage['is_weekend']]
    weekend_data = hourly_usage[hourly_usage['is_weekend']]

    return weekday_data, weekend_data


def create_monthly_trends(df):
    # tren bulanan terhadap tipe pengguna
    monthly_trends = df.groupby(['year', 'month']).agg({
        'casual': 'sum',
        'registered': 'sum',
        'cnt': 'sum'
    }).reset_index()

    # buat time series untuk plot
    monthly_trends['date'] = pd.to_datetime(monthly_trends[['year', 'month']].assign(day=1))
    return monthly_trends


def create_seasonal_ratio(df):
    seasonal_ratio = df.groupby('season', observed=True).agg({
        'casual': 'sum',
        'registered': 'sum',
        'cnt': 'sum'
    }).reset_index()

    seasonal_ratio['casual_percent'] = seasonal_ratio['casual'] / seasonal_ratio['cnt'] * 100
    seasonal_ratio['registered_percent'] = seasonal_ratio['registered'] / seasonal_ratio['cnt'] * 100
    return seasonal_ratio


min_date = hour_df['dteday'].min()
max_date = hour_df['dteday'].max()

st.header('Dashboard Penyewaan Sepeda :bike:')
with st.sidebar:
    st.title("Dashboard")

    # mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date, value=(min_date, max_date)
    )

main_df = hour_df[(hour_df['dteday'] >= str(start_date)) &
                  (hour_df['dteday'] <= str(end_date))]


st.subheader('Kinerja Penyewaan Sepeda Terhadap Temperatur Sepanjang Musim')
relative_optimal = create_relative_optimal(main_df)

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='season', y=('relative_performance', 'mean'),
            hue='temp_bin',
            data=relative_optimal.reset_index(),
            ax=ax)
plt.xticks(ticks=range(len(seasons)), labels=seasons, rotation=0)
plt.xlabel('Musim')
plt.ylabel('Rata-rata Penyewaan')
plt.legend(title='Rentang Temperatur', loc='upper right')
st.pyplot(fig)


st.subheader('Rerate Penyewaan Sepeda Pada Hari Kerja dan Akhir Pekan')
weekday_data, weekend_data = create_hourly_usage(main_df)

tab1, tab2 = st.tabs(["Hari Kerja", "Akhir Pekan"])

with tab1:
    fig_weekday = plt.figure(figsize=(10, 6))
    plt.plot(weekday_data['hr'], weekday_data['registered'], 'b-', label='Terdaftar')
    plt.plot(weekday_data['hr'], weekday_data['casual'], 'r-', label='Kasual')
    plt.title('Rerata Penyewaan Per Jam Pada Hari Kerja')
    plt.xlabel('Jam dalam hari')
    plt.ylabel('Rerata penyewaan')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_weekday)

with tab2:
    fig_weekend = plt.figure(figsize=(10, 6))
    plt.plot(weekend_data['hr'], weekend_data['registered'], 'b-', label='Terdaftar')
    plt.plot(weekend_data['hr'], weekend_data['casual'], 'r-', label='Kasual')
    plt.title('Rerata Penyewaan Per Jam Pada Akhir Pekan')
    plt.xlabel('Jam dalam hari')
    plt.ylabel('Rerata penyewaan')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_weekend)


st.subheader('Tren Bulanan Penyewaan Sepeda')
monthly_trends = create_monthly_trends(main_df)

fig_monthly = plt.figure(figsize=(12, 6))
plt.plot(monthly_trends['date'], monthly_trends['casual'], 'r-', label='Kasual')
plt.plot(monthly_trends['date'], monthly_trends['registered'], 'b-', label='Terdaftar')
plt.title('Penyewaan Sepeda Bulanan Berdasarkan Tipe Pengguna')
plt.xlabel('Bulan')
plt.ylabel('Total Penyewaan')
plt.legend()
plt.grid(True)
st.pyplot(fig_monthly)


st.subheader('Pengaruh Kondisi Cuaca Terhadap Penyewaan')
col1, col2 = st.columns(2)

with col1:
    # pengaruh temperatur
    temp_impact = main_df.groupby('temp_bin', observed=True).agg({
        'cnt': 'mean'
    }).reset_index()

    fig_temp = plt.figure(figsize=(8, 5))
    sns.barplot(x='temp_bin', y='cnt', data=temp_impact)
    plt.title('Rata-rata Penyewaan Berdasarkan Temperatur')
    plt.xlabel('Rentang Temperatur')
    plt.ylabel('Rata-rata Penyewaan')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_temp)

with col2:
    # pengaruh situasi cuaca
    weather_impact = main_df.groupby('weathersit', observed=True).agg({
        'cnt': 'mean'
    }).reset_index()

    fig_weather = plt.figure(figsize=(8, 5))
    sns.barplot(x='weathersit', y='cnt', data=weather_impact)
    plt.title('Rata-rata Penyewaan Berdasarkan Situasi Cuaca')
    plt.xlabel('Situasi Cuaca')
    plt.ylabel('Rata-rata Penyewaan')
    plt.tight_layout()
    st.pyplot(fig_weather)


st.subheader('Rasio Tipe Pengguna Berdasarkan Musim')
seasonal_ratio =  create_seasonal_ratio(main_df)

fig_ratio = plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(seasonal_ratio['season']))

plt.bar(index, seasonal_ratio['registered_percent'], bar_width, label='Terdaftar')
plt.bar(index, seasonal_ratio['casual_percent'], bar_width, bottom=seasonal_ratio['registered_percent'], label='Kasual')

plt.xlabel('Musim')
plt.ylabel('Persentase (%)')
plt.title('Komposisi Tipe Pengguna per Musim')
plt.xticks(ticks=range(len(seasons)), labels=seasons, rotation=0)
plt.legend()
plt.tight_layout()
st.pyplot(fig_ratio)
