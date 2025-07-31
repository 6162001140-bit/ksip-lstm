
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Forecast LSTM KSIP AGRO", layout="wide")
st.title("📈 Forecast LSTM untuk Komoditas KSIP AGRO")

uploaded_file = st.file_uploader("Upload file Excel (pastikan ada kolom TANGGAL dan HARGA)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "TANGGAL" not in df.columns or "HARGA" not in df.columns:
        st.error("File harus memiliki kolom 'TANGGAL' dan 'HARGA'")
        st.stop()

    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df = df.sort_values('TANGGAL')
    data = df[['TANGGAL', 'HARGA']].set_index('TANGGAL')
    data_weekly = data['HARGA'].resample('W').mean().dropna().to_frame()
    
    st.subheader("📄 Data Harga Aktual Kamu")
    data_formatted = data.copy()
    data_formatted["HARGA"] = data_formatted["HARGA"].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
    
    st.dataframe(data_formatted)
    st.line_chart(data)


    # ========================== MODELING ==========================
    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_weekly)

    def create_dataset(dataset, look_back=4):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 4
    X, y = create_dataset(scaled_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build & train model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=50, batch_size=1, verbose=0)

    
    # ======================== FORECASTING ========================
    # Forecast 4 minggu
    last_weeks_data = scaled_data[-look_back:].reshape(-1, 1)
    forecast, input_seq = [], last_weeks_data
    for _ in range(4):
        pred_scaled = model.predict(input_seq.reshape(1, look_back, 1), verbose=0)
        forecast.append(pred_scaled[0][0])
        input_seq = np.append(input_seq[1:], [[pred_scaled[0][0]]], axis=0)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    forecast_dates = [data_weekly.index[-1] + pd.Timedelta(weeks=i) for i in range(1, 5)]
    forecast_df = pd.DataFrame({"Tanggal": forecast_dates, "Harga Prediksi (Mingguan)": forecast})
    forecast_df_display = forecast_df.copy()
    forecast_df_display["Harga Prediksi (Mingguan)"] = forecast_df_display["Harga Prediksi (Mingguan)"].apply(lambda x: f"Rp {x:,.0f}")
    


    st.subheader("📊 Forecast Harga (Mingguan)")
    st.dataframe(forecast_df_display.set_index("Tanggal"))
    st.line_chart(pd.concat([data_weekly, forecast_df.set_index("Tanggal")], axis=0))

    # ==================== Interpolasi Harian =======================
    data_daily = df[['TANGGAL', 'HARGA']].set_index('TANGGAL')
    last_date = data_daily.index[-1]
    daily_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=31)
    interp_series = pd.Series([data_daily['HARGA'].iloc[-1]] + list(forecast), index=[last_date] + forecast_dates)
    daily_forecast = interp_series.reindex(interp_series.index.union(daily_range)).interpolate('time').loc[daily_range]

    st.subheader("📅 Forecast Harga Harian (Interpolasi Linear)")
    daily_df = pd.DataFrame({"Tanggal": daily_range, "Forecast Harian": daily_forecast.values})
    st.dataframe(daily_df.set_index("Tanggal"))

    # Plot gabungan
    st.subheader("📉 Visualisasi Harga")
    combined_df2 = pd.concat([
    pd.DataFrame({"Harga Aktual": data_daily["HARGA"]}),
    pd.DataFrame({"Forecast Harian": daily_forecast})], axis=1)

    st.line_chart(combined_df2)


