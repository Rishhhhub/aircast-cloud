# AirCast dashboard (cloud trial copy)

A trimmed copy of the AirCast dashboard for trying Streamlit Community Cloud. It shows the live station forecasts for the next 24, 48 and 72 hours, the estimated AQI between stations (live mode), the forecast history for the 2023 test year, and a map. Data is a snapshot up to 14 November 2026; the refresh button of the full project is not part of this copy.

What was left out to fit the free memory limit (about 1 GB): the 2017-2022 rows of the feature table and the historical mode of the between-stations tab.

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub. Allow access to this repository (it is private).
2. New app: repository `aircast-cloud`, branch `main`, main file `app/app.py`.
3. Advanced settings: choose **Python 3.12** (or 3.13). Deploy. The first start takes a few minutes while it installs `requirements.txt`.

If the app stops with a memory error, the free tier is too small for this copy.

## Run locally
```
pip install -r requirements.txt
streamlit run app/app.py
```
