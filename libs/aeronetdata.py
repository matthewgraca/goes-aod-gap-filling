import io
import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class AERONETData:
    """
    Fetches and processes AERONET sun-photometer observations.

    Pipeline:
        1. Fetch all-points (AVG=10) observations from the AERONET Web
           Service v3 for a date range and bounding box
        2. Parse the CSV response and discard missing-value sentinels (-999)
        3. Interpolate AOD to 550 nm using the Ångström Exponent derived
           from the 440 nm and 675 nm channels (which bracket 550 nm)
        4. Compute per-site hourly means; label each bin with the *end* of
           the hour (00:00–00:59 → 01:00), matching the GOESData convention
        5. Expose result as self.data — a DataFrame with columns:
               lat, lon, aod_550, timestamp, sensor_name
    """

    BASE_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"

    # Reference wavelengths used to derive the Ångström exponent.
    # 440 nm and 675 nm bracket 550 nm and are reliably present in the data.
    _LAMBDA_LOW  = 440   # nm
    _LAMBDA_HIGH = 675   # nm
    _LAMBDA_TARGET = 550  # nm

    def __init__(
        self,
        start_date="2023-08-02",
        end_date="2023-08-03",
        extent=(-118.615, -117.70, 33.60, 34.35),
        quality_level=15,
        cache_path=None,
        load_cache=False,
        save_cache=True,
        verbose=False,
    ):
        """
        Args:
            start_date:     "YYYY-MM-DD" start of the fetch window (inclusive)
            end_date:       "YYYY-MM-DD" end of the fetch window (inclusive)
            extent:         (lon_min, lon_max, lat_min, lat_max) bounding box
                            in decimal degrees; used for server-side filtering
            quality_level:  10 = Level 1.0 (raw), 15 = Level 1.5,
                            20 = Level 2.0 (cloud-screened, final quality)
            cache_path:     Path to a Parquet file for caching self.data.
                            Required when save_cache or load_cache is True.
            load_cache:     If True, load self.data from cache_path and skip
                            all network requests.
            save_cache:     If True, write self.data to cache_path after
                            processing.
            verbose:        Print progress information.
        """
        if save_cache or load_cache:
            if cache_path is None:
                raise ValueError("Please provide a cache_path.")
            parent = os.path.dirname(cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._validate_cache_path(load_cache, cache_path)

        if load_cache:
            self.data = self._load_cache(cache_path)
            return

        self._validate_quality_level(quality_level)

        raw = self._fetch_observations(start_date, end_date, extent, quality_level, verbose)
        raw = self._interpolate_to_550nm(raw)
        self.data = self._compute_hourly_means(raw)

        if cache_path is not None and save_cache:
            self._save_cache(cache_path, self.data)

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    def _load_cache(self, cache_path):
        print(f"Loading AERONET data from {cache_path}...", end=" ")
        df = pd.read_parquet(cache_path)
        print("Complete!")
        return df

    def _save_cache(self, cache_path, df):
        print(f"Saving AERONET data to {cache_path}...", end=" ")
        df.to_parquet(cache_path, index=False)
        print("Complete!")

    def _validate_cache_path(self, load_cache, cache_path):
        if load_cache and not os.path.exists(cache_path):
            raise ValueError(
                f"Cache file does not exist: {cache_path}. "
                "Set load_cache=False to fetch fresh data from AERONET."
            )

    def _validate_quality_level(self, quality_level):
        valid = {10, 15, 20}
        if quality_level not in valid:
            raise ValueError(
                f"quality_level must be one of {valid}, got {quality_level}. "
                "Use 10 for Level 1.0, 15 for Level 1.5, or 20 for Level 2.0."
            )

    # -------------------------------------------------------------------------
    # Fetch and parse
    # -------------------------------------------------------------------------

    def _build_params(self, start, end, extent, quality_level):
        """Return query-parameter dict for the AERONET v3 endpoint."""
        lon_min, lon_max, lat_min, lat_max = extent
        return {
            f"AOD{quality_level}": 1,
            "AVG": 10,          # all points (not daily average)
            "if_no_html": 1,    # plain CSV; no HTML wrapper
            "year":   start.year,
            "month":  start.month,
            "day":    start.day,
            "hour":   0,
            "year2":  end.year,
            "month2": end.month,
            "day2":   end.day,
            "hour2":  23,
            "lat1": lat_min,
            "lon1": lon_min,
            "lat2": lat_max,
            "lon2": lon_max,
        }

    def _parse_csv(self, text):
        """
        Parse an AERONET v3 CSV response.

        The response body contains several preamble lines before the actual
        header row, which always begins with "AERONET_Site".  Everything
        from that row onward is valid CSV.  Missing values are coded as
        -999 or -999.000000.
        """
        lines = text.splitlines()

        header_idx = next(
            (i for i, ln in enumerate(lines) if ln.startswith("AERONET_Site")),
            None,
        )
        if header_idx is None:
            return pd.DataFrame()

        csv_body = "\n".join(lines[header_idx:])
        df = pd.read_csv(
            io.StringIO(csv_body),
            na_values=["-999", "-999.000000"],
            low_memory=False,
        )
        return df

    def _fetch_observations(self, start_date, end_date, extent, quality_level, verbose):
        """
        Issue a single request to the AERONET API and return a parsed,
        lightly cleaned DataFrame of raw per-observation records.
        """
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
        params = self._build_params(start, end, extent, quality_level)

        if verbose:
            print(
                f"Fetching AERONET L{quality_level / 10:.1f} data "
                f"{start.date()} → {end.date()} ..."
            )

        response = requests.get(self.BASE_URL, params=params, timeout=120)
        response.raise_for_status()

        df = self._parse_csv(response.text)
        if df.empty:
            if verbose:
                print("No AERONET observations found for the given parameters.")
            return df

        # parse the combined date+time into a single Timestamp column
        timestamp_raw = pd.to_datetime(
            df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
            format="%d:%m:%Y %H:%M:%S",
        )

        df = df.rename(columns={
            "AERONET_Site":           "sensor_name",
            "Site_Latitude(Degrees)": "lat",
            "Site_Longitude(Degrees)": "lon",
        }).copy()

        df["timestamp_raw"] = timestamp_raw.values

        if verbose:
            print(
                f"  {len(df):,} observations from "
                f"{df['sensor_name'].nunique()} site(s)."
            )

        return df

    # -------------------------------------------------------------------------
    # Spectral interpolation to 550 nm
    # -------------------------------------------------------------------------

    def _interpolate_to_550nm(self, df):
        """
        Derive the Ångström Exponent (α) from the 440 nm and 675 nm channels,
        then power-law interpolate to 550 nm.

            α   = −log(AOD_440 / AOD_675) / log(440 / 675)
            AOD_550 = AOD_440 × (550 / 440)^(−α)

        Rows where either reference channel is missing or non-positive are
        set to NaN and excluded from the hourly mean downstream.
        """
        if df.empty:
            return df

        col_low  = f"AOD_{self._LAMBDA_LOW}nm"
        col_high = f"AOD_{self._LAMBDA_HIGH}nm"

        for col in (col_low, col_high):
            if col not in df.columns:
                raise KeyError(
                    f"Expected column '{col}' not found in AERONET response. "
                    f"Available AOD columns: "
                    f"{[c for c in df.columns if c.startswith('AOD_')]}"
                )

        aod_low  = df[col_low].values.astype(float)
        aod_high = df[col_high].values.astype(float)

        valid = (aod_low > 0) & (aod_high > 0)

        alpha = np.full(len(df), np.nan)
        alpha[valid] = (
            -np.log(aod_low[valid] / aod_high[valid])
            / np.log(self._LAMBDA_LOW / self._LAMBDA_HIGH)
        )

        aod_550 = np.full(len(df), np.nan)
        aod_550[valid] = aod_low[valid] * (self._LAMBDA_TARGET / self._LAMBDA_LOW) ** (-alpha[valid])

        df = df.copy()
        df["aod_550"] = aod_550
        return df

    # -------------------------------------------------------------------------
    # Hourly aggregation
    # -------------------------------------------------------------------------

    def _compute_hourly_means(self, df):
        """
        Aggregate per-observation AOD into per-site hourly means.

        Timestamp convention (matching GOESData._realigned_date_range):
            observations in [HH:00, HH:59] → label = (HH+1):00
            e.g. 00:00–00:59 observations → timestamp 01:00

        Returns a DataFrame with columns:
            lat, lon, aod_550, timestamp, sensor_name
        """
        if df.empty:
            return pd.DataFrame(
                columns=["lat", "lon", "aod_550", "timestamp", "sensor_name"]
            )

        df = df.copy()
        # floor to the hour then shift forward by one hour
        df["timestamp"] = df["timestamp_raw"].dt.floor("h") + pd.Timedelta(hours=1)

        result = (
            df.dropna(subset=["aod_550"])
            .groupby(["sensor_name", "timestamp"], as_index=False)
            .agg(
                lat=("lat", "first"),
                lon=("lon", "first"),
                aod_550=("aod_550", "mean"),
            )
        )

        return result[["lat", "lon", "aod_550", "timestamp", "sensor_name"]]


if __name__ == "__main__":
    extent = (-118.615, -117.70, 33.60, 34.35)  # California

    aeronet = AERONETData(
        start_date="2023-08-02",
        end_date="2023-08-03",
        extent=extent,
        quality_level=15,
        save_cache=False,
        verbose=True,
    )

    print(aeronet.data)
    print(f"\nShape: {aeronet.data.shape}")
    print(f"Sites: {aeronet.data['sensor_name'].unique()}")
    print(f"Time range: {aeronet.data['timestamp'].min()} → {aeronet.data['timestamp'].max()}")
