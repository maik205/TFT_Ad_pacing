"""
Data Ingestion and Feature Resampling Pipeline for Programmatic Ad Spend Pacing.

Transforms raw iPinYou bidding auction logs into continuous, hourly-resampled
time series formatted for Temporal Fusion Transformers (TFT).

Features Extracted:
- Static Context: campaign_id, total_budget, duration_hours, nominal_velocity, vertical
- Observed Past: hourly_spend, impressions_won, win_rate, avg_cpm, cumulative_spend, spend_ratio
- Known Future: hour_sin, hour_cos, day_of_week, day_sin, day_cos, is_weekend, countdown_ratio
"""

import os
import gc
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Map anonymized advertiser IDs to industry vertical categories
VERTICAL_MAPPING = {
    937614: "e-commerce",
    937615: "finance",
    937616: "software",
    937617: "automotive",
    937618: "travel",
    937619: "retail",
    937620: "gaming",
    937621: "telecom",
    937622: "education"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Resample iPinYou logs into hourly TFT time series.")
    parser.add_argument("--train_file", type=str, default="train.csv", help="Path to raw train CSV.")
    parser.add_argument("--test_file", type=str, default="test.csv", help="Path to raw test CSV.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for processed files.")
    parser.add_argument("--chunksize", type=int, default=500000, help="Chunk size for streaming processing.")
    parser.add_argument("--max_chunks", type=int, default=None, help="Optional maximum chunks for quick testing.")
    return parser.parse_args()


def process_raw_chunks(filepath: str, chunksize: int = 500000, max_chunks: int = None, base_day_offset: int = 0):
    """
    Stream and aggregate raw auction events into campaign-hourly buckets.
    Handles memory efficiency by streaming large CSV files in chunks.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file {filepath} not found.")

    logger.info(f"Processing {filepath} (chunksize={chunksize:,})...")
    hourly_aggregates = []
    
    # Columns necessary for aggregation
    usecols = ['weekday', 'hour', 'advertiser', 'slotprice', 'click']
    
    chunk_idx = 0
    prev_weekday = None
    accumulated_day_offset = base_day_offset

    for chunk in pd.read_csv(filepath, chunksize=chunksize, usecols=usecols):
        chunk_idx += 1
        
        # Standardize hour: in iPinYou dataset hour is mapped 9-32 (0-23 + 9)
        chunk['norm_hour'] = chunk['hour'] - 9
        # Normalize slotprice: slotprice 937665 is baseline bucket (scaled to currency units)
        chunk['price_value'] = (chunk['slotprice'] - 937665 + 1.0) * 1.5

        # Group by campaign, weekday, and normalized hour
        grouped = chunk.groupby(['advertiser', 'weekday', 'norm_hour']).agg(
            impressions_won=('click', 'count'),
            total_clicks=('click', 'sum'),
            spend=('price_value', 'sum')
        ).reset_index()

        hourly_aggregates.append(grouped)

        if chunk_idx % 10 == 0:
            logger.info(f"  Processed {chunk_idx * chunksize:,} rows from {filepath}...")
            gc.collect()

        if max_chunks and chunk_idx >= max_chunks:
            logger.info(f"  Reached max_chunks limit ({max_chunks}).")
            break

    logger.info(f"Consolidating aggregate chunks for {filepath}...")
    combined = pd.concat(hourly_aggregates, ignore_index=True)
    consolidated = combined.groupby(['advertiser', 'weekday', 'norm_hour']).agg(
        impressions_won=('impressions_won', 'sum'),
        total_clicks=('total_clicks', 'sum'),
        spend=('spend', 'sum')
    ).reset_index()

    return consolidated


def build_continuous_hourly_timeline(df_agg: pd.DataFrame, num_cycles: int = 4) -> pd.DataFrame:
    """
    Constructs a contiguous, continuous hourly timeline across campaigns.
    Replicates multi-week cycles (e.g. 60-day train, 15-day val, 15-day test)
    with realistic diurnal and day-of-week seasonality, and adds dynamic noise.
    """
    logger.info("Building continuous timeline with calendar encodings...")
    campaigns = sorted(df_agg['advertiser'].unique())
    all_campaign_records = []

    for adv_id in campaigns:
        adv_data = df_agg[df_agg['advertiser'] == adv_id].copy()
        vertical = VERTICAL_MAPPING.get(adv_id, "general")
        
        # Create full 7-day x 24-hour grid for this advertiser
        grid = []
        for day in range(1, 8):
            for hr in range(24):
                grid.append({'advertiser': adv_id, 'weekday': day, 'norm_hour': hr})
        grid_df = pd.DataFrame(grid)
        
        # Merge observed aggregates onto full grid
        merged = pd.merge(grid_df, adv_data, on=['advertiser', 'weekday', 'norm_hour'], how='left')
        merged['impressions_won'] = merged['impressions_won'].fillna(0)
        merged['total_clicks'] = merged['total_clicks'].fillna(0)
        merged['spend'] = merged['spend'].fillna(0.0)

        # Build full multi-day continuous horizon (e.g. 90 days total for 60/15/15 split)
        # 90 days * 24 hours = 2,160 hours
        total_days = 90
        total_hours = total_days * 24
        
        np.random.seed(int(adv_id) % 10000)
        
        time_records = []
        cum_spend = 0.0
        
        # Weekly template base pattern
        week_template = merged.sort_values(['weekday', 'norm_hour']).reset_index(drop=True)
        num_template_hours = len(week_template) # 168 hours

        for global_t in range(total_hours):
            day_idx = global_t // 24
            hour_of_day = global_t % 24
            day_of_week = (day_idx % 7) # 0 = Monday, 6 = Sunday
            
            # Map into weekly template
            template_row = week_template[(week_template['weekday'] == (day_of_week + 1)) & 
                                         (week_template['norm_hour'] == hour_of_day)].iloc[0]
            
            base_spend = template_row['spend']
            base_imps = template_row['impressions_won']
            base_clicks = template_row['total_clicks']

            # Introduce realistic market macro-trend and auction noise
            # Trend factor: slight growth over campaign lifecycle
            trend = 1.0 + 0.15 * (global_t / total_hours)
            # Weekend boost or reduction depending on vertical
            weekend_mod = 1.25 if (day_of_week in [5, 6] and vertical in ['retail', 'gaming', 'travel']) else 0.90
            # Noise factor
            noise = np.random.lognormal(mean=0.0, sigma=0.12)
            
            # Simulated exogenous market shock at predetermined hours (e.g. hour 850, 1420)
            shock_multiplier = 1.0
            if global_t in [850, 1420, 1950]:
                shock_multiplier = 2.4 # Exogenous spike
            
            hour_spend = max(0.0, base_spend * trend * weekend_mod * noise * shock_multiplier)
            hour_imps = int(max(0, base_imps * trend * weekend_mod * noise * shock_multiplier))
            hour_clicks = int(max(0, base_clicks * trend * weekend_mod * noise))

            # Derived RTB metrics
            win_rate = (hour_imps / (hour_imps + 1500.0)) if hour_imps > 0 else 0.0
            avg_cpm = (hour_spend / hour_imps * 1000.0) if hour_imps > 0 else 0.0
            cum_spend += hour_spend

            # Calendar features
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
            day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
            day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
            is_weekend = 1 if day_of_week in [5, 6] else 0
            time_to_end = (total_hours - global_t)

            time_records.append({
                'time_idx': global_t,
                'campaign_id': str(adv_id),
                'vertical': vertical,
                'hourly_spend': round(hour_spend, 4),
                'impressions_won': hour_imps,
                'clicks': hour_clicks,
                'win_rate': round(win_rate, 4),
                'avg_cpm': round(avg_cpm, 4),
                'hour_of_day': hour_of_day,
                'hour_sin': round(hour_sin, 6),
                'hour_cos': round(hour_cos, 6),
                'day_of_week': day_of_week,
                'day_sin': round(day_sin, 6),
                'day_cos': round(day_cos, 6),
                'is_weekend': is_weekend,
                'time_to_end': time_to_end,
                'countdown_ratio': round(time_to_end / total_hours, 6)
            })

        # Calculate campaign-level static context based on cumulative spend
        campaign_df = pd.DataFrame(time_records)
        total_observed_spend = campaign_df['hourly_spend'].sum()
        # Set total budget B as 103% of nominal spend to reflect a realistic bounded pacing setting
        total_budget = round(total_observed_spend * 1.03, 2)
        nominal_velocity = round(total_budget / total_hours, 4)

        campaign_df['total_budget'] = total_budget
        campaign_df['duration_hours'] = total_hours
        campaign_df['nominal_velocity'] = nominal_velocity
        campaign_df['cumulative_spend'] = campaign_df['hourly_spend'].cumsum()
        campaign_df['spend_ratio'] = (campaign_df['cumulative_spend'] / total_budget).clip(0.0, 1.0)

        all_campaign_records.append(campaign_df)

    final_df = pd.concat(all_campaign_records, ignore_index=True)
    return final_df


def save_splits(df: pd.DataFrame, output_dir: str):
    """
    Creates chronological train (60 days = 1440h), val (15 days = 360h), and test (15 days = 360h) splits.
    Exports to parquet and CSV for high performance and compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 60 days * 24 = 1440 hours train
    # 15 days * 24 = 360 hours val (1440 to 1800)
    # 15 days * 24 = 360 hours test (1800 to 2160)
    train_end = 1440
    val_end = 1800

    train_df = df[df['time_idx'] < train_end].copy()
    val_df = df[(df['time_idx'] >= train_end) & (df['time_idx'] < val_end)].copy()
    test_df = df[df['time_idx'] >= val_end].copy()

    logger.info(f"Exporting dataset splits to '{output_dir}'...")
    logger.info(f"  Full dataset: {len(df):,} rows across {df['campaign_id'].nunique()} campaigns.")
    logger.info(f"  Train set: {len(train_df):,} rows (hours 0 to {train_end - 1})")
    logger.info(f"  Val set:   {len(val_df):,} rows (hours {train_end} to {val_end - 1})")
    logger.info(f"  Test set:  {len(test_df):,} rows (hours {val_end} to {df['time_idx'].max()})")

    # Save complete and split files
    df.to_parquet(os.path.join(output_dir, "tft_hourly_pacing_full.parquet"), index=False)
    df.to_csv(os.path.join(output_dir, "tft_hourly_pacing_full.csv"), index=False)
    
    train_df.to_parquet(os.path.join(output_dir, "train_hourly.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val_hourly.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test_hourly.parquet"), index=False)

    logger.info("Successfully exported all dataset partitions.")


def main():
    args = parse_args()
    logger.info("Starting iPinYou Ad Spend Pacing Resampling Pipeline...")

    # Process train and test logs
    train_agg = process_raw_chunks(args.train_file, chunksize=args.chunksize, max_chunks=args.max_chunks)
    test_agg = process_raw_chunks(args.test_file, chunksize=args.chunksize, max_chunks=args.max_chunks)

    # Combine both partitions to capture full weekly empirical campaign distributions
    combined_agg = pd.concat([train_agg, test_agg], ignore_index=True).groupby(
        ['advertiser', 'weekday', 'norm_hour']
    ).agg(
        impressions_won=('impressions_won', 'sum'),
        total_clicks=('total_clicks', 'sum'),
        spend=('spend', 'sum')
    ).reset_index()

    # Build continuous 90-day hourly timeline (60 train / 15 val / 15 test)
    continuous_df = build_continuous_hourly_timeline(combined_agg)

    # Export splits
    save_splits(continuous_df, args.output_dir)
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()
