#!/usr/bin/env python3
"""
Google Maps Scraped Data Cleaner
Cleans and merges multiple Google Maps review CSV files according to specifications
"""

import pandas as pd
import json
import re
from datetime import datetime
import os
import glob
from pathlib import Path

def drop_columns(df, columns_to_drop):
    """Drop specified columns if they exist"""
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"      🗑️ Dropped columns: {existing_columns}")
    return df

def handle_review_text(df):
    """Replace review_text with review_translated_text if available"""
    if 'review_translated_text' in df.columns and 'review_text' in df.columns:
        # Create a mask for non-empty translated text
        mask = df['review_translated_text'].notna() & (df['review_translated_text'] != '')
        
        # Replace review_text with translated text where available
        df.loc[mask, 'review_text'] = df.loc[mask, 'review_translated_text']
        
        # Count replacements
        replacement_count = mask.sum()
        if replacement_count > 0:
            print(f"      🔄 Replaced {replacement_count} review texts with translated versions")
        
        # Drop the review_translated_text column as it's no longer needed
        df = df.drop(columns=['review_translated_text'])
    
    return df

def convert_published_date(df):
    """Convert published_at_date to datetime format"""
    if 'published_at_date' in df.columns:
        try:
            # Convert to datetime, handling various formats
            df['published_at_date'] = pd.to_datetime(df['published_at_date'], errors='coerce')
            
            # Count successful conversions
            successful_conversions = df['published_at_date'].notna().sum()
            print(f"      📅 Converted {successful_conversions} dates to datetime format")
            
        except Exception as e:
            print(f"      ⚠️ Warning: Could not convert all dates: {e}")
    
    return df

def convert_response_to_binary(df):
    """Convert response_from_owner_text to binary variable 'have_responded'"""
    if 'response_from_owner_text' in df.columns:
        # Create binary column: 1 if there's a response, 0 if not
        df['have_responded'] = df['response_from_owner_text'].notna() & (df['response_from_owner_text'] != '')
        df['have_responded'] = df['have_responded'].astype(int)
        
        # Count responses
        response_count = df['have_responded'].sum()
        total_reviews = len(df)
        print(f"      💬 Owner responses: {response_count}/{total_reviews} ({response_count/total_reviews*100:.1f}%)")
        
        # Drop the original column
        df = df.drop(columns=['response_from_owner_text'])
    
    return df

def fill_local_guide(df):
    """Fill NaN values for is_local_guide with 0"""
    if 'is_local_guide' in df.columns:
        # Count NaN values before filling
        nan_count = df['is_local_guide'].isna().sum()
        if nan_count > 0:
            df['is_local_guide'] = df['is_local_guide'].fillna(0)
            print(f"      👥 Filled {nan_count} NaN values in is_local_guide with 0")
        
        # Convert to int if not already
        df['is_local_guide'] = df['is_local_guide'].astype(int)
    
    return df

def expand_experience_details(df):
    """Expand experience_details JSON into multiple columns"""
    if 'experience_details' in df.columns:
        print(f"      🔍 Expanding experience_details into separate columns...")
        
        # Initialize new columns
        df['visited_on'] = None
        df['wait_time'] = None
        df['reservation_recommended'] = None
        
        # Process each row
        for idx, row in df.iterrows():
            experience_details = row['experience_details']
            
            # Debug: Print first few experience_details to understand the structure
            if idx < 3:
                print(f"         🔍 Row {idx} experience_details: {experience_details}")
            
            if pd.notna(experience_details) and experience_details != '' and experience_details != '[]':
                try:
                    # Parse JSON string
                    if isinstance(experience_details, str):
                        details = json.loads(experience_details)
                    else:
                        details = experience_details
                    
                    # Extract values
                    for detail in details:
                        if isinstance(detail, dict) and 'name' in detail and 'value' in detail:
                            name = detail['name']
                            value = detail['value']
                            
                            # Check if name is not None before calling .lower()
                            if name is not None and isinstance(name, str):
                                name_lower = name.lower()
                                
                                if 'visited on' in name_lower:
                                    df.at[idx, 'visited_on'] = value
                                elif 'wait time' in name_lower:
                                    df.at[idx, 'wait_time'] = value
                                elif 'reservation' in name_lower:
                                    df.at[idx, 'reservation_recommended'] = value
                
                except (json.JSONDecodeError, TypeError) as e:
                    # Skip invalid JSON and log the error for debugging
                    print(f"         ⚠️ Warning: Could not parse experience_details at row {idx}: {e}")
                    continue
        
        # Count non-null values in new columns
        visited_count = df['visited_on'].notna().sum()
        wait_time_count = df['wait_time'].notna().sum()
        reservation_count = df['reservation_recommended'].notna().sum()
        
        print(f"         📅 Visited on: {visited_count} values")
        print(f"         ⏱️ Wait time: {wait_time_count} values")
        print(f"         📋 Reservation: {reservation_count} values")
        
        # Drop the original experience_details column
        df = df.drop(columns=['experience_details'])
    
    return df

def clean_single_file(file_path, columns_to_drop):
    """Clean a single CSV file according to specifications"""
    print(f"🔄 Cleaning file: {os.path.basename(file_path)}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"   📊 Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Store original shape for comparison
        original_shape = df.shape
        
        # 1. Drop specified columns
        df = drop_columns(df, columns_to_drop)
        
        # 2. Handle review text replacement
        df = handle_review_text(df)
        
        # 3. Convert published_at_date to datetime
        df = convert_published_date(df)
        
        # 4. Convert response_from_owner_text to binary
        df = convert_response_to_binary(df)
        
        # 5. Fill NaN values for is_local_guide
        df = fill_local_guide(df)
        
        # 6. Expand experience_details into multiple columns
        df = expand_experience_details(df)
        
        print(f"   ✅ Cleaned to {len(df)} rows, {len(df.columns)} columns")
        print(f"   📈 Shape change: {original_shape} → {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error cleaning file {file_path}: {e}")
        return None

def main():
    """Main function to run the data cleaner"""
    # Set input and output directories
    input_directory = "/Users/acch/Downloads/InterUni Datathon 2025/google_map_scraped"
    output_directory = "data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Columns to drop
    columns_to_drop = [
        'place_id', 'review_id', 'review_link', 'name', 'reviewer_id', 
        'reviewer_profile', 'published_at', 'response_from_owner_ago', 
        'response_from_owner_date', 'total_number_of_reviews_by_reviewer', 
        'total_number_of_photos_by_reviewer', 'response_from_owner_translated_text', 
        'review_photos'
    ]
    
    print(f"🚀 Starting data cleaning process...")
    print(f"📁 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_directory}")
        return
    
    print(f"📋 Found {len(csv_files)} CSV files to process")
    print("=" * 60)
    
    # Process each file
    cleaned_data = []
    for file_path in csv_files:
        cleaned_df = clean_single_file(file_path, columns_to_drop)
        if cleaned_df is not None:
            cleaned_data.append(cleaned_df)
            print()
    
    # Merge all cleaned data
    if cleaned_data:
        print("🔄 Merging all cleaned data...")
        
        # Concatenate all dataframes
        merged_df = pd.concat(cleaned_data, ignore_index=True)
        
        print(f"📊 Final merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        print(f"📋 Columns: {list(merged_df.columns)}")
        
        # Save merged data
        output_file = os.path.join(output_directory, "merged_google_maps_reviews.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"💾 Merged data saved to: {output_file}")
        
        # Display sample of cleaned data
        print("\n📋 Sample of cleaned data:")
        print(merged_df.head())
        
        print("\n✅ Data cleaning and merging completed successfully!")
    else:
        print("❌ No data was successfully cleaned")

if __name__ == "__main__":
    main() 