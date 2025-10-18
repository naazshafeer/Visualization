import pandas as pd
from astropy.io import ascii
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # parent folder
WISE_DIR = os.path.join(BASE_DIR, 'data')  # WISE .tbl files
SDSS_DIR = os.path.join(BASE_DIR, 'data', 'data/Dataframe_csv')  # SDSS CSVs
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')  # save unmatched CSVs here

os.makedirs(OUTPUT_DIR, exist_ok=True) #chat helped with this cuz i wanted to make the input and output more routine


def load_tbl(file_tbl):
    wise_table = ascii.read(file_tbl)
    wise_df = wise_table.to_pandas()
    return wise_df


def load_csv(file_csv):
    sdss_df = pd.read_csv(file_csv)
    return sdss_df

def compare_catalogs(wise_df, sdss_df, wise_col='designation', sdss_col='SDSS_designation'):
    unmatched = wise_df[~wise_df[wise_col].isin(sdss_df[sdss_col])]
    

    unmatched = unmatched.reset_index()
    return unmatched

def main():
    for wise_file in os.listdir(WISE_DIR):
        if wise_file.endswith('.tbl'):
            wise_path = os.path.join(WISE_DIR, wise_file)
            wise_df = load_tbl(wise_path)

            for sdss_file in os.listdir(SDSS_DIR):
                if sdss_file.endswith('.csv'):
                    sdss_path = os.path.join(SDSS_DIR, sdss_file)
                    sdss_df = load_csv(sdss_path)

                    unmatched_entries = compare_catalogs(
                        wise_df, sdss_df, wise_col='designation', sdss_col='SDSS designation'
                    )


                    output_file = f'unmatched_{os.path.splitext(wise_file)[0]}_{os.path.splitext(sdss_file)[0]}.csv'
                    output_path = os.path.join(OUTPUT_DIR, output_file)
                    unmatched_entries.to_csv(output_path, index=False)
                    print(f"Saved {len(unmatched_entries)} unmatched entries to {output_path}")

if __name__ == "__main__":
    main()