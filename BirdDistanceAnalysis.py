import argparse
import os
import librosa
import numpy as np
from scipy.fft import fft
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Process bird audio data from a CSV file.")

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "scientific_name",
        type=str,
        help="Scientific name of the bird"
    )

    parser.add_argument(
        "hz_interval",
        type=str,
        help="Frequency interval in Hz, e.g. '2500' or '2500-4000'"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path"
    )

    args = parser.parse_args()

    # Validate CSV file exists
    if not os.path.isfile(args.csv_file):
        parser.error(f"The file '{args.csv_file}' does not exist.")

    # Validate frequency format
    try:
        if '-' in args.hz_interval:
            start, end = map(float, args.hz_interval.split('-'))
            if start >= end:
                raise ValueError
        else:
            float(args.hz_interval)
    except ValueError:
        parser.error("hz_interval must be a float or a range in format 'start-end' where start < end.")

    return args

def load_librosa_file(file_path):
    print(f"loading {file_path}")
    y, sr = librosa.load(file_path)
    print(f"Sample rate: {sr} Hz")
    print(f"Total duration: {len(y)/sr:.2f} seconds")
    return y, sr

def analyze_segment(audio_segment: np.ndarray, sample_rate: int, frequency_range: tuple[int, int] = (500, 1000)) -> \
tuple[float, float, float, float, float]:
    # Perform Fourier Transform on the segment
    #print("Starting with Fourier Transform on the segment")
    fft_result = fft(audio_segment)
    frequencies = np.fft.fftfreq(len(audio_segment), 1 / sample_rate)

    # Calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fft_result)

    min_freq_hz, max_freq_hz = frequency_range

    # Find the indices corresponding to the desired frequency range
    freq_indices = np.where((frequencies >= min_freq_hz) & (frequencies <= max_freq_hz))[0]

    # Extract the amplitudes within the frequency range
    amplitude_in_range = amplitude_spectrum[freq_indices]

    # Calculate the overall amplitude in the specified frequency range (e.g., sum of amplitudes)
    total_amplitude_in_range = np.sum(amplitude_in_range)
    median_amplitude_in_range = np.median(amplitude_in_range)
    mean_amplitude_in_range = np.mean(amplitude_in_range)
    max_amplitude_in_range = np.max(amplitude_in_range)
    min_amplitude_in_range = np.min(amplitude_in_range)

    return total_amplitude_in_range, median_amplitude_in_range, mean_amplitude_in_range, max_amplitude_in_range, min_amplitude_in_range


def main(csv_file, target_bird, min_hz, max_hz, output_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    matches = df[(df["Scientific name"] == target_bird)]
    # Display the dataframe
    # print(df)

    # Pre‑load each unique file into a cache
    audio_cache = {}
    for file_path in matches["File"].unique():
        print(f"Loading {file_path}")
        y, sr = librosa.load(file_path)
        audio_cache[file_path] = (y, sr)

    # Do something for each match
    analysis_results = []
    for _, row in matches.iterrows():
        start = row["Start (s)"]
        end = row["End (s)"]
        confidence = row["Confidence"]
        file = row["File"]
        y, sr = audio_cache[file_path]

        # Do your custom action here
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = y[start_sample:end_sample]
        print(f"Found {target_bird} from {start}s to {end}s with confidence {confidence}")
        analysis_results.append(
            analyze_segment(audio_segment, sr, (min_hz, max_hz))
        )

    # Convert analysis_df into a DataFrame and add column names
    analysis_df = pd.DataFrame(analysis_results, columns=[
        'Total Amplitude',
        'Median Amplitude',
        'Mean Amplitude',
        'Max Amplitude',
        'Min Amplitude'
    ])
    combined_df = pd.concat([matches.reset_index(drop=True), analysis_df], axis=1)
    # Write to CSV
    if output_file is None:
        output_file = f'{csv_file.rstrip(".csv")}+analysis_freqAmplitude.csv'

    combined_df.to_csv(output_file, index=False)
    print("🔬 Done! Results written to CSV file.")

if __name__ == "__main__":
    args = parse_args()
    print("CSV File:", args.csv_file)
    csv_file = args.csv_file
    print("Scientific Name:", args.scientific_name)
    bird_name = args.scientific_name
    print("Hz Interval:", args.hz_interval)
    min_hz, max_hz = map(float, args.hz_interval.split('-')) if '-' in args.hz_interval else (args.hz_interval, args.hz_interval)
    print("Output Path:", args.output if args.output else "Not specified")
    output_file = args.output
    main(csv_file, bird_name, min_hz, max_hz, output_file)