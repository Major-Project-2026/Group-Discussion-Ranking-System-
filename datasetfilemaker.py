import os
import csv

audio_directory = 'C:\\Users\\palak\\Downloads\\emotions\\surprise\\wavs'
output_csv = 'C:\\Users\\palak\\OneDrive\\Desktop\\surprise.csv'


# Get a list of all audio files in the directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# Create a list to store the data for the CSV file
csv_data = []

# Extract the emotion labels from the filenames and create the CSV data
for file in audio_files:
    emotion = os.path.splitext(file)[0]  # Assuming the filename is the emotion
    csv_data.append([os.path.join(audio_directory, file), emotion])

# Write the data to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['path', 'label'])  # Write the header
    csvwriter.writerows(csv_data)

print(f'CSV file created at {output_csv}')
