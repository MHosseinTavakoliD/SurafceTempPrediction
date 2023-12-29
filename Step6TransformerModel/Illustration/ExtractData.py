import csv

# Open the file and read the lines
with open('result.txt', 'r') as file:
    data_lines = file.readlines()

# Filtering out non-epoch lines and extracting the current learning rate
epoch_data = []
current_learning_rate = ""
for line in data_lines:
    if "Current learning rate" in line:
        current_learning_rate = line.split(":")[1].strip()
    elif "Epoch" in line:
        parts = line.split(":")
        # Ensure that there are enough parts to extract the data
        if len(parts) >= 6:
            epoch_number = parts[0].split()[1]  # Extracting the epoch number without the colon
            train_loss = parts[2].split(",")[0].strip()  # Extracting the train loss value
            train_mae = parts[3].split(",")[0].strip()  # Extracting the train MAE value
            val_loss = parts[4].split(",")[0].strip()  # Extracting the val loss value
            val_mae = parts[5].strip()  # Extracting the val MAE value
            epoch_data.append([epoch_number, current_learning_rate, train_loss, train_mae, val_loss, val_mae])

# Defining CSV headers
headers = ["Epoch", "Learning Rate", "Train Loss", "Train MAE", "Val Loss", "Val MAE"]

# Writing to CSV
csv_file_path = "result.csv"  # Change this to your desired file path
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Writing headers
    writer.writerows(epoch_data)

print(f"Data written to {csv_file_path}")
