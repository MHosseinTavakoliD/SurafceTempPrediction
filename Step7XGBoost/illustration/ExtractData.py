import re
import pandas as pd

# File paths
input_txt_file = "resualt.txt"  # replace with your input file path
output_csv_file = "resualt.csv"  # replace with your desired output CSV file path



# Read the contents of the file
with open(input_txt_file, 'r') as file:
    lines = file.readlines()

# Process each line
data = []
for line in lines:
    # Extracting numbers following specific patterns (rmse and mae values)
    rmse_mae_values = re.findall(r'rmse:(\d+\.\d+)|mae:(\d+\.\d+)', line)
    numbers = [num for t in rmse_mae_values for num in t if num]  # flatten and filter out empty strings

    # Debug: Print extracted numbers
    print(f"Extracted Numbers: {numbers}")

    # Assuming the order is: [train_rmse, train_mae, val_rmse, val_mae]
    if len(numbers) == 4:
        train_rmse, train_mae, val_rmse, val_mae = map(float, numbers)
        estimator = int(re.search(r'\[(\d+)\]', line).group(1))  # assuming estimator number is within brackets
        data.append({
            "Estimator": estimator,
            "Train_RMSE": train_rmse,
            "Train_MAE": train_mae,
            "Val_RMSE": val_rmse,
            "Val_MAE": val_mae,
            "Train_MSE": train_rmse ** 2,
            "Val_MSE": val_rmse ** 2
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Check if data was found
if not df.empty:
    # Save to CSV
    df.to_csv(output_csv_file, index=False)
    print(f"Data successfully written to {output_csv_file}")
else:
    print("No data was extracted. Please check the input file and the format.")
