import pandas as pd
from app.config import get_settings

# Load configuration
settings = get_settings()

# Load database once
vehicles = pd.read_csv(settings.vehicle_database_path)

def get_vehicle_details(vehicle_no):
    vehicle = vehicles[vehicles["vehicle_no"] == vehicle_no]

    if vehicle.empty:
        return None

    vehicle = vehicle.iloc[0]

    return {
        "vehicle_no": vehicle["vehicle_no"],
        "type": vehicle["type"],
        "fuel": vehicle["fuel"],
        "engine_size": vehicle["engine_size"],
        "mileage": vehicle["mileage"]
    }


# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    print("🚗 Vehicle Information System")
    print("-----------------------------")

    vehicle_number = input("Enter Vehicle Number: ").strip()

    details = get_vehicle_details(vehicle_number)

    if details:
        print("\n✅ Vehicle Found!")
        print("-----------------------------")
        print(f"Vehicle Number : {details['vehicle_no']}")
        print(f"Type           : {details['type']}")
        print(f"Fuel           : {details['fuel']}")
        print(f"Engine Size    : {details['engine_size']} L")
        print(f"Mileage        : {details['mileage']} km/l")
    else:
        print("\n❌ Vehicle not found in database.")