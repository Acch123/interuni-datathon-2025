from serpapi import GoogleSearch
import csv

def searchAndSaveToCSV(params, filename):
    search = GoogleSearch(params)
    results = search.get_dict()

    propertyResults = results["properties"]

    clean_and_save_to_csv(propertyResults,filename)


# def clean_and_save_to_csv(data, filename):
#     """
#     Cleans a list of hotel data and saves it to a CSV file.

#     Args:
#         data (list): A list of dictionaries, where each dictionary contains
#                      information about a hotel.
#         filename (str): The name of the output CSV file.
#     """
#     # Define the headers for the CSV file. These are the keys we want to extract.
#     headers = [
#         'name', 'description', 'link', 'hotel_class', 'overall_rating',
#         'reviews', 'latitude', 'longitude', 'check_in_time',
#         'check_out_time', 'rate_per_night_lowest', 'total_rate_lowest',
#         'amenities'
#     ]

#     # Open the file in write mode with newline='' to prevent blank rows.
#     with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=headers)

#         # Write the header row to the CSV file.
#         writer.writeheader()

#         # Iterate over each hotel dictionary in the data list.
#         for hotel in data:
#             # Create a dictionary to hold the cleaned row data.
#             # We use .get() to avoid errors if a key is missing.
#             row = {
#                 'name': hotel.get('name'),
#                 'description': hotel.get('description'),
#                 'link': hotel.get('link'),
#                 'hotel_class': hotel.get('hotel_class'),
#                 'overall_rating': hotel.get('overall_rating'),
#                 'reviews': hotel.get('reviews'),
                
#                 # Safely get nested GPS coordinates
#                 'latitude': hotel.get('gps_coordinates', {}).get('latitude'),
#                 'longitude': hotel.get('gps_coordinates', {}).get('longitude'),
                
#                 'check_in_time': hotel.get('check_in_time'),
#                 'check_out_time': hotel.get('check_out_time'),
                
#                 # Safely get nested rate information
#                 'rate_per_night_lowest': hotel.get('rate_per_night', {}).get('lowest'),
#                 'total_rate_lowest': hotel.get('total_rate', {}).get('before_taxes_fees'),

#                 "lowest": "String - Lowest total rate for the entire trip formatted with currency",
#                 "extracted_lowest": "Float - Extracted lowest total rate for the entire trip",
#                 "before_taxes_fees": "String - Total rate before taxes and fees for the entire trip formatted with currency",
#                 "extracted_before_taxes_fees": "Float - Extracted total rate before taxes and fees for the entire trip"
                
#                 # Join the list of amenities into a single string
#                 'amenities': ', '.join(hotel.get('amenities', []))
#             }
#             # Write the processed row to the CSV file.
#             writer.writerow(row)
    
#     print(f"Data successfully saved to {filename}")

def clean_and_save_to_csv(data, filename):
    """
    Cleans a list of hotel data, including nearby places, and saves it to a CSV file.

    Args:
        data (list): A list of dictionaries, where each dictionary contains
                     information about a hotel.
        filename (str): The name of the output CSV file.
    """
    # Define the headers for the CSV file.
    headers = [
        'name', 'description', 'link', 'hotel_class', 'overall_rating',
        'reviews', 'latitude', 'longitude', 'check_in_time',
        'check_out_time', 'rate_per_night_lowest', 'total_rate_lowest',
        'amenities', 'nearby_places_formatted'
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for hotel in data:
            # Process Nearby Places into a formatted string.
            nearby_places_list = []
            if hotel.get('nearby_places'):
                for place in hotel.get('nearby_places', []):
                    place_name = place.get('name', 'N/A')
                    transport_info = []
                    for transport in place.get('transportations', []):
                        transport_type = transport.get('type', '')
                        duration = transport.get('duration', '')
                        transport_info.append(f"{transport_type} ({duration})")
                    transport_str = ", ".join(transport_info)
                    nearby_places_list.append(f"{place_name}: {transport_str}")
            nearby_places_formatted = "; ".join(nearby_places_list)

            # Create a dictionary for the row.
            row = {
                'name': hotel.get('name'),
                'description': hotel.get('description'),
                'link': hotel.get('link'),
                'hotel_class': hotel.get('hotel_class'),
                'overall_rating': hotel.get('overall_rating'),
                'reviews': hotel.get('reviews'),
                'latitude': hotel.get('gps_coordinates', {}).get('latitude'),
                'longitude': hotel.get('gps_coordinates', {}).get('longitude'),
                'check_in_time': hotel.get('check_in_time'),
                'check_out_time': hotel.get('check_out_time'),
                'rate_per_night_lowest': hotel.get('rate_per_night', {}).get('lowest'),
                'total_rate_lowest': hotel.get('total_rate', {}).get('extracted_lowest'),
                'amenities': ', '.join(hotel.get('amenities', [])),
                'nearby_places_formatted': nearby_places_formatted
            }
            writer.writerow(row)


#                 "lowest": "String - Lowest total rate for the entire trip formatted with currency",
#                 "extracted_lowest": "Float - Extracted lowest total rate for the entire trip",
#                 "before_taxes_fees": "String - Total rate before taxes and fees for the entire trip formatted with currency",
#                 "extracted_before_taxes_fees": "Float - Extracted total rate before taxes and fees for the entire trip"
    
    print(f"Data successfully saved to {filename}")


dateList = [
    ["2026-06-09", "2026-06-15"],
    ["2026-06-16", "2026-06-22"],
    ["2026-06-23", "2026-06-29"],
    ["2026-06-30", "2026-07-06"],
    ["2026-07-07", "2026-07-13"],
    ["2026-07-14", "2026-07-20"],
    ["2026-07-21", "2026-07-27"],
    ["2026-07-28", "2026-08-03"],
    ["2026-08-04", "2026-08-10"],
    ["2026-08-11", "2026-08-17"],
    ["2026-08-18", "2026-08-24"],
    ["2026-08-25", "2026-08-31"],
    ["2026-09-01", "2026-09-07"],
    ["2026-09-08", "2026-09-14"],
    ["2026-09-15", "2026-09-21"]
]

try:
    for i in range(15):
        currentDate = dateList[i]
        checkIn = currentDate[0]
        checkOut = currentDate[1]

        params = {
            "engine": "google_hotels",
            "q": "Mount Baw Baw accommodation",
            "check_in_date": checkIn,
            "check_out_date": checkOut,
            "adults": "2",
            "currency": "AUD",
            "gl": "au",
            "hl": "en",
            "property_types": "14,15,16,17,18,19,20,21",
            "api_key": "key"
        }

        searchAndSaveToCSV(params, f"Mount_Baw_Baw_skiWeek{i}.csv")
except Exception as e:
    print("Failed to scrape: " + str(e))



# params = {
#     "engine": "google_hotels",
#     "q": "mount buller accommodation",
#     "check_in_date": "2026-06-09",
#     "check_out_date": "2026-06-15",
#     "adults": "2",
#     "currency": "AUD",
#     "gl": "au",
#     "hl": "en",
#     "property_types": "14,15,16,17,18,19,20,21",
#     "api_key": "b16f55269dc7a00406f622187b3663cce9e4f7f455529f2390fae89639a1ca3f"
# }

# search = GoogleSearch(params)
# results = search.get_dict()

# print(results)

# propertyResults = results["properties"]

# print(propertyResults)

# # clean_and_save_to_csv(propertyResults,filename)