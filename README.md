# Shortest-path-through-urban-green-spaces-software

The Shortest path through urban green spaces software is a Python script designed to help users plan walking routes that incorporate urban green spaces within a specified city or area. The script utilizes the OpenStreetMap (OSM) data and the osmnx library to extract street network information, identify green spaces, and calculate optimal walking routes that maximize exposure to green areas. It also compares the travel time between the direct route and the route that includes urban green spaces.

# Prerequisites
Before running the script, ensure you have the following:

Python (3.10)
Required Python packages (install using pip):
osmnx
geopandas
matplotlib

Run the urban_green_spaces_route_planner.py script using your preferred Python interpreter.

Specify the Location:

In the script, specify the location for which you want to plan the route by setting the place_name variable to the desired city or area, e.g., 'Vancouver,Canada'.

Run the Script:

Execute the script, and it will generate a map showing various route options. The script will identify urban green spaces, compute route lengths, and provide time comparisons between direct routes and routes through green spaces.

Script Description
The script performs the following steps:

Extracts the OSM street network using the osmnx library for the specified location.
Identifies urban green spaces using specified tags (e.g., 'leisure': 'park').
Prompts the user to click on the map to select start and end points for the route.
Computes the shortest path routes using the ox.shortest_path function.
Compares route options based on the presence of urban green spaces.
Plots the routes on a map, highlighting urban green spaces and providing time comparisons.
Additional Information
The script offers flexibility to handle different scenarios based on the number and size of urban green spaces within the specified area.
The walking speed for route calculations is set as a constant, which can be adjusted based on preferences.
Disclaimer
Please note that the accuracy and availability of OSM data can vary. The script's functionality is dependent on the data available for the specified location.
