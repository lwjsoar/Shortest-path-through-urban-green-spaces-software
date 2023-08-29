# Shortest path through urban green spaces software
# This is the code for Weijun Li's dissertation

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.geometry import box

def calculate_longitude_diff_for_fixed_distance(start_lat, start_lon, distance_meters):
    """
    Calculate the difference in longitude for a given fixed distance, moving eastward.

    Args:
        start_lat (float): Starting latitude.
        start_lon (float): Starting longitude.
        distance_meters (float): Fixed distance in meters.

    Returns:
        float: Difference in longitude for the given distance.
    """
    new_coords = geodesic(meters=distance_meters).destination((start_lat, start_lon), 90)
    return new_coords[1] - start_lon

def get_user_input(graph):
    """
    Get user input and create a map with overlays and annotations.

    Args:
        graph: Street graph data.

    Returns:
        None
    """
    # Plot the street graph first with larger figsize
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_color='gray', node_size=5, edge_color='gray',
                            edge_alpha=0.5, bgcolor='white', figsize=(16, 16))

    # Overlay parks with green color
    if not parks.empty:
        parks.plot(ax=ax, facecolor='green', alpha=0.7)

    # Overlay buildings
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor='khaki', edgecolor='black', alpha=0.7)

    # Add a North Arrow using Arrow patch
    arrow = mpatches.Arrow(0.05, 0.85, 0, 0.1, width=0.02, transform=ax.transAxes, color='black')
    ax.add_patch(arrow)
    ax.text(0.05, 0.96, 'N', transform=ax.transAxes, ha='center', va='center', color='black', fontsize=20)

    # Calculate the Longitude Difference for 500 meters at the current latitude and longitude
    current_lat = ax.get_ylim()[0]
    current_lon = ax.get_xlim()[0]
    long_diff = calculate_longitude_diff_for_fixed_distance(current_lat, current_lon, 500)  # for 500 meters

    # Define scale bar position and width, place it closer to the lower-left corner
    lon_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    lat_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    scalebar_x = current_lon + 0.01 * lon_range
    scalebar_y = current_lat + 0.01 * lat_range
    scalebar_width = long_diff

    # Draw the Scale Bar
    scalebar = mpatches.Rectangle((scalebar_x, scalebar_y), scalebar_width, 0.002, color='black')
    ax.add_patch(scalebar)

    # Label the Scale Bar, placing the text slightly below the scale bar
    ax.text(scalebar_x + scalebar_width / 2, scalebar_y - 0.003, '500 m', ha='center', va='top', color='black')

    # Add a Legend
    gray_patch = mpatches.Patch(color='gray', label='Street Network')
    green_patch = mpatches.Patch(color='green', label='Parks')
    khaki_patch = mpatches.Patch(color='khaki', label='Buildings')
    ax.legend(handles=[gray_patch, green_patch, khaki_patch])

    # Add a hint on the map
    ax.text(0.5, 0.95, "Click on the map for the start and end points, respectively.",
            transform=ax.transAxes, ha='center', va='center', color='blue', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='blue', boxstyle='round'))

    coords = []  # Placeholder for user input coordinates

    def onclick(event):
        """
        Callback function for handling mouse click events on the map.

        Args:
            event: Mouse click event containing x and y data.

        Returns:
            None
        """
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))

        # Close the figure after 2 clicks
        if len(coords) == 2:
            plt.close()

    # Connect the click event to the callback function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Extract start and end coordinates
    start_coords, end_coords = coords

    return start_coords, end_coords

def find_nearest_park(point, parks):
    """
    Find the nearest park to a given point.

    Args:
        point: Point coordinates in the same CRS as the parks.
        parks: GeoDataFrame containing park geometries.

    Returns:
        Series: Information about the nearest park.
    """
    # Project parks to a local CRS
    parks_projected = parks.to_crs("EPSG:3153")

    # Create a GeoSeries from the point, with the same CRS as the projected parks
    point_geoseries = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3153")

    # Calculate distances to the point and find the index of the closest park
    nearest_park_idx = parks_projected.geometry.distance(point_geoseries[0]).idxmin()
    return parks.loc[nearest_park_idx]

def compute_route_length(graph, route):
    """
    Compute the length of a route using edge weights.

    Args:
        graph: Street graph data.
        route: List of nodes representing the route.

    Returns:
        float: Length of the route.
    """
    route_length = 0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        # Extract data for all edges between u and v
        data = graph.get_edge_data(u, v)

        # If there are multiple edges, choose the one with the shortest length
        if data:
            min_length = min([d['length'] for d in data.values()])
            route_length += min_length
    return route_length

# Specify the name that is used to search for the data
place_name = 'Vancouver, Canada'

# Fetch OSM street network from the location, set the network type as 'walk'
graph = ox.graph_from_place(place_name, network_type='walk')

# Retrieve nodes and edges
nodes, edges = ox.graph_to_gdfs(graph)

# Get place boundary related to the place name as a geodataframe
area = ox.geocode_to_gdf(place_name)

# List key-value pairs for tags
tags = {'building': True}
try:
    buildings = ox.features_from_place(place_name, tags)
except Exception as e:
    print(f"No features found for {tags}. Exception: {e}")

tags = {'leisure': 'park', 'landuse': 'grass'}
try:
    parks = ox.features_from_place(place_name, tags)
except Exception as e:
    print(f"No features found for {tags}. Exception: {e}")

# Use the function
start_coords, end_coords = get_user_input(graph)

# Convert start and end coordinates to Points
start_point = Point(start_coords)
end_point = Point(end_coords)

# Get the nearest nodes to the start and end point
start_node = ox.nearest_nodes(graph, start_point.x, start_point.y)
end_node = ox.nearest_nodes(graph, end_point.x, end_point.y)

# Compute the shortest path
shortest_path = ox.shortest_path(graph, start_node, end_node, weight='length')

# Plot the full path
fig, ax = ox.plot_graph_route(graph, shortest_path, route_color='r', route_linewidth=6, node_size=0, show=False,
                               close=False)

# Get the bounding box of the full_path
path_nodes = [graph.nodes[node] for node in shortest_path]
xs = [node['x'] for node in path_nodes]
ys = [node['y'] for node in path_nodes]
minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)

# Apply a little padding to the bounding box for better visualization
padding = 0.01  # this value might need adjustment based on your data
ax.set_xlim([minx - padding, maxx + padding])
ax.set_ylim([miny - padding, maxy + padding])

# Plot the parks
parks.plot(ax=ax, color='green', alpha=0.5)
plt.show()

# Define the walking speed
walking_speed = 1.43  # m/s

# Compute the distance for the direct route (start to end)
direct_route_distance = compute_route_length(graph, shortest_path)

# Compute the time for the direct route
direct_route_time = direct_route_distance / walking_speed  # seconds

# Create a rectangle using the start and end points as diagonal points.
minx, miny = min(start_point.x, end_point.x), min(start_point.y, end_point.y)
maxx, maxy = max(start_point.x, end_point.x), max(start_point.y, end_point.y)
rectangle = box(minx, miny, maxx, maxy)

# Search for any urban green space inside this rectangle.
greenspaces_in_rectangle = parks[parks.geometry.intersects(rectangle)]

# Handle different cases based on the count of urban green spaces
num_greenspaces = len(greenspaces_in_rectangle)


if num_greenspaces == 0:
    print("No urban green spaces found within the rectangle.")

    # Find nearest parks
    nearest_park_start = find_nearest_park(start_point, parks)
    nearest_park_end = find_nearest_park(end_point, parks)

    # Identify the nearest nodes to the start point, first park, second park, and end point
    park1_node = ox.nearest_nodes(graph, nearest_park_start.geometry.centroid.x, nearest_park_start.geometry.centroid.y)
    park2_node = ox.nearest_nodes(graph, nearest_park_end.geometry.centroid.x, nearest_park_end.geometry.centroid.y)

    # Calculate the shortest paths from start point to park 1 to end point
    shortest_path_start_to_park1 = ox.shortest_path(graph, start_node, park1_node, weight='length')
    shortest_path_park1_to_end = ox.shortest_path(graph, park1_node, end_node, weight='length')

    # Combine the paths from start point to park 1 to end point
    park1_path = shortest_path_start_to_park1 + shortest_path_park1_to_end[1:]

    # Calculate the shortest paths from start point to park 2 to end point
    shortest_path_start_to_park2 = ox.shortest_path(graph, start_node, park2_node, weight='length')
    shortest_path_park2_to_end = ox.shortest_path(graph, park2_node, end_node, weight='length')

    # Combine the paths from start point to park 2 to end point
    park2_path = shortest_path_start_to_park2 + shortest_path_park2_to_end[1:]

    # Calculate the length of both paths
    path1_length = compute_route_length(graph, park1_path)
    path2_length = compute_route_length(graph, park2_path)

    # Compare the magnitude of the distances between the two and choose the path that has the shortest distance
    if path1_length < path2_length:
        best_path = park1_path
    else:
        best_path = park2_path

    # Plot the full path
    fig, ax = ox.plot_graph_route(graph, best_path, route_color='r', route_linewidth=6, node_size=0, show=False,
                                  close=False)

    # Get the bounding box of the full_path
    path_nodes = [graph.nodes[node] for node in best_path]
    xs = [node['x'] for node in path_nodes]
    ys = [node['y'] for node in path_nodes]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Apply a little padding to the bounding box for better visualization
    padding = 0.01  # this value might need adjustment based on your data
    ax.set_xlim([minx - padding, maxx + padding])
    ax.set_ylim([miny - padding, maxy + padding])

    # Plot the parks
    parks.plot(ax=ax, color='green', alpha=0.5)
    plt.show()

    # Compute the time for the direct route
    best_path_time = compute_route_length(graph, best_path) / walking_speed  # seconds
    # Compute the time difference
    time_difference = best_path_time - direct_route_time  # seconds

    print(f"Time for direct route: {direct_route_time / 60:.2f} minutes")
    print(f"Time for route through parks: {best_path_time / 60:.2f} minutes")
    print(f"Time difference: {time_difference / 60:.2f} minutes")

elif num_greenspaces == 1:
    print("One urban green space found within the rectangle.")

    # Get the single urban green space
    green_space = greenspaces_in_rectangle.iloc[0]

    # Find all nodes within the urban green space
    nodes_in_green_space = nodes[nodes.geometry.within(green_space.geometry)]

    if not nodes_in_green_space.empty:
        # If there are nodes within the urban green space, plan a path through its centroid

        # Identify the centroid of the green space
        centroid = green_space.geometry.centroid

        # Find the nearest node to the centroid
        nearest_node_to_centroid = ox.nearest_nodes(graph, centroid.x, centroid.y)

        # Plan a path from the start node to the centroid node, then to the end node
        path_to_centroid = ox.shortest_path(graph, start_node, nearest_node_to_centroid, weight='length')
        path_from_centroid = ox.shortest_path(graph, nearest_node_to_centroid, end_node, weight='length')

        # Combine the two paths
        full_path_through_centroid = path_to_centroid + path_from_centroid[1:]

        # Plot the full path
        fig, ax = ox.plot_graph_route(graph, full_path_through_centroid, route_color='r', route_linewidth=6,
                                      node_size=0, show=False,
                                      close=False)

        # Get the bounding box of the full_path
        path_nodes = [graph.nodes[node] for node in full_path_through_centroid]
        xs = [node['x'] for node in path_nodes]
        ys = [node['y'] for node in path_nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Apply a little padding to the bounding box for better visualization
        padding = 0.01  # this value might need adjustment based on your data
        ax.set_xlim([minx - padding, maxx + padding])
        ax.set_ylim([miny - padding, maxy + padding])

        # Plot the parks
        parks.plot(ax=ax, color='green', alpha=0.5)

        plt.show()

        # Compute the time for the direct route
        best_path_time = compute_route_length(graph, full_path_through_centroid) / walking_speed  # seconds
        # Compute the time difference
        time_difference = best_path_time - direct_route_time  # seconds

        print(f"Time for direct route: {direct_route_time / 60:.2f} minutes")
        print(f"Time for route through parks: {best_path_time / 60:.2f} minutes")
        print(f"Time difference: {time_difference / 60:.2f} minutes")

    else:
        print("No nodes found within the urban green space.")

        # Get the only green space
        single_green_space = greenspaces_in_rectangle.iloc[0]

        # Find nearest nodes to the start point and the centroid of the green space
        greenspace_centroid = single_green_space.geometry.centroid
        greenspace_node = ox.nearest_nodes(graph, greenspace_centroid.x, greenspace_centroid.y)

        # Compute the shortest path from start to green space
        path_start_to_greenspace = ox.shortest_path(graph, start_node, greenspace_node, weight='length')

        # Compute the shortest path from green space to end
        path_greenspace_to_end = ox.shortest_path(graph, greenspace_node, end_node, weight='length')

        # Combine the paths
        full_path_through_greenspace = path_start_to_greenspace + path_greenspace_to_end[1:]

        # Plot the full path
        fig, ax = ox.plot_graph_route(graph, full_path_through_greenspace, route_color='r', route_linewidth=6,
                                      node_size=0, show=False,
                                      close=False)

        # Get the bounding box of the full_path
        path_nodes = [graph.nodes[node] for node in full_path_through_greenspace]
        xs = [node['x'] for node in path_nodes]
        ys = [node['y'] for node in path_nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Apply a little padding to the bounding box for better visualization
        padding = 0.01  # this value might need adjustment based on your data
        ax.set_xlim([minx - padding, maxx + padding])
        ax.set_ylim([miny - padding, maxy + padding])

        # Plot the parks
        parks.plot(ax=ax, color='green', alpha=0.5)

        plt.show()

        # Compute the time for the direct route
        best_path_time = compute_route_length(graph, full_path_through_greenspace) / walking_speed  # seconds
        # Compute the time difference
        time_difference = best_path_time - direct_route_time  # seconds

        print(f"Time for direct route: {direct_route_time / 60:.2f} minutes")
        print(f"Time for route through parks: {best_path_time / 60:.2f} minutes")
        print(f"Time difference: {time_difference / 60:.2f} minutes")


else:  # For cases with more than 1 green spaces
    print(f"{num_greenspaces} urban green spaces found within the rectangle.")
    # From all the urban green spaces within the rectangle, select the largest one.
    largest_green_space = greenspaces_in_rectangle.loc[greenspaces_in_rectangle.geometry.area.idxmax()]
    # Find all nodes within the urban green space
    nodes_in_green_space = nodes[nodes.geometry.within(largest_green_space.geometry)]

    if not nodes_in_green_space.empty:
        # If there are nodes within the urban green space, plan a path through its centroid

        # Identify the centroid of the green space
        largest_centroid = largest_green_space.geometry.centroid

        # Find the nearest node to the centroid
        nearest_node_to_centroid = ox.nearest_nodes(graph, largest_centroid.x, largest_centroid.y)

        # Plan a path from the start node to the centroid node, then to the end node
        largest_path_to_centroid = ox.shortest_path(graph, start_node, nearest_node_to_centroid, weight='length')
        largest_path_from_centroid = ox.shortest_path(graph, nearest_node_to_centroid, end_node, weight='length')

        # Combine the two paths
        full_path_through_largest_centroid = largest_path_to_centroid + largest_path_from_centroid[1:]

        # Plot the full path
        fig, ax = ox.plot_graph_route(graph, full_path_through_largest_centroid, route_color='r', route_linewidth=6, node_size=0, show=False,
                                      close=False)

        # Get the bounding box of the full_path
        path_nodes = [graph.nodes[node] for node in full_path_through_largest_centroid]
        xs = [node['x'] for node in path_nodes]
        ys = [node['y'] for node in path_nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Apply a little padding to the bounding box for better visualization
        padding = 0.01  # this value might need adjustment based on your data
        ax.set_xlim([minx - padding, maxx + padding])
        ax.set_ylim([miny - padding, maxy + padding])

        # Plot the parks
        parks.plot(ax=ax, color='green', alpha=0.5)

        plt.show()

        # Compute the time for the direct route
        best_path_time = compute_route_length(graph, full_path_through_largest_centroid) / walking_speed  # seconds
        # Compute the time difference
        time_difference = best_path_time - direct_route_time  # seconds

        print(f"Time for direct route: {direct_route_time / 60:.2f} minutes")
        print(f"Time for route through parks: {best_path_time / 60:.2f} minutes")
        print(f"Time difference: {time_difference / 60:.2f} minutes")

    else:
        print("No nodes found within the urban green space.")

        # Find nearest nodes to the start point and the centroid of the green space
        largest_greenspace_centroid = largest_green_space.geometry.centroid
        largest_greenspace_node = ox.nearest_nodes(graph, largest_greenspace_centroid.x, largest_greenspace_centroid.y)

        # Compute the shortest path from start to green space
        path_start_to_largest_greenspace = ox.shortest_path(graph, start_node, largest_greenspace_node, weight='length')

        # Compute the shortest path from green space to end
        path_largest_greenspace_to_end = ox.shortest_path(graph, largest_greenspace_node, end_node, weight='length')

        # Combine the paths
        full_path_through_largest_greenspace = path_start_to_largest_greenspace + path_largest_greenspace_to_end[1:]

        # Plot the full path
        fig, ax = ox.plot_graph_route(graph, full_path_through_largest_greenspace, route_color='r', route_linewidth=6, node_size=0, show=False,
                                      close=False)

        # Get the bounding box of the full_path
        path_nodes = [graph.nodes[node] for node in full_path_through_largest_greenspace]
        xs = [node['x'] for node in path_nodes]
        ys = [node['y'] for node in path_nodes]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # Apply a little padding to the bounding box for better visualization
        padding = 0.01  # this value might need adjustment based on your data
        ax.set_xlim([minx - padding, maxx + padding])
        ax.set_ylim([miny - padding, maxy + padding])

        # Plot the parks
        parks.plot(ax=ax, color='green', alpha=0.5)

        plt.show()

        # Compute the time for the direct route
        best_path_time = compute_route_length(graph, full_path_through_largest_greenspace) / walking_speed  # seconds
        # Compute the time difference
        time_difference = best_path_time - direct_route_time  # seconds

        print(f"Time for direct route: {direct_route_time / 60:.2f} minutes")
        print(f"Time for route through parks: {best_path_time / 60:.2f} minutes")
        print(f"Time difference: {time_difference / 60:.2f} minutes")

