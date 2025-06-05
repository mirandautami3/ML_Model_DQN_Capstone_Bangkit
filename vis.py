import folium
from test import routes_new
from main import locations, depot_location, paths, G
from matplotlib import cm
import matplotlib.colors as mcolors


# Visualization with folium
m = folium.Map(location=depot_location, zoom_start=13)
folium.Marker(depot_location, popup="Depot", icon=folium.Icon(color="red")).add_to(m)

# Add markers for customers
for idx, loc in enumerate(locations[1:], start=1):
    folium.Marker(loc, popup=f"Customer {idx}", icon=folium.Icon(color="blue")).add_to(m)

# Color palette for routes
num_routes = len(routes_new)
color_palette = cm.get_cmap('rainbow', num_routes)

for i, route in enumerate(routes_new):
    rgba_color = color_palette(i / num_routes)
    route_color = mcolors.to_hex(rgba_color)

    for j in range(len(route) - 1):
        path = paths.get((route[j], route[j + 1]), [])
        if not path:
            print(f"Warning: No valid path between node {route[j]} and {route[j + 1]}")
            continue

        edge_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
        folium.PolyLine(
            edge_coords,
            color=route_color,
            weight=5,
            opacity=1,
            tooltip=f"Route {i + 1}"
        ).add_to(m)

# Save the map
m.save("vrp_routes.html")
print("Routes visualized in 'vrp_routes.html'")