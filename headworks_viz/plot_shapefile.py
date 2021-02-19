import geopandas as gpd
import matplotlib.pyplot as plt

infile = r"C:\Users\u77932\Documents\github\borehole_gfx_utils\headworks_viz\AUS_2016_AUST.shp"

gdf = gpd.read_file(infile)

ax = gdf.plot()

plt.show()
