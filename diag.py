import sys, os
log = open(os.path.join(os.path.dirname(__file__), "diag.log"), "w", buffering=1)
log.write("start\n"); log.flush()

log.write("importing numpy\n"); log.flush()
import numpy as np
log.write("numpy OK\n"); log.flush()

log.write("importing pandas\n"); log.flush()
import pandas as pd
log.write("pandas OK\n"); log.flush()

log.write("importing geopandas\n"); log.flush()
import geopandas as gpd
log.write("geopandas OK\n"); log.flush()

log.write("importing shapefile\n"); log.flush()
import shapefile
log.write("shapefile OK\n"); log.flush()

log.write("importing shapely\n"); log.flush()
from shapely.geometry import Point
from shapely.ops import nearest_points
log.write("shapely OK\n"); log.flush()

log.write("importing folium\n"); log.flush()
import folium
log.write("folium OK\n"); log.flush()

log.write("ALL IMPORTS OK\n"); log.flush()
log.close()
