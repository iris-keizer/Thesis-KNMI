"""
File containing the Python functions to create the timmerman regions

Author: Iris Keizer
https://github.com/iris-keizer/Thesis-KNMI

These functions are used in the notebooks or files:
- Load_prep_data.py

"""


# Import necessary packages
import numpy as np
import regionmask





# Create Timmerman regions
#-------------------------
def timmerman_regions():
    

    # As first coordinates take most South-West point and than go anti-clockwise
    Channel = np.array([[-5.1, 48.6], [1.5, 50.1], [1.5, 50.9], [-5.1, 49.9]])
    South = np.array([[0.5, 50.8], [3.2, 51.3], [5.3, 53.1], [1.7, 52.3]])
    Mid_West = np.array([[1.7, 52.3], [5.3, 53.1], [3.7, 55.7], [-1.3, 55.1], [0.5, 53.1], [1.8, 52.7]])
    Mid_East = np.array([[5.3, 53.1], [8.9, 53.9], [7.8, 57.0], [3.7, 55.7]])
    North_West = np.array([[-1.3, 55.1], [3.7, 55.7], [1.1, 59.3], [-3.0, 58.7], [-1.7, 57.5], [-1.5, 55.5]])
    North_East = np.array([[3.7, 55.7], [7.8, 57.0], [7.4, 58.0], [6.1, 58.6], [4.9, 60.3], [1.1, 59.3]])

    region_names = ["Channel", "South", "Mid-West", "Mid-East", "North-West", "North-East"]
    region_abbrevs = ["C", "S", "MW", "ME", "NW", "NE"]
    region_numbers = [1,2,3,4,5,6]
    Timmerman_regions = regionmask.Regions([Channel, South, Mid_West, Mid_East, North_West, North_East], numbers = region_numbers, 
                                           names=region_names, abbrevs=region_abbrevs, name="Timmerman")
    
    return Timmerman_regions
    