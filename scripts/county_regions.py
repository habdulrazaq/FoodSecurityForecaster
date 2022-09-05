import json
import io
import numpy as np
import requests
import zipfile

def load_county_regions(alpha3_code='SSD'):
    counties = {}
    json_file = f"{alpha3_code}_counties.json"
    try:
        with open(json_file, 'r') as f:
            data = json.loads(f.read())
    except FileNotFoundError:
        data = requests.get(f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{alpha3_code}_2.json.zip").content
        zip_file = zipfile.ZipFile(io.BytesIO(data))
        data = zip_file.read(zip_file.infolist()[0]).decode('utf8')
        with open(json_file, 'w+') as f:
            f.write(data)
        data = json.loads(data)
    for county in data['features']:
        name = county['properties']['NAME_2']
        polygon = np.array(county['geometry']['coordinates'])[0,0]
        counties[name] = polygon
    return counties
