#static folders/API
#LANG_MODEL = fasttext.load_model("lid.176.ftz")
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_SAVE_DIR = "data/overpass_fetch"
BASE_FOLDER = "images_test_overpass"
MODEL_STAGE1 = "runs/europe_bollard_basic/weights/best.pt"
MODEL_STAGE2 = "runs/europe_bollard_all/weights/best.pt"
MODEL_NUM_PLATES = "runs/detect/numplate_detector3/weights/best.pt"
SAVE_IMAGES = True
IMAGE_SIZE_STAGE = 640
CONF_THRESHOLD_STAGE1 = 0.5
CONF_THRESHOLD_STAGE2 = 0.75
#pan / zoom steps
pan_steps = [4]
zoom_scrolls = [0]
#saves the found detections for the current round
found_detections = []
#saves the found links
zoomed_links = []
zero_five_sleep_time_in_seconds = 0.5
zero_one_sleep_time_in_seconds = 0.1
one_sleep_time_in_seconds = 1
one_five_sleep_time_in_seconds = 1.5
two_sleep_time_in_seconds = 2
four_sleep_time_in_seconds = 4
settings = {
    "save_filename": "locations.json",
    "append_mode": True,
    "debug_mode": False,
}
current_save_file = None
#metas_to_fetch = {"croatia_bollard":5,"spain_bollard":5,"portugal_bollard":5,"polen_bollard":5,"czechia_bollard":5,"france_bollard":5,"slovenia_bollard":5,"austria_bollard":5,"germany_bollard":5,"luxenburg_bollard":5}
#metas_to_fetch = {"estonia_bollard":5,"latvia_bollard":5,"lithuania_bollard":5,"finland_bollard":5,"sweden_bollard":5,"norway_bollard":5,"iceland_bollard":5}
#metas_to_fetch = {"finland_bollard":5,"sweden_bollard":5,"norway_bollard":5,"iceland_bollard":5}
#since some metas create many false negatives (look nearly the same) (like austra/slovenia bollards) we use them also
meta_validate_results = {"austria_bollard":("austria_bollard","slovenia_bollard"),"germany_bollard":("germany_bollard","germany_bollard_reflector","luxenburg_bollard","spain_bollard"),"luxenburg_bollard":("luxenburg_bollard","luxenburg_bollard_reflector","germany_bollard","hungary_bollard"),"slovenia_bollard":("slovenia_bollard","slovenia_bollard_reflector","austria_bollard"),"france_bollard":("france_bollard","france_bollard"),"spain_bollard":("spain_bollard","germany_bollard"),"portugal_bollard":("germany_bollard","portugal_bollard","spain_bollard","france_bollard"),"italy_bollard":("italy_bollard","italy_bollard"),"polen_bollard":("polen_bollard","france_bollard"),"hungary_bollard":("hungary_bollard","polen_bollard","germany_bollard"),"czechia_bollard":("czechia_bollard","italy_bollard"),"croatia_bollard":("croatia_bollard","germany_bollard"),"lithuania_bollard":("lithuania_bollard","germany_bollard"),"latvia_bollard":("latvia_bollard","estonia_bollard"),"estonia_bollard":("estonia_bollard","latvia_bollard","latvia_bollard"),"finland_bollard":("finland_bollard","estonia_bollard"),
                         "sweden_bollard":("sweden_bollard","germany_bollard","estonia_bollard"),"norway_bollard":("norway_bollard","finland_bollard"),"iceland_bollard":("iceland_bollard","iceland_bollard")}
#TODO special bollard in spain (france_bollard - is a type in spain)
SAVE_RESUME_FETCHING_FILE = "resume_progress_save_state.json"
SAVE_STATE_FILE = "save_state_application.json"
metas_query_mapping = {
    "austria_bollard": None,
    "germany_bollard": None,
    "luxenburg_bollard": None,
    "slovenia_bollard": None,
    "france_bollard": None,
    "spain_bollard": None,
    "portugal_bollard": None,
    "italy_bollard": None,
    "polen_bollard": None,
    "hungary_bollard": None,
    "czechia_bollard": None,
    "croatia_bollard": None,
    "lithuania_bollard": None,
    "latvia_bollard": None,
    "estonia_bollard": None,
    "sweden_bollard": None,
    "finland_bollard": None,
    "norway_bollard": None,
    "iceland_bollard": None,
    "great_britain_num_plate": None,
    "us_district_of_columbia_num_plate": None,
    "us_new_york_num_plate":None
}

#overpass queries
exclude_residential_areas = """
(
  way(area.searchArea)["landuse"="residential"];
  relation(area.searchArea)["landuse"="residential"];
)->.residential;
"""

roads_query_highway = """
way(area.searchArea)["highway"~"trunk|primary"]->.roads;
(.roads; - .residential;)->.result;
.result out geom;
"""
roads_query_highway_intersections = """
// Select all highways (motorway, trunk, primary, secondary, tertiary, etc.)
way(area.searchArea)["highway"~"motorway|trunk|primary|secondary|tertiary"];
node(w) -> .highway_nodes;

// Select all roads (any kind of highway)
way(area.searchArea)["highway"];
node(w) -> .all_nodes;

// Intersections = nodes used both by "highways" and "all roads"
node.highway_nodes.all_nodes -> .intersection_nodes;

.intersection_nodes out geom;
"""

austria_bollard_query = f"""
[out:json][timeout:60];
area["ISO3166-1"="AT"][admin_level=2]->.searchArea;
{exclude_residential_areas}
{roads_query_highway}
"""
luxembourg_bollard_query = f"""
[out:json][timeout:60];
area["ISO3166-1"="LU"][admin_level=2]->.searchArea;
{exclude_residential_areas}
{roads_query_highway}
"""
slovenia_bollard_query = f"""
[out:json][timeout:60];
area["ISO3166-1"="SI"][admin_level=2]->.searchArea;
{exclude_residential_areas}
{roads_query_highway}
"""

exclude_residential_areas = """
(
  way(area.searchArea)["landuse"="residential"];
  relation(area.searchArea)["landuse"="residential"];
)->.residential;
"""
bundeslaender_de = [
    "DE-BW", "DE-BY", "DE-BE", "DE-BB", "DE-HB", "DE-HH",
    "DE-HE", "DE-MV", "DE-NI", "DE-NW", "DE-RP", "DE-SL",
    "DE-SN", "DE-ST", "DE-SH", "DE-TH"
]

germany_bollard_queries = {}
for bl_code in bundeslaender_de:
    germany_bollard_queries[bl_code] = f"""
    [out:json][timeout:120];
    area["ISO3166-2"="{bl_code}"][admin_level=4]->.searchArea;
    {exclude_residential_areas}
    {roads_query_highway}
    """

#TODO: Rework france
"""
regions_fr = [
    "FR-ARA", "FR-BFC", "FR-BRE", "FR-CVL", "FR-COR", "FR-GES",
    "FR-HDF", "FR-IDF", "FR-NOR", "FR-NAQ", "FR-OCC", "FR-PDL", "FR-PAC"
]
"""

regions_fr = [
    "FR-ARA", "FR-BFC"
]

france_bollard_queries = {}
for region_code in regions_fr:
    france_bollard_queries[region_code] = f"""
    [out:json][timeout:120];
    area["ISO3166-2"="{region_code}"][admin_level=4]->.searchArea;
    {exclude_residential_areas}
    {roads_query_highway_intersections}
    """

regions_es = [
    "ES-AN", "ES-AR", "ES-AS", "ES-CN", "ES-CB", "ES-CL",
    "ES-CM", "ES-CT", "ES-EX", "ES-GA", "ES-IB", "ES-RI",
    "ES-MD", "ES-MC", "ES-NC", "ES-PV", "ES-VC"
]

spain_bollard_queries = {}
for region_code in regions_es:
    spain_bollard_queries[region_code] = f"""
    [out:json][timeout:120];
    area["ISO3166-2"="{region_code}"][admin_level=4]->.searchArea;
    {exclude_residential_areas}
    {roads_query_highway}
    """


#all roads from portugal AXXX and NX - many bollards on these roads -region boxes AI
portugal_bollard_query ="""
[out:json][timeout:300];
(
  way["highway"]["ref"~"^A ?[0-9]+$"](36.8,-9.5,42.2,-6.2);
  way["highway"]["ref"~"^N ?[1-9]$"](36.8,-9.5,42.2,-6.2);
);

out geom;
"""
regions_it = [
    "Lombardia", "Toscana", "Piemonte", "Lazio", "Veneto", "Emilia-Romagna",
    "Sicilia", "Campania", "Liguria", "Friuli-Venezia Giulia", "Puglia",
    "Marche", "Abruzzo", "Sardegna", "Calabria", "Trentino-Alto Adige",
    "Umbria", "Basilicata", "Molise", "Valle d'Aosta"
]

italy_bollard_queries = {}
#roads with SSXXX - are mostly "Connection - roads " from bigger roads - more likely to have bollards imo
for region_code in regions_it:
    italy_bollard_queries[region_code] = f"""
    
[out:json][timeout:180];
area["name"="{region_code}"]["admin_level"=4]->.searchArea;
way(area.searchArea)["highway"]["ref"~"^SS[0-9]{{3}}"];
out geom;
    """

#use the bounding boxees beacuse region codes did not work (region boxes with AI)
region_bboxes = {
    "PL-DS": (50.0,14.9,51.8,17.9),
    "PL-KP": (52.8,17.5,54.0,19.5),
    "PL-LB": (53.5,22.0,54.0,24.0),
    "PL-LD": (49.0,18.5,51.0,20.5),
    "PL-LU": (50.8,22.5,51.5,24.0),
    "PL-MA": (49.5,19.5,50.5,21.5),
    "PL-MZ": (51.5,19.0,52.5,21.0),
    "PL-OP": (50.0,17.5,51.5,19.5),
    "PL-PK": (49.5,21.5,50.5,23.5),
    "PL-PD": (52.0,14.0,53.0,16.0),
    "PL-PM": (53.0,16.5,53.5,17.5),
    "PL-SL": (49.5,18.0,50.5,19.5),
    "PL-SK": (50.5,19.0,51.0,20.0),
    "PL-WN": (53.5,14.0,54.0,16.0),
    "PL-WP": (51.5,16.5,52.5,18.5),
    "PL-ZP": (51.5,14.5,52.5,16.0)
}

poland_bollard_queries = {}
for region_code, bbox in region_bboxes.items():
    south, west, north, east = bbox
    poland_bollard_queries[region_code] = f"""
    [out:json][timeout:120];
    way["ref"~"^[0-9]{{3}}$"]({south},{west},{north},{east});
    out geom;
    """
hungary_bollard_query = """[out:json][timeout:300];
area["ISO3166-1"="HU"]->.searchArea;
way(area.searchArea)["highway"]["ref"~"^[0-9]{1,3}$"];

out geom;"""

czechia_bollard_query = """[out:json][timeout:300];

// Define Czechia and Slovakia using ISO country codes
area["ISO3166-1"="CZ"]->.cz;
area["ISO3166-1"="SK"]->.sk;

(
  way(area.cz)["highway"="primary"];
  way(area.cz)["highway"]["ref"~"^[0-9]{2,3}$"];
  
  way(area.sk)["highway"="primary"];
);

out geom;"""
croatia_bollard_query = """[out:json][timeout:300];
area["ISO3166-1"="HR"]->.searchArea;
(
  way(area.searchArea)["highway"="primary"];
  way(area.searchArea)["highway"]["ref"~"^[0-9]{1,3}$"];
);

out geom;"""


lithuania_bollard_query = """[out:json][timeout:60];
area["ISO3166-1"="LT"][admin_level=2]->.searchArea;
way(area.searchArea)["highway"~"trunk|primary"]->.roads;
(.roads; - .residential;)->.result;
.result out geom;
"""
latvia_bollard_query = """[out:json][timeout:60];
area["ISO3166-1"="LV"][admin_level=2]->.searchArea;
way(area.searchArea)["highway"~"trunk|primary"]->.roads;
(.roads; - .residential;)->.result;
.result out geom;
"""
estonia_bollard_query = """
[out:json][timeout:60];
area["ISO3166-1"="EE"][admin_level=2]->.searchArea;
way(area.searchArea)["highway"~"trunk|primary|secondary"]->.roads;
(.roads; - .residential;)->.result;
.result out geom;"""
#just use the south of finland - in the north there aren't many bollards anyway -ONLY 1200 nodes though
finland_bollard_query = """[out:json][timeout:60];
area["ISO3166-1"="FI"][admin_level=2]->.searchArea;
way(area.searchArea)["ref"~"^(41|44|46|52|53|54|57|50|55)$"](59.0,19.0,62.0,32.0)->.roads;
(.roads; - .residential;)->.result;
.result out geom;"""
sweden_bollard_query = """[out:json][timeout:60];
area["ISO3166-1"="SE"][admin_level=2]->.searchArea;
way(area.searchArea)["ref"~"^[0-9]{2}[ ]?[A-Za-z]?$"](55.0,10.0,59.0,25.0)->.roads;
(.roads; - .residential;)->.result;
.result out geom;"""
iceland_bollard_query = """[out:json][timeout:120];
area["ISO3166-1"="IS"][admin_level=2]->.searchArea;
way(area.searchArea)["highway"~"^(motorway|trunk|primary|secondary)$"]->.roads;
(.roads; >;);
out geom;"""
#just get the E134 from norway - there are bollards - these are rare in norway
norway_bollard_query = """[out:json][timeout:300]; way["ref"="E 134"]["highway"]; (._;>;); out geom;"""

great_britain_num_plate_query = """
[out:json][timeout:300];
area["name"="London"]->.london;
area["name"="Edinburgh"]->.edinburgh;
area["name"="Glasgow"]->.glasgow;
(
  way["highway"](area.london);
  way["highway"](area.edinburgh);
  way["highway"](area.glasgow);
);
out geom;
"""
us_district_of_columbia_num_plate_query = """
[out:json][timeout:60];

// Polygon around Washington D.C.
(
  way["highway"](poly:"39.0069126 -77.1501160
                      38.8888950 -77.1920013
                      38.8076105 -77.0608521
                      38.8744631 -76.9145966
                      39.0031775 -76.9015503
                      39.0069126 -77.1501160");
);

// Output ways with full geometry
out geom;
"""

us_new_york_num_plate_query = """
[out:json][timeout:60];
area["name"="New York"]->.searchArea;
(
  way["highway"](area.searchArea);
);
out body;
>;
out skel qt;
"""
metas_query_mapping = {"austria_bollard":austria_bollard_query,"germany_bollard":germany_bollard_queries,"luxenburg_bollard":luxembourg_bollard_query,"slovenia_bollard":slovenia_bollard_query,"france_bollard":france_bollard_queries,"spain_bollard":spain_bollard_queries,"portugal_bollard":portugal_bollard_query,"italy_bollard":italy_bollard_queries,"polen_bollard":poland_bollard_queries,"hungary_bollard":hungary_bollard_query,"czechia_bollard":czechia_bollard_query,"croatia_bollard":croatia_bollard_query,"lithuania_bollard":lithuania_bollard_query,"latvia_bollard":latvia_bollard_query,"estonia_bollard":estonia_bollard_query,"sweden_bollard":sweden_bollard_query,"finland_bollard":finland_bollard_query,"norway_bollard":norway_bollard_query,"iceland_bollard":iceland_bollard_query,"great_britain_num_plate":great_britain_num_plate_query,
                       "us_new_york_num_plate":us_new_york_num_plate_query,"us_district_of_columbia_num_plate":us_district_of_columbia_num_plate_query}



