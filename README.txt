Data Dictionary

shapes FIELDS
* shape_id
* points = count of vertices in polyline
* trip_id = unique trip_id linked to the shape_id
* headsign = end destination, identical with trip_headsign in GTFS feed
* direction = 0 or 1, not necessarily inbound/outbound
* route_id = route_id linked to the trip_id
* route_name = identical with route_short_name in GTFS feed
* route_long = identical with route_long_name in GTFS feed

stops FIELDS
* stop_id
* stop_name
* stop_lat
* stop_long
* location_t = identical to location_type in GTFS feed
* parent_sta = identical to parent_station in GTFS feed
* trip_count = count of unique trip_id's linked to the stop_id
* trip_ids = concatenated string of unique trip_id's linked to the stop_id, separated by " "
* route_ids = concatenated string of route_id's linked 1:1 to the trip_id's, separated by " ", some duplication
* route_nams = concatenated string of route_short_name's linked 1:1 to the route_id's, separated by " ", some duplication as with previous field

Digital Matatus Project
University of Nairobi & MIT Civic Data Design Lab
September 2019