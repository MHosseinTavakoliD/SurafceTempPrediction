import requests
import json
import pandas as pd
#
# # Request a new access token by logging in
# request_URL = "https://www.viewmondousa.com/Token"
# credentials = {
#     "grant_type": "password",
#     "Username": "*****",
#     "Password": "***"
# }
#
# response = requests.post(request_URL, data=credentials)
# ######   Get all stations
# # if 'access_token' in response.text:
# #     access_token = response.json()["access_token"]  # Extract the new access token
# #     print("Access token:", access_token)
# #     # You can now use the new access token to make authorized requests to other endpoints.
# #     api_url = ("https://www.viewmondousa.com/api/v1/GetStations")
# #     headers = {
# #         "Authorization": f"Bearer {access_token}"
# #     }
# #
# #     response1 = requests.get(api_url, headers=headers)
# #     print('after Token', response1.text)
#
# ####  Get one station information
# if 'access_token' in response.text:
#     access_token = response.json()["access_token"]  # Extract the new access token
#     print("Access token:", access_token)
#     # You can now use the new access token to make authorized requests to other endpoints.
#     api_url = ("https://www.viewmondousa.com/api/v1/GetStationInfo?station_id=1598cf0e-c473-484e-9ca2-6cce8102e3f8")
#     headers = {
#         "Authorization": f"Bearer {access_token}"
#     }
#
#     response1 = requests.get(api_url, headers=headers)
#     print('after Token', response1.text)


#  So far
# {"StationId":"0a405208-b0e9-44ae-a2f0-ae91daac5962","StationName":"Mt. Pleasant"},
# {"StationId":"96ffc852-5d3b-4103-8e16-ccc37249e031","StationName":"Princeton"}, all surface temp is zero
# {"StationId":"1598cf0e-c473-484e-9ca2-6cce8102e3f8","StationName":"Trego"},

# "8c9541f8-8a52-4a2a-b2d6-ddf64b9bc764# Station 1,
# "238797a5-542c-45a6-a5f4-d734ab19484b" Station 2,
# "96ffc852-5d3b-4103-8e16-ccc37249e031" Station 3,
# "ddb05ff0-c3cc-475f-b8cc-6f9198bea53f" Station 4,
# "0a405208-b0e9-44ae-a2f0-ae91daac5962" Station 5
#{"StationId":"694f3214-64a6-4932-9bae-178cce013ded","StationName":"Greenfield"}, Just wind data
# {"StationId":"9b056292-2a47-4fff-a9e0-4c44a55283f8","StationName":"Cam Trego"}, no sensor channels configured for this station
# {"StationId":"69111e17-4d4f-4783-8544-6c766732f61d","StationName":"Wisconsin MARWIS 90817"}, No data for the whole year
# {"StationId":"69111e17-4d4f-4783-8544-6c766732f61d","StationName":"Wisconsin MARWIS 90817"}, No data

#{"StationId": "63badd17-bfbd-4b83-81e3-9e95298918b8", "StationName": "Johnson Creek"}, Freezed **
#   {"StationId": "5962bb18-17aa-48e4-8e41-3e18fc6c7334", "StationName": "Byron"} , Freezed  **
# {"StationId":"2c7beb18-4515-4b27-a940-b23ba14d088a","StationName":"Dickeyville"},Freezed  **
# {"StationId": "bbae3f19-5eef-4595-a583-271bd9cea23f", "StationName": "Newville"}, No surface temp
# {"StationId": "0db3ff1f-3afc-4dc3-a042-d546128fcb91", "StationName": "Arcadia"}, Looks goodYYYYYYYYYYYYYYYYYYYYYYYY
# {"StationId": "5bc44226-830a-44a0-9b5e-54014a067f58", "StationName": "Haugen"}, No surface temp
# {"StationId": "1e735129-bf68-49d7-98c7-1dea484b0c78", "StationName": "Plow truck mounted unit number is 10.320.1007.206"}, No sensor
# {"StationId":"fd7bc02a-fcb9-45e7-8aa4-b4d63a19932a","StationName":"Monico"}, No surface temp sensor
# {"StationId":"0c5adb2d-8dcc-4f0e-9f37-37ab5b68c2fe","StationName":"Tomahawk"}, No surface temp
# {"StationId":"32591632-bf79-42ca-aee2-efa45913127a","StationName":"Jackson"}, Freezed **
#  {"StationId":"5c8b5333-0181-4a49-9ff5-6348f469f0a4","StationName":"Neenah"}, freezed
#  {"StationId":"72ff8b35-8bdd-4d0d-b60d-5dd4273aaded","StationName":"Packwaukee"}, freezed
#  {"StationId":"ac7b7b38-35d3-4db2-b5af-fc3df5db1e97","StationName":"Stanley"}, freezed
# {"StationId":"5203573b-c9a3-450b-bb35-594bb8099d86","StationName":"Cudahy"}, freezed
# {"StationId":"38c28f3d-8d22-41f8-9840-7cad5b3970cd","StationName":"Manitowish"}, freezed
#   {"StationId":"ae3b923d-a5c2-4e9c-8b47-07fa48e52164","StationName":"Theresa"},, freezed
#   {"StationId":"6d98e23f-ab16-4ce7-8400-20ec39d72788","StationName":"St. Croix Falls"}, freezed
#    {"StationId":"e327fb41-c2a4-40c2-9909-bcc6cb1868bb","StationName":"Bruce"}, freezed
# {"StationId": "aaa78047-935b-4911-a3d8-a78961085bbe", "StationName": "Ft. McCoy"},freezed
#   {"StationId":"08aac14c-c858-4f2e-a867-6cf9ef46b8d4","StationName":"Superior (14 mi. SE)"}, freezed
#   {"StationId":"8c2c894d-aac2-4705-9dc8-74f390483b3a","StationName":"Mt. Sterling"}, freezed
#   {"StationId":"54fbfc52-2210-438c-940f-80232d7aafe4","StationName":"Knowlton"}, freezed
# {"StationId": "21fec254-7bbc-4439-b899-01cceaa66fd0", "StationName": "Tipler"},freezed
# {"StationId":"e001e369-00c8-47ac-ae05-b487c13de746","StationName":"Tipler Cam"},freezed
# {"StationId":"9948f254-6ce1-493c-b94b-5327efca2143","StationName":"Pewaukee"}, freezed but no surface temp sensor
#  {"StationId":"76448d55-490c-4f81-ad2c-1bb984d56019","StationName":"Appleton"}, freezed
# {"StationId":"1e00f055-087a-42bc-90dc-c795709071f6","StationName":"Kenosha"}, freezed
#    {"StationId":"11c93c57-cd1c-48db-ba32-f95e03229353","StationName":"Stevens Point (US 10 WI River Bridge"}, freezed
#  {"StationId":"2481b159-c715-4ec5-aeb4-253bcf96b5c1","StationName":"La Crosse"},freezed
# {"StationId":"de92195d-9b17-4060-a110-0c2848d9a669","StationName":"Woodruff"}, freezed but has sensors
# {"StationId":"9747095e-7255-4c98-8af1-a88932b67491","StationName":"Green Bay (Leo Frigo Bridge"}, Seems to be a good chance but freezes
# {"StationId": "14ff5964-0d85-4c80-b862-08162d454099", "StationName": "Mt. Horeb"},freezed
#   {"StationId":"476c1965-d256-4e2c-a0da-a21ff4c75c71","StationName":"Oshkosh"},No surface temp
#   {"StationId":"ee003867-a6d7-420e-880d-96adedaa5113","StationName":"Wausau"},No surface temp
#  {"StationId":"abc51468-dd91-40cc-896d-fc45026c8c2f","StationName":"Menomonee Falls"},No surface temp
#   {"StationId":"d30b0970-3a80-4e99-9a34-069f251df0b0","StationName":"Bellevue"}, no sensor
#   {"StationId":"5c8b5333-0181-4a49-9ff5-6348f469f0a4","StationName":"Neenah"}, Good one YYYYYYYYYYYYYYYYYYYYYYYY
# {"StationId": "72ff8b35-8bdd-4d0d-b60d-5dd4273aaded", "StationName": "Packwaukee"}, Responded as a web site
#  {"StationId":"ac7b7b38-35d3-4db2-b5af-fc3df5db1e97","StationName":"Stanley"}, Responded as a web site
# {"StationId":"5203573b-c9a3-450b-bb35-594bb8099d86","StationName":"Cudahy"},Responded as a web site
# {"StationId":"38c28f3d-8d22-41f8-9840-7cad5b3970cd","StationName":"Manitowish"},Good one YYYYYYYYYYYYYYYYYYYYYYYY
#   {"StationId":"ae3b923d-a5c2-4e9c-8b47-07fa48e52164","StationName":"Theresa"}, Responded as a web site
# {"StationId":"6d98e23f-ab16-4ce7-8400-20ec39d72788","StationName":"St. Croix Falls"},  No surface temp
#    {"StationId":"e327fb41-c2a4-40c2-9909-bcc6cb1868bb","StationName":"Bruce"}, Responded as a web site
#  {"StationId": "aaa78047-935b-4911-a3d8-a78961085bbe", "StationName": "Ft. McCoy"},Good one YYYYYYYYYYYYYYYYYYYYYYYY but no air temp!!!
# {"StationId": "08aac14c-c858-4f2e-a867-6cf9ef46b8d4", "StationName": "Superior (14 mi. SE)"}, No surface temp
#    {"StationId":"8c2c894d-aac2-4705-9dc8-74f390483b3a","StationName":"Mt. Sterling"},Good one YYYYYYYYYYYYYYYYYYYYYYYY but no air temp for some monthes!!!
#   {"StationId":"54fbfc52-2210-438c-940f-80232d7aafe4","StationName":"Knowlton"}, No surface temp
# {"StationId": "21fec254-7bbc-4439-b899-01cceaa66fd0", "StationName": "Tipler"}, Good one YYYYYYYYYYYYYYYYYYYYYYYY

#  {"StationId":"e001e369-00c8-47ac-ae05-b487c13de746","StationName":"Tipler Cam"}, no sensor
# {"StationId": "76448d55-490c-4f81-ad2c-1bb984d56019", "StationName": "Appleton"},Good one YYYYYYYYYYYYYYYYYYYYYYYY
# {"StationId":"1e00f055-087a-42bc-90dc-c795709071f6","StationName":"Kenosha"}, Good one YYYYYYYYYYYYYYYYYYYYYYYY for some monthes
#     {"StationId":"11c93c57-cd1c-48db-ba32-f95e03229353","StationName":"Stevens Point (US 10 WI River Bridge"}, Responded as a web site
#  {"StationId":"2481b159-c715-4ec5-aeb4-253bcf96b5c1","StationName":"La Crosse"}, no sensor
# {"StationId":"de92195d-9b17-4060-a110-0c2848d9a669","StationName":"Woodruff"} , no subsurface temp

Good_stations_for_2022 = {"Appleton", "Kenosha", "Tipler" , "Arcadia", "Neenah"}
Remaining_stations =[



]

# {"StationId":"0a405208-b0e9-44ae-a2f0-ae91daac5962","StationName":"Mt. Pleasant"},
# {"StationId":"96ffc852-5d3b-4103-8e16-ccc37249e031","StationName":"Princeton"},
# {"StationId":"1598cf0e-c473-484e-9ca2-6cce8102e3f8","StationName":"Trego"},
# {"StationId":"69111e17-4d4f-4783-8544-6c766732f61d","StationName":"Wisconsin MARWIS 90817"},
# {"StationId": "63badd17-bfbd-4b83-81e3-9e95298918b8", "StationName": "Johnson Creek"},
#   {"StationId": "5962bb18-17aa-48e4-8e41-3e18fc6c7334", "StationName": "Byron"} ,
# {"StationId":"2c7beb18-4515-4b27-a940-b23ba14d088a","StationName":"Dickeyville"},
# {"StationId": "0db3ff1f-3afc-4dc3-a042-d546128fcb91", "StationName": "Arcadia"},
# {"StationId":"32591632-bf79-42ca-aee2-efa45913127a","StationName":"Jackson"},
#  {"StationId":"5c8b5333-0181-4a49-9ff5-6348f469f0a4","StationName":"Neenah"},
#  {"StationId":"72ff8b35-8bdd-4d0d-b60d-5dd4273aaded","StationName":"Packwaukee"},
#  {"StationId":"ac7b7b38-35d3-4db2-b5af-fc3df5db1e97","StationName":"Stanley"},
# {"StationId":"5203573b-c9a3-450b-bb35-594bb8099d86","StationName":"Cudahy"},
# {"StationId":"38c28f3d-8d22-41f8-9840-7cad5b3970cd","StationName":"Manitowish"},
#   {"StationId":"ae3b923d-a5c2-4e9c-8b47-07fa48e52164","StationName":"Theresa"},
#    {"StationId":"e327fb41-c2a4-40c2-9909-bcc6cb1868bb","StationName":"Bruce"},
# {"StationId": "aaa78047-935b-4911-a3d8-a78961085bbe", "StationName": "Ft. McCoy"},
#   {"StationId":"8c2c894d-aac2-4705-9dc8-74f390483b3a","StationName":"Mt. Sterling"},
#  {"StationId":"76448d55-490c-4f81-ad2c-1bb984d56019","StationName":"Appleton"},
# {"StationId":"1e00f055-087a-42bc-90dc-c795709071f6","StationName":"Kenosha"},
#    {"StationId":"11c93c57-cd1c-48db-ba32-f95e03229353","StationName":"Stevens Point (US 10 WI River Bridge"},
# {"StationId":"9747095e-7255-4c98-8af1-a88932b67491","StationName":"Green Bay (Leo Frigo Bridge"},
# {"StationId": "14ff5964-0d85-4c80-b862-08162d454099", "StationName": "Mt. Horeb"},
#   {"StationId":"5c8b5333-0181-4a49-9ff5-6348f469f0a4","StationName":"Neenah"},
# {"StationId": "72ff8b35-8bdd-4d0d-b60d-5dd4273aaded", "StationName": "Packwaukee"},...
#  {"StationId":"ac7b7b38-35d3-4db2-b5af-fc3df5db1e97","StationName":"Stanley"},
# {"StationId":"5203573b-c9a3-450b-bb35-594bb8099d86","StationName":"Cudahy"},
# {"StationId":"38c28f3d-8d22-41f8-9840-7cad5b3970cd","StationName":"Manitowish"},
#   {"StationId":"ae3b923d-a5c2-4e9c-8b47-07fa48e52164","StationName":"Theresa"},
# {"StationId":"6d98e23f-ab16-4ce7-8400-20ec39d72788","StationName":"St. Croix Falls"},
#    {"StationId":"e327fb41-c2a4-40c2-9909-bcc6cb1868bb","StationName":"Bruce"},
#  {"StationId": "aaa78047-935b-4911-a3d8-a78961085bbe", "StationName": "Ft. McCoy"},
# {"StationId": "08aac14c-c858-4f2e-a867-6cf9ef46b8d4", "StationName": "Superior (14 mi. SE)"},
#    {"StationId":"8c2c894d-aac2-4705-9dc8-74f390483b3a","StationName":"Mt. Sterling"},
# {"StationId": "21fec254-7bbc-4439-b899-01cceaa66fd0", "StationName": "Tipler"},
#     {"StationId":"11c93c57-cd1c-48db-ba32-f95e03229353","StationName":"Stevens Point (US 10 WI River Bridge"},
# {"StationId":"0c5adb2d-8dcc-4f0e-9f37-37ab5b68c2fe","StationName":"Tomahawk"},...
# {"StationId": "bbae3f19-5eef-4595-a583-271bd9cea23f", "StationName": "Newville"},
# {"StationId":"de92195d-9b17-4060-a110-0c2848d9a669","StationName":"Woodruff"},
# {"StationId": "5bc44226-830a-44a0-9b5e-54014a067f58", "StationName": "Haugen"},
# {"StationId":"476c1965-d256-4e2c-a0da-a21ff4c75c71","StationName":"Oshkosh"},
# {"StationId":"ee003867-a6d7-420e-880d-96adedaa5113","StationName":"Wausau"},
# {"StationId":"abc51468-dd91-40cc-896d-fc45026c8c2f","StationName":"Menomonee Falls"},
# {"StationId":"54fbfc52-2210-438c-940f-80232d7aafe4","StationName":"Knowlton"},
