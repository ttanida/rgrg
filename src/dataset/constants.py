ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right cardiophrenic angle": 7,
    "right hemidiaphragm": 8,
    "left lung": 9,
    "left upper lung zone": 10,
    "left mid lung zone": 11,
    "left lower lung zone": 12,
    "left hilar structures": 13,
    "left apical zone": 14,
    "left costophrenic angle": 15,
    "left hemidiaphragm": 16,
    "trachea": 17,
    "spine": 18,
    "right clavicle": 19,
    "left clavicle": 20,
    "aortic arch": 21,
    "mediastinum": 22,
    "upper mediastinum": 23,
    "svc": 24,
    "cardiac silhouette": 25,
    "left cardiac silhouette": 26,
    "right cardiac silhouette": 27,
    "cavoatrial junction": 28,
    "right atrium": 29,
    "descending aorta": 30,
    "carina": 31,
    "left upper abdomen": 32,
    "right upper abdomen": 33,
    "abdomen": 34,
    "left cardiophrenic angle": 35,
}


IMAGE_IDS_TO_IGNORE = {
    "0518c887-b80608ca-830de2d5-89acf0e2-bd3ec900",
    "03b2e67c-70631ff8-685825fb-6c989456-621ca64d",
    "786d69d0-08d16a2c-dd260165-682e66e9-acf7e942",
    "1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584",
    "f55a5fe2-395fc452-4e6b63d9-3341534a-ebb882d5",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179",
    "9c42d877-dfa63a03-a1f2eb8c-127c60c3-b20b7e01",
    "996fb121-fab58dd2-7521fd7e-f9f3133c-bc202556",
    "56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557",
    "93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6",
    "f57b4a53-5fecd631-2fe14e8a-f4780ee0-b8471007",
    "d496943d-153ec9a5-c6dfe4c0-4fb9e57f-675596eb",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50m",
    "422689b1-40e06ae8-d6151ff3-2780c186-6bd67271",
    "8385a8ad-ad5e02a8-8e1fa7f3-d822c648-2a41a205",
    "e180a7b6-684946d6-fe1782de-45ed1033-1a6f8a51",
    "f5f82c2f-e99a7a06-6ecc9991-072adb2f-497dae52",
    "6d54a492-7aade003-a238dc5c-019ccdd2-05661649",
    "2b5edbbf-116df0e3-d0fea755-fabd7b85-cbb19d84",
    "db9511e3-ee0359ab-489c3556-4a9b2277-c0bf0369",
    "87495016-a6efd89e-a3697ec7-89a81d53-627a2e13",
    "810a8e3b-2cf85e71-7ed0b3d3-531b6b68-24a5ca89",
    "a9f0620b-6e256cbd-a7f66357-2fe78c8a-49caac26",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
}

SUBSTRINGS_TO_REMOVE = "WET READ VERSION|WET READ|UPRIGHT PORTABLE AP CHEST RADIOGRAPH:|UPRIGHT AP VIEW OF THE CHEST:|UPRIGHT AP AND LATERAL VIEWS OF THE CHEST:|TECHNOLOGIST'S NOTE:|TECHNIQUE:|SUPINE PORTABLE RADIOGRAPH:|SUPINE PORTABLE CHEST RADIOGRAPHS:|SUPINE PORTABLE CHEST RADIOGRAPH:|SUPINE PORTABLE AP CHEST RADIOGRAPH:|SUPINE FRONTAL CHEST RADIOGRAPH:|SUPINE CHEST RADIOGRAPH:|SUPINE AP VIEW OF THE CHEST:|SINGLE SUPINE PORTABLE VIEW OF THE CHEST:|SINGLE SEMI-ERECT AP PORTABLE VIEW OF THE CHEST:|SINGLE PORTABLE UPRIGHT CHEST RADIOGRAPH:|SINGLE PORTABLE CHEST RADIOGRAPH:|SINGLE PORTABLE AP CHEST RADIOGRAPH:|SINGLE FRONTAL VIEW OF THE CHEST:|SINGLE FRONTAL PORTABLE VIEW OF THE CHEST:|SINGLE AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|SINGLE AP UPRIGHT CHEST RADIOGRAPH:|SINGLE AP PORTABLE CHEST RADIOGRAPH:|SEMIERECT PORTABLE RADIOGRAPH OF THE CHEST:|SEMIERECT AP VIEW OF THE CHEST:|SEMI-UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT PORTABLE CHEST RADIOGRAPH:|SEMI-UPRIGHT PORTABLE AP RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT AP VIEW OF THE CHEST:|SEMI-ERECT PORTABLE FRONTAL CHEST RADIOGRAPH:|SEMI-ERECT PORTABLE CHEST:|SEMI-ERECT PORTABLE CHEST RADIOGRAPH:|REPORT:|PORTABLES SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE UPRIGHT FRONTAL VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE ABDOMEN:|PORTABLE SUPINE FRONTAL VIEW OF THE CHEST:|PORTABLE SUPINE FRONTAL CHEST RADIOGRAPH:|PORTABLE SUPINE CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT RADIOGRAPH:|PORTABLE SEMI-UPRIGHT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT AP CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPHS:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT AP AND PA CHEST RADIOGRAPH:|PORTABLE FRONTAL VIEW OF THE CHEST:|PORTABLE FRONTAL CHEST RADIOGRAPH:|PORTABLE ERECT RADIOGRAPH:|PORTABLE CHEST RADIOGRAPH:|PORTABLE AP VIEW OF THE CHEST:|PORTABLE AP UPRIGHT CHEST RADIOGRAPH:|PORTABLE AP CHEST RADIOGRAPH:|PA AND LATERAL VIEWS OF THE CHEST:|PA AND LATERAL CHEST RADIOGRAPHS:|PA AND LATERAL CHEST RADIOGRAPH:|PA AND LAT CHEST RADIOGRAPH:|PA AND AP CHEST RADIOGRAPH:|NOTIFICATION:|IMPRESSON:|IMPRESSION: AP CHEST:|IMPRESSION: AP|IMPRESSION:|IMPRESSION AP|IMPRESSION|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPPER ABDOMINAL RADIOGRAPH, TWO IMAGES:|FRONTAL SUPINE PORTABLE CHEST:|FRONTAL SEMI-UPRIGHT PORTABLE CHEST:|FRONTAL RADIOGRAPH OF THE CHEST:|FRONTAL PORTABLE SUPINE CHEST:|FRONTAL PORTABLE CHEST:|FRONTAL PORTABLE CHEST RADIOGRAPH:|FRONTAL LATERAL VIEWS CHEST:|FRONTAL LATERAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPHS:|FRONTAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPH WITH THE PATIENT IN SUPINE AND UPRIGHT POSITIONS:|FRONTAL AND LATERAL VIEWS OF THE CHEST:|FRONTAL AND LATERAL FRONTAL CHEST RADIOGRAPH:|FRONTAL AND LATERAL CHEST RADIOGRAPHS:|FRONTAL AND LATERAL CHEST RADIOGRAPH:|FRONTAL|FINIDNGS:|FINDNGS:|FINDINGS:|FINDINGS/IMPRESSION:|FINDINGS AND IMPRESSION:|FINDINGS|FINDING:|FINAL REPORT FINDINGS:|FINAL REPORT EXAMINATION:|FINAL REPORT|FINAL ADDENDUM ADDENDUM:|FINAL ADDENDUM ADDENDUM|FINAL ADDENDUM \*\*\*\*\*\*\*\*\*\*ADDENDUM\*\*\*\*\*\*\*\*\*\*\*|FINAL ADDENDUM|CONCLUSION:|COMPARISONS:|COMPARISON:|COMPARISON.|CHEST:|CHEST/ABDOMEN RADIOGRAPHS:|CHEST, TWO VIEWS:|CHEST, SINGLE AP PORTABLE VIEW:|CHEST, PA AND LATERAL:|CHEST, AP:|CHEST, AP UPRIGHT:|CHEST, AP UPRIGHT AND LATERAL:|CHEST, AP SUPINE:|CHEST, AP SEMI-UPRIGHT:|CHEST, AP PORTABLE, UPRIGHT:|CHEST, AP AND LATERAL:|CHEST SUPINE:|CHEST RADIOGRAPH:|CHEST PA AND LATERAL RADIOGRAPH:|CHEST AP:|BEDSIDE UPRIGHT FRONTAL CHEST RADIOGRAPH:|AP:|AP,|AP VIEW OF THE CHEST:|AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|AP UPRIGHT CHEST RADIOGRAPH:|AP UPRIGHT AND LATERAL CHEST RADIOGRAPHS:|AP PORTABLE SUPINE CHEST RADIOGRAPH:|AP PORTABLE CHEST RADIOGRAPH:|AP FRONTAL CHEST RADIOGRAPH:|AP CHEST:|AP CHEST RADIOGRAPH:|AP AND LATERAL VIEWS OF THE CHEST:|AP AND LATERAL CHEST RADIOGRAPHS:|AP AND LATERAL CHEST RADIOGRAPH:|5. |4. |3. |2. |1. |#1 |#2 |#3 |#4 |#5 "