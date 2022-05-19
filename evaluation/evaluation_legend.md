## Evaluated models

### Syntax

Models are annoted in the following format where notes where kept:

version \<number> = \<accuracy>(\<balanced accuracy>) || \<processing step> & \<processing step> & \<...>|| \<train extents>

Shorthand:
- calculated layers (4) refers to: calculating and adding NDVI, Moisture, NDWI, NDSI
- cluster (n = \<int>) refers to: running k-means clustering with n as cluster # and adding cluster labels as a feature
- edge (\<band name>, sig = \<int>) refers to: running sklearn canny edge detection on input band name tiff with a sigma of input
- geocoords refers to: retrieving the geocoordinates of each pixel in the scene and converting them to two input features `lat` and `long`.

### Version list

`evaluation` main folder

&rarr; `test_data`: evaluation run on test data, ie. data split from the extents prior to training but still all pixels from those train extents and with equal class distribution

- `toronto`, `simcoe`, `sask`, `labrador` = || 5K/class || toronto, simcoe, sask, labrador
- `base_run` = || 5K/class || sim and lab 
- `scaled` = ||  5K/class, StandardScaler() || sask, tor
    - improves SVC no effect on RF
- `large_set` = 18K/class || sim and lab
- `gaussian` = bands transformed via gaussian filter (sigma 1), 18K/class || sim and lab

&rarr; `demo`: evaluation run on demo site data (extent provided by Fraser for the working group) ie. entire extent completely unknown to model and with unequal class distribution. Divided into subfolders based on model used. Below are processing details for each version. *RFC hypers:* unless otherwise stated used 300 trees, max depth 25 and max features 2. This changes for *version 10 and later* where max depth 15 is used instead; Detailed map (15 color values) starts with version 10.
- version_1 = 31.41% || 18K each || sim, lab 
- version_2 =  31.55% || gaussian (sigma 5) bands, 18K/class || sim, lab
- version_3 = 33.22% || gaussian (sigma 1) bands, 18K/class || sim, lab
- version_4 = 37.49% || gaussian (sigma 1), 40K/class || sim, lab, tor and james
- version_5 = 34.21% || 40K/class|| sim, lab, tor, james  [?? missing something can't recreate map]
- version_6 = 33.97% (34.84%) || raws bands and gaussian (sig 1) bands 40K/class || sim, lab, tor, james 
- version_7 = 33.37% (b:33.95%) || raws & median (size 10) transformed bands 40K/class || sim, lab, tor, james
- version_8 = 42.11% (b:37.22%) || raws & calculated layers (4) || sim, lab, tor, james 
- version_9 = 36.50 (b: 35.80%) || raws & calculated layres (4) and gaussian (sig 1) bands, 40K/class || sim, lab, tor, james
- version_10 = 50.87 (b: 40.89%) || raws & calculated layers (4) & B01f with B01 where outliers < q3x2 = median & 80K/class, 47K snow class || sim, lab, tor, james, sjames all bands
- version 11 = 46.11 (b: 39.43) || raws with calculated layers & B01f with B01 where outliers < q3x2 = median & 60K/class no snow || sim, lab, tor, james, sjames, calgary
- version 12 = 48.10 (b: 39.83) || raws & calculated layers (4) & cluster (n = 4) & 100K/class 49K snow class || sim, lab, tor, james, sjames, cal, trois
- version 13 = 48.11 (40.06) || raws & calculated layers (4) & edge ('B03', sig = 1) & gaussian (sig 5) & clustering & 150K/class 49K snow class || sim, lab, tor, james, sjames, cal
- version 14 = 51.74 (42.00) || raws & calc layers (4) & edge ('B8A, sig = 3) & 100K/class 47K snow class || sim , lab, tor, james, sjames
- version 15 = 53.86 (38.94) || raws & calc layers (4) & geocoords & edge ('B8A, sig = 3) & 150K/class 49K snow || sim, lab, tor, james, sjames, cal, trois, winn
- version 16 = 48.84 (39.60) || raws & calc layers (4) & geocoords & edge ('B8A', sig = 3) & gaussian filter (sig 3) & 250K/class 49K snow 150K class 11 || sim, lab, tor, james, sjames, cal, trois, winn, 

## Other models

Binary versions
- bin14 = binary class 14, trained on is or is not (1, 0) class 14 (wetland) using james, sjames raws with calculated layers (10) and 300K of is and is not; raw bands + calc layers (extra)
    - forest: using 300 trees, max depth 15 and max features 2, threshold 50
    - xgb: no hyperparam set, threshold 20

Combination models
- 10_bin14_13: base of version 10 with add on forest_bin14 for class 14 and taking 15 and 17 from version 13 in that order
    - demo accuracy 52.40 (b: 41.22%)
- 10_xgbin14_13: base of version 10 with add on xgb_bin14 for class 14 and taking 15 and 17 from version 13 in that order
    - demo accuracy 52.45 (b: 41.35%)
- 14_13: base of version 14 taking 15 from version 13 in that order
    - demo accuracy 58.05 (b: 43.66)
- 14_xgbin14_13: base of version 14 with add on xg_bin14 for class 14 and taking 15 from version 13 in that order
    - demo accuracy 55.92 (b: 43.04)
    - trois accuracy 57.00 (b: 41.27)
    - calgary accuracy 42.49 (b: 25.05)
- 15_xgbin14_13: base of version 15 with add on xg_bin14 for class 14 & taking 15 from v. 13 in that order
    - demo: 55.60 (38.28)
    - cal: 53.52 (31.75)
- 15_xgbin14_16: base v. 15 w/ xg_bin14 for class 14 and v. 13 for class 15
    - demo: 56.82 (39.71)
    - cal: 32.18 (55.74)
    
    