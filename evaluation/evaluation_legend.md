## Evaluated models

`evaluation` main folder

&rarr; `test_data`: evaluation run on test data, ie. data split from the extents prior to training but still all pixels from those train extents and with equal class distribution

- `toronto`, `simcoe`, `sask`, `labrador` = standard bands, no parameters, 5K each value, no scaling, in turn toronto, simcoe, sask, labrador extents only
- `base_run` = standard bands, no parameters, 5K each value, no scaling, sim and lab
- `scaled` = standard bands, no params, 5K each value, StandardScaler() -> improved SVC, sask and tor
- `large_set` = standard bands, no parameters, 18K each value, no scaling, sim and lab
- `gaussian` = standard bands transformed via gaussian filter, 18K each value, no scaling, sim and lab

&rarr; `demo`: evaluation run on demo site data (extent provided by Fraser for the working group) ie. entire extent completely unknown to model and with unequal class distribution. Divided into subfolders based on model used. Below are processing details for each version. *RFC hypers:* unless otherwise stated used 300 trees, max depth 25 and max features 2. This changes for version 11 and later where max depth 15 is used instead; Detailed map (15 color values) starts with version 10.
- version_1 = 31.41% trained on sim & lab 18K each
- version_2 =  31.55% trained on gaussian (sigma 5) sim & lab, 18K each
- version_3 = 33.22% trained on gaussian (sigma 1) sim & lab, 18K each
- version_4 = 37.49% trained on gaussian (sigma 1) sim, lab, tor and james, 40K each random
- version_5 = 34.21% trained on sim, lab, tor, james all bands, 40K each random [?? missing something can't recreate map]
- version_6 = 33.97% (b: 34.84%) trained on raw and gaussian (sig 1) sim, lab, tor, james all bands, 40k each random
- version_7 = 33.37% (b:33.95%) raw and median (size 10) sim, lab, tor, james all bands, 40k each random
- version_8 = 42.11% (b:37.22%) raw with calculated layers sim, lab, tor, james all bands, 40k each
- version_9 = 36.50 (b: 35.80%) raw with calculated layres and gauss (sig 1) sim, lab, tor, james all bands 40k each
- version_10 = 50.87 (b: 40.89%) raw with calculated layers and column B01f with B01 where outliers < q3x2 = median sim, lab, tor, james, sjames all bands, 80k each 47K snow ; 15 max depth
- version 11 = 46.11 (b: 39.43) raw with calculated layers and column B01f with B01 where outliers < q3x2 = median sim, lab, tor, james, sjames, calgary all bands, 60k each no snow
- version 12 = 48.10 (b: 39.83) raw with calculated layers and cluster (n = 4) column; sim, lab, tor, james, sjames, cal, trois 100K classes with 49K snow class
- version 13 = 48.11 (40.06) raw with calculated layers (4) and edge ('B03', sig = 1); sim, lab, tor, james, sjames, cal 150K with 49K snow class
- version 14 = 54.39 (40.91) raw with calc layers (4), outlier B01f same as v. 10, and edge ('B8A, sig = 3); sim , lab, tor, james, sjames 100K with 47K snow
- version 15 = 53.86 (38.94) raw with calc layers(4), geocoords, and edge ('B8A, sig = 3); sim, lab, tor, james, sjames, cal, trois, winn, 150K with 49K snow

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
- 14_xgbin14_13: base of version 14 with add on xg_bin14 for class 14 and taking 15 from version 13 in that order
    - demo accuracy 55.92 (b: 43.04)
    - trois accuracy 57.00 (b: 41.27)
    - calgary accuracy 42.49 (b: 25.05)
- 15_xgbin14_13: base of version 15 with add on xg_bin14 for class 14 & taking 17 from v. 13 in that order
    - demo: 55.60 (38.28)
    - cal: 53.52 (31.75)
    
    