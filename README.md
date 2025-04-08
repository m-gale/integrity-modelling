Continental-scale prediction of ecosystem integrity using a contemporary ‘best-on-offer’ reference approach.

1. reference_suitability.py
   Generate a hierarchy of inferred ecosystem integrity layers from protected area, native vegetation, and land use layers.

2. classify_gdm_kmeans.py
   Generate a non-anthropogenic 'ecotype' classification for Australia based on Mokany et al. (2022) GDM transformed compositional dissimilarity predictor layers.

3. compute_ref_sites_main.py and compute_ref_sites_functions.py
   Generate a set of best-on-offer reference sites by running through the integrity hierarchy for each ecostype, implementing a convex hull expansion to ensure that new sites are only added as required.

4. standardise_rasters.py
   Take predictor and response layers and resample to common extent, resolution, CRS, and water mask

5. sample_rasters.py
   Sample predictor and response valies for best-on-offer reference sites

6. predict_BRT.py
   Train GBRT models for each response using the sampled datatable. Predict from these models using the standardised rasters. Export predicted best-on-offer reference layers.

7. Use best-on-offer reference layers to calculate ecosystem integriy for each response, as a fcuntion of the departure between reference and contemporary layers.
