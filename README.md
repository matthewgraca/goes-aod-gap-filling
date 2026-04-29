# GOES AOD Gap Filling

Satellite retrievals of Aerosol Optical Depth (AOD) are widely used for monitoring smoke, dust, and air quality, but the resulting fields are spatially incomplete because of clouds, optically thick plumes, surface-related retrieval failures, and quality screening. These gaps bias downstream analysis toward scenes that are easiest for the satellite retrieval algorithm to observe.

This project develops a deep learning system that reconstructs missing GOES AOD over California using a UNet3+ architecture conditioned on lagged GOES observations together with meteorological, fire, smoke, vegetation, and intended static terrain predictors. The work follows the gap-filling approach recently demonstrated for NASA Deep Blue AOD over the contiguous United States by Lee et al.[^2], and adapts that idea to the higher spatial and temporal resolution of GOES AOD.

The repository is organized into two main components:

- `libs/`: ingestion and preprocessing code for satellite, meteorological, vegetation, AERONET, and static predictor data.
- `model/`: distributed PyTorch training code for the UNet3+ gap-filling model.

The ingestion and training layers are related, but they are not yet tied together into one complete end-to-end pipeline. Integration, evaluation, and validation against ground-based references are planned next steps.

![Aerosol Optical Depth](https://www.goes-r.gov/imagesContent/users/products/Aerosol-Optical-Depth.jpg)

## Problem Domain

### What Is AOD?

AOD measures how much aerosol is present in a vertical column of the atmosphere, inferred from how much sunlight is scattered or absorbed along that column.[^1] When AOD is high, the atmosphere contains more aerosol particles such as smoke, dust, or anthropogenic pollution.

AOD is not a direct measurement of pollution at ground level. It represents the total aerosol loading through the full atmospheric column, from the surface upward.

### Relationship With PM2.5

AOD is related to fine particulate matter, or PM2.5, but it is not equivalent to it. PM2.5 refers to the concentration of particulate matter smaller than 2.5 micrometers near the surface, where people breathe.

AOD often carries useful information about PM2.5 because both quantities respond to aerosol presence, but their relationship depends on several physical factors:

- Planetary boundary layer depth
- Ambient humidity
- Aerosol type and hygroscopicity
- Vertical layering of aerosols
- Meteorological transport

A high-AOD day can correspond to high PM2.5, but not always in a one-to-one manner. This is one reason meteorological predictors are central to the approach.

### Why Gap Filling Is Needed

Satellite AOD products are incomplete by nature. Retrievals can fail when clouds obscure the column, when smoke is optically thick enough to violate retrieval assumptions, when surface conditions are bright or heterogeneous, or when quality screening rejects an observation.

In the GOES pipeline used here, the data quality flags provided with the Level 2 product are an explicit part of preprocessing, so only higher-quality retrievals are retained. This improves the quality of the observed AOD pixels, but it also increases the number of missing pixels.

If these gaps are left untreated, downstream analysis becomes biased toward clearer scenes and more easily observed conditions. For applications such as wildfire smoke monitoring, exposure assessment, and PM2.5 modeling, this is a meaningful limitation. The objective of this model is to reconstruct a more complete AOD field from the information that remains available within the same scene and across nearby times.

### UNet and UNet3+

The model architecture used here is UNet3+, an extension of the original UNet architecture. A standard UNet is an encoder-decoder convolutional neural network with skip connections. The encoder progressively compresses the input and learns larger-scale spatial patterns, while the decoder upsamples the compressed representation back to the original resolution. Skip connections from the encoder to the decoder help preserve fine spatial detail that would otherwise be lost during downsampling.

UNet3+ extends this design with full-scale skip connections. Rather than connecting only matching encoder and decoder levels, each decoder stage receives information from every encoder stage and from coarser decoder stages. This matters for AOD gap filling because missing values do not depend only on neighboring pixels. Sometimes the model needs local texture, sometimes plume-scale structure, and sometimes broader meteorological context.

UNet3+ is designed to combine all of those scales more effectively than a simpler encoder-decoder model, and a closely related formulation has recently been used for CONUS-scale AOD gap filling.[^2]

## Model Overview

### Intended Use

The model is intended to reconstruct missing hourly GOES AOD fields over California. A gap-filled product of this kind can support smoke mapping during wildfire events, retrospective air quality studies, exposure analyses, and downstream PM2.5 estimation models that ingest AOD as a predictor.

### Data Sources

- Aerosol Optical Depth: [GOES Aerosol Optical Depth](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C01511/html)
- Meteorology: [High Resolution Rapid Refresh](https://rapidrefresh.noaa.gov/hrrr/)[^3]
- Vegetation: MODIS NDVI
- Intended orography source: [ASTER Global Digital Elevation Map](https://asterweb.jpl.nasa.gov/gdem.asp)
- Fire detection: [GOES Fire/Hot Spot Characterization](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01520)
- Smoke detection: [GOES Aerosol Detection](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01510)
- Ground-based evaluation data: AERONET observations

Orography and land mask are currently represented in the repository as placeholder static fields. They are part of the intended predictor set, but their ingestion modules still need to be implemented.

### Model Architecture

The model is implemented in PyTorch in `model/model.py`. It uses a five-level encoder with channel dimensions 64, 128, 256, 512, and 1024. Each encoder block applies repeated 3 x 3 convolutions followed by batch normalization and ReLU activations. The decoder follows the UNet3+ pattern, fusing features from multiple encoder and decoder resolutions before producing a single-channel output through a final 3 x 3 convolution.

The current training configuration expects 85 input channels. This count comes from stacking lagged continuous predictors together with one-hot encoded categorical variables. The continuous predictors include lagged AOD, fire radiative power, meteorological fields such as temperature, dewpoint, relative humidity, surface pressure, planetary boundary layer height, and wind components, plus intended static predictors such as terrain and land mask. The training configuration also includes categorical aerosol and retrieval-flag variables such as `Aerosol_Type_Land_Ocean`, `Algorithm_Flag_Ocean`, and `Algorithm_Flag_Land`.

### Why This Architecture Fits the Problem

AOD gap filling is not a tabular regression problem. It is a spatial reconstruction problem. Missing values occur in structured patterns rather than as isolated random points. Smoke plumes, cloud edges, and aerosol gradients have characteristic shapes and length scales, and meteorology adds additional structure through transport, mixing, moisture, and dispersion.

A fully convolutional, multi-scale architecture such as UNet3+ is therefore well suited to this task. It can exploit local texture, mid-scale plume structure, and broader meteorological context while still producing an output field at the original spatial resolution of the input patch. Compared to a baseline UNet, the full-scale skip connections of UNet3+ allow each decoder stage to draw on every level of the encoder, which matches the multi-scale nature of AOD gap filling.

## Prior Work

This project is primarily based on the work of Lee et al.[^2] We are tackling the same general problem, but using different data sources. The goals of this adaptation are:

1. Use an AOD product with higher spatial and temporal resolution.
2. Identify and use independent variables that can match this new resolution.

Due to the increased spatial resolution, the current target area is reduced from the entire continental United States to California.

## References

[^1]: https://aeronet.gsfc.nasa.gov/new_web/Documents/Aerosol_Optical_Depth.pdf
[^2]: Lee, J. S. M., Loría-Salazar, S. M., Holmes, H. A., & Sayer, A. M. (2025). Spatiotemporal gap-filling of NASA deep blue satellite aerosol optical depth over the contiguous United States (CONUS) using the UNet 3+ architecture. *Earth and Space Science, 12*, e2025EA004338. https://doi.org/10.1029/2025EA004338
[^3]: https://journals.ametsoc.org/configurable/content/journals$002fwefo$002f37$002f8$002fWAF-D-21-0151.1.xml
