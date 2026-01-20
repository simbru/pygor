## Wrote pygor implementation of STRF calculation



Below is comparison of STRF calculated in pygor vs in IGOR Pro using the same data, parameters, and same terminal (hand picked). The results are very similar, with minor differences likely due to float precision and slightly different pixel inclusions in the ROI.

Correlation projection in IGOR via OS_scripts pipeline:

<img src="..\docs\modules\images\igor_calculated_strf.png" width="220">

Python plotting to get as close to equivalent as I can

<img src="..\docs\modules\images\pygor_calculated_strf.png" width="250">

``` python 
img = pygor.strf.spatial.corr_spacetime(obj.strfs[0])
# Vectorized polarity and SD calculation
max_locs = np.argmax(obj.strfs[0], axis=0)
min_locs = np.argmin(obj.strfs[0], axis=0)
# Polarity: -1 where max comes before min
pols = np.where(max_locs < min_locs, -1, 1)
plt.imshow(img*pols, cmap = "bwr", clim = (-10, 10))
plt.colorbar()
```

To further check STRF scripts, I did the following:
- Preprocess the data in pygor (motion correction, artifact removal, etc) and segment ROIs
- Export the preprocessed data to H5 
- Load the preprocessed data in IGOR Pro
- Calculate STRFs in IGOR Pro using the same parameters as in pygor
- Calculate STRFs in pygor
- Compare the STRFs from IGOR Pro and pygor

And when both plotted with the same code in Python after one is processed in IGOR and the other in pygor, they look virtually identical:

![igor_vs_pygor](..\docs\modules\images\strf_reimport_compare.png)

```
Statistics comparison (ROI 10 as example):
Pygor - min: -37.282, max: 7.561, std: 2.173
IGOR  - min: -37.385, max: 7.612, std: 2.178
Scale ratio (IGOR/pygor std): 1.002
```

```
All ROIs correlation: 0.9992 ± 0.0007
Min: 0.9965, Max: 0.9998
```

I then tested for multicolour STRFs, where each colour channel is treated separately. Again, very similar results:

```
Result:
Number of ROIs: 63, 4 colour channels (252 RFs)
obj.strfs shape: (252, 20, 24, 40)
obj_igor.strfs shape: (252, 20, 24, 40)

obj.strfs[3] shape: (20, 24, 40)
obj_igor.strfs[3] shape: (20, 24, 40)
ROI 3 STRF correlation: 1.0000

All ROIs correlation: 0.9993 ± 0.0015
Min: 0.9926, Max: 1.0000
```

Here are the images

![pygor_strf](..\docs\modules\images\pygor_multicolour_strf.png)

![igor_strf](..\docs\modules\images\igor_multicolour_strf.png)

Differences are likely due to float precision. Overall, the STRF implementation in pygor appears to be validated against the IGOR Pro implementation.