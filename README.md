# Inference 
using official model `./models/nestfuse_1e2.model`

# Evaluate using FusionToolBox

## FLIR RoadScene dataset[^1]
| EN | MI | SCD | SD | SF | SSIM | Qabf | VIF | 
|----|----|-----|----|----|------|------|-----|
6.78 | 1.78 | 1.39 | 31.03 | 9.07 | 0.93 | 0.34 | 0.48 |


## NIR Country dataset[^2]
| EN | MI | SCD | SD | SF | SSIM | Qabf | VIF | 
|----|----|-----|----|----|------|------|-----|
7.11 | 2.75 | 1.12 | 41.62 | 14.26 | 1.4 | 0.53 | 0.77 |


## TNO dataset[^3]
| EN | MI | SCD | SD | SF | SSIM | Qabf | VIF | 
|----|----|-----|----|----|------|------|-----|
6.82 | 1.84 | 1.56 | 41.77 | 13.4 | 0.79 | 0.34 | 0.68 |

[^1]:https://github.com/StaRainJ/road-scene-infrared-visible-images.git

[^2]:Brown, Matthew, and Sabine Süsstrunk. "Multi-spectral SIFT for scene category recognition." CVPR 2011. IEEE, 2011

[^3]:Toet, Alexander, and Maarten A. Hogervorst. "Progress in color night vision." Optical Engineering 51.1 (2012): 010901-010901.