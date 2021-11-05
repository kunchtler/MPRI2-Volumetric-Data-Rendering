---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python slideshow={"slide_type": "skip"}
import sys
import os
import warnings

os.chdir("../ProjVizGit")
warnings.filterwarnings('ignore')
```

<!-- #region slideshow={"slide_type": "slide"} -->
# VOLUMETRIC RAY CASTING
 
Léo Kulinski, Emma Caizergues

<center>
    <img src="Images\volume rendering 2.png" width="450"/>
</center>

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Goal : implement the ray casting technique from scratch
- In python.

- with numpy.

- and matplotlib / tkinter.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Implementation
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### What we're working with :

 - 3D scalar values in a regular grid, parsed from online volumetric data sets.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## 3D scene

<center>
    <img src="Images\boundingbox3D.png" width="600"/>
</center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## 4 STEPS :


<center>
    <img src="Images\volume_ray_casting.png" width="1300"/>
</center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## STEP 1 and 2 : Emitting a ray and sampling points

 - Given a ray, compute the entry and exit point in the bounding box.

 - We only sample in the bounding box.

<center>
    <img src="Images\boundingbox2d.png" width="700"/>
</center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## STEP 3 : Obtaining sample values by trilinear interpolation

<center>
    <img src="Images\interpolation_tri2.png" width="500"/>
</center>

- For every sample of the ray : we obtain a scalar value.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## STEP 3 bis : Computing the colour from the value

- Translate the value of the sample to a colour using a user-defined function.

- Apply shading using those specific vectors :
<center>
    <img src="Images\ombre_vecteurs.png" width="500"/>
</center>

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Computing the normal
<center>
    <img src="Images\computing_normal.png" width="900"/>
</center>
$g_x = \frac{f(x-1,y,z) - f(x+1,y,z)}{2}$ $ \quad g_y = \frac{f(x,y-1,z) - f(x,y+1,z)}{2}$
 $ \quad g_x = \frac{f(x,y,z-1) - f(x+1,y,z+1)}{2}$
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
## STEP 4 : Combining colours

- Apply a recursive formula from back to front.
<center>
    <img src="Images\shading_recursive_formula.png" width="800"/>
</center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Demo and results 
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Foot Dataset

<center>
    <img src="Images\FootIso.png" width="600"/>
</center>
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
from parse_data import *

file_name = r"data/Foot.vol"
_, data = parse_vol_file(file_name)
show_data_slice(data[:,:,60].T)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
<center>
    <img src="Images\couleuros.png" width="800"/>
</center>
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
%matplotlib inline
from colour_fun import colour_Foot
import camera

camera.colour_function = lambda x : colour_Foot(x, bone_colour=(1.0, 0.5, 0.0))
image = camera.tk_compute_image(no_shadows=False)
camera.show_image(image)
```

<!-- #region slideshow={"slide_type": "subslide"} -->
 ### The buckyball

Bucky Ball dataset        | Result
:-------------------------:|:-------------------------:
<img width="500" src="Images\Buckyball.png"/>  |  <img width="500" src="Images\c60_hd2.png"/>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "subslide"} -->
### Shininess impact

Shininess = 15      | Shininess = 30
:-------------------------:|:-------------------------:
<img width="500" src="Images\sh15.png"/>  |  <img width="500" src="Images\sh30.png"/>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
# Conclusion

 - Play more with lighting effects and parameters.
 - Possible optimization : Normal of interpolations $\rightarrow$ Interpolation of normals ?
 - Make it in a more efficient programming language.
<!-- #endregion -->
