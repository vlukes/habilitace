# Dvouškálové numerické simulace - nelineární a slabě lineární problémy

## Požadavky

* [SfePy](https://sfepy.org)
* [PyVista](https://pyvista.org) - vizualizace výsledků, 3D
* [Matplotlib](https://matplotlib.org) - vizualizace výsledků, 2D

## Kapitola 2: Hyperelastické materiály s uvažováním velkých deformací

* Command line processing:

      python pucgen.py <input_file>

* Run the GUI:

      python pucgen.py

## Input file examples:

#### `example1.puc`:

```
BaseCell;size=(1, 1, 1);el_size=0.1;mat_id=5
SphericalInclusion;radius=0.3;central_point=(0, 0, 0);el_size=0.5;mat_id=2
CylindricalChannel;radius=0.1;central_point=(0, 0, 0);direction=x;el_size=0.5;mat_id=2
CylindricalChannel;radius=0.15;central_point=(0, 0, 0);direction=y;el_size=0.5;mat_id=2
CylindricalChannel;radius=0.2;central_point=(0, 0, 0);direction=z;el_size=0.5;mat_id=2
```
