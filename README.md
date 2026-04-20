# Dvouškálové numerické simulace - nelineární a slabě lineární problémy

## Požadavky

* [SfePy](https://sfepy.org)
* [PyVista](https://pyvista.org) - vizualizace výsledků, 3D
* [Matplotlib](https://matplotlib.org) - vizualizace výsledků, 2D

## Kapitola 2: Hyperelastické materiály s uvažováním velkých deformací

### Kapitola 2.4.1

* Přímá numerická simulace:

      sfepy-run dns_hyper.py -d "output_dir='output_dns2'"

* FE2 výpočet:

      sfepy-run fe2_makro_dns -d "output_dir='output_fe2'"

* Generování grafů a obrázků:

      python plot_kap241.py

### Kapitola 2.4.2

* FE2 výpočet:
      sfepy-run fe2_makro.py -d "output_dir='output_fe2_a',recovery_idxs=[(0,0),(22,0),(27,0)],save_qp=True"

* Generování grafů a obrázků:

      python plot_kap242.py

### Kapitola 2.4.4

* FE2 výpočet (3D):

      sfepy-run fe2_makro_3d.py

### Kapitola 2.5.1

* POD výpočet:

      sfepy-run pod_makro.py -d "output_dir='output_pod'"

### Kapitola 2.6.4

* CSA výpočet:

      sfepy-run csa_makro.py -d "delta=0.005,output_dir='output_fe2_b',save_qp=True"

* Generování grafů a obrázků:

      python plot_kap264.py

### Kapitola 2.6.8

* CSA výpočet (3D):

      sfepy-run csa_makro_3D.py
