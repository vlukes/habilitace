# Dvouškálové numerické simulace - nelineární a slabě lineární problémy

## Požadavky

* [SfePy](https://sfepy.org)
* [PyVista](https://pyvista.org) - vizualizace výsledků, 3D
* [Matplotlib](https://matplotlib.org) - vizualizace výsledků, 2D

## Kapitola 2: Hyperelastické materiály s uvažováním velkých deformací

### Kapitola 2.4.1

* Přímá numerická simulace (DNS):

      sfepy-run dns_hyper.py -d "output_dir='output_dns'"

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

## Kapitola 3: Hyperelastické porézních struktury nasycené kapalinou

### Kapitola 3.5

* DNS výpočet:

      sfepy-run dns_hyper_perf.py -d "output_dir='output_dns_perf'"

* FE2 výpočet:

      sfepy-run fe2_perf_makro_dns.py -d "output_dir='output_fe2_perf'"

* Generování grafů:

      python plot_kap35.py

### Kapitola 3.6

* FE2 výpočet:

      sfepy-run fe2_perf_makro.py -d "output_dir='output_perf'"

* Generování grafů a obrázků:

      python plot_kap36.py

## Kapitola 4: Výpočetní homogenizace pro slabě nelineární úlohy

### Kapitola 4.1.4

* Homogenizace - lineární model:

      sfepy-run poroela_makro.py -d "output_dir='output_poroela', nonlinear=False"

* Homogenizace - nelineární model:

      sfepy-run poroela_makro.py -d "output_dir='output_poroela', nonlinear=True"

* Generování grafů:

      python plot_kap414.py

### Kapitola 4.2.3

* Homogenizace - výpočet:

      sfepy-run poropiezo_makro.py -d "N=20,output_dir='output_poropiezo',eps0=0.005,phi_ampl=[4e5,0],micro_recovery=True"

* DNS výpočet:

      sfepy-run dns_poropiezo.py

* Generování grafů:

      python plot_kap423.py

