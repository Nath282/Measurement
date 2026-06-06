Cette librairie a été crée dans le but d'automatiser la gestion et la propagation des incertitudes liées aux mesures physiques
Elle repose en son coeur sur numpy pour la vectorisation et les performances du C 
Elle définie une classe Measure dont une instance est définie par une série de mesure (sous la forme d'un array numpy) auquel est associé une série d'incertitudes
Les opérateurs élémentaires ont été définie permettant une utilisation intuitive dans le cas d'opérations élémentaires
Pour des propagations à travers des fonctions complexes, la méthode propagate utilise JAX pour propager à travers une fonction quelconque crée à partir de jax.numpy
