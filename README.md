Ce package a été crée dans le but d'automatiser la gestion et la propagation des incertitudes dans le cadre de mesures physiques. 

Il implémente une classe Measure dont une instance est définie par une serie de mesure ainsi que les incertitudes associés. Les opérateurs élémentaires (addition, multiplication,...) ont été définies permettant des calculs explicites. 

Elle encapsule aussi la gestion des arrondi, qui ne sont effectués seulement à l'affichage et qui permet de ne pas surestimer les incertitudes dues aux arrondis dans les calculs intermédiaires. Une méthode Measure.errobar (basé sur la fonction errorbar de matplotlib) permet de gerer directement l'affichage des barres d'erreurs dans les graphes. 

Pour des propagations à travers des fonctions plus complexes, une méthode Measure.JAXpropagate reposant sur la librairie jax permet de calculer la propagation à travers une fonction func passé en argument à condition que la libairie jax soit installé (pip install jax) et que func soit définie à partir des fonctions élémentaires de jax.numpy (équivalentes à celle de numpy, import jax.numpy as jnp et remplacer les fonctions usuelles telle que np.sqrt par jnp.sqrt).


