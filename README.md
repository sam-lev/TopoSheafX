# TopoSheafX

Democratizing sheaf theory

## List of desired features

### Sheaves

* Data stucture for sheaves of Euclidean over vector spaces, cell complexes, etc.
  * support stalks of different dimensions
  
  * support restriction maps from different classes
    * orthogonal matrices
    * projection maps
    * diagonal matrices
    * positive definition matrices
    * arbitrary matrices
* Sheaf cohomology
  * boundary opeator: $$\delta: C^k(X; \mathcal{F}) \to C^{k+1}(X; \mathcal{F})$$
  * Sheaf Laplacian: $$L_k = \delta^\ast \delta + \delta \delta^\ast$$
  * Sheaf diffusion: $$\dot{x} = - L_k x$$
  * Nonlinear sheaf Laplacian: $$L_k = \delta^\ast \nabla U_{k+1} \delta + \delta \nabla U_{k}\delta^\ast$$
* Homological programs $$\underset{x \in C^k(X;\mathcal{F})}{\text{minimize}} \quad \sum_{\sigma \in X_k} f_\sigma(x_\sigma) \\ \text{subject to } \quad L_k x = 0$$
* Ideally (although difficult) foster eventual support for stalks from 
  * different data categories
    * groups / matrix groups / Lie groups
    * suplattices / preorders
    * metric spaces
    * Hilbert spaces
  * analgous message passing and sheaf Laplacians  

### Sheaf neural networks

## Meetings

 * July 14, 2026
 * [July 16, 2026](meetings/16_07_2026.md)

## References