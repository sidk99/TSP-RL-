# Traveling Salesman Problem-RL-
Public access files for subtasks

# Christofides TSP Implementation
The Christofides algorithm is an algorithm for finding approximate solutions to the travelling salesman problem, on instances where the distances form a metric space (symmetric and obey the triangle inequality). It is an approximation algorithm that guarantees that its solutions will be within 1.5 times the optimal solution length.

The Algorithm is the following:
- Find a minimum weight spanning tree T 
- Let W be the set of vertices having an odd degree in T
- Find a minimum weight matching M of nodes from W
- Merge of T and M forms a multigraph (V (Kn), E(T) ∪ M)
in which we find the Eulerian walk L
- Transform the Eulerian walk L into the Hamiltonian circuit H
in the complete graph
