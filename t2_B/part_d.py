Bose-Einstein Condensate (BEC) - Quantum Partition Function

We have N indistinguishable bosons in a 2-level system with:
- Ground state energy = 0
- Excited state energy = epsilon

### Quantum Partition Function in Canonical Ensemble
Unlike the classical case, the quantum partition function considers the Bose-Einstein statistics, where multiple bosons can occupy the same energy state.

The quantum partition function is given by:

Z = sum from ne=0 to N of e^(-beta * ne * epsilon)

Using the formula for a geometric series, this simplifies to:

Z = (1 - e^(-beta * (N+1) * epsilon)) / (1 - e^(-beta * epsilon))

where:
- beta = 1 / (kB * T) is the inverse temperature
- epsilon is the energy of the excited state

### Interpretation
- This partition function accounts for the bosonic nature of the particles, ensuring that multiple bosons can occupy the same state.
- It differs from the classical case and leads to different thermodynamic properties such as energy distribution and condensation effects.

