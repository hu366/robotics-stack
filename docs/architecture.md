# Architecture

System layers:

1. Parse instruction into a `TaskSpec`
2. Ground semantic references into a geometric `WorldState`
3. Build a plan from reusable skills
4. Execute via control loops
5. Evaluate with reproducible metrics and traces
