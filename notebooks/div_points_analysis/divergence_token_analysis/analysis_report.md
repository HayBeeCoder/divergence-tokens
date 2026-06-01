# Divergence Token Evolution Summary

- Experiment directory: /home/abasso_aims_ac_za/divergence-tokens/workspace/multihop
- Model: gemma
- Target preference: raven
- Seeds analyzed: [42]
- Hops analyzed: ['hop0', 'hop1', 'hop2', 'hop3', 'hop4', 'hop5', 'hop6']

## Hop-level summary
- Hop 0: mean dpoints/sample = 6.343 [CI 6.343, 6.343], mean position = 20.264 [CI 20.264, 20.264]
- Hop 1: mean dpoints/sample = 6.156 [CI 6.156, 6.156], mean position = 19.794 [CI 19.794, 19.794]
- Hop 2: mean dpoints/sample = 3.737 [CI 3.737, 3.737], mean position = 23.262 [CI 23.262, 23.262]
- Hop 3: mean dpoints/sample = 3.551 [CI 3.551, 3.551], mean position = 22.386 [CI 22.386, 22.386]
- Hop 4: mean dpoints/sample = 3.367 [CI 3.367, 3.367], mean position = 22.652 [CI 22.652, 22.652]
- Hop 5: mean dpoints/sample = 3.045 [CI 3.045, 3.045], mean position = 23.560 [CI 23.560, 23.560]
- Hop 6: mean dpoints/sample = 2.648 [CI 2.648, 2.648], mean position = 21.549 [CI 21.549, 21.549]

## Adjacent-hop changes
- Hop 0 -> Hop 1: delta dpoints/sample = -0.187 (non-overlapping CIs), delta mean position = -0.470 (non-overlapping CIs)
- Hop 1 -> Hop 2: delta dpoints/sample = -2.419 (non-overlapping CIs), delta mean position = +3.468 (non-overlapping CIs)
- Hop 2 -> Hop 3: delta dpoints/sample = -0.186 (non-overlapping CIs), delta mean position = -0.876 (non-overlapping CIs)
- Hop 3 -> Hop 4: delta dpoints/sample = -0.183 (non-overlapping CIs), delta mean position = +0.267 (non-overlapping CIs)
- Hop 4 -> Hop 5: delta dpoints/sample = -0.322 (non-overlapping CIs), delta mean position = +0.908 (non-overlapping CIs)
- Hop 5 -> Hop 6: delta dpoints/sample = -0.397 (non-overlapping CIs), delta mean position = -2.012 (non-overlapping CIs)

## Emergence / disappearance
- Aligned comparisons available: 13637
- Gained divergence tokens: 472
- Lost divergence tokens: 585
- Consistent nonzero divergence tokens: 12412

## Correctness matrices
- Hop 0: target accuracy = 0.828, target accuracy at decision points = 0.613, matrix density = 0.829
- Hop 1: target accuracy = 0.798, target accuracy at decision points = 0.636, matrix density = 0.796
- Hop 2: target accuracy = 0.820, target accuracy at decision points = 0.619, matrix density = 0.824
- Hop 3: target accuracy = 0.732, target accuracy at decision points = 0.684, matrix density = 0.729
- Hop 4: target accuracy = 0.712, target accuracy at decision points = 0.688, matrix density = 0.708
- Hop 5: target accuracy = 0.689, target accuracy at decision points = 0.673, matrix density = 0.686
- Hop 6: target accuracy = 0.629, target accuracy at decision points = 0.692, matrix density = 0.625
