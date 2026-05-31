# Divergence Token Evolution Summary

- Experiment directory: /home/abasso_aims_ac_za/divergence-tokens/workspace/multihop
- Model: qwen
- Target preference: panda
- Seeds analyzed: [42]
- Hops analyzed: ['hop0', 'hop1', 'hop2', 'hop3', 'hop4']

## Hop-level summary
- Hop 0: mean dpoints/sample = 2.139 [CI 2.139, 2.139], mean position = 16.405 [CI 16.405, 16.405]
- Hop 1: mean dpoints/sample = 2.286 [CI 2.286, 2.286], mean position = 16.821 [CI 16.821, 16.821]
- Hop 2: mean dpoints/sample = 2.173 [CI 2.173, 2.173], mean position = 16.840 [CI 16.840, 16.840]
- Hop 3: mean dpoints/sample = 2.084 [CI 2.084, 2.084], mean position = 16.983 [CI 16.983, 16.983]
- Hop 4: mean dpoints/sample = 2.025 [CI 2.025, 2.025], mean position = 17.002 [CI 17.002, 17.002]

## Adjacent-hop changes
- Hop 0 -> Hop 1: delta dpoints/sample = +0.147 (non-overlapping CIs), delta mean position = +0.416 (non-overlapping CIs)
- Hop 1 -> Hop 2: delta dpoints/sample = -0.113 (non-overlapping CIs), delta mean position = +0.019 (non-overlapping CIs)
- Hop 2 -> Hop 3: delta dpoints/sample = -0.089 (non-overlapping CIs), delta mean position = +0.143 (non-overlapping CIs)
- Hop 3 -> Hop 4: delta dpoints/sample = -0.059 (non-overlapping CIs), delta mean position = +0.020 (non-overlapping CIs)

## Emergence / disappearance
- Aligned comparisons available: 114673
- Gained divergence tokens: 9676
- Lost divergence tokens: 10174
- Consistent nonzero divergence tokens: 84561

## Correctness matrices
- Hop 0: target accuracy = 0.837, target accuracy at decision points = 0.564, matrix density = 0.837
- Hop 1: target accuracy = 0.865, target accuracy at decision points = 0.557, matrix density = 0.866
- Hop 2: target accuracy = 0.862, target accuracy at decision points = 0.562, matrix density = 0.863
- Hop 3: target accuracy = 0.855, target accuracy at decision points = 0.564, matrix density = 0.856
- Hop 4: target accuracy = 0.848, target accuracy at decision points = 0.562, matrix density = 0.849
