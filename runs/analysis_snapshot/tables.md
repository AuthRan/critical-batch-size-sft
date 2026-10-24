| model | dataset | B (tokens) | mean tokens/step | lr* | 68% CI | loss* | flag | seeds | diverged |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-360M | dolly | 512 | 604 | 1.71e-05 | [1.7e-05, 1.7e-05] | 1.9107 | ok | 3 | 1.0e-03: 1 |
| SmolLM2-360M | dolly | 1024 | 1117 | 2.21e-05 | [2.2e-05, 2.2e-05] | 1.9104 | ok | 1 | - |

| model | dataset | variant | alpha | 68% CI | CBS_fixed_data (tokens) | n_B |
|---|---|---|---|---|---|---|
| SmolLM2-360M | dolly | adamw+noclip | nan | [nan, nan] | 1024 | 2 |
