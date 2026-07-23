# Data Contract

- Endpoint `/predict` returns field `top_class`.
- Endpoint `/history` returns field `predicted_class` (display name; intentionally different from predict).
- `recycling_score` is on a scale of 0 to 100.
- The upload field must be named exactly `file`.
- 5 classes: Glass, Metal, Paper, Plastic, Waste.
