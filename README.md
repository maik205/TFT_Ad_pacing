# Forecast-Driven Daily Budget Pacing

- **[v2/](v2/)**: the current study. It runs on real Criteo campaign logs, follows a pre-registered
  [protocol](v2/PROTOCOL.md), and every reported number is generated from the results files.
- **[v4_ctr/](v4_ctr/)**: the click-probability (CTR) model for the mgads ad server: an ad-format fork
  of the v3 synthetic generator, a CADET-style model and a logistic-regression baseline, exported to ONNX.
  Synthetic only; see [v4_ctr/RESULTS.md](v4_ctr/RESULTS.md).
- **[legacy/](legacy/)**: the v1 material. It is **invalid**: the data was synthetic, the
  ablations were fabricated and the figures were drawn from formulas.
  See [legacy/README.md](legacy/README.md). It is kept only as a record.
