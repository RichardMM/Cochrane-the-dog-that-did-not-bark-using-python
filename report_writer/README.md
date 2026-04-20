# report_writer

Drop-in logging helper for EDA scripts that currently rely on many `print(...)` calls.

## Quick start

```python
from report_writer import HTMLReport, report

r = HTMLReport(title="Breakout EDA")

# Log-style replacement for print
report("Starting regressions", r)
report(phi, r, title="phi")

# Tabular output (DataFrame / Series / dict / numpy array)
report(mydata.dtypes, r, kind="table", title="Input dtypes")
report({"bd": bd, "br": br, "rho": rho}, r, kind="table", title="Model parameters")

# Chart location (save with matplotlib first)
plt.savefig("charts/bd_br_scatter.png", bbox_inches="tight")
report(
    "charts/bd_br_scatter.png",
    r,
    kind="chart",
    title="Breakout: br vs bd",
    caption="Simulated return coefficient vs simulated growth coefficient",
)

# Produce HTML output
html_path = r.write("reports/breakout_report.html")
print(f"Report written to {html_path}")
```

## Suggested replacements in your breakout script

- `print(mydata.dtypes.index)` -> `report(mydata.dtypes, r, kind="table", title="Input dtypes")`
- `print(phi)` -> `report(phi, r, title="phi")`
- `print(bd)` -> `report(bd, r, title="bd")`
- `print(simulated_divyields.shape)` -> `report(simulated_divyields.shape, r, title="simulated_divyields shape")`
- `print(led_dividendyields[:, 1])` -> `report(led_dividendyields[:, 1], r, kind="table", title="Sample led dividend yield")`
- `plt.show()` -> save figure then `report(..., kind="chart")`

This keeps every log, table, and chart together in a single neat HTML file.
