# Defect log

## Pass 0 — Plan review
- Status: 
- Open risks: 

Resolved: one left-to-right source-to-use story; no Q1-only numbers, forecasts, or optimization details; emergency purchase is visually and semantically separated as a contingency path.

## Screenshot review cycles

### Cycle 1
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

Static review: Draw.io validation reports 19 cells (11 vertices, 6 edges), no duplicate IDs, no empty labels, no embedded rasters, and the intended 1600 × 900 canvas.

### Cycle 2
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

Quality review: normalized stroke palette to three semantic stroke colors and combined the legend background with its label; `validate_visual_quality.py` reports 0 FAIL and 0 WARN.

### Cycle 3
- Screenshot (canvas-only): 
- P0/P1 inventory: 
- Fixes and verification: 

Export note: the local Quick Look SVG thumbnail crops wide canvases, so it is not retained as an export. The editable Draw.io source and full-canvas SVG are the delivered files.

## Red-team audit
- Text: 
- Arrows: 
- Boxes/overlap: 
- Spacing/layout: 
- Color/typography: 
- Icons/assets: 
- Semantics/regressions: 

## Self-score
| Dimension | Score /10 | Evidence |
|---|---:|---|
| Text readability | | |
| Arrow accuracy | | |
| Color coherence | | |
| Layout consistency | | |
| Style/spec match | | |
| **Total /50** | | |

## Remaining gaps
- A final PNG can be exported directly from diagrams.net when it is available; no semantic or layout changes are required.
