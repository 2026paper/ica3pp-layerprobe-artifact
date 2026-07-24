# Figure 1 Method Overview QA

Generated: `2026-07-25T03:02:34+08:00`

Output stem: `fig0_layerprobe_overview`

Output directory: `${ARTIFACT_ROOT}\results\figure1_method_final_20260725`

## Outcome

This is a fully original LayerProbe method figure. Its drawing pipeline uses
only project semantics and programmatically generated vector/text primitives.
An optional external image, when explicitly supplied, is used only for
non-destructive provenance and side-by-side visual review.

## Source and provenance

| Item | Path | SHA-256 |
| --- | --- | --- |
| Earlier LayerProbe figure (optional QA) | ${ARTIFACT_ROOT}\results\deadline_figures_distancefix_final_20260724_xeon\fig0_layerprobe_overview.png | 7b22a8687da3e009a9b9795e4ee9a9f4be8f58e704e965a775322c4808e645b3 |
| Paper method text (optional QA) | ${LOCAL_WORKSPACE}\ICA3PP_截止日前增强实验版_2026-07-23\07_论文\manuscript\draft\paper.tex | dc7266097eacadac01660c6e607c6c51a868246cf61eb1a921d1f894c3331818 |
| LayerProbe evaluator | ${ARTIFACT_ROOT}\src\layerprobe\evaluator.py | e9b183db3df914db4496f88ee6d654f7175be2b1e6e44cf7b034a9ce9f3f82b1 |
| Independent-oracle boundary | ${ARTIFACT_ROOT}\experiments\INDEPENDENT_TRACE_ORACLE_README.md | 743bd3f8e86d01e27836a2c1d91d71bb09c8336380887d91fdade0033f4b6150 |
| Figure generator | ${ARTIFACT_ROOT}\experiments\build_figure1_layerprobe_method.py | b1700f389e2cda359257583bde85b61a53c61eebbb72c2601f5366374e6244ed |

No optional style reference was supplied. Figure construction and publication QA were completed from project semantics and original Matplotlib primitives only.

Earlier LayerProbe figure dimensions: `1888 × 1104` pixels (aspect ratio `1.710`).

New Figure 1 aspect ratio: `1.873`.

## Method-semantic audit

| Element | Meaning encoded in the figure | Status |
| --- | --- | --- |
| Problem | A presentation may change observation, memory, action, and later trajectory; state-only reuse is unsafe. | PASS |
| Reuse level 1 | Validate each mechanism once because presentations cannot change physics, terminal rules, or validity. | PASS |
| Reuse level 2 | Within fixed mechanism k and agent a, reuse only on q = (world state s, pre-ingest memory m, observation o). | PASS |
| Cached result | r(q) contains action, next state, next memory, and next status; miss computes and stores it. | PASS |
| Local state | Observation generation, display history, and declared presentation traces are never merged. | PASS |
| Output | Declared traces produce six-bit signatures; an all-p intersection and exact cover form one downstream query. | PASS |
| Independent validation | A separate interpreter rebuilds semantics and compares validity, every trace, and signatures over the complete finite domain. | PASS |

The cache is deliberately labeled as scoped to a fixed `(k, a)`. Therefore the
displayed complete execution key is exactly `q = (s, m, o)`, without implying
that mechanism or agent identity can be omitted from a globally shared cache.

## Originality and external-image independence audit

| Feature | Optional external input | New Figure 1 |
| --- | --- | --- |
| Global composition | No external reference supplied | One unlettered left-to-right semantic ledger plus a bottom audit rail |
| Primary visual grammar | Not used by the drawing pipeline | Four open ledger stages separated by hairlines; no dashed enclosure |
| Motifs | Not imported or embedded | Finite-product matrix, contract, exact-key index, signature ledger |
| Information flow | Not used to position any element | Invariant lane fans into presentation-local lanes, then robust aggregation |
| Panel labels | Not copied | None |
| Construction | Optional file is provenance-only | Original Matplotlib vector/text primitives generated from method semantics |

Verdict: **PASS — clearly different composition and visual grammar.**

The new design contains no time-series plot, Transformer/GNN block, graph
adjacency matrix, lettered subpanel, dashed outer container, or left-side
vertical input-to-output pipeline.

## Publication QA

| Check | Result | Status |
| --- | --- | --- |
| PDF size | 4.720 × 2.520 in | PASS |
| Colour PNG | 1888 × 1008 px at 400.00 dpi | PASS |
| Grayscale PNG | 1888 × 1008 px; mode L | PASS |
| Visible text | Times New Roman; minimum 6.0 pt; 78 objects | PASS |
| PDF fonts | 3 embedded subsets; no Type 3 | PASS |
| Vector integrity | 0 raster image XObjects | PASS |
| Canvas bounds | minimum text clearance 36.02 px; 0 text-overlap pairs | PASS |
| External-image independence | no external style-reference supplied or read | PASS |

Embedded PDF base fonts: `/CHAXTK+TimesNewRomanPSMT, /DEBKYM+TimesNewRomanPS-BoldMT, /HHLWDR+TimesNewRomanPS-ItalicMT`.

## Exact 400-dpi spacing audit

All measurements below are computed on the delivered `1888 × 1008` canvas,
after the final Matplotlib draw and before export.

| Relationship | Measured minimum | Gate | Status |
| --- | --- | --- | --- |
| Canvas edge → text | 36.021 px | ≥ 24 px | PASS |
| Text ↔ text | 4.231 px; 0 overlaps | ≥ 4 px | PASS |
| Arrow ↔ text | 6.262 px across 17 arrows | ≥ 5 px | PASS |
| Routed line ↔ text | 14.156 px across 10 connector lines | ≥ 10 px | PASS |
| Card edge → contained text | 6.015 px minimum | ≥ 4 px | PASS |
| Green contract → fan-out gutter | 18.880 px | ≥ 18 px | PASS |
| Green entry arrow → p pill | 13.216 px | ≥ 12 px | PASS |
| Blue query collector → index card | 16.992 px | ≥ 16 px | PASS |
| Purple return line → output label | 17.103 px | ≥ 16 px | PASS |

Curated card-containment margins:

| Card | Minimum internal text margin | Status |
| --- | --- | --- |
| state warning | 12.428 px | PASS |
| validate once | 7.007 px | PASS |
| presentation contract | 6.015 px | PASS |
| exact-step index | 6.341 px | PASS |
| local-state note | 6.015 px | PASS |
| robust-suite card | 11.055 px | PASS |
| audit comparison | 8.031 px | PASS |
| audit outcome | 8.031 px | PASS |

## Visual QA

Status: **PASS**

- Round 1 identified excessive density in the complete-key card, green fan-out, and purple validation return; labels were shortened and the connectors were rerouted through dedicated gutters and ports.
- Round 2 inspected the 1888 × 1008 colour rendering: the left-to-right reading order, two reuse levels, local presentation lanes, output ledger, and independent audit rail are clear with no clipping or collision.
- Round 3 inspected the true-grayscale rendering and exact 400-dpi geometry: stage boundaries, arrows, key segments, trace lanes, signature bits, and audit comparison remain readable without relying on hue.
- The generator contains no external-image loading path in its drawing pipeline and emits only original vector/text primitives.
- External-reference comparison: not requested for this run; reference-independent construction verified.

## Reproduction

From the project root:

```powershell
python experiments/build_figure1_layerprobe_method.py
```

To generate the manuscript replacement directly:

```powershell
python experiments/build_figure1_layerprobe_method.py `
  --output-dir ../../07_论文/manuscript/draft/figures `
  --stem fig0_layerprobe_overview
```

The script writes only `fig0_layerprobe_overview.pdf`, `fig0_layerprobe_overview.png`,
`fig0_layerprobe_overview_grayscale.png`, and `FIGURE1_METHOD_QA.md` in the selected directory.
It does not edit `paper.tex`, does not require a third-party image, and never
writes to an optional style-reference path.
