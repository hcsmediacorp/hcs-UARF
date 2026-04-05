# TurboQuant Research Notes (Google, 2026) + UARF Alignment

**Stand:** 2026-04-05  
**Kontext:** Konsolidierung der UARF-TurboQuant-Roadmap mit öffentlich verfügbaren Google-Quellen.

## 1) Relevante Google-Quellen

1. **Google Research Blog (2026-03-24):**  
   *TurboQuant: Redefining AI efficiency with extreme compression*  
   Kernaussagen:
   - Fokus auf speicherkritische KI-Pfade (insb. KV-Cache bei langen Kontexten).
   - Kombination theoretisch fundierter Quantisierung mit praktischer Recall-/Distortion-Optimierung.
   - Benchmarks auf LongBench/Needle-in-a-Haystack/ZeroSCROLLS/RULER/L-Eval.
   - Vergleich u. a. gegen KIVI-Baselines.

2. **Google AQT (Accurate Quantized Training, GitHub):**  
   Kernaussagen:
   - WYTIWYS-Prinzip („what you train is what you serve“).
   - Quantization-aware und serving-orientierte Konvertierungspfade.
   - Produktions- und Forschungseinsatz in JAX/Flax/Pax/MaxText-Kontexten.

## 2) Implikationen für UARF TurboQuant

Für UARF ergibt sich ein sinnvoller 3-Stufen-Pfad:

1. **Numerische Robustheit zuerst**  
   - stabile Quantisierung bei Degenerationsfällen (z. B. konstante Tensoren),
   - defensive Dequantisierung.

2. **AQT-inspirierter Modus als Einstieg**  
   - expliziter `aqt_int8`-Modus als API-Flag,
   - später Erweiterung um per-channel/per-group Parameterisierung.

3. **Benchmark- und Qualitäts-Contracts**  
   - reproduzierbare Distortion/Recall/KV-Memory-Metriken,
   - später optional LongBench-nahe Auswertung für Regression-Checks.

## 3) In dieser Änderung umgesetzt

- `aqt_int8` als zusätzlicher Quantisierungsmodus im TurboQuant-Level ergänzt.
- INT8/INT4-Quantisierung gegen Zero-Range/konstante Tensoren abgesichert.
- INT4-Packlogik auf flache Speicherrepräsentation stabilisiert.

Damit ist ein robusterer Ausgangspunkt vorhanden, um die größeren Google-inspirierten Themen (KV-Cache-spezifische Verfahren, Distortion-gesteuerte Optimierung, Benchmark-Integration) in den nächsten Iterationen sauber aufzubauen.

