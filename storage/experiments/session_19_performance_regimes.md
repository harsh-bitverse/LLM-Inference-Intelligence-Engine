# Session 19 — Empirical Performance Regimes Characterization

**Experiment Manifest:** `exp-session-19-sustained-load`  
**Dataset Corpus:** 93 empirical observations across Sessions 12B–19  
**Hardware Platform:** Intel(R) Core(TM) i7-14700 (20 cores, 28 threads, 31.69 GB RAM)  
**Inference Runtime:** Ollama `0.40.11` / `llama3.2:1b` (Q8_0, 1.2B parameters, GGUF)

---

## 1. Operating Regimes Taxonomy

Based on controlled physical measurements under varying prefix caching states, model residency, generation lengths, and host system load, IIE identifies five distinct, statistically separable operating regimes:

1. **`COLD_START`**: The model is not resident in host memory (`keep_alive: 0` or fresh daemon startup). The request incurs a mandatory $1,038\text{ ms} \pm 70\text{ ms}$ disk-to-RAM loading phase before prompt prefill begins.
2. **`WARM_SHORT_BURST`**: Model resident in RAM; short generation ($O \le 32$ tokens) dispatched from an isolated or post-idle state. Decode throughput is high ($21.8\text{--}24.1\text{ tok/s}$) with minimal latency variance.
3. **`WARM_SUSTAINED_GENERATION`**: Model resident in RAM; prolonged generation ($O \ge 64$ tokens) operating without background CPU contention. Decode throughput remains stable at $20.7\text{--}22.8\text{ tok/s}$.
4. **`PROMPT_PREFIX_REUSE`**: Model resident in RAM with warm prompt prefix KV cache hit. Prompt evaluation drops from $\approx 3.03\text{ ms/tok}$ to $\approx 0.04\text{ ms/tok}$ ($\approx 49\text{ ms}$ fixed re-evaluation time), eliminating prefill bottleneck.
5. **`CONSECUTIVE_SATURATED` / `CONTENTION_DEGRADED`**: Model resident in RAM under sustained multi-request generation or background host thread contention. Decode throughput degrades by $\approx 45\%$ (dropping from $\approx 22.7\text{ tok/s}$ to $12.56\text{ tok/s}$, or $79.5\text{ ms/tok}$), producing extreme latency tails up to $10,302.9\text{ ms}$.

---

## 2. Quantitative Regime Metrics Summary

| Operating Regime | Observation Count | Mean Client Latency (ms) | Median Client Latency (ms) | Latency IQR (ms) | Worst-Case Latency (ms) | Median Decode TPS (tok/s) | Min/Max Decode TPS (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`COLD_START`** | 5 | $3,051.29$ | $3,062.53$ | $200.96$ | **$3,173.81$** | $21.00$ | $20.39 / 23.79$ |
| **`WARM_SHORT_BURST`** | 15 | $1,809.27$ | $1,545.01$ | $375.75$ | **$4,754.48$** | $21.82$ | $18.68 / 24.13$ |
| **`WARM_SUSTAINED_GENERATION`** | 15 | $4,379.90$ | $4,972.63$ | $2,301.60$ | **$7,149.57$** | $20.71$ | $18.28 / 23.79$ |
| **`PROMPT_PREFIX_REUSE`** | 49 | $3,180.24$ | $2,829.56$ | $4,161.61$ | **$5,995.86$** | $22.79$ | $20.13 / 24.08$ |
| **`CONSECUTIVE_SATURATED`** | 9 | $5,471.10$ | $2,738.62$ | $6,991.27$ | **$10,302.90$** | **$12.56$** | $12.25 / 14.76$ |

---

## 3. Comparative Observations & Empirical Takeaways

1. **The Decode Degradation Phenomenon:**
   - In unconstrained environments, decode throughput operates tightly between $20.7$ and $22.8\text{ tok/s}$.
   - Under sustained back-to-back request bursts or background OS core contention across Intel hybrid cores (P-cores vs E-cores), decode throughput falls to $12.56\text{ tok/s}$.
   - For an $O=128$ token workload, this drops generation speed from $44\text{ ms/tok}$ to $79.5\text{ ms/tok}$, inflating decode duration by $+4,554\text{ ms}$.
2. **Cold Start Determinism:**
   - Model loading duration from local NVMe storage into host RAM is remarkably consistent: $1,038\text{ ms} \pm 70\text{ ms}$, with total cold request duration bounded at $3,173.81\text{ ms}$ for $O=32$.
3. **Prefix Replay Efficiency:**
   - Prefix cache hits preserve decode throughput ($22.79\text{ tok/s}$) while slashing prefill latency by over $98\%$ (e.g., $49.55\text{ ms}$ vs $3,190.74\text{ ms}$ on 677 prompt tokens).
4. **Impact on SLA Decisions:**
   - Recommending configurations assuming the best-case decode rate ($22.72\text{ tok/s}$) produces catastrophic SLA violations when the system operates in the `CONSECUTIVE_SATURATED` regime.
   - An SLA policy must incorporate conservative worst-case bounds ($12.5\text{ tok/s}$) and report empirical violation risks.
