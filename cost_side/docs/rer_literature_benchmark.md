# Exchange-Rate Cost-Shifter Literature Benchmark

This note benchmarks our cost-side exchange-rate specification against closely related papers in top field or general-interest economics journals, plus NBER working papers that are the corresponding working-paper versions.

## Scope

The relevant comparison class is papers that use exchange rates to move costs, markups, or pass-through, rather than papers that use exchange rates only as macro controls.

## Main Papers

### 1. Feenstra (1989 JIE; NBER Working Paper 2453)

Source:

- NBER working paper page: <https://www.nber.org/papers/w2453>
- NBER PDF: <https://www.nber.org/system/files/working_papers/w2453/w2453.pdf>

What the paper does:

- Studies pass-through of tariffs and exchange rates to U.S. prices of Japanese cars, trucks, and motorcycles.
- Writes the import-pricing equation as a function of foreign costs in domestic currency, with the key cost term entering through `w*/e`.

Relevant specification detail:

- The paper's regression specification is a log-linear pricing equation with a time trend and the exchange-rate-adjusted foreign cost term:
  - `ln p_t = alpha_t + beta ln(w/e_t) + gamma ln(1+t_t) + ...`
- See the NBER PDF discussion of the pricing equation and regression specification at pp. 4-11, especially the pricing equation and equation `(11)`.

Implication for us:

- This is an early benchmark where exchange rates enter directly through a cost term.
- The paper does not add a separate source-country fixed effect on top of the pricing equation. The identification comes from exchange-rate-driven cost movement plus standard controls.

### 2. Knetter (1993 AER), as summarized in Goldberg and Knetter (1997 JEL; NBER Working Paper 5862)

Source:

- NBER review page: <https://www.nber.org/papers/w5862>
- Search snippet with the core regression: <https://www.nber.org/system/files/working_papers/w5862/w5862.pdf>

What the paper does:

- Sets out the classic pricing-to-market fixed-effects regression for export prices across destinations.

Relevant specification detail:

- Goldberg and Knetter summarize Knetter's benchmark regression as
  - `ln p_it = theta_t + lambda_i + beta_i ln E_it + u_it`
- with `theta_t` as time effects and `lambda_i` as destination-country effects.
- The purpose of the time effects is to absorb common marginal-cost movements across destinations, while destination effects absorb persistent destination-level price differences.

Implication for us:

- The standard benchmark is time effects plus destination effects, not source-country fixed effects layered on top of a product effect.
- This is the clearest reduced-form precedent for treating exchange rates as cost/markup shifters while holding common cost movements fixed with time effects.

### 3. Goldberg and Verboven (2001 ReStud; NBER Working Paper 6818)

Source:

- NBER working paper page: <https://www.nber.org/papers/w6818>

What the paper does:

- Structural auto paper on the European car market.
- Uses exchange-rate fluctuations to explain local-currency price stability and to separate markup versus cost channels in automobile prices.

Relevant result:

- The NBER abstract states that local-currency price stability can come from either a local component in marginal costs or markup adjustment, and reports that roughly `2/3` of the observed price inertia is attributed to local costs and `1/3` to markup adjustment.

Implication for us:

- This is directly relevant because it is an auto-market structural paper.
- The paper's logic is to decompose exchange-rate pass-through into cost and markup channels, not to absorb the source country with an extra fixed effect.

### 4. Goldberg and Hellerstein (2013 ReStud; NBER Working Paper 13183)

Source:

- NBER working paper page: <https://www.nber.org/papers/w13183>
- NBER PDF: <https://www.nber.org/system/files/working_papers/w13183/w13183.pdf>

What the paper does:

- Structural decomposition of incomplete exchange-rate pass-through in the beer market.
- Lets producer costs have both a traded component and a non-traded local destination-market component.
- Uses firms' first-order conditions to back out markups and marginal costs, and then decomposes incomplete pass-through.

Relevant specification detail:

- The introduction states that the supply side allows for both a traded and a non-traded local cost component.
- The paper then uses first-order conditions to back out marginal costs and markups, and explicitly decomposes marginal costs into traded and non-traded pieces.

Relevant result:

- The abstract reports that about `60%` of incomplete pass-through is due to local non-traded costs, `8%` to markup adjustment, and `30%` to price-adjustment costs.

Implication for us:

- This is another benchmark where exchange-rate sensitivity is analyzed through an explicit cost decomposition rather than through source-country fixed effects.
- The relevant empirical choice is whether to model traded versus local cost components and markup adjustment, not whether to add a standalone source-country FE.

### 5. Amiti, Itskhoki, and Konings (2014 AER; NBER Working Paper 18615)

Source:

- NBER working paper page: <https://www.nber.org/papers/w18615>
- NBER PDF: <https://www.nber.org/system/files/working_papers/w18615/w18615.pdf>

What the paper does:

- Studies how import intensity and market share shape exchange-rate pass-through using Belgian firm-product-destination data.
- This is the closest benchmark to our interaction-based approach because import intensity is used to proxy marginal-cost sensitivity to exchange rates.

Relevant specification detail:

- The paper states that import intensity captures the marginal-cost channel and destination-specific market share captures the markup channel.
- The benchmark empirical specification regresses price changes on exchange-rate changes interacted with import intensity and market share.
- The paper explicitly notes that the demanding version of the specification includes `sector-destination` effects and then `sector-destination-year` fixed effects.

Implication for us:

- This is the strongest empirical precedent for using an interaction between exchange rates and an exposure measure that proxies marginal-cost sensitivity.
- The fixed effects benchmark is richer on the destination side, not on the source-country side.
- That is closer to `year` or market-time absorption plus exposure interactions than to adding a primary-source-country FE in our canonical domestic panel.

### 6. Auer, Burstein, and Lein (2021 AER; NBER Working Paper 28404)

Source:

- NBER working paper page: <https://www.nber.org/papers/w28404>
- AEA article page: <https://www.aeaweb.org/articles?id=10.1257/aer.20181415>

What the paper does:

- Uses the 2015 Swiss franc appreciation to trace exchange-rate movements through border prices, retail prices, and expenditure shares.

Implication for us:

- This paper is less about a marginal-cost interaction regression than the papers above, but it reinforces the same empirical design logic: use exchange-rate variation to move border or traded costs, and trace the pass-through rather than saturating the regression with source-country fixed effects.

## Takeaways For Our Specification

1. The literature's common reduced-form template is exchange-rate variation plus time effects and market/destination/product fixed effects.

2. The literature's common structural template is to decompose costs into traded and local components and then separate cost from markup adjustment.

3. The closest benchmark to our `rho x RER` interaction is Amiti, Itskhoki, and Konings (2014), where import intensity proxies marginal-cost sensitivity and the fixed effects are sector-destination or sector-destination-year, not source-country FE.

4. None of these benchmark papers suggests that the right default robustness move is to add a source-country fixed effect on top of a product fixed effect when source country is already time-invariant within product.

5. In our setting, once we restrict to the canonical domestic panel, `pcOth1_code1` is effectively nested inside `make_model` in levels. That makes a source-country FE redundant in the levels regressions and weakly motivated as a benchmark robustness in first differences.

## Bottom Line

Removing the source-country-FE variants is consistent with the literature benchmark.

If we want the specification to look more like the published exchange-rate pass-through literature, the stronger moves are:

- keep `make_model + year` in levels and `year` in first differences,
- focus on exposure interactions and timing/placebo tests,
- optionally absorb richer market-time conditions when the data structure permits, and
- interpret the exchange-rate term as moving traded-cost exposure rather than as something that requires a separate source-country FE for credibility.
