# Arithmetic check

Every derived number in the abstract, recomputed by hand from the analysis CSVs
under `data/processed/horizon/`, so that an error of the kind the first revision
made (a level printed as a gap; a ratio computed on the wrong basis) would show
here. Inputs are quoted to the precision the CSV holds; the arithmetic is written
out. The conclusion quotes no number.

`tests/python/test_paper_numbers.py` recomputes the abstract's headline numbers
from the raw seed files, independently of the analysis scripts, and fails if a
printed macro disagrees. This note is the hand check beside it.

## The round-99 gap and the peak

Source `e09_canonical/analysis/summary.csv`, confirmation set, 20 seeds.

- Peak of the seed-mean gap: 0.313134, at round 6. Printed +0.313 at round 6.
- Round-99 gap: mean 0.025330, 95% t bounds 0.021693 and 0.028967. Half-width
  0.028967 − 0.025330 = 0.003637, and 0.025330 − 0.021693 = 0.003637: symmetric,
  as a t interval is. With t(0.975, 19) = 2.093, the implied seed sd is
  0.003637 × √20 / 2.093 = 0.00777.
- BCa bounds 0.022393 and 0.029153. Differences from the t bounds: 0.000700 and
  0.000186. The larger, 0.0007, is the "agree to" figure of the Design section.

## The entry rounds

Source `e09_canonical/analysis/per_seed.csv`, both sets. Confirmation: minimum
199, maximum 261; exploratory: minimum 194, maximum 269. Over all forty: 194 to
269. No seed is censored in either set (summary `n_censored` 0 and 0).

## The equivalence test on the tail

Source as above, confirmation set: tail mean 0.000043, seed sd 0.000393, n = 20.

- Standard error 0.000393 / √20 = 0.00008788.
- t against −0.002: (0.000043 + 0.002) / 0.00008788 = 23.25. The CSV's 23.237
  uses unrounded inputs. Printed 23.2.
- t against +0.002: (0.002 − 0.000043) / 0.00008788 = 22.27; the CSV's 22.264.
  Printed 22.3.
- 90% interval: 0.000043 ± 1.729 × 0.0000879 = [−0.000109, +0.000195], inside
  ±0.002. The CSV's `tost_lo90` and `tost_hi90` are −0.000109 and +0.000195.

## The dose elasticity

Source `e13a_dose/analysis/by_dose.csv`. Mean T(0.05) at 0.5, 1, 2 and 4 passes
per round: 173.9, 87.1, 43.6, 22.0.

- ln dose: −0.6931, 0, 0.6931, 1.3863 (mean 0.3466). ln T: 5.1585, 4.4671,
  3.7751, 3.0910 (mean 4.1229).
- Deviations x − x̄: −1.0397, −0.3466, 0.3466, 1.0397; y − ȳ: 1.0356, 0.3442,
  −0.3478, −1.0319.
- Slope = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)² = (−1.0767 − 0.1193 − 0.1205 − 1.0729) /
  (2 × 1.0810 + 2 × 0.1201) = −2.3894 / 2.4023 = −0.9946.
- The CSV's elasticity, fitted on the forty per-seed values rather than the four
  means, is −0.994. Printed −0.99.

## The half-life

Source `e10_long/analysis/fits.csv`, pure exponential: rate λ = 0.0234531.
Half-life ln 2 / λ = 0.693147 / 0.0234531 = 29.55 rounds. Printed 29.6.

## The long tail

Source: the per-seed means over rounds 950 to 999, ten seeds:
0.00088, 0.00020, 0.00006, 0.00070, 0.00128, 0.00002, −0.00015, 0.00094, 0.00038,
0.00033. Sum 0.00464, mean 0.000464; nine of ten positive. The CSV's tail mean is
0.000463 (from unrounded values). Printed +0.00046. As a share of the 0.002
margin: 0.000463 / 0.002 = 23 percent.

## The open market

Source `e14_arrivals/analysis/summary.csv`, ten seeds per rate, mean over rounds
500 to 599: 0.01648, 0.04034, 0.10113 at 0.5, 1 and 2 percent per round. Printed
+0.016, +0.040, +0.101. Every lower 95% bound (0.01466, 0.03659, 0.09592) is above
the 0.002 margin, which is the abstract's "does not close".

## The warm start ("about four fifths")

Source `e13d_warm/analysis/summary.csv`, `gap_round0_F299`: confirmation 0.2075
(95% t 0.1789 to 0.2362), exploratory 0.2403. Removed: 1 − 0.2075 = 0.79 and
1 − 0.2403 = 0.76. Both are about four fifths.

That quantity is a share: the CSV holds it as a fraction and the paper prints it
as a percentage, 20.8 and 24.0. Checking the fraction against the percent reads as
a factor-of-100 error, which is the one arithmetic slip this note made when it was
recomputed by script. Every other line was reproduced from the CSVs to the digits
printed, including the two equivalence statistics (23.237 and 22.264, printed as
23.2 and 22.3) and the 90% interval, whose hand bounds (−0.000109 and +0.000195)
match `tost_lo90` and `tost_hi90`.
