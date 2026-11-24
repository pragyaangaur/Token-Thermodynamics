/-
Token Thermodynamics: the two-level model behind the melting law.
-/
import TokenThermo.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The two-level model

One ground state at energy `0` and `g` states at energy `Δ > 0` (a vocabulary of `V` tokens has
`g = V - 1`, and the paper writes `g = V`; the difference between `log (V - 1)` and `log V` is below
`1 / (V - 1)`). With `x = Δ / T` and `u = g e^{-x}` the heat capacity is `C = x^2 u / (1 + u)^2`.

Everything in this file about the peak of `C` is new relative to the paper, which states the peak
equation `x* = 2 (1 + u) / (1 - u)` and the form `x* = log V + c` without proof.

Main results, for a real `g ≥ 1`:

* `heatCap_twoLevel`: the general `C` of `TokenThermo.Basic` equals `x^2 u / (1 + u)^2` for the
  energy vector `twoLevel g Δ`.
* `two_level_peak`: the peak equation has exactly one solution `x*`, it exceeds `log g` and `2`, and
  `C` has a strict global maximum there, in `x` and in `T`.
* `peakOccupancy`, `peakHeight`: `p_top(T_melt) = 1/2 + 1/x*` and `C_max = (x*^2 - 4) / 4`.
* `c_lower`, `c_upper`: with `c = x* - log g`, `c (log g + c) > 4` and
  `c^2 + (log g - 2) c ≤ 4`, which pin `c` to within about 20 percent and give `c ≈ 4 / log g`.
-/

open Finset Real

namespace TokenThermo

namespace TwoLevel

variable (g : ℝ)

/-- `u = g e^{-x}`, the ratio of the bulk weight to the ground-state weight. -/
noncomputable def u (x : ℝ) : ℝ := g * exp (-x)

/-- The two-level heat capacity as a function of `x = Δ / T`. -/
noncomputable def cTwo (x : ℝ) : ℝ := x ^ 2 * u g x / (1 + u g x) ^ 2

/-- Minus the numerator of the stationarity condition: `x (1 - u) - 2 (1 + u)`. -/
noncomputable def H (x : ℝ) : ℝ := x * (1 - u g x) - 2 * (1 + u g x)

variable {g}

lemma u_pos (hg : 0 < g) (x : ℝ) : 0 < u g x := mul_pos hg (exp_pos _)

lemma hasDerivAt_u (x : ℝ) : HasDerivAt (u g) (-u g x) x := by
  have := ((hasDerivAt_neg x).exp).const_mul g
  have h : u g = fun y => g * exp (-y) := rfl
  rw [h]
  convert this using 1
  simp

lemma hasDerivAt_H (x : ℝ) : HasDerivAt (H g) (1 + u g x * (1 + x)) x := by
  have h1 : HasDerivAt (fun y => y * (1 - u g y)) _ x :=
    (hasDerivAt_id' x).mul ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_u (g := g) x))
  have h2 : HasDerivAt (fun y => 2 * (1 + u g y)) _ x :=
    ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_u (g := g) x)).const_mul 2
  have := h1.sub h2
  unfold H
  convert this using 1
  ring

/-- `dC/dx = - x u H / (1 + u)^3`. -/
lemma hasDerivAt_cTwo (hg : 0 < g) (x : ℝ) :
    HasDerivAt (cTwo g) (-(x * u g x / (1 + u g x) ^ 3 * H g x)) x := by
  have hu := u_pos hg x
  have h1 : HasDerivAt (fun y => y ^ 2 * u g y) _ x :=
    (hasDerivAt_pow 2 x).mul (hasDerivAt_u (g := g) x)
  have h2 : HasDerivAt (fun y => (1 + u g y) ^ 2) _ x :=
    ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_u (g := g) x)).pow 2
  have h3 : HasDerivAt (fun y => y ^ 2 * u g y / (1 + u g y) ^ 2) _ x :=
    h1.div h2 (by positivity)
  unfold cTwo
  convert h3 using 1
  unfold H
  have : (1 + u g x) ≠ 0 := by positivity
  simp only [Nat.cast_ofNat, Pi.add_apply]
  field_simp
  ring

lemma H_zero (hg : 0 < g) : H g 0 < 0 := by
  have := u_pos hg 0
  unfold H; nlinarith

lemma H_strictMonoOn (hg : 0 < g) : StrictMonoOn (H g) (Set.Ici 0) := by
  apply strictMonoOn_of_deriv_pos (convex_Ici 0)
  · exact fun x _ => (hasDerivAt_H x).continuousAt.continuousWithinAt
  · intro x hx
    rw [interior_Ici] at hx
    rw [(hasDerivAt_H x).deriv]
    have := u_pos hg x
    have hx' : 0 < x := hx
    positivity

/-- A point where `H` is positive. -/
lemma H_pos_far (hg : 1 ≤ g) : 0 < H g (log g + 4) := by
  have hL : 0 ≤ log g := log_nonneg hg
  have hu : u g (log g + 4) = exp (-4) := by
    unfold u
    rw [neg_add, exp_add, exp_neg (log g), exp_log (by linarith)]
    field_simp
  have hq : exp (-4 : ℝ) ≤ 1 / 13 := by
    rw [exp_neg, inv_eq_one_div]
    apply one_div_le_one_div_of_le (by norm_num)
    have := quadratic_le_exp_of_nonneg (show (0:ℝ) ≤ 4 by norm_num)
    linarith
  have hqp := exp_pos (-4 : ℝ)
  unfold H
  rw [hu]
  nlinarith

lemma H_continuous : Continuous (H g) := by
  unfold H u; fun_prop

lemma exists_root (hg : 1 ≤ g) : ∃ x, 0 < x ∧ H g x = 0 := by
  have hg0 : 0 < g := by linarith
  have hL : 0 ≤ log g := log_nonneg hg
  have h0 := H_zero hg0
  have h1 := H_pos_far hg
  obtain ⟨x, hx, hx0⟩ := intermediate_value_Icc (show (0:ℝ) ≤ log g + 4 by linarith)
    H_continuous.continuousOn ⟨h0.le, h1.le⟩ |> fun h => by simpa using (h : (0:ℝ) ∈ _)
  refine ⟨x, ?_, hx0⟩
  rcases hx.1.eq_or_lt with h | h
  · rw [← h] at hx0; linarith
  · exact h

lemma root_unique (hg : 0 < g) {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hHx : H g x = 0)
    (hHy : H g y = 0) : x = y := by
  have := (H_strictMonoOn hg).injOn (Set.mem_Ici.mpr hx.le) (Set.mem_Ici.mpr hy.le)
    (hHx.trans hHy.symm)
  exact this

lemma H_neg_of_lt (hg : 0 < g) {x xs : ℝ} (hx : 0 ≤ x) (hxs : x < xs) (hH : H g xs = 0) :
    H g x < 0 := by
  have := (H_strictMonoOn hg) (Set.mem_Ici.mpr hx) (Set.mem_Ici.mpr (hx.trans hxs.le)) hxs
  linarith

lemma H_pos_of_gt (hg : 0 < g) {x xs : ℝ} (hxs0 : 0 ≤ xs) (hxs : xs < x) (hH : H g xs = 0) :
    0 < H g x := by
  have := (H_strictMonoOn hg) (Set.mem_Ici.mpr hxs0) (Set.mem_Ici.mpr (hxs0.trans hxs.le)) hxs
  linarith

/-- There is no root of `H` at or below `log g`, because there `u ≥ 1`. -/
lemma log_lt_of_root (hg : 1 ≤ g) {x : ℝ} (hx : 0 < x) (hH : H g x = 0) : log g < x := by
  by_contra hle
  have hle := not_lt.mp hle
  have hg0 : 0 < g := by linarith
  have hu1 : 1 ≤ u g x := by
    unfold u
    have : exp x ≤ exp (log g) := exp_le_exp.mpr hle
    rw [exp_log hg0] at this
    have h2 : g * exp (-x) = g / exp x := by rw [exp_neg]; ring
    rw [h2, le_div_iff₀ (exp_pos x)]
    linarith
  unfold H at hH
  nlinarith

/-- The peak equation `x* = 2 (1 + u) / (1 - u)` is `H = 0` once `u < 1`. -/
lemma u_lt_one_of_root (hg : 1 ≤ g) {x : ℝ} (hx : 0 < x) (hH : H g x = 0) : u g x < 1 := by
  have hl := log_lt_of_root hg hx hH
  have hg0 : 0 < g := by linarith
  unfold u
  have : exp (log g) < exp x := exp_lt_exp.mpr hl
  rw [exp_log hg0] at this
  have h2 : g * exp (-x) = g / exp x := by rw [exp_neg]; ring
  rw [h2, div_lt_one (exp_pos x)]
  exact this

/-- **The peak equation** (new). For every real `g ≥ 1` there is exactly one `x* > 0` with
`x* = 2 (1 + u) / (1 - u)`, it exceeds `log g` and `2`, and `C` has a strict global maximum there
as a function of `x`. -/
theorem two_level_peak (hg : 1 ≤ g) :
    ∃ xs : ℝ, 0 < xs ∧ log g < xs ∧ 2 < xs ∧ u g xs < 1 ∧
      xs = 2 * (1 + u g xs) / (1 - u g xs) ∧
      (∀ y, 0 < y → H g y = 0 → y = xs) ∧
      (∀ x, 0 < x → x ≠ xs → cTwo g x < cTwo g xs) := by
  have hg0 : 0 < g := by linarith
  obtain ⟨xs, hxs, hH⟩ := exists_root hg
  have hlt := log_lt_of_root hg hxs hH
  have hu1 := u_lt_one_of_root hg hxs hH
  have hup := u_pos hg0 xs
  refine ⟨xs, hxs, hlt, ?_, hu1, ?_, fun y hy hHy => root_unique hg0 hy hxs hHy hH, ?_⟩
  · unfold H at hH; nlinarith
  · unfold H at hH
    rw [eq_div_iff (by linarith)]
    linarith
  · intro x hx hne
    have hcont : Continuous (cTwo g) :=
      continuous_iff_continuousAt.2 fun x => (hasDerivAt_cTwo hg0 x).continuousAt
    rcases lt_or_gt_of_ne hne with hlt' | hgt
    · have hmono : StrictMonoOn (cTwo g) (Set.Icc 0 xs) := by
        apply strictMonoOn_of_deriv_pos (convex_Icc 0 xs) hcont.continuousOn
        intro y hy
        rw [interior_Icc] at hy
        rw [(hasDerivAt_cTwo hg0 y).deriv]
        have hHy := H_neg_of_lt hg0 hy.1.le hy.2 hH
        have := u_pos hg0 y
        have hy0 : 0 < y := hy.1
        have : 0 < y * u g y / (1 + u g y) ^ 3 := by positivity
        nlinarith
      exact hmono ⟨hx.le, hlt'.le⟩ ⟨hxs.le, le_refl _⟩ hlt'
    · have hanti : StrictAntiOn (cTwo g) (Set.Ici xs) := by
        apply strictAntiOn_of_deriv_neg (convex_Ici xs) hcont.continuousOn
        intro y hy
        rw [interior_Ici] at hy
        rw [(hasDerivAt_cTwo hg0 y).deriv]
        have hHy := H_pos_of_gt hg0 hxs.le hy hH
        have hy0 : 0 < y := hxs.trans hy
        have := u_pos hg0 y
        have : 0 < y * u g y / (1 + u g y) ^ 3 := by positivity
        nlinarith
      exact hanti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hgt.le) hgt

section peakvalues

variable (hg : 1 ≤ g) {xs : ℝ} (hxs : 0 < xs) (hH : H g xs = 0)
include hg hxs hH

/-- At the peak, `u = (x* - 2) / (x* + 2)`. -/
lemma u_peak : u g xs = (xs - 2) / (xs + 2) := by
  have hup := u_pos (by linarith : 0 < g) xs
  unfold H at hH
  have : 0 < xs + 2 := by linarith
  rw [eq_div_iff this.ne']
  nlinarith

/-- **Peak occupancy** (validate.py section 2): the top token has probability `1/2 + 1/x*` at the
melting temperature. -/
theorem peakOccupancy : 1 / (1 + u g xs) = 1 / 2 + 1 / xs := by
  have hup := u_pos (by linarith : 0 < g) xs
  unfold H at hH
  have hx2 : 2 < xs := by nlinarith
  rw [u_peak hg hxs hH]
  have : xs + 2 ≠ 0 := by linarith
  field_simp
  ring

/-- **Peak height** (validate.py section 2): `C_max = (x*^2 - 4) / 4`. -/
theorem peakHeight : cTwo g xs = (xs ^ 2 - 4) / 4 := by
  have hx2 : 2 < xs := by
    have hup := u_pos (by linarith : 0 < g) xs
    unfold H at hH
    nlinarith
  unfold cTwo
  rw [u_peak hg hxs hH]
  have : xs + 2 ≠ 0 := by linarith
  field_simp
  ring

omit hxs hH in
/-- `u = exp (-c)` at the peak, where `c = x* - log g`. -/
lemma u_peak_eq_exp : u g xs = exp (-(xs - log g)) := by
  unfold u
  rw [show -(xs - log g) = log g + -xs by ring, exp_add, exp_log (by linarith : 0 < g)]

/-- With `t = exp c` the peak equation reads `x* (t - 1) = 2 (t + 1)`. -/
lemma peak_exp_form : xs * (exp (xs - log g) - 1) = 2 * (exp (xs - log g) + 1) := by
  have hq := u_peak_eq_exp hg (xs := xs)
  have ht : exp (xs - log g) * exp (-(xs - log g)) = 1 := by rw [← exp_add]; simp
  unfold H at hH
  rw [hq] at hH
  set t := exp (xs - log g)
  set q := exp (-(xs - log g))
  have h0 : xs * (1 - q) = 2 * (1 + q) := by linarith
  have : xs * (1 - q) * t = 2 * (1 + q) * t := by rw [h0]
  nlinarith

/-- **Lower bound on the offset** (new): `c (log g + c) > 4`, that is `c x* > 4`. -/
theorem c_lower : 4 < (xs - log g) * xs := by
  have hlt := log_lt_of_root hg hxs hH
  have hc : 0 < xs - log g := by linarith
  have hte := peak_exp_form hg hxs hH
  set t := exp (xs - log g) with ht
  have ht1 : 1 < t := by
    have := exp_lt_exp.mpr hc
    rw [exp_zero] at this; exact this
  have key := lt_log_one_add_of_pos (show 0 < t - 1 by linarith)
  rw [show 1 + (t - 1) = t by ring, ht, log_exp] at key
  rw [← ht] at key
  have hxs2 : 2 * (t - 1) / (t - 1 + 2) = 4 / xs := by
    rw [div_eq_div_iff (by linarith) hxs.ne']
    nlinarith
  rw [hxs2, div_lt_iff₀ hxs] at key
  linarith

/-- **Upper bound on the offset** (new): `c x* ≤ 2 c + 4`. -/
theorem c_upper : (xs - log g) * xs ≤ 2 * (xs - log g) + 4 := by
  have hlt := log_lt_of_root hg hxs hH
  have hc : 0 < xs - log g := by linarith
  have hte := peak_exp_form hg hxs hH
  have h1 := add_one_le_exp (xs - log g)
  set t := exp (xs - log g) with ht
  have ht1 : 1 < t := by
    have := exp_lt_exp.mpr hc
    rw [exp_zero] at this; exact this
  have h2 : xs = 2 * (t + 1) / (t - 1) := by
    rw [eq_div_iff (by linarith)]; linarith
  have : (xs - log g) * xs ≤ 2 * (xs - log g) + 4 := by
    rw [h2, mul_div_assoc', div_le_iff₀ (by linarith)]
    nlinarith
  exact this

/-- The offset is positive and bounded by `c^2 ≤ 2 c + 4` (new), so `c < 1 + √5` for every `g ≥ 1`. -/
theorem c_pos_sq_le : 0 < xs - log g ∧ (xs - log g) ^ 2 ≤ 2 * (xs - log g) + 4 := by
  have hlt := log_lt_of_root hg hxs hH
  have hL : 0 ≤ log g := log_nonneg hg
  have hc : 0 < xs - log g := by linarith
  have := c_upper hg hxs hH
  refine ⟨hc, ?_⟩
  nlinarith

end peakvalues

end TwoLevel

/-- The two-level energy vector: one ground state at `0` and `g` states at `Δ`. -/
def twoLevelE (g : ℕ) (Δ : ℝ) : Fin (g + 1) → ℝ := Fin.cons 0 (fun _ => Δ)

lemma Z_twoLevel (g : ℕ) (Δ T : ℝ) :
    Z (twoLevelE g Δ) T = 1 + g * exp (-Δ / T) := by
  simp [Z, twoLevelE, Fin.sum_univ_succ]

/-- The general heat capacity `Var(E) / T^2` of the energy vector `twoLevelE g Δ` equals the
closed form `x^2 u / (1 + u)^2` with `x = Δ / T` and `u = g e^{-x}`. -/
theorem C_twoLevel (g : ℕ) (Δ : ℝ) {T : ℝ} (hT : T ≠ 0) :
    C (twoLevelE g Δ) T = TwoLevel.cTwo (g : ℝ) (Δ / T) := by
  have hZ := Z_twoLevel g Δ T
  have hZp : (0 : ℝ) < 1 + g * exp (-Δ / T) := by positivity
  have hneg : -(Δ / T) = -Δ / T := by ring
  unfold C
  rw [var_eq]
  simp only [U, prob, hZ]
  simp only [TwoLevel.cTwo, TwoLevel.u, hneg, twoLevelE, Fin.sum_univ_succ,
    Fin.cons_zero, Fin.cons_succ, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  field_simp
  ring

/-- **The melting temperature of the two-level model** (new). For `g ≥ 1` and `Δ > 0` there is an
`x* > log g` with `x* = 2 (1 + u) / (1 - u)` such that `T_melt = Δ / x*` is the unique maximiser of
the heat capacity of `twoLevelE g Δ` over all `T > 0`. -/
theorem two_level_melting (g : ℕ) (hg : 1 ≤ g) {Δ : ℝ} (hΔ : 0 < Δ) :
    ∃ xs : ℝ, 0 < xs ∧ Real.log g < xs ∧ 2 < xs ∧
      xs = 2 * (1 + TwoLevel.u g xs) / (1 - TwoLevel.u g xs) ∧
      ∀ T : ℝ, 0 < T → T ≠ Δ / xs →
        C (twoLevelE g Δ) T < C (twoLevelE g Δ) (Δ / xs) := by
  have hg' : (1 : ℝ) ≤ g := by exact_mod_cast hg
  obtain ⟨xs, hxs, hlog, h2, -, heq, -, hmax⟩ := TwoLevel.two_level_peak hg'
  refine ⟨xs, hxs, hlog, h2, heq, fun T hT hne => ?_⟩
  have hTm : Δ / xs ≠ 0 := (div_pos hΔ hxs).ne'
  rw [C_twoLevel g Δ hT.ne', C_twoLevel g Δ hTm]
  have : Δ / (Δ / xs) = xs := by field_simp
  rw [this]
  apply hmax _ (div_pos hΔ hT)
  intro h
  apply hne
  field_simp at h ⊢
  linarith

end TokenThermo
