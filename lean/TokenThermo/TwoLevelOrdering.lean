/-
Token Thermodynamics: in the two-level model every turning point lies below the melting temperature.
-/
import TokenThermo.Ordering
import TokenThermo.TwoLevel

/-!
# Turning points of the two-level model lie below `T_melt`

`TokenThermo.Ordering` shows that every entropy turning point lies where `C' > 0`. For a general
energy vector that does not place the turning point below the global maximum of `C`, because `C`
can have more than one peak. For the two-level model `C` has one peak (`two_level_peak`), so it
does. This is new: `FINDINGS.md` section 7m states it only as an observation on synthetic spectra.
-/

open Real

namespace TokenThermo

open TwoLevel

variable (g : ℕ) (Δ : ℝ)

lemma twoLevelE_nonconstant (hg : 1 ≤ g) (hΔ : Δ ≠ 0) :
    ∃ i j : Fin (g + 1), twoLevelE g Δ i ≠ twoLevelE g Δ j := by
  refine ⟨0, Fin.succ ⟨0, hg⟩, ?_⟩
  simp only [twoLevelE, Fin.cons_zero, Fin.cons_succ]
  exact hΔ.symm

/-- Above `T_melt` the heat capacity of the two-level model does not increase. -/
lemma deriv_C_twoLevel_nonpos (hg : 1 ≤ g) {Δ : ℝ} (hΔ : 0 < Δ) {xs : ℝ} (hxs : 0 < xs)
    (hH : H (g : ℝ) xs = 0) {t : ℝ} (ht : Δ / xs ≤ t) : deriv (C (twoLevelE g Δ)) t ≤ 0 := by
  have hg' : (1 : ℝ) ≤ g := by exact_mod_cast hg
  have hg0 : (0 : ℝ) < g := by linarith
  have htpos : 0 < t := lt_of_lt_of_le (div_pos hΔ hxs) ht
  have hev : C (twoLevelE g Δ) =ᶠ[nhds t] fun s => cTwo (g : ℝ) (Δ / s) := by
    filter_upwards [Ioi_mem_nhds htpos] with s hs
    exact C_twoLevel g Δ (ne_of_gt hs)
  rw [hev.deriv_eq]
  have hx : HasDerivAt (fun s : ℝ => Δ / s) (-Δ / t ^ 2) t := by
    have := (hasDerivAt_inv htpos.ne').const_mul Δ
    convert this using 1
    · funext s; simp [div_eq_mul_inv]
    · field_simp
  have hcomp := (hasDerivAt_cTwo hg0 (Δ / t)).comp t hx
  rw [show (fun s => cTwo (g : ℝ) (Δ / s)) = cTwo (g : ℝ) ∘ fun s => Δ / s from rfl, hcomp.deriv]
  have hxt : Δ / t ≤ xs := by
    rw [div_le_iff₀ htpos]
    have := (div_le_iff₀ hxs).mp (show Δ / xs ≤ t from ht)
    nlinarith
  have hup := u_pos hg0 (Δ / t)
  have hHx : H (g : ℝ) (Δ / t) ≤ 0 := by
    rcases hxt.lt_or_eq with h | h
    · exact (H_neg_of_lt hg0 (div_pos hΔ htpos).le h hH).le
    · rw [h]; exact hH.le
  have hxp : 0 < Δ / t := div_pos hΔ htpos
  have hfac : 0 ≤ Δ / t * u (g : ℝ) (Δ / t) / (1 + u (g : ℝ) (Δ / t)) ^ 3 := by positivity
  have h1 : -(Δ / t * u (g : ℝ) (Δ / t) / (1 + u (g : ℝ) (Δ / t)) ^ 3 * H (g : ℝ) (Δ / t)) ≥ 0 := by
    nlinarith
  have h2 : -Δ / t ^ 2 < 0 := by
    have : 0 < Δ / t ^ 2 := by positivity
    rw [neg_div]; linarith
  nlinarith

/-- **New.** In the two-level model every inflection of `S` against `T` lies strictly below
`T_melt = Δ / x*`, for each of the three conventions. -/
theorem two_level_turning_lt_melt (hg : 1 ≤ g) {Δ : ℝ} (hΔ : 0 < Δ) :
    ∃ xs : ℝ, 0 < xs ∧
      (∀ T : ℝ, 0 < T → deriv (deriv (S (twoLevelE g Δ))) T = 0 → T < Δ / xs) ∧
      (∀ T : ℝ, 0 < T →
        deriv (deriv (fun u => Real.log (S (twoLevelE g Δ) (Real.exp u)))) (Real.log T) = 0 →
          T < Δ / xs) ∧
      (∀ T : ℝ, 0 < T →
        deriv (deriv (fun t => Real.log (S (twoLevelE g Δ) t))) T = 0 → T < Δ / xs) := by
  have hg' : (1 : ℝ) ≤ g := by exact_mod_cast hg
  obtain ⟨xs, hxs, hH, -⟩ := two_level_peak hg'
  have hne := twoLevelE_nonconstant g Δ hg hΔ.ne'
  have key : ∀ T : ℝ, 0 < T → 0 < deriv (C (twoLevelE g Δ)) T → T < Δ / xs := by
    intro T hT hpos
    by_contra hge
    have hge := not_lt.mp hge
    have := deriv_C_twoLevel_nonpos g hg hΔ hxs hH hge
    linarith
  refine ⟨xs, hxs, fun T hT h => key T hT (ordering_linear _ hne hT h).2,
    fun T hT h => key T hT (ordering_loglog _ hne hT h).2,
    fun T hT h => key T hT (ordering_logS_linear _ hne hT h).2⟩

end TokenThermo
