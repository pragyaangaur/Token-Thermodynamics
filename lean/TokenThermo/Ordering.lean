/-
Token Thermodynamics: every entropy turning point lies on a rising flank of the heat capacity.
-/
import TokenThermo.Deriv

/-!
# The ordering theorem (FINDINGS.md section 7m)

Because `S'(T) = C(T) / T`, each of the three natural ways of writing an inflection of the entropy
curve reduces to a condition of the form `T C' / C = R` with `R > 0`:

* `S` against `T`: `S'' = 0` gives `T C' / C = 1`.
* `log S` against `log T`: the second derivative vanishes exactly when `T C' / C = C / S`.
* `log S` against `T`: the second derivative vanishes exactly when `T C' / C = 1 + C / S`.

Since `C > 0` and `S > 0` for a non-constant energy vector, `C' > 0` at every such point. The
statement is local. It says every turning point lies on a rising flank of `C`. That the turning
point lies below the global maximum of `C` needs the extra assumption that `C` is falling beyond
its peak, which is `turning_point_lt_of_decreasing_after`.
-/

open Finset Real

namespace TokenThermo

variable {ι : Type*} [Fintype ι] [Nonempty ι] (E : ι → ℝ)

omit [Fintype ι] [Nonempty ι] in
lemma exists_ne_of_energies (h : ∃ i j, E i ≠ E j) : ∃ i j : ι, i ≠ j := by
  obtain ⟨i, j, hij⟩ := h
  exact ⟨i, j, fun h' => hij (by rw [h'])⟩

/-- `S''(T) = (T C'(T) - C(T)) / T^2`. -/
theorem deriv_deriv_S {T : ℝ} (hT : 0 < T) :
    deriv (deriv (S E)) T = (deriv (C E) T * T - C E T) / T ^ 2 := by
  have hev : deriv (S E) =ᶠ[nhds T] fun t => C E t / t := by
    filter_upwards [Ioi_mem_nhds hT] with t ht
    exact (hasDerivAt_S E (ne_of_gt ht)).deriv
  rw [hev.deriv_eq]
  have h : HasDerivAt (fun t => C E t / t) _ T :=
    (hasDerivAt_C E hT.ne').div (hasDerivAt_id' T) hT.ne'
  rw [h.deriv, (hasDerivAt_C E hT.ne').deriv]
  simp

/-- The derivative of `u ↦ log S (exp u)`. -/
lemma hasDerivAt_logS_exp (hS : ∀ T, 0 < S E T) (u : ℝ) :
    HasDerivAt (fun u => log (S E (exp u))) (C E (exp u) / S E (exp u)) u := by
  have h := ((hasDerivAt_S E (exp_pos u).ne').comp u (hasDerivAt_exp u)).log (hS _).ne'
  convert h using 1
  have : exp u ≠ 0 := (exp_pos u).ne'
  simp only [Function.comp]
  field_simp
  rfl

/-- The second derivative of `log S` against `log T`. -/
theorem deriv_deriv_logS_exp (hS : ∀ T, 0 < S E T) {T : ℝ} (hT : 0 < T) :
    deriv (deriv (fun u => log (S E (exp u)))) (log T) =
      (deriv (C E) T * T * S E T - C E T ^ 2) / S E T ^ 2 := by
  have hfun : deriv (fun u => log (S E (exp u))) = fun u => C E (exp u) / S E (exp u) :=
    funext fun u => (hasDerivAt_logS_exp E hS u).deriv
  rw [hfun]
  have hu : exp (log T) = T := exp_log hT
  have hC := (hasDerivAt_C E (exp_pos (log T)).ne').comp (log T) (hasDerivAt_exp (log T))
  have hS' := (hasDerivAt_S E (exp_pos (log T)).ne').comp (log T) (hasDerivAt_exp (log T))
  have h : HasDerivAt (fun u => C E (exp u) / S E (exp u)) _ (log T) :=
    hC.div hS' (hS _).ne'
  rw [h.deriv, (hasDerivAt_C E hT.ne').deriv]
  simp only [Function.comp, hu]
  have hTne : T ≠ 0 := hT.ne'
  field_simp

/-- The second derivative of `log S` against `T`. -/
theorem deriv_deriv_logS {T : ℝ} (hS : ∀ T, 0 < S E T) (hT : 0 < T) :
    deriv (deriv (fun t => log (S E t))) T =
      (deriv (C E) T * (T * S E T) - C E T * (S E T + C E T)) / (T * S E T) ^ 2 := by
  have hd : ∀ t, 0 < t → HasDerivAt (fun t => log (S E t)) (C E t / (t * S E t)) t := by
    intro t ht
    have := (hasDerivAt_S E ht.ne').log (hS t).ne'
    convert this using 1
    field_simp
  have hev : deriv (fun t => log (S E t)) =ᶠ[nhds T] fun t => C E t / (t * S E t) := by
    filter_upwards [Ioi_mem_nhds hT] with t ht using (hd t ht).deriv
  rw [hev.deriv_eq]
  have hden : HasDerivAt (fun t => t * S E t) (1 * S E T + T * (C E T / T)) T :=
    (hasDerivAt_id' T).mul (hasDerivAt_S E hT.ne')
  have h : HasDerivAt (fun t => C E t / (t * S E t)) _ T :=
    (hasDerivAt_C E hT.ne').div hden (mul_pos hT (hS T)).ne'
  rw [h.deriv, (hasDerivAt_C E hT.ne').deriv]
  have hTne : T ≠ 0 := hT.ne'
  field_simp

section ordering

variable (hne : ∃ i j, E i ≠ E j)
include hne

lemma S_pos' (T : ℝ) : 0 < S E T := S_pos E (exists_ne_of_energies E hne) T

/-- First convention: an inflection of `S` against `T` has `T C' / C = 1` and `C' > 0`. -/
theorem ordering_linear {T : ℝ} (hT : 0 < T) (h : deriv (deriv (S E)) T = 0) :
    T * deriv (C E) T / C E T = 1 ∧ 0 < deriv (C E) T := by
  have hC := C_pos E hT.ne' hne
  rw [deriv_deriv_S E hT] at h
  have h' : deriv (C E) T * T = C E T := by
    have : deriv (C E) T * T - C E T = 0 := by
      have := (div_eq_zero_iff.mp h).resolve_right (pow_ne_zero 2 hT.ne')
      exact this
    linarith
  refine ⟨?_, ?_⟩
  · rw [mul_comm, h', div_self hC.ne']
  · by_contra hneg
    have hneg := not_lt.mp hneg
    nlinarith

/-- Second convention: `log S` against `log T` has `T C' / C = C / S` and `C' > 0`. -/
theorem ordering_loglog {T : ℝ} (hT : 0 < T)
    (h : deriv (deriv (fun u => log (S E (exp u)))) (log T) = 0) :
    T * deriv (C E) T / C E T = C E T / S E T ∧ 0 < deriv (C E) T := by
  have hC := C_pos E hT.ne' hne
  have hS := S_pos' E hne
  rw [deriv_deriv_logS_exp E hS hT] at h
  have hSp := hS T
  have h' : deriv (C E) T * T * S E T = C E T ^ 2 := by
    have := (div_eq_zero_iff.mp h).resolve_right (pow_ne_zero 2 hSp.ne')
    linarith
  refine ⟨?_, ?_⟩
  · field_simp
    linarith
  · by_contra hneg
    have hneg := not_lt.mp hneg
    nlinarith [mul_pos hT hSp, sq_pos_of_pos hC]

/-- Third convention: `log S` against `T` has `T C' / C = 1 + C / S` and `C' > 0`. -/
theorem ordering_logS_linear {T : ℝ} (hT : 0 < T)
    (h : deriv (deriv (fun t => log (S E t))) T = 0) :
    T * deriv (C E) T / C E T = 1 + C E T / S E T ∧ 0 < deriv (C E) T := by
  have hC := C_pos E hT.ne' hne
  have hS := S_pos' E hne
  have hSp := hS T
  rw [deriv_deriv_logS E hS hT] at h
  have h' : deriv (C E) T * (T * S E T) = C E T * (S E T + C E T) := by
    have := (div_eq_zero_iff.mp h).resolve_right (pow_ne_zero 2 (mul_pos hT hSp).ne')
    linarith
  refine ⟨?_, ?_⟩
  · field_simp
    linarith
  · by_contra hneg
    have hneg := not_lt.mp hneg
    nlinarith [mul_pos hT hSp, mul_pos hC hSp, sq_pos_of_pos hC]

/-- Any point with `C' = 0` is not a turning point of the first convention. -/
theorem not_turning_of_critical {T : ℝ} (hT : 0 < T) (hc : deriv (C E) T = 0) :
    deriv (deriv (S E)) T ≠ 0 := fun h => by
  have := (ordering_linear E hne hT h).2
  linarith

/-- If `C` is strictly falling beyond a critical point `Tm`, every turning point in the first
convention lies below `Tm`. -/
theorem turning_point_lt_of_decreasing_after {T Tm : ℝ} (hT : 0 < T)
    (hdec : ∀ t, Tm ≤ t → deriv (C E) t ≤ 0)
    (h : deriv (deriv (S E)) T = 0) : T < Tm := by
  by_contra hge
  have hge := not_lt.mp hge
  have := (ordering_linear E hne hT h).2
  have := hdec T hge
  linarith

end ordering

end TokenThermo
