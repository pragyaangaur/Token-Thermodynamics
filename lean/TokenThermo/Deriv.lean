/-
Token Thermodynamics: temperature derivatives of the Boltzmann quantities.
-/
import TokenThermo.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Derivatives in the temperature

For a finite energy vector `E` and `T ≠ 0` we prove

* `dU/dT = C`
* `dS/dT = C / T`
* `dF/dT = -S`
* `d(log Z)/dT = U / T^2`
* `dC/dT = μ₃ / T^4 - 2 Var / T^3`, where `μ₃` is the third central moment of the energy.

The unnormalised moments `m_k(T) = ∑ exp (-E_i/T) E_i^k` satisfy `m_k' = m_{k+1} / T^2`, and every
statement follows from this one derivative by the quotient rule.
-/

open Finset Real

namespace TokenThermo

variable {ι : Type*} [Fintype ι] [Nonempty ι] (E : ι → ℝ)

/-- The unnormalised moment `m_k(T) = ∑ exp (-E_i / T) E_i^k`. -/
noncomputable def m (k : ℕ) (T : ℝ) : ℝ := ∑ i, exp (-E i / T) * E i ^ k

/-- The normalised moment `⟨E^k⟩ = m_k / m_0`. -/
noncomputable def r (k : ℕ) (T : ℝ) : ℝ := m E k T / m E 0 T

omit [Nonempty ι] in
lemma m_zero (T : ℝ) : m E 0 T = Z E T := by simp [m, Z]

omit [Nonempty ι] in
lemma r_eq (k : ℕ) (T : ℝ) : r E k T = ∑ i, prob E T i * E i ^ k := by
  unfold r prob
  rw [m_zero]
  simp_rw [div_mul_eq_mul_div]
  rw [← Finset.sum_div]
  rfl

omit [Nonempty ι] in
lemma U_eq_r (T : ℝ) : U E T = r E 1 T := by simp [r_eq, U]

lemma var_eq_r (T : ℝ) : var E T = r E 2 T - r E 1 T ^ 2 := by
  rw [var_eq, ← U_eq_r, r_eq]

omit [Nonempty ι] in
lemma hasDerivAt_m (k : ℕ) {T : ℝ} (hT : T ≠ 0) :
    HasDerivAt (m E k) (m E (k + 1) T / T ^ 2) T := by
  unfold m
  have h : ∀ i ∈ (Finset.univ : Finset ι),
      HasDerivAt (fun T => exp (-E i / T) * E i ^ k)
        (exp (-E i / T) * E i ^ (k + 1) / T ^ 2) T := by
    intro i _
    have h1 : HasDerivAt (fun T : ℝ => -E i / T) (E i / T ^ 2) T := by
      have := (hasDerivAt_inv hT).const_mul (-E i)
      convert this using 1
      · funext t; simp [div_eq_mul_inv]
      · field_simp
    have := (h1.exp).mul_const (E i ^ k)
    convert this using 1
    field_simp
    ring
  have := HasDerivAt.sum h
  convert this using 1
  · funext t; simp
  · rw [Finset.sum_div]

lemma hasDerivAt_r (k : ℕ) {T : ℝ} (hT : T ≠ 0) :
    HasDerivAt (r E k) ((r E (k + 1) T - r E k T * r E 1 T) / T ^ 2) T := by
  have h0 : m E 0 T ≠ 0 := by rw [m_zero]; exact (Z_pos E T).ne'
  have := (hasDerivAt_m E k hT).div (hasDerivAt_m E 0 hT) h0
  unfold r
  convert this using 1
  have h1 : m E (0 + 1) T = m E 1 T := rfl
  rw [h1]
  field_simp

/-- `dU/dT = C`. -/
theorem hasDerivAt_U {T : ℝ} (hT : T ≠ 0) : HasDerivAt (U E) (C E T) T := by
  have := hasDerivAt_r E 1 hT
  have hU : U E = r E 1 := funext fun t => U_eq_r E t
  rw [hU]
  convert this using 1
  unfold C
  rw [var_eq_r]
  ring

/-- `d(log Z)/dT = U / T^2`. -/
theorem hasDerivAt_logZ {T : ℝ} (hT : T ≠ 0) :
    HasDerivAt (fun t => log (Z E t)) (U E T / T ^ 2) T := by
  have h := hasDerivAt_m E 0 hT
  have hZ : (fun t => Z E t) = m E 0 := funext fun t => (m_zero E t).symm
  have := (h.log (by rw [m_zero]; exact (Z_pos E T).ne'))
  convert this using 1
  · funext t; rw [m_zero]
  · rw [m_zero, U_eq_r]
    unfold r
    rw [m_zero]
    simp [m]
    field_simp

/-- `dS/dT = C / T`. -/
theorem hasDerivAt_S {T : ℝ} (hT : T ≠ 0) : HasDerivAt (S E) (C E T / T) T := by
  have h1 := hasDerivAt_logZ E hT
  have h2 := (hasDerivAt_U E hT).div (hasDerivAt_id' T) hT
  have := h1.add h2
  have hS : S E = fun t => log (Z E t) + U E t / t := funext fun t => S_eq E t
  rw [hS]
  convert this using 1
  field_simp
  ring

/-- `dF/dT = -S`. -/
theorem hasDerivAt_F {T : ℝ} (hT : T ≠ 0) : HasDerivAt (F E) (-S E T) T := by
  have h1 := (hasDerivAt_id' T).neg.mul (hasDerivAt_logZ E hT)
  have hF : F E = fun t => -t * log (Z E t) := by
    funext t; simp [F]
  rw [hF]
  convert h1 using 1
  rw [S_eq]
  simp only [Pi.neg_apply]
  field_simp
  ring

/-- The third central moment of the energy under the Boltzmann distribution. -/
noncomputable def mu3 (T : ℝ) : ℝ := ∑ i, prob E T i * (E i - U E T) ^ 3

lemma mu3_eq_r (T : ℝ) : mu3 E T = r E 3 T - 3 * r E 1 T * r E 2 T + 2 * r E 1 T ^ 3 := by
  unfold mu3
  have : ∀ i, prob E T i * (E i - U E T) ^ 3 =
      prob E T i * E i ^ 3 - 3 * U E T * (prob E T i * E i ^ 2)
        + 3 * U E T ^ 2 * (prob E T i * E i) - U E T ^ 3 * prob E T i :=
    fun i => by ring
  simp_rw [this, Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  simp_rw [← Finset.mul_sum, sum_prob]
  have h1 : ∑ i, prob E T i * E i = r E 1 T := by simp [r_eq]
  have h2 : ∑ i, prob E T i * E i ^ 2 = r E 2 T := (r_eq E 2 T).symm
  have h3 : ∑ i, prob E T i * E i ^ 3 = r E 3 T := (r_eq E 3 T).symm
  rw [h1, h2, h3, U_eq_r]
  ring

/-- `d Var / dT = μ₃ / T^2`. -/
theorem hasDerivAt_var {T : ℝ} (hT : T ≠ 0) : HasDerivAt (var E) (mu3 E T / T ^ 2) T := by
  have h2 := hasDerivAt_r E 2 hT
  have h1 := (hasDerivAt_r E 1 hT).pow 2
  have := h2.sub h1
  have hv : var E = fun t => r E 2 t - r E 1 t ^ 2 := funext fun t => var_eq_r E t
  rw [hv]
  convert this using 1
  rw [mu3_eq_r]
  field_simp
  ring

/-- `dC/dT = μ₃ / T^4 - 2 Var / T^3`. -/
theorem hasDerivAt_C {T : ℝ} (hT : T ≠ 0) :
    HasDerivAt (C E) (mu3 E T / T ^ 4 - 2 * var E T / T ^ 3) T := by
  have h := (hasDerivAt_var E hT).div ((hasDerivAt_pow 2 T)) (pow_ne_zero 2 hT)
  have hC : C E = fun t => var E t / t ^ 2 := rfl
  rw [hC]
  convert h using 1
  field_simp
  ring

end TokenThermo
