/-
Token Thermodynamics: the finite Boltzmann model of a next-token distribution.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Field

/-!
# The Boltzmann form of softmax

A next-token distribution over a finite vocabulary `ι` has logits `z : ι → ℝ`. The paper sets the
energy of token `i` to `E i = - z i` and reads sampling at temperature `T` as the Boltzmann
distribution `p i = exp (-E i / T) / Z T`.

These definitions are our own. They do not use the canonical ensemble of Physlib, whose
construction is measure theoretic and carries the Boltzmann constant, a phase space unit and a
`Temperature` type. `TokenThermo.Bridge` relates the two for the case of finitely many microstates.

All quantities are in nats and the Boltzmann constant is 1.
-/

open Finset Real

namespace TokenThermo

variable {ι : Type*} [Fintype ι]

/-- The partition function `Z(T) = ∑ exp (-E_i / T)`. -/
noncomputable def Z (E : ι → ℝ) (T : ℝ) : ℝ := ∑ i, exp (-E i / T)

/-- The Boltzmann probability `p_i(T) = exp (-E_i / T) / Z(T)`. -/
noncomputable def prob (E : ι → ℝ) (T : ℝ) (i : ι) : ℝ := exp (-E i / T) / Z E T

/-- The internal energy `U(T) = ∑ p_i E_i`. -/
noncomputable def U (E : ι → ℝ) (T : ℝ) : ℝ := ∑ i, prob E T i * E i

/-- The energy variance `Var_p(E) = ∑ p_i (E_i - U)^2`. -/
noncomputable def var (E : ι → ℝ) (T : ℝ) : ℝ := ∑ i, prob E T i * (E i - U E T) ^ 2

/-- The Shannon entropy `S(T) = - ∑ p_i log p_i` in nats. -/
noncomputable def S (E : ι → ℝ) (T : ℝ) : ℝ := -∑ i, prob E T i * log (prob E T i)

/-- The heat capacity `C(T) = Var_p(E) / T^2`. -/
noncomputable def C (E : ι → ℝ) (T : ℝ) : ℝ := var E T / T ^ 2

/-- The free energy `F(T) = - T log Z(T)`. -/
noncomputable def F (E : ι → ℝ) (T : ℝ) : ℝ := -T * log (Z E T)

/-- The softmax of a logit vector. -/
noncomputable def softmax (z : ι → ℝ) (i : ι) : ℝ := exp (z i) / ∑ j, exp (z j)

/-- The predictive entropy of a probability vector, `- ∑ p_i log p_i`. -/
noncomputable def predEntropy (p : ι → ℝ) : ℝ := -∑ i, p i * log (p i)

/-- The varentropy of a probability vector, the variance of the surprisal `- log p_i`. -/
noncomputable def varentropy (p : ι → ℝ) : ℝ :=
  ∑ i, p i * (-log (p i) - predEntropy p) ^ 2

section basic

variable [Nonempty ι] (E : ι → ℝ) {T : ℝ}

lemma Z_pos (T : ℝ) : 0 < Z E T :=
  Finset.sum_pos (fun _ _ => exp_pos _) Finset.univ_nonempty

lemma prob_pos (T : ℝ) (i : ι) : 0 < prob E T i := div_pos (exp_pos _) (Z_pos E T)

lemma sum_prob (T : ℝ) : ∑ i, prob E T i = 1 := by
  unfold prob
  rw [← Finset.sum_div]
  exact div_self (Z_pos E T).ne'

lemma prob_lt_one_of_ne (T : ℝ) {i j : ι} (h : i ≠ j) : prob E T i < 1 := by
  classical
  have h1 : prob E T i + prob E T j ≤ 1 := by
    rw [← sum_prob E T]
    calc prob E T i + prob E T j = ∑ k ∈ ({i, j} : Finset ι), prob E T k := by
          rw [Finset.sum_pair h]
      _ ≤ ∑ k, prob E T k :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
            (fun k _ _ => (prob_pos E T k).le)
  linarith [prob_pos E T j]

omit [Nonempty ι] in
/-- The Boltzmann form of softmax: softmax of the logits `z / T` is the Boltzmann distribution
with energies `-z`. -/
theorem softmax_eq_prob (z : ι → ℝ) (T : ℝ) (i : ι) :
    softmax (fun j => z j / T) i = prob (fun j => -z j) T i := by
  simp [softmax, prob, Z, div_eq_mul_inv]

/-- The logarithm of the Boltzmann probability is `-E_i / T - log Z`. -/
lemma log_prob (T : ℝ) (i : ι) : log (prob E T i) = -E i / T - log (Z E T) := by
  unfold prob
  rw [log_div (exp_pos _).ne' (Z_pos E T).ne', log_exp]

/-- The internal energy shifted by `log Z` is the entropy: `S = log Z + U / T`. -/
theorem S_eq (T : ℝ) : S E T = log (Z E T) + U E T / T := by
  unfold S
  simp_rw [log_prob, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, sum_prob]
  unfold U
  rw [Finset.sum_div]
  have : ∀ i, prob E T i * (-E i / T) = -(prob E T i * E i / T) := fun i => by ring
  simp_rw [this, Finset.sum_neg_distrib]
  ring

/-- The free energy is `F = U - T S` for `T ≠ 0`. -/
theorem F_eq_U_sub_TS (hT : T ≠ 0) : F E T = U E T - T * S E T := by
  rw [S_eq]
  unfold F
  field_simp
  ring

/-- The variance is the mean square minus the square of the mean. -/
lemma var_eq (T : ℝ) :
    var E T = ∑ i, prob E T i * E i ^ 2 - U E T ^ 2 := by
  unfold var
  have : ∀ i, prob E T i * (E i - U E T) ^ 2 =
      prob E T i * E i ^ 2 - 2 * U E T * (prob E T i * E i) + U E T ^ 2 * prob E T i :=
    fun i => by ring
  simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, sum_prob]
  change _ - 2 * U E T * U E T + U E T ^ 2 * 1 = _
  ring

/-- The heat capacity is non-negative at positive temperature. -/
theorem C_nonneg (T : ℝ) : 0 ≤ C E T :=
  div_nonneg (Finset.sum_nonneg fun i _ => mul_nonneg (prob_pos E T i).le (sq_nonneg _))
    (sq_nonneg _)

/-- The heat capacity vanishes exactly when all energies are equal. -/
theorem C_eq_zero_iff (hT : T ≠ 0) : C E T = 0 ↔ ∀ i j, E i = E j := by
  unfold C
  rw [div_eq_zero_iff, or_iff_left (pow_ne_zero 2 hT)]
  unfold var
  rw [Finset.sum_eq_zero_iff_of_nonneg
    (fun i _ => mul_nonneg (prob_pos E T i).le (sq_nonneg _))]
  constructor
  · intro h i j
    have hi := h i (Finset.mem_univ i)
    have hj := h j (Finset.mem_univ j)
    rcases mul_eq_zero.mp hi with h1 | h1
    · exact absurd h1 (prob_pos E T i).ne'
    rcases mul_eq_zero.mp hj with h2 | h2
    · exact absurd h2 (prob_pos E T j).ne'
    have := pow_eq_zero_iff (two_ne_zero) |>.mp h1
    have := pow_eq_zero_iff (two_ne_zero) |>.mp h2
    linarith
  · intro h i _
    have hU : U E T = E i := by
      unfold U
      have : ∀ k, prob E T k * E k = prob E T k * E i := fun k => by rw [h k i]
      simp_rw [this, ← Finset.sum_mul, sum_prob, one_mul]
    rw [hU]; simp

/-- Strictly positive heat capacity when the energies are not all equal. -/
theorem C_pos (hT : T ≠ 0) (h : ∃ i j, E i ≠ E j) : 0 < C E T := by
  refine lt_of_le_of_ne (C_nonneg E T) (fun h0 => ?_)
  obtain ⟨i, j, hij⟩ := h
  exact hij ((C_eq_zero_iff E hT).mp h0.symm i j)

/-- The entropy is strictly positive when there are at least two tokens. -/
theorem S_pos (h : ∃ i j : ι, i ≠ j) (T : ℝ) : 0 < S E T := by
  obtain ⟨i, j, hij⟩ := h
  unfold S
  rw [neg_eq_neg_one_mul, Finset.mul_sum]
  have key : ∀ k, 0 ≤ -1 * (prob E T k * log (prob E T k)) := fun k => by
    have h1 := prob_pos E T k
    have h2 : prob E T k ≤ 1 := by
      rw [← sum_prob E T]
      exact Finset.single_le_sum (f := fun k => prob E T k)
        (fun k _ => (prob_pos E T k).le) (Finset.mem_univ k)
    have := log_nonpos h1.le h2
    nlinarith
  refine lt_of_lt_of_le ?_ (Finset.single_le_sum (fun k _ => key k) (Finset.mem_univ i))
  have h1 := prob_pos E T i
  have h2 := prob_lt_one_of_ne E T hij
  have := log_neg h1 h2
  nlinarith

end basic

/-- At `T = 1` the Boltzmann entropy is the predictive entropy of the softmax of the logits. -/
theorem S_one_eq_predEntropy [Nonempty ι] (z : ι → ℝ) :
    S (fun i => -z i) 1 = predEntropy (softmax z) := by
  have h : ∀ i, softmax z i = prob (fun j => -z j) 1 i := fun i => by
    have := softmax_eq_prob z 1 i
    simpa using this
  simp only [S, predEntropy, h]

/-- At `T = 1` the heat capacity is the varentropy of the softmax of the logits. -/
theorem C_one_eq_varentropy [Nonempty ι] (z : ι → ℝ) :
    C (fun i => -z i) 1 = varentropy (softmax z) := by
  set E : ι → ℝ := fun i => -z i with hE
  have h : ∀ i, softmax z i = prob E 1 i := fun i => by
    have := softmax_eq_prob z 1 i
    simpa using this
  have hH : predEntropy (softmax z) = log (Z E 1) + U E 1 := by
    have := S_eq E 1
    simp only [div_one] at this
    rw [← this]
    simp only [S, predEntropy, h]
  unfold varentropy
  simp_rw [h, hH, log_prob, div_one]
  unfold C var
  simp only [one_pow, div_one]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

end TokenThermo
