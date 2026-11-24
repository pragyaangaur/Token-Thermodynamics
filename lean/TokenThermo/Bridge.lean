/-
Token Thermodynamics: agreement with Physlib's canonical ensemble.
-/
import TokenThermo.Basic
import Physlib.StatisticalMechanics.CanonicalEnsemble.Finite

/-!
# Bridge to Physlib's finite canonical ensemble

`TokenThermo.Basic` states its own definitions on a plain energy vector. This file shows that they
agree with Physlib's `CanonicalEnsemble`, for a finite ensemble (`IsFinite`) and the Boltzmann
constant `1`. For such an ensemble the temperature `T : Temperature` has real value `T.val`, and:

* the probability is `prob 𝓒.energy T.val`,
* the partition function is `Z 𝓒.energy T.val`,
* the mean energy and energy variance are `U` and `var`,
* the Shannon entropy is `S`,
* the Helmholtz free energy is `F`.
-/

open Real Finset

namespace TokenThermo

open CanonicalEnsemble Temperature

variable {ι : Type} [Fintype ι] [MeasurableSpace ι] [MeasurableSingletonClass ι]
  (kB : BoltzmannConstant) (hk : (kB : ℝ) = 1) (𝓒 : CanonicalEnsemble ι) [IsFinite 𝓒]
  (T : Temperature)

include hk

lemma beta_real : (T.β kB : ℝ) = 1 / (T.val : ℝ) := by
  have : (T.β kB : ℝ) = 1 / ((kB : ℝ) * (T.val : ℝ)) := rfl
  rw [this, hk, one_mul]

lemma exp_form (i : ι) : exp (-(T.β kB : ℝ) * 𝓒.energy i) = exp (-𝓒.energy i / (T.val : ℝ)) := by
  rw [beta_real kB hk T]; congr 1; ring

theorem bridge_partitionFunction : 𝓒.partitionFunction kB T = Z 𝓒.energy (T.val : ℝ) := by
  rw [partitionFunction_of_fintype]
  unfold Z
  exact Finset.sum_congr rfl fun i _ => exp_form kB hk 𝓒 T i

theorem bridge_probability (i : ι) : 𝓒.probability kB T i = prob 𝓒.energy (T.val : ℝ) i := by
  unfold CanonicalEnsemble.probability prob
  rw [mathematicalPartitionFunction_of_fintype, exp_form kB hk 𝓒 T i]
  congr 1
  unfold Z
  exact Finset.sum_congr rfl fun j _ => exp_form kB hk 𝓒 T j

theorem bridge_meanEnergy : 𝓒.meanEnergy kB T = U 𝓒.energy (T.val : ℝ) := by
  rw [meanEnergy_of_fintype]
  unfold U
  exact Finset.sum_congr rfl fun i _ => by rw [bridge_probability kB hk 𝓒 T i, mul_comm]

theorem bridge_shannonEntropy : 𝓒.shannonEntropy kB T = S 𝓒.energy (T.val : ℝ) := by
  rw [entropy_of_fintype, hk]
  unfold S
  simp only [bridge_probability kB hk 𝓒 T]
  ring

theorem bridge_energyVariance [Nonempty ι] : 𝓒.energyVariance kB T = var 𝓒.energy (T.val : ℝ) := by
  rw [energyVariance_of_fintype, bridge_meanEnergy kB hk, var_eq]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by rw [bridge_probability kB hk 𝓒 T i, mul_comm]

theorem bridge_helmholtz : 𝓒.helmholtzFreeEnergy kB T = F 𝓒.energy (T.val : ℝ) := by
  rw [helmholtzFreeEnergy_def, bridge_partitionFunction kB hk, hk]
  unfold F
  ring

end TokenThermo
