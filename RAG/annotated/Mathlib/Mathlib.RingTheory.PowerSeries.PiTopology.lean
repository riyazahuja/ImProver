/-- The pointwise topology on `PowerSeries` -/
scoped instance : TopologicalSpace (PowerSeries R) :=
  Pi.topologicalSpace


/-- Separation of the topology on `PowerSeries` -/
@[scoped instance]
theorem instT0Space [T0Space R] : T0Space (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instT0Space


/-- `PowerSeries` on a `T2Space` form a `T2Space` -/
@[scoped instance]
theorem instT2Space [T2Space R] : T2Space (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instT2Space


/-- Coefficients are continuous -/
theorem continuous_coeff [Semiring R] (d : ℕ) : Continuous (PowerSeries.coeff R d) :=
  continuous_pi_iff.mp continuous_id (Finsupp.single () d)


/-- The constant coefficient is continuous -/
theorem continuous_constantCoeff [Semiring R] : Continuous (constantCoeff R) :=
  coeff_zero_eq_constantCoeff (R := R) ▸ continuous_coeff R 0


/-- A family of power series converges iff it converges coefficientwise -/
theorem tendsto_iff_coeff_tendsto [Semiring R] {ι : Type*}
    (f : ι → PowerSeries R) (u : Filter ι) (g : PowerSeries R) :
    Tendsto f u (nhds g) ↔
    ∀ d : ℕ, Tendsto (fun i => coeff R d (f i)) u (nhds (coeff R d g)) := by
  /-
    R : Type u_1
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_2
    f : ι → PowerSeries R
    u : Filter ι
    g : PowerSeries R
    ⊢ Iff (Filter.Tendsto f u (nhds g)) (∀ (d : Nat), Filter.Tendsto (fun i => (Po …
  -/
  rw [MvPowerSeries.WithPiTopology.tendsto_iff_coeff_tendsto]
  /-
    R : Type u_1
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_2
    f : ι → PowerSeries R
    u : Filter ι
    g : PowerSeries R
    ⊢ Iff (∀ (d : Finsupp Unit Nat), Filter.Tendsto (fun i => (MvPowerSeries.coeff …
  -/
  apply (Finsupp.LinearEquiv.finsuppUnique ℕ ℕ Unit).toEquiv.forall_congr
  /-
    R : Type u_1
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_2
    f : ι → PowerSeries R
    u : Filter ι
    g : PowerSeries R
    ⊢ ∀ (a : Finsupp Unit Nat), Iff (Filter.Tendsto (fun i => (MvPowerSeries.coeff …
  -/
  intro d
  simp only [LinearEquiv.coe_toEquiv, Finsupp.LinearEquiv.finsuppUnique_apply,
    PUnit.default_eq_unit, coeff]
  /-
    R : Type u_1
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_2
    f : ι → PowerSeries R
    u : Filter ι
    g : PowerSeries R
    d : Finsupp Unit Nat
    ⊢ Iff (Filter.Tendsto (fun i => (MvPowerSeries.coeff R d) (f i)) u (nhds ((MvP …
  -/
  apply iff_of_eq
  /-
    case a
    R : Type u_1
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_2
    f : ι → PowerSeries R
    u : Filter ι
    g : PowerSeries R
    d : Finsupp Unit Nat
    ⊢ Eq (Filter.Tendsto (fun i => (MvPowerSeries.coeff R d) (f i)) u (nhds ((MvPo …
  -/
  congr
    /-
      case a.e_f
      R : Type u_1
      inst✝¹ : TopologicalSpace R
      inst✝ : Semiring R
      ι : Type u_2
      f : ι → PowerSeries R
      u : Filter ι
      g : PowerSeries R
      d : Finsupp Unit Nat
      ⊢ Eq (fun i => (MvPowerSeries.coeff R d) (f i)) fun i => (MvPowerSeries.coeff  …
    -/
  · ext _; congr; ext; simp
                       /-
                         🎉 no goals
                       -/
    /-
      case a.e_l₂.e_x.e_a.e_n
      R : Type u_1
      inst✝¹ : TopologicalSpace R
      inst✝ : Semiring R
      ι : Type u_2
      f : ι → PowerSeries R
      u : Filter ι
      g : PowerSeries R
      d : Finsupp Unit Nat
      ⊢ Eq d (Finsupp.single Unit.unit (d PUnit.unit))
    -/
  · ext; simp
         /-
           🎉 no goals
         -/


/-- The semiring topology on `PowerSeries` of a topological semiring -/
@[scoped instance]
theorem instTopologicalSemiring [Semiring R] [TopologicalSemiring R] :
    TopologicalSemiring (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instTopologicalSemiring Unit R


/-- The ring topology on `PowerSeries` of a topological ring -/
@[scoped instance]
theorem instTopologicalRing [Ring R] [TopologicalRing R] :
    TopologicalRing (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instTopologicalRing Unit R


/-- The product uniformity on `PowerSeries` -/
scoped instance : UniformSpace (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instUniformSpace


/-- Coefficients are uniformly continuous -/
theorem uniformContinuous_coeff [Semiring R] (d : ℕ) :
    UniformContinuous fun f : PowerSeries R ↦ coeff R d f :=
  uniformContinuous_pi.mp uniformContinuous_id (Finsupp.single () d)


/-- Completeness of the uniform structure on `PowerSeries` -/
@[scoped instance]
theorem instCompleteSpace [CompleteSpace R] :
    CompleteSpace (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instCompleteSpace


/-- The `UniformAddGroup` structure on `PowerSeries` of a `UniformAddGroup` -/
@[scoped instance]
theorem instUniformAddGroup [AddGroup R] [UniformAddGroup R] :
    UniformAddGroup (PowerSeries R) :=
  MvPowerSeries.WithPiTopology.instUniformAddGroup


theorem continuous_C [Semiring R] : Continuous (C R) :=
  MvPowerSeries.WithPiTopology.continuous_C


theorem tendsto_pow_zero_of_constantCoeff_nilpotent [CommSemiring R]
    {f : PowerSeries R} (hf : IsNilpotent (constantCoeff R f)) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) :=
  MvPowerSeries.WithPiTopology.tendsto_pow_zero_of_constantCoeff_nilpotent hf


theorem tendsto_pow_zero_of_constantCoeff_zero [CommSemiring R]
    {f : PowerSeries R} (hf : constantCoeff R f = 0) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) :=
  MvPowerSeries.WithPiTopology.tendsto_pow_zero_of_constantCoeff_zero hf


/-- The powers of a `PowerSeries` converge to 0 iff its constant coefficient is nilpotent.
N. Bourbaki, *Algebra II*, [bourbaki1981] (chap. 4, §4, n°2, corollaire de la prop. 3) -/
theorem tendsto_pow_zero_of_constantCoeff_nilpotent_iff
    [CommRing R] [DiscreteTopology R] (f : PowerSeries R) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) ↔
      IsNilpotent (constantCoeff R f) :=
  MvPowerSeries.WithPiTopology.tendsto_pow_of_constantCoeff_nilpotent_iff f


/-- A power series is the sum (in the sense of summable families) of its monomials -/
theorem hasSum_of_monomials_self (f : PowerSeries R) :
    HasSum (fun d : ℕ => monomial R d (coeff R d f)) f := by
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : TopologicalSpace R
    f : PowerSeries R
    ⊢ HasSum (fun d => (PowerSeries.monomial R d) ((PowerSeries.coeff R d) f)) f
  -/
  rw [← (Finsupp.LinearEquiv.finsuppUnique ℕ ℕ Unit).toEquiv.hasSum_iff]
  /-
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : TopologicalSpace R
    f : PowerSeries R
    ⊢ HasSum (Function.comp (fun d => (PowerSeries.monomial R d) ((PowerSeries.coe …
  -/
  convert MvPowerSeries.WithPiTopology.hasSum_of_monomials_self f
  simp only [LinearEquiv.coe_toEquiv, comp_apply, monomial, coeff,
    Finsupp.LinearEquiv.finsuppUnique_apply, PUnit.default_eq_unit]
  /-
    case h.e'_5.h
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : TopologicalSpace R
    f : PowerSeries R
    x✝ : Finsupp Unit Nat
    ⊢ Eq ((MvPowerSeries.monomial R (Finsupp.single Unit.unit (x✝ PUnit.unit))) (( …
  -/
  congr
  /-
    case h.e'_5.h.h.e_5.h.e_n
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : TopologicalSpace R
    f : PowerSeries R
    x✝ : Finsupp Unit Nat
    ⊢ Eq (Finsupp.single Unit.unit (x✝ PUnit.unit)) x✝
  -/
  all_goals { ext; simp }
  /-
    🎉 no goals
  -/


/-- If the coefficient space is T2, then the power series is `tsum` of its monomials -/
theorem as_tsum [T2Space R] (f : PowerSeries R) :
    f = tsum fun d : ℕ => monomial R d (coeff R d f) :=
  (HasSum.tsum_eq (hasSum_of_monomials_self f)).symm


