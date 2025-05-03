variable (R) in
/-- The pointwise topology on `MvPowerSeries` -/
scoped instance : TopologicalSpace (MvPowerSeries σ R) :=
  Pi.topologicalSpace


/-- `MvPowerSeries` on a `T0Space` form a `T0Space` -/
@[scoped instance]
theorem instT0Space [T0Space R] : T0Space (MvPowerSeries σ R) := Pi.instT0Space


/-- `MvPowerSeries` on a `T2Space` form a `T2Space` -/
@[scoped instance]
theorem instT2Space [T2Space R] : T2Space (MvPowerSeries σ R) := Pi.t2Space


variable (R) in
/-- `MvPowerSeries.coeff` is continuous. -/
@[fun_prop]
theorem continuous_coeff [Semiring R] (d : σ →₀ ℕ) :
    Continuous (MvPowerSeries.coeff R d) :=
  continuous_pi_iff.mp continuous_id d


variable (R) in
/-- `MvPolynomial.constantCoeff` is continuous -/
theorem continuous_constantCoeff [Semiring R] : Continuous (constantCoeff σ R) :=
  continuous_coeff R 0


/-- A family of power series converges iff it converges coefficientwise -/
theorem tendsto_iff_coeff_tendsto [Semiring R] {ι : Type*}
    (f : ι → MvPowerSeries σ R) (u : Filter ι) (g : MvPowerSeries σ R) :
    Tendsto f u (nhds g) ↔
    ∀ d : σ →₀ ℕ, Tendsto (fun i => coeff R d (f i)) u (nhds (coeff R d g)) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_3
    f : ι → MvPowerSeries σ R
    u : Filter ι
    g : MvPowerSeries σ R
    ⊢ Iff (Filter.Tendsto f u (nhds g)) (∀ (d : Finsupp σ Nat), Filter.Tendsto (fu …
  -/
  rw [nhds_pi, tendsto_pi]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    ι : Type u_3
    f : ι → MvPowerSeries σ R
    u : Filter ι
    g : MvPowerSeries σ R
    ⊢ Iff (∀ (i : Finsupp σ Nat), Filter.Tendsto (fun x => f x i) u (nhds (g i)))  …
  -/
  exact forall_congr' (fun d => Iff.rfl)
  /-
    🎉 no goals
  -/


/-- The semiring topology on `MvPowerSeries` of a topological semiring -/
@[scoped instance]
theorem instTopologicalSemiring [Semiring R] [TopologicalSemiring R] :
    TopologicalSemiring (MvPowerSeries σ R) where
    continuous_add := continuous_pi fun d => continuous_add.comp
      (((continuous_coeff R d).fst').prod_mk (continuous_coeff R d).snd')
    continuous_mul := continuous_pi fun _ =>
      continuous_finset_sum _ fun i _ => continuous_mul.comp
        ((continuous_coeff R i.fst).fst'.prod_mk (continuous_coeff R i.snd).snd')


/-- The ring topology on `MvPowerSeries` of a topological ring -/
@[scoped instance]
theorem instTopologicalRing [Ring R] [TopologicalRing R] :
    TopologicalRing (MvPowerSeries σ R) :=
  { instTopologicalSemiring σ R with
    continuous_neg := continuous_pi fun d ↦ Continuous.comp continuous_neg
      (continuous_coeff R d) }


@[fun_prop]
theorem continuous_C [Semiring R] :
    Continuous (C σ R) := by
  classical
  simp only [continuous_iff_continuousAt]
  refine fun r ↦ (tendsto_iff_coeff_tendsto _ _ _).mpr fun d ↦ ?_
  simp only [coeff_C]
  split_ifs
  · exact tendsto_id
  · exact tendsto_const_nhds


theorem variables_tendsto_zero [Semiring R] :
    Tendsto (X · : σ → MvPowerSeries σ R) cofinite (nhds 0) := by
  classical
  simp only [tendsto_iff_coeff_tendsto, ← coeff_apply, coeff_X, coeff_zero]
  refine fun d ↦ tendsto_nhds_of_eventually_eq ?_
  by_cases h : ∃ i, d = Finsupp.single i 1
  · obtain ⟨i, hi⟩ := h
    filter_upwards [eventually_cofinite_ne i] with j hj
    simp [hi, Finsupp.single_eq_single_iff, hj.symm]
  · simpa only [ite_eq_right_iff] using
      Eventually.of_forall fun x h' ↦ (not_exists.mp h x h').elim


theorem tendsto_pow_zero_of_constantCoeff_nilpotent [CommSemiring R]
    {f} (hf : IsNilpotent (constantCoeff σ R f)) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) := by
  classical
  obtain ⟨m, hm⟩ := hf
  simp_rw [tendsto_iff_coeff_tendsto, coeff_zero]
  exact fun d ↦ tendsto_atTop_of_eventually_const fun n hn ↦
    coeff_eq_zero_of_constantCoeff_nilpotent hm hn


theorem tendsto_pow_zero_of_constantCoeff_zero [CommSemiring R]
    {f} (hf : constantCoeff σ R f = 0) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : CommSemiring R
    f : MvPowerSeries σ R
    hf : Eq ((MvPowerSeries.constantCoeff σ R) f) 0
    ⊢ Filter.Tendsto (fun n => HPow.hPow f n) Filter.atTop (nhds 0)
  -/
  apply tendsto_pow_zero_of_constantCoeff_nilpotent
  /-
    case hf
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : CommSemiring R
    f : MvPowerSeries σ R
    hf : Eq ((MvPowerSeries.constantCoeff σ R) f) 0
    ⊢ IsNilpotent ((MvPowerSeries.constantCoeff σ R) f)
  -/
  rw [hf]
  /-
    case hf
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : CommSemiring R
    f : MvPowerSeries σ R
    hf : Eq ((MvPowerSeries.constantCoeff σ R) f) 0
    ⊢ IsNilpotent 0
  -/
  exact IsNilpotent.zero
  /-
    🎉 no goals
  -/


/-- The powers of a `MvPowerSeries` converge to 0 iff its constant coefficient is nilpotent.
N. Bourbaki, *Algebra II*, [bourbaki1981] (chap. 4, §4, n°2, corollaire de la prop. 3) -/
theorem tendsto_pow_of_constantCoeff_nilpotent_iff [CommRing R] [DiscreteTopology R] (f) :
    Tendsto (fun n : ℕ => f ^ n) atTop (nhds 0) ↔
      IsNilpotent (constantCoeff σ R f) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    ⊢ Iff (Filter.Tendsto (fun n => HPow.hPow f n) Filter.atTop (nhds 0)) (IsNilpo …
  -/
  refine ⟨?_, tendsto_pow_zero_of_constantCoeff_nilpotent⟩
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    ⊢ Filter.Tendsto (fun n => HPow.hPow f n) Filter.atTop (nhds 0) → IsNilpotent  …
  -/
  intro h
  suffices Tendsto (fun n : ℕ => constantCoeff σ R (f ^ n)) atTop (nhds 0) by
    simp only [tendsto_def] at this
    specialize this {0} _
    suffices ∀ x : R, {x} ∈ nhds x by exact this 0
    rw [← discreteTopology_iff_singleton_mem_nhds]; infer_instance
    simp only [map_pow, mem_atTop_sets, ge_iff_le, Set.mem_preimage,
      Set.mem_singleton_iff] at this
    obtain ⟨m, hm⟩ := this
    use m
    apply hm m (le_refl m)
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    h : Filter.Tendsto (fun n => HPow.hPow f n) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun n => (MvPowerSeries.constantCoeff σ R) (HPow.hPow f n))  …
  -/
  simp only [← @comp_apply _ R ℕ, ← tendsto_map'_iff]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    h : Filter.Tendsto (fun n => HPow.hPow f n) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (⇑(MvPowerSeries.constantCoeff σ R)) (Filter.map (HPow.hPow f …
  -/
  simp only [Tendsto, map_le_iff_le_comap] at h ⊢
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    h : LE.le Filter.atTop (Filter.comap (fun n => HPow.hPow f n) (nhds 0))
    ⊢ LE.le Filter.atTop (Filter.comap (HPow.hPow f) (Filter.comap (⇑(MvPowerSerie …
  -/
  refine le_trans h (comap_mono ?_)
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    h : LE.le Filter.atTop (Filter.comap (fun n => HPow.hPow f n) (nhds 0))
    ⊢ LE.le (nhds 0) (Filter.comap (⇑(MvPowerSeries.constantCoeff σ R)) (nhds 0))
  -/
  rw [← map_le_iff_le_comap]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : TopologicalSpace R
    inst✝¹ : CommRing R
    inst✝ : DiscreteTopology R
    f : MvPowerSeries σ R
    h : LE.le Filter.atTop (Filter.comap (fun n => HPow.hPow f n) (nhds 0))
    ⊢ LE.le (Filter.map (⇑(MvPowerSeries.constantCoeff σ R)) (nhds 0)) (nhds 0)
  -/
  exact Continuous.continuousAt (continuous_constantCoeff R)
  /-
    🎉 no goals
  -/


/-- A multivariate power series is the sum (in the sense of summable families) of its monomials -/
theorem hasSum_of_monomials_self (f : MvPowerSeries σ R) :
    HasSum (fun d : σ →₀ ℕ => monomial R d (coeff R d f)) f := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    ⊢ HasSum (fun d => (MvPowerSeries.monomial R d) ((MvPowerSeries.coeff R d) f)) f
  -/
  rw [Pi.hasSum]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    ⊢ ∀ (x : Finsupp σ Nat), HasSum (fun i => (MvPowerSeries.monomial R i) ((MvPow …
  -/
  intro d
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : TopologicalSpace R
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    ⊢ HasSum (fun i => (MvPowerSeries.monomial R i) ((MvPowerSeries.coeff R i) f)  …
  -/
  convert hasSum_single d ?_ using 1
    /-
      case h.e'_6
      σ : Type u_1
      R : Type u_2
      inst✝¹ : TopologicalSpace R
      inst✝ : Semiring R
      f : MvPowerSeries σ R
      d : Finsupp σ Nat
      ⊢ Eq (f d) ((MvPowerSeries.monomial R d) ((MvPowerSeries.coeff R d) f) d)
    -/
  · exact (coeff_monomial_same d _).symm
    /-
      🎉 no goals
    -/
    /-
      case convert_5
      σ : Type u_1
      R : Type u_2
      inst✝¹ : TopologicalSpace R
      inst✝ : Semiring R
      f : MvPowerSeries σ R
      d : Finsupp σ Nat
      ⊢ ∀ (b' : Finsupp σ Nat), Ne b' d → Eq ((MvPowerSeries.monomial R b') ((MvPowe …
    -/
  · exact fun d' h ↦ coeff_monomial_ne (Ne.symm h) _
    /-
      🎉 no goals
    -/


/-- If the coefficient space is T2, then the multivariate power series is `tsum` of its monomials -/
theorem as_tsum [T2Space R] (f : MvPowerSeries σ R) :
    f = tsum fun d : σ →₀ ℕ => monomial R d (coeff R d f) :=
  (HasSum.tsum_eq (hasSum_of_monomials_self _)).symm


/-- The componentwise uniformity on `MvPowerSeries` -/
scoped instance : UniformSpace (MvPowerSeries σ R) :=
  Pi.uniformSpace fun _ : σ →₀ ℕ => R


variable (R) in
/-- Coefficients of a multivariate power series are uniformly continuous -/
theorem uniformContinuous_coeff [Semiring R] (d : σ →₀ ℕ) :
    UniformContinuous fun f : MvPowerSeries σ R => coeff R d f :=
  uniformContinuous_pi.mp uniformContinuous_id d


/-- Completeness of the uniform structure on `MvPowerSeries` -/
@[scoped instance]
theorem instCompleteSpace [CompleteSpace R] :
    CompleteSpace (MvPowerSeries σ R) := Pi.complete _


/-- The `UniformAddGroup` structure on `MvPowerSeries` of a `UniformAddGroup` -/
@[scoped instance]
theorem instUniformAddGroup [AddGroup R] [UniformAddGroup R] :
    UniformAddGroup (MvPowerSeries σ R) := Pi.instUniformAddGroup


