/-- The covolume of a `ℤ`-lattice is the volume of some fundamental domain; see
`ZLattice.covolume_eq_volume` for the proof that the volume does not depend on the choice of
the fundamental domain. -/
def covolume (μ : Measure E := by volume_tac) : ℝ := (addCovolume L E μ).toReal


variable (μ : Measure E := by volume_tac) [Measure.IsAddHaarMeasure μ]


theorem covolume_eq_measure_fundamentalDomain {F : Set E} (h : IsAddFundamentalDomain L F μ) :
    covolume L μ = (μ F).toReal := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure μ
    F : Set E
    h : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L x) …
    ⊢ Eq (ZLattice.covolume L μ) (μ F).toReal
  -/
  have : MeasurableVAdd L E := (inferInstance : MeasurableVAdd L.toAddSubgroup E)
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure μ
    F : Set E
    h : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L x) …
    this : MeasurableVAdd (Subtype fun x => Membership.mem L x) E
    ⊢ Eq (ZLattice.covolume L μ) (μ F).toReal
  -/
  have : VAddInvariantMeasure L E μ := (inferInstance : VAddInvariantMeasure L.toAddSubgroup E μ)
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure μ
    F : Set E
    h : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L x) …
    this✝ : MeasurableVAdd (Subtype fun x => Membership.mem L x) E
    this : MeasureTheory.VAddInvariantMeasure (Subtype fun x => Membership.mem L x …
    ⊢ Eq (ZLattice.covolume L μ) (μ F).toReal
  -/
  exact congr_arg ENNReal.toReal (h.covolume_eq_volume μ)
  /-
    🎉 no goals
  -/


theorem covolume_ne_zero : covolume L μ ≠ 0 := by
  rw [covolume_eq_measure_fundamentalDomain L μ (isAddFundamentalDomain (Free.chooseBasis ℤ L) μ),
    ENNReal.toReal_ne_zero]
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure μ
    ⊢ And (Ne (μ (ZSpan.fundamentalDomain (Basis.ofZLatticeBasis Real L (Module.Fr …
  -/
  refine ⟨measure_fundamentalDomain_ne_zero _, ne_of_lt ?_⟩
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure μ
    ⊢ LT.lt (μ (ZSpan.fundamentalDomain (Basis.ofZLatticeBasis Real L (Module.Free …
  -/
  exact Bornology.IsBounded.measure_lt_top (fundamentalDomain_isBounded _)
  /-
    🎉 no goals
  -/


theorem covolume_pos : 0 < covolume L μ :=
  lt_of_le_of_ne ENNReal.toReal_nonneg (covolume_ne_zero L μ).symm


theorem covolume_comap {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F]
    [MeasurableSpace F] [BorelSpace F] (ν : Measure F := by volume_tac) [Measure.IsAddHaarMeasure ν]
    {e : F ≃L[ℝ] E} (he : MeasurePreserving e ν μ) :
    covolume (ZLattice.comap ℝ L e.toLinearMap) ν = covolume L μ := by
  rw [covolume_eq_measure_fundamentalDomain _ _ (isAddFundamentalDomain (Free.chooseBasis ℤ L) μ),
    covolume_eq_measure_fundamentalDomain _ _ ((isAddFundamentalDomain
    ((Free.chooseBasis ℤ L).ofZLatticeComap ℝ L e.toLinearEquiv) ν)), ← he.measure_preimage
    (fundamentalDomain_measurableSet _).nullMeasurableSet, ← e.image_symm_eq_preimage,
    ← e.symm.coe_toLinearEquiv, map_fundamentalDomain]
  /-
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    L : Submodule Int E
    inst✝⁸ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝⁷ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝⁶ : MeasureTheory.Measure.IsAddHaarMeasure μ
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    ν : autoParam (MeasureTheory.Measure F) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure ν
    e : ContinuousLinearEquiv (RingHom.id Real) F E
    he : MeasureTheory.MeasurePreserving (⇑e) ν μ
    ⊢ Eq (ν (ZSpan.fundamentalDomain (Basis.ofZLatticeBasis Real (ZLattice.comap R …
  -/
  congr!
  /-
    case h.e'_1.h.e'_6.h.e'_7
    E : Type u_1
    inst✝¹³ : NormedAddCommGroup E
    inst✝¹² : NormedSpace Real E
    inst✝¹¹ : FiniteDimensional Real E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    L : Submodule Int E
    inst✝⁸ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝⁷ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝⁶ : MeasureTheory.Measure.IsAddHaarMeasure μ
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace Real F
    inst✝³ : FiniteDimensional Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    ν : autoParam (MeasureTheory.Measure F) _auto✝
    inst✝ : MeasureTheory.Measure.IsAddHaarMeasure ν
    e : ContinuousLinearEquiv (RingHom.id Real) F E
    he : MeasureTheory.MeasurePreserving (⇑e) ν μ
    ⊢ Eq (Basis.ofZLatticeBasis Real (ZLattice.comap Real L ↑e.toLinearEquiv) (Bas …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem covolume_eq_det_mul_measure {ι : Type*} [Fintype ι] [DecidableEq ι] (b : Basis ι ℤ L)
    (b₀ : Basis ι ℝ E) :
    covolume L μ = |b₀.det ((↑) ∘ b)| * (μ (fundamentalDomain b₀)).toReal := by
  rw [covolume_eq_measure_fundamentalDomain L μ (isAddFundamentalDomain b μ),
    measure_fundamentalDomain _ _ b₀,
    measure_congr (fundamentalDomain_ae_parallelepiped b₀ μ), ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (by positivity)]
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : FiniteDimensional Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    L : Submodule Int E
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝³ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝² : MeasureTheory.Measure.IsAddHaarMeasure μ
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    b₀ : Basis ι Real E
    ⊢ Eq (HMul.hMul (abs (b₀.det ⇑(Basis.ofZLatticeBasis Real L b))) (μ (parallele …
  -/
  congr
  /-
    case e_a.e_a.h.e_6.h
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : FiniteDimensional Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    L : Submodule Int E
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝³ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝² : MeasureTheory.Measure.IsAddHaarMeasure μ
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    b₀ : Basis ι Real E
    ⊢ Eq (⇑(Basis.ofZLatticeBasis Real L b)) (Function.comp Subtype.val ⇑b)
  -/
  ext
  /-
    case e_a.e_a.h.e_6.h.h
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    inst✝⁷ : FiniteDimensional Real E
    inst✝⁶ : MeasurableSpace E
    inst✝⁵ : BorelSpace E
    L : Submodule Int E
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝³ : IsZLattice Real L
    μ : autoParam (MeasureTheory.Measure E) _auto✝
    inst✝² : MeasureTheory.Measure.IsAddHaarMeasure μ
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    b₀ : Basis ι Real E
    x✝ : ι
    ⊢ Eq ((Basis.ofZLatticeBasis Real L b) x✝) (Function.comp Subtype.val (⇑b) x✝)
  -/
  exact b.ofZLatticeBasis_apply ℝ L _
  /-
    🎉 no goals
  -/


theorem covolume_eq_det {ι : Type*} [Fintype ι] [DecidableEq ι] (L : Submodule ℤ (ι → ℝ))
    [DiscreteTopology L] [IsZLattice ℝ L] (b : Basis ι ℤ L) :
    /-
      E : Type u_1
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : FiniteDimensional Real E
      inst✝⁸ : MeasurableSpace E
      inst✝⁷ : BorelSpace E
      L✝ : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
      inst✝⁵ : IsZLattice Real L✝
      μ : autoParam (MeasureTheory.Measure E) _auto✝
      inst✝⁴ : MeasureTheory.Measure.IsAddHaarMeasure μ
      ι : Type u_2
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      L : Submodule Int (ι → Real)
      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝ : IsZLattice Real L
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      ⊢ MeasureTheory.Measure (ι → Real)
    -/
    covolume L = |(Matrix.of ((↑) ∘ b)).det| := by
    /-
      🎉 no goals
    -/
  rw [covolume_eq_measure_fundamentalDomain L volume (isAddFundamentalDomain b volume),
    volume_fundamentalDomain, ENNReal.toReal_ofReal (by positivity)]
  /-
    ι : Type u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    L : Submodule Int (ι → Real)
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    ⊢ Eq (abs (Matrix.of ⇑(Basis.ofZLatticeBasis Real L b)).det) (abs (Matrix.of ( …
  -/
  congr
  /-
    case e_a.e_M.h.e_6.h
    ι : Type u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    L : Submodule Int (ι → Real)
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    ⊢ Eq (⇑(Basis.ofZLatticeBasis Real L b)) (Function.comp Subtype.val ⇑b)
  -/
  ext1
  /-
    case e_a.e_M.h.e_6.h.h
    ι : Type u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    L : Submodule Int (ι → Real)
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    x✝ : ι
    ⊢ Eq ((Basis.ofZLatticeBasis Real L b) x✝) (Function.comp Subtype.val (⇑b) x✝)
  -/
  exact b.ofZLatticeBasis_apply ℝ L _
  /-
    🎉 no goals
  -/


theorem covolume_eq_det_inv {ι : Type*} [Fintype ι] [DecidableEq ι] (L : Submodule ℤ (ι → ℝ))
    [DiscreteTopology L] [IsZLattice ℝ L] (b : Basis ι ℤ L) :
    /-
      E : Type u_1
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : FiniteDimensional Real E
      inst✝⁸ : MeasurableSpace E
      inst✝⁷ : BorelSpace E
      L✝ : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
      inst✝⁵ : IsZLattice Real L✝
      μ : autoParam (MeasureTheory.Measure E) _auto✝
      inst✝⁴ : MeasureTheory.Measure.IsAddHaarMeasure μ
      ι : Type u_2
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      L : Submodule Int (ι → Real)
      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝ : IsZLattice Real L
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      ⊢ MeasureTheory.Measure (ι → Real)
    -/
    covolume L = |(LinearEquiv.det (b.ofZLatticeBasis ℝ L).equivFun : ℝ)|⁻¹ := by
    /-
      🎉 no goals
    -/
  rw [covolume_eq_det L b, ← Pi.basisFun_det_apply, show (((↑) : L → _) ∘ ⇑b) =
    (b.ofZLatticeBasis ℝ) by ext; simp, ← Basis.det_inv, ← abs_inv, Units.val_inv_eq_inv_val,
    IsUnit.unit_spec, ← Basis.det_basis, LinearEquiv.coe_det]
  /-
    ι : Type u_2
    inst✝³ : Fintype ι
    inst✝² : DecidableEq ι
    L : Submodule Int (ι → Real)
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    ⊢ Eq (abs (Inv.inv (LinearMap.det ↑((Basis.ofZLatticeBasis Real L b).equiv (Pi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem volume_image_eq_volume_div_covolume {ι : Type*} [Fintype ι] [DecidableEq ι]
    (L : Submodule ℤ (ι → ℝ)) [DiscreteTopology L] [IsZLattice ℝ L] (b : Basis ι ℤ L)
    {s : Set (ι → ℝ)} :
                                                                                /-
                                                                                  E : Type u_1
                                                                                  inst✝¹¹ : NormedAddCommGroup E
                                                                                  inst✝¹⁰ : NormedSpace Real E
                                                                                  inst✝⁹ : FiniteDimensional Real E
                                                                                  inst✝⁸ : MeasurableSpace E
                                                                                  inst✝⁷ : BorelSpace E
                                                                                  L✝ : Submodule Int E
                                                                                  inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
                                                                                  inst✝⁵ : IsZLattice Real L✝
                                                                                  μ : autoParam (MeasureTheory.Measure E) _auto✝
                                                                                  inst✝⁴ : MeasureTheory.Measure.IsAddHaarMeasure μ
                                                                                  ι : Type u_2
                                                                                  inst✝³ : Fintype ι
                                                                                  inst✝² : DecidableEq ι
                                                                                  L : Submodule Int (ι → Real)
                                                                                  inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                                                                  inst✝ : IsZLattice Real L
                                                                                  b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                                                                  s : Set (ι → Real)
                                                                                  ⊢ MeasureTheory.Measure (ι → Real)
                                                                                -/
    volume ((b.ofZLatticeBasis ℝ L).equivFun '' s) = volume s / ENNReal.ofReal (covolume L) := by
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  rw [LinearEquiv.image_eq_preimage, Measure.addHaar_preimage_linearEquiv, LinearEquiv.symm_symm,
    covolume_eq_det_inv L b, ENNReal.div_eq_inv_mul, ENNReal.ofReal_inv_of_pos
    (abs_pos.mpr (LinearEquiv.det _).ne_zero), inv_inv, LinearEquiv.coe_det]


/-- A more general version of `ZLattice.volume_image_eq_volume_div_covolume`;
see the `Naming conventions` section in the introduction. -/
theorem volume_image_eq_volume_div_covolume' {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    (L : Submodule ℤ E) [DiscreteTopology L] [IsZLattice ℝ L] {ι : Type*} [Fintype ι]
                                        /-
                                          E✝ : Type u_1
                                          inst✝¹⁵ : NormedAddCommGroup E✝
                                          inst✝¹⁴ : NormedSpace Real E✝
                                          inst✝¹³ : FiniteDimensional Real E✝
                                          inst✝¹² : MeasurableSpace E✝
                                          inst✝¹¹ : BorelSpace E✝
                                          L✝ : Submodule Int E✝
                                          inst✝¹⁰ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
                                          inst✝⁹ : IsZLattice Real L✝
                                          μ : autoParam (MeasureTheory.Measure E✝) _auto✝
                                          inst✝⁸ : MeasureTheory.Measure.IsAddHaarMeasure μ
                                          E : Type u_2
                                          inst✝⁷ : NormedAddCommGroup E
                                          inst✝⁶ : InnerProductSpace Real E
                                          inst✝⁵ : FiniteDimensional Real E
                                          inst✝⁴ : MeasurableSpace E
                                          inst✝³ : BorelSpace E
                                          L : Submodule Int E
                                          inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                          inst✝¹ : IsZLattice Real L
                                          ι : Type u_3
                                          inst✝ : Fintype ι
                                          b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                          s : Set E
                                          ⊢ MeasureTheory.Measure E
                                        -/
    (b : Basis ι ℤ L) {s : Set E} (hs : NullMeasurableSet s) :
                                        /-
                                          🎉 no goals
                                        -/
                                                                              /-
                                                                                E✝ : Type u_1
                                                                                inst✝¹⁵ : NormedAddCommGroup E✝
                                                                                inst✝¹⁴ : NormedSpace Real E✝
                                                                                inst✝¹³ : FiniteDimensional Real E✝
                                                                                inst✝¹² : MeasurableSpace E✝
                                                                                inst✝¹¹ : BorelSpace E✝
                                                                                L✝ : Submodule Int E✝
                                                                                inst✝¹⁰ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
                                                                                inst✝⁹ : IsZLattice Real L✝
                                                                                μ : autoParam (MeasureTheory.Measure E✝) _auto✝
                                                                                inst✝⁸ : MeasureTheory.Measure.IsAddHaarMeasure μ
                                                                                E : Type u_2
                                                                                inst✝⁷ : NormedAddCommGroup E
                                                                                inst✝⁶ : InnerProductSpace Real E
                                                                                inst✝⁵ : FiniteDimensional Real E
                                                                                inst✝⁴ : MeasurableSpace E
                                                                                inst✝³ : BorelSpace E
                                                                                L : Submodule Int E
                                                                                inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                                                                inst✝¹ : IsZLattice Real L
                                                                                ι : Type u_3
                                                                                inst✝ : Fintype ι
                                                                                b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                                                                s : Set E
                                                                                hs : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
                                                                                ⊢ MeasureTheory.Measure E
                                                                              -/
    volume ((b.ofZLatticeBasis ℝ).equivFun '' s) = volume s / ENNReal.ofReal (covolume L) := by
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  classical
  let e : Fin (finrank ℝ E) ≃ ι :=
    Fintype.equivOfCardEq (by rw [Fintype.card_fin, finrank_eq_card_basis (b.ofZLatticeBasis ℝ)])
  let f := (EuclideanSpace.equiv ι ℝ).symm.trans
    ((stdOrthonormalBasis ℝ E).reindex e).repr.toContinuousLinearEquiv.symm
  have hf : MeasurePreserving f :=
    ((stdOrthonormalBasis ℝ E).reindex e).measurePreserving_repr_symm.comp
      (EuclideanSpace.volume_preserving_measurableEquiv ι).symm
  rw [← hf.measure_preimage hs, ← (covolume_comap L volume volume hf),
    ← volume_image_eq_volume_div_covolume (ZLattice.comap ℝ L f.toLinearMap)
    (b.ofZLatticeComap ℝ L f.toLinearEquiv), Basis.ofZLatticeBasis_comap,
    ← f.image_symm_eq_preimage, ← Set.image_comp]
  simp only [Basis.equivFun_apply, ContinuousLinearEquiv.symm_toLinearEquiv, Basis.map_equivFun,
    LinearEquiv.symm_symm, Function.comp_apply, LinearEquiv.trans_apply,
    ContinuousLinearEquiv.coe_toLinearEquiv, ContinuousLinearEquiv.apply_symm_apply]


/-- A version of `ZLattice.covolume.tendsto_card_div_pow` for the general case;
see the `Naming convention` section in the introduction. -/
theorem tendsto_card_div_pow'' [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    {s : Set E} (hs₁ : IsBounded s) (hs₂ : MeasurableSet s)
    (hs₃ : volume (frontier ((b.ofZLatticeBasis ℝ).equivFun '' s)) = 0):
    Tendsto (fun n : ℕ ↦ (Nat.card (s ∩ (n : ℝ)⁻¹ • L : Set E) : ℝ) / n ^ card ι)
      atTop (𝓝 (volume ((b.ofZLatticeBasis ℝ).equivFun '' s)).toReal) := by
  refine Tendsto.congr' ?_
    (tendsto_card_div_pow_atTop_volume ((b.ofZLatticeBasis ℝ).equivFun '' s) ?_ ?_ hs₃)
    /-
      case refine_1
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      ⊢ Filter.atTop.EventuallyEq (fun n => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set …
    -/
  · filter_upwards [eventually_gt_atTop 0] with n hn
    /-
      case h
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      n : Nat
      hn : LT.lt 0 n
      ⊢ Eq (HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image (⇑(Basis.ofZLatticeBasis  …
    -/
    congr
    /-
      case h.e_a.e_a
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      n : Nat
      hn : LT.lt 0 n
      ⊢ Eq (Nat.card ↑(Inter.inter (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equ …
    -/
    refine Nat.card_congr <| ((b.ofZLatticeBasis ℝ).equivFun.toEquiv.subtypeEquiv fun x ↦ ?_).symm
    simp_rw [Set.mem_inter_iff, ← b.ofZLatticeBasis_span ℝ, LinearEquiv.coe_toEquiv,
      Basis.equivFun_apply, Set.mem_image, DFunLike.coe_fn_eq, EmbeddingLike.apply_eq_iff_eq,
      exists_eq_right, and_congr_right_iff, Set.mem_inv_smul_set_iff₀
      (mod_cast hn.ne' : (n : ℝ) ≠ 0), ← Finsupp.coe_smul, ← LinearEquiv.map_smul, SetLike.mem_coe,
      Basis.mem_span_iff_repr_mem, Pi.basisFun_repr, implies_true]
    /-
      case refine_2
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      ⊢ Bornology.IsBounded (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equivFun) s)
    -/
  · rw [← NormedSpace.isVonNBounded_iff ℝ] at hs₁ ⊢
    /-
      case refine_2
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsVonNBounded Real s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      ⊢ Bornology.IsVonNBounded Real (Set.image (⇑(Basis.ofZLatticeBasis Real L b).e …
    -/
    exact Bornology.IsVonNBounded.image hs₁ ((b.ofZLatticeBasis ℝ).equivFunL : E →L[ℝ] ι → ℝ)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁴ : IsZLattice Real L
      ι : Type u_2
      inst✝³ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝² : FiniteDimensional Real E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZL …
      ⊢ MeasurableSet (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equivFun) s)
    -/
  · exact (b.ofZLatticeBasis ℝ).equivFunL.toHomeomorph.toMeasurableEquiv.measurableSet_image.mpr hs₂
    /-
      🎉 no goals
    -/


private theorem tendsto_card_le_div''_aux {X : Set E} (hX : ∀ ⦃x⦄ ⦃r:ℝ⦄, x ∈ X → 0 < r → r • x ∈ X)
    {F : E → ℝ} (hF₁ : ∀ x ⦃r : ℝ⦄, 0 ≤ r → F (r • x) = r ^ card ι * (F x)) {c : ℝ} (hc : 0 < c) :
    c • {x ∈ X | F x ≤ 1} = {x ∈ X | F x ≤ c ^ card ι} := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ι : Type u_2
    inst✝ : Fintype ι
    X : Set E
    hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
    F : E → Real
    hF₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (H …
    c : Real
    hc : LT.lt 0 c
    ⊢ Eq (HSMul.hSMul c (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))) …
  -/
  ext x
  simp_rw [Set.mem_smul_set_iff_inv_smul_mem₀ hc.ne', Set.mem_setOf_eq, hF₁ _
    (inv_pos_of_pos hc).le, inv_pow, inv_mul_le_iff₀ (pow_pos hc _), mul_one, and_congr_left_iff]
  /-
    case h
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    ι : Type u_2
    inst✝ : Fintype ι
    X : Set E
    hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
    F : E → Real
    hF₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (H …
    c : Real
    hc : LT.lt 0 c
    x : E
    ⊢ LE.le (F x) (HPow.hPow c (Fintype.card ι)) → Iff (Membership.mem X (HSMul.hS …
  -/
  exact fun _ ↦ ⟨fun h ↦ (smul_inv_smul₀ hc.ne' x) ▸ hX h hc, fun h ↦ hX h (inv_pos_of_pos hc)⟩
  /-
    🎉 no goals
  -/


/-- A version of `ZLattice.covolume.tendsto_card_le_div` for the general case;
see the `Naming conventions` section in the introduction. -/
theorem tendsto_card_le_div'' [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]
    [Nonempty ι] {X : Set E} (hX : ∀ ⦃x⦄ ⦃r : ℝ⦄, x ∈ X → 0 < r → r • x ∈ X)
    {F : E → ℝ} (h₁ : ∀ x ⦃r : ℝ⦄, 0 ≤ r →  F (r • x) = r ^ card ι * (F x))
    (h₂ : IsBounded {x ∈ X | F x ≤ 1}) (h₃ : MeasurableSet {x ∈ X | F x ≤ 1})
    (h₄ : volume (frontier ((b.ofZLatticeBasis ℝ L).equivFun '' {x | x ∈ X ∧ F x ≤ 1})) = 0) :
    Tendsto (fun c : ℝ ↦
      Nat.card ({x ∈ X | F x ≤ c} ∩ L : Set E) / (c : ℝ))
        atTop (𝓝 (volume ((b.ofZLatticeBasis ℝ).equivFun '' {x ∈ X | F x ≤ 1})).toReal) := by

  refine Tendsto.congr' ?_ <| (tendsto_card_div_pow_atTop_volume'
      ((b.ofZLatticeBasis ℝ).equivFun '' {x ∈ X | F x ≤ 1}) ?_ ?_ h₄ fun x y hx hy ↦ ?_).comp
        (tendsto_rpow_atTop <| inv_pos.mpr
          (Nat.cast_pos.mpr card_pos) : Tendsto (fun x ↦ x ^ (card ι : ℝ)⁻¹) atTop atTop)
    /-
      case refine_1
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      ⊢ Filter.atTop.EventuallyEq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(I …
    -/
  · filter_upwards [eventually_gt_atTop 0] with c hc
    /-
      case h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    have aux₁ : (card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr card_ne_zero
    /-
      case h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    have aux₂ : 0 < c ^ (card ι : ℝ)⁻¹ := Real.rpow_pos_of_pos hc _
    /-
      case h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      aux₂ : LT.lt 0 (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    have aux₃ : (c ^ (card ι : ℝ)⁻¹)⁻¹ ≠ 0 := inv_ne_zero aux₂.ne'
    /-
      case h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      aux₂ : LT.lt 0 (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))
      aux₃ : Ne (Inv.inv (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))) 0
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    have aux₄ : c ^ (-(card ι : ℝ)⁻¹) ≠ 0 := (Real.rpow_pos_of_pos hc _).ne'
    /-
      case h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      aux₂ : LT.lt 0 (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))
      aux₃ : Ne (Inv.inv (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))) 0
      aux₄ : Ne (HPow.hPow c (Neg.neg (Inv.inv ↑(Fintype.card ι)))) 0
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    obtain ⟨hc₁, hc₂⟩ := lt_iff_le_and_ne.mp hc
    /-
      case h.intro
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      aux₂ : LT.lt 0 (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))
      aux₃ : Ne (Inv.inv (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))) 0
      aux₄ : Ne (HPow.hPow c (Neg.neg (Inv.inv ↑(Fintype.card ι)))) 0
      hc₁ : LE.le 0 c
      hc₂ : Ne 0 c
      ⊢ Eq (Function.comp (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (Set.image ( …
    -/
    rw [Function.comp_apply, ← Real.rpow_natCast, Real.rpow_inv_rpow hc₁ aux₁, eq_comm]
    /-
      case h.intro
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      c : Real
      hc : LT.lt 0 c
      aux₁ : Ne (↑(Fintype.card ι)) 0
      aux₂ : LT.lt 0 (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))
      aux₃ : Ne (Inv.inv (HPow.hPow c (Inv.inv ↑(Fintype.card ι)))) 0
      aux₄ : Ne (HPow.hPow c (Neg.neg (Inv.inv ↑(Fintype.card ι)))) 0
      hc₁ : LE.le 0 c
      hc₂ : Ne 0 c
      ⊢ Eq (HDiv.hDiv (↑(Nat.card ↑(Inter.inter (setOf fun x => And (Membership.mem  …
    -/
    congr
    refine Nat.card_congr <| Equiv.subtypeEquiv ((b.ofZLatticeBasis ℝ).equivFun.toEquiv.trans
          (Equiv.smulRight aux₄)) fun _ ↦ ?_
    rw [Set.mem_inter_iff, Set.mem_inter_iff, Equiv.trans_apply, LinearEquiv.coe_toEquiv,
      Equiv.smulRight_apply, Real.rpow_neg hc₁, Set.smul_mem_smul_set_iff₀ aux₃,
      ← Set.mem_smul_set_iff_inv_smul_mem₀ aux₂.ne', ← image_smul_set,
      tendsto_card_le_div''_aux hX h₁ aux₂, ← Real.rpow_natCast, ← Real.rpow_mul hc₁,
      inv_mul_cancel₀ aux₁, Real.rpow_one]
    simp_rw [SetLike.mem_coe, Set.mem_image, EmbeddingLike.apply_eq_iff_eq, exists_eq_right,
      and_congr_right_iff, ← b.ofZLatticeBasis_span ℝ, Basis.mem_span_iff_repr_mem,
      Pi.basisFun_repr, Basis.equivFun_apply, implies_true]
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      ⊢ Bornology.IsBounded (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equivFun)  …
    -/
  · rw [← NormedSpace.isVonNBounded_iff ℝ] at h₂ ⊢
    /-
      case refine_2
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsVonNBounded Real (setOf fun x => And (Membership.mem X x) (LE …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      ⊢ Bornology.IsVonNBounded Real (Set.image (⇑(Basis.ofZLatticeBasis Real L b).e …
    -/
    exact Bornology.IsVonNBounded.image h₂ ((b.ofZLatticeBasis ℝ).equivFunL : E →L[ℝ] ι → ℝ)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      ⊢ MeasurableSet (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equivFun) (setOf …
    -/
  · exact (b.ofZLatticeBasis ℝ).equivFunL.toHomeomorph.toMeasurableEquiv.measurableSet_image.mpr h₃
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le x y
      ⊢ HasSubset.Subset (HSMul.hSMul x (Set.image (⇑(Basis.ofZLatticeBasis Real L b …
    -/
  · simp_rw [← image_smul_set]
    /-
      case refine_4
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le x y
      ⊢ HasSubset.Subset (Set.image (⇑(Basis.ofZLatticeBasis Real L b).equivFun) (HS …
    -/
    apply Set.image_mono
    rw [tendsto_card_le_div''_aux hX h₁ hx,
      tendsto_card_le_div''_aux hX h₁ (lt_of_lt_of_le hx hy)]
    /-
      case refine_4.h
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      L : Submodule Int E
      inst✝⁶ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝⁵ : IsZLattice Real L
      ι : Type u_2
      inst✝⁴ : Fintype ι
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      inst✝³ : FiniteDimensional Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : BorelSpace E
      inst✝ : Nonempty ι
      X : Set E
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      F : E → Real
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLa …
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le x y
      ⊢ HasSubset.Subset (setOf fun x_1 => And (Membership.mem X x_1) (LE.le (F x_1) …
    -/
    exact fun a ⟨ha₁, ha₂⟩ ↦ ⟨ha₁, le_trans ha₂ <| pow_le_pow_left₀ (le_of_lt hx) hy _⟩
    /-
      🎉 no goals
    -/


private theorem frontier_equivFun {E : Type*} [AddCommGroup E] [Module ℝ E] {ι : Type*} [Fintype ι]
    [TopologicalSpace E] [TopologicalAddGroup E] [ContinuousSMul ℝ E] [T2Space E]
    (b : Basis ι ℝ E) (s : Set E) :
    frontier (b.equivFun '' s) = b.equivFun '' (frontier s) := by
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    ι : Type u_2
    inst✝⁴ : Fintype ι
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T2Space E
    b : Basis ι Real E
    s : Set E
    ⊢ Eq (frontier (Set.image (⇑b.equivFun) s)) (Set.image (⇑b.equivFun) (frontier …
  -/
  rw [LinearEquiv.image_eq_preimage, LinearEquiv.image_eq_preimage]
  /-
    E : Type u_1
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    ι : Type u_2
    inst✝⁴ : Fintype ι
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T2Space E
    b : Basis ι Real E
    s : Set E
    ⊢ Eq (frontier (Set.preimage (⇑b.equivFun.symm) s)) (Set.preimage (⇑b.equivFun …
  -/
  exact (Homeomorph.preimage_frontier b.equivFunL.toHomeomorph.symm s).symm
  /-
    🎉 no goals
  -/


theorem tendsto_card_div_pow (b : Basis ι ℤ L) {s : Set (ι → ℝ)} (hs₁ : IsBounded s)
    (hs₂ : MeasurableSet s) (hs₃ : volume (frontier s) = 0) :
    Tendsto (fun n : ℕ ↦ (Nat.card (s ∩ (n : ℝ)⁻¹ • L : Set (ι → ℝ)) : ℝ) / n ^ card ι)
                                    /-
                                      ι : Type u_1
                                      inst✝² : Fintype ι
                                      L : Submodule Int (ι → Real)
                                      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                      inst✝ : IsZLattice Real L
                                      b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                      s : Set (ι → Real)
                                      hs₁ : Bornology.IsBounded s
                                      hs₂ : MeasurableSet s
                                      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
                                      ⊢ MeasureTheory.Measure (ι → Real)
                                    -/
      atTop (𝓝 ((volume s).toReal / covolume L)) := by
                                    /-
                                      🎉 no goals
                                    -/
  classical
  convert tendsto_card_div_pow'' b hs₁ hs₂ ?_
  · rw [volume_image_eq_volume_div_covolume L b, ENNReal.toReal_div,
      ENNReal.toReal_ofReal (covolume_pos L volume).le]
  · rw [frontier_equivFun, volume_image_eq_volume_div_covolume, hs₃, ENNReal.zero_div]


theorem tendsto_card_le_div {X : Set (ι → ℝ)} (hX : ∀ ⦃x⦄ ⦃r : ℝ⦄, x ∈ X → 0 < r → r • x ∈ X)
    {F : (ι → ℝ) → ℝ} (h₁ : ∀ x ⦃r : ℝ⦄, 0 ≤ r →  F (r • x) = r ^ card ι * (F x))
    (h₂ : IsBounded {x ∈ X | F x ≤ 1}) (h₃ : MeasurableSet {x ∈ X | F x ≤ 1})
    (h₄ : volume (frontier {x | x ∈ X ∧ F x ≤ 1}) = 0) [Nonempty ι] :
    Tendsto (fun c : ℝ ↦
      Nat.card ({x ∈ X | F x ≤ c} ∩ L : Set (ι → ℝ)) / (c : ℝ))
                                                      /-
                                                        ι : Type u_1
                                                        inst✝³ : Fintype ι
                                                        L : Submodule Int (ι → Real)
                                                        inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                                        inst✝¹ : IsZLattice Real L
                                                        X : Set (ι → Real)
                                                        hX : ∀ ⦃x : ι → Real⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership. …
                                                        F : (ι → Real) → Real
                                                        h₁ : ∀ (x : ι → Real) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.h …
                                                        h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
                                                        h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
                                                        h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
                                                        inst✝ : Nonempty ι
                                                        ⊢ MeasureTheory.Measure (ι → Real)
                                                      -/
        atTop (𝓝 ((volume {x ∈ X | F x ≤ 1}).toReal / covolume L)) := by
                                                      /-
                                                        🎉 no goals
                                                      -/
  classical
  let e : Free.ChooseBasisIndex ℤ ↥L ≃ ι := by
    refine Fintype.equivOfCardEq ?_
    rw [← finrank_eq_card_chooseBasisIndex, ZLattice.rank ℝ, finrank_fintype_fun_eq_card]
  let b := (Module.Free.chooseBasis ℤ L).reindex e
  convert tendsto_card_le_div'' b hX h₁ h₂ h₃ ?_
  · rw [volume_image_eq_volume_div_covolume L b, ENNReal.toReal_div,
      ENNReal.toReal_ofReal (covolume_pos L volume).le]
  · rw [frontier_equivFun, volume_image_eq_volume_div_covolume, h₄, ENNReal.zero_div]


/-- A version of `ZLattice.covolume.tendsto_card_div_pow` for the `InnerProductSpace` case;
see the `Naming convention` section in the introduction. -/
theorem tendsto_card_div_pow' {s : Set E} (hs₁ : IsBounded s) (hs₂ : MeasurableSet s)
    (hs₃ : volume (frontier s) = 0) :
    Tendsto (fun n : ℕ ↦ (Nat.card (s ∩ (n : ℝ)⁻¹ • L : Set E) : ℝ) / n ^ finrank ℝ E)
                                    /-
                                      E : Type u_1
                                      inst✝⁶ : NormedAddCommGroup E
                                      inst✝⁵ : InnerProductSpace Real E
                                      inst✝⁴ : FiniteDimensional Real E
                                      inst✝³ : MeasurableSpace E
                                      inst✝² : BorelSpace E
                                      L : Submodule Int E
                                      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                      inst✝ : IsZLattice Real L
                                      s : Set E
                                      hs₁ : Bornology.IsBounded s
                                      hs₂ : MeasurableSet s
                                      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
                                      ⊢ MeasureTheory.Measure E
                                    -/
      atTop (𝓝 ((volume s).toReal / covolume L)) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace Real E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    s : Set E
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
  -/
  let b := Module.Free.chooseBasis ℤ L
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : InnerProductSpace Real E
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    s : Set E
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
  -/
  convert tendsto_card_div_pow'' b hs₁ hs₂ ?_
    /-
      case h.e'_3.h.h.e'_6.h.e'_6
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : InnerProductSpace Real E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      L : Submodule Int E
      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝ : IsZLattice Real L
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      x✝ : Nat
      ⊢ Eq (Module.finrank Real E) (Fintype.card (Module.Free.ChooseBasisIndex Int ( …
    -/
  · rw [← finrank_eq_card_chooseBasisIndex, ZLattice.rank ℝ L]
    /-
      🎉 no goals
    -/
  · rw [volume_image_eq_volume_div_covolume' L b hs₂.nullMeasurableSet, ENNReal.toReal_div,
      ENNReal.toReal_ofReal (covolume_pos L volume).le]
    /-
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : InnerProductSpace Real E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      L : Submodule Int E
      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝ : IsZLattice Real L
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLatti …
    -/
  · rw [frontier_equivFun, volume_image_eq_volume_div_covolume', hs₃, ENNReal.zero_div]
    /-
      case hs
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : InnerProductSpace Real E
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : MeasurableSpace E
      inst✝² : BorelSpace E
      L : Submodule Int E
      inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝ : IsZLattice Real L
      s : Set E
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ MeasureTheory.NullMeasurableSet (frontier s) MeasureTheory.MeasureSpace.volume
    -/
    exact NullMeasurableSet.of_null hs₃
    /-
      🎉 no goals
    -/


/-- A version of `ZLattice.covolume.tendsto_card_le_div` for the `InnerProductSpace` case;
see the `Naming convention` section in the introduction. -/
theorem tendsto_card_le_div' [Nontrivial E] {X : Set E} {F : E → ℝ}
    (hX : ∀ ⦃x⦄ ⦃r : ℝ⦄, x ∈ X → 0 < r → r • x ∈ X)
    (h₁ : ∀ x ⦃r : ℝ⦄, 0 ≤ r →  F (r • x) = r ^ finrank ℝ E * (F x))
    (h₂ : IsBounded {x ∈ X | F x ≤ 1}) (h₃ : MeasurableSet {x ∈ X | F x ≤ 1})
    (h₄ : volume (frontier {x ∈ X | F x ≤ 1}) = 0) :
    Tendsto (fun c : ℝ ↦
      Nat.card ({x ∈ X | F x ≤ c} ∩ L : Set E) / (c : ℝ))
                                                      /-
                                                        E : Type u_1
                                                        inst✝⁷ : NormedAddCommGroup E
                                                        inst✝⁶ : InnerProductSpace Real E
                                                        inst✝⁵ : FiniteDimensional Real E
                                                        inst✝⁴ : MeasurableSpace E
                                                        inst✝³ : BorelSpace E
                                                        L : Submodule Int E
                                                        inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                                        inst✝¹ : IsZLattice Real L
                                                        inst✝ : Nontrivial E
                                                        X : Set E
                                                        F : E → Real
                                                        hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
                                                        h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
                                                        h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
                                                        h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
                                                        h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
                                                        ⊢ MeasureTheory.Measure E
                                                      -/
        atTop (𝓝 ((volume {x ∈ X | F x ≤ 1}).toReal / covolume L)) := by
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    inst✝ : Nontrivial E
    X : Set E
    F : E → Real
    hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
    h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
    h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
    h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
    h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
    ⊢ Filter.Tendsto (fun c => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (setOf fun x => …
  -/
  let b := Module.Free.chooseBasis ℤ L
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    L : Submodule Int E
    inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝¹ : IsZLattice Real L
    inst✝ : Nontrivial E
    X : Set E
    F : E → Real
    hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
    h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
    h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
    h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
    h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
    b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
    ⊢ Filter.Tendsto (fun c => HDiv.hDiv (↑(Nat.card ↑(Inter.inter (setOf fun x => …
  -/
  convert tendsto_card_le_div'' b hX ?_ h₂ h₃ ?_
  · rw [volume_image_eq_volume_div_covolume' L b h₃.nullMeasurableSet, ENNReal.toReal_div,
      ENNReal.toReal_ofReal (covolume_pos L volume).le]
    /-
      case convert_1
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : InnerProductSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      L : Submodule Int E
      inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝¹ : IsZLattice Real L
      inst✝ : Nontrivial E
      X : Set E
      F : E → Real
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ Nonempty (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem  …
    -/
  · have : Nontrivial L := nontrivial_of_finrank_pos <| (ZLattice.rank ℝ L).symm ▸ finrank_pos
    /-
      case convert_1
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : InnerProductSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      L : Submodule Int E
      inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝¹ : IsZLattice Real L
      inst✝ : Nontrivial E
      X : Set E
      F : E → Real
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      this : Nontrivial (Subtype fun x => Membership.mem L x)
      ⊢ Nonempty (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : InnerProductSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      L : Submodule Int E
      inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝¹ : IsZLattice Real L
      inst✝ : Nontrivial E
      X : Set E
      F : E → Real
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HPow. …
    -/
  · rwa [← finrank_eq_card_chooseBasisIndex, ZLattice.rank ℝ L]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : InnerProductSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      L : Submodule Int E
      inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝¹ : IsZLattice Real L
      inst✝ : Nontrivial E
      X : Set E
      F : E → Real
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (frontier (Set.image (⇑(Basis.ofZLatti …
    -/
  · rw [frontier_equivFun, volume_image_eq_volume_div_covolume', h₄, ENNReal.zero_div]
    /-
      case convert_3.hs
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : InnerProductSpace Real E
      inst✝⁵ : FiniteDimensional Real E
      inst✝⁴ : MeasurableSpace E
      inst✝³ : BorelSpace E
      L : Submodule Int E
      inst✝² : DiscreteTopology (Subtype fun x => Membership.mem L x)
      inst✝¹ : IsZLattice Real L
      inst✝ : Nontrivial E
      X : Set E
      F : E → Real
      hX : ∀ ⦃x : E⦄ ⦃r : Real⦄, Membership.mem X x → LT.lt 0 r → Membership.mem X ( …
      h₁ : ∀ (x : E) ⦃r : Real⦄, LE.le 0 r → Eq (F (HSMul.hSMul r x)) (HMul.hMul (HP …
      h₂ : Bornology.IsBounded (setOf fun x => And (Membership.mem X x) (LE.le (F x) …
      h₃ : MeasurableSet (setOf fun x => And (Membership.mem X x) (LE.le (F x) 1))
      h₄ : Eq (MeasureTheory.MeasureSpace.volume (frontier (setOf fun x => And (Memb …
      b : Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      ⊢ MeasureTheory.NullMeasurableSet (frontier (setOf fun x => And (Membership.me …
    -/
    exact NullMeasurableSet.of_null h₄
    /-
      🎉 no goals
    -/


