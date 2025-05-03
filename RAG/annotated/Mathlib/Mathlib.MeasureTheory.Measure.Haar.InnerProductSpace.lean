/-- Every linear isometry equivalence is a measurable equivalence. -/
def toMeasureEquiv : E ≃ᵐ F where
  toEquiv := f
  measurable_toFun := f.continuous.measurable
  measurable_invFun := f.symm.continuous.measurable


@[simp] theorem coe_toMeasureEquiv : (f.toMeasureEquiv : E → F) = f := rfl


theorem toMeasureEquiv_symm : f.toMeasureEquiv.symm = f.symm.toMeasureEquiv := rfl


/-- The volume form coming from an orientation in an inner product space gives measure `1` to the
parallelepiped associated to any orthonormal basis. This is a rephrasing of
`abs_volumeForm_apply_of_orthonormal` in terms of measures. -/
theorem Orientation.measure_orthonormalBasis (o : Orientation ℝ F (Fin n))
    (b : OrthonormalBasis ι ℝ F) : o.volumeForm.measure (parallelepiped b) = 1 := by
  have e : ι ≃ Fin n := by
    refine Fintype.equivFinOfCardEq ?_
    rw [← _i.out, finrank_eq_card_basis b.toBasis]
  have A : ⇑b = b.reindex e ∘ e := by
    ext x
    simp only [OrthonormalBasis.coe_reindex, Function.comp_apply, Equiv.symm_apply_apply]
  rw [A, parallelepiped_comp_equiv, AlternatingMap.measure_parallelepiped,
    o.abs_volumeForm_apply_of_orthonormal, ENNReal.ofReal_one]


/-- In an oriented inner product space, the measure coming from the canonical volume form
associated to an orientation coincides with the volume. -/
theorem Orientation.measure_eq_volume (o : Orientation ℝ F (Fin n)) :
    o.volumeForm.measure = volume := by
  have A : o.volumeForm.measure (stdOrthonormalBasis ℝ F).toBasis.parallelepiped = 1 :=
    Orientation.measure_orthonormalBasis o (stdOrthonormalBasis ℝ F)
  rw [addHaarMeasure_unique o.volumeForm.measure
    (stdOrthonormalBasis ℝ F).toBasis.parallelepiped, A, one_smul]
  /-
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace Real F
    inst✝² : MeasurableSpace F
    inst✝¹ : BorelSpace F
    inst✝ : FiniteDimensional Real F
    n : Nat
    _i : Fact (Eq (Module.finrank Real F) n)
    o : Orientation Real F (Fin n)
    A : Eq (o.volumeForm.measure ↑(stdOrthonormalBasis Real F).toBasis.parallelepi …
    ⊢ Eq (MeasureTheory.Measure.addHaarMeasure (stdOrthonormalBasis Real F).toBasi …
  -/
  simp only [volume, Basis.addHaar]
  /-
    🎉 no goals
  -/


/-- The volume measure in a finite-dimensional inner product space gives measure `1` to the
parallelepiped spanned by any orthonormal basis. -/
theorem OrthonormalBasis.volume_parallelepiped (b : OrthonormalBasis ι ℝ F) :
    volume (parallelepiped b) = 1 := by
  /-
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (parallelepiped ⇑b)) 1
  -/
  haveI : Fact (finrank ℝ F = finrank ℝ F) := ⟨rfl⟩
  /-
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    this : Fact (Eq (Module.finrank Real F) (Module.finrank Real F))
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (parallelepiped ⇑b)) 1
  -/
  let o := (stdOrthonormalBasis ℝ F).toBasis.orientation
  /-
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    this : Fact (Eq (Module.finrank Real F) (Module.finrank Real F))
    o : Orientation Real F (Fin (Module.finrank Real F)) := (stdOrthonormalBasis R …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (parallelepiped ⇑b)) 1
  -/
  rw [← o.measure_eq_volume]
  /-
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    this : Fact (Eq (Module.finrank Real F) (Module.finrank Real F))
    o : Orientation Real F (Fin (Module.finrank Real F)) := (stdOrthonormalBasis R …
    ⊢ Eq (o.volumeForm.measure (parallelepiped ⇑b)) 1
  -/
  exact o.measure_orthonormalBasis b
  /-
    🎉 no goals
  -/


/-- The Haar measure defined by any orthonormal basis of a finite-dimensional inner product space
is equal to its volume measure. -/
theorem OrthonormalBasis.addHaar_eq_volume {ι F : Type*} [Fintype ι] [NormedAddCommGroup F]
    [InnerProductSpace ℝ F] [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F]
    (b : OrthonormalBasis ι ℝ F) :
    b.toBasis.addHaar = volume := by
  /-
    ι : Type u_4
    F : Type u_5
    inst✝⁵ : Fintype ι
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace Real F
    inst✝² : FiniteDimensional Real F
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    b : OrthonormalBasis ι Real F
    ⊢ Eq b.toBasis.addHaar MeasureTheory.MeasureSpace.volume
  -/
  rw [Basis.addHaar_eq_iff]
  /-
    ι : Type u_4
    F : Type u_5
    inst✝⁵ : Fintype ι
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace Real F
    inst✝² : FiniteDimensional Real F
    inst✝¹ : MeasurableSpace F
    inst✝ : BorelSpace F
    b : OrthonormalBasis ι Real F
    ⊢ Eq (MeasureTheory.MeasureSpace.volume ↑b.toBasis.parallelepiped) 1
  -/
  exact b.volume_parallelepiped
  /-
    🎉 no goals
  -/


/-- An orthonormal basis of a finite-dimensional inner product space defines a measurable
equivalence between the space and the Euclidean space of the same dimension. -/
noncomputable def OrthonormalBasis.measurableEquiv (b : OrthonormalBasis ι ℝ F) :
    F ≃ᵐ EuclideanSpace ℝ ι := b.repr.toHomeomorph.toMeasurableEquiv


/-- The measurable equivalence defined by an orthonormal basis is volume preserving. -/
theorem OrthonormalBasis.measurePreserving_measurableEquiv (b : OrthonormalBasis ι ℝ F) :
    MeasurePreserving b.measurableEquiv volume volume := by
  /-
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    ⊢ MeasureTheory.MeasurePreserving (⇑b.measurableEquiv) MeasureTheory.MeasureSp …
  -/
  convert (b.measurableEquiv.symm.measurable.measurePreserving _).symm
  /-
    case h.e'_6.h
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    e_3✝ : Eq MeasureTheory.MeasureSpace.toMeasurableSpace inst✝³
    ⊢ Eq MeasureTheory.MeasureSpace.volume (MeasureTheory.Measure.map (⇑b.measurab …
  -/
  rw [← (EuclideanSpace.basisFun ι ℝ).addHaar_eq_volume]
  /-
    case h.e'_6.h
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    e_3✝ : Eq MeasureTheory.MeasureSpace.toMeasurableSpace inst✝³
    ⊢ Eq MeasureTheory.MeasureSpace.volume (MeasureTheory.Measure.map (⇑b.measurab …
  -/
  erw [MeasurableEquiv.coe_toEquiv_symm, Basis.map_addHaar _ b.repr.symm.toContinuousLinearEquiv]
  /-
    case h.e'_6.h
    ι : Type u_1
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : InnerProductSpace Real F
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : Fintype ι
    inst✝ : FiniteDimensional Real F
    b : OrthonormalBasis ι Real F
    e_3✝ : Eq MeasureTheory.MeasureSpace.toMeasurableSpace inst✝³
    ⊢ Eq MeasureTheory.MeasureSpace.volume ((EuclideanSpace.basisFun ι Real).toBas …
  -/
  exact b.addHaar_eq_volume.symm
  /-
    🎉 no goals
  -/


theorem OrthonormalBasis.measurePreserving_repr (b : OrthonormalBasis ι ℝ F) :
    MeasurePreserving b.repr volume volume := b.measurePreserving_measurableEquiv


theorem OrthonormalBasis.measurePreserving_repr_symm (b : OrthonormalBasis ι ℝ F) :
    MeasurePreserving b.repr.symm volume volume := b.measurePreserving_measurableEquiv.symm


/-- The measure equivalence between `EuclideanSpace ℝ ι` and `ι → ℝ` is volume preserving. -/
theorem EuclideanSpace.volume_preserving_measurableEquiv :
    /-
      ι✝ : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹¹ : NormedAddCommGroup F
      inst✝¹⁰ : InnerProductSpace Real F
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : InnerProductSpace Real E
      inst✝⁷ : MeasurableSpace E
      inst✝⁶ : BorelSpace E
      inst✝⁵ : MeasurableSpace F
      inst✝⁴ : BorelSpace F
      inst✝³ : Fintype ι✝
      inst✝² : FiniteDimensional Real E
      inst✝¹ : FiniteDimensional Real F
      ι : Type u_4
      inst✝ : Fintype ι
      ⊢ MeasureTheory.Measure (EuclideanSpace Real ι)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (EuclideanSpace.measurableEquiv ι) := by
    /-
      🎉 no goals
    -/
  suffices volume = map (EuclideanSpace.measurableEquiv ι).symm volume by
    convert ((EuclideanSpace.measurableEquiv ι).symm.measurable.measurePreserving _).symm
  rw [← addHaarMeasure_eq_volume_pi, ← Basis.parallelepiped_basisFun, ← Basis.addHaar_def,
    coe_measurableEquiv_symm, ← PiLp.continuousLinearEquiv_symm_apply 2 ℝ, Basis.map_addHaar]
  /-
    ι : Type u_4
    inst✝ : Fintype ι
    ⊢ Eq MeasureTheory.MeasureSpace.volume ((Pi.basisFun Real ι).map (PiLp.continu …
  -/
  exact (EuclideanSpace.basisFun _ _).addHaar_eq_volume.symm
  /-
    🎉 no goals
  -/


/-- A copy of `EuclideanSpace.volume_preserving_measurableEquiv` for the canonical spelling of the
equivalence. -/
                                       /-
                                         ι✝ : Type u_1
                                         E : Type u_2
                                         F : Type u_3
                                         inst✝¹¹ : NormedAddCommGroup F
                                         inst✝¹⁰ : InnerProductSpace Real F
                                         inst✝⁹ : NormedAddCommGroup E
                                         inst✝⁸ : InnerProductSpace Real E
                                         inst✝⁷ : MeasurableSpace E
                                         inst✝⁶ : BorelSpace E
                                         inst✝⁵ : MeasurableSpace F
                                         inst✝⁴ : BorelSpace F
                                         inst✝³ : Fintype ι✝
                                         inst✝² : FiniteDimensional Real E
                                         inst✝¹ : FiniteDimensional Real F
                                         ι : Type u_4
                                         inst✝ : Fintype ι
                                         ⊢ MeasureTheory.Measure (WithLp 2 (ι → Real))
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
theorem PiLp.volume_preserving_equiv : MeasurePreserving (WithLp.equiv 2 (ι → ℝ)) :=
                                       /-
                                         🎉 no goals
                                       -/
  EuclideanSpace.volume_preserving_measurableEquiv ι


/-- The reverse direction of `PiLp.volume_preserving_measurableEquiv`, since
`MeasurePreserving.symm` only works for `MeasurableEquiv`s. -/
                                            /-
                                              ι✝ : Type u_1
                                              E : Type u_2
                                              F : Type u_3
                                              inst✝¹¹ : NormedAddCommGroup F
                                              inst✝¹⁰ : InnerProductSpace Real F
                                              inst✝⁹ : NormedAddCommGroup E
                                              inst✝⁸ : InnerProductSpace Real E
                                              inst✝⁷ : MeasurableSpace E
                                              inst✝⁶ : BorelSpace E
                                              inst✝⁵ : MeasurableSpace F
                                              inst✝⁴ : BorelSpace F
                                              inst✝³ : Fintype ι✝
                                              inst✝² : FiniteDimensional Real E
                                              inst✝¹ : FiniteDimensional Real F
                                              ι : Type u_4
                                              inst✝ : Fintype ι
                                              ⊢ MeasureTheory.Measure (ι → Real)
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
theorem PiLp.volume_preserving_equiv_symm : MeasurePreserving (WithLp.equiv 2 (ι → ℝ)).symm :=
                                            /-
                                              🎉 no goals
                                            -/
  (EuclideanSpace.volume_preserving_measurableEquiv ι).symm


lemma volume_euclideanSpace_eq_dirac [IsEmpty ι] :
    (volume : Measure (EuclideanSpace ℝ ι)) = Measure.dirac 0 := by
  rw [← ((EuclideanSpace.volume_preserving_measurableEquiv ι).symm).map_eq,
    volume_pi_eq_dirac 0, map_dirac (MeasurableEquiv.measurable _),
    EuclideanSpace.coe_measurableEquiv_symm, WithLp.equiv_symm_zero]


/-- Every linear isometry on a real finite dimensional Hilbert space is measure-preserving. -/
theorem measurePreserving (f : E ≃ₗᵢ[ℝ] F) :
    /-
      ι : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : InnerProductSpace Real F
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : InnerProductSpace Real E
      inst✝⁶ : MeasurableSpace E
      inst✝⁵ : BorelSpace E
      inst✝⁴ : MeasurableSpace F
      inst✝³ : BorelSpace F
      inst✝² : Fintype ι
      inst✝¹ : FiniteDimensional Real E
      inst✝ : FiniteDimensional Real F
      f : LinearIsometryEquiv (RingHom.id Real) E F
      ⊢ MeasureTheory.Measure E
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving f := by
    /-
      🎉 no goals
    -/
  /-
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : InnerProductSpace Real F
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : FiniteDimensional Real E
    inst✝ : FiniteDimensional Real F
    f : LinearIsometryEquiv (RingHom.id Real) E F
    ⊢ MeasureTheory.MeasurePreserving (⇑f) MeasureTheory.MeasureSpace.volume Measu …
  -/
  refine ⟨f.continuous.measurable, ?_⟩
  /-
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : InnerProductSpace Real F
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : FiniteDimensional Real E
    inst✝ : FiniteDimensional Real F
    f : LinearIsometryEquiv (RingHom.id Real) E F
    ⊢ Eq (MeasureTheory.Measure.map (⇑f) MeasureTheory.MeasureSpace.volume) Measur …
  -/
  rcases exists_orthonormalBasis ℝ E with ⟨w, b, _hw⟩
  erw [← OrthonormalBasis.addHaar_eq_volume b, ← OrthonormalBasis.addHaar_eq_volume (b.map f),
    Basis.map_addHaar _ f.toContinuousLinearEquiv]
  /-
    case intro.intro
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : InnerProductSpace Real F
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : InnerProductSpace Real E
    inst✝⁵ : MeasurableSpace E
    inst✝⁴ : BorelSpace E
    inst✝³ : MeasurableSpace F
    inst✝² : BorelSpace F
    inst✝¹ : FiniteDimensional Real E
    inst✝ : FiniteDimensional Real F
    f : LinearIsometryEquiv (RingHom.id Real) E F
    w : Finset E
    b : OrthonormalBasis (Subtype fun x => Membership.mem w x) Real E
    _hw : Eq (⇑b) Subtype.val
    ⊢ Eq (b.toBasis.map f.toContinuousLinearEquiv.toLinearEquiv).addHaar (b.map f) …
  -/
  congr
  /-
    🎉 no goals
  -/


