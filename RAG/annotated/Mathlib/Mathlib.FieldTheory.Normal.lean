/-- Typeclass for normal field extension: `K` is a normal extension of `F` iff the minimal
polynomial of every element `x` in `K` splits in `K`, i.e. every conjugate of `x` is in `K`. -/
@[stacks 09HM]
class Normal extends Algebra.IsAlgebraic F K : Prop where
  splits' (x : K) : Splits (algebraMap F K) (minpoly F x)


theorem Normal.isIntegral (_ : Normal F K) (x : K) : IsIntegral F x :=
  Algebra.IsIntegral.isIntegral x


theorem Normal.splits (_ : Normal F K) (x : K) : Splits (algebraMap F K) (minpoly F x) :=
  Normal.splits' x


theorem normal_iff : Normal F K ↔ ∀ x : K, IsIntegral F x ∧ Splits (algebraMap F K) (minpoly F x) :=
  ⟨fun h x => ⟨h.isIntegral x, h.splits x⟩, fun h =>
    { isAlgebraic := fun x => (h x).1.isAlgebraic
      splits' := fun x => (h x).2 }⟩


theorem Normal.out : Normal F K → ∀ x : K, IsIntegral F x ∧ Splits (algebraMap F K) (minpoly F x) :=
  normal_iff.1


instance normal_self : Normal F F where
  isAlgebraic := fun _ => isIntegral_algebraMap.isAlgebraic
  splits' := fun x => (minpoly.eq_X_sub_C' x).symm ▸ splits_X_sub_C _


theorem Normal.exists_isSplittingField [h : Normal F K] [FiniteDimensional F K] :
    ∃ p : F[X], IsSplittingField F K p := by
  classical
  let s := Basis.ofVectorSpace F K
  refine
    ⟨∏ x, minpoly F (s x), splits_prod _ fun x _ => h.splits (s x),
      Subalgebra.toSubmodule.injective ?_⟩
  rw [Algebra.top_toSubmodule, eq_top_iff, ← s.span_eq, Submodule.span_le, Set.range_subset_iff]
  refine fun x =>
    Algebra.subset_adjoin
      (Multiset.mem_toFinset.mpr <|
        (mem_roots <|
              mt (Polynomial.map_eq_zero <| algebraMap F K).1 <|
                Finset.prod_ne_zero_iff.2 fun x _ => ?_).2 ?_)
  · exact minpoly.ne_zero (h.isIntegral (s x))
  rw [IsRoot.def, eval_map, ← aeval_def, map_prod]
  exact Finset.prod_eq_zero (Finset.mem_univ _) (minpoly.aeval _ _)


@[stacks 09HN]
theorem Normal.tower_top_of_normal [h : Normal F E] : Normal K E :=
  normal_iff.2 fun x => by
    /-
      F : Type u_1
      K : Type u_2
      inst✝⁶ : Field F
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      E : Type u_3
      inst✝³ : Field E
      inst✝² : Algebra F E
      inst✝¹ : Algebra K E
      inst✝ : IsScalarTower F K E
      h : Normal F E
      x : E
      ⊢ And (IsIntegral K x) (Polynomial.Splits (algebraMap K E) (minpoly K x))
    -/
    cases' h.out x with hx hhx
    /-
      case intro
      F : Type u_1
      K : Type u_2
      inst✝⁶ : Field F
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      E : Type u_3
      inst✝³ : Field E
      inst✝² : Algebra F E
      inst✝¹ : Algebra K E
      inst✝ : IsScalarTower F K E
      h : Normal F E
      x : E
      hx : IsIntegral F x
      hhx : Polynomial.Splits (algebraMap F E) (minpoly F x)
      ⊢ And (IsIntegral K x) (Polynomial.Splits (algebraMap K E) (minpoly K x))
    -/
    rw [algebraMap_eq F K E] at hhx
    exact
      ⟨hx.tower_top,
        Polynomial.splits_of_splits_of_dvd (algebraMap K E)
          (Polynomial.map_ne_zero (minpoly.ne_zero hx))
          ((Polynomial.splits_map_iff (algebraMap F K) (algebraMap K E)).mpr hhx)
          (minpoly.dvd_map_of_isScalarTower F K x)⟩


theorem AlgHom.normal_bijective [h : Normal F E] (ϕ : E →ₐ[F] K) : Function.Bijective ϕ :=
  h.toIsAlgebraic.bijective_of_isScalarTower' ϕ


theorem Normal.of_algEquiv [h : Normal F E] (f : E ≃ₐ[F] E') : Normal F E' := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    inst✝³ : Field E
    inst✝² : Algebra F E
    E' : Type u_4
    inst✝¹ : Field E'
    inst✝ : Algebra F E'
    h : Normal F E
    f : AlgEquiv F E E'
    ⊢ Normal F E'
  -/
  rw [normal_iff] at h ⊢
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    inst✝³ : Field E
    inst✝² : Algebra F E
    E' : Type u_4
    inst✝¹ : Field E'
    inst✝ : Algebra F E'
    h : ∀ (x : E), And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpo …
    f : AlgEquiv F E E'
    ⊢ ∀ (x : E'), And (IsIntegral F x) (Polynomial.Splits (algebraMap F E') (minpo …
  -/
  intro x; specialize h (f.symm x)
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    inst✝³ : Field E
    inst✝² : Algebra F E
    E' : Type u_4
    inst✝¹ : Field E'
    inst✝ : Algebra F E'
    f : AlgEquiv F E E'
    x : E'
    h : And (IsIntegral F (f.symm x)) (Polynomial.Splits (algebraMap F E) (minpoly …
    ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E') (minpoly F x))
  -/
  rw [← f.apply_symm_apply x, minpoly.algEquiv_eq, ← f.toAlgHom.comp_algebraMap]
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_3
    inst✝³ : Field E
    inst✝² : Algebra F E
    E' : Type u_4
    inst✝¹ : Field E'
    inst✝ : Algebra F E'
    f : AlgEquiv F E E'
    x : E'
    h : And (IsIntegral F (f.symm x)) (Polynomial.Splits (algebraMap F E) (minpoly …
    ⊢ And (IsIntegral F (f (f.symm x))) (Polynomial.Splits ((↑↑f).comp (algebraMap …
  -/
  exact ⟨h.1.map f, splits_comp_of_splits _ _ h.2⟩
  /-
    🎉 no goals
  -/


theorem AlgEquiv.transfer_normal (f : E ≃ₐ[F] E') : Normal F E ↔ Normal F E' :=
  ⟨fun _ ↦ Normal.of_algEquiv f, fun _ ↦ Normal.of_algEquiv f.symm⟩


@[stacks 09HU "Normal part"]
theorem Normal.of_isSplittingField (p : F[X]) [hFEp : IsSplittingField F E p] : Normal F E := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    ⊢ Normal F E
  -/
  rcases eq_or_ne p 0 with (rfl | hp)
    /-
      case inl
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hFEp : Polynomial.IsSplittingField F E 0
      ⊢ Normal F E
    -/
  · have := hFEp.adjoin_rootSet
    /-
      case inl
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      hFEp : Polynomial.IsSplittingField F E 0
      this : Eq (Algebra.adjoin F (Polynomial.rootSet 0 E)) Top.top
      ⊢ Normal F E
    -/
    rw [rootSet_zero, Algebra.adjoin_empty] at this
    exact Normal.of_algEquiv
      (AlgEquiv.ofBijective (Algebra.ofId F E) (Algebra.bijective_algebraMap_iff.2 this.symm))
  /-
    case inr
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    hp : Ne p 0
    ⊢ Normal F E
  -/
  refine normal_iff.mpr fun x ↦ ?_
  /-
    case inr
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    hp : Ne p 0
    x : E
    ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
  -/
  haveI : FiniteDimensional F E := IsSplittingField.finiteDimensional E p
  /-
    case inr
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    hp : Ne p 0
    x : E
    this : FiniteDimensional F E
    ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
  -/
  have hx := IsIntegral.of_finite F x
  /-
    case inr
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    hp : Ne p 0
    x : E
    this : FiniteDimensional F E
    hx : IsIntegral F x
    ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
  -/
  let L := (p * minpoly F x).SplittingField
  /-
    case inr
    F : Type u_1
    inst✝² : Field F
    E : Type u_3
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    hFEp : Polynomial.IsSplittingField F E p
    hp : Ne p 0
    x : E
    this : FiniteDimensional F E
    hx : IsIntegral F x
    L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
    ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
  -/
  have hL := splits_of_splits_mul' _ ?_ (SplittingField.splits (p * minpoly F x))
    /-
      case inr.refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
      ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
    -/
  · let j : E →ₐ[F] L := IsSplittingField.lift E p hL.1
    /-
      case inr.refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
      j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
      ⊢ And (IsIntegral F x) (Polynomial.Splits (algebraMap F E) (minpoly F x))
    -/
    refine ⟨hx, splits_of_comp _ (j : E →+* L) (j.comp_algebraMap ▸ hL.2) fun a ha ↦ ?_⟩
    /-
      case inr.refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
      j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
      a : L
      ha : Membership.mem (Polynomial.map ((↑j).comp (algebraMap F E)) (minpoly F x) …
      ⊢ Membership.mem (↑j).range a
    -/
    rw [j.comp_algebraMap] at ha
    /-
      case inr.refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
      j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
      a : L
      ha : Membership.mem (Polynomial.map (algebraMap F L) (minpoly F x)).roots a
      ⊢ Membership.mem (↑j).range a
    -/
    letI : Algebra F⟮x⟯ L := ((algHomAdjoinIntegralEquiv F hx).symm ⟨a, ha⟩).toRingHom.toAlgebra
    /-
      case inr.refine_2
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this✝ : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
      j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
      a : L
      ha : Membership.mem (Polynomial.map (algebraMap F L) (minpoly F x)).roots a
      this : Algebra (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F  …
      ⊢ Membership.mem (↑j).range a
    -/
    let j' : E →ₐ[F⟮x⟯] L := IsSplittingField.lift E (p.map (algebraMap F F⟮x⟯)) ?_
      /-
        case inr.refine_2.refine_2
        F : Type u_1
        inst✝² : Field F
        E : Type u_3
        inst✝¹ : Field E
        inst✝ : Algebra F E
        p : Polynomial F
        hFEp : Polynomial.IsSplittingField F E p
        hp : Ne p 0
        x : E
        this✝ : FiniteDimensional F E
        hx : IsIntegral F x
        L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
        hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
        j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
        a : L
        ha : Membership.mem (Polynomial.map (algebraMap F L) (minpoly F x)).roots a
        this : Algebra (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F  …
        j' : AlgHom (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F (Si …
        ⊢ Membership.mem (↑j).range a
      -/
    · change a ∈ j.range
      rw [← IsSplittingField.adjoin_rootSet_eq_range E p j,
            IsSplittingField.adjoin_rootSet_eq_range E p (j'.restrictScalars F)]
      /-
        case inr.refine_2.refine_2
        F : Type u_1
        inst✝² : Field F
        E : Type u_3
        inst✝¹ : Field E
        inst✝ : Algebra F E
        p : Polynomial F
        hFEp : Polynomial.IsSplittingField F E p
        hp : Ne p 0
        x : E
        this✝ : FiniteDimensional F E
        hx : IsIntegral F x
        L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
        hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
        j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
        a : L
        ha : Membership.mem (Polynomial.map (algebraMap F L) (minpoly F x)).roots a
        this : Algebra (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F  …
        j' : AlgHom (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F (Si …
        ⊢ Membership.mem (AlgHom.restrictScalars F j').range a
      -/
      exact ⟨x, (j'.commutes _).trans (algHomAdjoinIntegralEquiv_symm_apply_gen F hx _)⟩
      /-
        🎉 no goals
      -/
      /-
        case inr.refine_2.refine_1
        F : Type u_1
        inst✝² : Field F
        E : Type u_3
        inst✝¹ : Field E
        inst✝ : Algebra F E
        p : Polynomial F
        hFEp : Polynomial.IsSplittingField F E p
        hp : Ne p 0
        x : E
        this✝ : FiniteDimensional F E
        hx : IsIntegral F x
        L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
        hL : And (Polynomial.Splits (algebraMap F (HMul.hMul p (minpoly F x)).Splittin …
        j : AlgHom F E L := Polynomial.IsSplittingField.lift E p ⋯
        a : L
        ha : Membership.mem (Polynomial.map (algebraMap F L) (minpoly F x)).roots a
        this : Algebra (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin F  …
        ⊢ Polynomial.Splits (algebraMap (Subtype fun x_1 => Membership.mem (Intermedia …
      -/
    · rw [splits_map_iff, ← IsScalarTower.algebraMap_eq]; exact hL.1
                                                          /-
                                                            🎉 no goals
                                                          -/
    /-
      case inr.refine_1
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      ⊢ Ne (Polynomial.map (algebraMap F (HMul.hMul p (minpoly F x)).SplittingField) …
    -/
  · rw [Polynomial.map_ne_zero_iff (algebraMap F L).injective, mul_ne_zero_iff]
    /-
      case inr.refine_1
      F : Type u_1
      inst✝² : Field F
      E : Type u_3
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      hFEp : Polynomial.IsSplittingField F E p
      hp : Ne p 0
      x : E
      this : FiniteDimensional F E
      hx : IsIntegral F x
      L : Type u_1 := (HMul.hMul p (minpoly F x)).SplittingField
      ⊢ And (Ne p 0) (Ne (minpoly F x) 0)
    -/
    exact ⟨hp, minpoly.ne_zero hx⟩
    /-
      🎉 no goals
    -/


instance Polynomial.SplittingField.instNormal (p : F[X]) : Normal F p.SplittingField :=
  Normal.of_isSplittingField p


/-- A compositum of normal extensions is normal. -/
instance normal_iSup {ι : Type*} (t : ι → IntermediateField F K) [h : ∀ i, Normal F (t i)] :
    Normal F (⨆ i, t i : IntermediateField F K) := by
  /-
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    ⊢ Normal F (Subtype fun x => Membership.mem (iSup fun i => t i) x)
  -/
  refine { toIsAlgebraic := isAlgebraic_iSup fun i => (h i).1, splits' := fun x => ?_ }
  /-
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    x : Subtype fun x => Membership.mem (iSup fun i => t i) x
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iSup fun i …
  -/
  obtain ⟨s, hx⟩ := exists_finset_of_mem_supr'' (fun i => (h i).1) x.2
  /-
    case intro
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    x : Subtype fun x => Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin F (( …
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iSup fun i …
  -/
  let E : IntermediateField F K := ⨆ i ∈ s, adjoin F ((minpoly F (i.2 : _)).rootSet K)
  have hF : Normal F E := by
    haveI : IsSplittingField F E (∏ i ∈ s, minpoly F i.snd) := by
      refine isSplittingField_iSup ?_ fun i _ => adjoin_rootSet_isSplittingField ?_
      · exact Finset.prod_ne_zero_iff.mpr fun i _ => minpoly.ne_zero ((h i.1).isIntegral i.2)
      · exact Polynomial.splits_comp_of_splits _ (algebraMap (t i.1) K) ((h i.1).splits i.2)
    apply Normal.of_isSplittingField (∏ i ∈ s, minpoly F i.2)
  have hE : E ≤ ⨆ i, t i := by
    refine iSup_le fun i => iSup_le fun _ => le_iSup_of_le i.1 ?_
    rw [adjoin_le_iff, ← image_rootSet ((h i.1).splits i.2) (t i.1).val]
    exact fun _ ⟨a, _, h⟩ => h ▸ a.2
  /-
    case intro
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    x : Subtype fun x => Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin F (( …
    E : IntermediateField F K := iSup fun i => iSup fun h => IntermediateField.adj …
    hF : Normal F (Subtype fun x => Membership.mem E x)
    hE : LE.le E (iSup fun i => t i)
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iSup fun i …
  -/
  have := hF.splits ⟨x, hx⟩
  /-
    case intro
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    x : Subtype fun x => Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin F (( …
    E : IntermediateField F K := iSup fun i => iSup fun h => IntermediateField.adj …
    hF : Normal F (Subtype fun x => Membership.mem E x)
    hE : LE.le E (iSup fun i => t i)
    this : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem E x))  …
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iSup fun i …
  -/
  rw [minpoly_eq, Subtype.coe_mk, ← minpoly_eq] at this
  /-
    case intro
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    x : Subtype fun x => Membership.mem (iSup fun i => t i) x
    s : Finset (Sigma fun i => Subtype fun x => Membership.mem (t i) x)
    hx : Membership.mem (iSup fun i => iSup fun h => IntermediateField.adjoin F (( …
    E : IntermediateField F K := iSup fun i => iSup fun h => IntermediateField.adj …
    hF : Normal F (Subtype fun x => Membership.mem E x)
    hE : LE.le E (iSup fun i => t i)
    this : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem E x))  …
    ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iSup fun i …
  -/
  exact Polynomial.splits_comp_of_splits _ (inclusion hE).toRingHom this
  /-
    🎉 no goals
  -/


/-- If a set of algebraic elements in a field extension `K/F` have minimal polynomials that
  split in another extension `L/F`, then all minimal polynomials in the intermediate field
  generated by the set also split in `L/F`. -/
@[stacks 0BR3 "first part"]
theorem splits_of_mem_adjoin {L} [Field L] [Algebra F L] {S : Set K}
    (splits : ∀ x ∈ S, IsIntegral F x ∧ (minpoly F x).Splits (algebraMap F L)) {x : K}
    (hx : x ∈ adjoin F S) : (minpoly F x).Splits (algebraMap F L) := by
  /-
    F : Type u_1
    K : Type u_2
    inst✝⁴ : Field F
    inst✝³ : Field K
    inst✝² : Algebra F K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra F L
    S : Set K
    splits : ∀ (x : K), Membership.mem S x → And (IsIntegral F x) (Polynomial.Spli …
    x : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    ⊢ Polynomial.Splits (algebraMap F L) (minpoly F x)
  -/
  let E : IntermediateField F L := ⨆ x : S, adjoin F ((minpoly F x.val).rootSet L)
  have normal : Normal F E := normal_iSup (h := fun x ↦
    Normal.of_isSplittingField (hFEp := adjoin_rootSet_isSplittingField (splits x x.2).2))
  have : ∀ x ∈ S, (minpoly F x).Splits (algebraMap F E) := fun x hx ↦ splits_of_splits
    (splits x hx).2 fun y hy ↦ (le_iSup _ ⟨x, hx⟩ : _ ≤ E) (subset_adjoin F _ <| by exact hy)
  /-
    F : Type u_1
    K : Type u_2
    inst✝⁴ : Field F
    inst✝³ : Field K
    inst✝² : Algebra F K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra F L
    S : Set K
    splits : ∀ (x : K), Membership.mem S x → And (IsIntegral F x) (Polynomial.Spli …
    x : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    E : IntermediateField F L := iSup fun x => IntermediateField.adjoin F ((minpol …
    normal : Normal F (Subtype fun x => Membership.mem E x)
    this : ∀ (x : K), Membership.mem S x → Polynomial.Splits (algebraMap F (Subtyp …
    ⊢ Polynomial.Splits (algebraMap F L) (minpoly F x)
  -/
  obtain ⟨φ⟩ := nonempty_algHom_adjoin_of_splits fun x hx ↦ ⟨(splits x hx).1, this x hx⟩
  /-
    case intro
    F : Type u_1
    K : Type u_2
    inst✝⁴ : Field F
    inst✝³ : Field K
    inst✝² : Algebra F K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra F L
    S : Set K
    splits : ∀ (x : K), Membership.mem S x → And (IsIntegral F x) (Polynomial.Spli …
    x : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    E : IntermediateField F L := iSup fun x => IntermediateField.adjoin F ((minpol …
    normal : Normal F (Subtype fun x => Membership.mem E x)
    this : ∀ (x : K), Membership.mem S x → Polynomial.Splits (algebraMap F (Subtyp …
    φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F S) x …
    ⊢ Polynomial.Splits (algebraMap F L) (minpoly F x)
  -/
  convert splits_comp_of_splits _ E.val.toRingHom (normal.splits <| φ ⟨x, hx⟩)
  /-
    case h.e'_6
    F : Type u_1
    K : Type u_2
    inst✝⁴ : Field F
    inst✝³ : Field K
    inst✝² : Algebra F K
    L : Type u_3
    inst✝¹ : Field L
    inst✝ : Algebra F L
    S : Set K
    splits : ∀ (x : K), Membership.mem S x → And (IsIntegral F x) (Polynomial.Spli …
    x : K
    hx : Membership.mem (IntermediateField.adjoin F S) x
    E : IntermediateField F L := iSup fun x => IntermediateField.adjoin F ((minpol …
    normal : Normal F (Subtype fun x => Membership.mem E x)
    this : ∀ (x : K), Membership.mem S x → Polynomial.Splits (algebraMap F (Subtyp …
    φ : AlgHom F (Subtype fun x => Membership.mem (IntermediateField.adjoin F S) x …
    ⊢ Eq (minpoly F x) (minpoly F (φ ⟨x, hx⟩))
  -/
  rw [minpoly.algHom_eq _ φ.injective, ← minpoly.algHom_eq _ (adjoin F S).val.injective, val_mk]
  /-
    🎉 no goals
  -/


instance normal_sup
    (E E' : IntermediateField F K) [Normal F E] [Normal F E'] :
    Normal F (E ⊔ E' : IntermediateField F K) :=
                                                           /-
                                                             F : Type u_1
                                                             K : Type u_2
                                                             inst✝⁴ : Field F
                                                             inst✝³ : Field K
                                                             inst✝² : Algebra F K
                                                             E E' : IntermediateField F K
                                                             inst✝¹ : Normal F (Subtype fun x => Membership.mem E x)
                                                             inst✝ : Normal F (Subtype fun x => Membership.mem E' x)
                                                             ⊢ ∀ (i : Bool), Normal F (Subtype fun x => Membership.mem (Bool.rec E' E i) x)
                                                           -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  iSup_bool_eq (f := Bool.rec E' E) ▸ normal_iSup (h := by rintro (_|_) <;> infer_instance)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- An intersection of normal extensions is normal. -/
@[stacks 09HP]
instance normal_iInf {ι : Type*} [hι : Nonempty ι]
    (t : ι → IntermediateField F K) [h : ∀ i, Normal F (t i)] :
    Normal F (⨅ i, t i : IntermediateField F K) := by
  /-
    F : Type u_1
    K : Type u_2
    inst✝² : Field F
    inst✝¹ : Field K
    inst✝ : Algebra F K
    ι : Type u_3
    hι : Nonempty ι
    t : ι → IntermediateField F K
    h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
    ⊢ Normal F (Subtype fun x => Membership.mem (iInf fun i => t i) x)
  -/
  refine { toIsAlgebraic := ?_, splits' := fun x => ?_ }
    /-
      case refine_1
      F : Type u_1
      K : Type u_2
      inst✝² : Field F
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ι : Type u_3
      hι : Nonempty ι
      t : ι → IntermediateField F K
      h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
      ⊢ Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (iInf fun i => t i) x)
    -/
  · let f := inclusion (iInf_le t hι.some)
    /-
      case refine_1
      F : Type u_1
      K : Type u_2
      inst✝² : Field F
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ι : Type u_3
      hι : Nonempty ι
      t : ι → IntermediateField F K
      h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
      f : AlgHom F (Subtype fun x => Membership.mem (iInf t) x) (Subtype fun x => Me …
      ⊢ Algebra.IsAlgebraic F (Subtype fun x => Membership.mem (iInf fun i => t i) x)
    -/
    exact Algebra.IsAlgebraic.of_injective f f.injective
    /-
      🎉 no goals
    -/
  · have hx : ∀ i, Splits (algebraMap F (t i)) (minpoly F x) := by
      intro i
      rw [← minpoly.algHom_eq (inclusion (iInf_le t i)) (inclusion (iInf_le t i)).injective]
      exact (h i).splits' (inclusion (iInf_le t i) x)
    /-
      case refine_2
      F : Type u_1
      K : Type u_2
      inst✝² : Field F
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ι : Type u_3
      hι : Nonempty ι
      t : ι → IntermediateField F K
      h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
      x : Subtype fun x => Membership.mem (iInf fun i => t i) x
      hx : ∀ (i : ι), Polynomial.Splits (algebraMap F (Subtype fun x => Membership.m …
      ⊢ Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (iInf fun i …
    -/
    simp only [splits_iff_mem (splits_of_isScalarTower K (hx hι.some))] at hx ⊢
    /-
      case refine_2
      F : Type u_1
      K : Type u_2
      inst✝² : Field F
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ι : Type u_3
      hι : Nonempty ι
      t : ι → IntermediateField F K
      h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
      x : Subtype fun x => Membership.mem (iInf fun i => t i) x
      hx : ∀ (i : ι) (x_1 : K), Membership.mem ((minpoly F x).rootSet K) x_1 → Membe …
      ⊢ ∀ (x_1 : K), Membership.mem ((minpoly F x).rootSet K) x_1 → Membership.mem ( …
    -/
    rintro y hy - ⟨-, ⟨i, rfl⟩, rfl⟩
    /-
      case refine_2.intro.intro.intro
      F : Type u_1
      K : Type u_2
      inst✝² : Field F
      inst✝¹ : Field K
      inst✝ : Algebra F K
      ι : Type u_3
      hι : Nonempty ι
      t : ι → IntermediateField F K
      h : ∀ (i : ι), Normal F (Subtype fun x => Membership.mem (t i) x)
      x : Subtype fun x => Membership.mem (iInf fun i => t i) x
      hx : ∀ (i : ι) (x_1 : K), Membership.mem ((minpoly F x).rootSet K) x_1 → Membe …
      y : K
      hy : Membership.mem ((minpoly F x).rootSet K) y
      i : ι
      ⊢ Membership.mem ((fun x => ↑x) ((fun i => t i) i)) y
    -/
    exact hx i y hy
    /-
      🎉 no goals
    -/


@[stacks 09HP]
instance normal_inf
    (E E' : IntermediateField F K) [Normal F E] [Normal F E'] :
    Normal F (E ⊓ E' : IntermediateField F K) :=
                                                           /-
                                                             F : Type u_1
                                                             K : Type u_2
                                                             inst✝⁴ : Field F
                                                             inst✝³ : Field K
                                                             inst✝² : Algebra F K
                                                             E E' : IntermediateField F K
                                                             inst✝¹ : Normal F (Subtype fun x => Membership.mem E x)
                                                             inst✝ : Normal F (Subtype fun x => Membership.mem E' x)
                                                             ⊢ ∀ (i : Bool), Normal F (Subtype fun x => Membership.mem (Bool.rec E' E i) x)
                                                           -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  iInf_bool_eq (f := Bool.rec E' E) ▸ normal_iInf (h := by rintro (_|_) <;> infer_instance)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem restrictScalars_normal {E : IntermediateField K L} :
    Normal F (E.restrictScalars F) ↔ Normal F E :=
  Iff.rfl


/-- Restrict algebra homomorphism to image of normal subfield -/
def AlgHom.restrictNormalAux [h : Normal F E] :
    (toAlgHom F E K₁).range →ₐ[F] (toAlgHom F E K₂).range where
  toFun x :=
    ⟨ϕ x, by
      /-
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        ⊢ Membership.mem (IsScalarTower.toAlgHom F E K₂).range (ϕ ↑x)
      -/
      suffices (toAlgHom F E K₁).range.map ϕ ≤ _ by exact this ⟨x, Subtype.mem x, rfl⟩
      /-
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        ⊢ LE.le (Subalgebra.map ϕ (IsScalarTower.toAlgHom F E K₁).range) (IsScalarTowe …
      -/
      rintro x ⟨y, ⟨z, hy⟩, hx⟩
      /-
        case intro.intro.intro
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x✝ : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        x : K₂
        y : K₁
        hx : Eq (↑ϕ y) x
        z : E
        hy : Eq ((IsScalarTower.toAlgHom F E K₁).toRingHom z) y
        ⊢ Membership.mem (IsScalarTower.toAlgHom F E K₂).range x
      -/
      rw [← hx, ← hy]
      /-
        case intro.intro.intro
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x✝ : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        x : K₂
        y : K₁
        hx : Eq (↑ϕ y) x
        z : E
        hy : Eq ((IsScalarTower.toAlgHom F E K₁).toRingHom z) y
        ⊢ Membership.mem (IsScalarTower.toAlgHom F E K₂).range (↑ϕ ((IsScalarTower.toA …
      -/
      apply minpoly.mem_range_of_degree_eq_one E
      refine
        Or.resolve_left (h.splits z).def (minpoly.ne_zero (h.isIntegral z)) (minpoly.irreducible ?_)
          (minpoly.dvd E _ (by simp [aeval_algHom_apply]))
      /-
        case intro.intro.intro.hx
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x✝ : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        x : K₂
        y : K₁
        hx : Eq (↑ϕ y) x
        z : E
        hy : Eq ((IsScalarTower.toAlgHom F E K₁).toRingHom z) y
        ⊢ IsIntegral E (↑ϕ ((IsScalarTower.toAlgHom F E K₁).toRingHom z))
      -/
      simp only [AlgHom.toRingHom_eq_coe, AlgHom.coe_toRingHom]
      /-
        case intro.intro.intro.hx
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x✝ : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        x : K₂
        y : K₁
        hx : Eq (↑ϕ y) x
        z : E
        hy : Eq ((IsScalarTower.toAlgHom F E K₁).toRingHom z) y
        ⊢ IsIntegral E (ϕ ((IsScalarTower.toAlgHom F E K₁) z))
      -/
      suffices IsIntegral F _ by exact this.tower_top
      /-
        case intro.intro.intro.hx
        F : Type u_1
        K : Type u_2
        inst✝¹⁶ : Field F
        inst✝¹⁵ : Field K
        inst✝¹⁴ : Algebra F K
        K₁ : Type u_3
        K₂ : Type u_4
        K₃ : Type u_5
        inst✝¹³ : Field K₁
        inst✝¹² : Field K₂
        inst✝¹¹ : Field K₃
        inst✝¹⁰ : Algebra F K₁
        inst✝⁹ : Algebra F K₂
        inst✝⁸ : Algebra F K₃
        ϕ : AlgHom F K₁ K₂
        χ : AlgEquiv F K₁ K₂
        ψ : AlgHom F K₂ K₃
        ω : AlgEquiv F K₂ K₃
        E : Type u_6
        inst✝⁷ : Field E
        inst✝⁶ : Algebra F E
        inst✝⁵ : Algebra E K₁
        inst✝⁴ : Algebra E K₂
        inst✝³ : Algebra E K₃
        inst✝² : IsScalarTower F E K₁
        inst✝¹ : IsScalarTower F E K₂
        inst✝ : IsScalarTower F E K₃
        h : Normal F E
        x✝ : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
        x : K₂
        y : K₁
        hx : Eq (↑ϕ y) x
        z : E
        hy : Eq ((IsScalarTower.toAlgHom F E K₁).toRingHom z) y
        ⊢ IsIntegral F (ϕ ((IsScalarTower.toAlgHom F E K₁) z))
      -/
      exact ((h.isIntegral z).map <| toAlgHom F E K₁).map ϕ⟩
      /-
        🎉 no goals
      -/
  map_zero' := Subtype.ext (map_zero _)
  map_one' := Subtype.ext (map_one _)
                                    /-
                                      F : Type u_1
                                      K : Type u_2
                                      inst✝¹⁶ : Field F
                                      inst✝¹⁵ : Field K
                                      inst✝¹⁴ : Algebra F K
                                      K₁ : Type u_3
                                      K₂ : Type u_4
                                      K₃ : Type u_5
                                      inst✝¹³ : Field K₁
                                      inst✝¹² : Field K₂
                                      inst✝¹¹ : Field K₃
                                      inst✝¹⁰ : Algebra F K₁
                                      inst✝⁹ : Algebra F K₂
                                      inst✝⁸ : Algebra F K₃
                                      ϕ : AlgHom F K₁ K₂
                                      χ : AlgEquiv F K₁ K₂
                                      ψ : AlgHom F K₂ K₃
                                      ω : AlgEquiv F K₂ K₃
                                      E : Type u_6
                                      inst✝⁷ : Field E
                                      inst✝⁶ : Algebra F E
                                      inst✝⁵ : Algebra E K₁
                                      inst✝⁴ : Algebra E K₂
                                      inst✝³ : Algebra E K₃
                                      inst✝² : IsScalarTower F E K₁
                                      inst✝¹ : IsScalarTower F E K₂
                                      inst✝ : IsScalarTower F E K₃
                                      h : Normal F E
                                      x y : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
                                      ⊢ Eq ↑((↑{ toFun := fun x => ⟨ϕ ↑x, ⋯⟩, map_one' := ⋯, map_mul' := ⋯ }).toFun  …
                                    -/
                                    /-
                                      F : Type u_1
                                      K : Type u_2
                                      inst✝¹⁶ : Field F
                                      inst✝¹⁵ : Field K
                                      inst✝¹⁴ : Algebra F K
                                      K₁ : Type u_3
                                      K₂ : Type u_4
                                      K₃ : Type u_5
                                      inst✝¹³ : Field K₁
                                      inst✝¹² : Field K₂
                                      inst✝¹¹ : Field K₃
                                      inst✝¹⁰ : Algebra F K₁
                                      inst✝⁹ : Algebra F K₂
                                      inst✝⁸ : Algebra F K₃
                                      ϕ : AlgHom F K₁ K₂
                                      χ : AlgEquiv F K₁ K₂
                                      ψ : AlgHom F K₂ K₃
                                      ω : AlgEquiv F K₂ K₃
                                      E : Type u_6
                                      inst✝⁷ : Field E
                                      inst✝⁶ : Algebra F E
                                      inst✝⁵ : Algebra E K₁
                                      inst✝⁴ : Algebra E K₂
                                      inst✝³ : Algebra E K₃
                                      inst✝² : IsScalarTower F E K₁
                                      inst✝¹ : IsScalarTower F E K₂
                                      inst✝ : IsScalarTower F E K₃
                                      h : Normal F E
                                      x y : Subtype fun x => Membership.mem (IsScalarTower.toAlgHom F E K₁).range x
                                      ⊢ Eq ↑({ toFun := fun x => ⟨ϕ ↑x, ⋯⟩, map_one' := ⋯ }.toFun (HMul.hMul x y)) ↑ …
                                    -/
  map_add' x y := Subtype.ext <| by simp
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  map_mul' x y := Subtype.ext <| by simp
  commutes' x := Subtype.ext (ϕ.commutes x)


/-- Restrict algebra homomorphism to normal subfield. -/
@[stacks 0BME "Part 1"]
def AlgHom.restrictNormal [Normal F E] : E →ₐ[F] E :=
  ((AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F E K₂)).symm.toAlgHom.comp
        (ϕ.restrictNormalAux E)).comp
    (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F E K₁)).toAlgHom


/-- Restrict algebra homomorphism to normal subfield (`AlgEquiv` version) -/
def AlgHom.restrictNormal' [Normal F E] : E ≃ₐ[F] E :=
  AlgEquiv.ofBijective (AlgHom.restrictNormal ϕ E) (AlgHom.normal_bijective F E E _)


@[simp]
theorem AlgHom.restrictNormal_commutes [Normal F E] (x : E) :
    algebraMap E K₂ (ϕ.restrictNormal E x) = ϕ (algebraMap E K₁ x) :=
  Subtype.ext_iff.mp
    (AlgEquiv.apply_symm_apply (AlgEquiv.ofInjectiveField (IsScalarTower.toAlgHom F E K₂))
      (ϕ.restrictNormalAux E ⟨IsScalarTower.toAlgHom F E K₁ x, x, rfl⟩))


theorem AlgHom.restrictNormal_comp [Normal F E] :
    (ψ.restrictNormal E).comp (ϕ.restrictNormal E) = (ψ.comp ϕ).restrictNormal E :=
  AlgHom.ext fun _ =>
                                    /-
                                      F : Type u_1
                                      inst✝¹⁵ : Field F
                                      K₁ : Type u_3
                                      K₂ : Type u_4
                                      K₃ : Type u_5
                                      inst✝¹⁴ : Field K₁
                                      inst✝¹³ : Field K₂
                                      inst✝¹² : Field K₃
                                      inst✝¹¹ : Algebra F K₁
                                      inst✝¹⁰ : Algebra F K₂
                                      inst✝⁹ : Algebra F K₃
                                      ϕ : AlgHom F K₁ K₂
                                      ψ : AlgHom F K₂ K₃
                                      E : Type u_6
                                      inst✝⁸ : Field E
                                      inst✝⁷ : Algebra F E
                                      inst✝⁶ : Algebra E K₁
                                      inst✝⁵ : Algebra E K₂
                                      inst✝⁴ : Algebra E K₃
                                      inst✝³ : IsScalarTower F E K₁
                                      inst✝² : IsScalarTower F E K₂
                                      inst✝¹ : IsScalarTower F E K₃
                                      inst✝ : Normal F E
                                      x✝ : E
                                      ⊢ Eq ((algebraMap E K₃) (((ψ.restrictNormal E).comp (ϕ.restrictNormal E)) x✝)) …
                                    -/
    (algebraMap E K₃).injective (by simp only [AlgHom.comp_apply, AlgHom.restrictNormal_commutes])
                                    /-
                                      🎉 no goals
                                    -/


theorem AlgHom.fieldRange_of_normal {E : IntermediateField F K} [Normal F E]
    (f : E →ₐ[F] K) : f.fieldRange = E := by
  /-
    F : Type u_1
    K : Type u_2
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    E : IntermediateField F K
    inst✝ : Normal F (Subtype fun x => Membership.mem E x)
    f : AlgHom F (Subtype fun x => Membership.mem E x) K
    ⊢ Eq f.fieldRange E
  -/
  let g := f.restrictNormal' E
  rw [← show E.val.comp ↑g = f from DFunLike.ext_iff.mpr (f.restrictNormal_commutes E),
    ← AlgHom.map_fieldRange, AlgEquiv.fieldRange_eq_top g, ← AlgHom.fieldRange_eq_map,
    IntermediateField.fieldRange_val]


/-- Restrict algebra isomorphism to a normal subfield -/
def AlgEquiv.restrictNormal [Normal F E] : E ≃ₐ[F] E :=
  AlgHom.restrictNormal' χ.toAlgHom E


@[simp]
theorem AlgEquiv.restrictNormal_commutes [Normal F E] (x : E) :
    algebraMap E K₂ (χ.restrictNormal E x) = χ (algebraMap E K₁ x) :=
  χ.toAlgHom.restrictNormal_commutes E x


theorem AlgEquiv.restrictNormal_trans [Normal F E] :
    (χ.trans ω).restrictNormal E = (χ.restrictNormal E).trans (ω.restrictNormal E) :=
  AlgEquiv.ext fun _ =>
    (algebraMap E K₃).injective
          /-
            F : Type u_1
            inst✝¹⁵ : Field F
            K₁ : Type u_3
            K₂ : Type u_4
            K₃ : Type u_5
            inst✝¹⁴ : Field K₁
            inst✝¹³ : Field K₂
            inst✝¹² : Field K₃
            inst✝¹¹ : Algebra F K₁
            inst✝¹⁰ : Algebra F K₂
            inst✝⁹ : Algebra F K₃
            χ : AlgEquiv F K₁ K₂
            ω : AlgEquiv F K₂ K₃
            E : Type u_6
            inst✝⁸ : Field E
            inst✝⁷ : Algebra F E
            inst✝⁶ : Algebra E K₁
            inst✝⁵ : Algebra E K₂
            inst✝⁴ : Algebra E K₃
            inst✝³ : IsScalarTower F E K₁
            inst✝² : IsScalarTower F E K₂
            inst✝¹ : IsScalarTower F E K₃
            inst✝ : Normal F E
            x✝ : E
            ⊢ Eq ((algebraMap E K₃) (((χ.trans ω).restrictNormal E) x✝)) ((algebraMap E K₃ …
          -/
      (by simp only [AlgEquiv.trans_apply, AlgEquiv.restrictNormal_commutes])
          /-
            🎉 no goals
          -/


/-- Restriction to a normal subfield as a group homomorphism -/
def AlgEquiv.restrictNormalHom [Normal F E] : (K₁ ≃ₐ[F] K₁) →* E ≃ₐ[F] E :=
  MonoidHom.mk' (fun χ => χ.restrictNormal E) fun ω χ => χ.restrictNormal_trans ω E


lemma AlgEquiv.restrictNormalHom_apply (L : IntermediateField F K₁) [Normal F L]
    (σ : (K₁ ≃ₐ[F] K₁)) (x : L) : restrictNormalHom L σ x = σ x :=
  AlgEquiv.restrictNormal_commutes σ L x


/-- If `K₁/E/F` is a tower of fields with `E/F` normal then `AlgHom.restrictNormal'` is an
 equivalence. -/
@[simps, stacks 0BR4]
def Normal.algHomEquivAut [Normal F E] : (E →ₐ[F] K₁) ≃ E ≃ₐ[F] E where
  toFun σ := AlgHom.restrictNormal' σ E
  invFun σ := (IsScalarTower.toAlgHom F E K₁).comp σ.toAlgHom
  left_inv σ := by
    /-
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgHom F E K₁
      ⊢ Eq ((fun σ => (IsScalarTower.toAlgHom F E K₁).comp ↑σ) ((fun σ => σ.restrict …
    -/
    ext
    /-
      case H
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgHom F E K₁
      x✝ : E
      ⊢ Eq (((fun σ => (IsScalarTower.toAlgHom F E K₁).comp ↑σ) ((fun σ => σ.restric …
    -/
    simp [AlgHom.restrictNormal']
    /-
      🎉 no goals
    -/
  right_inv σ := by
    /-
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgEquiv F E E
      ⊢ Eq ((fun σ => σ.restrictNormal' E) ((fun σ => (IsScalarTower.toAlgHom F E K₁ …
    -/
    ext
    /-
      case h
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgEquiv F E E
      a✝ : E
      ⊢ Eq (((fun σ => σ.restrictNormal' E) ((fun σ => (IsScalarTower.toAlgHom F E K …
    -/
    simp only [AlgHom.restrictNormal', AlgEquiv.toAlgHom_eq_coe, AlgEquiv.coe_ofBijective]
    /-
      case h
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgEquiv F E E
      a✝ : E
      ⊢ Eq ((((IsScalarTower.toAlgHom F E K₁).comp ↑σ).restrictNormal E) a✝) (σ a✝)
    -/
    apply NoZeroSMulDivisors.algebraMap_injective E K₁
    /-
      case h.a
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgEquiv F E E
      a✝ : E
      ⊢ Eq ((algebraMap E K₁) ((((IsScalarTower.toAlgHom F E K₁).comp ↑σ).restrictNo …
    -/
    rw [AlgHom.restrictNormal_commutes]
    /-
      case h.a
      F : Type u_1
      K : Type u_2
      inst✝¹⁷ : Field F
      inst✝¹⁶ : Field K
      inst✝¹⁵ : Algebra F K
      K₁ : Type u_3
      K₂ : Type u_4
      K₃ : Type u_5
      inst✝¹⁴ : Field K₁
      inst✝¹³ : Field K₂
      inst✝¹² : Field K₃
      inst✝¹¹ : Algebra F K₁
      inst✝¹⁰ : Algebra F K₂
      inst✝⁹ : Algebra F K₃
      ϕ : AlgHom F K₁ K₂
      χ : AlgEquiv F K₁ K₂
      ψ : AlgHom F K₂ K₃
      ω : AlgEquiv F K₂ K₃
      E : Type u_6
      inst✝⁸ : Field E
      inst✝⁷ : Algebra F E
      inst✝⁶ : Algebra E K₁
      inst✝⁵ : Algebra E K₂
      inst✝⁴ : Algebra E K₃
      inst✝³ : IsScalarTower F E K₁
      inst✝² : IsScalarTower F E K₂
      inst✝¹ : IsScalarTower F E K₃
      inst✝ : Normal F E
      σ : AlgEquiv F E E
      a✝ : E
      ⊢ Eq (((IsScalarTower.toAlgHom F E K₁).comp ↑σ) ((algebraMap E E) a✝)) ((algeb …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- If `E/Kᵢ/F` are towers of fields with `E/F` normal then we can lift
  an algebra homomorphism `ϕ : K₁ →ₐ[F] K₂` to `ϕ.liftNormal E : E →ₐ[F] E`. -/
@[stacks 0BME "Part 2"]
noncomputable def AlgHom.liftNormal [h : Normal F E] : E →ₐ[F] E :=
  @AlgHom.restrictScalars F K₁ E E _ _ _ _ _ _
      ((IsScalarTower.toAlgHom F K₂ E).comp ϕ).toRingHom.toAlgebra _ _ _ _ <|
    Nonempty.some <|
      @IntermediateField.nonempty_algHom_of_adjoin_splits _ _ _ _ _ _ _
        ((IsScalarTower.toAlgHom F K₂ E).comp ϕ).toRingHom.toAlgebra _
        (fun x _ ↦ ⟨(h.out x).1.tower_top,
          splits_of_splits_of_dvd _ (map_ne_zero (minpoly.ne_zero (h.out x).1))
            -- Porting note: had to override typeclass inference below using `(_)`
                /-
                  F : Type u_1
                  K : Type u_2
                  inst✝¹⁴ : Field F
                  inst✝¹³ : Field K
                  inst✝¹² : Algebra F K
                  K₁ : Type u_3
                  K₂ : Type u_4
                  K₃ : Type u_5
                  inst✝¹¹ : Field K₁
                  inst✝¹⁰ : Field K₂
                  inst✝⁹ : Field K₃
                  inst✝⁸ : Algebra F K₁
                  inst✝⁷ : Algebra F K₂
                  inst✝⁶ : Algebra F K₃
                  ϕ : AlgHom F K₁ K₂
                  χ : AlgEquiv F K₁ K₂
                  ψ : AlgHom F K₂ K₃
                  ω : AlgEquiv F K₂ K₃
                  E : Type u_6
                  inst✝⁵ : Field E
                  inst✝⁴ : Algebra F E
                  inst✝³ : Algebra K₁ E
                  inst✝² : Algebra K₂ E
                  inst✝¹ : IsScalarTower F K₁ E
                  inst✝ : IsScalarTower F K₂ E
                  h : Normal F E
                  x : E
                  x✝ : Membership.mem Set.univ x
                  ⊢ Polynomial.Splits (algebraMap K₁ E) (Polynomial.map (algebraMap F K₁) (minpo …
                -/
            (by rw [splits_map_iff, ← @IsScalarTower.algebraMap_eq _ _ _ _ _ _ (_) (_) (_)]
                /-
                  F : Type u_1
                  K : Type u_2
                  inst✝¹⁴ : Field F
                  inst✝¹³ : Field K
                  inst✝¹² : Algebra F K
                  K₁ : Type u_3
                  K₂ : Type u_4
                  K₃ : Type u_5
                  inst✝¹¹ : Field K₁
                  inst✝¹⁰ : Field K₂
                  inst✝⁹ : Field K₃
                  inst✝⁸ : Algebra F K₁
                  inst✝⁷ : Algebra F K₂
                  inst✝⁶ : Algebra F K₃
                  ϕ : AlgHom F K₁ K₂
                  χ : AlgEquiv F K₁ K₂
                  ψ : AlgHom F K₂ K₃
                  ω : AlgEquiv F K₂ K₃
                  E : Type u_6
                  inst✝⁵ : Field E
                  inst✝⁴ : Algebra F E
                  inst✝³ : Algebra K₁ E
                  inst✝² : Algebra K₂ E
                  inst✝¹ : IsScalarTower F K₁ E
                  inst✝ : IsScalarTower F K₂ E
                  h : Normal F E
                  x : E
                  x✝ : Membership.mem Set.univ x
                  ⊢ Polynomial.Splits (algebraMap F E) (minpoly F x)
                -/
                exact (h.out x).2)
                /-
                  🎉 no goals
                -/
            (minpoly.dvd_map_of_isScalarTower F K₁ x)⟩)
        (IntermediateField.adjoin_univ _ _)


@[simp]
theorem AlgHom.liftNormal_commutes [Normal F E] (x : K₁) :
    ϕ.liftNormal E (algebraMap K₁ E x) = algebraMap K₂ E (ϕ x) :=
  -- Porting note: This seems to have been some sort of typeclass override trickery using `by apply`
  -- Now we explicitly specify which typeclass to override, using `(_)` instead of `_`
  @AlgHom.commutes K₁ E E _ _ _ _ (_) _ _


@[simp]
theorem AlgHom.restrict_liftNormal (ϕ : K₁ →ₐ[F] K₁) [Normal F K₁] [Normal F E] :
    (ϕ.liftNormal E).restrictNormal K₁ = ϕ :=
  AlgHom.ext fun x =>
    (algebraMap K₁ E).injective
      (Eq.trans (AlgHom.restrictNormal_commutes _ K₁ x) (ϕ.liftNormal_commutes E x))


/-- If `E/Kᵢ/F` are towers of fields with `E/F` normal then we can lift
  an algebra isomorphism `ϕ : K₁ ≃ₐ[F] K₂` to `ϕ.liftNormal E : E ≃ₐ[F] E`. -/
noncomputable def AlgEquiv.liftNormal [Normal F E] : E ≃ₐ[F] E :=
  AlgEquiv.ofBijective (χ.toAlgHom.liftNormal E) (AlgHom.normal_bijective F E E _)


@[simp]
theorem AlgEquiv.liftNormal_commutes [Normal F E] (x : K₁) :
    χ.liftNormal E (algebraMap K₁ E x) = algebraMap K₂ E (χ x) :=
  χ.toAlgHom.liftNormal_commutes E x


@[simp]
theorem AlgEquiv.restrict_liftNormal (χ : K₁ ≃ₐ[F] K₁) [Normal F K₁] [Normal F E] :
    (χ.liftNormal E).restrictNormal K₁ = χ :=
  AlgEquiv.ext fun x =>
    (algebraMap K₁ E).injective
      (Eq.trans (AlgEquiv.restrictNormal_commutes _ K₁ x) (χ.liftNormal_commutes E x))


/-- The group homomorphism given by restricting an algebra isomorphism to a normal subfield
is surjective. -/
theorem AlgEquiv.restrictNormalHom_surjective [Normal F K₁] [Normal F E] :
    Function.Surjective (AlgEquiv.restrictNormalHom K₁ : (E ≃ₐ[F] E) → K₁ ≃ₐ[F] K₁) := fun χ =>
  ⟨χ.liftNormal E, χ.restrict_liftNormal E⟩


/-- The group homomorphism given by restricting an algebra isomorphism to itself
is the identity map. -/
@[simp]
theorem AlgEquiv.restrictNormalHom_id (F K : Type*)
    [Field F] [Field K] [Algebra F K] [Normal F K] :
    AlgEquiv.restrictNormalHom K = MonoidHom.id (K ≃ₐ[F] K) := by
  /-
    F : Type u_7
    K : Type u_8
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Normal F K
    ⊢ Eq (AlgEquiv.restrictNormalHom K) (MonoidHom.id (AlgEquiv F K K))
  -/
  ext f x
  /-
    case h.h
    F : Type u_7
    K : Type u_8
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Normal F K
    f : AlgEquiv F K K
    x : K
    ⊢ Eq (((AlgEquiv.restrictNormalHom K) f) x) (((MonoidHom.id (AlgEquiv F K K))  …
  -/
  dsimp only [restrictNormalHom, MonoidHom.mk'_apply, MonoidHom.id_apply]
  /-
    case h.h
    F : Type u_7
    K : Type u_8
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Normal F K
    f : AlgEquiv F K K
    x : K
    ⊢ Eq ((f.restrictNormal K) x) (f x)
  -/
  apply (algebraMap K K).injective
  /-
    case h.h.a
    F : Type u_7
    K : Type u_8
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Normal F K
    f : AlgEquiv F K K
    x : K
    ⊢ Eq ((algebraMap K K) ((f.restrictNormal K) x)) ((algebraMap K K) (f x))
  -/
  rw [AlgEquiv.restrictNormal_commutes]
  /-
    case h.h.a
    F : Type u_7
    K : Type u_8
    inst✝³ : Field F
    inst✝² : Field K
    inst✝¹ : Algebra F K
    inst✝ : Normal F K
    f : AlgEquiv F K K
    x : K
    ⊢ Eq (f ((algebraMap K K) x)) ((algebraMap K K) (f x))
  -/
  simp only [Algebra.id.map_eq_id, RingHom.id_apply]
  /-
    🎉 no goals
  -/


/-- In a scalar tower `K₃/K₂/K₁/F` with `K₁` and `K₂` are normal over `F`, the group homomorphism
given by the restriction of algebra isomorphisms of `K₃` to `K₁` is equal to the composition of
the group homomorphism given by the restricting an algebra isomorphism of `K₃` to `K₂` and
the group homomorphism given by the restricting an algebra isomorphism of `K₂` to `K₁` -/
theorem AlgEquiv.restrictNormalHom_comp (F K₁ K₂ K₃ : Type*)
    [Field F] [Field K₁] [Field K₂] [Field K₃]
    [Algebra F K₁] [Algebra F K₂] [Algebra F K₃] [Algebra K₁ K₂] [Algebra K₁ K₃] [Algebra K₂ K₃]
    [IsScalarTower F K₁ K₃] [IsScalarTower F K₁ K₂] [IsScalarTower F K₂ K₃] [IsScalarTower K₁ K₂ K₃]
    [Normal F K₁] [Normal F K₂] :
    AlgEquiv.restrictNormalHom K₁ =
    (AlgEquiv.restrictNormalHom K₁).comp
    (AlgEquiv.restrictNormalHom (F := F) (K₁ := K₃) K₂) := by
  /-
    F : Type u_7
    K₁ : Type u_8
    K₂ : Type u_9
    K₃ : Type u_10
    inst✝¹⁵ : Field F
    inst✝¹⁴ : Field K₁
    inst✝¹³ : Field K₂
    inst✝¹² : Field K₃
    inst✝¹¹ : Algebra F K₁
    inst✝¹⁰ : Algebra F K₂
    inst✝⁹ : Algebra F K₃
    inst✝⁸ : Algebra K₁ K₂
    inst✝⁷ : Algebra K₁ K₃
    inst✝⁶ : Algebra K₂ K₃
    inst✝⁵ : IsScalarTower F K₁ K₃
    inst✝⁴ : IsScalarTower F K₁ K₂
    inst✝³ : IsScalarTower F K₂ K₃
    inst✝² : IsScalarTower K₁ K₂ K₃
    inst✝¹ : Normal F K₁
    inst✝ : Normal F K₂
    ⊢ Eq (AlgEquiv.restrictNormalHom K₁) ((AlgEquiv.restrictNormalHom K₁).comp (Al …
  -/
  ext f x
  /-
    case h.h
    F : Type u_7
    K₁ : Type u_8
    K₂ : Type u_9
    K₃ : Type u_10
    inst✝¹⁵ : Field F
    inst✝¹⁴ : Field K₁
    inst✝¹³ : Field K₂
    inst✝¹² : Field K₃
    inst✝¹¹ : Algebra F K₁
    inst✝¹⁰ : Algebra F K₂
    inst✝⁹ : Algebra F K₃
    inst✝⁸ : Algebra K₁ K₂
    inst✝⁷ : Algebra K₁ K₃
    inst✝⁶ : Algebra K₂ K₃
    inst✝⁵ : IsScalarTower F K₁ K₃
    inst✝⁴ : IsScalarTower F K₁ K₂
    inst✝³ : IsScalarTower F K₂ K₃
    inst✝² : IsScalarTower K₁ K₂ K₃
    inst✝¹ : Normal F K₁
    inst✝ : Normal F K₂
    f : AlgEquiv F K₃ K₃
    x : K₁
    ⊢ Eq (((AlgEquiv.restrictNormalHom K₁) f) x) ((((AlgEquiv.restrictNormalHom K₁ …
  -/
  apply (algebraMap K₁ K₃).injective
  /-
    case h.h.a
    F : Type u_7
    K₁ : Type u_8
    K₂ : Type u_9
    K₃ : Type u_10
    inst✝¹⁵ : Field F
    inst✝¹⁴ : Field K₁
    inst✝¹³ : Field K₂
    inst✝¹² : Field K₃
    inst✝¹¹ : Algebra F K₁
    inst✝¹⁰ : Algebra F K₂
    inst✝⁹ : Algebra F K₃
    inst✝⁸ : Algebra K₁ K₂
    inst✝⁷ : Algebra K₁ K₃
    inst✝⁶ : Algebra K₂ K₃
    inst✝⁵ : IsScalarTower F K₁ K₃
    inst✝⁴ : IsScalarTower F K₁ K₂
    inst✝³ : IsScalarTower F K₂ K₃
    inst✝² : IsScalarTower K₁ K₂ K₃
    inst✝¹ : Normal F K₁
    inst✝ : Normal F K₂
    f : AlgEquiv F K₃ K₃
    x : K₁
    ⊢ Eq ((algebraMap K₁ K₃) (((AlgEquiv.restrictNormalHom K₁) f) x)) ((algebraMap …
  -/
  rw [IsScalarTower.algebraMap_eq K₁ K₂ K₃]
  simp only [AlgEquiv.restrictNormalHom, MonoidHom.mk'_apply, RingHom.coe_comp, Function.comp_apply,
    ← algebraMap_apply, AlgEquiv.restrictNormal_commutes, MonoidHom.coe_comp]


theorem AlgEquiv.restrictNormalHom_comp_apply (K₁ K₂ : Type*) {F K₃ : Type*}
    [Field F] [Field K₁] [Field K₂] [Field K₃]
    [Algebra F K₁] [Algebra F K₂] [Algebra F K₃] [Algebra K₁ K₂] [Algebra K₁ K₃] [Algebra K₂ K₃]
    [IsScalarTower F K₁ K₃] [IsScalarTower F K₁ K₂] [IsScalarTower F K₂ K₃] [IsScalarTower K₁ K₂ K₃]
    [Normal F K₁] [Normal F K₂] (f : K₃ ≃ₐ[F] K₃) :
    AlgEquiv.restrictNormalHom K₁ f =
    (AlgEquiv.restrictNormalHom K₁) (AlgEquiv.restrictNormalHom K₂ f) := by
  /-
    K₁ : Type u_7
    K₂ : Type u_8
    F : Type u_9
    K₃ : Type u_10
    inst✝¹⁵ : Field F
    inst✝¹⁴ : Field K₁
    inst✝¹³ : Field K₂
    inst✝¹² : Field K₃
    inst✝¹¹ : Algebra F K₁
    inst✝¹⁰ : Algebra F K₂
    inst✝⁹ : Algebra F K₃
    inst✝⁸ : Algebra K₁ K₂
    inst✝⁷ : Algebra K₁ K₃
    inst✝⁶ : Algebra K₂ K₃
    inst✝⁵ : IsScalarTower F K₁ K₃
    inst✝⁴ : IsScalarTower F K₁ K₂
    inst✝³ : IsScalarTower F K₂ K₃
    inst✝² : IsScalarTower K₁ K₂ K₃
    inst✝¹ : Normal F K₁
    inst✝ : Normal F K₂
    f : AlgEquiv F K₃ K₃
    ⊢ Eq ((AlgEquiv.restrictNormalHom K₁) f) ((AlgEquiv.restrictNormalHom K₁) ((Al …
  -/
  rw [IsScalarTower.AlgEquiv.restrictNormalHom_comp F K₁ K₂ K₃, MonoidHom.comp_apply]
  /-
    🎉 no goals
  -/


open IntermediateField in
theorem Normal.minpoly_eq_iff_mem_orbit [h : Normal F E] {x y : E} :
    minpoly F x = minpoly F y ↔ x ∈ MulAction.orbit (E ≃ₐ[F] E) y := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_6
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : Normal F E
    x y : E
    ⊢ Iff (Eq (minpoly F x) (minpoly F y)) (Membership.mem (MulAction.orbit (AlgEq …
  -/
  refine ⟨fun he ↦ ?_, fun ⟨f, he⟩ ↦ he ▸ minpoly.algEquiv_eq f y⟩
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_6
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : Normal F E
    x y : E
    he : Eq (minpoly F x) (minpoly F y)
    ⊢ Membership.mem (MulAction.orbit (AlgEquiv F E E) y) x
  -/
  obtain ⟨φ, hφ⟩ := exists_algHom_of_splits_of_aeval (normal_iff.mp h) (he ▸ minpoly.aeval F x)
  /-
    case intro
    F : Type u_1
    inst✝² : Field F
    E : Type u_6
    inst✝¹ : Field E
    inst✝ : Algebra F E
    h : Normal F E
    x y : E
    he : Eq (minpoly F x) (minpoly F y)
    φ : AlgHom F E E
    hφ : Eq (φ y) x
    ⊢ Membership.mem (MulAction.orbit (AlgEquiv F E E) y) x
  -/
  exact ⟨AlgEquiv.ofBijective φ (φ.normal_bijective F E E), hφ⟩
  /-
    🎉 no goals
  -/


theorem isSolvable_of_isScalarTower [Normal F K₁] [h1 : IsSolvable (K₁ ≃ₐ[F] K₁)]
    [h2 : IsSolvable (E ≃ₐ[K₁] E)] : IsSolvable (E ≃ₐ[F] E) := by
  let f : (E ≃ₐ[K₁] E) →* E ≃ₐ[F] E :=
    { toFun := fun ϕ =>
        AlgEquiv.ofAlgHom (ϕ.toAlgHom.restrictScalars F) (ϕ.symm.toAlgHom.restrictScalars F)
          (AlgHom.ext fun x => ϕ.apply_symm_apply x) (AlgHom.ext fun x => ϕ.symm_apply_apply x)
      map_one' := AlgEquiv.ext fun _ => rfl
      map_mul' := fun _ _ => AlgEquiv.ext fun _ => rfl }
  refine
    solvable_of_ker_le_range f (AlgEquiv.restrictNormalHom K₁) fun ϕ hϕ =>
      ⟨{ ϕ with commutes' := fun x => ?_ }, AlgEquiv.ext fun _ => rfl⟩
  /-
    F : Type u_1
    inst✝⁷ : Field F
    K₁ : Type u_3
    inst✝⁶ : Field K₁
    inst✝⁵ : Algebra F K₁
    E : Type u_6
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    inst✝² : Algebra K₁ E
    inst✝¹ : IsScalarTower F K₁ E
    inst✝ : Normal F K₁
    h1 : IsSolvable (AlgEquiv F K₁ K₁)
    h2 : IsSolvable (AlgEquiv K₁ E E)
    f : MonoidHom (AlgEquiv K₁ E E) (AlgEquiv F E E) := { toFun := fun ϕ => AlgEqu …
    ϕ : AlgEquiv F E E
    hϕ : Membership.mem (AlgEquiv.restrictNormalHom K₁).ker ϕ
    x : K₁
    ⊢ Eq (ϕ.toFun ((algebraMap K₁ E) x)) ((algebraMap K₁ E) x)
  -/
  exact Eq.trans (ϕ.restrictNormal_commutes K₁ x).symm (congr_arg _ (AlgEquiv.ext_iff.mp hϕ x))
  /-
    🎉 no goals
  -/


/-- If `x : L` is a root of `minpoly K y`, then we can find `(σ : L ≃ₐ[K] L)` with `σ x = y`.
  That is, `x` and `y` are Galois conjugates. -/
theorem exists_algEquiv_of_root [Normal K L] {x y : L} (hy : IsAlgebraic K y)
    (h_ev : (Polynomial.aeval x) (minpoly K y) = 0) : ∃ σ : L ≃ₐ[K] L, σ x = y := by
  /-
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    ⊢ Exists fun σ => Eq (σ x) y
  -/
  have hx : IsAlgebraic K x := ⟨minpoly K y, ne_zero hy.isIntegral, h_ev⟩
  /-
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    hx : IsAlgebraic K x
    ⊢ Exists fun σ => Eq (σ x) y
  -/
  set f : K⟮x⟯ ≃ₐ[K] K⟮y⟯ := algEquiv hx (eq_of_root hy h_ev)
  have hxy : (liftNormal f L) ((algebraMap (↥K⟮x⟯) L) (AdjoinSimple.gen K x)) = y := by
    rw [liftNormal_commutes f L, algEquiv_apply, AdjoinSimple.algebraMap_gen K y]
  /-
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    hx : IsAlgebraic K x
    f : AlgEquiv K (Subtype fun x_1 => Membership.mem (IntermediateField.adjoin K  …
    hxy : Eq ((f.liftNormal L) ((algebraMap (Subtype fun x_1 => Membership.mem (In …
    ⊢ Exists fun σ => Eq (σ x) y
  -/
  exact ⟨(liftNormal f L), hxy⟩
  /-
    🎉 no goals
  -/


/-- If `x : L` is a root of `minpoly K y`, then we can find `(σ : L ≃ₐ[K] L)` with `σ y = x`.
  That is, `x` and `y` are Galois conjugates. -/
theorem exists_algEquiv_of_root' [Normal K L]{x y : L} (hy : IsAlgebraic K y)
    (h_ev : (Polynomial.aeval x) (minpoly K y) = 0) : ∃ σ : L ≃ₐ[K] L, σ y = x := by
  /-
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    ⊢ Exists fun σ => Eq (σ y) x
  -/
  obtain ⟨σ, hσ⟩ := exists_algEquiv_of_root hy h_ev
  /-
    case intro
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    σ : AlgEquiv K L L
    hσ : Eq (σ x) y
    ⊢ Exists fun σ => Eq (σ y) x
  -/
  use σ.symm
  /-
    case h
    K : Type u_6
    L : Type u_7
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Normal K L
    x y : L
    hy : IsAlgebraic K y
    h_ev : Eq ((Polynomial.aeval x) (minpoly K y)) 0
    σ : AlgEquiv K L L
    hσ : Eq (σ x) y
    ⊢ Eq (σ.symm y) x
  -/
  rw [← hσ, symm_apply_apply]
  /-
    🎉 no goals
  -/


