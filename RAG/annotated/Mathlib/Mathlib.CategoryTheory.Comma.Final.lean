private lemma final_fst_small [R.Final] : (fst L R).Final := by
  /-
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  rw  [Functor.final_iff_isIso_colimit_pre]
  /-
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    ⊢ ∀ (G : CategoryTheory.Functor A (Type v₁)), CategoryTheory.IsIso (CategoryTh …
  -/
  intro G
  let i : colimit G ≅ colimit (fst L R ⋙ G) :=
    colimitIsoColimitGrothendieck L G ≪≫
    (Final.colimitIso (Grothendieck.pre (functor L) R) (grothendieckProj L ⋙ G)).symm ≪≫
    HasColimit.isoOfNatIso (Iso.refl _) ≪≫
    Final.colimitIso (grothendieckPrecompFunctorEquivalence L R).functor (fst L R ⋙ G)
  /-
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    G : CategoryTheory.Functor A (Type v₁)
    i : CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limit …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimit.pre G (CategoryTheory.Co …
  -/
  convert i.isIso_inv
  /-
    case h.e'_5
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    G : CategoryTheory.Functor A (Type v₁)
    i : CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limit …
    ⊢ Eq (CategoryTheory.Limits.colimit.pre G (CategoryTheory.Comma.fst L R)) i.inv
  -/
  apply colimit.hom_ext
  /-
    case h.e'_5.w
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    G : CategoryTheory.Functor A (Type v₁)
    i : CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limit …
    ⊢ ∀ (j : CategoryTheory.Comma L R), Eq (CategoryTheory.CategoryStruct.comp (Ca …
  -/
  intro ⟨a, b, f⟩
  simp only [colimit.ι_pre, comp_obj, fst_obj, grothendieckPrecompFunctorEquivalence_functor,
    Iso.trans_inv, Iso.symm_inv, Category.assoc, i]
  change _ = colimit.ι (fst L R ⋙ G)
    ((grothendieckPrecompFunctorToComma L R).obj ⟨b, CostructuredArrow.mk f⟩) ≫ _
  /-
    case h.e'_5.w
    A : Type v₁
    inst✝³ : CategoryTheory.Category.{v₁, v₁} A
    B : Type v₁
    inst✝² : CategoryTheory.Category.{v₁, v₁} B
    T : Type v₁
    inst✝¹ : CategoryTheory.Category.{v₁, v₁} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    G : CategoryTheory.Functor A (Type v₁)
    i : CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limit …
    a : A
    b : B
    f : Quiver.Hom (L.obj a) (R.obj b)
    ⊢ Eq (CategoryTheory.Limits.colimit.ι G a) (CategoryTheory.CategoryStruct.comp …
  -/
  simp
  /-
    🎉 no goals
  -/


instance final_fst [R.Final] : (fst L R).Final := by
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  let sA : A ≌ AsSmall.{max u₁ u₂ u₃ v₁ v₂ v₃} A := AsSmall.equiv
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  let sB : B ≌ AsSmall.{max u₁ u₂ u₃ v₁ v₂ v₃} B := AsSmall.equiv
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  let sT : T ≌ AsSmall.{max u₁ u₂ u₃ v₁ v₂ v₃} T := AsSmall.equiv
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    sT : CategoryTheory.Equivalence T (CategoryTheory.AsSmall T) := CategoryTheory …
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  let L' := sA.inverse ⋙ L ⋙ sT.functor
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    sT : CategoryTheory.Equivalence T (CategoryTheory.AsSmall T) := CategoryTheory …
    L' : CategoryTheory.Functor (CategoryTheory.AsSmall A) (CategoryTheory.AsSmall …
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  let R' := sB.inverse ⋙ R ⋙ sT.functor
  let fC : Comma L R ⥤ Comma L' R' :=
    map (F₁ := sA.functor) (F := sT.functor) (F₂ := sB.functor)
      (isoWhiskerRight sA.unitIso (L ⋙ sT.functor)).hom
      (isoWhiskerRight sB.unitIso (R ⋙ sT.functor)).hom
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    sT : CategoryTheory.Equivalence T (CategoryTheory.AsSmall T) := CategoryTheory …
    L' : CategoryTheory.Functor (CategoryTheory.AsSmall A) (CategoryTheory.AsSmall …
    R' : CategoryTheory.Functor (CategoryTheory.AsSmall B) (CategoryTheory.AsSmall …
    fC : CategoryTheory.Functor (CategoryTheory.Comma L R) (CategoryTheory.Comma L …
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  have : Final (fst L' R') := final_fst_small _ _
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    sT : CategoryTheory.Equivalence T (CategoryTheory.AsSmall T) := CategoryTheory …
    L' : CategoryTheory.Functor (CategoryTheory.AsSmall A) (CategoryTheory.AsSmall …
    R' : CategoryTheory.Functor (CategoryTheory.AsSmall B) (CategoryTheory.AsSmall …
    fC : CategoryTheory.Functor (CategoryTheory.Comma L R) (CategoryTheory.Comma L …
    this : (CategoryTheory.Comma.fst L' R').Final
    ⊢ (CategoryTheory.Comma.fst L R).Final
  -/
  apply final_of_natIso (F := (fC ⋙ fst L' R' ⋙ sA.inverse))
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : R.Final
    sA : CategoryTheory.Equivalence A (CategoryTheory.AsSmall A) := CategoryTheory …
    sB : CategoryTheory.Equivalence B (CategoryTheory.AsSmall B) := CategoryTheory …
    sT : CategoryTheory.Equivalence T (CategoryTheory.AsSmall T) := CategoryTheory …
    L' : CategoryTheory.Functor (CategoryTheory.AsSmall A) (CategoryTheory.AsSmall …
    R' : CategoryTheory.Functor (CategoryTheory.AsSmall B) (CategoryTheory.AsSmall …
    fC : CategoryTheory.Functor (CategoryTheory.Comma L R) (CategoryTheory.Comma L …
    this : (CategoryTheory.Comma.fst L' R').Final
    ⊢ CategoryTheory.Iso (fC.comp ((CategoryTheory.Comma.fst L' R').comp sA.invers …
  -/
  exact (Functor.associator _ _ _).symm.trans (Iso.compInverseIso (mapFst _ _))
  /-
    🎉 no goals
  -/


instance initial_snd [L.Initial] : (snd L R).Initial := by
  haveI : ((opFunctor L R).leftOp ⋙ fst R.op L.op).Final :=
    final_equivalence_comp (opEquiv L R).functor.leftOp (fst R.op L.op)
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : L.Initial
    this : ((CategoryTheory.Comma.opFunctor L R).leftOp.comp (CategoryTheory.Comma …
    ⊢ (CategoryTheory.Comma.snd L R).Initial
  -/
  haveI : (snd L R).op.Final := final_of_natIso (opFunctorCompFst _ _)
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝ : L.Initial
    this✝ : ((CategoryTheory.Comma.opFunctor L R).leftOp.comp (CategoryTheory.Comm …
    this : (CategoryTheory.Comma.snd L R).op.Final
    ⊢ (CategoryTheory.Comma.snd L R).Initial
  -/
  apply initial_of_final_op
  /-
    🎉 no goals
  -/


/-- `Comma L R` with `L : A ⥤ T` and `R : B ⥤ T` is connected if `R` is final and `A` is
connected. -/
instance isConnected_comma_of_final [IsConnected A] [R.Final] : IsConnected (Comma L R) := by
  /-
    A : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝¹ : CategoryTheory.IsConnected A
    inst✝ : R.Final
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Comma L R)
  -/
  rwa [isConnected_iff_of_final (fst L R)]
  /-
    🎉 no goals
  -/


/-- `Comma L R` with `L : A ⥤ T` and `R : B ⥤ T` is connected if `L` is initial and `B` is
connected. -/
instance isConnected_comma_of_initial [IsConnected B] [L.Initial] : IsConnected (Comma L R) := by
  /-
    A : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} B
    T : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} T
    L : CategoryTheory.Functor A T
    R : CategoryTheory.Functor B T
    inst✝¹ : CategoryTheory.IsConnected B
    inst✝ : L.Initial
    ⊢ CategoryTheory.IsConnected (CategoryTheory.Comma L R)
  -/
  rwa [isConnected_iff_of_initial (snd L R)]
  /-
    🎉 no goals
  -/


