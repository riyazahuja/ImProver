/-- The scalar multiplication of family of elements of a presheaf of modules `M` over `R`
by a family of elements of `R`. -/
def smul : FamilyOfElements (M.presheaf ⋙ forget _) P := fun Y f hf =>
  HSMul.hSMul (α := R.obj (Opposite.op Y)) (β := M.obj (Opposite.op Y)) (r f hf) (m f hf)


lemma _root_.PresheafOfModules.Sheafify.app_eq_of_isLocallyInjective
    {Y : C} (r₀ r₀' : R₀.obj (Opposite.op Y))
    (m₀ m₀' : M₀.obj (Opposite.op Y))
    (hr₀ : α.app _ r₀ = α.app _ r₀')
    (hm₀ : φ.app _ m₀ = φ.app _ m₀') :
    φ.app _ (r₀ • m₀) = φ.app _ (r₀' • m₀') := by
  apply hA _ (Presheaf.equalizerSieve (D := RingCat) r₀ r₀' ⊓
      Presheaf.equalizerSieve (F := M₀.presheaf) m₀ m₀')
    /-
      case x
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ R : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R₀ R
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Functor (Opposite C) AddCommGrp
      φ : Quiver.Hom M₀.presheaf A
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      hA : CategoryTheory.Presheaf.IsSeparated J A
      Y : C
      r₀ r₀' : ↑(R₀.obj { unop := Y })
      m₀ m₀' : ↑(M₀.obj { unop := Y })
      hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((α.app { unop := Y }).hom r₀')
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((φ.app { unop := Y }) m₀')
      ⊢ Membership.mem (J (Opposite.unop { unop := Y })) (Min.min (CategoryTheory.Pr …
    -/
  · apply J.intersection_covering
      /-
        case x.rj
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ R : CategoryTheory.Functor (Opposite C) RingCat
        α : Quiver.Hom R₀ R
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Functor (Opposite C) AddCommGrp
        φ : Quiver.Hom M₀.presheaf A
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        hA : CategoryTheory.Presheaf.IsSeparated J A
        Y : C
        r₀ r₀' : ↑(R₀.obj { unop := Y })
        m₀ m₀' : ↑(M₀.obj { unop := Y })
        hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((α.app { unop := Y }).hom r₀')
        hm₀ : Eq ((φ.app { unop := Y }) m₀) ((φ.app { unop := Y }) m₀')
        ⊢ Membership.mem (J (Opposite.unop { unop := Y })) (CategoryTheory.Presheaf.eq …
      -/
    · exact Presheaf.equalizerSieve_mem J α _ _ hr₀
      /-
        🎉 no goals
      -/
      /-
        case x.sj
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ R : CategoryTheory.Functor (Opposite C) RingCat
        α : Quiver.Hom R₀ R
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Functor (Opposite C) AddCommGrp
        φ : Quiver.Hom M₀.presheaf A
        inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        hA : CategoryTheory.Presheaf.IsSeparated J A
        Y : C
        r₀ r₀' : ↑(R₀.obj { unop := Y })
        m₀ m₀' : ↑(M₀.obj { unop := Y })
        hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((α.app { unop := Y }).hom r₀')
        hm₀ : Eq ((φ.app { unop := Y }) m₀) ((φ.app { unop := Y }) m₀')
        ⊢ Membership.mem (J (Opposite.unop { unop := Y })) (CategoryTheory.Presheaf.eq …
      -/
    · exact Presheaf.equalizerSieve_mem J φ _ _ hm₀
      /-
        🎉 no goals
      -/
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ R : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R₀ R
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Functor (Opposite C) AddCommGrp
      φ : Quiver.Hom M₀.presheaf A
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      hA : CategoryTheory.Presheaf.IsSeparated J A
      Y : C
      r₀ r₀' : ↑(R₀.obj { unop := Y })
      m₀ m₀' : ↑(M₀.obj { unop := Y })
      hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((α.app { unop := Y }).hom r₀')
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((φ.app { unop := Y }) m₀')
      ⊢ ∀ (Y_1 : C) (f : Quiver.Hom Y_1 (Opposite.unop { unop := Y })), (Min.min (Ca …
    -/
  · intro Z g hg
    erw [← NatTrans.naturality_apply, ← NatTrans.naturality_apply, M₀.map_smul, M₀.map_smul,
      hg.1, hg.2]
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ R : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R₀ R
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Functor (Opposite C) AddCommGrp
      φ : Quiver.Hom M₀.presheaf A
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      hA : CategoryTheory.Presheaf.IsSeparated J A
      Y : C
      r₀ r₀' : ↑(R₀.obj { unop := Y })
      m₀ m₀' : ↑(M₀.obj { unop := Y })
      hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((α.app { unop := Y }).hom r₀')
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((φ.app { unop := Y }) m₀')
      Z : C
      g : Quiver.Hom Z (Opposite.unop { unop := Y })
      hg : (Min.min (CategoryTheory.Presheaf.equalizerSieve r₀ r₀') (CategoryTheory. …
      ⊢ Eq ((φ.app { unop := Z }) (HSMul.hSMul ((R₀.map g.op) r₀') ((M₀.presheaf.map …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma isCompatible_map_smul_aux {Y Z : C} (f : Y ⟶ X) (g : Z ⟶ Y)
    (r₀ : R₀.obj (Opposite.op Y)) (r₀' : R₀.obj (Opposite.op Z))
    (m₀ : M₀.obj (Opposite.op Y)) (m₀' : M₀.obj (Opposite.op Z))
    (hr₀ : α.app _ r₀ = R.map f.op r) (hr₀' : α.app _ r₀' = R.map (f.op ≫ g.op) r)
    (hm₀ : φ.app _ m₀ = A.map f.op m) (hm₀' : φ.app _ m₀' = A.map (f.op ≫ g.op) m) :
    φ.app _ (M₀.map g.op (r₀ • m₀)) = φ.app _ (r₀' • m₀') := by
  rw [← PresheafOfModules.Sheafify.app_eq_of_isLocallyInjective α φ hA (R₀.map g.op r₀) r₀'
    (M₀.map g.op m₀) m₀', M₀.map_smul]
  · rw [hr₀', R.map_comp, RingCat.comp_apply, ← hr₀, ← RingCat.comp_apply, NatTrans.naturality,
      RingCat.comp_apply]
    /-
      case hm₀
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ R : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R₀ R
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Functor (Opposite C) AddCommGrp
      φ : Quiver.Hom M₀.presheaf A
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      hA : CategoryTheory.Presheaf.IsSeparated J A
      X : C
      r : ↑(R.obj { unop := X })
      m : ↑(A.obj { unop := X })
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      r₀ : ↑(R₀.obj { unop := Y })
      r₀' : ↑(R₀.obj { unop := Z })
      m₀ : ↑(M₀.obj { unop := Y })
      m₀' : ↑(M₀.obj { unop := Z })
      hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((R.map f.op).hom r)
      hr₀' : Eq ((α.app { unop := Z }).hom r₀') ((R.map (CategoryTheory.CategoryStru …
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((A.map f.op) m)
      hm₀' : Eq ((φ.app { unop := Z }) m₀') ((A.map (CategoryTheory.CategoryStruct.c …
      ⊢ Eq ((φ.app { unop := Z }) ((M₀.map g.op).hom m₀)) ((φ.app { unop := Z }) m₀')
    -/
  · rw [hm₀', A.map_comp, AddCommGrp.coe_comp, Function.comp_apply, ← hm₀]
    /-
      case hm₀
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ R : CategoryTheory.Functor (Opposite C) RingCat
      α : Quiver.Hom R₀ R
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Functor (Opposite C) AddCommGrp
      φ : Quiver.Hom M₀.presheaf A
      inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      hA : CategoryTheory.Presheaf.IsSeparated J A
      X : C
      r : ↑(R.obj { unop := X })
      m : ↑(A.obj { unop := X })
      Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      r₀ : ↑(R₀.obj { unop := Y })
      r₀' : ↑(R₀.obj { unop := Z })
      m₀ : ↑(M₀.obj { unop := Y })
      m₀' : ↑(M₀.obj { unop := Z })
      hr₀ : Eq ((α.app { unop := Y }).hom r₀) ((R.map f.op).hom r)
      hr₀' : Eq ((α.app { unop := Z }).hom r₀') ((R.map (CategoryTheory.CategoryStru …
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((A.map f.op) m)
      hm₀' : Eq ((φ.app { unop := Z }) m₀') ((A.map (CategoryTheory.CategoryStruct.c …
      ⊢ Eq ((φ.app { unop := Z }) ((M₀.map g.op).hom m₀)) ((A.map g.op) ((φ.app { un …
    -/
    erw [NatTrans.naturality_apply]
    /-
      🎉 no goals
    -/


include hr₀ hm₀ in
lemma isCompatible_map_smul : ((r₀.smul m₀).map (whiskerRight φ (forget _))).Compatible := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    ⊢ ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab)) …
  -/
  intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ fac
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let a₁ := r₀ f₁ h₁
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let b₁ := m₀ f₁ h₁
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let a₂ := r₀ f₂ h₂
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let b₂ := m₀ f₂ h₂
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let a₀ := R₀.map g₁.op a₁
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  let b₀ := M₀.map g₁.op b₁
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  have ha₁ : (α.app (Opposite.op Y₁)) a₁ = (R.map f₁.op) r := (hr₀ f₁ h₁).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ha₁ : Eq ((α.app { unop := Y₁ }).hom a₁) ((R.map f₁.op).hom r)
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  have ha₂ : (α.app (Opposite.op Y₂)) a₂ = (R.map f₂.op) r := (hr₀ f₂ h₂).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ha₁ : Eq ((α.app { unop := Y₁ }).hom a₁) ((R.map f₁.op).hom r)
    ha₂ : Eq ((α.app { unop := Y₂ }).hom a₂) ((R.map f₂.op).hom r)
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  have hb₁ : (φ.app (Opposite.op Y₁)) b₁ = (A.map f₁.op) m := (hm₀ f₁ h₁).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ha₁ : Eq ((α.app { unop := Y₁ }).hom a₁) ((R.map f₁.op).hom r)
    ha₂ : Eq ((α.app { unop := Y₂ }).hom a₂) ((R.map f₂.op).hom r)
    hb₁ : Eq ((φ.app { unop := Y₁ }) b₁) ((A.map f₁.op) m)
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  have hb₂ : (φ.app (Opposite.op Y₂)) b₂ = (A.map f₂.op) m := (hm₀ f₂ h₂).symm
  have ha₀ : (α.app (Opposite.op Z)) a₀ = (R.map (f₁.op ≫ g₁.op)) r := by
    dsimp [a₀]
    rw [← RingCat.comp_apply, NatTrans.naturality, RingCat.comp_apply, ha₁, Functor.map_comp,
      RingCat.comp_apply]
  have hb₀ : (φ.app (Opposite.op Z)) b₀ = (A.map (f₁.op ≫ g₁.op)) m := by
    dsimp [b₀]
    erw [NatTrans.naturality_apply, hb₁, Functor.map_comp, comp_apply]
  have ha₀' : (α.app (Opposite.op Z)) a₀ = (R.map (f₂.op ≫ g₂.op)) r := by
    rw [ha₀, ← op_comp, fac, op_comp]
  have hb₀' : (φ.app (Opposite.op Z)) b₀ = (A.map (f₂.op ≫ g₂.op)) m := by
    rw [hb₀, ← op_comp, fac, op_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ha₁ : Eq ((α.app { unop := Y₁ }).hom a₁) ((R.map f₁.op).hom r)
    ha₂ : Eq ((α.app { unop := Y₂ }).hom a₂) ((R.map f₂.op).hom r)
    hb₁ : Eq ((φ.app { unop := Y₁ }) b₁) ((A.map f₁.op) m)
    hb₂ : Eq ((φ.app { unop := Y₂ }) b₂) ((A.map f₂.op) m)
    ha₀ : Eq ((α.app { unop := Z }).hom a₀) ((R.map (CategoryTheory.CategoryStruct …
    hb₀ : Eq ((φ.app { unop := Z }) b₀) ((A.map (CategoryTheory.CategoryStruct.com …
    ha₀' : Eq ((α.app { unop := Z }).hom a₀) ((R.map (CategoryTheory.CategoryStruc …
    hb₀' : Eq ((φ.app { unop := Z }) b₀) ((A.map (CategoryTheory.CategoryStruct.co …
    ⊢ Eq ((A.comp (CategoryTheory.forget Ab)).map g₁.op ((r₀.smul m₀).map (Categor …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ R : CategoryTheory.Functor (Opposite C) RingCat
    α : Quiver.Hom R₀ R
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Functor (Opposite C) AddCommGrp
    φ : Quiver.Hom M₀.presheaf A
    inst✝ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    hA : CategoryTheory.Presheaf.IsSeparated J A
    X : C
    r : ↑(R.obj { unop := X })
    m : ↑(A.obj { unop := X })
    P : CategoryTheory.Presieve X
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
    hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
    hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
    Y₁ Y₂ Z : C
    g₁ : Quiver.Hom Z Y₁
    g₂ : Quiver.Hom Z Y₂
    f₁ : Quiver.Hom Y₁ X
    f₂ : Quiver.Hom Y₂ X
    h₁ : P f₁
    h₂ : P f₂
    fac : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategorySt …
    a₁ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₁ } := r₀ f₁ h₁
    b₁ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₁ } := m₀ f₁ …
    a₂ : (R₀.comp (CategoryTheory.forget RingCat)).obj { unop := Y₂ } := r₀ f₂ h₂
    b₂ : (M₀.presheaf.comp (CategoryTheory.forget Ab)).obj { unop := Y₂ } := m₀ f₂ …
    a₀ : ↑(R₀.obj { unop := Z }) := (R₀.map g₁.op).hom a₁
    b₀ : ↑((ModuleCat.restrictScalars (R₀.map g₁.op).hom).obj (M₀.obj { unop := Z  …
    ha₁ : Eq ((α.app { unop := Y₁ }).hom a₁) ((R.map f₁.op).hom r)
    ha₂ : Eq ((α.app { unop := Y₂ }).hom a₂) ((R.map f₂.op).hom r)
    hb₁ : Eq ((φ.app { unop := Y₁ }) b₁) ((A.map f₁.op) m)
    hb₂ : Eq ((φ.app { unop := Y₂ }) b₂) ((A.map f₂.op) m)
    ha₀ : Eq ((α.app { unop := Z }).hom a₀) ((R.map (CategoryTheory.CategoryStruct …
    hb₀ : Eq ((φ.app { unop := Z }) b₀) ((A.map (CategoryTheory.CategoryStruct.com …
    ha₀' : Eq ((α.app { unop := Z }).hom a₀) ((R.map (CategoryTheory.CategoryStruc …
    hb₀' : Eq ((φ.app { unop := Z }) b₀) ((A.map (CategoryTheory.CategoryStruct.co …
    ⊢ Eq ((A.map g₁.op) ((φ.app { unop := Y₁ }) (r₀.smul m₀ f₁ h₁))) ((A.map g₂.op …
  -/
  erw [← NatTrans.naturality_apply, ← NatTrans.naturality_apply]
  exact (isCompatible_map_smul_aux α φ hA r m f₁ g₁ a₁ a₀ b₁ b₀ ha₁ ha₀ hb₁ hb₀).trans
    (isCompatible_map_smul_aux α φ hA r m f₂ g₂ a₂ a₀ b₂ b₀ ha₂ ha₀' hb₂ hb₀').symm


/-- Assuming `α : R₀ ⟶ R.val` is the sheafification map of a presheaf of rings `R₀`
and `φ : M₀.presheaf ⟶ A.val` is the sheafification map of the underlying
sheaf of abelian groups of a presheaf of modules `M₀` over `R₀`, then given
`r : R.val.obj X` and `m : A.val.obj X`, this structure contains the data
of `x : A.val.obj X` along with the property which makes `x` a good candidate
for the definition of the scalar multiplication `r • m`. -/
structure SMulCandidate where
  /-- The candidate for the scalar product `r • m`. -/
  x : A.val.obj X
  h ⦃Y : Cᵒᵖ⦄ (f : X ⟶ Y) (r₀ : R₀.obj Y) (hr₀ : α.app Y r₀ = R.val.map f r)
    (m₀ : M₀.obj Y) (hm₀ : φ.app Y m₀ = A.val.map f m) : A.val.map f x = φ.app Y (r₀ • m₀)


/-- Constructor for `SMulCandidate`. -/
def SMulCandidate.mk' (S : Sieve X.unop) (hS : S ∈ J X.unop)
    (r₀ : Presieve.FamilyOfElements (R₀ ⋙ forget _) S.arrows)
    (m₀ : Presieve.FamilyOfElements (M₀.presheaf ⋙ forget _) S.arrows)
    (hr₀ : (r₀.map (whiskerRight α (forget _))).IsAmalgamation r)
    (hm₀ : (m₀.map (whiskerRight φ (forget _))).IsAmalgamation m)
    (a : A.val.obj X)
    (ha : ((r₀.smul m₀).map (whiskerRight φ (forget _))).IsAmalgamation a) :
    SMulCandidate α φ r m where
  x := a
  h Y f a₀ ha₀ b₀ hb₀ := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      S : CategoryTheory.Sieve (Opposite.unop X)
      hS : Membership.mem (J (Opposite.unop X)) S
      r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
      m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
      hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
      hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
      a : ↑(A.val.obj X)
      ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
      Y : Opposite C
      f : Quiver.Hom X Y
      a₀ : ↑(R₀.obj Y)
      ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
      b₀ : ↑(M₀.obj Y)
      hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
      ⊢ Eq ((A.val.map f) a) ((φ.app Y) (HSMul.hSMul a₀ b₀))
    -/
    apply A.isSeparated _ _ (J.pullback_stable f.unop hS)
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      S : CategoryTheory.Sieve (Opposite.unop X)
      hS : Membership.mem (J (Opposite.unop X)) S
      r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
      m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
      hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
      hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
      a : ↑(A.val.obj X)
      ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
      Y : Opposite C
      f : Quiver.Hom X Y
      a₀ : ↑(R₀.obj Y)
      ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
      b₀ : ↑(M₀.obj Y)
      hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
      ⊢ ∀ (Y_1 : C) (f_1 : Quiver.Hom Y_1 (Opposite.unop Y)), (CategoryTheory.Sieve. …
    -/
    rintro Z g hg
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      S : CategoryTheory.Sieve (Opposite.unop X)
      hS : Membership.mem (J (Opposite.unop X)) S
      r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
      m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
      hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
      hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
      a : ↑(A.val.obj X)
      ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
      Y : Opposite C
      f : Quiver.Hom X Y
      a₀ : ↑(R₀.obj Y)
      ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
      b₀ : ↑(M₀.obj Y)
      hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
      Z : C
      g : Quiver.Hom Z (Opposite.unop Y)
      hg : (CategoryTheory.Sieve.pullback f.unop S).arrows g
      ⊢ Eq ((A.val.map g.op) ((A.val.map f) a)) ((A.val.map g.op) ((φ.app Y) (HSMul. …
    -/
    dsimp at hg
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      S : CategoryTheory.Sieve (Opposite.unop X)
      hS : Membership.mem (J (Opposite.unop X)) S
      r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
      m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
      hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
      hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
      a : ↑(A.val.obj X)
      ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
      Y : Opposite C
      f : Quiver.Hom X Y
      a₀ : ↑(R₀.obj Y)
      ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
      b₀ : ↑(M₀.obj Y)
      hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
      Z : C
      g : Quiver.Hom Z (Opposite.unop Y)
      hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
      ⊢ Eq ((A.val.map g.op) ((A.val.map f) a)) ((A.val.map g.op) ((φ.app Y) (HSMul. …
    -/
    erw [← comp_apply, ← A.val.map_comp, ← NatTrans.naturality_apply, M₀.map_smul]
    /-
      case a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      S : CategoryTheory.Sieve (Opposite.unop X)
      hS : Membership.mem (J (Opposite.unop X)) S
      r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
      m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
      hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
      hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
      a : ↑(A.val.obj X)
      ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
      Y : Opposite C
      f : Quiver.Hom X Y
      a₀ : ↑(R₀.obj Y)
      ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
      b₀ : ↑(M₀.obj Y)
      hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
      Z : C
      g : Quiver.Hom Z (Opposite.unop Y)
      hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
      ⊢ Eq ((A.val.map (CategoryTheory.CategoryStruct.comp f g.op)) a) ((φ.app { uno …
    -/
    refine (ha _ hg).trans (app_eq_of_isLocallyInjective α φ A.isSeparated _ _ _ _ ?_ ?_)
      /-
        case a.refine_1
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((α.app { unop := Z }).hom (r₀ (CategoryTheory.CategoryStruct.comp g f.un …
      -/
    · rw [← RingCat.comp_apply, NatTrans.naturality, RingCat.comp_apply, ha₀]
      /-
        case a.refine_1
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((α.app { unop := Z }).hom (r₀ (CategoryTheory.CategoryStruct.comp g f.un …
      -/
      apply (hr₀ _ hg).symm.trans
      /-
        case a.refine_1
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((R.val.comp (CategoryTheory.forget RingCat)).map (CategoryTheory.Categor …
      -/
      simp [RingCat.forget_map]
      /-
        🎉 no goals
      -/
      /-
        case a.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((φ.app { unop := Z }) (m₀ (CategoryTheory.CategoryStruct.comp g f.unop)  …
      -/
    · erw [NatTrans.naturality_apply, hb₀]
      /-
        case a.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((φ.app { unop := Z }) (m₀ (CategoryTheory.CategoryStruct.comp g f.unop)  …
      -/
      apply (hm₀ _ hg).symm.trans
      /-
        case a.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((A.val.comp (CategoryTheory.forget Ab)).map (CategoryTheory.CategoryStru …
      -/
      dsimp
      /-
        case a.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((A.val.map (CategoryTheory.CategoryStruct.comp f g.op)) m) ((A.val.map g …
      -/
      rw [Functor.map_comp]
      /-
        case a.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        J : CategoryTheory.GrothendieckTopology C
        R₀ : CategoryTheory.Functor (Opposite C) RingCat
        R : CategoryTheory.Sheaf J RingCat
        α : Quiver.Hom R₀ R.val
        inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
        inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
        M₀ : PresheafOfModules R₀
        A : CategoryTheory.Sheaf J AddCommGrp
        φ : Quiver.Hom M₀.presheaf A.val
        inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
        inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
        X Y✝ : Opposite C
        π : Quiver.Hom X Y✝
        r r' : ↑(R.val.obj X)
        m m' : ↑(A.val.obj X)
        S : CategoryTheory.Sieve (Opposite.unop X)
        hS : Membership.mem (J (Opposite.unop X)) S
        r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
        m₀ : CategoryTheory.Presieve.FamilyOfElements (M₀.presheaf.comp (CategoryTheor …
        hr₀ : (r₀.map (CategoryTheory.whiskerRight α (CategoryTheory.forget RingCat))) …
        hm₀ : (m₀.map (CategoryTheory.whiskerRight φ (CategoryTheory.forget Ab))).IsAm …
        a : ↑(A.val.obj X)
        ha : ((r₀.smul m₀).map (CategoryTheory.whiskerRight φ (CategoryTheory.forget A …
        Y : Opposite C
        f : Quiver.Hom X Y
        a₀ : ↑(R₀.obj Y)
        ha₀ : Eq ((α.app Y).hom a₀) ((R.val.map f).hom r)
        b₀ : ↑(M₀.obj Y)
        hb₀ : Eq ((φ.app Y) b₀) ((A.val.map f) m)
        Z : C
        g : Quiver.Hom Z (Opposite.unop Y)
        hg : S.arrows (CategoryTheory.CategoryStruct.comp g f.unop)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (A.val.map f) (A.val.map g.op)) m) ( …
      -/
      rfl
      /-
        🎉 no goals
      -/


instance : Nonempty (SMulCandidate α φ r m) := ⟨by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r r' : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    ⊢ PresheafOfModules.Sheafify.SMulCandidate α φ r m
  -/
  let S := (Presheaf.imageSieve α r ⊓ Presheaf.imageSieve φ m)
  have hS : S ∈ J _ := by
    apply J.intersection_covering
    all_goals apply Presheaf.imageSieve_mem
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r r' : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
    hS : Membership.mem (J (Opposite.unop X)) S
    ⊢ PresheafOfModules.Sheafify.SMulCandidate α φ r m
  -/
  have h₁ : S ≤ Presheaf.imageSieve α r := fun _ _ h => h.1
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r r' : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
    hS : Membership.mem (J (Opposite.unop X)) S
    h₁ : LE.le S (CategoryTheory.Presheaf.imageSieve α r)
    ⊢ PresheafOfModules.Sheafify.SMulCandidate α φ r m
  -/
  have h₂ : S ≤ Presheaf.imageSieve φ m := fun _ _ h => h.2
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r r' : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
    hS : Membership.mem (J (Opposite.unop X)) S
    h₁ : LE.le S (CategoryTheory.Presheaf.imageSieve α r)
    h₂ : LE.le S (CategoryTheory.Presheaf.imageSieve φ m)
    ⊢ PresheafOfModules.Sheafify.SMulCandidate α φ r m
  -/
  let r₀ := (Presieve.FamilyOfElements.localPreimage (whiskerRight α (forget _)) r).restrict h₁
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r r' : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
    hS : Membership.mem (J (Opposite.unop X)) S
    h₁ : LE.le S (CategoryTheory.Presheaf.imageSieve α r)
    h₂ : LE.le S (CategoryTheory.Presheaf.imageSieve φ m)
    r₀ : CategoryTheory.Presieve.FamilyOfElements (R₀.comp (CategoryTheory.forget  …
    ⊢ PresheafOfModules.Sheafify.SMulCandidate α φ r m
  -/
  let m₀ := (Presieve.FamilyOfElements.localPreimage (whiskerRight φ (forget _)) m).restrict h₂
  have hr₀ : (r₀.map (whiskerRight α (forget _))).IsAmalgamation r := by
    rw [Presieve.FamilyOfElements.restrict_map]
    apply Presieve.isAmalgamation_restrict
    apply Presieve.FamilyOfElements.isAmalgamation_map_localPreimage
  have hm₀ : (m₀.map (whiskerRight φ (forget _))).IsAmalgamation m := by
    rw [Presieve.FamilyOfElements.restrict_map]
    apply Presieve.isAmalgamation_restrict
    apply Presieve.FamilyOfElements.isAmalgamation_map_localPreimage
  exact SMulCandidate.mk' α φ r m S hS r₀ m₀ hr₀ hm₀ _ (Presieve.IsSheafFor.isAmalgamation
    (((sheafCompose J (forget _)).obj A).2.isSheafFor S hS)
    (Presieve.FamilyOfElements.isCompatible_map_smul α φ A.isSeparated r m r₀ m₀ hr₀ hm₀))⟩


instance : Subsingleton (SMulCandidate α φ r m) where
  allEq := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y : Opposite C
      π : Quiver.Hom X Y
      r r' : ↑(R.val.obj X)
      m m' : ↑(A.val.obj X)
      ⊢ ∀ (a b : PresheafOfModules.Sheafify.SMulCandidate α φ r m), Eq a b
    -/
    rintro ⟨x₁, h₁⟩ ⟨x₂, h₂⟩
    /-
      case mk.mk
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y : Opposite C
      π : Quiver.Hom X Y
      r r' : ↑(R.val.obj X)
      m m' x₁ : ↑(A.val.obj X)
      h₁ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      x₂ : ↑(A.val.obj X)
      h₂ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      ⊢ Eq { x := x₁, h := h₁ } { x := x₂, h := h₂ }
    -/
    simp only [SMulCandidate.mk.injEq]
    /-
      case mk.mk
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y : Opposite C
      π : Quiver.Hom X Y
      r r' : ↑(R.val.obj X)
      m m' x₁ : ↑(A.val.obj X)
      h₁ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      x₂ : ↑(A.val.obj X)
      h₂ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      ⊢ Eq x₁ x₂
    -/
    let S := (Presheaf.imageSieve α r ⊓ Presheaf.imageSieve φ m)
    have hS : S ∈ J _ := by
      apply J.intersection_covering
      all_goals apply Presheaf.imageSieve_mem
    /-
      case mk.mk
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y : Opposite C
      π : Quiver.Hom X Y
      r r' : ↑(R.val.obj X)
      m m' x₁ : ↑(A.val.obj X)
      h₁ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      x₂ : ↑(A.val.obj X)
      h₂ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
      hS : Membership.mem (J (Opposite.unop X)) S
      ⊢ Eq x₁ x₂
    -/
    apply A.isSeparated _ _ hS
    /-
      case mk.mk.a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y : Opposite C
      π : Quiver.Hom X Y
      r r' : ↑(R.val.obj X)
      m m' x₁ : ↑(A.val.obj X)
      h₁ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      x₂ : ↑(A.val.obj X)
      h₂ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
      hS : Membership.mem (J (Opposite.unop X)) S
      ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), S.arrows f → Eq ((A.val.map  …
    -/
    intro Y f ⟨⟨r₀, hr₀⟩, ⟨m₀, hm₀⟩⟩
    /-
      case mk.mk.a
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      X Y✝ : Opposite C
      π : Quiver.Hom X Y✝
      r r' : ↑(R.val.obj X)
      m m' x₁ : ↑(A.val.obj X)
      h₁ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      x₂ : ↑(A.val.obj X)
      h₂ : ∀ ⦃Y : Opposite C⦄ (f : Quiver.Hom X Y) (r₀ : ↑(R₀.obj Y)), Eq ((α.app Y) …
      S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (CategoryTheory.Presheaf …
      hS : Membership.mem (J (Opposite.unop X)) S
      Y : C
      f : Quiver.Hom Y (Opposite.unop X)
      r₀ : (CategoryTheory.forget RingCat).obj (R₀.obj { unop := Y })
      hr₀ : Eq ((α.app { unop := Y }) r₀) ((R.val.map f.op) r)
      m₀ : (CategoryTheory.forget Ab).obj (M₀.presheaf.obj { unop := Y })
      hm₀ : Eq ((φ.app { unop := Y }) m₀) ((A.val.map f.op) m)
      ⊢ Eq ((A.val.map f.op) x₁) ((A.val.map f.op) x₂)
    -/
    rw [h₁ f.op r₀ hr₀ m₀ hm₀, h₂ f.op r₀ hr₀ m₀ hm₀]
    /-
      🎉 no goals
    -/


noncomputable instance : Unique (SMulCandidate α φ r m) :=
  uniqueOfSubsingleton (Nonempty.some inferInstance)


/-- The (unique) element in `SMulCandidate α φ r m`. -/
noncomputable def smulCandidate : SMulCandidate α φ r m := default


/-- The scalar multiplication on the sheafification of a presheaf of modules. -/
noncomputable def smul : A.val.obj X := (smulCandidate α φ r m).x


lemma map_smul_eq {Y : Cᵒᵖ} (f : X ⟶ Y) (r₀ : R₀.obj Y) (hr₀ : α.app Y r₀ = R.val.map f r)
    (m₀ : M₀.obj Y) (hm₀ : φ.app Y m₀ = A.val.map f m) :
    A.val.map f (smul α φ r m) = φ.app Y (r₀ • m₀) :=
  (smulCandidate α φ r m).h f r₀ hr₀ m₀ hm₀


protected lemma one_smul : smul α φ 1 m = m := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    m : ↑(A.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ 1 m) m
  -/
  apply A.isSeparated _ _ (Presheaf.imageSieve_mem J φ m)
  /-
    case a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    m : ↑(A.val.obj X)
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), (CategoryTheory.Presheaf.ima …
  -/
  rintro Y f ⟨m₀, hm₀⟩
  /-
    case a.intro
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    m : ↑(A.val.obj X)
    Y : C
    f : Quiver.Hom Y (Opposite.unop X)
    m₀ : (CategoryTheory.forget Ab).obj (M₀.presheaf.obj { unop := Y })
    hm₀ : Eq ((φ.app { unop := Y }) m₀) ((A.val.map f.op) m)
    ⊢ Eq ((A.val.map f.op) (PresheafOfModules.Sheafify.smul α φ 1 m)) ((A.val.map  …
  -/
  rw [← hm₀, map_smul_eq α φ 1 m f.op 1 (by simp) m₀ hm₀, one_smul]
  /-
    🎉 no goals
  -/


protected lemma zero_smul : smul α φ 0 m = 0 := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    m : ↑(A.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ 0 m) 0
  -/
  apply A.isSeparated _ _ (Presheaf.imageSieve_mem J φ m)
  /-
    case a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    m : ↑(A.val.obj X)
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), (CategoryTheory.Presheaf.ima …
  -/
  rintro Y f ⟨m₀, hm₀⟩
  rw [map_smul_eq α φ 0 m f.op 0 (by simp) m₀ hm₀, zero_smul, map_zero,
    (A.val.map f.op).map_zero]


protected lemma smul_zero : smul α φ r 0 = 0 := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r : ↑(R.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ r 0) 0
  -/
  apply A.isSeparated _ _ (Presheaf.imageSieve_mem J α r)
  /-
    case a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r : ↑(R.val.obj X)
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), (CategoryTheory.Presheaf.ima …
  -/
  rintro Y f ⟨r₀, hr₀⟩
  rw [(A.val.map f.op).map_zero, map_smul_eq α φ r 0 f.op r₀ hr₀ 0 (by simp),
    smul_zero, map_zero]


protected lemma smul_add : smul α φ r (m + m') = smul α φ r m + smul α φ r m' := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ r (HAdd.hAdd m m')) (HAdd.hAdd (Pres …
  -/
  let S := Presheaf.imageSieve α r ⊓ Presheaf.imageSieve φ m ⊓ Presheaf.imageSieve φ m'
  have hS : S ∈ J X.unop := by
    refine J.intersection_covering (J.intersection_covering ?_ ?_) ?_
    all_goals apply Presheaf.imageSieve_mem
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (Min.min (CategoryTheory …
    hS : Membership.mem (J (Opposite.unop X)) S
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ r (HAdd.hAdd m m')) (HAdd.hAdd (Pres …
  -/
  apply A.isSeparated _ _ hS
  /-
    case a
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r : ↑(R.val.obj X)
    m m' : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (Min.min (CategoryTheory …
    hS : Membership.mem (J (Opposite.unop X)) S
    ⊢ ∀ (Y : C) (f : Quiver.Hom Y (Opposite.unop X)), S.arrows f → Eq ((A.val.map  …
  -/
  rintro Y f ⟨⟨⟨r₀, hr₀⟩, ⟨m₀ : M₀.obj _, hm₀⟩⟩, ⟨m₀' : M₀.obj _, hm₀'⟩⟩
  rw [(A.val.map f.op).map_add, map_smul_eq α φ r m f.op r₀ hr₀ m₀ hm₀,
    map_smul_eq α φ r m' f.op r₀ hr₀ m₀' hm₀',
    map_smul_eq α φ r (m + m') f.op r₀ hr₀ (m₀ + m₀')
      (by rw [map_add, map_add, hm₀, hm₀']),
    smul_add, map_add]


protected lemma add_smul : smul α φ (r + r') m = smul α φ r m + smul α φ r' m := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r r' : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ (HAdd.hAdd r r') m) (HAdd.hAdd (Pres …
  -/
  let S := Presheaf.imageSieve α r ⊓ Presheaf.imageSieve α r' ⊓ Presheaf.imageSieve φ m
  have hS : S ∈ J X.unop := by
    refine J.intersection_covering (J.intersection_covering ?_ ?_) ?_
    all_goals apply Presheaf.imageSieve_mem
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r r' : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (Min.min (CategoryTheory …
    hS : Membership.mem (J (Opposite.unop X)) S
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ (HAdd.hAdd r r') m) (HAdd.hAdd (Pres …
  -/
  apply A.isSeparated _ _ hS
  rintro Y f ⟨⟨⟨r₀ : R₀.obj _, (hr₀ : (α.app (Opposite.op Y)) r₀ = (R.val.map f.op) r)⟩,
    ⟨r₀' : R₀.obj _, (hr₀' : (α.app (Opposite.op Y)) r₀' = (R.val.map f.op) r')⟩⟩, ⟨m₀, hm₀⟩⟩
  rw [(A.val.map f.op).map_add, map_smul_eq α φ r m f.op r₀ hr₀ m₀ hm₀,
    map_smul_eq α φ r' m f.op r₀' hr₀' m₀ hm₀,
    map_smul_eq α φ (r + r') m f.op (r₀ + r₀') (by rw [map_add, map_add, hr₀, hr₀'])
      m₀ hm₀, add_smul, map_add]


protected lemma mul_smul : smul α φ (r * r') m = smul α φ r (smul α φ r' m) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r r' : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ (HMul.hMul r r') m) (PresheafOfModul …
  -/
  let S := Presheaf.imageSieve α r ⊓ Presheaf.imageSieve α r' ⊓ Presheaf.imageSieve φ m
  have hS : S ∈ J X.unop := by
    refine J.intersection_covering (J.intersection_covering ?_ ?_) ?_
    all_goals apply Presheaf.imageSieve_mem
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X : Opposite C
    r r' : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop X) := Min.min (Min.min (CategoryTheory …
    hS : Membership.mem (J (Opposite.unop X)) S
    ⊢ Eq (PresheafOfModules.Sheafify.smul α φ (HMul.hMul r r') m) (PresheafOfModul …
  -/
  apply A.isSeparated _ _ hS
  rintro Y f ⟨⟨⟨r₀ : R₀.obj _, (hr₀ : (α.app (Opposite.op Y)) r₀ = (R.val.map f.op) r)⟩,
    ⟨r₀' : R₀.obj _, (hr₀' : (α.app (Opposite.op Y)) r₀' = (R.val.map f.op) r')⟩⟩,
    ⟨m₀ : M₀.obj _, hm₀⟩⟩
  rw [map_smul_eq α φ (r * r') m f.op (r₀ * r₀')
    (by rw [map_mul, map_mul, hr₀, hr₀']) m₀ hm₀, mul_smul,
    map_smul_eq α φ r (smul α φ r' m) f.op r₀ hr₀ (r₀' • m₀)
      (map_smul_eq α φ r' m f.op r₀' hr₀' m₀ hm₀).symm]


/-- The module structure on the sections of the sheafification of the underlying
presheaf of abelian groups of a presheaf of modules. -/
noncomputable def module : Module (R.val.obj X) (A.val.obj X) where
  smul r m := smul α φ r m
  one_smul := Sheafify.one_smul α φ
  zero_smul := Sheafify.zero_smul α φ
  smul_zero := Sheafify.smul_zero α φ
  smul_add := Sheafify.smul_add α φ
  add_smul := Sheafify.add_smul α φ
  mul_smul := Sheafify.mul_smul α φ


lemma map_smul :
    A.val.map π (smul α φ r m) = smul α φ (R.val.map π r) (A.val.map π m) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    ⊢ Eq ((A.val.map π) (PresheafOfModules.Sheafify.smul α φ r m)) (PresheafOfModu …
  -/
  let S := Presheaf.imageSieve α (R.val.map π r) ⊓ Presheaf.imageSieve φ (A.val.map π m)
  have hS : S ∈ J Y.unop := by
    apply J.intersection_covering
    all_goals apply Presheaf.imageSieve_mem
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    X Y : Opposite C
    π : Quiver.Hom X Y
    r : ↑(R.val.obj X)
    m : ↑(A.val.obj X)
    S : CategoryTheory.Sieve (Opposite.unop Y) := Min.min (CategoryTheory.Presheaf …
    hS : Membership.mem (J (Opposite.unop Y)) S
    ⊢ Eq ((A.val.map π) (PresheafOfModules.Sheafify.smul α φ r m)) (PresheafOfModu …
  -/
  apply A.isSeparated _ _ hS
  rintro Y f ⟨⟨r₀,
    (hr₀ : (α.app (Opposite.op Y)).hom r₀ = (R.val.map f.op).hom ((R.val.map π).hom r))⟩, ⟨m₀, hm₀⟩⟩
  rw [← comp_apply, ← Functor.map_comp,
    map_smul_eq α φ r m (π ≫ f.op) r₀ (by rw [hr₀, Functor.map_comp, RingCat.comp_apply]) m₀
      (by rw [hm₀, Functor.map_comp, comp_apply]),
    map_smul_eq α φ (R.val.map π r) (A.val.map π m) f.op r₀ hr₀ m₀ hm₀]


/-- Assuming `α : R₀ ⟶ R.val` is the sheafification map of a presheaf of rings `R₀`
and `φ : M₀.presheaf ⟶ A.val` is the sheafification map of the underlying
sheaf of abelian groups of a presheaf of modules `M₀` over `R₀`, this is
the sheaf of modules over `R` which is obtained by endowing the sections of
`A.val` with a scalar multiplication. -/
noncomputable def sheafify : SheafOfModules.{v} R where
  val := letI := Sheafify.module α φ; ofPresheaf A.val (Sheafify.map_smul _ _)
  isSheaf := A.cond


/-- The canonical morphism from a presheaf of modules to its associated sheaf. -/
def toSheafify : M₀ ⟶ (restrictScalars α).obj (sheafify α φ).val :=
  homMk φ (fun X r₀ m₀ ↦ by
    simpa using (Sheafify.map_smul_eq α φ (α.app _ r₀) (φ.app _ m₀) (𝟙 _)
      r₀ (by aesop) m₀ (by simp)).symm)


lemma toSheafify_app_apply (X : Cᵒᵖ) (x : M₀.obj X) :
    ((toSheafify α φ).app X).hom x = φ.app X x := rfl


/-- `@[simp]`-normal form of `toSheafify_app_apply`. -/
@[simp]
lemma toSheafify_app_apply' (X : Cᵒᵖ) (x : M₀.obj X) :
    DFunLike.coe (F := (_ →ₗ[_] ↑((ModuleCat.restrictScalars (α.app X).hom).obj _)))
    ((toSheafify α φ).app X).hom x = φ.app X x := rfl


@[simp]
lemma toPresheaf_map_toSheafify : (toPresheaf R₀).map (toSheafify α φ) = φ := rfl


instance : IsLocallyInjective J (toSheafify α φ) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    ⊢ PresheafOfModules.IsLocallyInjective J (PresheafOfModules.toSheafify α φ)
  -/
  dsimp [IsLocallyInjective]; infer_instance
                              /-
                                🎉 no goals
                              -/


instance : IsLocallySurjective J (toSheafify α φ) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    R₀ : CategoryTheory.Functor (Opposite C) RingCat
    R : CategoryTheory.Sheaf J RingCat
    α : Quiver.Hom R₀ R.val
    inst✝³ : CategoryTheory.Presheaf.IsLocallyInjective J α
    inst✝² : CategoryTheory.Presheaf.IsLocallySurjective J α
    M₀ : PresheafOfModules R₀
    A : CategoryTheory.Sheaf J AddCommGrp
    φ : Quiver.Hom M₀.presheaf A.val
    inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ
    inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ
    ⊢ PresheafOfModules.IsLocallySurjective J (PresheafOfModules.toSheafify α φ)
  -/
  dsimp [IsLocallySurjective]; infer_instance
                               /-
                                 🎉 no goals
                               -/


/-- The bijection `((sheafify α φ).val ⟶ F) ≃ (M₀ ⟶ (restrictScalars α).obj F)` which
is part of the universal property of the sheafification of the presheaf of modules `M₀`,
when `F` is a presheaf of modules which is a sheaf. -/
noncomputable def sheafifyHomEquiv' {F : PresheafOfModules.{v} R.val}
    (hF : Presheaf.IsSheaf J F.presheaf) :
    ((sheafify α φ).val ⟶ F) ≃ (M₀ ⟶ (restrictScalars α).obj F) :=
  (restrictHomEquivOfIsLocallySurjective α hF).trans
    (homEquivOfIsLocallyBijective (f := toSheafify α φ)
      (N := (restrictScalars α).obj F) hF)


lemma comp_toPresheaf_map_sheafifyHomEquiv'_symm_hom {F : PresheafOfModules.{v} R.val}
    (hF : Presheaf.IsSheaf J F.presheaf) (f : M₀ ⟶ (restrictScalars α).obj F) :
    φ ≫ (toPresheaf R.val).map ((sheafifyHomEquiv' α φ hF).symm f) = (toPresheaf R₀).map f :=
  (toPresheaf _).congr_map ((sheafifyHomEquiv' α φ hF).apply_symm_apply f)


/-- The bijection
`(sheafify α φ ⟶ F) ≃ (M₀ ⟶ (restrictScalars α).obj ((SheafOfModules.forget _).obj F))`
which is part of the universal property of the sheafification of the presheaf of modules `M₀`,
for any sheaf of modules `F`, see `PresheafOfModules.sheafificationAdjunction` -/
noncomputable def sheafifyHomEquiv {F : SheafOfModules.{v} R} :
    (sheafify α φ ⟶ F) ≃
      (M₀ ⟶ (restrictScalars α).obj ((SheafOfModules.forget _).obj F)) :=
  (SheafOfModules.fullyFaithfulForget R).homEquiv.trans
    (sheafifyHomEquiv' α φ F.isSheaf)


/-- The morphism of sheaves of modules `sheafify α φ ⟶ sheafify α φ'`
induced by morphisms `τ₀ : M₀ ⟶ M₀'` and `τ : A ⟶ A'`
which satisfy `τ₀.hom ≫ φ' = φ ≫ τ.val`. -/
@[simps]
def sheafifyMap (fac : (toPresheaf R₀).map τ₀ ≫ φ' = φ ≫ τ.val) :
    sheafify α φ ⟶ sheafify α φ' where
  val := homMk τ.val (fun X r m ↦ by
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      ⊢ Eq ((τ.val.app X) (HSMul.hSMul r m)) (HSMul.hSMul r ((τ.val.app X) m))
    -/
    let f := (sheafifyHomEquiv' α φ (by exact A'.cond)).symm (τ₀ ≫ toSheafify α φ')
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      f : Quiver.Hom (PresheafOfModules.sheafify α φ).val (PresheafOfModules.sheafif …
      ⊢ Eq ((τ.val.app X) (HSMul.hSMul r m)) (HSMul.hSMul r ((τ.val.app X) m))
    -/
    suffices τ.val = (toPresheaf _).map f by simpa only [this] using (f.app X).hom.map_smul r m
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      f : Quiver.Hom (PresheafOfModules.sheafify α φ).val (PresheafOfModules.sheafif …
      ⊢ Eq τ.val ((PresheafOfModules.toPresheaf R.val).map f)
    -/
    apply ((J.W_of_isLocallyBijective φ).homEquiv _ A'.cond).injective
    /-
      case a
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      f : Quiver.Hom (PresheafOfModules.sheafify α φ).val (PresheafOfModules.sheafif …
      ⊢ Eq ((CategoryTheory.Localization.LeftBousfield.W.homEquiv ⋯ A'.val ⋯) τ.val) …
    -/
    dsimp [f]
    /-
      case a
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      f : Quiver.Hom (PresheafOfModules.sheafify α φ).val (PresheafOfModules.sheafif …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ τ.val) (CategoryTheory.CategoryStru …
    -/
    erw [comp_toPresheaf_map_sheafifyHomEquiv'_symm_hom]
    /-
      case a
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      R₀ : CategoryTheory.Functor (Opposite C) RingCat
      R : CategoryTheory.Sheaf J RingCat
      α : Quiver.Hom R₀ R.val
      inst✝⁶ : CategoryTheory.Presheaf.IsLocallyInjective J α
      inst✝⁵ : CategoryTheory.Presheaf.IsLocallySurjective J α
      M₀ : PresheafOfModules R₀
      A : CategoryTheory.Sheaf J AddCommGrp
      φ : Quiver.Hom M₀.presheaf A.val
      inst✝⁴ : CategoryTheory.Presheaf.IsLocallyInjective J φ
      inst✝³ : CategoryTheory.Presheaf.IsLocallySurjective J φ
      inst✝² : J.WEqualsLocallyBijective AddCommGrp
      M₀' : PresheafOfModules R₀
      A' : CategoryTheory.Sheaf J AddCommGrp
      φ' : Quiver.Hom M₀'.presheaf A'.val
      inst✝¹ : CategoryTheory.Presheaf.IsLocallyInjective J φ'
      inst✝ : CategoryTheory.Presheaf.IsLocallySurjective J φ'
      τ₀ : Quiver.Hom M₀ M₀'
      τ : Quiver.Hom A A'
      fac : Eq (CategoryTheory.CategoryStruct.comp ((PresheafOfModules.toPresheaf R₀ …
      X : Opposite C
      r : ↑(R.val.obj X)
      m : ↑((PresheafOfModules.sheafify α φ).val.obj X)
      f : Quiver.Hom (PresheafOfModules.sheafify α φ).val (PresheafOfModules.sheafif …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ τ.val) ((PresheafOfModules.toPreshe …
    -/
    rw [← fac, Functor.map_comp, toPresheaf_map_toSheafify])
    /-
      🎉 no goals
    -/


