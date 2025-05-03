theorem has_injective_coseparator [HasLimits C] [EnoughInjectives C] (G : C) (hG : IsSeparator G) :
    ∃ G : C, Injective G ∧ IsCoseparator G := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    ⊢ Exists fun G => And (CategoryTheory.Injective G) (CategoryTheory.IsCoseparat …
  -/
  haveI : WellPowered.{v} C := wellPowered_of_isDetector G hG.isDetector
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this : CategoryTheory.WellPowered.{v, v, u} C
    ⊢ Exists fun G => And (CategoryTheory.Injective G) (CategoryTheory.IsCoseparat …
  -/
  haveI : HasProductsOfShape (Subobject (op G)) C := hasProductsOfShape_of_small _ _
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this✝ : CategoryTheory.WellPowered.{v, v, u} C
    this : CategoryTheory.Limits.HasProductsOfShape (CategoryTheory.Subobject { un …
    ⊢ Exists fun G => And (CategoryTheory.Injective G) (CategoryTheory.IsCoseparat …
  -/
  let T : C := Injective.under (piObj fun P : Subobject (op G) => unop P)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this✝ : CategoryTheory.WellPowered.{v, v, u} C
    this : CategoryTheory.Limits.HasProductsOfShape (CategoryTheory.Subobject { un …
    T : C := CategoryTheory.Injective.under (CategoryTheory.Limits.piObj fun P =>  …
    ⊢ Exists fun G => And (CategoryTheory.Injective G) (CategoryTheory.IsCoseparat …
  -/
  refine ⟨T, inferInstance, (Preadditive.isCoseparator_iff _).2 fun X Y f hf => ?_⟩
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this✝ : CategoryTheory.WellPowered.{v, v, u} C
    this : CategoryTheory.Limits.HasProductsOfShape (CategoryTheory.Subobject { un …
    T : C := CategoryTheory.Injective.under (CategoryTheory.Limits.piObj fun P =>  …
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (h : Quiver.Hom Y T), Eq (CategoryTheory.CategoryStruct.comp f h) 0
    ⊢ Eq f 0
  -/
  refine (Preadditive.isSeparator_iff _).1 hG _ fun h => ?_
  suffices hh : factorThruImage (h ≫ f) = 0 by
    rw [← Limits.image.fac (h ≫ f), hh, zero_comp]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this✝ : CategoryTheory.WellPowered.{v, v, u} C
    this : CategoryTheory.Limits.HasProductsOfShape (CategoryTheory.Subobject { un …
    T : C := CategoryTheory.Injective.under (CategoryTheory.Limits.piObj fun P =>  …
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (h : Quiver.Hom Y T), Eq (CategoryTheory.CategoryStruct.comp f h) 0
    h : Quiver.Hom G X
    ⊢ Eq (CategoryTheory.Limits.factorThruImage (CategoryTheory.CategoryStruct.com …
  -/
  let R := Subobject.mk (factorThruImage (h ≫ f)).op
  let q₁ : image (h ≫ f) ⟶ unop R :=
    (Subobject.underlyingIso (factorThruImage (h ≫ f)).op).unop.hom
  let q₂ : unop (R : Cᵒᵖ) ⟶ piObj fun P : Subobject (op G) => unop P :=
    section_ (Pi.π (fun P : Subobject (op G) => (unop P : C)) R)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasLimits C
    inst✝ : CategoryTheory.EnoughInjectives C
    G : C
    hG : CategoryTheory.IsSeparator G
    this✝ : CategoryTheory.WellPowered.{v, v, u} C
    this : CategoryTheory.Limits.HasProductsOfShape (CategoryTheory.Subobject { un …
    T : C := CategoryTheory.Injective.under (CategoryTheory.Limits.piObj fun P =>  …
    X Y : C
    f : Quiver.Hom X Y
    hf : ∀ (h : Quiver.Hom Y T), Eq (CategoryTheory.CategoryStruct.comp f h) 0
    h : Quiver.Hom G X
    R : CategoryTheory.Subobject { unop := G } := CategoryTheory.Subobject.mk (Cat …
    q₁ : Quiver.Hom (CategoryTheory.Limits.image (CategoryTheory.CategoryStruct.co …
    q₂ : Quiver.Hom (Opposite.unop (CategoryTheory.Subobject.underlying.obj R)) (C …
    ⊢ Eq (CategoryTheory.Limits.factorThruImage (CategoryTheory.CategoryStruct.com …
  -/
  let q : image (h ≫ f) ⟶ T := q₁ ≫ q₂ ≫ Injective.ι _
  exact zero_of_comp_mono q
    (by rw [← Injective.comp_factorThru q (Limits.image.ι (h ≫ f)), Limits.image.fac_assoc,
      Category.assoc, hf, comp_zero])


theorem has_projective_separator [HasColimits C] [EnoughProjectives C] (G : C)
    (hG : IsCoseparator G) : ∃ G : C, Projective G ∧ IsSeparator G := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    inst✝ : CategoryTheory.EnoughProjectives C
    G : C
    hG : CategoryTheory.IsCoseparator G
    ⊢ Exists fun G => And (CategoryTheory.Projective G) (CategoryTheory.IsSeparato …
  -/
  obtain ⟨T, hT₁, hT₂⟩ := has_injective_coseparator (op G) ((isSeparator_op_iff _).2 hG)
  /-
    case intro.intro
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    inst✝ : CategoryTheory.EnoughProjectives C
    G : C
    hG : CategoryTheory.IsCoseparator G
    T : Opposite C
    hT₁ : CategoryTheory.Injective T
    hT₂ : CategoryTheory.IsCoseparator T
    ⊢ Exists fun G => And (CategoryTheory.Projective G) (CategoryTheory.IsSeparato …
  -/
  exact ⟨unop T, inferInstance, (isSeparator_unop_iff _).2 hT₂⟩
  /-
    🎉 no goals
  -/


