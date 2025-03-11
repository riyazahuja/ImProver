variable (X) in
/--
The projection maps of a product to the products indexed by a subset and its complement, as a
binary fan.
-/
noncomputable def Pi.binaryFanOfProp : BinaryFan (∏ᶜ (fun (i : {x : I // P x}) ↦ X i.val))
    (∏ᶜ (fun (i : {x : I // ¬ P x}) ↦ X i.val)) :=
  BinaryFan.mk (P := ∏ᶜ X) (Pi.map' Subtype.val fun _ ↦ 𝟙 _)
    (Pi.map' Subtype.val fun _ ↦ 𝟙 _)


variable (X) in
/--
A product indexed by `I` is a binary product of the products indexed by a subset of `I` and its
complement.
-/
noncomputable def Pi.binaryFanOfPropIsLimit [∀ i, Decidable (P i)] :
    IsLimit (Pi.binaryFanOfProp X P) :=
  BinaryFan.isLimitMk
    (fun s ↦ Pi.lift fun b ↦ if h : P b then
      s.π.app ⟨WalkingPair.left⟩ ≫ Pi.π (fun (i : {x : I // P x}) ↦ X i.val) ⟨b, h⟩ else
      s.π.app ⟨WalkingPair.right⟩ ≫ Pi.π (fun (i : {x : I // ¬ P x}) ↦ X i.val) ⟨b, h⟩)
        /-
          C : Type u_1
          I : Type u_2
          inst✝⁷ : CategoryTheory.Category.{?u.4723, u_1} C
          X Y : I → C
          f : (i : I) → Quiver.Hom (X i) (Y i)
          P : I → Prop
          inst✝⁶ : CategoryTheory.Limits.HasProduct X
          inst✝⁵ : CategoryTheory.Limits.HasProduct Y
          inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝ : (i : I) → Decidable (P i)
          ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan (CategoryTheory.Limits.piObj fun i => …
        -/
        /-
          🎉 no goals
        -/
    (by aesop) (by aesop)
                   /-
                     🎉 no goals
                   -/
    (fun _ _ h₁ h₂ ↦ Pi.hom_ext _ _ fun b ↦ by
      /-
        C : Type u_1
        I : Type u_2
        inst✝⁷ : CategoryTheory.Category.{?u.4723, u_1} C
        X Y : I → C
        f : (i : I) → Quiver.Hom (X i) (Y i)
        P : I → Prop
        inst✝⁶ : CategoryTheory.Limits.HasProduct X
        inst✝⁵ : CategoryTheory.Limits.HasProduct Y
        inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
        inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
        inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
        inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
        inst✝ : (i : I) → Decidable (P i)
        x✝¹ : CategoryTheory.Limits.BinaryFan (CategoryTheory.Limits.piObj fun i => X  …
        x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.piObj X)
        h₁ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
        b : I
        ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.π X b))  …
      -/
      by_cases h : P b
        /-
          case pos
          C : Type u_1
          I : Type u_2
          inst✝⁷ : CategoryTheory.Category.{?u.4723, u_1} C
          X Y : I → C
          f : (i : I) → Quiver.Hom (X i) (Y i)
          P : I → Prop
          inst✝⁶ : CategoryTheory.Limits.HasProduct X
          inst✝⁵ : CategoryTheory.Limits.HasProduct Y
          inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝ : (i : I) → Decidable (P i)
          x✝¹ : CategoryTheory.Limits.BinaryFan (CategoryTheory.Limits.piObj fun i => X  …
          x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.piObj X)
          h₁ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
          b : I
          h : P b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.π X b))  …
        -/
      · simp [← h₁, dif_pos h]
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          I : Type u_2
          inst✝⁷ : CategoryTheory.Category.{?u.4723, u_1} C
          X Y : I → C
          f : (i : I) → Quiver.Hom (X i) (Y i)
          P : I → Prop
          inst✝⁶ : CategoryTheory.Limits.HasProduct X
          inst✝⁵ : CategoryTheory.Limits.HasProduct Y
          inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
          inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
          inst✝ : (i : I) → Decidable (P i)
          x✝¹ : CategoryTheory.Limits.BinaryFan (CategoryTheory.Limits.piObj fun i => X  …
          x✝ : Quiver.Hom x✝¹.pt (CategoryTheory.Limits.piObj X)
          h₁ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.map'  …
          b : I
          h : Not (P b)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Pi.π X b))  …
        -/
      · simp [← h₂, dif_neg h])
        /-
          🎉 no goals
        -/


lemma hasBinaryProduct_of_products : HasBinaryProduct (∏ᶜ (fun (i : {x : I // P x}) ↦ X i.val))
    (∏ᶜ (fun (i : {x : I // ¬ P x}) ↦ X i.val)) := by
  /-
    C : Type u_1
    I : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    X : I → C
    P : I → Prop
    inst✝² : CategoryTheory.Limits.HasProduct X
    inst✝¹ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    ⊢ CategoryTheory.Limits.HasBinaryProduct (CategoryTheory.Limits.piObj fun i => …
  -/
  classical exact ⟨Pi.binaryFanOfProp X P, Pi.binaryFanOfPropIsLimit X P⟩
  /-
    🎉 no goals
  -/


lemma Pi.map_eq_prod_map [∀ i, Decidable (P i)] : Pi.map f =
    ((Pi.binaryFanOfPropIsLimit X P).conePointUniqueUpToIso (prodIsProd _ _)).hom ≫
      prod.map (Pi.map (fun (i : {x : I // P x}) ↦ f i.val))
      (Pi.map (fun (i : {x : I // ¬ P x}) ↦ f i.val)) ≫
        ((Pi.binaryFanOfPropIsLimit Y P).conePointUniqueUpToIso (prodIsProd _ _)).inv := by
  /-
    C : Type u_1
    I : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    X Y : I → C
    f : (i : I) → Quiver.Hom (X i) (Y i)
    P : I → Prop
    inst✝⁶ : CategoryTheory.Limits.HasProduct X
    inst✝⁵ : CategoryTheory.Limits.HasProduct Y
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝ : (i : I) → Decidable (P i)
    ⊢ Eq (CategoryTheory.Limits.Pi.map f) (CategoryTheory.CategoryStruct.comp ((Ca …
  -/
  rw [← Category.assoc, Iso.eq_comp_inv]
  /-
    C : Type u_1
    I : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    X Y : I → C
    f : (i : I) → Quiver.Hom (X i) (Y i)
    P : I → Prop
    inst✝⁶ : CategoryTheory.Limits.HasProduct X
    inst✝⁵ : CategoryTheory.Limits.HasProduct Y
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝ : (i : I) → Decidable (P i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.map f) ((Ca …
  -/
  dsimp only [IsLimit.conePointUniqueUpToIso, binaryFanOfProp, prodIsProd]
  /-
    C : Type u_1
    I : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    X Y : I → C
    f : (i : I) → Quiver.Hom (X i) (Y i)
    P : I → Prop
    inst✝⁶ : CategoryTheory.Limits.HasProduct X
    inst✝⁵ : CategoryTheory.Limits.HasProduct Y
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝ : (i : I) → Decidable (P i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.map f) ((Ca …
  -/
  apply prod.hom_ext
  /-
    case h₁
    C : Type u_1
    I : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    X Y : I → C
    f : (i : I) → Quiver.Hom (X i) (Y i)
    P : I → Prop
    inst✝⁶ : CategoryTheory.Limits.HasProduct X
    inst✝⁵ : CategoryTheory.Limits.HasProduct Y
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝³ : CategoryTheory.Limits.HasProduct fun i => X ↑i
    inst✝² : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝¹ : CategoryTheory.Limits.HasProduct fun i => Y ↑i
    inst✝ : (i : I) → Decidable (P i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  all_goals aesop_cat
  /-
    🎉 no goals
  -/


