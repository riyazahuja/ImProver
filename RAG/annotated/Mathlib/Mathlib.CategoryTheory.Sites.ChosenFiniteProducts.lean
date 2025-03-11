lemma tensorProd_isSheaf : Presheaf.IsSheaf J (X.val ⊗ Y.val) := by
  apply isSheaf_of_isLimit (E := (Cones.postcompose (pairComp X Y (sheafToPresheaf J A)).inv).obj
    (ChosenFiniteProducts.product X.val Y.val).cone)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : CategoryTheory.ChosenFiniteProducts A
    X Y : CategoryTheory.Sheaf J A
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
  -/
  exact (IsLimit.postcomposeInvEquiv _ _).invFun (ChosenFiniteProducts.product X.val Y.val).isLimit
  /-
    🎉 no goals
  -/


lemma tensorUnit_isSheaf : Presheaf.IsSheaf J (𝟙_ (Cᵒᵖ ⥤ A)) := by
  apply isSheaf_of_isLimit (E := (Cones.postcompose (Functor.uniqueFromEmpty _).inv).obj
    ChosenFiniteProducts.terminal.cone)
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      inst✝ : CategoryTheory.ChosenFiniteProducts A
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
    -/
  · exact (IsLimit.postcomposeInvEquiv _ _).invFun ChosenFiniteProducts.terminal.isLimit
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      A : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
      J : CategoryTheory.GrothendieckTopology C
      inst✝ : CategoryTheory.ChosenFiniteProducts A
      ⊢ CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) (CategoryTheory. …
    -/
  · exact Functor.empty _
    /-
      🎉 no goals
    -/


/-- Any `ChosenFiniteProducts` on `A` induce a `ChosenFiniteProducts` structures on `A`-valued
sheaves. -/
@[simps! product_cone_pt_val terminal_cone_pt_val_obj terminal_cone_pt_val_map]
noncomputable instance chosenFiniteProducts : ChosenFiniteProducts (Sheaf J A) where
  product X Y :=
    { cone := BinaryFan.mk
          (P := { val := X.val ⊗ Y.val
                  cond := tensorProd_isSheaf J X Y})
          ⟨(ChosenFiniteProducts.fst _ _)⟩ ⟨(ChosenFiniteProducts.snd _ _)⟩
      isLimit :=
        { lift := fun f ↦ ⟨ChosenFiniteProducts.lift (BinaryFan.fst f).val (BinaryFan.snd f).val⟩
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      A : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                      J : CategoryTheory.GrothendieckTopology C
                      inst✝ : CategoryTheory.ChosenFiniteProducts A
                      X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)) (j : Cat …
                    -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
          fac := by rintro s ⟨⟨j⟩⟩ <;> apply Sheaf.hom_ext <;> simp
                                                               /-
                                                                 🎉 no goals
                                                               -/
          uniq := by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X✝ Y✝ X Y : CategoryTheory.Sheaf J A
              ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)) (m : Qui …
            -/
            intro x f h
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X✝ Y✝ X Y : CategoryTheory.Sheaf J A
              x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
              f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
              h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
              ⊢ Eq f ((fun f => { val := CategoryTheory.ChosenFiniteProducts.lift (CategoryT …
            -/
            apply Sheaf.hom_ext
            /-
              case h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X✝ Y✝ X Y : CategoryTheory.Sheaf J A
              x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
              f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
              h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
              ⊢ Eq f.val ((fun f => { val := CategoryTheory.ChosenFiniteProducts.lift (Categ …
            -/
            apply ChosenFiniteProducts.hom_ext
              /-
                case h.h_fst
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
            · specialize h ⟨WalkingPair.left⟩
              /-
                case h.h_fst
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limits.BinaryFan …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
              rw [Sheaf.hom_ext_iff] at h
              /-
                case h.h_fst
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limits.BinaryFan …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
              simpa using h
              /-
                🎉 no goals
              -/
              /-
                case h.h_snd
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
            · specialize h ⟨WalkingPair.right⟩
              /-
                case h.h_snd
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limits.BinaryFan …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
              rw [Sheaf.hom_ext_iff] at h
              /-
                case h.h_snd
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                A : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                J : CategoryTheory.GrothendieckTopology C
                inst✝ : CategoryTheory.ChosenFiniteProducts A
                X✝ Y✝ X Y : CategoryTheory.Sheaf J A
                x : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
                f : Quiver.Hom x.pt (CategoryTheory.Limits.BinaryFan.mk { val := CategoryTheor …
                h : Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limits.BinaryFan …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f.val (CategoryTheory.ChosenFinitePro …
              -/
              simpa using h } }
              /-
                🎉 no goals
              -/
  terminal :=
    { cone := asEmptyCone { val := 𝟙_ (Cᵒᵖ ⥤ A)
                            cond := tensorUnit_isSheaf _}
      isLimit :=
        { lift := fun f ↦ ⟨ChosenFiniteProducts.toUnit f.pt.val⟩
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      A : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
                      J : CategoryTheory.GrothendieckTopology C
                      inst✝ : CategoryTheory.ChosenFiniteProducts A
                      X Y : CategoryTheory.Sheaf J A
                      ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryThe …
                    -/
          fac := by intro s ⟨e⟩; cases e
                                 /-
                                   🎉 no goals
                                 -/
          uniq := by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X Y : CategoryTheory.Sheaf J A
              ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryThe …
            -/
            intro x f h
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X Y : CategoryTheory.Sheaf J A
              x : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.S …
              f : Quiver.Hom x.pt (CategoryTheory.Limits.asEmptyCone { val := CategoryTheory …
              h : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStr …
              ⊢ Eq f ((fun f => { val := CategoryTheory.ChosenFiniteProducts.toUnit f.pt.val …
            -/
            apply Sheaf.hom_ext
            /-
              case h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              A : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} A
              J : CategoryTheory.GrothendieckTopology C
              inst✝ : CategoryTheory.ChosenFiniteProducts A
              X Y : CategoryTheory.Sheaf J A
              x : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (CategoryTheory.S …
              f : Quiver.Hom x.pt (CategoryTheory.Limits.asEmptyCone { val := CategoryTheory …
              h : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStr …
              ⊢ Eq f.val ((fun f => { val := CategoryTheory.ChosenFiniteProducts.toUnit f.pt …
            -/
            exact ChosenFiniteProducts.toUnit_unique f.val _} }
            /-
              🎉 no goals
            -/


@[simp]
lemma chosenFiniteProducts_fst_val : (ChosenFiniteProducts.fst X Y).val =
    ChosenFiniteProducts.fst X.val Y.val := rfl


@[simp]
lemma chosenFiniteProducts_snd_val : (ChosenFiniteProducts.snd X Y).val =
    ChosenFiniteProducts.snd X.val Y.val := rfl


@[simp]
lemma chosenFiniteProducts_lift_val : (ChosenFiniteProducts.lift f g).val =
    ChosenFiniteProducts.lift f.val g.val := rfl


@[simp]
lemma chosenFiniteProducts_whiskerLeft_val : (X ◁ f).val = (X.val ◁ f.val) := rfl

@[simp]
lemma chosenFiniteProducts_whiskerRight_val : (f ▷ X).val = (f.val ▷ X.val) := rfl


/-- The inclusion from sheaves to presheaves is monoidal with respect to the cartesian monoidal
structures. -/
noncomputable instance sheafToPresheafMonoidal : (sheafToPresheaf J A).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun F G ↦ Iso.refl _ }


@[simp]
lemma sheafToPresheaf_ε : ε (sheafToPresheaf J A) = 𝟙 _ := rfl

@[simp]
lemma sheafToPresheaf_η : η (sheafToPresheaf J A) = 𝟙 _ := rfl


@[simp]
lemma sheafToPresheaf_μ (X Y : Sheaf J A) : μ (sheafToPresheaf J A) X Y = 𝟙 _ := rfl

@[simp]
lemma sheafToPresheaf_δ (X Y : Sheaf J A) : δ (sheafToPresheaf J A) X Y = 𝟙 _ := rfl


