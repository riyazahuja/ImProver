theorem id_tensor_π_preserves_coequalizer_inv_desc {W X Y Z : C} (f g : X ⟶ Y) (h : Z ⊗ Y ⟶ W)
    (wh : (Z ◁ f) ≫ h = (Z ◁ g) ≫ h) :
    (Z ◁ coequalizer.π f g) ≫
        (PreservesCoequalizer.iso (tensorLeft Z) f g).inv ≫ coequalizer.desc h wh =
      h :=
  map_π_preserves_coequalizer_inv_desc (tensorLeft Z) f g h wh


theorem id_tensor_π_preserves_coequalizer_inv_colimMap_desc {X Y Z X' Y' Z' : C} (f g : X ⟶ Y)
    (f' g' : X' ⟶ Y') (p : Z ⊗ X ⟶ X') (q : Z ⊗ Y ⟶ Y') (wf : (Z ◁ f) ≫ q = p ≫ f')
    (wg : (Z ◁ g) ≫ q = p ≫ g') (h : Y' ⟶ Z') (wh : f' ≫ h = g' ≫ h) :
    (Z ◁ coequalizer.π f g) ≫
        (PreservesCoequalizer.iso (tensorLeft Z) f g).inv ≫
          colimMap (parallelPairHom (Z ◁ f) (Z ◁ g) f' g' p q wf wg) ≫ coequalizer.desc h wh =
      q ≫ h :=
  map_π_preserves_coequalizer_inv_colimMap_desc (tensorLeft Z) f g f' g' p q wf wg h wh


theorem π_tensor_id_preserves_coequalizer_inv_desc {W X Y Z : C} (f g : X ⟶ Y) (h : Y ⊗ Z ⟶ W)
    (wh : (f ▷ Z) ≫ h = (g ▷ Z) ≫ h) :
    (coequalizer.π f g ▷ Z) ≫
        (PreservesCoequalizer.iso (tensorRight Z) f g).inv ≫ coequalizer.desc h wh =
      h :=
  map_π_preserves_coequalizer_inv_desc (tensorRight Z) f g h wh


theorem π_tensor_id_preserves_coequalizer_inv_colimMap_desc {X Y Z X' Y' Z' : C} (f g : X ⟶ Y)
    (f' g' : X' ⟶ Y') (p : X ⊗ Z ⟶ X') (q : Y ⊗ Z ⟶ Y') (wf : (f ▷ Z) ≫ q = p ≫ f')
    (wg : (g ▷ Z) ≫ q = p ≫ g') (h : Y' ⟶ Z') (wh : f' ≫ h = g' ≫ h) :
    (coequalizer.π f g ▷ Z) ≫
        (PreservesCoequalizer.iso (tensorRight Z) f g).inv ≫
          colimMap (parallelPairHom (f ▷ Z) (g ▷ Z) f' g' p q wf wg) ≫ coequalizer.desc h wh =
      q ≫ h :=
  map_π_preserves_coequalizer_inv_colimMap_desc (tensorRight Z) f g f' g' p q wf wg h wh


/-- A bimodule object for a pair of monoid objects, all internal to some monoidal category. -/
structure Bimod (A B : Mon_ C) where
  X : C
  actLeft : A.X ⊗ X ⟶ X
  one_actLeft : (A.one ▷ X) ≫ actLeft = (λ_ X).hom := by aesop_cat
  left_assoc :
    (A.mul ▷ X) ≫ actLeft = (α_ A.X A.X X).hom ≫ (A.X ◁ actLeft) ≫ actLeft := by aesop_cat
  actRight : X ⊗ B.X ⟶ X
  actRight_one : (X ◁ B.one) ≫ actRight = (ρ_ X).hom := by aesop_cat
  right_assoc :
    (X ◁ B.mul) ≫ actRight = (α_ X B.X B.X).inv ≫ (actRight ▷ B.X) ≫ actRight := by
    aesop_cat
  middle_assoc :
    (actLeft ▷ B.X) ≫ actRight = (α_ A.X X B.X).hom ≫ (A.X ◁ actRight) ≫ actLeft := by
    aesop_cat


attribute [reassoc (attr := simp)] Bimod.one_actLeft Bimod.actRight_one Bimod.left_assoc
  Bimod.right_assoc Bimod.middle_assoc


/-- A morphism of bimodule objects. -/
@[ext]
structure Hom (M N : Bimod A B) where
  hom : M.X ⟶ N.X
  left_act_hom : M.actLeft ≫ hom = (A.X ◁ hom) ≫ N.actLeft := by aesop_cat
  right_act_hom : M.actRight ≫ hom = (hom ▷ B.X) ≫ N.actRight := by aesop_cat


attribute [reassoc (attr := simp)] Hom.left_act_hom Hom.right_act_hom


/-- The identity morphism on a bimodule object. -/
@[simps]
def id' (M : Bimod A B) : Hom M M where hom := 𝟙 M.X


instance homInhabited (M : Bimod A B) : Inhabited (Hom M M) :=
  ⟨id' M⟩


/-- Composition of bimodule object morphisms. -/
@[simps]
def comp {M N O : Bimod A B} (f : Hom M N) (g : Hom N O) : Hom M O where hom := f.hom ≫ g.hom


instance : Category (Bimod A B) where
  Hom M N := Hom M N
  id := id'
  comp f g := comp f g


@[ext]
lemma hom_ext {M N : Bimod A B} (f g : M ⟶ N) (h : f.hom = g.hom) : f = g :=
  Hom.ext h


@[simp]
theorem id_hom' (M : Bimod A B) : (𝟙 M : Hom M M).hom = 𝟙 M.X :=
  rfl


@[simp]
theorem comp_hom' {M N K : Bimod A B} (f : M ⟶ N) (g : N ⟶ K) :
    (f ≫ g : Hom M K).hom = f.hom ≫ g.hom :=
  rfl


/-- Construct an isomorphism of bimodules by giving an isomorphism between the underlying objects
and checking compatibility with left and right actions only in the forward direction.
-/
@[simps]
def isoOfIso {X Y : Mon_ C} {P Q : Bimod X Y} (f : P.X ≅ Q.X)
    (f_left_act_hom : P.actLeft ≫ f.hom = (X.X ◁ f.hom) ≫ Q.actLeft)
    (f_right_act_hom : P.actRight ≫ f.hom = (f.hom ▷ Y.X) ≫ Q.actRight) : P ≅ Q where
  hom :=
    { hom := f.hom }
  inv :=
    { hom := f.inv
      left_act_hom := by
        rw [← cancel_mono f.hom, Category.assoc, Category.assoc, Iso.inv_hom_id, Category.comp_id,
          f_left_act_hom, ← Category.assoc, ← MonoidalCategory.whiskerLeft_comp, Iso.inv_hom_id,
          MonoidalCategory.whiskerLeft_id, Category.id_comp]
      right_act_hom := by
        rw [← cancel_mono f.hom, Category.assoc, Category.assoc, Iso.inv_hom_id, Category.comp_id,
          f_right_act_hom, ← Category.assoc, ← comp_whiskerRight, Iso.inv_hom_id,
          MonoidalCategory.id_whiskerRight, Category.id_comp] }
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     inst✝ : CategoryTheory.MonoidalCategory C
                     A B : Mon_ C
                     M : Bimod A B
                     X Y : Mon_ C
                     P Q : Bimod X Y
                     f : CategoryTheory.Iso P.X Q.X
                     f_left_act_hom : Eq (CategoryTheory.CategoryStruct.comp P.actLeft f.hom) (Cate …
                     f_right_act_hom : Eq (CategoryTheory.CategoryStruct.comp P.actRight f.hom) (Ca …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := f.hom, left_act_hom := ⋯, ri …
                   -/
  hom_inv_id := by ext; dsimp; rw [Iso.hom_inv_id]
                               /-
                                 🎉 no goals
                               -/
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     inst✝ : CategoryTheory.MonoidalCategory C
                     A B : Mon_ C
                     M : Bimod A B
                     X Y : Mon_ C
                     P Q : Bimod X Y
                     f : CategoryTheory.Iso P.X Q.X
                     f_left_act_hom : Eq (CategoryTheory.CategoryStruct.comp P.actLeft f.hom) (Cate …
                     f_right_act_hom : Eq (CategoryTheory.CategoryStruct.comp P.actRight f.hom) (Ca …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := f.inv, left_act_hom := ⋯, ri …
                   -/
  inv_hom_id := by ext; dsimp; rw [Iso.inv_hom_id]
                               /-
                                 🎉 no goals
                               -/


/-- A monoid object as a bimodule over itself. -/
@[simps]
def regular : Bimod A A where
  X := A.X
  actLeft := A.mul
  actRight := A.mul


instance : Inhabited (Bimod A A) :=
  ⟨regular A⟩


/-- The forgetful functor from bimodule objects to the ambient category. -/
def forget : Bimod A B ⥤ C where
  obj A := A.X
  map f := f.hom


/-- The underlying object of the tensor product of two bimodules. -/
noncomputable def X : C :=
  coequalizer (P.actRight ▷ Q.X) ((α_ _ _ _).hom ≫ (P.X ◁ Q.actLeft))


/-- Left action for the tensor product of two bimodules. -/
noncomputable def actLeft : R.X ⊗ X P Q ⟶ X P Q :=
  (PreservesCoequalizer.iso (tensorLeft R.X) _ _).inv ≫
    colimMap
      (parallelPairHom _ _ _ _
        ((α_ _ _ _).inv ≫ ((α_ _ _ _).inv ▷ _) ≫ (P.actLeft ▷ S.X ▷ Q.X))
        ((α_ _ _ _).inv ≫ (P.actLeft ▷ Q.X))
        (by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          simp only [Category.assoc]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          slice_lhs 1 2 => rw [associator_inv_naturality_middle]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          slice_rhs 3 4 => rw [← comp_whiskerRight, middle_assoc, comp_whiskerRight]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          monoidal)
          /-
            🎉 no goals
          -/
        (by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          slice_lhs 1 1 => rw [MonoidalCategory.whiskerLeft_comp]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          slice_lhs 2 3 => rw [associator_inv_naturality_right]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          slice_lhs 3 4 => rw [whisker_exchange]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          monoidal))
          /-
            🎉 no goals
          -/


theorem whiskerLeft_π_actLeft :
    (R.X ◁ coequalizer.π _ _) ≫ actLeft P Q =
      (α_ _ _ _).inv ≫ (P.actLeft ▷ Q.X) ≫ coequalizer.π _ _ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [map_π_preserves_coequalizer_inv_colimMap (tensorLeft _)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


theorem one_act_left' : (R.one ▷ _) ≫ actLeft P Q = (λ_ _).hom := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp [X]
  -- Porting note: had to replace `rw` by `erw`
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => erw [whisker_exchange]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [whiskerLeft_π_actLeft]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_inv_naturality_left]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [← comp_whiskerRight, one_actLeft]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [leftUnitor_naturality]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem left_assoc' :
    (R.mul ▷ _) ≫ actLeft P Q = (α_ R.X R.X _).hom ≫ (R.X ◁ actLeft P Q) ≫ actLeft P Q := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp [X]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [whisker_exchange]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [whiskerLeft_π_actLeft]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_inv_naturality_left]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [← comp_whiskerRight, left_assoc, comp_whiskerRight, comp_whiskerRight]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [associator_naturality_right]
  slice_rhs 2 3 =>
    rw [← MonoidalCategory.whiskerLeft_comp, whiskerLeft_π_actLeft,
      MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [whiskerLeft_π_actLeft]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [associator_inv_naturality_middle]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


/-- Right action for the tensor product of two bimodules. -/
noncomputable def actRight : X P Q ⊗ T.X ⟶ X P Q :=
  (PreservesCoequalizer.iso (tensorRight T.X) _ _).inv ≫
    colimMap
      (parallelPairHom _ _ _ _
        ((α_ _ _ _).hom ≫ (α_ _ _ _).hom ≫ (P.X ◁ S.X ◁ Q.actRight) ≫ (α_ _ _ _).inv)
        ((α_ _ _ _).hom ≫ (P.X ◁ Q.actRight))
        (by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          slice_lhs 1 2 => rw [associator_naturality_left]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          slice_lhs 2 3 => rw [← whisker_exchange]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          simp)
          /-
            🎉 no goals
          -/
        (by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          simp only [comp_whiskerRight, whisker_assoc, Category.assoc, Iso.inv_hom_id_assoc]
          slice_lhs 3 4 =>
            rw [← MonoidalCategory.whiskerLeft_comp, middle_assoc,
              MonoidalCategory.whiskerLeft_comp]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            inst✝² : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
            R S T : Mon_ C
            P : Bimod R S
            Q : Bimod S T
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          simp))
          /-
            🎉 no goals
          -/


theorem π_tensor_id_actRight :
    (coequalizer.π _ _ ▷ T.X) ≫ actRight P Q =
      (α_ _ _ _).hom ≫ (P.X ◁ Q.actRight) ≫ coequalizer.π _ _ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  erw [map_π_preserves_coequalizer_inv_colimMap (tensorRight _)]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


theorem actRight_one' : (_ ◁ T.one) ≫ actRight P Q = (ρ_ _).hom := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp [X]
  -- Porting note: had to replace `rw` by `erw`
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 =>erw [← whisker_exchange]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [π_tensor_id_actRight]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_naturality_right]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, actRight_one]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem right_assoc' :
    (_ ◁ T.mul) ≫ actRight P Q =
      (α_ _ T.X T.X).inv ≫ (actRight P Q ▷ T.X) ≫ actRight P Q := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp [X]
  -- Porting note: had to replace some `rw` by `erw`
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [← whisker_exchange]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [π_tensor_id_actRight]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_naturality_right]
  slice_lhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, right_assoc,
    MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [associator_inv_naturality_left]
  slice_rhs 2 3 => rw [← comp_whiskerRight, π_tensor_id_actRight, comp_whiskerRight,
    comp_whiskerRight]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [π_tensor_id_actRight]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    inst✝¹ : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem middle_assoc' :
    (actLeft P Q ▷ T.X) ≫ actRight P Q =
      (α_ R.X _ T.X).hom ≫ (R.X ◁ actRight P Q) ≫ actLeft P Q := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorLeft _ ⋙ tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
  -/
  dsimp [X]
  slice_lhs 1 2 => rw [← comp_whiskerRight, whiskerLeft_π_actLeft, comp_whiskerRight,
    comp_whiskerRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [π_tensor_id_actRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [associator_naturality_left]
  -- Porting note: had to replace `rw` by `erw`
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [associator_naturality_middle]
  slice_rhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, π_tensor_id_actRight,
    MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [whiskerLeft_π_actLeft]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [associator_inv_naturality_right]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [whisker_exchange]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S T : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Tensor product of two bimodule objects as a bimodule object. -/
@[simps]
noncomputable def tensorBimod {X Y Z : Mon_ C} (M : Bimod X Y) (N : Bimod Y Z) : Bimod X Z where
  X := TensorBimod.X M N
  actLeft := TensorBimod.actLeft M N
  actRight := TensorBimod.actRight M N
  one_actLeft := TensorBimod.one_act_left' M N
  actRight_one := TensorBimod.actRight_one' M N
  left_assoc := TensorBimod.left_assoc' M N
  right_assoc := TensorBimod.right_assoc' M N
  middle_assoc := TensorBimod.middle_assoc' M N


/-- Left whiskering for morphisms of bimodule objects. -/
@[simps]
noncomputable def whiskerLeft {X Y Z : Mon_ C} (M : Bimod X Y) {N₁ N₂ : Bimod Y Z} (f : N₁ ⟶ N₂) :
    M.tensorBimod N₁ ⟶ M.tensorBimod N₂ where
  hom :=
    colimMap
      (parallelPairHom _ _ _ _ (_ ◁ f.hom) (_ ◁ f.hom)
            /-
              C : Type u₁
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
              inst✝³ : CategoryTheory.MonoidalCategory C
              A B : Mon_ C
              M✝ : Bimod A B
              inst✝² : CategoryTheory.Limits.HasCoequalizers C
              inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
              inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
              X Y Z : Mon_ C
              M : Bimod X Y
              N₁ N₂ : Bimod Y Z
              f : Quiver.Hom N₁ N₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
            -/
        (by rw [whisker_exchange])
            /-
              🎉 no goals
            -/
        (by
          simp only [Category.assoc, tensor_whiskerLeft, Iso.inv_hom_id_assoc,
            Iso.cancel_iso_hom_left]
          /-
            C : Type u₁
            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
            inst✝³ : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M✝ : Bimod A B
            inst✝² : CategoryTheory.Limits.HasCoequalizers C
            inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            X Y Z : Mon_ C
            M : Bimod X Y
            N₁ N₂ : Bimod Y Z
            f : Quiver.Hom N₁ N₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          slice_lhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, Hom.left_act_hom]
          /-
            C : Type u₁
            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
            inst✝³ : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M✝ : Bimod A B
            inst✝² : CategoryTheory.Limits.HasCoequalizers C
            inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            X Y Z : Mon_ C
            M : Bimod X Y
            N₁ N₂ : Bimod Y Z
            f : Quiver.Hom N₁ N₂
            ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft M.X (CategoryTheory.Ca …
          -/
          simp))
          /-
            🎉 no goals
          -/
  left_act_hom := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.tensorBimod N₁).actLeft (CategoryT …
    -/
    refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 1 2 => rw [TensorBimod.whiskerLeft_π_actLeft]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
    slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, ι_colimMap, parallelPairHom_app_one,
      MonoidalCategory.whiskerLeft_comp]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 2 3 => rw [TensorBimod.whiskerLeft_π_actLeft]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 1 2 => rw [associator_inv_naturality_right]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 2 3 => rw [whisker_exchange]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_act_hom := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.tensorBimod N₁).actRight (Category …
    -/
    refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 1 2 => rw [TensorBimod.π_tensor_id_actRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, Hom.right_act_hom]
    slice_rhs 1 2 =>
      rw [← comp_whiskerRight, ι_colimMap, parallelPairHom_app_one, comp_whiskerRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 2 3 => rw [TensorBimod.π_tensor_id_actRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M✝ : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M : Bimod X Y
      N₁ N₂ : Bimod Y Z
      f : Quiver.Hom N₁ N₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Right whiskering for morphisms of bimodule objects. -/
@[simps]
noncomputable def whiskerRight {X Y Z : Mon_ C} {M₁ M₂ : Bimod X Y} (f : M₁ ⟶ M₂) (N : Bimod Y Z) :
    M₁.tensorBimod N ⟶ M₂.tensorBimod N where
  hom :=
    colimMap
      (parallelPairHom _ _ _ _ (f.hom ▷ _ ▷ _) (f.hom ▷ _)
            /-
              C : Type u₁
              inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
              inst✝³ : CategoryTheory.MonoidalCategory C
              A B : Mon_ C
              M : Bimod A B
              inst✝² : CategoryTheory.Limits.HasCoequalizers C
              inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
              inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
              X Y Z : Mon_ C
              M₁ M₂ : Bimod X Y
              f : Quiver.Hom M₁ M₂
              N : Bimod Y Z
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
            -/
        (by rw [← comp_whiskerRight, Hom.right_act_hom, comp_whiskerRight])
            /-
              🎉 no goals
            -/
        (by
          /-
            C : Type u₁
            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
            inst✝³ : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝² : CategoryTheory.Limits.HasCoequalizers C
            inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            X Y Z : Mon_ C
            M₁ M₂ : Bimod X Y
            f : Quiver.Hom M₁ M₂
            N : Bimod Y Z
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          slice_lhs 2 3 => rw [whisker_exchange]
          /-
            C : Type u₁
            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
            inst✝³ : CategoryTheory.MonoidalCategory C
            A B : Mon_ C
            M : Bimod A B
            inst✝² : CategoryTheory.Limits.HasCoequalizers C
            inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
            inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
            X Y Z : Mon_ C
            M₁ M₂ : Bimod X Y
            f : Quiver.Hom M₁ M₂
            N : Bimod Y Z
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
          -/
          simp))
          /-
            🎉 no goals
          -/
  left_act_hom := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.tensorBimod N).actLeft (CategoryT …
    -/
    refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 1 2 => rw [TensorBimod.whiskerLeft_π_actLeft]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 2 3 => rw [← comp_whiskerRight, Hom.left_act_hom]
    slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, ι_colimMap, parallelPairHom_app_one,
      MonoidalCategory.whiskerLeft_comp]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 2 3 => rw [TensorBimod.whiskerLeft_π_actLeft]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 1 2 => rw [associator_inv_naturality_middle]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_act_hom := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (M₁.tensorBimod N).actRight (Category …
    -/
    refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 1 2 => rw [TensorBimod.π_tensor_id_actRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_lhs 2 3 => rw [whisker_exchange]
    slice_rhs 1 2 => rw [← comp_whiskerRight, ι_colimMap, parallelPairHom_app_one,
      comp_whiskerRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    slice_rhs 2 3 => rw [TensorBimod.π_tensor_id_actRight]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      A B : Mon_ C
      M : Bimod A B
      inst✝² : CategoryTheory.Limits.HasCoequalizers C
      inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
      inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
      X Y Z : Mon_ C
      M₁ M₂ : Bimod X Y
      f : Quiver.Hom M₁ M₂
      N : Bimod Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- An auxiliary morphism for the definition of the underlying morphism of the forward component of
the associator isomorphism. -/
noncomputable def homAux : (P.tensorBimod Q).X ⊗ L.X ⟶ (P.tensorBimod (Q.tensorBimod L)).X :=
  (PreservesCoequalizer.iso (tensorRight L.X) _ _).inv ≫
    coequalizer.desc ((α_ _ _ _).hom ≫ (P.X ◁ coequalizer.π _ _) ≫ coequalizer.π _ _)
      (by
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
        -/
        dsimp; dsimp [TensorBimod.X]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 1 2 => rw [associator_naturality_left]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_lhs 2 3 => rw [← whisker_exchange]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 3 4 => rw [coequalizer.condition]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 2 3 => rw [associator_naturality_right]
        slice_lhs 3 4 =>
          rw [← MonoidalCategory.whiskerLeft_comp,
            TensorBimod.whiskerLeft_π_actLeft, MonoidalCategory.whiskerLeft_comp]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- The underlying morphism of the forward component of the associator isomorphism. -/
noncomputable def hom :
    ((P.tensorBimod Q).tensorBimod L).X ⟶ (P.tensorBimod (Q.tensorBimod L)).X :=
  coequalizer.desc (homAux P Q L)
    (by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp [homAux]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      refine (cancel_epi ((tensorRight _ ⋙ tensorRight _).map (coequalizer.π _ _))).1 ?_
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
      -/
      dsimp [TensorBimod.X]
      slice_lhs 1 2 => rw [← comp_whiskerRight, TensorBimod.π_tensor_id_actRight,
        comp_whiskerRight, comp_whiskerRight]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      slice_lhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_lhs 2 3 => rw [associator_naturality_middle]
      slice_lhs 3 4 =>
        rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.condition,
          MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 1 2 => rw [associator_naturality_left]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 2 3 => rw [← whisker_exchange]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      simp)
      /-
        🎉 no goals
      -/


theorem hom_left_act_hom' :
    ((P.tensorBimod Q).tensorBimod L).actLeft ≫ hom P Q L =
      (R.X ◁ hom P Q L) ≫ (P.tensorBimod (Q.tensorBimod L)).actLeft := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((P.tensorBimod Q).tensorBimod L).act …
  -/
  dsimp; dsimp [hom, homAux]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.TensorBimod.actLeft (P.tensorB …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorLeft_map]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [TensorBimod.whiskerLeft_π_actLeft]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.π_desc,
    MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _ ⋙ tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
  -/
  dsimp; dsimp [TensorBimod.X]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_inv_naturality_middle]
  slice_lhs 2 3 =>
    rw [← comp_whiskerRight, TensorBimod.whiskerLeft_π_actLeft,
      comp_whiskerRight, comp_whiskerRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 4 6 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [associator_naturality_left]
  slice_rhs 1 3 =>
    rw [← MonoidalCategory.whiskerLeft_comp, ← MonoidalCategory.whiskerLeft_comp,
      π_tensor_id_preserves_coequalizer_inv_desc, MonoidalCategory.whiskerLeft_comp,
      MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => erw [TensorBimod.whiskerLeft_π_actLeft P (Q.tensorBimod L)]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => erw [associator_inv_naturality_right]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => erw [whisker_exchange]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem hom_right_act_hom' :
    ((P.tensorBimod Q).tensorBimod L).actRight ≫ hom P Q L =
      (hom P Q L ▷ U.X) ≫ (P.tensorBimod (Q.tensorBimod L)).actRight := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((P.tensorBimod Q).tensorBimod L).act …
  -/
  dsimp; dsimp [hom, homAux]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.TensorBimod.actRight (P.tensor …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorRight_map]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [TensorBimod.π_tensor_id_actRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← comp_whiskerRight, coequalizer.π_desc, comp_whiskerRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _ ⋙ tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
  -/
  dsimp; dsimp [TensorBimod.X]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [associator_naturality_left]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [← whisker_exchange]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [associator_naturality_right]
  slice_rhs 1 3 =>
    rw [← comp_whiskerRight, ← comp_whiskerRight, π_tensor_id_preserves_coequalizer_inv_desc,
      comp_whiskerRight, comp_whiskerRight]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => erw [TensorBimod.π_tensor_id_actRight P (Q.tensorBimod L)]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => erw [associator_naturality_middle]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp
  slice_rhs 3 4 =>
    rw [← MonoidalCategory.whiskerLeft_comp, TensorBimod.π_tensor_id_actRight,
      MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


/-- An auxiliary morphism for the definition of the underlying morphism of the inverse component of
the associator isomorphism. -/
noncomputable def invAux : P.X ⊗ (Q.tensorBimod L).X ⟶ ((P.tensorBimod Q).tensorBimod L).X :=
  (PreservesCoequalizer.iso (tensorLeft P.X) _ _).inv ≫
    coequalizer.desc ((α_ _ _ _).inv ≫ (coequalizer.π _ _ ▷ L.X) ≫ coequalizer.π _ _)
      (by
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
        -/
        dsimp; dsimp [TensorBimod.X]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 1 2 => rw [associator_inv_naturality_middle]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [← Iso.inv_hom_id_assoc (α_ _ _ _) (P.X ◁ Q.actRight), comp_whiskerRight]
        slice_lhs 3 4 =>
          rw [← comp_whiskerRight, Category.assoc, ← TensorBimod.π_tensor_id_actRight,
            comp_whiskerRight]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 4 5 => rw [coequalizer.condition]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 3 4 => rw [associator_naturality_left]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 1 2 => rw [MonoidalCategory.whiskerLeft_comp]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 2 3 => rw [associator_inv_naturality_right]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_rhs 3 4 => rw [whisker_exchange]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          A B : Mon_ C
          M : Bimod A B
          inst✝² : CategoryTheory.Limits.HasCoequalizers C
          inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
          inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
          R S T U : Mon_ C
          P : Bimod R S
          Q : Bimod S T
          L : Bimod T U
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        monoidal)
        /-
          🎉 no goals
        -/


/-- The underlying morphism of the inverse component of the associator isomorphism. -/
noncomputable def inv :
    (P.tensorBimod (Q.tensorBimod L)).X ⟶ ((P.tensorBimod Q).tensorBimod L).X :=
  coequalizer.desc (invAux P Q L)
    (by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      dsimp [invAux]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
      -/
      dsimp [TensorBimod.X]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_lhs 1 2 => rw [whisker_exchange]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      slice_lhs 2 4 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_lhs 1 2 => rw [associator_inv_naturality_left]
      slice_lhs 2 3 =>
        rw [← comp_whiskerRight, coequalizer.condition, comp_whiskerRight, comp_whiskerRight]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 1 2 => rw [associator_naturality_right]
      slice_rhs 2 3 =>
        rw [← MonoidalCategory.whiskerLeft_comp, TensorBimod.whiskerLeft_π_actLeft,
          MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 4 6 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      slice_rhs 3 4 => rw [associator_inv_naturality_middle]
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.MonoidalCategory C
        A B : Mon_ C
        M : Bimod A B
        inst✝² : CategoryTheory.Limits.HasCoequalizers C
        inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
        inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
        R S T U : Mon_ C
        P : Bimod R S
        Q : Bimod S T
        L : Bimod T U
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
      -/
      monoidal)
      /-
        🎉 no goals
      -/


theorem hom_inv_id : hom P Q L ≫ inv P Q L = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.AssociatorBimod.hom P Q L) (Bi …
  -/
  dsimp [hom, homAux, inv, invAux]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.de …
  -/
  apply coequalizer.hom_ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorRight_map]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 4 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [Iso.hom_inv_id_assoc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp only [TensorBimod.X]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [Category.comp_id]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem inv_hom_id : inv P Q L ≫ hom P Q L = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.AssociatorBimod.inv P Q L) (Bi …
  -/
  dsimp [hom, homAux, inv, invAux]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.de …
  -/
  apply coequalizer.hom_ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorLeft_map]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 4 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [Iso.inv_hom_id_assoc]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp only [TensorBimod.X]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [Category.comp_id]
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    R S T U : Mon_ C
    P : Bimod R S
    Q : Bimod S T
    L : Bimod T U
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The underlying morphism of the forward component of the left unitor isomorphism. -/
noncomputable def hom : TensorBimod.X (regular R) P ⟶ P.X :=
                                 /-
                                   C : Type u₁
                                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                   inst✝¹ : CategoryTheory.MonoidalCategory C
                                   A B : Mon_ C
                                   M : Bimod A B
                                   inst✝ : CategoryTheory.Limits.HasCoequalizers C
                                   R S : Mon_ C
                                   P : Bimod R S
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                 -/
  coequalizer.desc P.actLeft (by dsimp; rw [Category.assoc, left_assoc])
                                        /-
                                          🎉 no goals
                                        -/


/-- The underlying morphism of the inverse component of the left unitor isomorphism. -/
noncomputable def inv : P.X ⟶ TensorBimod.X (regular R) P :=
  (λ_ P.X).inv ≫ (R.one ▷ _) ≫ coequalizer.π _ _


theorem hom_inv_id : hom P ≫ inv P = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.LeftUnitorBimod.hom P) (Bimod. …
  -/
  dsimp only [hom, inv, TensorBimod.X]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.de …
  -/
  ext; dsimp
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 1 2 => rw [leftUnitor_inv_naturality]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [whisker_exchange]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 3 => rw [← Iso.inv_hom_id_assoc (α_ R.X R.X P.X) (R.X ◁ P.actLeft)]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 4 6 => rw [← Category.assoc, ← coequalizer.condition]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [associator_inv_naturality_left]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [← comp_whiskerRight, Mon_.one_mul]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [Category.comp_id]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem inv_hom_id : inv P ≫ hom P = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.LeftUnitorBimod.inv P) (Bimod. …
  -/
  dsimp [hom, inv]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [one_actLeft, Iso.inv_hom_id]
  /-
    🎉 no goals
  -/


theorem hom_left_act_hom' :
    ((regular R).tensorBimod P).actLeft ≫ hom P = (R.X ◁ hom P) ≫ P.actLeft := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Bimod.regular R).tensorBimod P).act …
  -/
  dsimp; dsimp [hom, TensorBimod.actLeft, regular]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 4 => rw [id_tensor_π_preserves_coequalizer_inv_colimMap_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [left_assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


theorem hom_right_act_hom' :
    ((regular R).tensorBimod P).actRight ≫ hom P = (hom P ▷ S.X) ≫ P.actRight := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Bimod.regular R).tensorBimod P).act …
  -/
  dsimp; dsimp [hom, TensorBimod.actRight, regular]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 4 => rw [π_tensor_id_preserves_coequalizer_inv_colimMap_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 1 2 => rw [← comp_whiskerRight, coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 1 2 => rw [middle_assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


/-- The underlying morphism of the forward component of the right unitor isomorphism. -/
noncomputable def hom : TensorBimod.X P (regular S) ⟶ P.X :=
                                  /-
                                    C : Type u₁
                                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                    inst✝¹ : CategoryTheory.MonoidalCategory C
                                    A B : Mon_ C
                                    M : Bimod A B
                                    inst✝ : CategoryTheory.Limits.HasCoequalizers C
                                    R S : Mon_ C
                                    P : Bimod R S
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                  -/
  coequalizer.desc P.actRight (by dsimp; rw [Category.assoc, right_assoc, Iso.hom_inv_id_assoc])
                                         /-
                                           🎉 no goals
                                         -/


/-- The underlying morphism of the inverse component of the right unitor isomorphism. -/
noncomputable def inv : P.X ⟶ TensorBimod.X P (regular S) :=
  (ρ_ P.X).inv ≫ (_ ◁ S.one) ≫ coequalizer.π _ _


theorem hom_inv_id : hom P ≫ inv P = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.RightUnitorBimod.hom P) (Bimod …
  -/
  dsimp only [hom, inv, TensorBimod.X]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.de …
  -/
  ext; dsimp
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 1 2 => rw [rightUnitor_inv_naturality]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [← whisker_exchange]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [coequalizer.condition]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [associator_naturality_right]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [← MonoidalCategory.whiskerLeft_comp, Mon_.mul_one]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [Category.comp_id]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem inv_hom_id : inv P ≫ hom P = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.RightUnitorBimod.inv P) (Bimod …
  -/
  dsimp [hom, inv]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [actRight_one, Iso.inv_hom_id]
  /-
    🎉 no goals
  -/


theorem hom_left_act_hom' :
    (P.tensorBimod (regular S)).actLeft ≫ hom P = (R.X ◁ hom P) ≫ P.actLeft := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.tensorBimod (Bimod.regular S)).act …
  -/
  dsimp; dsimp [hom, TensorBimod.actLeft, regular]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 4 => rw [id_tensor_π_preserves_coequalizer_inv_colimMap_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [middle_assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Iso.inv_hom_id_assoc]
  /-
    🎉 no goals
  -/


theorem hom_right_act_hom' :
    (P.tensorBimod (regular S)).actRight ≫ hom P = (hom P ▷ S.X) ≫ P.actRight := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.tensorBimod (Bimod.regular S)).act …
  -/
  dsimp; dsimp [hom, TensorBimod.actRight, regular]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 4 => rw [π_tensor_id_preserves_coequalizer_inv_colimMap_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [right_assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← comp_whiskerRight, coequalizer.π_desc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    R S : Mon_ C
    P : Bimod R S
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/-- The associator as a bimodule isomorphism. -/
noncomputable def associatorBimod {W X Y Z : Mon_ C} (L : Bimod W X) (M : Bimod X Y)
    (N : Bimod Y Z) : (L.tensorBimod M).tensorBimod N ≅ L.tensorBimod (M.tensorBimod N) :=
  isoOfIso
    { hom := AssociatorBimod.hom L M N
      inv := AssociatorBimod.inv L M N
      hom_inv_id := AssociatorBimod.hom_inv_id L M N
      inv_hom_id := AssociatorBimod.inv_hom_id L M N } (AssociatorBimod.hom_left_act_hom' L M N)
    (AssociatorBimod.hom_right_act_hom' L M N)


/-- The left unitor as a bimodule isomorphism. -/
noncomputable def leftUnitorBimod {X Y : Mon_ C} (M : Bimod X Y) : (regular X).tensorBimod M ≅ M :=
  isoOfIso
    { hom := LeftUnitorBimod.hom M
      inv := LeftUnitorBimod.inv M
      hom_inv_id := LeftUnitorBimod.hom_inv_id M
      inv_hom_id := LeftUnitorBimod.inv_hom_id M } (LeftUnitorBimod.hom_left_act_hom' M)
    (LeftUnitorBimod.hom_right_act_hom' M)


/-- The right unitor as a bimodule isomorphism. -/
noncomputable def rightUnitorBimod {X Y : Mon_ C} (M : Bimod X Y) : M.tensorBimod (regular Y) ≅ M :=
  isoOfIso
    { hom := RightUnitorBimod.hom M
      inv := RightUnitorBimod.inv M
      hom_inv_id := RightUnitorBimod.hom_inv_id M
      inv_hom_id := RightUnitorBimod.inv_hom_id M } (RightUnitorBimod.hom_left_act_hom' M)
    (RightUnitorBimod.hom_right_act_hom' M)


theorem whiskerLeft_id_bimod {X Y Z : Mon_ C} {M : Bimod X Y} {N : Bimod Y Z} :
    whiskerLeft M (𝟙 N) = 𝟙 (M.tensorBimod N) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (M.whiskerLeft (CategoryTheory.CategoryStruct.id N)) (CategoryTheory.Cate …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (M.whiskerLeft (CategoryTheory.CategoryStruct.id N)).hom (CategoryTheory. …
  -/
  apply Limits.coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp only [tensorBimod_X, whiskerLeft_hom, id_hom']
  simp only [MonoidalCategory.whiskerLeft_id, ι_colimMap, parallelPair_obj_one,
    parallelPairHom_app_one, Category.id_comp]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.parallelPair (Cat …
  -/
  erw [Category.comp_id]
  /-
    🎉 no goals
  -/


theorem id_whiskerRight_bimod {X Y Z : Mon_ C} {M : Bimod X Y} {N : Bimod Y Z} :
    whiskerRight (𝟙 M) N = 𝟙 (M.tensorBimod N) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (CategoryTheory.CategoryStruct.id M) N) (CategoryTheo …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (CategoryTheory.CategoryStruct.id M) N).hom (Category …
  -/
  apply Limits.coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp only [tensorBimod_X, whiskerRight_hom, id_hom']
  simp only [MonoidalCategory.id_whiskerRight, ι_colimMap, parallelPair_obj_one,
    parallelPairHom_app_one, Category.id_comp]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.parallelPair (Cat …
  -/
  erw [Category.comp_id]
  /-
    🎉 no goals
  -/


theorem whiskerLeft_comp_bimod {X Y Z : Mon_ C} (M : Bimod X Y) {N P Q : Bimod Y Z} (f : N ⟶ P)
    (g : P ⟶ Q) : whiskerLeft M (f ≫ g) = whiskerLeft M f ≫ whiskerLeft M g := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N P Q : Bimod Y Z
    f : Quiver.Hom N P
    g : Quiver.Hom P Q
    ⊢ Eq (M.whiskerLeft (CategoryTheory.CategoryStruct.comp f g)) (CategoryTheory. …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N P Q : Bimod Y Z
    f : Quiver.Hom N P
    g : Quiver.Hom P Q
    ⊢ Eq (M.whiskerLeft (CategoryTheory.CategoryStruct.comp f g)).hom (CategoryThe …
  -/
  apply Limits.coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N P Q : Bimod Y Z
    f : Quiver.Hom N P
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem id_whiskerLeft_bimod {X Y : Mon_ C} {M N : Bimod X Y} (f : M ⟶ N) :
    whiskerLeft (regular X) f = (leftUnitorBimod M).hom ≫ f ≫ (leftUnitorBimod N).inv := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq ((Bimod.regular X).whiskerLeft f) (CategoryTheory.CategoryStruct.comp M.l …
  -/
  dsimp [tensorHom, regular, leftUnitorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq ({ X := X.X, actLeft := X.mul, one_actLeft := ⋯, left_assoc := ⋯, actRigh …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq ({ X := X.X, actLeft := X.mul, one_actLeft := ⋯, left_assoc := ⋯, actRigh …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [LeftUnitorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [LeftUnitorBimod.inv]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [Hom.left_act_hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [leftUnitor_inv_naturality]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 4 => rw [← Iso.inv_hom_id_assoc (α_ X.X X.X N.X) (X.X ◁ N.actLeft)]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 5 7 => rw [← Category.assoc, ← coequalizer.condition]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [associator_inv_naturality_left]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [← comp_whiskerRight, Mon_.one_mul]
  have : (λ_ (X.X ⊗ N.X)).inv ≫ (α_ (𝟙_ C) X.X N.X).inv ≫ ((λ_ X.X).hom ▷ N.X) = 𝟙 _ := by
    monoidal
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 4 => rw [this]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [Category.comp_id]
  /-
    🎉 no goals
  -/


theorem comp_whiskerLeft_bimod {W X Y Z : Mon_ C} (M : Bimod W X) (N : Bimod X Y)
    {P P' : Bimod Y Z} (f : P ⟶ P') :
    whiskerLeft (M.tensorBimod N) f =
      (associatorBimod M N P).hom ≫
        whiskerLeft M (whiskerLeft N f) ≫ (associatorBimod M N P').inv := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq ((M.tensorBimod N).whiskerLeft f) (CategoryTheory.CategoryStruct.comp (M. …
  -/
  dsimp [tensorHom, tensorBimod, associatorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq ({ X := Bimod.TensorBimod.X M N, actLeft := Bimod.TensorBimod.actLeft M N …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq ({ X := Bimod.TensorBimod.X M N, actLeft := Bimod.TensorBimod.actLeft M N …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [TensorBimod.X, AssociatorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.homAux, AssociatorBimod.inv]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorRight_map]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.invAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 2 => rw [MonoidalCategory.whiskerLeft_comp]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 5 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [associator_inv_naturality_right]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [Iso.hom_inv_id_assoc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [← whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N : Bimod X Y
    P P' : Bimod Y Z
    f : Quiver.Hom P P'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem comp_whiskerRight_bimod {X Y Z : Mon_ C} {M N P : Bimod X Y} (f : M ⟶ N) (g : N ⟶ P)
    (Q : Bimod Y Z) : whiskerRight (f ≫ g) Q = whiskerRight f Q ≫ whiskerRight g Q := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N P : Bimod X Y
    f : Quiver.Hom M N
    g : Quiver.Hom N P
    Q : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (CategoryTheory.CategoryStruct.comp f g) Q) (Category …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N P : Bimod X Y
    f : Quiver.Hom M N
    g : Quiver.Hom N P
    Q : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (CategoryTheory.CategoryStruct.comp f g) Q).hom (Cate …
  -/
  apply Limits.coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N P : Bimod X Y
    f : Quiver.Hom M N
    g : Quiver.Hom N P
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem whiskerRight_id_bimod {X Y : Mon_ C} {M N : Bimod X Y} (f : M ⟶ N) :
    whiskerRight f (regular Y) = (rightUnitorBimod M).hom ≫ f ≫ (rightUnitorBimod N).inv := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (Bimod.whiskerRight f (Bimod.regular Y)) (CategoryTheory.CategoryStruct.c …
  -/
  dsimp [tensorHom, regular, rightUnitorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (Bimod.whiskerRight f { X := Y.X, actLeft := Y.mul, one_actLeft := ⋯, lef …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (Bimod.whiskerRight f { X := Y.X, actLeft := Y.mul, one_actLeft := ⋯, lef …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [RightUnitorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [RightUnitorBimod.inv]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [Hom.right_act_hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [rightUnitor_inv_naturality]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [← whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [coequalizer.condition]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [associator_naturality_right]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 4 5 => rw [← MonoidalCategory.whiskerLeft_comp, Mon_.mul_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y : Mon_ C
    M N : Bimod X Y
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem whiskerRight_comp_bimod {W X Y Z : Mon_ C} {M M' : Bimod W X} (f : M ⟶ M') (N : Bimod X Y)
    (P : Bimod Y Z) :
    whiskerRight f (N.tensorBimod P) =
      (associatorBimod M N P).inv ≫
        whiskerRight (whiskerRight f N) P ≫ (associatorBimod M' N P).hom := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight f (N.tensorBimod P)) (CategoryTheory.CategoryStruct.c …
  -/
  dsimp [tensorHom, tensorBimod, associatorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight f { X := Bimod.TensorBimod.X N P, actLeft := Bimod.Te …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight f { X := Bimod.TensorBimod.X N P, actLeft := Bimod.Te …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [TensorBimod.X, AssociatorBimod.inv]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.invAux, AssociatorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorLeft _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorLeft_map]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [← comp_whiskerRight, ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.homAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 2 => rw [comp_whiskerRight]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [associator_naturality_left]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [Iso.inv_hom_id_assoc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M M' : Bimod W X
    f : Quiver.Hom M M'
    N : Bimod X Y
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem whisker_assoc_bimod {W X Y Z : Mon_ C} (M : Bimod W X) {N N' : Bimod X Y} (f : N ⟶ N')
    (P : Bimod Y Z) :
    whiskerRight (whiskerLeft M f) P =
      (associatorBimod M N P).hom ≫
        whiskerLeft M (whiskerRight f P) ≫ (associatorBimod M N' P).inv := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (M.whiskerLeft f) P) (CategoryTheory.CategoryStruct.c …
  -/
  dsimp [tensorHom, tensorBimod, associatorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (M.whiskerLeft f) P) (CategoryTheory.CategoryStruct.c …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (Bimod.whiskerRight (M.whiskerLeft f) P).hom (CategoryTheory.CategoryStru …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.homAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  rw [tensorRight_map]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [← comp_whiskerRight, ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.inv]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.invAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 2 => rw [MonoidalCategory.whiskerLeft_comp]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 5 => rw [id_tensor_π_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [associator_inv_naturality_middle]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 3 => rw [Iso.hom_inv_id_assoc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    W X Y Z : Mon_ C
    M : Bimod W X
    N N' : Bimod X Y
    f : Quiver.Hom N N'
    P : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 1 => rw [comp_whiskerRight]
  /-
    🎉 no goals
  -/


theorem whisker_exchange_bimod {X Y Z : Mon_ C} {M N : Bimod X Y} {P Q : Bimod Y Z} (f : M ⟶ N)
    (g : P ⟶ Q) : whiskerLeft M g ≫ whiskerRight f Q =
      whiskerRight f P ≫ whiskerLeft N g := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.whiskerLeft g) (Bimod.whiskerRight …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.whiskerLeft g) (Bimod.whiskerRight …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 2 3 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M N : Bimod X Y
    P Q : Bimod Y Z
    f : Quiver.Hom M N
    g : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


theorem pentagon_bimod {V W X Y Z : Mon_ C} (M : Bimod V W) (N : Bimod W X) (P : Bimod X Y)
    (Q : Bimod Y Z) :
    whiskerRight (associatorBimod M N P).hom Q ≫
      (associatorBimod M (N.tensorBimod P) Q).hom ≫
        whiskerLeft M (associatorBimod N P Q).hom =
      (associatorBimod (M.tensorBimod N) P Q).hom ≫
        (associatorBimod M N (P.tensorBimod Q)).hom := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.whiskerRight (M.associatorBimo …
  -/
  dsimp [associatorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.whiskerRight (Bimod.isoOfIso { …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.whiskerRight (Bimod.isoOfIso { …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp only [AssociatorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [AssociatorBimod.homAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [← comp_whiskerRight, coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 1 3 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 3 4 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorRight _ ⋙ tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.MonoidalCategory.te …
  -/
  dsimp
  slice_lhs 1 2 =>
    rw [← comp_whiskerRight, π_tensor_id_preserves_coequalizer_inv_desc, comp_whiskerRight,
      comp_whiskerRight]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp only [TensorBimod.X]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [associator_naturality_middle]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 5 6 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 4 5 => rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.π_desc]
  slice_lhs 3 4 =>
    rw [← MonoidalCategory.whiskerLeft_comp, π_tensor_id_preserves_coequalizer_inv_desc,
      MonoidalCategory.whiskerLeft_comp, MonoidalCategory.whiskerLeft_comp]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [associator_naturality_left]
  slice_rhs 2 3 =>
    rw [← whisker_exchange]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 3 5 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 2 3 => rw [associator_naturality_right]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    V W X Y Z : Mon_ C
    M : Bimod V W
    N : Bimod W X
    P : Bimod X Y
    Q : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem triangle_bimod {X Y Z : Mon_ C} (M : Bimod X Y) (N : Bimod Y Z) :
    (associatorBimod M (regular Y) N).hom ≫ whiskerLeft M (leftUnitorBimod N).hom =
      whiskerRight (rightUnitorBimod M).hom N := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.associatorBimod (Bimod.regular Y)  …
  -/
  dsimp [associatorBimod, leftUnitorBimod, rightUnitorBimod]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.isoOfIso { hom := Bimod.Associ …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.isoOfIso { hom := Bimod.Associ …
  -/
  apply coequalizer.hom_ext
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  dsimp [AssociatorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  slice_lhs 1 2 => rw [coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Bimod.AssociatorBimod.homAux M (Bimo …
  -/
  dsimp [AssociatorBimod.homAux]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_rhs 1 2 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [RightUnitorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine (cancel_epi ((tensorRight _).map (coequalizer.π _ _))).1 ?_
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategory.ten …
  -/
  dsimp [regular]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [π_tensor_id_preserves_coequalizer_inv_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [ι_colimMap, parallelPairHom_app_one]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [LeftUnitorBimod.hom]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [← MonoidalCategory.whiskerLeft_comp, coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← comp_whiskerRight, coequalizer.π_desc]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [coequalizer.condition]
  /-
    case h.h
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.MonoidalCategory C
    inst✝² : CategoryTheory.Limits.HasCoequalizers C
    inst✝¹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v …
    inst✝ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfSize.{0, 0, v₁, v₁ …
    X Y Z : Mon_ C
    M : Bimod X Y
    N : Bimod Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


/-- The bicategory of algebras (monoids) and bimodules, all internal to some monoidal category. -/
noncomputable def monBicategory : Bicategory (Mon_ C) where
  Hom X Y := Bimod X Y
  homCategory X Y := (inferInstance : Category (Bimod X Y))
  id X := regular X
  comp M N := tensorBimod M N
  whiskerLeft L _ _ f := whiskerLeft L f
  whiskerRight f N := whiskerRight f N
  associator := associatorBimod
  leftUnitor := leftUnitorBimod
  rightUnitor := rightUnitorBimod
  whiskerLeft_id _ _ := whiskerLeft_id_bimod
  whiskerLeft_comp M _ _ _ f g := whiskerLeft_comp_bimod M f g
  id_whiskerLeft := id_whiskerLeft_bimod
  comp_whiskerLeft M N _ _ f := comp_whiskerLeft_bimod M N f
  id_whiskerRight _ _ := id_whiskerRight_bimod
  comp_whiskerRight f g Q := comp_whiskerRight_bimod f g Q
  whiskerRight_id := whiskerRight_id_bimod
  whiskerRight_comp := whiskerRight_comp_bimod
  whisker_assoc M _ _ f P := whisker_assoc_bimod M f P
  whisker_exchange := whisker_exchange_bimod
  pentagon := pentagon_bimod
  triangle := triangle_bimod


