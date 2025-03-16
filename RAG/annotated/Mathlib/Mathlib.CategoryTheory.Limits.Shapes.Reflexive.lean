/-- The pair `f g : A ⟶ B` is reflexive if there is a morphism `B ⟶ A` which is a section for both.
-/
class IsReflexivePair (f g : A ⟶ B) : Prop where
  common_section' : ∃ s : B ⟶ A, s ≫ f = 𝟙 B ∧ s ≫ g = 𝟙 B


theorem IsReflexivePair.common_section (f g : A ⟶ B) [IsReflexivePair f g] :
    ∃ s : B ⟶ A, s ≫ f = 𝟙 B ∧ s ≫ g = 𝟙 B := IsReflexivePair.common_section'


/--
The pair `f g : A ⟶ B` is coreflexive if there is a morphism `B ⟶ A` which is a retraction for both.
-/
class IsCoreflexivePair (f g : A ⟶ B) : Prop where
  common_retraction' : ∃ s : B ⟶ A, f ≫ s = 𝟙 A ∧ g ≫ s = 𝟙 A


theorem IsCoreflexivePair.common_retraction (f g : A ⟶ B) [IsCoreflexivePair f g] :
    ∃ s : B ⟶ A, f ≫ s = 𝟙 A ∧ g ≫ s = 𝟙 A := IsCoreflexivePair.common_retraction'


theorem IsReflexivePair.mk' (s : B ⟶ A) (sf : s ≫ f = 𝟙 B) (sg : s ≫ g = 𝟙 B) :
    IsReflexivePair f g :=
  ⟨⟨s, sf, sg⟩⟩


theorem IsCoreflexivePair.mk' (s : B ⟶ A) (fs : f ≫ s = 𝟙 A) (gs : g ≫ s = 𝟙 A) :
    IsCoreflexivePair f g :=
  ⟨⟨s, fs, gs⟩⟩


/-- Get the common section for a reflexive pair. -/
noncomputable def commonSection (f g : A ⟶ B) [IsReflexivePair f g] : B ⟶ A :=
  (IsReflexivePair.common_section f g).choose


@[reassoc (attr := simp)]
theorem section_comp_left (f g : A ⟶ B) [IsReflexivePair f g] : commonSection f g ≫ f = 𝟙 B :=
  (IsReflexivePair.common_section f g).choose_spec.1


@[reassoc (attr := simp)]
theorem section_comp_right (f g : A ⟶ B) [IsReflexivePair f g] : commonSection f g ≫ g = 𝟙 B :=
  (IsReflexivePair.common_section f g).choose_spec.2


/-- Get the common retraction for a coreflexive pair. -/
noncomputable def commonRetraction (f g : A ⟶ B) [IsCoreflexivePair f g] : B ⟶ A :=
  (IsCoreflexivePair.common_retraction f g).choose


@[reassoc (attr := simp)]
theorem left_comp_retraction (f g : A ⟶ B) [IsCoreflexivePair f g] :
    f ≫ commonRetraction f g = 𝟙 A :=
  (IsCoreflexivePair.common_retraction f g).choose_spec.1


@[reassoc (attr := simp)]
theorem right_comp_retraction (f g : A ⟶ B) [IsCoreflexivePair f g] :
    g ≫ commonRetraction f g = 𝟙 A :=
  (IsCoreflexivePair.common_retraction f g).choose_spec.2


/-- If `f,g` is a kernel pair for some morphism `q`, then it is reflexive. -/
theorem IsKernelPair.isReflexivePair {R : C} {f g : R ⟶ A} {q : A ⟶ B} (h : IsKernelPair q f g) :
    IsReflexivePair f g :=
  IsReflexivePair.mk' _ (h.lift' _ _ rfl).2.1 (h.lift' _ _ _).2.2

-- This shouldn't be an instance as it would instantly loop.

/-- If `f,g` is reflexive, then `g,f` is reflexive. -/
theorem IsReflexivePair.swap [IsReflexivePair f g] : IsReflexivePair g f :=
  IsReflexivePair.mk' _ (section_comp_right f g) (section_comp_left f g)

-- This shouldn't be an instance as it would instantly loop.

/-- If `f,g` is coreflexive, then `g,f` is coreflexive. -/
theorem IsCoreflexivePair.swap [IsCoreflexivePair f g] : IsCoreflexivePair g f :=
  IsCoreflexivePair.mk' _ (right_comp_retraction f g) (left_comp_retraction f g)


/-- For an adjunction `F ⊣ G` with counit `ε`, the pair `(FGε_B, ε_FGB)` is reflexive. -/
instance (B : D) :
    IsReflexivePair (F.map (G.map (adj.counit.app B))) (adj.counit.app (F.obj (G.obj B))) :=
  IsReflexivePair.mk' (F.map (adj.unit.app (G.obj B)))
    (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A B✝ : C
        f g : Quiver.Hom A B✝
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        B : D
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app (G.obj B))) (F.m …
      -/
      rw [← F.map_comp, adj.right_triangle_components]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        A B✝ : C
        f g : Quiver.Hom A B✝
        F : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj : CategoryTheory.Adjunction F G
        B : D
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id (G.obj B))) (CategoryTheory.Cate …
      -/
      apply F.map_id)
      /-
        🎉 no goals
      -/
    (adj.left_triangle_components _)


/-- `C` has reflexive coequalizers if it has coequalizers for every reflexive pair. -/
class HasReflexiveCoequalizers : Prop where
  has_coeq : ∀ ⦃A B : C⦄ (f g : A ⟶ B) [IsReflexivePair f g], HasCoequalizer f g


/-- `C` has coreflexive equalizers if it has equalizers for every coreflexive pair. -/
class HasCoreflexiveEqualizers : Prop where
  has_eq : ∀ ⦃A B : C⦄ (f g : A ⟶ B) [IsCoreflexivePair f g], HasEqualizer f g


theorem hasCoequalizer_of_common_section [HasReflexiveCoequalizers C] {A B : C} {f g : A ⟶ B}
    (r : B ⟶ A) (rf : r ≫ f = 𝟙 _) (rg : r ≫ g = 𝟙 _) : HasCoequalizer f g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasReflexiveCoequalizers C
    A B : C
    f g : Quiver.Hom A B
    r : Quiver.Hom B A
    rf : Eq (CategoryTheory.CategoryStruct.comp r f) (CategoryTheory.CategoryStruc …
    rg : Eq (CategoryTheory.CategoryStruct.comp r g) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.Limits.HasCoequalizer f g
  -/
  letI := IsReflexivePair.mk' r rf rg
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasReflexiveCoequalizers C
    A B : C
    f g : Quiver.Hom A B
    r : Quiver.Hom B A
    rf : Eq (CategoryTheory.CategoryStruct.comp r f) (CategoryTheory.CategoryStruc …
    rg : Eq (CategoryTheory.CategoryStruct.comp r g) (CategoryTheory.CategoryStruc …
    this : CategoryTheory.IsReflexivePair f g := CategoryTheory.IsReflexivePair.mk …
    ⊢ CategoryTheory.Limits.HasCoequalizer f g
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem hasEqualizer_of_common_retraction [HasCoreflexiveEqualizers C] {A B : C} {f g : A ⟶ B}
    (r : B ⟶ A) (fr : f ≫ r = 𝟙 _) (gr : g ≫ r = 𝟙 _) : HasEqualizer f g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    A B : C
    f g : Quiver.Hom A B
    r : Quiver.Hom B A
    fr : Eq (CategoryTheory.CategoryStruct.comp f r) (CategoryTheory.CategoryStruc …
    gr : Eq (CategoryTheory.CategoryStruct.comp g r) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.Limits.HasEqualizer f g
  -/
  letI := IsCoreflexivePair.mk' r fr gr
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasCoreflexiveEqualizers C
    A B : C
    f g : Quiver.Hom A B
    r : Quiver.Hom B A
    fr : Eq (CategoryTheory.CategoryStruct.comp f r) (CategoryTheory.CategoryStruc …
    gr : Eq (CategoryTheory.CategoryStruct.comp g r) (CategoryTheory.CategoryStruc …
    this : CategoryTheory.IsCoreflexivePair f g := CategoryTheory.IsCoreflexivePai …
    ⊢ CategoryTheory.Limits.HasEqualizer f g
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `C` has coequalizers, then it has reflexive coequalizers. -/
instance (priority := 100) hasReflexiveCoequalizers_of_hasCoequalizers [HasCoequalizers C] :
                                                              /-
                                                                C : Type u
                                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                                D : Type u₂
                                                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                A✝ B✝ : C
                                                                f✝ g✝ : Quiver.Hom A✝ B✝
                                                                F : CategoryTheory.Functor C D
                                                                G : CategoryTheory.Functor D C
                                                                adj : CategoryTheory.Adjunction F G
                                                                inst✝ : CategoryTheory.Limits.HasCoequalizers C
                                                                A B : C
                                                                f g : Quiver.Hom A B
                                                                x✝ : CategoryTheory.IsReflexivePair f g
                                                                ⊢ CategoryTheory.Limits.HasCoequalizer f g
                                                              -/
    HasReflexiveCoequalizers C where has_coeq A B f g _ := by infer_instance
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If `C` has equalizers, then it has coreflexive equalizers. -/
instance (priority := 100) hasCoreflexiveEqualizers_of_hasEqualizers [HasEqualizers C] :
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              D : Type u₂
                                                              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                              A✝ B✝ : C
                                                              f✝ g✝ : Quiver.Hom A✝ B✝
                                                              F : CategoryTheory.Functor C D
                                                              G : CategoryTheory.Functor D C
                                                              adj : CategoryTheory.Adjunction F G
                                                              inst✝ : CategoryTheory.Limits.HasEqualizers C
                                                              A B : C
                                                              f g : Quiver.Hom A B
                                                              x✝ : CategoryTheory.IsCoreflexivePair f g
                                                              ⊢ CategoryTheory.Limits.HasEqualizer f g
                                                            -/
    HasCoreflexiveEqualizers C where has_eq A B f g _ := by infer_instance
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The type of objects for the diagram indexing reflexive (co)equalizers -/
inductive WalkingReflexivePair : Type where
  | zero
  | one
  deriving DecidableEq, Inhabited


/-- The type of morphisms for the diagram indexing reflexive (co)equalizers -/
inductive Hom : (WalkingReflexivePair → WalkingReflexivePair → Type)
  | left : Hom one zero
  | right : Hom one zero
  | reflexion : Hom zero one
  | leftCompReflexion : Hom one one
  | rightCompReflexion : Hom one one
  | id (X : WalkingReflexivePair) : Hom X X
  deriving DecidableEq


/-- Composition of morphisms in the diagram indexing reflexive (co)equalizers -/
def Hom.comp :
    ∀ { X Y Z : WalkingReflexivePair } (_ : Hom X Y)
      (_ : Hom Y Z), Hom X Z
  | _, _, _, id _, h => h
  | _, _, _, h, id _ => h
  | _, _, _, reflexion, left => id zero
  | _, _, _, reflexion, right => id zero
  | _, _, _, reflexion, rightCompReflexion => reflexion
  | _, _, _, reflexion, leftCompReflexion => reflexion
  | _, _, _, left, reflexion => leftCompReflexion
  | _, _, _, right, reflexion => rightCompReflexion
  | _, _, _, rightCompReflexion, rightCompReflexion => rightCompReflexion
  | _, _, _, rightCompReflexion, leftCompReflexion => rightCompReflexion
  | _, _, _, rightCompReflexion, right => right
  | _, _, _, rightCompReflexion, left => right
  | _, _, _, leftCompReflexion, left => left
  | _, _, _, leftCompReflexion, right => left
  | _, _, _, leftCompReflexion, rightCompReflexion => leftCompReflexion
  | _, _, _, leftCompReflexion, leftCompReflexion => leftCompReflexion


instance category : SmallCategory WalkingReflexivePair where
  Hom := Hom
  id := Hom.id
  comp := Hom.comp
                /-
                  ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingReflexivePair} (f : Quiver.Hom X Y), E …
                -/
                /-
                  ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingReflexivePair} (f : Quiver.Hom X Y), E …
                -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  comp_id := by intro _ _ f; cases f <;> rfl
                                         /-
                                           🎉 no goals
                                         -/
  id_comp := by intro _ _ f; cases f <;> rfl
              /-
                ⊢ ∀ {W X Y Z : CategoryTheory.Limits.WalkingReflexivePair} (f : Quiver.Hom W X …
              -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  assoc := by intro _ _ _ _ f g h; cases f <;> cases g <;> cases h <;> rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
lemma Hom.id_eq (X : WalkingReflexivePair) :
                         /-
                           X : CategoryTheory.Limits.WalkingReflexivePair
                           ⊢ Eq (CategoryTheory.Limits.WalkingReflexivePair.Hom.id X) (CategoryTheory.Cat …
                         -/
    Hom.id X = 𝟙 X := by rfl
                         /-
                           🎉 no goals
                         -/


@[reassoc (attr := simp)]
lemma reflexion_comp_left : reflexion ≫ left = 𝟙 zero := rfl


@[reassoc (attr := simp)]
lemma reflexion_comp_right : reflexion ≫ right = 𝟙 zero := rfl


@[simp]
lemma leftCompReflexion_eq : leftCompReflexion = (left ≫ reflexion : one ⟶ one) := rfl


@[simp]
lemma rightCompReflexion_eq : rightCompReflexion = (right ≫ reflexion : one ⟶ one) := rfl


@[reassoc (attr := simp)]
lemma map_reflexion_comp_map_left (F : WalkingReflexivePair ⥤ C) :
    F.map reflexion ≫ F.map left = 𝟙 (F.obj zero) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
  -/
  rw [← F.map_comp, reflexion_comp_left, F.map_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma map_reflexion_comp_map_right (F : WalkingReflexivePair ⥤ C) :
    F.map reflexion ≫ F.map right = 𝟙 (F.obj zero) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
  -/
  rw [← F.map_comp, reflexion_comp_right, F.map_id]
  /-
    🎉 no goals
  -/


/-- The inclusion functor forgetting the common section -/
@[simps!]
def inclusionWalkingReflexivePair : WalkingParallelPair ⥤ WalkingReflexivePair where
  obj := fun x => match x with
    | one => WalkingReflexivePair.zero
    | zero => WalkingReflexivePair.one
  map := fun f => match f with
    | .left => WalkingReflexivePair.Hom.left
    | .right => WalkingReflexivePair.Hom.right
    | .id _ => WalkingReflexivePair.Hom.id _
  map_comp := by
    /-
      ⊢ ∀ {X Y Z : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X Y) ( …
    -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
    intro _ _ _ f g; cases f <;> cases g <;> rfl
                                             /-
                                               🎉 no goals
                                             -/


instance (X : WalkingReflexivePair) :
    Nonempty (StructuredArrow X inclusionWalkingReflexivePair) := by
  cases X with
  | zero => exact ⟨StructuredArrow.mk (Y := one) (𝟙 _)⟩
  | one => exact ⟨StructuredArrow.mk (Y := zero) (𝟙 _)⟩


open WalkingReflexivePair.Hom in
instance (X : WalkingReflexivePair) :
    IsConnected (StructuredArrow X inclusionWalkingReflexivePair) := by
  cases X with
  | zero =>
      refine IsConnected.of_induct  (j₀ := StructuredArrow.mk (Y := one) (𝟙 _)) ?_
      rintro p h₁ h₂ ⟨⟨⟨⟩⟩, (_ | _), ⟨_⟩⟩
      · exact (h₂ (StructuredArrow.homMk .left)).2 h₁
      · exact h₁
  | one =>
      refine IsConnected.of_induct  (j₀ := StructuredArrow.mk (Y := zero) (𝟙 _))
        (fun p h₁ h₂ ↦ ?_)
      have hₗ : StructuredArrow.mk left ∈ p := (h₂ (StructuredArrow.homMk .left)).1 h₁
      have hᵣ : StructuredArrow.mk right ∈ p := (h₂ (StructuredArrow.homMk .right)).1 h₁
      rintro ⟨⟨⟨⟩⟩, (_ | _), ⟨_⟩⟩
      · exact (h₂ (StructuredArrow.homMk .left)).2 hₗ
      · exact (h₂ (StructuredArrow.homMk .right)).2 hᵣ
      all_goals assumption


/-- The inclusion functor is a final functor -/
instance inclusionWalkingReflexivePair_final : Functor.Final inclusionWalkingReflexivePair where
  out := inferInstance


/-- Bundle the data of a parallel pair along with a common section as a functor out of the walking
reflexive pair -/
def reflexivePair (f g : A ⟶ B) (s : B ⟶ A)
    (sl : s ≫ f = 𝟙 B := by aesop_cat) (sr : s ≫ g = 𝟙 B := by aesop_cat) :
    (WalkingReflexivePair ⥤ C) where
  obj x :=
    match x with
    | zero => B
    | one => A
  map h :=
    match h with
    | .id _ => 𝟙 _
    | .left => f
    | .right => g
    | .reflexion => s
    | .rightCompReflexion => g ≫ s
    | .leftCompReflexion => f ≫ s
  map_comp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A B : C
      f g : Quiver.Hom A B
      s : Quiver.Hom B A
      sl : autoParam (Eq (CategoryTheory.CategoryStruct.comp s f) (CategoryTheory.Ca …
      sr : autoParam (Eq (CategoryTheory.CategoryStruct.comp s g) (CategoryTheory.Ca …
      ⊢ ∀ {X Y Z : CategoryTheory.Limits.WalkingReflexivePair} (f_1 : Quiver.Hom X Y …
    -/
    rintro _ _ _ ⟨⟩ g <;> cases g <;>
      simp only [Category.id_comp, Category.comp_id, Category.assoc, sl, sr,
                                            /-
                                              case left.reflexion
                                              C : Type u
                                              inst✝ : CategoryTheory.Category.{v, u} C
                                              A B : C
                                              f g : Quiver.Hom A B
                                              s : Quiver.Hom B A
                                              sl : autoParam (Eq (CategoryTheory.CategoryStruct.comp s f) (CategoryTheory.Ca …
                                              sr : autoParam (Eq (CategoryTheory.CategoryStruct.comp s g) (CategoryTheory.Ca …
                                              ⊢ Eq (CategoryTheory.Limits.reflexivePair.match_2 (fun X Y h => Quiver.Hom (Ca …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
        reassoc_of% sl, reassoc_of% sr] <;> rfl
                                            /-
                                              🎉 no goals
                                            -/


@[simp] lemma reflexivePair_obj_zero : (reflexivePair f g s sl sr).obj zero = B := rfl


@[simp] lemma reflexivePair_obj_one : (reflexivePair f g s sl sr).obj one = A := rfl


@[simp] lemma reflexivePair_map_right : (reflexivePair f g s sl sr).map .left = f := rfl


@[simp] lemma reflexivePair_map_left : (reflexivePair f g s sl sr).map .right = g := rfl


@[simp] lemma reflexivePair_map_reflexion : (reflexivePair f g s sl sr).map .reflexion = s := rfl


/-- (Noncomputably) bundle the data of a reflexive pair as a functor out of the walking reflexive
pair -/
noncomputable def ofIsReflexivePair (f g : A ⟶ B) [IsReflexivePair f g] :
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  A B : C
                                  f g : Quiver.Hom A B
                                  inst✝ : CategoryTheory.IsReflexivePair f g
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.commonSection f g) f) …
                                -/
                                /-
                                  🎉 no goals
                                -/
    WalkingReflexivePair ⥤ C := reflexivePair f g (commonSection f g)
                                /-
                                  🎉 no goals
                                -/


@[simp]
lemma ofIsReflexivePair_map_left (f g : A ⟶ B) [IsReflexivePair f g] :
    (ofIsReflexivePair f g).map .left = f := rfl


@[simp]
lemma ofIsReflexivePair_map_right (f g : A ⟶ B) [IsReflexivePair f g] :
    (ofIsReflexivePair f g).map .right = g := rfl


/-- The natural isomorphism between the diagram obtained by forgetting the reflexion of
`ofIsReflexivePair f g` and the original parallel pair. -/
noncomputable def inclusionWalkingReflexivePairOfIsReflexivePairIso
    (f g : A ⟶ B) [IsReflexivePair f g] :
    WalkingParallelPair.inclusionWalkingReflexivePair ⋙ (ofIsReflexivePair f g) ≅
      parallelPair f g :=
  diagramIsoParallelPair _


variable {F G : WalkingReflexivePair ⥤ C}
  (e₀ : F.obj zero ⟶ G.obj zero) (e₁ : F.obj one ⟶ G.obj one)
  (h₁ : F.map left ≫ e₀ = e₁ ≫ G.map left := by aesop_cat)
  (h₂ : F.map right ≫ e₀ = e₁ ≫ G.map right := by aesop_cat)
  (h₃ : F.map reflexion ≫ e₁ = e₀ ≫ G.map reflexion := by aesop_cat)


/-- A constructor for natural transformations between functors from `WalkingReflexivePair`. -/
def mkNatTrans : F ⟶ G where
  app := fun x ↦ match x with
    | zero => e₀
    | one => e₁
  naturality _ _ f := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
      e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
      e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
      h₁ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
      h₂ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
      h₃ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
      x✝¹ x✝ : CategoryTheory.Limits.WalkingReflexivePair
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun x => CategoryTheory.L …
    -/
    cases f
    all_goals
      dsimp
      simp only [Functor.map_id, Category.id_comp, Category.comp_id,
        Functor.map_comp, h₁, h₂, h₃, reassoc_of% h₁, reassoc_of% h₂,
        reflexivePair_map_reflexion, reflexivePair_map_left, reflexivePair_map_right,
        Category.assoc]


@[simp]
lemma mkNatTrans_app_zero : (mkNatTrans e₀ e₁ h₁ h₂ h₃).app zero = e₀ := rfl


@[simp]
lemma mkNatTrans_app_one : (mkNatTrans e₀ e₁ h₁ h₂ h₃).app one = e₁ := rfl


/-- Constructor for natural isomorphisms between functors out of `WalkingReflexivePair`. -/
@[simps!]
def mkNatIso (e₀ : F.obj zero ≅ G.obj zero) (e₁ : F.obj one ≅ G.obj one)
    (h₁ : F.map left ≫ e₀.hom = e₁.hom ≫ G.map left := by aesop_cat)
    (h₂ : F.map right ≫ e₀.hom = e₁.hom ≫ G.map right := by aesop_cat)
    (h₃ : F.map reflexion ≫ e₁.hom = e₀.hom ≫ G.map reflexion := by aesop_cat) :
    F ≅ G where
         /-
           C : Type u
           inst✝ : CategoryTheory.Category.{v, u} C
           F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
           e₀ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero …
           e₁ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) …
           h₁ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
           h₂ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
           h₃ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
  hom := mkNatTrans e₀.hom e₁.hom
         /-
           🎉 no goals
         -/
  inv := mkNatTrans e₀.inv e₁.inv
        (by rw [← cancel_epi e₁.hom, e₁.hom_inv_id_assoc, ← reassoc_of% h₁, e₀.hom_inv_id,
            Category.comp_id])
        (by rw [← cancel_epi e₁.hom, e₁.hom_inv_id_assoc, ← reassoc_of% h₂, e₀.hom_inv_id,
            Category.comp_id])
        (by rw [← cancel_epi e₀.hom, e₀.hom_inv_id_assoc, ← reassoc_of% h₃, e₁.hom_inv_id,
            Category.comp_id])
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                     e₀ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero …
                     e₁ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) …
                     h₁ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     h₂ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     h₃ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.reflexivePair. …
                   -/
                                      /-
                                        🎉 no goals
                                      -/
  hom_inv_id := by ext x; cases x <;> simp
                                      /-
                                        🎉 no goals
                                      -/
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                     e₀ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero …
                     e₁ : CategoryTheory.Iso (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) …
                     h₁ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     h₂ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     h₃ : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.L …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.reflexivePair. …
                   -/
                                      /-
                                        🎉 no goals
                                      -/
  inv_hom_id := by ext x; cases x <;> simp
                                      /-
                                        🎉 no goals
                                      -/


/-- Every functor out of `WalkingReflexivePair` is isomorphic to the `reflexivePair` given by
its components -/
@[simps!]
def diagramIsoReflexivePair :
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
        -/
        /-
          🎉 no goals
        -/
    F ≅ reflexivePair (F.map left) (F.map right) (F.map reflexion) :=
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  mkNatIso (Iso.refl _) (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A `reflexivePair` composed with a functor is isomorphic to the `reflexivePair` obtained by
applying the functor at each map. -/
@[simps!]
def compRightIso {D : Type u₂} [Category.{v₂} D] {A B : C}
    (f g : A ⟶ B) (s : B ⟶ A) (sl : s ≫ f = 𝟙 B) (sr : s ≫ g = 𝟙 B) (F : C ⥤ D) :
    (reflexivePair f g s sl sr) ⋙ F ≅ reflexivePair (F.map f) (F.map g) (F.map s)
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            A B : C
            f g : Quiver.Hom A B
            s : Quiver.Hom B A
            sl : Eq (CategoryTheory.CategoryStruct.comp s f) (CategoryTheory.CategoryStruc …
            sr : Eq (CategoryTheory.CategoryStruct.comp s g) (CategoryTheory.CategoryStruc …
            F : CategoryTheory.Functor C D
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map s) (F.map f)) (CategoryTheory. …
          -/
      (by simp only [← Functor.map_comp, sl, Functor.map_id])
          /-
            🎉 no goals
          -/
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            A B : C
            f g : Quiver.Hom A B
            s : Quiver.Hom B A
            sl : Eq (CategoryTheory.CategoryStruct.comp s f) (CategoryTheory.CategoryStruc …
            sr : Eq (CategoryTheory.CategoryStruct.comp s g) (CategoryTheory.CategoryStruc …
            F : CategoryTheory.Functor C D
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map s) (F.map g)) (CategoryTheory. …
          -/
      (by simp only [← Functor.map_comp, sr, Functor.map_id]) :=
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    A B : C
    f g : Quiver.Hom A B
    s : Quiver.Hom B A
    sl : Eq (CategoryTheory.CategoryStruct.comp s f) (CategoryTheory.CategoryStruc …
    sr : Eq (CategoryTheory.CategoryStruct.comp s g) (CategoryTheory.CategoryStruc …
    F : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Limits.reflexivePai …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  mkNatIso (Iso.refl _) (Iso.refl _)
  /-
    🎉 no goals
  -/


lemma whiskerRightMkNatTrans {F G : WalkingReflexivePair ⥤ C}
    (e₀ : F.obj zero ⟶ G.obj zero) (e₁ : F.obj one ⟶ G.obj one)
    {h₁ : F.map left ≫ e₀ = e₁ ≫ G.map left}
    {h₂ : F.map right ≫ e₀ = e₁ ≫ G.map right}
    {h₃ : F.map reflexion ≫ e₁ = e₀ ≫ G.map reflexion}
    {D : Type u₂} [Category.{v₂} D] (H : C ⥤ D) :
                  /-
                    C : Type u
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                    e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
                    e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
                    h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                    h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                    h₃ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                    D : Type u₂
                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                    H : CategoryTheory.Functor C D
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
    whiskerRight (mkNatTrans e₀ e₁ : F ⟶ G) H =
                  /-
                    🎉 no goals
                  -/
      mkNatTrans (H.map e₀) (H.map e₁)
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
                e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
                h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₃ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                D : Type u₂
                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                H : CategoryTheory.Functor C D
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp H).map CategoryTheory.Limits …
              -/
          (by simp only [Functor.comp_obj, Functor.comp_map, ← Functor.map_comp, h₁])
              /-
                🎉 no goals
              -/
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
                e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
                h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₃ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                D : Type u₂
                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                H : CategoryTheory.Functor C D
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp H).map CategoryTheory.Limits …
              -/
          (by simp only [Functor.comp_obj, Functor.comp_map, ← Functor.map_comp, h₂])
              /-
                🎉 no goals
              -/
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
                e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
                h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                h₃ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
                D : Type u₂
                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                H : CategoryTheory.Functor C D
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp H).map CategoryTheory.Limits …
              -/
          (by simp only [Functor.comp_obj, Functor.comp_map, ← Functor.map_comp, h₃]) := by
              /-
                🎉 no goals
              -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    e₀ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) (G.obj …
    e₁ : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.one) (G.obj  …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walki …
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    H : CategoryTheory.Functor C D
    ⊢ Eq (CategoryTheory.whiskerRight (CategoryTheory.Limits.reflexivePair.mkNatTr …
  -/
                     /-
                       🎉 no goals
                     -/
  ext x; cases x <;> simp
                     /-
                       🎉 no goals
                     -/


/-- Any functor out of the WalkingReflexivePair yields a reflexive pair -/
instance to_isReflexivePair {F : WalkingReflexivePair ⥤ C} :
    IsReflexivePair (F.map .left) (F.map .right) :=
  ⟨F.map .reflexion, map_reflexion_comp_map_left F, map_reflexion_comp_map_right F⟩


/-- A `ReflexiveCofork` is a cocone over a `WalkingReflexivePair`-shaped diagram. -/
abbrev ReflexiveCofork (F : WalkingReflexivePair ⥤ C) := Cocone F


/-- The tail morphism of a reflexive cofork. -/
abbrev π (G : ReflexiveCofork F) : F.obj zero ⟶ G.pt := G.ι.app zero


/-- Constructor for `ReflexiveCofork` -/
@[simps pt]
def mk {X : C} (π : F.obj zero ⟶ X) (h : F.map left ≫ π = F.map right ≫ π) :
    ReflexiveCofork F where
  pt := X
       /-
         C : Type u
         inst✝ : CategoryTheory.Category.{v, u} C
         F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
         X : C
         π : Quiver.Hom (F.obj CategoryTheory.Limits.WalkingReflexivePair.zero) X
         h : Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.Walkin …
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
       -/
       /-
         🎉 no goals
       -/
       /-
         🎉 no goals
       -/
  ι := reflexivePair.mkNatTrans π (F.map left ≫ π)
       /-
         🎉 no goals
       -/


@[simp]
lemma mk_π {X : C} (π : F.obj zero ⟶ X) (h : F.map left ≫ π = F.map right ≫ π) :
    (mk π h).π = π := rfl


lemma condition (G : ReflexiveCofork F) : F.map left ≫ G.π = F.map right ≫ G.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.ReflexiveCofork F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
  -/
  rw [Cocone.w G left, Cocone.w G right]
  /-
    🎉 no goals
  -/


@[simp]
lemma app_one_eq_π (G : ReflexiveCofork F) : G.ι.app zero = G.π := rfl


/-- The underlying `Cofork` of a `ReflexiveCofork`. -/
abbrev toCofork (G : ReflexiveCofork F) : Cofork (F.map left) (F.map right) :=
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
                       G : CategoryTheory.Limits.ReflexiveCofork F
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.WalkingR …
                     -/
  Cofork.ofπ G.π (by simp)
                     /-
                       🎉 no goals
                     -/


/-- Forgetting the reflexion yields an equivalence between cocones over a bundled reflexive pair and
coforks on the underlying parallel pair. -/
@[simps! functor_obj_pt inverse_obj_pt]
def reflexiveCoforkEquivCofork :
    ReflexiveCofork F ≌ Cofork (F.map left) (F.map right) :=
  (Functor.Final.coconesEquiv _ F).symm.trans (Cocones.precomposeEquivalence
    (diagramIsoParallelPair (WalkingParallelPair.inclusionWalkingReflexivePair ⋙ F))).symm


@[simp]
lemma reflexiveCoforkEquivCofork_functor_obj_π (G : ReflexiveCofork F) :
    ((reflexiveCoforkEquivCofork F).functor.obj G).π = G.π := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.ReflexiveCofork F
    ⊢ Eq ((CategoryTheory.Limits.reflexiveCoforkEquivCofork F).functor.obj G).π G.π
  -/
  dsimp [reflexiveCoforkEquivCofork]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.ReflexiveCofork F
    ⊢ Eq ((CategoryTheory.Limits.Cocones.precompose (CategoryTheory.Limits.diagram …
  -/
  rw [ReflexiveCofork.π, Cofork.π]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.ReflexiveCofork F
    ⊢ Eq (((CategoryTheory.Limits.Cocones.precompose (CategoryTheory.Limits.diagra …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
lemma reflexiveCoforkEquivCofork_inverse_obj_π
    (G : Cofork (F.map left) (F.map right)) :
    ((reflexiveCoforkEquivCofork F).inverse.obj G).π = G.π := by
  dsimp only [reflexiveCoforkEquivCofork, Equivalence.symm, Equivalence.trans,
    ReflexiveCofork.π, Cocones.precomposeEquivalence, Cocones.precompose,
    Functor.comp, Functor.Final.coconesEquiv]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.Cofork (F.map CategoryTheory.Limits.WalkingReflexive …
    ⊢ Eq ((CategoryTheory.Functor.Final.extendCocone.obj { pt := G.pt, ι := Catego …
  -/
  rw [Functor.Final.extendCocone_obj_ι_app' (Y := .one) (f := 𝟙 zero)]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    G : CategoryTheory.Limits.Cofork (F.map CategoryTheory.Limits.WalkingReflexive …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The equivalence between reflexive coforks and coforks sends a reflexive cofork to its underlying
cofork. -/
def reflexiveCoforkEquivCoforkObjIso (G : ReflexiveCofork F) :
    (reflexiveCoforkEquivCofork F).functor.obj G ≅ G.toCofork :=
  Cofork.ext (Iso.refl _)
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
          G : CategoryTheory.Limits.ReflexiveCofork F
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.reflexiveCofo …
        -/
    (by simp [reflexiveCoforkEquivCofork, Cofork.π])
        /-
          🎉 no goals
        -/


lemma hasReflexiveCoequalizer_iff_hasCoequalizer :
    HasColimit F ↔ HasCoequalizer (F.map left) (F.map right) := by
  simpa only [hasColimit_iff_hasInitial_cocone]
    using Equivalence.hasInitial_iff (reflexiveCoforkEquivCofork F)


instance reflexivePair_hasColimit_of_hasCoequalizer
    [h : HasCoequalizer (F.map left) (F.map right)] : HasColimit F :=
  hasReflexiveCoequalizer_iff_hasCoequalizer _|>.mpr h


/-- A reflexive cofork is a colimit cocone if and only if the underlying cofork is. -/
def ReflexiveCofork.isColimitEquiv (G : ReflexiveCofork F) :
    IsColimit (G.toCofork) ≃ IsColimit G :=
  IsColimit.equivIsoColimit (reflexiveCoforkEquivCoforkObjIso F G).symm|>.trans <|
    (IsColimit.precomposeHomEquiv (diagramIsoParallelPair _).symm (G.whisker _)).trans <|
      Functor.Final.isColimitWhiskerEquiv _ _


/-- The colimit of a functor out of the walking reflexive pair is the same as the colimit of the
underlying parallel pair. -/
def reflexiveCoequalizerIsoCoequalizer :
    colimit F ≅ coequalizer (F.map left) (F.map right) :=
  ((ReflexiveCofork.isColimitEquiv _ _).symm (colimit.isColimit F)).coconePointUniqueUpToIso
    (colimit.isColimit _)


@[reassoc (attr := simp)]
lemma ι_reflexiveCoequalizerIsoCoequalizer_hom :
    colimit.ι F zero ≫ (reflexiveCoequalizerIsoCoequalizer F).hom =
      coequalizer.π (F.map left) (F.map right) :=
  IsColimit.comp_coconePointUniqueUpToIso_hom
    ((ReflexiveCofork.isColimitEquiv F _).symm _) _ WalkingParallelPair.one


@[reassoc (attr := simp)]
lemma π_reflexiveCoequalizerIsoCoequalizer_inv :
    coequalizer.π _ _ ≫ (reflexiveCoequalizerIsoCoequalizer F).inv = colimit.ι F _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor CategoryTheory.Limits.WalkingReflexivePair C
    inst✝ : CategoryTheory.Limits.HasCoequalizer (F.map CategoryTheory.Limits.Walk …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  rw [reflexiveCoequalizerIsoCoequalizer]
  simp only [colimit.comp_coconePointUniqueUpToIso_inv, Cofork.ofπ_pt, colimit.cocone_x,
    Cofork.ofπ_ι_app, colimit.cocone_ι]


instance ofIsReflexivePair_hasColimit_of_hasCoequalizer :
    HasColimit (ofIsReflexivePair f g) :=
  hasReflexiveCoequalizer_iff_hasCoequalizer _|>.mpr h


/-- The coequalizer of a reflexive pair can be promoted to the colimit of a diagram out of the
walking reflexive pair -/
def colimitOfIsReflexivePairIsoCoequalizer :
    colimit (ofIsReflexivePair f g) ≅ coequalizer f g :=
  @reflexiveCoequalizerIsoCoequalizer _ _ (ofIsReflexivePair f g) h



@[reassoc (attr := simp)]
lemma ι_colimitOfIsReflexivePairIsoCoequalizer_hom :
    colimit.ι (ofIsReflexivePair f g) zero ≫ colimitOfIsReflexivePairIsoCoequalizer.hom =
      coequalizer.π f g := @ι_reflexiveCoequalizerIsoCoequalizer_hom _ _ _ h


@[reassoc (attr := simp)]
lemma π_colimitOfIsReflexivePairIsoCoequalizer_inv :
    coequalizer.π f g ≫ colimitOfIsReflexivePairIsoCoequalizer.inv =
      colimit.ι (ofIsReflexivePair f g) zero :=
  @π_reflexiveCoequalizerIsoCoequalizer_inv _ _ (ofIsReflexivePair f g) h


/-- A category has coequalizers of reflexive pairs if and only if it has all colimits indexed by the
walking reflexive pair. -/
theorem hasReflexiveCoequalizers_iff :
    HasColimitsOfShape WalkingReflexivePair C ↔ HasReflexiveCoequalizers C :=
  ⟨fun _ ↦ ⟨fun _ _ f g _ ↦ (hasReflexiveCoequalizer_iff_hasCoequalizer
       /-
         C : Type u
         inst✝ : CategoryTheory.Category.{v, u} C
         x✝³ : CategoryTheory.Limits.HasColimitsOfShape CategoryTheory.Limits.WalkingRe …
         x✝² x✝¹ : C
         f g : Quiver.Hom x✝² x✝¹
         x✝ : CategoryTheory.IsReflexivePair f g
         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.commonSection f g) f) …
       -/
       /-
         🎉 no goals
       -/
      (reflexivePair f g (commonSection f g))).1 inferInstance⟩,
       /-
         🎉 no goals
       -/
    fun _ ↦ ⟨inferInstance⟩⟩


