/-- Bundled left-exact functors. -/
def LeftExactFunctor :=
  FullSubcategory fun F : C ⥤ D => PreservesFiniteLimits F


instance : Category (LeftExactFunctor C D) :=
  FullSubcategory.category _


/-- `C ⥤ₗ D` denotes left exact functors `C ⥤ D` -/
infixr:26 " ⥤ₗ " => LeftExactFunctor


/-- A left exact functor is in particular a functor. -/
def LeftExactFunctor.forget : (C ⥤ₗ D) ⥤ C ⥤ D :=
  fullSubcategoryInclusion _


instance : (LeftExactFunctor.forget C D).Full :=
  FullSubcategory.full _


instance : (LeftExactFunctor.forget C D).Faithful :=
  FullSubcategory.faithful _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Bundled right-exact functors. -/
def RightExactFunctor :=
  FullSubcategory fun F : C ⥤ D => PreservesFiniteColimits F


instance : Category (RightExactFunctor C D) :=
  FullSubcategory.category _


/-- `C ⥤ᵣ D` denotes right exact functors `C ⥤ D` -/
infixr:26 " ⥤ᵣ " => RightExactFunctor


/-- A right exact functor is in particular a functor. -/
def RightExactFunctor.forget : (C ⥤ᵣ D) ⥤ C ⥤ D :=
  fullSubcategoryInclusion _


instance : (RightExactFunctor.forget C D).Full :=
  FullSubcategory.full _


instance : (RightExactFunctor.forget C D).Faithful :=
  FullSubcategory.faithful _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Bundled exact functors. -/
def ExactFunctor :=
  FullSubcategory fun F : C ⥤ D =>
    PreservesFiniteLimits F ∧ PreservesFiniteColimits F


instance : Category (ExactFunctor C D) :=
  FullSubcategory.category _


/-- `C ⥤ₑ D` denotes exact functors `C ⥤ D` -/
infixr:26 " ⥤ₑ " => ExactFunctor


/-- An exact functor is in particular a functor. -/
def ExactFunctor.forget : (C ⥤ₑ D) ⥤ C ⥤ D :=
  fullSubcategoryInclusion _


instance : (ExactFunctor.forget C D).Full :=
  FullSubcategory.full _


instance : (ExactFunctor.forget C D).Faithful :=
  FullSubcategory.faithful _


/-- Turn an exact functor into a left exact functor. -/
def LeftExactFunctor.ofExact : (C ⥤ₑ D) ⥤ C ⥤ₗ D :=
  FullSubcategory.map fun _ => And.left


instance : (LeftExactFunctor.ofExact C D).Full :=
  FullSubcategory.full_map _


instance : (LeftExactFunctor.ofExact C D).Faithful :=
  FullSubcategory.faithful_map _


/-- Turn an exact functor into a left exact functor. -/
def RightExactFunctor.ofExact : (C ⥤ₑ D) ⥤ C ⥤ᵣ D :=
  FullSubcategory.map fun _ => And.right


instance : (RightExactFunctor.ofExact C D).Full :=
  FullSubcategory.full_map _


instance : (RightExactFunctor.ofExact C D).Faithful :=
  FullSubcategory.faithful_map _


@[simp]
theorem LeftExactFunctor.ofExact_obj (F : C ⥤ₑ D) :
    (LeftExactFunctor.ofExact C D).obj F = ⟨F.1, F.2.1⟩ :=
  rfl


@[simp]
theorem RightExactFunctor.ofExact_obj (F : C ⥤ₑ D) :
    (RightExactFunctor.ofExact C D).obj F = ⟨F.1, F.2.2⟩ :=
  rfl


@[simp]
theorem LeftExactFunctor.ofExact_map {F G : C ⥤ₑ D} (α : F ⟶ G) :
    (LeftExactFunctor.ofExact C D).map α = α :=
  rfl


@[simp]
theorem RightExactFunctor.ofExact_map {F G : C ⥤ₑ D} (α : F ⟶ G) :
    (RightExactFunctor.ofExact C D).map α = α :=
  rfl


@[simp]
theorem LeftExactFunctor.forget_obj (F : C ⥤ₗ D) : (LeftExactFunctor.forget C D).obj F = F.1 :=
  rfl


@[simp]
theorem RightExactFunctor.forget_obj (F : C ⥤ᵣ D) : (RightExactFunctor.forget C D).obj F = F.1 :=
  rfl


@[simp]
theorem ExactFunctor.forget_obj (F : C ⥤ₑ D) : (ExactFunctor.forget C D).obj F = F.1 :=
  rfl


@[simp]
theorem LeftExactFunctor.forget_map {F G : C ⥤ₗ D} (α : F ⟶ G) :
    (LeftExactFunctor.forget C D).map α = α :=
  rfl


@[simp]
theorem RightExactFunctor.forget_map {F G : C ⥤ᵣ D} (α : F ⟶ G) :
    (RightExactFunctor.forget C D).map α = α :=
  rfl


@[simp]
theorem ExactFunctor.forget_map {F G : C ⥤ₑ D} (α : F ⟶ G) : (ExactFunctor.forget C D).map α = α :=
  rfl


/-- Turn a left exact functor into an object of the category `LeftExactFunctor C D`. -/
def LeftExactFunctor.of (F : C ⥤ D) [PreservesFiniteLimits F] : C ⥤ₗ D :=
  ⟨F, inferInstance⟩


/-- Turn a right exact functor into an object of the category `RightExactFunctor C D`. -/
def RightExactFunctor.of (F : C ⥤ D) [PreservesFiniteColimits F] : C ⥤ᵣ D :=
  ⟨F, inferInstance⟩


/-- Turn an exact functor into an object of the category `ExactFunctor C D`. -/
def ExactFunctor.of (F : C ⥤ D) [PreservesFiniteLimits F] [PreservesFiniteColimits F] : C ⥤ₑ D :=
  ⟨F, ⟨inferInstance, inferInstance⟩⟩


@[simp]
theorem LeftExactFunctor.of_fst (F : C ⥤ D) [PreservesFiniteLimits F] :
    (LeftExactFunctor.of F).obj = F :=
  rfl


@[simp]
theorem RightExactFunctor.of_fst (F : C ⥤ D) [PreservesFiniteColimits F] :
    (RightExactFunctor.of F).obj = F :=
  rfl


@[simp]
theorem ExactFunctor.of_fst (F : C ⥤ D) [PreservesFiniteLimits F] [PreservesFiniteColimits F] :
    (ExactFunctor.of F).obj = F :=
  rfl


theorem LeftExactFunctor.forget_obj_of (F : C ⥤ D) [PreservesFiniteLimits F] :
    (LeftExactFunctor.forget C D).obj (LeftExactFunctor.of F) = F :=
  rfl


theorem RightExactFunctor.forget_obj_of (F : C ⥤ D) [PreservesFiniteColimits F] :
    (RightExactFunctor.forget C D).obj (RightExactFunctor.of F) = F :=
  rfl


theorem ExactFunctor.forget_obj_of (F : C ⥤ D) [PreservesFiniteLimits F]
    [PreservesFiniteColimits F] : (ExactFunctor.forget C D).obj (ExactFunctor.of F) = F :=
  rfl


noncomputable instance (F : C ⥤ₗ D) : PreservesFiniteLimits F.obj :=
  F.property


noncomputable instance (F : C ⥤ᵣ D) : PreservesFiniteColimits F.obj :=
  F.property


noncomputable instance (F : C ⥤ₑ D) : PreservesFiniteLimits F.obj :=
  F.property.1


noncomputable instance (F : C ⥤ₑ D) : PreservesFiniteColimits F.obj :=
  F.property.2


/-- Whiskering a left exact functor by a left exact functor yields a left exact functor. -/
@[simps!]
def LeftExactFunctor.whiskeringLeft : (C ⥤ₗ D) ⥤ (D ⥤ₗ E) ⥤ (C ⥤ₗ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringLeft C D E).obj F.obj)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.LeftExactFunctor C D
                   G : CategoryTheory.LeftExactFunctor D E
                   ⊢ CategoryTheory.Limits.PreservesFiniteLimits (((CategoryTheory.LeftExactFunct …
                 -/
    (fun G => by dsimp; exact comp_preservesFiniteLimits _ _)
                        /-
                          🎉 no goals
                        -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringLeft C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringLeft C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.LeftExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.LeftExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.LeftExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.LeftExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Whiskering a left exact functor by a left exact functor yields a left exact functor. -/
@[simps!]
def LeftExactFunctor.whiskeringRight : (D ⥤ₗ E) ⥤ (C ⥤ₗ D) ⥤ (C ⥤ₗ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringRight C D E).obj F.obj)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.LeftExactFunctor D E
                   G : CategoryTheory.LeftExactFunctor C D
                   ⊢ CategoryTheory.Limits.PreservesFiniteLimits (((CategoryTheory.LeftExactFunct …
                 -/
    (fun G => by dsimp; exact comp_preservesFiniteLimits _ _)
                        /-
                          🎉 no goals
                        -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringRight C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringRight C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.LeftExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.LeftExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.LeftExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.LeftExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Whiskering a right exact functor by a right exact functor yields a right exact functor. -/
@[simps!]
def RightExactFunctor.whiskeringLeft : (C ⥤ᵣ D) ⥤ (D ⥤ᵣ E) ⥤ (C ⥤ᵣ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringLeft C D E).obj F.obj)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.RightExactFunctor C D
                   G : CategoryTheory.RightExactFunctor D E
                   ⊢ CategoryTheory.Limits.PreservesFiniteColimits (((CategoryTheory.RightExactFu …
                 -/
    (fun G => by dsimp; exact comp_preservesFiniteColimits _ _)
                        /-
                          🎉 no goals
                        -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringLeft C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringLeft C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.RightExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.RightExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.RightExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.RightExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Whiskering a right exact functor by a right exact functor yields a right exact functor. -/
@[simps!]
def RightExactFunctor.whiskeringRight : (D ⥤ᵣ E) ⥤ (C ⥤ᵣ D) ⥤ (C ⥤ᵣ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringRight C D E).obj F.obj)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   F : CategoryTheory.RightExactFunctor D E
                   G : CategoryTheory.RightExactFunctor C D
                   ⊢ CategoryTheory.Limits.PreservesFiniteColimits (((CategoryTheory.RightExactFu …
                 -/
    (fun G => by dsimp; exact comp_preservesFiniteColimits _ _)
                        /-
                          🎉 no goals
                        -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringRight C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringRight C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.RightExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.RightExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.RightExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.RightExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => Category …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Whiskering an exact functor by an exact functor yields an exact functor. -/
@[simps!]
def ExactFunctor.whiskeringLeft : (C ⥤ₑ D) ⥤ (D ⥤ₑ E) ⥤ (C ⥤ₑ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringLeft C D E).obj F.obj)
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    E : Type u₃
                    inst✝ : CategoryTheory.Category.{v₃, u₃} E
                    F : CategoryTheory.ExactFunctor C D
                    G : CategoryTheory.ExactFunctor D E
                    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (((CategoryTheory.ExactFunctor.f …
                  -/
    (fun G => ⟨by dsimp; exact comp_preservesFiniteLimits _ _,
                         /-
                           🎉 no goals
                         -/
         /-
           C : Type u₁
           inst✝² : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
           E : Type u₃
           inst✝ : CategoryTheory.Category.{v₃, u₃} E
           F : CategoryTheory.ExactFunctor C D
           G : CategoryTheory.ExactFunctor D E
           ⊢ CategoryTheory.Limits.PreservesFiniteColimits (((CategoryTheory.ExactFunctor …
         -/
      by dsimp; exact comp_preservesFiniteColimits _ _⟩)
                /-
                  🎉 no goals
                -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringLeft C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringLeft C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.ExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.ExactFunctor C D
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.ExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.ExactFunctor C D
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- Whiskering an exact functor by an exact functor yields an exact functor. -/
@[simps!]
def ExactFunctor.whiskeringRight : (D ⥤ₑ E) ⥤ (C ⥤ₑ D) ⥤ (C ⥤ₑ E) where
  obj F := FullSubcategory.lift _ (forget _ _ ⋙ (CategoryTheory.whiskeringRight C D E).obj F.obj)
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    D : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                    E : Type u₃
                    inst✝ : CategoryTheory.Category.{v₃, u₃} E
                    F : CategoryTheory.ExactFunctor D E
                    G : CategoryTheory.ExactFunctor C D
                    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (((CategoryTheory.ExactFunctor.f …
                  -/
    (fun G => ⟨by dsimp; exact comp_preservesFiniteLimits _ _,
                         /-
                           🎉 no goals
                         -/
         /-
           C : Type u₁
           inst✝² : CategoryTheory.Category.{v₁, u₁} C
           D : Type u₂
           inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
           E : Type u₃
           inst✝ : CategoryTheory.Category.{v₃, u₃} E
           F : CategoryTheory.ExactFunctor D E
           G : CategoryTheory.ExactFunctor C D
           ⊢ CategoryTheory.Limits.PreservesFiniteColimits (((CategoryTheory.ExactFunctor …
         -/
      by dsimp; exact comp_preservesFiniteColimits _ _⟩)
                /-
                  🎉 no goals
                -/
  map {F G} η :=
    { app := fun H => ((CategoryTheory.whiskeringRight C D E).map η).app H.obj
      naturality := fun _ _ f => ((CategoryTheory.whiskeringRight C D E).map η).naturality f }
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.ExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    rw [FullSubcategory.id_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X : CategoryTheory.ExactFunctor D E
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.ExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    rw [FullSubcategory.comp_def]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      X✝ Y✝ Z✝ : CategoryTheory.ExactFunctor D E
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun F => CategoryTheory.FullSubcategory.lift (fun F => And (Cat …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


