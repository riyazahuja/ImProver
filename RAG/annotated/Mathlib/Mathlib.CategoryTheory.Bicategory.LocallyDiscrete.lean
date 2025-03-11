/-- A wrapper for promoting any category to a bicategory,
with the only 2-morphisms being equalities.
-/
@[ext]
structure LocallyDiscrete (C : Type u) where
  /-- A wrapper for promoting any category to a bicategory,
  with the only 2-morphisms being equalities.
  -/
  as : C


@[simp]
theorem mk_as (a : LocallyDiscrete C) : mk a.as = a := rfl


/-- `LocallyDiscrete C` is equivalent to the original type `C`. -/
@[simps]
def locallyDiscreteEquiv : LocallyDiscrete C ≃ C where
  toFun := LocallyDiscrete.as
  invFun := LocallyDiscrete.mk
                 /-
                   C : Type u
                   ⊢ Function.LeftInverse CategoryTheory.LocallyDiscrete.mk CategoryTheory.Locall …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    ⊢ Function.RightInverse CategoryTheory.LocallyDiscrete.mk CategoryTheory.Local …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


instance [DecidableEq C] : DecidableEq (LocallyDiscrete C) :=
  locallyDiscreteEquiv.decidableEq


instance [Inhabited C] : Inhabited (LocallyDiscrete C) :=
  ⟨⟨default⟩⟩


instance categoryStruct [CategoryStruct.{v} C] : CategoryStruct (LocallyDiscrete C) where
  Hom a b := Discrete (a.as ⟶ b.as)
  id a := ⟨𝟙 a.as⟩
  comp f g := ⟨f.as ≫ g.as⟩


@[simp]
lemma id_as (a : LocallyDiscrete C) : (𝟙 a : Discrete (a.as ⟶ a.as)).as = 𝟙 a.as :=
  rfl


@[simp]
lemma comp_as {a b c : LocallyDiscrete C} (f : a ⟶ b) (g : b ⟶ c) : (f ≫ g).as = f.as ≫ g.as :=
  rfl


instance (priority := 900) homSmallCategory (a b : LocallyDiscrete C) : SmallCategory (a ⟶ b) :=
  CategoryTheory.discreteCategory (a.as ⟶ b.as)

-- Porting note: Manually adding this instance (inferInstance doesn't work)

instance subsingleton2Hom {a b : LocallyDiscrete C} (f g : a ⟶ b) : Subsingleton (f ⟶ g) :=
  instSubsingletonDiscreteHom f g


/-- Extract the equation from a 2-morphism in a locally discrete 2-category. -/
theorem eq_of_hom {X Y : LocallyDiscrete C} {f g : X ⟶ Y} (η : f ⟶ g) : f = g :=
  Discrete.ext η.1.1


/-- The locally discrete bicategory on a category is a bicategory in which the objects and the
1-morphisms are the same as those in the underlying category, and the 2-morphisms are the
equalities between 1-morphisms.
-/
instance locallyDiscreteBicategory : Bicategory (LocallyDiscrete C) where
  whiskerLeft _ _ _ η := eqToHom (congr_arg₂ (· ≫ ·) rfl (LocallyDiscrete.eq_of_hom η))
  whiskerRight η _ := eqToHom (congr_arg₂ (· ≫ ·) (LocallyDiscrete.eq_of_hom η) rfl)
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      a✝ b✝ c✝ d✝ : CategoryTheory.LocallyDiscrete C
                                      f : Quiver.Hom a✝ b✝
                                      g : Quiver.Hom b✝ c✝
                                      h : Quiver.Hom c✝ d✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                    -/
  associator f g h := eqToIso <| by apply Discrete.ext; simp
                                                        /-
                                                          🎉 no goals
                                                        -/
                                /-
                                  C : Type u
                                  inst✝ : CategoryTheory.Category.{v, u} C
                                  a✝ b✝ : CategoryTheory.LocallyDiscrete C
                                  f : Quiver.Hom a✝ b✝
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id a✝) …
                                -/
  leftUnitor f := eqToIso <| by apply Discrete.ext; simp
                                                    /-
                                                      🎉 no goals
                                                    -/
                                 /-
                                   C : Type u
                                   inst✝ : CategoryTheory.Category.{v, u} C
                                   a✝ b✝ : CategoryTheory.LocallyDiscrete C
                                   f : Quiver.Hom a✝ b✝
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id b …
                                 -/
  rightUnitor f := eqToIso <| by apply Discrete.ext; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A locally discrete bicategory is strict. -/
instance locallyDiscreteBicategory.strict : Strict (LocallyDiscrete C) where
  id_comp _ := Discrete.ext (Category.id_comp _)
  comp_id _ := Discrete.ext (Category.comp_id _)
  assoc _ _ _ := Discrete.ext (Category.assoc _ _ _)


@[simp]
lemma PrelaxFunctor.map₂_eqToHom (F : PrelaxFunctor B C) {a b : B} {f g : a ⟶ b} (h : f = g) :
    F.map₂ (eqToHom h) = eqToHom (F.congr_map h) := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F : CategoryTheory.PrelaxFunctor B C
    a b : B
    f g : Quiver.Hom a b
    h : Eq f g
    ⊢ Eq (F.map₂ (CategoryTheory.eqToHom h)) (CategoryTheory.eqToHom ⋯)
  -/
  subst h; simp only [eqToHom_refl, PrelaxFunctor.map₂_id]
           /-
             🎉 no goals
           -/


/-- A bicategory is locally discrete if the categories of 1-morphisms are discrete. -/
abbrev IsLocallyDiscrete (B : Type*) [Bicategory B] := ∀ (b c : B), IsDiscrete (b ⟶ c)


instance (C : Type*) [Category C] : IsLocallyDiscrete (LocallyDiscrete C) :=
  fun _ _ ↦ Discrete.isDiscrete _


instance (B : Type*) [Bicategory B] [IsLocallyDiscrete B] : Strict B where
  id_comp f := obj_ext_of_isDiscrete (leftUnitor f).hom
  comp_id f := obj_ext_of_isDiscrete (rightUnitor f).hom
  assoc f g h := obj_ext_of_isDiscrete (associator f g h).hom


/-- The 1-morphism in `LocallyDiscrete C` associated to a given morphism `f : a ⟶ b` in `C` -/
@[simps]
def toLoc {a b : C} (f : a ⟶ b) : LocallyDiscrete.mk a ⟶ LocallyDiscrete.mk b :=
  ⟨f⟩


@[simp]
lemma id_toLoc (a : C) : (𝟙 a).toLoc = 𝟙 (LocallyDiscrete.mk a) :=
  rfl


@[simp]
lemma comp_toLoc {a b c : C} (f : a ⟶ b) (g : b ⟶ c) : (f ≫ g).toLoc = f.toLoc ≫ g.toLoc :=
  rfl


@[simp]
lemma CategoryTheory.LocallyDiscrete.eqToHom_toLoc {C : Type u} [Category.{v} C] {a b : C}
    (h : a = b) : (eqToHom h).toLoc = eqToHom (congrArg LocallyDiscrete.mk h) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    a b : C
    h : Eq a b
    ⊢ Eq (CategoryTheory.eqToHom h).toLoc (CategoryTheory.eqToHom ⋯)
  -/
  subst h; rfl
           /-
             🎉 no goals
           -/


