/-- A monoid object internal to a monoidal category.

When the monoidal category is preadditive, this is also sometimes called an "algebra object".
-/
class Mon_Class (X : C) where
  /-- The unit morphism of a monoid object. -/
  one : 𝟙_ C ⟶ X
  /-- The multiplication morphism of a monoid object. -/
  mul : X ⊗ X ⟶ X
  /- For the names of the conditions below, the unprimed names are reserved for the version where
  the argument `X` is explicit. -/
  one_mul' : one ▷ X ≫ mul = (λ_ X).hom := by aesop_cat
  mul_one' : X ◁ one ≫ mul = (ρ_ X).hom := by aesop_cat
  -- Obviously there is some flexibility stating this axiom.
  -- This one has left- and right-hand sides matching the statement of `Monoid.mul_assoc`,
  -- and chooses to place the associator on the right-hand side.
  -- The heuristic is that unitors and associators "don't have much weight".
  mul_assoc' : (mul ▷ X) ≫ mul = (α_ X X X).hom ≫ (X ◁ mul) ≫ mul := by aesop_cat


@[inherit_doc] scoped notation "μ" => Mon_Class.mul

@[inherit_doc] scoped notation "μ["M"]" => Mon_Class.mul (X := M)

@[inherit_doc] scoped notation "η" => Mon_Class.one

@[inherit_doc] scoped notation "η["M"]" => Mon_Class.one (X := M)

/- The simp attribute is reserved for the unprimed versions. -/

attribute [reassoc] one_mul' mul_one' mul_assoc'


@[reassoc (attr := simp)]
theorem one_mul (X : C) [Mon_Class X] : η ▷ X ≫ μ = (λ_ X).hom := one_mul'


@[reassoc (attr := simp)]
theorem mul_one (X : C) [Mon_Class X] : X ◁ η ≫ μ = (ρ_ X).hom := mul_one'


@[reassoc (attr := simp)]
theorem mul_assoc (X : C) [Mon_Class X] : μ ▷ X ≫ μ = (α_ X X X).hom ≫ X ◁ μ ≫ μ := mul_assoc'


/-- The property that a morphism between monoid objects is a monoid morphism. -/
class IsMon_Hom (f : M ⟶ N) : Prop where
  one_hom : η ≫ f = η := by aesop_cat
  mul_hom : μ ≫ f = (f ⊗ f) ≫ μ := by aesop_cat


attribute [reassoc (attr := simp)] IsMon_Hom.one_hom IsMon_Hom.mul_hom


/-- A monoid object internal to a monoidal category.

When the monoidal category is preadditive, this is also sometimes called an "algebra object".
-/
structure Mon_ where
  X : C
  one : 𝟙_ C ⟶ X
  mul : X ⊗ X ⟶ X
  one_mul : (one ▷ X) ≫ mul = (λ_ X).hom := by aesop_cat
  mul_one : (X ◁ one) ≫ mul = (ρ_ X).hom := by aesop_cat
  -- Obviously there is some flexibility stating this axiom.
  -- This one has left- and right-hand sides matching the statement of `Monoid.mul_assoc`,
  -- and chooses to place the associator on the right-hand side.
  -- The heuristic is that unitors and associators "don't have much weight".
  mul_assoc : (mul ▷ X) ≫ mul = (α_ X X X).hom ≫ (X ◁ mul) ≫ mul := by aesop_cat


attribute [reassoc] Mon_.one_mul Mon_.mul_one


attribute [reassoc (attr := simp)] Mon_.mul_assoc


/-- Construct an object of `Mon_ C` from an object `X : C` and `Mon_Class X` instance. -/
@[simps]
def mk' (X : C) [Mon_Class X] : Mon_ C where
  X := X
  one := η
  mul := μ


instance {M : Mon_ C} : Mon_Class M.X where
  one := M.one
  mul := M.mul
  one_mul' := M.one_mul
  mul_one' := M.mul_one
  mul_assoc' := M.mul_assoc


/-- The trivial monoid object. We later show this is initial in `Mon_ C`.
-/
@[simps]
def trivial : Mon_ C where
  X := 𝟙_ C
  one := 𝟙 _
  mul := (λ_ _).hom
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.MonoidalCategory C
                    M N : C
                    inst✝¹ : Mon_Class M
                    inst✝ : Mon_Class N
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                  -/
                /-
                  C : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝² : CategoryTheory.MonoidalCategory C
                  M N : C
                  inst✝¹ : Mon_Class M
                  inst✝ : Mon_Class N
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                -/
  mul_assoc := by monoidal_coherence
                /-
                  🎉 no goals
                -/
                  /-
                    🎉 no goals
                  -/
  mul_one := by monoidal_coherence


instance : Inhabited (Mon_ C) :=
  ⟨trivial C⟩


@[simp]
theorem one_mul_hom {Z : C} (f : Z ⟶ M.X) : (M.one ⊗ f) ≫ M.mul = (λ_ Z).hom ≫ f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Mon_ C
    Z : C
    f : Quiver.Hom Z M.X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [tensorHom_def'_assoc, M.one_mul, leftUnitor_naturality]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_one_hom {Z : C} (f : Z ⟶ M.X) : (f ⊗ M.one) ≫ M.mul = (ρ_ Z).hom ≫ f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Mon_ C
    Z : C
    f : Quiver.Hom Z M.X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [tensorHom_def_assoc, M.mul_one, rightUnitor_naturality]
  /-
    🎉 no goals
  -/


theorem mul_assoc_flip :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝ : CategoryTheory.MonoidalCategory C
                                                                                 M : Mon_ C
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                                               -/
    (M.X ◁ M.mul) ≫ M.mul = (α_ M.X M.X M.X).inv ≫ (M.mul ▷ M.X) ≫ M.mul := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A morphism of monoid objects. -/
@[ext]
structure Hom (M N : Mon_ C) where
  hom : M.X ⟶ N.X
  one_hom : M.one ≫ hom = N.one := by aesop_cat
  mul_hom : M.mul ≫ hom = (hom ⊗ hom) ≫ N.mul := by aesop_cat


attribute [reassoc (attr := simp)] Hom.one_hom Hom.mul_hom


/-- The identity morphism on a monoid object. -/
@[simps]
def id (M : Mon_ C) : Hom M M where
  hom := 𝟙 M.X


instance homInhabited (M : Mon_ C) : Inhabited (Hom M M) :=
  ⟨id M⟩


/-- Composition of morphisms of monoid objects. -/
@[simps]
def comp {M N O : Mon_ C} (f : Hom M N) (g : Hom N O) : Hom M O where
  hom := f.hom ≫ g.hom


instance : Category (Mon_ C) where
  Hom M N := Hom M N
  id := id
  comp f g := comp f g


@[ext]
lemma ext {X Y : Mon_ C} {f g : X ⟶ Y} (w : f.hom = g.hom) : f = g :=
  Hom.ext w


@[simp]
theorem id_hom' (M : Mon_ C) : (𝟙 M : Hom M M).hom = 𝟙 M.X :=
  rfl


@[simp]
theorem comp_hom' {M N K : Mon_ C} (f : M ⟶ N) (g : N ⟶ K) :
    (f ≫ g : Hom M K).hom = f.hom ≫ g.hom :=
  rfl


/-- The forgetful functor from monoid objects to the ambient category. -/
@[simps]
def forget : Mon_ C ⥤ C where
  obj A := A.X
  map f := f.hom


instance forget_faithful : (forget C).Faithful where


instance {A B : Mon_ C} (f : A ⟶ B) [e : IsIso ((forget C).map f)] : IsIso f.hom :=
  e


/-- The forgetful functor from monoid objects to the ambient category reflects isomorphisms. -/
instance : (forget C).ReflectsIsomorphisms where
                                             /-
                                               C : Type u₁
                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                               inst✝² : CategoryTheory.MonoidalCategory C
                                               M✝ N : C
                                               inst✝¹ : Mon_Class M✝
                                               inst✝ : Mon_Class N
                                               M A✝ B✝ : Mon_ C
                                               f : Quiver.Hom A✝ B✝
                                               e : CategoryTheory.IsIso ((Mon_.forget C).map f)
                                               ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f { hom := CategoryTheory.inv f. …
                                             -/
  reflects f e := ⟨⟨{ hom := inv f.hom }, by aesop_cat⟩⟩
                                             /-
                                               🎉 no goals
                                             -/


/-- Construct an isomorphism of monoids by giving an isomorphism between the underlying objects
and checking compatibility with unit and multiplication only in the forward direction.
-/
@[simps]
def mkIso {M N : Mon_ C} (f : M.X ≅ N.X) (one_f : M.one ≫ f.hom = N.one := by aesop_cat)
    (mul_f : M.mul ≫ f.hom = (f.hom ⊗ f.hom) ≫ N.mul := by aesop_cat) : M ≅ N where
  hom := { hom := f.hom }
  inv :=
  { hom := f.inv
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.MonoidalCategory C
                    M✝¹ N✝ : C
                    inst✝¹ : Mon_Class M✝¹
                    inst✝ : Mon_Class N✝
                    M✝ M N : Mon_ C
                    f : CategoryTheory.Iso M.X N.X
                    one_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.one f.hom) N.one)  …
                    mul_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.mul f.hom) (Catego …
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp N.one f.inv) M.one
                  -/
    one_hom := by rw [← one_f]; simp
                                /-
                                  🎉 no goals
                                -/
    mul_hom := by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        M✝¹ N✝ : C
        inst✝¹ : Mon_Class M✝¹
        inst✝ : Mon_Class N✝
        M✝ M N : Mon_ C
        f : CategoryTheory.Iso M.X N.X
        one_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.one f.hom) N.one)  …
        mul_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.mul f.hom) (Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp N.mul f.inv) (CategoryTheory.Category …
      -/
      rw [← cancel_mono f.hom]
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        M✝¹ N✝ : C
        inst✝¹ : Mon_Class M✝¹
        inst✝ : Mon_Class N✝
        M✝ M N : Mon_ C
        f : CategoryTheory.Iso M.X N.X
        one_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.one f.hom) N.one)  …
        mul_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.mul f.hom) (Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp N …
      -/
      slice_rhs 2 3 => rw [mul_f]
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        inst✝² : CategoryTheory.MonoidalCategory C
        M✝¹ N✝ : C
        inst✝¹ : Mon_Class M✝¹
        inst✝ : Mon_Class N✝
        M✝ M N : Mon_ C
        f : CategoryTheory.Iso M.X N.X
        one_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.one f.hom) N.one)  …
        mul_f : autoParam (Eq (CategoryTheory.CategoryStruct.comp M.mul f.hom) (Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp N …
      -/
      simp }
      /-
        🎉 no goals
      -/


@[simps]
instance uniqueHomFromTrivial (A : Mon_ C) : Unique (trivial C ⟶ A) where
  default :=
  { hom := A.one
                  /-
                    C : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝² : CategoryTheory.MonoidalCategory C
                    M✝ N : C
                    inst✝¹ : Mon_Class M✝
                    inst✝ : Mon_Class N
                    M A : Mon_ C
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Mon_.trivial C).mul A.one) (Category …
                  -/
    mul_hom := by simp [A.one_mul, unitors_equal] }
                  /-
                    🎉 no goals
                  -/
  uniq f := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Mon_Class M✝
      inst✝ : Mon_Class N
      M A : Mon_ C
      f : Quiver.Hom (Mon_.trivial C) A
      ⊢ Eq f Inhabited.default
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Mon_Class M✝
      inst✝ : Mon_Class N
      M A : Mon_ C
      f : Quiver.Hom (Mon_.trivial C) A
      ⊢ Eq f.hom Inhabited.default.hom
    -/
    simp only [trivial_X]
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Mon_Class M✝
      inst✝ : Mon_Class N
      M A : Mon_ C
      f : Quiver.Hom (Mon_.trivial C) A
      ⊢ Eq f.hom A.one
    -/
    rw [← Category.id_comp f.hom]
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Mon_Class M✝
      inst✝ : Mon_Class N
      M A : Mon_ C
      f : Quiver.Hom (Mon_.trivial C) A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Mo …
    -/
    erw [f.one_hom]
    /-
      🎉 no goals
    -/


instance : HasInitial (Mon_ C) :=
  hasInitial_of_unique (trivial C)


set_option maxHeartbeats 400000 in
-- TODO: mapMod F A : Mod A ⥤ Mod (F.mapMon A)
/-- A lax monoidal functor takes monoid objects to monoid objects.

That is, a lax monoidal functor `F : C ⥤ D` induces a functor `Mon_ C ⥤ Mon_ D`.
-/
@[simps]
def mapMon (F : C ⥤ D) [F.LaxMonoidal] : Mon_ C ⥤ Mon_ D where
  obj A :=
    { X := F.obj A.X
      one := ε F ≫ F.map A.one
      mul := «μ» F _ _ ≫ F.map A.mul
      one_mul := by
        simp_rw [comp_whiskerRight, Category.assoc, μ_natural_left_assoc,
          LaxMonoidal.left_unitality]
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Mon_Class M
          inst✝³ : Mon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.LaxMonoidal
          A : Mon_ C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 3 4 => rw [← F.map_comp, A.one_mul]
        /-
          🎉 no goals
        -/
      mul_one := by
        simp_rw [MonoidalCategory.whiskerLeft_comp, Category.assoc, μ_natural_right_assoc,
          LaxMonoidal.right_unitality]
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Mon_Class M
          inst✝³ : Mon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.LaxMonoidal
          A : Mon_ C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 3 4 => rw [← F.map_comp, A.mul_one]
        /-
          🎉 no goals
        -/
      mul_assoc := by
        simp_rw [comp_whiskerRight, Category.assoc, μ_natural_left_assoc,
          MonoidalCategory.whiskerLeft_comp, Category.assoc, μ_natural_right_assoc]
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Mon_Class M
          inst✝³ : Mon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.LaxMonoidal
          A : Mon_ C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        slice_lhs 3 4 => rw [← F.map_comp, A.mul_assoc]
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Mon_Class M
          inst✝³ : Mon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.LaxMonoidal
          A : Mon_ C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
        -/
        simp }
        /-
          🎉 no goals
        -/
  map f :=
    { hom := F.map f.hom
                    /-
                      C : Type u₁
                      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                      inst✝⁵ : CategoryTheory.MonoidalCategory C
                      M N : C
                      inst✝⁴ : Mon_Class M
                      inst✝³ : Mon_Class N
                      D : Type u₂
                      inst✝² : CategoryTheory.Category.{v₂, u₂} D
                      inst✝¹ : CategoryTheory.MonoidalCategory D
                      F : CategoryTheory.Functor C D
                      inst✝ : F.LaxMonoidal
                      X✝ Y✝ : Mon_ C
                      f : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun A => { X := F.obj A.X, one := C …
                    -/
      one_hom := by dsimp; rw [Category.assoc, ← F.map_comp, f.one_hom]
                           /-
                             🎉 no goals
                           -/
      mul_hom := by
        rw [Category.assoc, μ_natural_assoc, ← F.map_comp, ← F.map_comp,
          f.mul_hom] }


/-- `mapMon` is functorial in the lax monoidal functor. -/
@[simps] -- Porting note: added this, not sure how it worked previously without.
def mapMonFunctor : LaxMonoidalFunctor C D ⥤ Mon_ C ⥤ Mon_ D where
  obj F := F.mapMon
  map α := { app := fun A => { hom := α.hom.app A.X } }
  map_comp _ _ := rfl


/-- Implementation of `Mon_.equivLaxMonoidalFunctorPUnit`. -/
@[simps]
def laxMonoidalToMon : LaxMonoidalFunctor (Discrete PUnit.{u + 1}) C ⥤ Mon_ C where
  obj F := (F.mapMon : Mon_ _ ⥤ Mon_ C).obj (trivial (Discrete PUnit))
  map α := ((Functor.mapMonFunctor (Discrete PUnit) C).map α).app _


/-- Implementation of `Mon_.equivLaxMonoidalFunctorPUnit`. -/
@[simps!]
def monToLaxMonoidalObj (A : Mon_ C) :
    Discrete PUnit.{u + 1} ⥤ C := (Functor.const _).obj A.X


instance (A : Mon_ C) : (monToLaxMonoidalObj A).LaxMonoidal where
  ε' := A.one
  μ' := fun _ _ => A.mul


@[simp]
lemma monToLaxMonoidalObj_ε (A : Mon_ C) :
    ε (monToLaxMonoidalObj A) = A.one := rfl


@[simp]
lemma monToLaxMonoidalObj_μ (A : Mon_ C) (X Y) :
    «μ» (monToLaxMonoidalObj A) X Y = A.mul := rfl


/-- Implementation of `Mon_.equivLaxMonoidalFunctorPUnit`. -/
@[simps]
def monToLaxMonoidal : Mon_ C ⥤ LaxMonoidalFunctor (Discrete PUnit.{u + 1}) C where
  obj A := LaxMonoidalFunctor.of (monToLaxMonoidalObj A)
  map f :=
    { hom := { app := fun _ => f.hom }
      isMonoidal := { } }


set_option maxHeartbeats 400000 in
/-- Implementation of `Mon_.equivLaxMonoidalFunctorPUnit`. -/
@[simps!]
def unitIso :
    𝟭 (LaxMonoidalFunctor (Discrete PUnit.{u + 1}) C) ≅ laxMonoidalToMon C ⋙ monToLaxMonoidal C :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    M N : C
    inst✝¹ : Mon_Class M
    inst✝ : Mon_Class N
    ⊢ ∀ {X Y : CategoryTheory.LaxMonoidalFunctor (CategoryTheory.Discrete PUnit.{u …
  -/
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝² : CategoryTheory.MonoidalCategory C
                                                                                 M N : C
                                                                                 inst✝¹ : Mon_Class M
                                                                                 inst✝ : Mon_Class N
                                                                                 F : CategoryTheory.LaxMonoidalFunctor (CategoryTheory.Discrete PUnit.{u + 1}) C
                                                                                 x✝ : CategoryTheory.Discrete PUnit.{u + 1}
                                                                                 ⊢ Eq x✝ (Mon_.trivial (CategoryTheory.Discrete PUnit.{u + 1})).X
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
  NatIso.ofComponents
             /-
               🎉 no goals
             -/
  /-
    🎉 no goals
  -/
    (fun F ↦ LaxMonoidalFunctor.isoOfComponents (fun _ ↦ F.mapIso (eqToIso (by ext))))


/-- Implementation of `Mon_.equivLaxMonoidalFunctorPUnit`. -/
@[simps!]
def counitIso : monToLaxMonoidal C ⋙ laxMonoidalToMon C ≅ 𝟭 (Mon_ C) :=
                               /-
                                 C : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                 inst✝² : CategoryTheory.MonoidalCategory C
                                 M N : C
                                 inst✝¹ : Mon_Class M
                                 inst✝ : Mon_Class N
                                 F : Mon_ C
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (((Mon_.EquivLaxMonoidalFunctorPUnit. …
                               -/
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents (fun F ↦ mkIso (Iso.refl _))
  /-
    🎉 no goals
  -/


/--
Monoid objects in `C` are "just" lax monoidal functors from the trivial monoidal category to `C`.
-/
@[simps]
def equivLaxMonoidalFunctorPUnit : LaxMonoidalFunctor (Discrete PUnit.{u + 1}) C ≌ Mon_ C where
  functor := laxMonoidalToMon C
  inverse := monToLaxMonoidal C
  unitIso := unitIso C
  counitIso := counitIso C


theorem one_associator {M N P : Mon_ C} :
    ((λ_ (𝟙_ C)).inv ≫ ((λ_ (𝟙_ C)).inv ≫ (M.one ⊗ N.one) ⊗ P.one)) ≫ (α_ M.X N.X P.X).hom =
      (λ_ (𝟙_ C)).inv ≫ (M.one ⊗ (λ_ (𝟙_ C)).inv ≫ (N.one ⊗ P.one)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, Iso.cancel_iso_inv_left]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [← Category.id_comp P.one, tensor_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 2 3 => rw [associator_naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_rhs 1 2 => rw [← Category.id_comp M.one, tensor_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [tensorHom_id, ← leftUnitor_tensor_inv]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [← cancel_epi (λ_ (𝟙_ C)).inv]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 2 => rw [leftUnitor_inv_naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem one_leftUnitor {M : Mon_ C} :
    ((λ_ (𝟙_ C)).inv ≫ (𝟙 (𝟙_ C) ⊗ M.one)) ≫ (λ_ M.X).hom = M.one := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem one_rightUnitor {M : Mon_ C} :
    ((λ_ (𝟙_ C)).inv ≫ (M.one ⊗ 𝟙 (𝟙_ C))) ≫ (ρ_ M.X).hom = M.one := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [← unitors_equal]
  /-
    🎉 no goals
  -/


theorem Mon_tensor_one_mul (M N : Mon_ C) :
    (((λ_ (𝟙_ C)).inv ≫ (M.one ⊗ N.one)) ▷ (M.X ⊗ N.X)) ≫
        tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) =
      (λ_ (M.X ⊗ N.X)).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [comp_whiskerRight_assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [tensorμ_natural_left]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [← tensor_comp, one_mul M, one_mul N]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor (CategoryTheory.Monoida …
  -/
  exact tensor_left_unitality M.X N.X
  /-
    🎉 no goals
  -/


theorem Mon_tensor_mul_one (M N : Mon_ C) :
    (M.X ⊗ N.X) ◁ ((λ_ (𝟙_ C)).inv ≫ (M.one ⊗ N.one)) ≫
        tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) =
      (ρ_ (M.X ⊗ N.X)).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [MonoidalCategory.whiskerLeft_comp_assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [tensorμ_natural_right]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [← tensor_comp, mul_one M, mul_one N]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor (CategoryTheory.Monoid …
  -/
  exact tensor_right_unitality M.X N.X
  /-
    🎉 no goals
  -/


theorem Mon_tensor_mul_assoc (M N : Mon_ C) :
    ((tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul)) ▷ (M.X ⊗ N.X)) ≫
        tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) =
      (α_ (M.X ⊗ N.X) (M.X ⊗ N.X) (M.X ⊗ N.X)).hom ≫
        ((M.X ⊗ N.X) ◁ (tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul))) ≫
          tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [comp_whiskerRight_assoc, MonoidalCategory.whiskerLeft_comp_assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 2 3 => rw [tensorμ_natural_left]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 3 4 => rw [← tensor_comp, mul_assoc M, mul_assoc N, tensor_comp, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  slice_lhs 1 3 => rw [tensor_associativity]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [← tensorμ_natural_right]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mul_associator {M N P : Mon_ C} :
    (tensorμ (M.X ⊗ N.X) P.X (M.X ⊗ N.X) P.X ≫
          (tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) ⊗ P.mul)) ≫
        (α_ M.X N.X P.X).hom =
      ((α_ M.X N.X P.X).hom ⊗ (α_ M.X N.X P.X).hom) ≫
        tensorμ M.X (N.X ⊗ P.X) M.X (N.X ⊗ P.X) ≫
          (M.mul ⊗ tensorμ N.X P.X N.X P.X ≫ (N.mul ⊗ P.mul)) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [tensor_obj, prodMonoidal_tensorObj, Category.assoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_lhs 2 3 => rw [← Category.id_comp P.mul, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_lhs 3 4 => rw [associator_naturality]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_rhs 3 4 => rw [← Category.id_comp M.mul, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  simp only [tensorHom_id, id_tensorHom]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_lhs 1 3 => rw [associator_monoidal]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N P : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


theorem mul_leftUnitor {M : Mon_ C} :
    (tensorμ (𝟙_ C) M.X (𝟙_ C) M.X ≫ ((λ_ (𝟙_ C)).hom ⊗ M.mul)) ≫ (λ_ M.X).hom =
      ((λ_ M.X).hom ⊗ (λ_ M.X).hom) ≫ M.mul := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← Category.comp_id (λ_ (𝟙_ C)).hom, ← Category.id_comp M.mul, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [tensorHom_id, id_tensorHom]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [leftUnitor_naturality]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_lhs 1 3 => rw [← leftUnitor_monoidal]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [Category.assoc, Category.id_comp]
  /-
    🎉 no goals
  -/


theorem mul_rightUnitor {M : Mon_ C} :
    (tensorμ M.X (𝟙_ C) M.X (𝟙_ C) ≫ (M.mul ⊗ (λ_ (𝟙_ C)).hom)) ≫ (ρ_ M.X).hom =
      ((ρ_ M.X).hom ⊗ (ρ_ M.X).hom) ≫ M.mul := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← Category.id_comp M.mul, ← Category.comp_id (λ_ (𝟙_ C)).hom, tensor_comp]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [tensorHom_id, id_tensorHom]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  slice_lhs 3 4 => rw [rightUnitor_naturality]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
  -/
  slice_lhs 1 3 => rw [← rightUnitor_monoidal]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simp only [Category.assoc, Category.id_comp]
  /-
    🎉 no goals
  -/


@[simps tensorObj_X tensorHom_hom]
instance monMonoidalStruct : MonoidalCategoryStruct (Mon_ C) :=
  let tensorObj (M N : Mon_ C) : Mon_ C :=
    { X := M.X ⊗ N.X
      one := (λ_ (𝟙_ C)).inv ≫ (M.one ⊗ N.one)
      mul := tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul)
      one_mul := Mon_tensor_one_mul M N
      mul_one := Mon_tensor_mul_one M N
      mul_assoc := Mon_tensor_mul_assoc M N }
  let tensorHom {X₁ Y₁ X₂ Y₂ : Mon_ C} (f : X₁ ⟶ Y₁) (g : X₂ ⟶ Y₂) :
      tensorObj _ _ ⟶ tensorObj _ _ :=
    { hom := f.hom ⊗ g.hom
      one_hom := by
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (tensorObj X₁ X₂).one (CategoryTheory …
        -/
        dsimp [tensorObj]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_lhs 2 3 => rw [← tensor_comp, Hom.one_hom f, Hom.one_hom g]
        /-
          🎉 no goals
        -/
      mul_hom := by
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (tensorObj X₁ X₂).mul (CategoryTheory …
        -/
        dsimp [tensorObj]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_rhs 1 2 => rw [tensorμ_natural]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        slice_lhs 2 3 => rw [← tensor_comp, Hom.mul_hom f, Hom.mul_hom g, tensor_comp]
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          inst✝³ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝² : Mon_Class M
          inst✝¹ : Mon_Class N
          inst✝ : CategoryTheory.BraidedCategory C
          tensorObj : Mon_ C → Mon_ C → Mon_ C := fun M N => { X := CategoryTheory.Monoi …
          X₁ Y₁ X₂ Y₂ : Mon_ C
          f : Quiver.Hom X₁ Y₁
          g : Quiver.Hom X₂ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory.tens …
        -/
        simp only [Category.assoc] }
        /-
          🎉 no goals
        -/
  { tensorObj := tensorObj
    tensorHom := tensorHom
    whiskerRight := fun f Y => tensorHom f (𝟙 Y)
    whiskerLeft := fun X _ _ g => tensorHom (𝟙 X) g
    tensorUnit := trivial C
    associator := fun M N P ↦ mkIso (α_ M.X N.X P.X) one_associator mul_associator
    leftUnitor := fun M ↦ mkIso (λ_ M.X) one_leftUnitor mul_leftUnitor
    rightUnitor := fun M ↦ mkIso (ρ_ M.X) one_rightUnitor mul_rightUnitor }


@[simp]
theorem tensorUnit_X : (𝟙_ (Mon_ C)).X = 𝟙_ C := rfl


@[simp]
theorem tensorUnit_one : (𝟙_ (Mon_ C)).one = 𝟙 (𝟙_ C) := rfl


@[simp]
theorem tensorUnit_mul : (𝟙_ (Mon_ C)).mul = (λ_ (𝟙_ C)).hom := rfl


@[simp]
theorem tensorObj_one (X Y : Mon_ C) : (X ⊗ Y).one = (λ_ (𝟙_ C)).inv ≫ (X.one ⊗ Y.one) := rfl


@[simp]
theorem tensorObj_mul (X Y : Mon_ C) :
    (X ⊗ Y).mul = tensorμ X.X Y.X X.X Y.X ≫ (X.mul ⊗ Y.mul) := rfl


@[simp]
theorem whiskerLeft_hom {X Y : Mon_ C} (f : X ⟶ Y) (Z : Mon_ C) :
    (f ▷ Z).hom = f.hom ▷ Z.X := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y : Mon_ C
    f : Quiver.Hom X Y
    Z : Mon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight f Z).hom (CategoryThe …
  -/
  rw [← tensorHom_id]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem whiskerRight_hom (X : Mon_ C) {Y Z : Mon_ C} (f : Y ⟶ Z) :
    (X ◁ f).hom = X.X ◁ f.hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y Z : Mon_ C
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X f).hom (CategoryTheo …
  -/
  rw [← id_tensorHom]; rfl
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem leftUnitor_hom_hom (X : Mon_ C) : (λ_ X).hom.hom = (λ_ X.X).hom := rfl


@[simp]
theorem leftUnitor_inv_hom (X : Mon_ C) : (λ_ X).inv.hom = (λ_ X.X).inv := rfl


@[simp]
theorem rightUnitor_hom_hom (X : Mon_ C) : (ρ_ X).hom.hom = (ρ_ X.X).hom := rfl


@[simp]
theorem rightUnitor_inv_hom (X : Mon_ C) : (ρ_ X).inv.hom = (ρ_ X.X).inv := rfl


@[simp]
theorem associator_hom_hom (X Y Z : Mon_ C) : (α_ X Y Z).hom.hom = (α_ X.X Y.X Z.X).hom := rfl


@[simp]
theorem associator_inv_hom (X Y Z : Mon_ C) : (α_ X Y Z).inv.hom = (α_ X.X Y.X Z.X).inv := rfl


@[simp]
theorem tensor_one (M N : Mon_ C) : (M ⊗ N).one = (λ_ (𝟙_ C)).inv ≫ (M.one ⊗ N.one) := rfl


@[simp]
theorem tensor_mul (M N : Mon_ C) : (M ⊗ N).mul =
    tensorμ M.X N.X M.X N.X ≫ (M.mul ⊗ N.mul) := rfl


instance monMonoidal : MonoidalCategory (Mon_ C) where
                      /-
                        C : Type u₁
                        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝³ : CategoryTheory.MonoidalCategory C
                        M N : C
                        inst✝² : Mon_Class M
                        inst✝¹ : Mon_Class N
                        inst✝ : CategoryTheory.BraidedCategory C
                        ⊢ ∀ {X₁ Y₁ X₂ Y₂ : Mon_ C} (f : Quiver.Hom X₁ Y₁) (g : Quiver.Hom X₂ Y₂), Eq ( …
                      -/
  tensorHom_def := by intros; ext; simp [tensorHom_def]
                                   /-
                                     🎉 no goals
                                   -/


@[simps!]
instance {M N : C} [Mon_Class M] [Mon_Class N] : Mon_Class (M ⊗ N) :=
  inferInstanceAs <| Mon_Class (Mon_.mk' M ⊗ Mon_.mk' N).X


/-- The forgetful functor from `Mon_ C` to `C` is monoidal when `C` is monoidal. -/
instance : (forget C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _ }


@[simp] theorem forget_ε : ε (forget C) = 𝟙 (𝟙_ C) := rfl

@[simp] theorem forget_η : «η» (forget C) = 𝟙 (𝟙_ C) := rfl

@[simp] theorem forget_μ (X Y : Mon_ C) : «μ» (forget C) X Y = 𝟙 (X.X ⊗ Y.X) := rfl

@[simp] theorem forget_δ (X Y : Mon_ C) : δ (forget C) X Y = 𝟙 (X.X ⊗ Y.X) := rfl


theorem one_braiding {X Y : Mon_ C} : (X ⊗ Y).one ≫ (β_ X.X Y.X).hom = (Y ⊗ X).one := by
  simp only [monMonoidalStruct_tensorObj_X, tensor_one, Category.assoc,
    BraidedCategory.braiding_naturality, braiding_tensorUnit_right, Iso.cancel_iso_inv_left]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    X Y : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  monoidal
  /-
    🎉 no goals
  -/


theorem mul_braiding {X Y : Mon_ C} :
    (X ⊗ Y).mul ≫ (β_ X.X Y.X).hom = ((β_ X.X Y.X).hom ⊗ (β_ X.X Y.X).hom) ≫ (Y ⊗ X).mul := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.SymmetricCategory C
    X Y : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp
  simp only [tensorμ, Category.assoc, BraidedCategory.braiding_naturality,
    BraidedCategory.braiding_tensor_right, BraidedCategory.braiding_tensor_left,
    comp_whiskerRight, whisker_assoc, MonoidalCategory.whiskerLeft_comp, pentagon_assoc,
    pentagon_inv_hom_hom_hom_inv_assoc, Iso.inv_hom_id_assoc, whiskerLeft_hom_inv_assoc]
  slice_lhs 3 4 =>
    -- We use symmetry here:
    rw [← MonoidalCategory.whiskerLeft_comp, ← comp_whiskerRight, SymmetricCategory.symmetry]
  simp only [id_whiskerRight, MonoidalCategory.whiskerLeft_id, Category.id_comp, Category.assoc,
    pentagon_inv_assoc, Iso.hom_inv_id_assoc]
  slice_lhs 1 2 =>
    rw [← associator_inv_naturality_left]
  slice_lhs 2 3 =>
    rw [Iso.inv_hom_id]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.SymmetricCategory C
    X Y : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Category.id_comp]
  slice_lhs 2 3 =>
    rw [← associator_naturality_right]
  slice_lhs 1 2 =>
    rw [← tensorHom_def]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.SymmetricCategory C
    X Y : Mon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc]
  /-
    🎉 no goals
  -/


instance : SymmetricCategory (Mon_ C) where
  braiding := fun X Y => mkIso (β_ X.X Y.X) one_braiding mul_braiding
  symmetry := fun X Y => by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      M N : C
      inst✝² : Mon_Class M
      inst✝¹ : Mon_Class N
      inst✝ : CategoryTheory.SymmetricCategory C
      X Y : Mon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
    -/
    ext
    /-
      case w
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.MonoidalCategory C
      M N : C
      inst✝² : Mon_Class M
      inst✝¹ : Mon_Class N
      inst✝ : CategoryTheory.SymmetricCategory C
      X Y : Mon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
    -/
    simp [← SymmetricCategory.braiding_swap_eq_inv_braiding]
    /-
      🎉 no goals
    -/


