/-- A comonoid object internal to a monoidal category.

When the monoidal category is preadditive, this is also sometimes called a "coalgebra object".
-/
class Comon_Class (X : C) where
  /-- The counit morphism of a comonoid object. -/
  counit : X ⟶ 𝟙_ C
  /-- The comultiplication morphism of a comonoid object. -/
  comul : X ⟶ X ⊗ X
  /- For the names of the conditions below, the unprimed names are reserved for the version where
  the argument `X` is explicit. -/
  counit_comul' : comul ≫ counit ▷ X = (λ_ X).inv := by aesop_cat
  comul_counit' : comul ≫ X ◁ counit = (ρ_ X).inv := by aesop_cat
  comul_assoc' : comul ≫ X ◁ comul = comul ≫ (comul ▷ X) ≫ (α_ X X X).hom := by aesop_cat


@[inherit_doc] scoped notation "Δ" => Comon_Class.comul

@[inherit_doc] scoped notation "Δ["M"]" => Comon_Class.comul (X := M)

@[inherit_doc] scoped notation "ε" => Comon_Class.counit

@[inherit_doc] scoped notation "ε["M"]" => Comon_Class.counit (X := M)

/- The simp attribute is reserved for the unprimed versions. -/

attribute [reassoc] counit_comul' comul_counit' comul_assoc'


@[reassoc (attr := simp)]
theorem counit_comul (X : C) [Comon_Class X] : Δ ≫ ε ▷ X = (λ_ X).inv := counit_comul'


@[reassoc (attr := simp)]
theorem comul_counit (X : C) [Comon_Class X] : Δ ≫ X ◁ ε = (ρ_ X).inv := comul_counit'


@[reassoc (attr := simp)]
theorem comul_assoc (X : C) [Comon_Class X] :
    Δ ≫ X ◁ Δ = Δ ≫ Δ ▷ X ≫ (α_ X X X).hom :=
  comul_assoc'


/-- The property that a morphism between comonoid objects is a comonoid morphism. -/
class IsComon_Hom (f : M ⟶ N) : Prop where
  hom_counit : f ≫ ε = ε := by aesop_cat
  hom_comul : f ≫ Δ = Δ ≫ (f ⊗ f) := by aesop_cat


attribute [reassoc (attr := simp)] IsComon_Hom.hom_counit IsComon_Hom.hom_comul


/-- A comonoid object internal to a monoidal category.

When the monoidal category is preadditive, this is also sometimes called a "coalgebra object".
-/
structure Comon_ where
  /-- The underlying object of a comonoid object. -/
  X : C
  /-- The counit of a comonoid object. -/
  counit : X ⟶ 𝟙_ C
  /-- The comultiplication morphism of a comonoid object. -/
  comul : X ⟶ X ⊗ X
  counit_comul : comul ≫ (counit ▷ X) = (λ_ X).inv := by aesop_cat
  comul_counit : comul ≫ (X ◁ counit) = (ρ_ X).inv := by aesop_cat
  comul_assoc : comul ≫ (X ◁ comul) = comul ≫ (comul ▷ X) ≫ (α_ X X X).hom := by aesop_cat


attribute [reassoc (attr := simp)] Comon_.counit_comul Comon_.comul_counit


attribute [reassoc (attr := simp)] Comon_.comul_assoc


/-- Construct an object of `Comon_ C` from an object `X : C` and `Comon_Class X` instance. -/
@[simps]
def mk' (X : C) [Comon_Class X] : Comon_ C where
  X := X
  counit := ε
  comul := Δ


instance {M : Comon_ C} : Comon_Class M.X where
  counit := M.counit
  comul := M.comul
  counit_comul' := M.counit_comul
  comul_counit' := M.comul_counit
  comul_assoc' := M.comul_assoc


/-- The trivial comonoid object. We later show this is terminal in `Comon_ C`.
-/
@[simps]
def trivial : Comon_ C where
  X := 𝟙_ C
  counit := 𝟙 _
  comul := (λ_ _).inv
                    /-
                      C : Type u₁
                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                      inst✝² : CategoryTheory.MonoidalCategory C
                      M N : C
                      inst✝¹ : Comon_Class M
                      inst✝ : Comon_Class N
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                    -/
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝² : CategoryTheory.MonoidalCategory C
                       M N : C
                       inst✝¹ : Comon_Class M
                       inst✝ : Comon_Class N
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                     -/
  comul_assoc := by monoidal_coherence
                     /-
                       🎉 no goals
                     -/
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝² : CategoryTheory.MonoidalCategory C
                       M N : C
                       inst✝¹ : Comon_Class M
                       inst✝ : Comon_Class N
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                     -/
                    /-
                      🎉 no goals
                    -/
                     /-
                       🎉 no goals
                     -/
  counit_comul := by monoidal_coherence
  comul_counit := by monoidal_coherence


instance : Inhabited (Comon_ C) :=
  ⟨trivial C⟩


@[reassoc (attr := simp)]
theorem counit_comul_hom {Z : C} (f : M.X ⟶ Z) : M.comul ≫ (M.counit ⊗ f) = f ≫ (λ_ Z).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Comon_ C
    Z : C
    f : Quiver.Hom M.X Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul (CategoryTheory.MonoidalCateg …
  -/
  rw [leftUnitor_inv_naturality, tensorHom_def, counit_comul_assoc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem comul_counit_hom {Z : C} (f : M.X ⟶ Z) : M.comul ≫ (f ⊗ M.counit) = f ≫ (ρ_ Z).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Comon_ C
    Z : C
    f : Quiver.Hom M.X Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul (CategoryTheory.MonoidalCateg …
  -/
  rw [rightUnitor_inv_naturality, tensorHom_def', comul_counit_assoc]
  /-
    🎉 no goals
  -/


@[reassoc] theorem comul_assoc_flip :
    M.comul ≫ (M.comul ▷ M.X) = M.comul ≫ (M.X ◁ M.comul) ≫ (α_ M.X M.X M.X).inv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.MonoidalCategory C
    M : Comon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul (CategoryTheory.MonoidalCateg …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A morphism of comonoid objects. -/
@[ext]
structure Hom (M N : Comon_ C) where
  /-- The underlying morphism of a morphism of comonoid objects. -/
  hom : M.X ⟶ N.X
  hom_counit : hom ≫ N.counit = M.counit := by aesop_cat
  hom_comul : hom ≫ N.comul = M.comul ≫ (hom ⊗ hom) := by aesop_cat


attribute [reassoc (attr := simp)] Hom.hom_counit Hom.hom_comul


/-- The identity morphism on a comonoid object. -/
@[simps]
def id (M : Comon_ C) : Hom M M where
  hom := 𝟙 M.X


instance homInhabited (M : Comon_ C) : Inhabited (Hom M M) :=
  ⟨id M⟩


/-- Composition of morphisms of monoid objects. -/
@[simps]
def comp {M N O : Comon_ C} (f : Hom M N) (g : Hom N O) : Hom M O where
  hom := f.hom ≫ g.hom


instance : Category (Comon_ C) where
  Hom M N := Hom M N
  id := id
  comp f g := comp f g


@[ext] lemma ext {X Y : Comon_ C} {f g : X ⟶ Y} (w : f.hom = g.hom) : f = g := Hom.ext w


@[simp] theorem id_hom' (M : Comon_ C) : (𝟙 M : Hom M M).hom = 𝟙 M.X := rfl


@[simp]
theorem comp_hom' {M N K : Comon_ C} (f : M ⟶ N) (g : N ⟶ K) : (f ≫ g).hom = f.hom ≫ g.hom :=
  rfl


/-- The forgetful functor from comonoid objects to the ambient category. -/
@[simps]
def forget : Comon_ C ⥤ C where
  obj A := A.X
  map f := f.hom


instance forget_faithful : (@forget C _ _).Faithful where


instance {A B : Comon_ C} (f : A ⟶ B) [e : IsIso ((forget C).map f)] : IsIso f.hom := e


/-- The forgetful functor from comonoid objects to the ambient category reflects isomorphisms. -/
instance : (forget C).ReflectsIsomorphisms where
  reflects f e :=
                               /-
                                 C : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                 inst✝² : CategoryTheory.MonoidalCategory C
                                 M✝ N : C
                                 inst✝¹ : Comon_Class M✝
                                 inst✝ : Comon_Class N
                                 M A✝ B✝ : Comon_ C
                                 f : Quiver.Hom A✝ B✝
                                 e : CategoryTheory.IsIso ((Comon_.forget C).map f)
                                 ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f { hom := CategoryTheory.inv f. …
                               -/
    ⟨⟨{ hom := inv f.hom }, by aesop_cat⟩⟩
                               /-
                                 🎉 no goals
                               -/


/-- Construct an isomorphism of comonoids by giving an isomorphism between the underlying objects
and checking compatibility with counit and comultiplication only in the forward direction.
-/
@[simps]
def mkIso {M N : Comon_ C} (f : M.X ≅ N.X) (f_counit : f.hom ≫ N.counit = M.counit := by aesop_cat)
    (f_comul : f.hom ≫ N.comul = M.comul ≫ (f.hom ⊗ f.hom) := by aesop_cat) : M ≅ N where
  hom :=
    { hom := f.hom
      hom_counit := f_counit
      hom_comul := f_comul }
  inv :=
    { hom := f.inv
                       /-
                         C : Type u₁
                         inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                         inst✝² : CategoryTheory.MonoidalCategory C
                         M✝¹ N✝ : C
                         inst✝¹ : Comon_Class M✝¹
                         inst✝ : Comon_Class N✝
                         M✝ M N : Comon_ C
                         f : CategoryTheory.Iso M.X N.X
                         f_counit : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.counit) M …
                         f_comul : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.comul) (Ca …
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv M.counit) N.counit
                       -/
      hom_counit := by rw [← f_counit]; simp
                                        /-
                                          🎉 no goals
                                        -/
      hom_comul := by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.MonoidalCategory C
          M✝¹ N✝ : C
          inst✝¹ : Comon_Class M✝¹
          inst✝ : Comon_Class N✝
          M✝ M N : Comon_ C
          f : CategoryTheory.Iso M.X N.X
          f_counit : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.counit) M …
          f_comul : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.comul) (Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.inv M.comul) (CategoryTheory.Catego …
        -/
        rw [← cancel_epi f.hom]
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.MonoidalCategory C
          M✝¹ N✝ : C
          inst✝¹ : Comon_Class M✝¹
          inst✝ : Comon_Class N✝
          M✝ M N : Comon_ C
          f : CategoryTheory.Iso M.X N.X
          f_counit : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.counit) M …
          f_comul : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.comul) (Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
        -/
        slice_rhs 1 2 => rw [f_comul]
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          inst✝² : CategoryTheory.MonoidalCategory C
          M✝¹ N✝ : C
          inst✝¹ : Comon_Class M✝¹
          inst✝ : Comon_Class N✝
          M✝ M N : Comon_ C
          f : CategoryTheory.Iso M.X N.X
          f_counit : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.counit) M …
          f_comul : autoParam (Eq (CategoryTheory.CategoryStruct.comp f.hom N.comul) (Ca …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
        -/
        simp }
        /-
          🎉 no goals
        -/


@[simps]
instance uniqueHomToTrivial (A : Comon_ C) : Unique (A ⟶ trivial C) where
  default :=
    { hom := A.counit
                      /-
                        C : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝² : CategoryTheory.MonoidalCategory C
                        M✝ N : C
                        inst✝¹ : Comon_Class M✝
                        inst✝ : Comon_Class N
                        M A : Comon_ C
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp A.counit (Comon_.trivial C).comul) (C …
                      -/
      hom_comul := by simp [A.comul_counit, unitors_inv_equal] }
                      /-
                        🎉 no goals
                      -/
  uniq f := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      f : Quiver.Hom A (Comon_.trivial C)
      ⊢ Eq f Inhabited.default
    -/
    ext; simp
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      f : Quiver.Hom A (Comon_.trivial C)
      ⊢ Eq f.hom A.counit
    -/
    rw [← Category.comp_id f.hom]
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      f : Quiver.Hom A (Comon_.trivial C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom (CategoryTheory.CategoryStruct. …
    -/
    erw [f.hom_counit]
    /-
      🎉 no goals
    -/


instance : HasTerminal (Comon_ C) :=
  hasTerminal_of_unique (trivial C)


/--
Turn a comonoid object into a monoid object in the opposite category.
-/
@[simps] def Comon_ToMon_OpOp_obj' (A : Comon_ C) : Mon_ (Cᵒᵖ) where
  X := op A.X
  one := A.counit.op
  mul := A.comul.op
  one_mul := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [← op_whiskerRight, ← op_comp, counit_comul]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.leftUnitor A.X).inv.op (CategoryTh …
    -/
    rfl
    /-
      🎉 no goals
    -/
  mul_one := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rw [← op_whiskerLeft, ← op_comp, comul_counit]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.rightUnitor A.X).inv.op (CategoryT …
    -/
    rfl
    /-
      🎉 no goals
    -/
  mul_assoc := by
    rw [← op_inv_associator, ← op_whiskerRight, ← op_comp, ← op_whiskerLeft, ← op_comp,
      comul_assoc_flip, op_comp, op_comp_assoc]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M A : Comon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
The contravariant functor turning comonoid objects into monoid objects in the opposite category.
-/
@[simps] def Comon_ToMon_OpOp : Comon_ C ⥤ (Mon_ (Cᵒᵖ))ᵒᵖ where
  obj A := op (Comon_ToMon_OpOp_obj' C A)
  map := fun f => op <|
    { hom := f.hom.op
                    /-
                      C : Type u₁
                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                      inst✝² : CategoryTheory.MonoidalCategory C
                      M✝ N : C
                      inst✝¹ : Comon_Class M✝
                      inst✝ : Comon_Class N
                      M X✝ Y✝ : Comon_ C
                      f : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((fun A => { unop := C …
                    -/
      one_hom := by apply Quiver.Hom.unop_inj; simp
                                               /-
                                                 🎉 no goals
                                               -/
                    /-
                      C : Type u₁
                      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                      inst✝² : CategoryTheory.MonoidalCategory C
                      M✝ N : C
                      inst✝¹ : Comon_Class M✝
                      inst✝ : Comon_Class N
                      M X✝ Y✝ : Comon_ C
                      f : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((fun A => { unop := C …
                    -/
      mul_hom := by apply Quiver.Hom.unop_inj; simp [op_tensorHom] }
                                               /-
                                                 🎉 no goals
                                               -/


/--
Turn a monoid object in the opposite category into a comonoid object.
-/
@[simps] def Mon_OpOpToComon_obj' (A : (Mon_ (Cᵒᵖ))) : Comon_ C where
  X := unop A.X
  counit := A.one.unop
  comul := A.mul.unop
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝² : CategoryTheory.MonoidalCategory C
                       M✝ N : C
                       inst✝¹ : Comon_Class M✝
                       inst✝ : Comon_Class N
                       M : Comon_ C
                       A : Mon_ (Opposite C)
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp A.mul.unop (CategoryTheory.MonoidalCa …
                     -/
  counit_comul := by rw [← unop_whiskerRight, ← unop_comp, Mon_.one_mul]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                     /-
                       C : Type u₁
                       inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝² : CategoryTheory.MonoidalCategory C
                       M✝ N : C
                       inst✝¹ : Comon_Class M✝
                       inst✝ : Comon_Class N
                       M : Comon_ C
                       A : Mon_ (Opposite C)
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp A.mul.unop (CategoryTheory.MonoidalCa …
                     -/
  comul_counit := by rw [← unop_whiskerLeft, ← unop_comp, Mon_.mul_one]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
  comul_assoc := by
    rw [← unop_whiskerRight, ← unop_whiskerLeft, ← unop_comp_assoc, ← unop_comp,
      Mon_.mul_assoc_flip]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.MonoidalCategory C
      M✝ N : C
      inst✝¹ : Comon_Class M✝
      inst✝ : Comon_Class N
      M : Comon_ C
      A : Mon_ (Opposite C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
The contravariant functor turning monoid objects in the opposite category into comonoid objects.
-/
@[simps]
def Mon_OpOpToComon_ : (Mon_ (Cᵒᵖ))ᵒᵖ ⥤ Comon_ C where
  obj A := Mon_OpOpToComon_obj' C (unop A)
  map := fun f =>
    { hom := f.unop.hom.unop
                       /-
                         C : Type u₁
                         inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                         inst✝² : CategoryTheory.MonoidalCategory C
                         M✝ N : C
                         inst✝¹ : Comon_Class M✝
                         inst✝ : Comon_Class N
                         M : Comon_ C
                         X✝ Y✝ : Opposite (Mon_ (Opposite C))
                         f : Quiver.Hom X✝ Y✝
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.hom.unop ((fun A => Comon_.Mon …
                       -/
      hom_counit := by apply Quiver.Hom.op_inj; simp
                                                /-
                                                  🎉 no goals
                                                -/
                      /-
                        C : Type u₁
                        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                        inst✝² : CategoryTheory.MonoidalCategory C
                        M✝ N : C
                        inst✝¹ : Comon_Class M✝
                        inst✝ : Comon_Class N
                        M : Comon_ C
                        X✝ Y✝ : Opposite (Mon_ (Opposite C))
                        f : Quiver.Hom X✝ Y✝
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.hom.unop ((fun A => Comon_.Mon …
                      -/
      hom_comul := by apply Quiver.Hom.op_inj; simp [op_tensorHom] }
                                               /-
                                                 🎉 no goals
                                               -/


/--
Comonoid objects are contravariantly equivalent to monoid objects in the opposite category.
-/
@[simps]
def Comon_EquivMon_OpOp : Comon_ C ≌ (Mon_ (Cᵒᵖ))ᵒᵖ :=
  { functor := Comon_ToMon_OpOp C
    inverse := Mon_OpOpToComon_ C
               /-
                 C : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                 inst✝² : CategoryTheory.MonoidalCategory C
                 M✝ N : C
                 inst✝¹ : Comon_Class M✝
                 inst✝ : Comon_Class N
                 M : Comon_ C
                 ⊢ ∀ {X Y : Comon_ C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
               -/
    unitIso := NatIso.ofComponents (fun _ => Iso.refl _)
               /-
                 🎉 no goals
               -/
                 /-
                   C : Type u₁
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                   inst✝² : CategoryTheory.MonoidalCategory C
                   M✝ N : C
                   inst✝¹ : Comon_Class M✝
                   inst✝ : Comon_Class N
                   M : Comon_ C
                   ⊢ ∀ {X Y : Opposite (Mon_ (Opposite C))} (f : Quiver.Hom X Y), Eq (CategoryThe …
                 -/
    counitIso := NatIso.ofComponents (fun _ => Iso.refl _) }
                 /-
                   🎉 no goals
                 -/


/--
Comonoid objects in a braided category form a monoidal category.

This definition is via transporting back and forth to monoids in the opposite category,
-/
@[simps!]
instance monoidal [BraidedCategory C] : MonoidalCategory (Comon_ C) :=
  Monoidal.transport (Comon_EquivMon_OpOp C).symm


theorem tensorObj_X (A B : Comon_ C) : (A ⊗ B).X = A.X ⊗ B.X := rfl


instance (A B : C) [Comon_Class A] [Comon_Class B] : Comon_Class (A ⊗ B) :=
  inferInstanceAs <| Comon_Class (Comon_.mk' A ⊗ Comon_.mk' B).X


theorem tensorObj_counit (A B : Comon_ C) : (A ⊗ B).counit = (A.counit ⊗ B.counit) ≫ (λ_ _).hom :=
  rfl


/--
Preliminary statement of the comultiplication for a tensor product of comonoids.
This version is the definitional equality provided by transport, and not quite as good as
the version provided in `tensorObj_comul` below.
-/
theorem tensorObj_comul' (A B : Comon_ C) :
    (A ⊗ B).comul =
      (A.comul ⊗ B.comul) ≫ (tensorμ (op A.X) (op B.X) (op A.X) (op B.X)).unop := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorObj A B).comul (CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
The comultiplication on the tensor product of two comonoids is
the tensor product of the comultiplications followed by the tensor strength
(to shuffle the factors back into order).
-/
theorem tensorObj_comul (A B : Comon_ C) :
    (A ⊗ B).comul = (A.comul ⊗ B.comul) ≫ tensorμ A.X A.X B.X B.X := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.tensorObj A B).comul (CategoryTheo …
  -/
  rw [tensorObj_comul']
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  congr
  /-
    case e_a.e_f
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.MonoidalCategory.tensorμ { unop := A.X } { unop := B.X }  …
  -/
  simp only [tensorμ, unop_tensorObj, unop_op]
  /-
    case e_a.e_f
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  apply Quiver.Hom.unop_inj
  /-
    case e_a.e_f.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  dsimp [op_tensorObj, op_associator]
  /-
    case e_a.e_f.a
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    A B : Comon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, Category.assoc, Category.assoc]
  /-
    🎉 no goals
  -/


/-- The forgetful functor from `Comon_ C` to `C` is monoidal when `C` is monoidal. -/
instance : (forget C).Monoidal :=
  Functor.CoreMonoidal.toMonoidal
    { εIso := Iso.refl _
      μIso := fun _ _ ↦ Iso.refl _ }


@[simp] theorem forget_ε : «ε» (forget C) = 𝟙 (𝟙_ C) := rfl

@[simp] theorem forget_η : η (forget C) = 𝟙 (𝟙_ C) := rfl

@[simp] theorem forget_μ (X Y : Comon_ C) : μ (forget C) X Y = 𝟙 (X.X ⊗ Y.X) := rfl

@[simp] theorem forget_δ (X Y : Comon_ C) : δ (forget C) X Y = 𝟙 (X.X ⊗ Y.X) := rfl


/-- A oplax monoidal functor takes comonoid objects to comonoid objects.

That is, a oplax monoidal functor `F : C ⥤ D` induces a functor `Comon_ C ⥤ Comon_ D`.
-/
@[simps]
def mapComon (F : C ⥤ D) [F.OplaxMonoidal] : Comon_ C ⥤ Comon_ D where
  obj A :=
    { X := F.obj A.X
      counit := F.map A.counit ≫ η F
      comul := F.map A.comul ≫ δ F _ _
      counit_comul := by
        simp_rw [comp_whiskerRight, Category.assoc, δ_natural_left_assoc, left_unitality,
          ← F.map_comp_assoc, A.counit_comul]
      comul_counit := by
        simp_rw [MonoidalCategory.whiskerLeft_comp, Category.assoc, δ_natural_right_assoc,
          right_unitality, ← F.map_comp_assoc, A.comul_counit]
      comul_assoc := by
        simp_rw [comp_whiskerRight, Category.assoc, δ_natural_left_assoc,
          MonoidalCategory.whiskerLeft_comp, δ_natural_right_assoc,
          ← F.map_comp_assoc, A.comul_assoc, F.map_comp, Category.assoc, associativity] }
  map f :=
    { hom := F.map f.hom
                       /-
                         C : Type u₁
                         inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                         inst✝⁵ : CategoryTheory.MonoidalCategory C
                         M N : C
                         inst✝⁴ : Comon_Class M
                         inst✝³ : Comon_Class N
                         D : Type u₂
                         inst✝² : CategoryTheory.Category.{v₂, u₂} D
                         inst✝¹ : CategoryTheory.MonoidalCategory D
                         F : CategoryTheory.Functor C D
                         inst✝ : F.OplaxMonoidal
                         X✝ Y✝ : Comon_ C
                         f : Quiver.Hom X✝ Y✝
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.hom) ((fun A => { X := F.obj …
                       -/
      hom_counit := by dsimp; rw [← F.map_comp_assoc, f.hom_counit]
                              /-
                                🎉 no goals
                              -/
      hom_comul := by
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Comon_Class M
          inst✝³ : Comon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.OplaxMonoidal
          X✝ Y✝ : Comon_ C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.hom) ((fun A => { X := F.obj …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
          inst✝⁵ : CategoryTheory.MonoidalCategory C
          M N : C
          inst✝⁴ : Comon_Class M
          inst✝³ : Comon_Class N
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.MonoidalCategory D
          F : CategoryTheory.Functor C D
          inst✝ : F.OplaxMonoidal
          X✝ Y✝ : Comon_ C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f.hom) (CategoryTheory.Categor …
        -/
        rw [Category.assoc, δ_natural, ← F.map_comp_assoc, ← F.map_comp_assoc, f.hom_comul] }
        /-
          🎉 no goals
        -/
                 /-
                   C : Type u₁
                   inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                   inst✝⁵ : CategoryTheory.MonoidalCategory C
                   M N : C
                   inst✝⁴ : Comon_Class M
                   inst✝³ : Comon_Class N
                   D : Type u₂
                   inst✝² : CategoryTheory.Category.{v₂, u₂} D
                   inst✝¹ : CategoryTheory.MonoidalCategory D
                   F : CategoryTheory.Functor C D
                   inst✝ : F.OplaxMonoidal
                   A : Comon_ C
                   ⊢ Eq ({ obj := fun A => { X := F.obj A.X, counit := CategoryTheory.CategoryStr …
                 -/
  map_id A := by ext; simp
                      /-
                        🎉 no goals
                      -/
                     /-
                       C : Type u₁
                       inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝⁵ : CategoryTheory.MonoidalCategory C
                       M N : C
                       inst✝⁴ : Comon_Class M
                       inst✝³ : Comon_Class N
                       D : Type u₂
                       inst✝² : CategoryTheory.Category.{v₂, u₂} D
                       inst✝¹ : CategoryTheory.MonoidalCategory D
                       F : CategoryTheory.Functor C D
                       inst✝ : F.OplaxMonoidal
                       X✝ Y✝ Z✝ : Comon_ C
                       f : Quiver.Hom X✝ Y✝
                       g : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun A => { X := F.obj A.X, counit := CategoryTheory.CategoryStr …
                     -/
  map_comp f g := by ext; simp
                          /-
                            🎉 no goals
                          -/

-- TODO We haven't yet set up the category structure on `OplaxMonoidalFunctor C D`
-- and so can't state `mapComonFunctor : OplaxMonoidalFunctor C D ⥤ Comon_ C ⥤ Comon_ D`.


