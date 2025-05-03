/--
The functor from a cartesian monoidal category to comonoids in that category,
equipping every object with the diagonal map as a comultiplication.
-/
def cartesianComon_ : C ⥤ Comon_ C where
  obj := fun X =>
  { X := X
    comul := diag X
    counit := terminal.from X }
  map := fun f => { hom := f }


                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                                                     inst✝¹ : CategoryTheory.Limits.HasTerminal C
                                                                                     inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                                                     A : Comon_ C
                                                                                     ⊢ Eq A.counit (CategoryTheory.Limits.terminal.from A.X)
                                                                                   -/
@[simp] theorem counit_eq_from (A : Comon_ C) : A.counit = terminal.from A.X := by ext
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp] theorem comul_eq_diag (A : Comon_ C) : A.comul = diag A.X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    A : Comon_ C
    ⊢ Eq A.comul (CategoryTheory.Limits.diag A.X)
  -/
  ext
    /-
      case w₁
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasBinaryProducts C
      A : Comon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp A.comul CategoryTheory.Limits.prod.fs …
    -/
  · simpa using A.comul_counit =≫ prod.fst
    /-
      🎉 no goals
    -/
    /-
      case w₂
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasTerminal C
      inst✝ : CategoryTheory.Limits.HasBinaryProducts C
      A : Comon_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp A.comul CategoryTheory.Limits.prod.sn …
    -/
  · simpa using A.counit_comul =≫ prod.snd
    /-
      🎉 no goals
    -/


/--
Every comonoid object in a cartesian monoidal category is equivalent to
the canonical comonoid structure on the underlying object.
-/
@[simps] def iso_cartesianComon_ (A : Comon_ C) : A ≅ (cartesianComon_ C).obj A.X :=
  { hom := { hom := 𝟙 _ }
    inv := { hom := 𝟙 _ } }


/--
The category of comonoid objects in a cartesian monoidal category is equivalent
to the category itself, via the forgetful functor.
-/
@[simps] def comonEquiv : Comon_ C ≌ C where
  functor := Comon_.forget C
  inverse := cartesianComon_ C
             /-
               C : Type u
               inst✝² : CategoryTheory.Category.{v, u} C
               inst✝¹ : CategoryTheory.Limits.HasTerminal C
               inst✝ : CategoryTheory.Limits.HasBinaryProducts C
               ⊢ ∀ {X Y : Comon_ C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.c …
             -/
  unitIso := NatIso.ofComponents (fun A => iso_cartesianComon_ A)
             /-
               🎉 no goals
             -/
               /-
                 C : Type u
                 inst✝² : CategoryTheory.Category.{v, u} C
                 inst✝¹ : CategoryTheory.Limits.HasTerminal C
                 inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
               -/
  counitIso := NatIso.ofComponents (fun _ => Iso.refl _)
               /-
                 🎉 no goals
               -/

