/--
A bimonoid object in a braided category `C` is a object that is simultaneously monoid and comonoid
objects, and structure morphisms of them satisfy appropriate consistency conditions.
-/
class Bimon_Class (M : C) extends Mon_Class M, Comon_Class M where
  /- For the names of the conditions below, the unprimed names are reserved for the version where
  the argument `M` is explicit. -/
  mul_comul' : μ[M] ≫ Δ[M] = (Δ[M] ⊗ Δ[M]) ≫ tensorμ M M M M ≫ (μ[M] ⊗ μ[M]) := by aesop_cat
  one_comul' : η[M] ≫ Δ[M] = η[M ⊗ M] := by aesop_cat
  mul_counit' : μ[M] ≫ ε[M] = ε[M ⊗ M] := by aesop_cat
  one_counit' : η[M] ≫ ε[M] = 𝟙 (𝟙_ C) := by aesop_cat


attribute [reassoc] mul_comul' one_comul' mul_counit' one_counit'


@[reassoc (attr := simp)]
theorem mul_comul (M : C) [Bimon_Class M] :
    μ[M] ≫ Δ[M] = (Δ[M] ⊗ Δ[M]) ≫ tensorμ M M M M ≫ (μ[M] ⊗ μ[M]) :=
  mul_comul'


@[reassoc (attr := simp)]
theorem one_comul (M : C) [Bimon_Class M] : η[M] ≫ Δ[M] = η[M ⊗ M] := one_comul'


@[reassoc (attr := simp)]
theorem mul_counit (M : C) [Bimon_Class M] : μ[M] ≫ ε[M] = ε[M ⊗ M] := mul_counit'


@[reassoc (attr := simp)]
theorem one_counit (M : C) [Bimon_Class M] : η[M] ≫ ε[M] = 𝟙 (𝟙_ C) := one_counit'


/-- The property that a morphism between bimonoid objects is a bimonoid morphism. -/
class IsBimon_Hom {M N : C} [Bimon_Class M] [Bimon_Class N] (f : M ⟶ N) extends
    IsMon_Hom f, IsComon_Hom f : Prop


/--
A bimonoid object in a braided category `C` is a comonoid object in the (monoidal)
category of monoid objects in `C`.
-/
def Bimon_ := Comon_ (Mon_ C)


instance : Category (Bimon_ C) := inferInstanceAs (Category (Comon_ (Mon_ C)))


@[ext] lemma ext {X Y : Bimon_ C} {f g : X ⟶ Y} (w : f.hom.hom = g.hom.hom) : f = g :=
  Comon_.Hom.ext (Mon_.Hom.ext w)


@[simp] theorem id_hom' (M : Bimon_ C) : Comon_.Hom.hom (𝟙 M) = 𝟙 M.X := rfl


@[simp]
theorem comp_hom' {M N K : Bimon_ C} (f : M ⟶ N) (g : N ⟶ K) : (f ≫ g).hom = f.hom ≫ g.hom :=
  rfl


/-- The forgetful functor from bimonoid objects to monoid objects. -/
abbrev toMon_ : Bimon_ C ⥤ Mon_ C := Comon_.forget (Mon_ C)


/-- The forgetful functor from bimonoid objects to the underlying category. -/
def forget : Bimon_ C ⥤ C := toMon_ C ⋙ Mon_.forget C


@[simp]
theorem toMon_forget : toMon_ C ⋙ Mon_.forget C = forget C := rfl


/-- The forgetful functor from bimonoid objects to comonoid objects. -/
@[simps!]
def toComon_ : Bimon_ C ⥤ Comon_ C := (Mon_.forget C).mapComon


@[simp]
theorem toComon_forget : toComon_ C ⋙ Comon_.forget C = forget C := rfl


/-- The object level part of the forward direction of `Comon_ (Mon_ C) ≌ Mon_ (Comon_ C)` -/
def toMon_Comon_obj (M : Bimon_ C) : Mon_ (Comon_ C) where
  X := (toComon_ C).obj M
  one := { hom := M.X.one }
  mul :=
  { hom := M.X.mul,
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      inst✝¹ : CategoryTheory.MonoidalCategory C
                      inst✝ : CategoryTheory.BraidedCategory C
                      M : Bimon_ C
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp M.X.mul ((Bimon_.toComon_ C).obj M).c …
                    -/
    hom_comul := by dsimp; simp [tensor_μ] }
                           /-
                             🎉 no goals
                           -/


attribute [simps] toMon_Comon_obj -- We add this after the fact to avoid a timeout.


/-- The forward direction of `Comon_ (Mon_ C) ≌ Mon_ (Comon_ C)` -/
@[simps]
def toMon_Comon_ : Bimon_ C ⥤ Mon_ (Comon_ C) where
  obj := toMon_Comon_obj C
  map f :=
  { hom := (toComon_ C).map f }


/-- The object level part of the backward direction of `Comon_ (Mon_ C) ≌ Mon_ (Comon_ C)` -/
@[simps]
def ofMon_Comon_obj (M : Mon_ (Comon_ C)) : Bimon_ C where
  X := (Comon_.forget C).mapMon.obj M
  counit := { hom := M.X.counit }
  comul :=
  { hom := M.X.comul,
                  /-
                    C : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} C
                    inst✝¹ : CategoryTheory.MonoidalCategory C
                    inst✝ : CategoryTheory.BraidedCategory C
                    M : Mon_ (Comon_ C)
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Comon_.forget C).mapMon.obj M).mul  …
                  -/
    mul_hom := by dsimp; simp [tensor_μ] }
                         /-
                           🎉 no goals
                         -/


/-- The backward direction of `Comon_ (Mon_ C) ≌ Mon_ (Comon_ C)` -/
@[simps]
def ofMon_Comon_ : Mon_ (Comon_ C) ⥤ Bimon_ C where
  obj := ofMon_Comon_obj C
  map f :=
  { hom := (Comon_.forget C).mapMon.map f }


/-- The equivalence `Comon_ (Mon_ C) ≌ Mon_ (Comon_ C)` -/
def equivMon_Comon_ : Bimon_ C ≌ Mon_ (Comon_ C) where
  functor := toMon_Comon_ C
  inverse := ofMon_Comon_ C
                                                         /-
                                                           C : Type u₁
                                                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                           inst✝¹ : CategoryTheory.MonoidalCategory C
                                                           inst✝ : CategoryTheory.BraidedCategory C
                                                           x✝ : Bimon_ C
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Bimon_ C …
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
  unitIso := NatIso.ofComponents (fun _ => Comon_.mkIso (Mon_.mkIso (Iso.refl _)))
             /-
               🎉 no goals
             -/
                                                         /-
                                                           C : Type u₁
                                                           inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                           inst✝¹ : CategoryTheory.MonoidalCategory C
                                                           inst✝ : CategoryTheory.BraidedCategory C
                                                           x✝ : Mon_ (Comon_ C)
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (((Bimon_.of …
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
  counitIso := NatIso.ofComponents (fun _ => Mon_.mkIso (Comon_.mkIso (Iso.refl _)))
               /-
                 🎉 no goals
               -/


/-- The trivial bimonoid object. -/
@[simps!]
def trivial : Bimon_ C := Comon_.trivial (Mon_ C)


/-- The bimonoid morphism from the trivial bimonoid to any bimonoid. -/
@[simps]
def trivialTo (A : Bimon_ C) : trivial C ⟶ A :=
  { hom := (default : Mon_.trivial C ⟶ A.X), }


@[deprecated (since := "2024-12-07")] alias trivial_to := trivialTo

@[deprecated (since := "2024-12-07")] alias trivial_to_hom := trivialTo_hom


/-- The bimonoid morphism from any bimonoid to the trivial bimonoid. -/
@[simps!]
def toTrivial (A : Bimon_ C) : A ⟶ trivial C :=
  (default : @Quiver.Hom (Comon_ (Mon_ C)) _ A (Comon_.trivial (Mon_ C)))


@[deprecated (since := "2024-12-07")] alias to_trivial := toTrivial

@[deprecated (since := "2024-12-07")] alias to_trivial_hom := toTrivial_hom


@[reassoc]
theorem one_comul (M : Bimon_ C) :
    M.X.one ≫ M.comul.hom = (λ_ _).inv ≫ (M.X.one ⊗ M.X.one) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.X.one M.comul.hom) (CategoryTheory. …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem mul_counit (M : Bimon_ C) :
    M.X.mul ≫ M.counit.hom = (M.counit.hom ⊗ M.counit.hom) ≫ (λ_ _).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.X.mul M.counit.hom) (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Compatibility of the monoid and comonoid structures, in terms of morphisms in `C`. -/
@[reassoc (attr := simp)] theorem compatibility (M : Bimon_ C) :
    (M.comul.hom ⊗ M.comul.hom) ≫
      (α_ _ _ (M.X.X ⊗ M.X.X)).hom ≫ M.X.X ◁ (α_ _ _ _).inv ≫
      M.X.X ◁ (β_ M.X.X M.X.X).hom ▷ M.X.X ≫
      M.X.X ◁ (α_ _ _ _).hom ≫ (α_ _ _ _).inv ≫
      (M.X.mul ⊗ M.X.mul) =
    M.X.mul ≫ M.comul.hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  have := (Mon_.Hom.mul_hom M.comul).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  simpa [-Mon_.Hom.mul_hom, tensorμ] using this
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)] theorem comul_counit_hom (M : Bimon_ C) :
    M.comul.hom ≫ (_ ◁ M.counit.hom) = (ρ_ _).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul.hom (CategoryTheory.MonoidalC …
  -/
  simpa [- Comon_.comul_counit] using congr_arg Mon_.Hom.hom M.comul_counit
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)] theorem counit_comul_hom (M : Bimon_ C) :
    M.comul.hom ≫ (M.counit.hom ▷ _) = (λ_ _).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul.hom (CategoryTheory.MonoidalC …
  -/
  simpa [- Comon_.counit_comul] using congr_arg Mon_.Hom.hom M.counit_comul
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)] theorem comul_assoc_hom (M : Bimon_ C) :
    M.comul.hom ≫ (M.X.X ◁ M.comul.hom) =
      M.comul.hom ≫ (M.comul.hom ▷ M.X.X) ≫ (α_ M.X.X M.X.X M.X.X).hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul.hom (CategoryTheory.MonoidalC …
  -/
  simpa [- Comon_.comul_assoc] using congr_arg Mon_.Hom.hom M.comul_assoc
  /-
    🎉 no goals
  -/


@[reassoc] theorem comul_assoc_flip_hom (M : Bimon_ C) :
    M.comul.hom ≫ (M.comul.hom ▷ M.X.X) =
      M.comul.hom ≫ (M.X.X ◁ M.comul.hom) ≫ (α_ M.X.X M.X.X M.X.X).inv := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M : Bimon_ C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.comul.hom (CategoryTheory.MonoidalC …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc] theorem hom_comul_hom {M N : Bimon_ C} (f : M ⟶ N) :
    f.hom.hom ≫ N.comul.hom = M.comul.hom ≫ (f.hom.hom ⊗ f.hom.hom) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Bimon_ C
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.hom N.comul.hom) (CategoryTheor …
  -/
  simpa [- Comon_.Hom.hom_comul] using congr_arg Mon_.Hom.hom f.hom_comul
  /-
    🎉 no goals
  -/


@[reassoc] theorem hom_counit_hom {M N : Bimon_ C} (f : M ⟶ N) :
    f.hom.hom ≫ N.counit.hom = M.counit.hom := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.MonoidalCategory C
    inst✝ : CategoryTheory.BraidedCategory C
    M N : Bimon_ C
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.hom.hom N.counit.hom) M.counit.hom
  -/
  simpa [- Comon_.Hom.hom_counit] using congr_arg Mon_.Hom.hom f.hom_counit
  /-
    🎉 no goals
  -/


