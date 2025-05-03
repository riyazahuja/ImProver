/-- An algebra of an endofunctor; `str` stands for "structure morphism" -/
structure Algebra (F : C ⥤ C) where
  /-- carrier of the algebra -/
  a : C
  /-- structure morphism of the algebra -/
  str : F.obj a ⟶ a


instance [Inhabited C] : Inhabited (Algebra (𝟭 C)) :=
  ⟨⟨default, 𝟙 _⟩⟩


/-- A morphism between algebras of endofunctor `F` -/
@[ext]
structure Hom (A₀ A₁ : Algebra F) where
  /-- underlying morphism between the carriers -/
  f : A₀.1 ⟶ A₁.1
  /-- compatibility condition -/
  h : F.map f ≫ A₁.str = A₀.str ≫ f := by aesop_cat


attribute [reassoc (attr := simp)] Hom.h


/-- The identity morphism of an algebra of endofunctor `F` -/
def id : Hom A A where f := 𝟙 _


instance : Inhabited (Hom A A) :=
  ⟨{ f := 𝟙 _ }⟩


/-- The composition of morphisms between algebras of endofunctor `F` -/
def comp (f : Hom A₀ A₁) (g : Hom A₁ A₂) : Hom A₀ A₂ where f := f.1 ≫ g.1


instance (F : C ⥤ C) : CategoryStruct (Algebra F) where
  Hom := Hom
  id := Hom.id
  comp := @Hom.comp _ _ _


@[ext]
lemma ext {A B : Algebra F} {f g : A ⟶ B} (w : f.f = g.f := by aesop_cat) : f = g :=
  Hom.ext w


@[simp]
theorem id_eq_id : Algebra.Hom.id A = 𝟙 A :=
  rfl


@[simp]
theorem id_f : (𝟙 _ : A ⟶ A).1 = 𝟙 A.1 :=
  rfl


@[simp]
theorem comp_eq_comp : Algebra.Hom.comp f g = f ≫ g :=
  rfl


@[simp]
theorem comp_f : (f ≫ g).1 = f.1 ≫ g.1 :=
  rfl


/-- Algebras of an endofunctor `F` form a category -/
instance (F : C ⥤ C) : Category (Algebra F) := { }


/-- To construct an isomorphism of algebras, it suffices to give an isomorphism of the As which
commutes with the structure morphisms.
-/
@[simps!]
def isoMk (h : A₀.1 ≅ A₁.1) (w : F.map h.hom ≫ A₁.str = A₀.str ≫ h.hom := by aesop_cat) :
    A₀ ≅ A₁ where
  hom := { f := h.hom }
  inv :=
    { f := h.inv
      h := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₀ A₁
          g : Quiver.Hom A₁ A₂
          h : CategoryTheory.Iso A₀.a A₁.a
          w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map h.hom) A₁.str) (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map h.inv) A₀.str) (CategoryTheory …
        -/
        rw [h.eq_comp_inv, Category.assoc, ← w, ← Functor.map_comp_assoc]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₀ A₁
          g : Quiver.Hom A₁ A₂
          h : CategoryTheory.Iso A₀.a A₁.a
          w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map h.hom) A₁.str) (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The forgetful functor from the category of algebras, forgetting the algebraic structure. -/
@[simps]
def forget (F : C ⥤ C) : Algebra F ⥤ C where
  obj A := A.1
  map := Hom.f


/-- An algebra morphism with an underlying isomorphism hom in `C` is an algebra isomorphism. -/
theorem iso_of_iso (f : A₀ ⟶ A₁) [IsIso f.1] : IsIso f :=
  ⟨⟨{ f := inv f.1
      h := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          A₀ A₁ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₀ A₁
          inst✝ : CategoryTheory.IsIso f.f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.inv f.f)) A₀.s …
        -/
        rw [IsIso.eq_comp_inv f.1, Category.assoc, ← f.h]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          A₀ A₁ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₀ A₁
          inst✝ : CategoryTheory.IsIso f.f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.inv f.f)) (Cat …
        -/
        /-
          🎉 no goals
        -/
                   /-
                     🎉 no goals
                   -/
        simp }, by aesop_cat, by aesop_cat⟩⟩
                                 /-
                                   🎉 no goals
                                 -/


instance forget_reflects_iso : (forget F).ReflectsIsomorphisms where reflects := iso_of_iso


instance forget_faithful : (forget F).Faithful := { }


/-- An algebra morphism with an underlying epimorphism hom in `C` is an algebra epimorphism. -/
theorem epi_of_epi {X Y : Algebra F} (f : X ⟶ Y) [h : Epi f.1] : Epi f :=
  (forget F).epi_of_epi_map h


/-- An algebra morphism with an underlying monomorphism hom in `C` is an algebra monomorphism. -/
theorem mono_of_mono {X Y : Algebra F} (f : X ⟶ Y) [h : Mono f.1] : Mono f :=
  (forget F).mono_of_mono_map h


/-- From a natural transformation `α : G → F` we get a functor from
algebras of `F` to algebras of `G`.
-/
@[simps]
def functorOfNatTrans {F G : C ⥤ C} (α : G ⟶ F) : Algebra F ⥤ Algebra G where
  obj A :=
    { a := A.1
      str := α.app _ ≫ A.str }
  map f := { f := f.1 }


/-- The identity transformation induces the identity endofunctor on the category of algebras. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransId : functorOfNatTrans (𝟙 F) ≅ 𝟭 _ :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F : CategoryTheory.Functor C C
                                 A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                                 f : Quiver.Hom A₀ A₁
                                 g : Quiver.Hom A₁ A₂
                                 X : CategoryTheory.Endofunctor.Algebra F
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Iso.refl ((Cat …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A composition of natural transformations gives the composition of corresponding functors. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransComp {F₀ F₁ F₂ : C ⥤ C} (α : F₀ ⟶ F₁) (β : F₁ ⟶ F₂) :
    functorOfNatTrans (α ≫ β) ≅ functorOfNatTrans β ⋙ functorOfNatTrans α :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F : CategoryTheory.Functor C C
                                 A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                                 f : Quiver.Hom A₀ A₁
                                 g : Quiver.Hom A₁ A₂
                                 F₀ F₁ F₂ : CategoryTheory.Functor C C
                                 α : Quiver.Hom F₀ F₁
                                 β : Quiver.Hom F₁ F₂
                                 X : CategoryTheory.Endofunctor.Algebra F₂
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₀.map (CategoryTheory.Iso.refl ((Ca …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/--
If `α` and `β` are two equal natural transformations, then the functors of algebras induced by them
are isomorphic.
We define it like this as opposed to using `eq_to_iso` so that the components are nicer to prove
lemmas about.
-/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransEq {F G : C ⥤ C} {α β : F ⟶ G} (h : α = β) :
    functorOfNatTrans α ≅ functorOfNatTrans β :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F✝ : CategoryTheory.Functor C C
                                 A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F✝
                                 f : Quiver.Hom A₀ A₁
                                 g : Quiver.Hom A₁ A₂
                                 F G : CategoryTheory.Functor C C
                                 α β : Quiver.Hom F G
                                 h : Eq α β
                                 X : CategoryTheory.Endofunctor.Algebra G
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Iso.refl ((Cat …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Naturally isomorphic endofunctors give equivalent categories of algebras.
Furthermore, they are equivalent as categories over `C`, that is,
we have `equiv_of_nat_iso h ⋙ forget = forget`.
-/
@[simps]
def equivOfNatIso {F G : C ⥤ C} (α : F ≅ G) : Algebra F ≌ Algebra G where
  functor := functorOfNatTrans α.inv
  inverse := functorOfNatTrans α.hom
                                                                 /-
                                                                   C : Type u
                                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                                   F✝ : CategoryTheory.Functor C C
                                                                   A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F✝
                                                                   f : Quiver.Hom A₀ A₁
                                                                   g : Quiver.Hom A₁ A₂
                                                                   F G : CategoryTheory.Functor C C
                                                                   α : CategoryTheory.Iso F G
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.id F) (CategoryTheory.CategoryStruct.comp  …
                                                                 -/
  unitIso := functorOfNatTransId.symm ≪≫ functorOfNatTransEq (by simp) ≪≫ functorOfNatTransComp _ _
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  counitIso :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                                  F✝ : CategoryTheory.Functor C C
                                                                  A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F✝
                                                                  f : Quiver.Hom A₀ A₁
                                                                  g : Quiver.Hom A₁ A₂
                                                                  F G : CategoryTheory.Functor C C
                                                                  α : CategoryTheory.Iso F G
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp α.inv α.hom) (CategoryTheory.Category …
                                                                -/
    (functorOfNatTransComp _ _).symm ≪≫ functorOfNatTransEq (by simp) ≪≫ functorOfNatTransId
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The inverse of the structure map of an initial algebra -/
@[simp]
def strInv : A.1 ⟶ F.obj A.1 :=
  (h.to ⟨F.obj A.a, F.map A.str⟩).f


theorem left_inv' :
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            F : CategoryTheory.Functor C C
                            A A₀ A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
                            f : Quiver.Hom A₀ A₁
                            g : Quiver.Hom A₁ A₂
                            h : CategoryTheory.Limits.IsInitial A
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
                          -/
    ⟨strInv h ≫ A.str, by rw [← Category.assoc, F.map_comp, strInv, ← Hom.h]⟩ = 𝟙 A :=
                          /-
                            🎉 no goals
                          -/
  Limits.IsInitial.hom_ext h _ (𝟙 A)


theorem left_inv : strInv h ≫ A.str = 𝟙 _ :=
  congr_arg Hom.f (left_inv' h)


theorem right_inv : A.str ≫ strInv h = 𝟙 _ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C C
    A : CategoryTheory.Endofunctor.Algebra F
    h : CategoryTheory.Limits.IsInitial A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp A.str (CategoryTheory.Endofunctor.Alg …
  -/
  rw [strInv, ← (h.to ⟨F.obj A.1, F.map A.str⟩).h, ← F.map_id, ← F.map_comp]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C C
    A : CategoryTheory.Endofunctor.Algebra F
    h : CategoryTheory.Limits.IsInitial A
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (h.to { a := F.obj A.a, str := …
  -/
  congr
  /-
    case e_a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C C
    A : CategoryTheory.Endofunctor.Algebra F
    h : CategoryTheory.Limits.IsInitial A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.to { a := F.obj A.a, str := F.map  …
  -/
  exact left_inv h
  /-
    🎉 no goals
  -/


/-- The structure map of the initial algebra is an isomorphism,
hence endofunctors preserve their initial algebras
-/
theorem str_isIso (h : Limits.IsInitial A) : IsIso A.str :=
  { out := ⟨strInv h, right_inv _, left_inv _⟩ }


/-- A coalgebra of an endofunctor; `str` stands for "structure morphism" -/
structure Coalgebra (F : C ⥤ C) where
  /-- carrier of the coalgebra -/
  V : C
  /-- structure morphism of the coalgebra -/
  str : V ⟶ F.obj V


instance [Inhabited C] : Inhabited (Coalgebra (𝟭 C)) :=
  ⟨⟨default, 𝟙 _⟩⟩


/-- A morphism between coalgebras of an endofunctor `F` -/
@[ext]
structure Hom (V₀ V₁ : Coalgebra F) where
  /-- underlying morphism between two carriers -/
  f : V₀.1 ⟶ V₁.1
  /-- compatibility condition -/
  h : V₀.str ≫ F.map f = f ≫ V₁.str := by aesop_cat


/-- The identity morphism of an algebra of endofunctor `F` -/
def id : Hom V V where f := 𝟙 _


instance : Inhabited (Hom V V) :=
  ⟨{ f := 𝟙 _ }⟩


/-- The composition of morphisms between algebras of endofunctor `F` -/
def comp (f : Hom V₀ V₁) (g : Hom V₁ V₂) : Hom V₀ V₂ where f := f.1 ≫ g.1


instance (F : C ⥤ C) : CategoryStruct (Coalgebra F) where
  Hom := Hom
  id := Hom.id
  comp := @Hom.comp _ _ _


@[ext]
lemma ext {A B : Coalgebra F} {f g : A ⟶ B} (w : f.f = g.f := by aesop_cat) : f = g :=
  Hom.ext w


@[simp]
theorem id_eq_id : Coalgebra.Hom.id V = 𝟙 V :=
  rfl


@[simp]
theorem id_f : (𝟙 _ : V ⟶ V).1 = 𝟙 V.1 :=
  rfl


@[simp]
theorem comp_eq_comp : Coalgebra.Hom.comp f g = f ≫ g :=
  rfl


/-- Coalgebras of an endofunctor `F` form a category -/
instance (F : C ⥤ C) : Category (Coalgebra F) := { }


/-- To construct an isomorphism of coalgebras, it suffices to give an isomorphism of the Vs which
commutes with the structure morphisms.
-/
@[simps]
def isoMk (h : V₀.1 ≅ V₁.1) (w : V₀.str ≫ F.map h.hom = h.hom ≫ V₁.str := by aesop_cat) :
    V₀ ≅ V₁ where
  hom := { f := h.hom }
  inv :=
    { f := h.inv
      h := by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F
          f : Quiver.Hom V₀ V₁
          g : Quiver.Hom V₁ V₂
          h : CategoryTheory.Iso V₀.V V₁.V
          w : autoParam (Eq (CategoryTheory.CategoryStruct.comp V₀.str (F.map h.hom)) (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp V₁.str (F.map h.inv)) (CategoryTheory …
        -/
        rw [h.eq_inv_comp, ← Category.assoc, ← w, Category.assoc, ← F.map_comp]
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F
          f : Quiver.Hom V₀ V₁
          g : Quiver.Hom V₁ V₂
          h : CategoryTheory.Iso V₀.V V₁.V
          w : autoParam (Eq (CategoryTheory.CategoryStruct.comp V₀.str (F.map h.hom)) (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp V₀.str (F.map (CategoryTheory.Categor …
        -/
        simp only [Iso.hom_inv_id, Functor.map_id, Category.comp_id] }
        /-
          🎉 no goals
        -/


/-- The forgetful functor from the category of coalgebras, forgetting the coalgebraic structure. -/
@[simps]
def forget (F : C ⥤ C) : Coalgebra F ⥤ C where
  obj A := A.1
  map f := f.1


/-- A coalgebra morphism with an underlying isomorphism hom in `C` is a coalgebra isomorphism. -/
theorem iso_of_iso (f : V₀ ⟶ V₁) [IsIso f.1] : IsIso f :=
  ⟨⟨{ f := inv f.1
      h := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          V₀ V₁ : CategoryTheory.Endofunctor.Coalgebra F
          f : Quiver.Hom V₀ V₁
          inst✝ : CategoryTheory.IsIso f.f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp V₁.str (F.map (CategoryTheory.inv f.f …
        -/
        rw [IsIso.eq_inv_comp f.1, ← Category.assoc, ← f.h, Category.assoc]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          F : CategoryTheory.Functor C C
          V₀ V₁ : CategoryTheory.Endofunctor.Coalgebra F
          f : Quiver.Hom V₀ V₁
          inst✝ : CategoryTheory.IsIso f.f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp V₀.str (CategoryTheory.CategoryStruct …
        -/
        /-
          🎉 no goals
        -/
                   /-
                     🎉 no goals
                   -/
        simp }, by aesop_cat, by aesop_cat⟩⟩
                                 /-
                                   🎉 no goals
                                 -/


/-- An algebra morphism with an underlying epimorphism hom in `C` is an algebra epimorphism. -/
theorem epi_of_epi {X Y : Coalgebra F} (f : X ⟶ Y) [h : Epi f.1] : Epi f :=
  (forget F).epi_of_epi_map h


/-- An algebra morphism with an underlying monomorphism hom in `C` is an algebra monomorphism. -/
theorem mono_of_mono {X Y : Coalgebra F} (f : X ⟶ Y) [h : Mono f.1] : Mono f :=
  (forget F).mono_of_mono_map h


/-- From a natural transformation `α : F → G` we get a functor from
coalgebras of `F` to coalgebras of `G`.
-/
@[simps]
def functorOfNatTrans {F G : C ⥤ C} (α : F ⟶ G) : Coalgebra F ⥤ Coalgebra G where
  obj V :=
    { V := V.1
      str := V.str ≫ α.app V.1 }
  map f :=
    { f := f.1
              /-
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F✝ : CategoryTheory.Functor C C
                V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F✝
                f✝ : Quiver.Hom V₀ V₁
                g : Quiver.Hom V₁ V₂
                F G : CategoryTheory.Functor C C
                α : Quiver.Hom F G
                X✝ Y✝ : CategoryTheory.Endofunctor.Coalgebra F
                f : Quiver.Hom X✝ Y✝
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun V => { V := V.V, str := Categor …
              -/
      h := by rw [Category.assoc, ← α.naturality, ← Category.assoc, f.h, Category.assoc] }
              /-
                🎉 no goals
              -/


/-- The identity transformation induces the identity endofunctor on the category of coalgebras. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransId : functorOfNatTrans (𝟙 F) ≅ 𝟭 _ :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F : CategoryTheory.Functor C C
                                 V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F
                                 f : Quiver.Hom V₀ V₁
                                 g : Quiver.Hom V₁ V₂
                                 X : CategoryTheory.Endofunctor.Coalgebra F
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunctor.Coalgebr …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- A composition of natural transformations gives the composition of corresponding functors. -/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransComp {F₀ F₁ F₂ : C ⥤ C} (α : F₀ ⟶ F₁) (β : F₁ ⟶ F₂) :
    functorOfNatTrans (α ≫ β) ≅ functorOfNatTrans α ⋙ functorOfNatTrans β :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F : CategoryTheory.Functor C C
                                 V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F
                                 f : Quiver.Hom V₀ V₁
                                 g : Quiver.Hom V₁ V₂
                                 F₀ F₁ F₂ : CategoryTheory.Functor C C
                                 α : Quiver.Hom F₀ F₁
                                 β : Quiver.Hom F₁ F₂
                                 X : CategoryTheory.Endofunctor.Coalgebra F₀
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunctor.Coalgebr …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- If `α` and `β` are two equal natural transformations, then the functors of coalgebras induced by
them are isomorphic.
We define it like this as opposed to using `eq_to_iso` so that the components are nicer to prove
lemmas about.
-/
-- Porting note: removed @[simps (config := { rhsMd := semireducible })] and replaced with
@[simps!]
def functorOfNatTransEq {F G : C ⥤ C} {α β : F ⟶ G} (h : α = β) :
    functorOfNatTrans α ≅ functorOfNatTrans β :=
                               /-
                                 C : Type u
                                 inst✝ : CategoryTheory.Category.{v, u} C
                                 F✝ : CategoryTheory.Functor C C
                                 V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F✝
                                 f : Quiver.Hom V₀ V₁
                                 g : Quiver.Hom V₁ V₂
                                 F G : CategoryTheory.Functor C C
                                 α β : Quiver.Hom F G
                                 h : Eq α β
                                 X : CategoryTheory.Endofunctor.Coalgebra F
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunctor.Coalgebr …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => isoMk (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Naturally isomorphic endofunctors give equivalent categories of coalgebras.
Furthermore, they are equivalent as categories over `C`, that is,
we have `equiv_of_nat_iso h ⋙ forget = forget`.
-/
@[simps]
def equivOfNatIso {F G : C ⥤ C} (α : F ≅ G) : Coalgebra F ≌ Coalgebra G where
  functor := functorOfNatTrans α.hom
  inverse := functorOfNatTrans α.inv
                                                                 /-
                                                                   C : Type u
                                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                                   F✝ : CategoryTheory.Functor C C
                                                                   V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F✝
                                                                   f : Quiver.Hom V₀ V₁
                                                                   g : Quiver.Hom V₁ V₂
                                                                   F G : CategoryTheory.Functor C C
                                                                   α : CategoryTheory.Iso F G
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.id F) (CategoryTheory.CategoryStruct.comp  …
                                                                 -/
  unitIso := functorOfNatTransId.symm ≪≫ functorOfNatTransEq (by simp) ≪≫ functorOfNatTransComp _ _
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  counitIso :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                                  F✝ : CategoryTheory.Functor C C
                                                                  V V₀ V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra F✝
                                                                  f : Quiver.Hom V₀ V₁
                                                                  g : Quiver.Hom V₁ V₂
                                                                  F G : CategoryTheory.Functor C C
                                                                  α : CategoryTheory.Iso F G
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp α.inv α.hom) (CategoryTheory.Category …
                                                                -/
    (functorOfNatTransComp _ _).symm ≪≫ functorOfNatTransEq (by simp) ≪≫ functorOfNatTransId
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem Algebra.homEquiv_naturality_str (adj : F ⊣ G) (A₁ A₂ : Algebra F) (f : A₁ ⟶ A₂) :
    (adj.homEquiv A₁.a A₁.a) A₁.str ≫ G.map f.f = f.f ≫ (adj.homEquiv A₂.a A₂.a) A₂.str := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F G : CategoryTheory.Functor C C
    adj : CategoryTheory.Adjunction F G
    A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
    f : Quiver.Hom A₁ A₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj.homEquiv A₁.a A₁.a) A₁.str) (G. …
  -/
  rw [← Adjunction.homEquiv_naturality_right, ← Adjunction.homEquiv_naturality_left, f.h]
  /-
    🎉 no goals
  -/


theorem Coalgebra.homEquiv_naturality_str_symm (adj : F ⊣ G) (V₁ V₂ : Coalgebra G) (f : V₁ ⟶ V₂) :
    F.map f.f ≫ (adj.homEquiv V₂.V V₂.V).symm V₂.str =
    (adj.homEquiv V₁.V V₁.V).symm V₁.str ≫ f.f := by
  rw [← Adjunction.homEquiv_naturality_left_symm, ← Adjunction.homEquiv_naturality_right_symm,
    f.h]


/-- Given an adjunction `F ⊣ G`, the functor that associates to an algebra over `F` a
coalgebra over `G` defined via adjunction applied to the structure map. -/
def Algebra.toCoalgebraOf (adj : F ⊣ G) : Algebra F ⥤ Coalgebra G where
  obj A :=
    { V := A.1
      str := (adj.homEquiv A.1 A.1).toFun A.2 }
  map f :=
    { f := f.1
      h := Algebra.homEquiv_naturality_str adj _ _ f }


/-- Given an adjunction `F ⊣ G`, the functor that associates to a coalgebra over `G` an algebra over
`F` defined via adjunction applied to the structure map. -/
def Coalgebra.toAlgebraOf (adj : F ⊣ G) : Coalgebra G ⥤ Algebra F where
  obj V :=
    { a := V.1
      str := (adj.homEquiv V.1 V.1).invFun V.2 }
  map f :=
    { f := f.1
      h := Coalgebra.homEquiv_naturality_str_symm adj _ _ f }


/-- Given an adjunction, assigning to an algebra over the left adjoint a coalgebra over its right
adjoint and going back is isomorphic to the identity functor. -/
def AlgCoalgEquiv.unitIso (adj : F ⊣ G) :
    𝟭 (Algebra F) ≅ Algebra.toCoalgebraOf adj ⋙ Coalgebra.toAlgebraOf adj where
  hom :=
    { app := fun A =>
        { f := 𝟙 A.1
          h := by
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              A : CategoryTheory.Endofunctor.Algebra F
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
            -/
            erw [F.map_id, Category.id_comp, Category.comp_id]
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              A : CategoryTheory.Endofunctor.Algebra F
              ⊢ Eq (((CategoryTheory.Endofunctor.Adjunction.Algebra.toCoalgebraOf adj).comp  …
            -/
            apply (adj.homEquiv _ _).left_inv A.str } }
            /-
              🎉 no goals
            -/
  inv :=
    { app := fun A =>
        { f := 𝟙 A.1
          h := by
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              A : CategoryTheory.Endofunctor.Algebra F
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
            -/
            erw [F.map_id, Category.id_comp, Category.comp_id]
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              A : CategoryTheory.Endofunctor.Algebra F
              ⊢ Eq ((CategoryTheory.Functor.id (CategoryTheory.Endofunctor.Algebra F)).obj A …
            -/
            apply ((adj.homEquiv _ _).left_inv A.str).symm }
            /-
              🎉 no goals
            -/
      naturality := fun A₁ A₂ f => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₁ A₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Endofunctor.Adjunct …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₁ A₂
          ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Endofunc …
        -/
        dsimp
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₁ A₂
          ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunct …
        -/
        erw [Category.comp_id, Category.id_comp]
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          A₁ A₂ : CategoryTheory.Endofunctor.Algebra F
          f : Quiver.Hom A₁ A₂
          ⊢ autoParam (Eq ((CategoryTheory.Endofunctor.Adjunction.Coalgebra.toAlgebraOf  …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- Given an adjunction, assigning to a coalgebra over the right adjoint an algebra over the left
adjoint and going back is isomorphic to the identity functor. -/
def AlgCoalgEquiv.counitIso (adj : F ⊣ G) :
    Coalgebra.toAlgebraOf adj ⋙ Algebra.toCoalgebraOf adj ≅ 𝟭 (Coalgebra G) where
  hom :=
    { app := fun V =>
        { f := 𝟙 V.1
          h := by
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Endofunctor.Adjunct …
            -/
            dsimp
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunctor.Adjuncti …
            -/
            erw [G.map_id, Category.id_comp, Category.comp_id]
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq ((CategoryTheory.Endofunctor.Adjunction.Algebra.toCoalgebraOf adj).obj (( …
            -/
            apply (adj.homEquiv _ _).right_inv V.str }
            /-
              🎉 no goals
            -/
      naturality := fun V₁ V₂ f => by
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra G
          f : Quiver.Hom V₁ V₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Endofunctor.Adjunct …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra G
          f : Quiver.Hom V₁ V₂
          ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Endofunc …
        -/
        dsimp
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra G
          f : Quiver.Hom V₁ V₂
          ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunct …
        -/
        erw [Category.comp_id, Category.id_comp]
        /-
          case w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F G : CategoryTheory.Functor C C
          adj : CategoryTheory.Adjunction F G
          V₁ V₂ : CategoryTheory.Endofunctor.Coalgebra G
          f : Quiver.Hom V₁ V₂
          ⊢ autoParam (Eq ((CategoryTheory.Endofunctor.Adjunction.Algebra.toCoalgebraOf  …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  inv :=
    { app := fun V =>
        { f := 𝟙 V.1
          h := by
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
            -/
            dsimp
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq (CategoryTheory.CategoryStruct.comp V.str (G.map (CategoryTheory.Category …
            -/
            rw [G.map_id, Category.comp_id, Category.id_comp]
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F G : CategoryTheory.Functor C C
              adj : CategoryTheory.Adjunction F G
              V : CategoryTheory.Endofunctor.Coalgebra G
              ⊢ Eq V.str ((CategoryTheory.Endofunctor.Adjunction.Algebra.toCoalgebraOf adj). …
            -/
            apply ((adj.homEquiv _ _).right_inv V.str).symm } }
            /-
              🎉 no goals
            -/


/-- If `F` is left adjoint to `G`, then the category of algebras over `F` is equivalent to the
category of coalgebras over `G`. -/
def algebraCoalgebraEquiv (adj : F ⊣ G) : Algebra F ≌ Coalgebra G where
  functor := Algebra.toCoalgebraOf adj
  inverse := Coalgebra.toAlgebraOf adj
  unitIso := AlgCoalgEquiv.unitIso adj
  counitIso := AlgCoalgEquiv.counitIso adj
  functor_unitIso_comp A := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C C
      adj : CategoryTheory.Adjunction F G
      A : CategoryTheory.Endofunctor.Algebra F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunctor.Adjuncti …
    -/
    ext
    -- Porting note: why doesn't `simp` work here?
    /-
      case w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F G : CategoryTheory.Functor C C
      adj : CategoryTheory.Adjunction F G
      A : CategoryTheory.Endofunctor.Algebra F
      ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Endofunct …
    -/
    exact Category.comp_id _
    /-
      🎉 no goals
    -/


