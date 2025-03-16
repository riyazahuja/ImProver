/-- The data of a monad on C consists of an endofunctor T together with natural transformations
η : 𝟭 C ⟶ T and μ : T ⋙ T ⟶ T satisfying three equations:
- T μ_X ≫ μ_X = μ_(TX) ≫ μ_X (associativity)
- η_(TX) ≫ μ_X = 1_X (left unit)
- Tη_X ≫ μ_X = 1_X (right unit)
-/
structure Monad extends C ⥤ C where
  /-- The unit for the monad. -/
  η : 𝟭 _ ⟶ toFunctor
  /-- The multiplication for the monad. -/
  μ : toFunctor ⋙ toFunctor ⟶ toFunctor
  assoc : ∀ X, toFunctor.map (NatTrans.app μ X) ≫ μ.app _ = μ.app _ ≫ μ.app _ := by aesop_cat
  left_unit : ∀ X : C, η.app (toFunctor.obj X) ≫ μ.app _ = 𝟙 _ := by aesop_cat
  right_unit : ∀ X : C, toFunctor.map (η.app X) ≫ μ.app _ = 𝟙 _ := by aesop_cat


/-- The data of a comonad on C consists of an endofunctor G together with natural transformations
ε : G ⟶ 𝟭 C and δ : G ⟶ G ⋙ G satisfying three equations:
- δ_X ≫ G δ_X = δ_X ≫ δ_(GX) (coassociativity)
- δ_X ≫ ε_(GX) = 1_X (left counit)
- δ_X ≫ G ε_X = 1_X (right counit)
-/
structure Comonad extends C ⥤ C where
  /-- The counit for the comonad. -/
  ε : toFunctor ⟶ 𝟭 _
  /-- The comultiplication for the comonad. -/
  δ : toFunctor ⟶ toFunctor ⋙ toFunctor
  coassoc : ∀ X, NatTrans.app δ _ ≫ toFunctor.map (δ.app X) = δ.app _ ≫ δ.app _ := by
    aesop_cat
  left_counit : ∀ X : C, δ.app X ≫ ε.app (toFunctor.obj X) = 𝟙 _ := by aesop_cat
  right_counit : ∀ X : C, δ.app X ≫ toFunctor.map (ε.app X) = 𝟙 _ := by aesop_cat


instance coeMonad : Coe (Monad C) (C ⥤ C) :=
  ⟨fun T => T.toFunctor⟩


instance coeComonad : Coe (Comonad C) (C ⥤ C) :=
  ⟨fun G => G.toFunctor⟩

-- Porting note: these lemmas are syntactic tautologies
--@[simp]
--theorem monad_toFunctor_eq_coe : T.toFunctor = T :=
--  rfl
--
--@[simp]
--theorem comonad_toFunctor_eq_coe : G.toFunctor = G :=
--  rfl


attribute [reassoc (attr := simp)] Monad.left_unit Monad.right_unit

attribute [reassoc (attr := simp)] Comonad.coassoc Comonad.left_counit Comonad.right_counit


/-- A morphism of monads is a natural transformation compatible with η and μ. -/
@[ext]
structure MonadHom (T₁ T₂ : Monad C) extends NatTrans (T₁ : C ⥤ C) T₂ where
  app_η : ∀ X, T₁.η.app X ≫ app X = T₂.η.app X := by aesop_cat
  app_μ : ∀ X, T₁.μ.app X ≫ app X = (T₁.map (app X) ≫ app _) ≫ T₂.μ.app X := by
    aesop_cat


/-- A morphism of comonads is a natural transformation compatible with ε and δ. -/
@[ext]
structure ComonadHom (M N : Comonad C) extends NatTrans (M : C ⥤ C) N where
  app_ε : ∀ X, app X ≫ N.ε.app X = M.ε.app X := by aesop_cat
  app_δ : ∀ X, app X ≫ N.δ.app X = M.δ.app X ≫ app _ ≫ N.map (app X) := by aesop_cat


attribute [reassoc (attr := simp)] MonadHom.app_η MonadHom.app_μ

attribute [reassoc (attr := simp)] ComonadHom.app_ε ComonadHom.app_δ


instance : Quiver (Monad C) where
  Hom := MonadHom


instance : Quiver (Comonad C) where
  Hom := ComonadHom

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[ext]
lemma MonadHom.ext' {T₁ T₂ : Monad C} (f g : T₁ ⟶ T₂) (h : f.app = g.app) : f = g :=
  MonadHom.ext h

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[ext]
lemma ComonadHom.ext' {T₁ T₂ : Comonad C} (f g : T₁ ⟶ T₂) (h : f.app = g.app) : f = g :=
  ComonadHom.ext h


instance : Category (Monad C) where
  id M := { toNatTrans := 𝟙 (M : C ⥤ C) }
  comp f g :=
    { toNatTrans :=
        { app := fun X => f.app X ≫ g.app X
                                        /-
                                          C : Type u₁
                                          inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                          T : CategoryTheory.Monad C
                                          G : CategoryTheory.Comonad C
                                          X✝ Y✝ Z✝ : CategoryTheory.Monad C
                                          f : Quiver.Hom X✝ Y✝
                                          g : Quiver.Hom Y✝ Z✝
                                          X Y : C
                                          h : Quiver.Hom X Y
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.map h) ((fun X => CategoryTheory. …
                                        -/
          naturality := fun X Y h => by rw [assoc, f.1.naturality_assoc, g.1.naturality] } }
                                        /-
                                          🎉 no goals
                                        -/
  -- `aesop_cat` can fill in these proofs, but is unfortunately slightly slow.
                                /-
                                  C : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                  T : CategoryTheory.Monad C
                                  G : CategoryTheory.Comonad C
                                  X✝ Y✝ : CategoryTheory.Monad C
                                  x✝ : Quiver.Hom X✝ Y✝
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                                -/
  id_comp _ := MonadHom.ext (by funext; simp only [NatTrans.id_app, id_comp])
                                        /-
                                          🎉 no goals
                                        -/
                                /-
                                  C : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                  T : CategoryTheory.Monad C
                                  G : CategoryTheory.Comonad C
                                  X✝ Y✝ : CategoryTheory.Monad C
                                  x✝ : Quiver.Hom X✝ Y✝
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.CategoryStruct.id  …
                                -/
  comp_id _ := MonadHom.ext (by funext; simp only [NatTrans.id_app, comp_id])
                                        /-
                                          🎉 no goals
                                        -/
                                  /-
                                    C : Type u₁
                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                    T : CategoryTheory.Monad C
                                    G : CategoryTheory.Comonad C
                                    W✝ X✝ Y✝ Z✝ : CategoryTheory.Monad C
                                    x✝² : Quiver.Hom W✝ X✝
                                    x✝¹ : Quiver.Hom X✝ Y✝
                                    x✝ : Quiver.Hom Y✝ Z✝
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
                                  -/
  assoc _ _ _ := MonadHom.ext (by funext; simp only [assoc])
                                          /-
                                            🎉 no goals
                                          -/


instance : Category (Comonad C) where
  id M := { toNatTrans := 𝟙 (M : C ⥤ C) }
  comp f g :=
    { toNatTrans :=
        { app := fun X => f.app X ≫ g.app X
                                        /-
                                          C : Type u₁
                                          inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                          T : CategoryTheory.Monad C
                                          G X✝ Y✝ Z✝ : CategoryTheory.Comonad C
                                          f : Quiver.Hom X✝ Y✝
                                          g : Quiver.Hom Y✝ Z✝
                                          X Y : C
                                          h : Quiver.Hom X Y
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X✝.map h) ((fun X => CategoryTheory. …
                                        -/
          naturality := fun X Y h => by rw [assoc, f.1.naturality_assoc, g.1.naturality] } }
                                        /-
                                          🎉 no goals
                                        -/
  -- `aesop_cat` can fill in these proofs, but is unfortunately slightly slow.
                                  /-
                                    C : Type u₁
                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                    T : CategoryTheory.Monad C
                                    G X✝ Y✝ : CategoryTheory.Comonad C
                                    x✝ : Quiver.Hom X✝ Y✝
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                                  -/
  id_comp _ := ComonadHom.ext (by funext; simp only [NatTrans.id_app, id_comp])
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    C : Type u₁
                                    inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                    T : CategoryTheory.Monad C
                                    G X✝ Y✝ : CategoryTheory.Comonad C
                                    x✝ : Quiver.Hom X✝ Y✝
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.CategoryStruct.id  …
                                  -/
  comp_id _ := ComonadHom.ext (by funext; simp only [NatTrans.id_app, comp_id])
                                          /-
                                            🎉 no goals
                                          -/
                                    /-
                                      C : Type u₁
                                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                      T : CategoryTheory.Monad C
                                      G W✝ X✝ Y✝ Z✝ : CategoryTheory.Comonad C
                                      x✝² : Quiver.Hom W✝ X✝
                                      x✝¹ : Quiver.Hom X✝ Y✝
                                      x✝ : Quiver.Hom Y✝ Z✝
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
                                    -/
  assoc _ _ _ := ComonadHom.ext (by funext; simp only [assoc])
                                            /-
                                              🎉 no goals
                                            -/


instance {T : Monad C} : Inhabited (MonadHom T T) :=
  ⟨𝟙 T⟩


@[simp]
theorem MonadHom.id_toNatTrans (T : Monad C) : (𝟙 T : T ⟶ T).toNatTrans = 𝟙 (T : C ⥤ C) :=
  rfl


@[simp]
theorem MonadHom.comp_toNatTrans {T₁ T₂ T₃ : Monad C} (f : T₁ ⟶ T₂) (g : T₂ ⟶ T₃) :
    (f ≫ g).toNatTrans = ((f.toNatTrans : _ ⟶ (T₂ : C ⥤ C)) ≫ g.toNatTrans : (T₁ : C ⥤ C) ⟶ T₃) :=
  rfl


instance {G : Comonad C} : Inhabited (ComonadHom G G) :=
  ⟨𝟙 G⟩


@[simp]
theorem ComonadHom.id_toNatTrans (T : Comonad C) : (𝟙 T : T ⟶ T).toNatTrans = 𝟙 (T : C ⥤ C) :=
  rfl


@[simp]
theorem comp_toNatTrans {T₁ T₂ T₃ : Comonad C} (f : T₁ ⟶ T₂) (g : T₂ ⟶ T₃) :
    (f ≫ g).toNatTrans = ((f.toNatTrans : _ ⟶ (T₂ : C ⥤ C)) ≫ g.toNatTrans : (T₁ : C ⥤ C) ⟶ T₃) :=
  rfl


/-- Construct a monad isomorphism from a natural isomorphism of functors where the forward
direction is a monad morphism. -/
@[simps]
def MonadIso.mk {M N : Monad C} (f : (M : C ⥤ C) ≅ N)
    (f_η : ∀ (X : C), M.η.app X ≫ f.hom.app X = N.η.app X := by aesop_cat)
    (f_μ : ∀ (X : C), M.μ.app X ≫ f.hom.app X =
    (M.map (f.hom.app X) ≫ f.hom.app (N.obj X)) ≫ N.μ.app X := by aesop_cat) : M ≅ N where
  hom :=
    { toNatTrans := f.hom
      app_η := f_η
      app_μ := f_μ }
  inv :=
    { toNatTrans := f.inv
                           /-
                             C : Type u₁
                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                             T : CategoryTheory.Monad C
                             G : CategoryTheory.Comonad C
                             M N : CategoryTheory.Monad C
                             f : CategoryTheory.Iso M.toFunctor N.toFunctor
                             f_η : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.η.app X) …
                             f_μ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.μ.app X) …
                             X : C
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (N.η.app X) (f.inv.app X)) (M.η.app X)
                           -/
      app_η := fun X => by simp [← f_η]
                           /-
                             🎉 no goals
                           -/
      app_μ := fun X => by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G : CategoryTheory.Comonad C
          M N : CategoryTheory.Monad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_η : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.η.app X) …
          f_μ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.μ.app X) …
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (N.μ.app X) (f.inv.app X)) (CategoryT …
        -/
        rw [← NatIso.cancel_natIso_hom_right f]
        simp only [NatTrans.naturality, Iso.inv_hom_id_app, assoc, comp_id, f_μ,
          NatTrans.naturality_assoc, Iso.inv_hom_id_app_assoc, ← Functor.map_comp_assoc]
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G : CategoryTheory.Comonad C
          M N : CategoryTheory.Monad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_η : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.η.app X) …
          f_μ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.μ.app X) …
          X : C
          ⊢ Eq (N.μ.app X) (CategoryTheory.CategoryStruct.comp (N.map (CategoryTheory.Ca …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- Construct a comonad isomorphism from a natural isomorphism of functors where the forward
direction is a comonad morphism. -/
@[simps]
def ComonadIso.mk {M N : Comonad C} (f : (M : C ⥤ C) ≅ N)
    (f_ε : ∀ (X : C), f.hom.app X ≫ N.ε.app X = M.ε.app X := by aesop_cat)
    (f_δ : ∀ (X : C), f.hom.app X ≫ N.δ.app X =
    M.δ.app X ≫ f.hom.app (M.obj X) ≫ N.map (f.hom.app X) := by aesop_cat) : M ≅ N where
  hom :=
    { toNatTrans := f.hom
      app_ε := f_ε
      app_δ := f_δ }
  inv :=
    { toNatTrans := f.inv
                           /-
                             C : Type u₁
                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                             T : CategoryTheory.Monad C
                             G M N : CategoryTheory.Comonad C
                             f : CategoryTheory.Iso M.toFunctor N.toFunctor
                             f_ε : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
                             f_δ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
                             X : C
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.inv.app X) (M.ε.app X)) (N.ε.app X)
                           -/
      app_ε := fun X => by simp [← f_ε]
                           /-
                             🎉 no goals
                           -/
      app_δ := fun X => by
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G M N : CategoryTheory.Comonad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_ε : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          f_δ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.inv.app X) (M.δ.app X)) (CategoryT …
        -/
        rw [← NatIso.cancel_natIso_hom_left f]
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G M N : CategoryTheory.Comonad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_ε : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          f_δ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.hom.app X) (CategoryTheory.Categor …
        -/
        simp only [reassoc_of% (f_δ X), Iso.hom_inv_id_app_assoc, NatTrans.naturality_assoc]
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G M N : CategoryTheory.Comonad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_ε : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          f_δ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          X : C
          ⊢ Eq (M.δ.app X) (CategoryTheory.CategoryStruct.comp (M.δ.app X) (CategoryTheo …
        -/
        rw [← Functor.map_comp, Iso.hom_inv_id_app, Functor.map_id]
        /-
          C : Type u₁
          inst✝ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          G M N : CategoryTheory.Comonad C
          f : CategoryTheory.Iso M.toFunctor N.toFunctor
          f_ε : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          f_δ : autoParam (∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app  …
          X : C
          ⊢ Eq (M.δ.app X) (CategoryTheory.CategoryStruct.comp (M.δ.app X) (CategoryTheo …
        -/
        apply (comp_id _).symm }
        /-
          🎉 no goals
        -/


/-- The forgetful functor from the category of monads to the category of endofunctors.
-/
@[simps!]
def monadToFunctor : Monad C ⥤ C ⥤ C where
  obj T := T
  map f := f.toNatTrans


instance : (monadToFunctor C).Faithful where


theorem monadToFunctor_mapIso_monad_iso_mk {M N : Monad C} (f : (M : C ⥤ C) ≅ N) (f_η f_μ) :
    (monadToFunctor _).mapIso (MonadIso.mk f f_η f_μ) = f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    M N : CategoryTheory.Monad C
    f : CategoryTheory.Iso M.toFunctor N.toFunctor
    f_η : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.η.app X) (f.hom.app …
    f_μ : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.μ.app X) (f.hom.app …
    ⊢ Eq ((CategoryTheory.monadToFunctor C).mapIso (CategoryTheory.MonadIso.mk f f …
  -/
  ext
  /-
    case w.w.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    M N : CategoryTheory.Monad C
    f : CategoryTheory.Iso M.toFunctor N.toFunctor
    f_η : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.η.app X) (f.hom.app …
    f_μ : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (M.μ.app X) (f.hom.app …
    x✝ : C
    ⊢ Eq (((CategoryTheory.monadToFunctor C).mapIso (CategoryTheory.MonadIso.mk f  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : (monadToFunctor C).ReflectsIsomorphisms where
  reflects f _ := (MonadIso.mk (asIso ((monadToFunctor C).map f)) f.app_η f.app_μ).isIso_hom


/-- The forgetful functor from the category of comonads to the category of endofunctors.
-/
@[simps!]
def comonadToFunctor : Comonad C ⥤ C ⥤ C where
  obj G := G
  map f := f.toNatTrans


instance : (comonadToFunctor C).Faithful where


theorem comonadToFunctor_mapIso_comonad_iso_mk {M N : Comonad C} (f : (M : C ⥤ C) ≅ N) (f_ε f_δ) :
    (comonadToFunctor _).mapIso (ComonadIso.mk f f_ε f_δ) = f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    M N : CategoryTheory.Comonad C
    f : CategoryTheory.Iso M.toFunctor N.toFunctor
    f_ε : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app X) (N.ε.app …
    f_δ : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app X) (N.δ.app …
    ⊢ Eq ((CategoryTheory.comonadToFunctor C).mapIso (CategoryTheory.ComonadIso.mk …
  -/
  ext
  /-
    case w.w.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    M N : CategoryTheory.Comonad C
    f : CategoryTheory.Iso M.toFunctor N.toFunctor
    f_ε : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app X) (N.ε.app …
    f_δ : ∀ (X : C), Eq (CategoryTheory.CategoryStruct.comp (f.hom.app X) (N.δ.app …
    x✝ : C
    ⊢ Eq (((CategoryTheory.comonadToFunctor C).mapIso (CategoryTheory.ComonadIso.m …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : (comonadToFunctor C).ReflectsIsomorphisms where
  reflects f _ := (ComonadIso.mk (asIso ((comonadToFunctor C).map f)) f.app_ε f.app_δ).isIso_hom


/-- An isomorphism of monads gives a natural isomorphism of the underlying functors.
-/
/- Porting note: removed
`@[simps (config := { rhsMd := semireducible })]`
and replaced with `@[simps]` in the two declarations below -/
@[simps!]
def MonadIso.toNatIso {M N : Monad C} (h : M ≅ N) : (M : C ⥤ C) ≅ N :=
  (monadToFunctor C).mapIso h


/-- An isomorphism of comonads gives a natural isomorphism of the underlying functors.
-/
@[simps!]
def ComonadIso.toNatIso {M N : Comonad C} (h : M ≅ N) : (M : C ⥤ C) ≅ N :=
  (comonadToFunctor C).mapIso h


/-- The identity monad. -/
@[simps!]
def id : Monad C where
  toFunctor := 𝟭 C
  η := 𝟙 (𝟭 C)
  μ := 𝟙 (𝟭 C)


instance : Inhabited (Monad C) :=
  ⟨Monad.id C⟩


/-- The identity comonad. -/
@[simps!]
def id : Comonad C where
  toFunctor := 𝟭 _
  ε := 𝟙 (𝟭 C)
  δ := 𝟙 (𝟭 C)


instance : Inhabited (Comonad C) :=
  ⟨Comonad.id C⟩


/-- Transport a monad structure on a functor along an isomorphism of functors. -/
def transport {F : C ⥤ C} (T : Monad C) (i : (T : C ⥤ C) ≅ F) : Monad C where
  toFunctor := F
  η := T.η ≫ i.hom
  μ := (i.inv ◫ i.inv) ≫ T.μ ≫ i.hom
  left_unit X := by
    simp only [Functor.id_obj, NatTrans.comp_app, comp_obj, NatTrans.hcomp_app, Category.assoc,
      hom_inv_id_app_assoc]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.η.app (F.obj X)) (CategoryTheory.C …
    -/
    slice_lhs 1 2 => rw [← T.η.naturality (i.inv.app X), ]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_unit X := by
    simp only [id_obj, NatTrans.comp_app, Functor.map_comp, comp_obj, NatTrans.hcomp_app,
      Category.assoc, NatTrans.naturality_assoc]
    slice_lhs 2 4 =>
      simp only [← T.map_comp]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.inv.app ((CategoryTheory.Functor.i …
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (i.inv.app (F.obj X))) (Catego …
    -/
    simp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (i.inv.app (F.obj X))) (Catego …
    -/
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (i.inv.app (F.obj X))) (Catego …
    -/
  assoc X := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [comp_obj, NatTrans.comp_app, NatTrans.hcomp_app, Category.assoc, Functor.map_comp,
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (i.inv.app (F.obj X))) (Catego …
    -/
      NatTrans.naturality_assoc, hom_inv_id_app_assoc, NatIso.cancel_natIso_inv_left]
    /-
      case e_a
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (T.map (i.inv.app X))) (Catego …
    -/
    slice_lhs 4 5 => rw [← T.map_comp]
    /-
      case e_a
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [hom_inv_id_app, Functor.map_id, id_comp]
    /-
      case e_a
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Monad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    slice_lhs 1 2 => rw [← T.map_comp]
    /-
      🎉 no goals
    -/
    simp only [Functor.map_comp, Category.assoc]
    congr 1
    simp only [← Category.assoc, NatIso.cancel_natIso_hom_right]
    rw [← T.μ.naturality]
    simp [T.assoc X]


/-- Transport a comonad structure on a functor along an isomorphism of functors. -/
def transport {F : C ⥤ C} (T : Comonad C) (i : (T : C ⥤ C) ≅ F) : Comonad C where
  toFunctor := F
  ε := i.inv ≫ T.ε
  δ := i.inv ≫ T.δ ≫ (i.hom ◫ i.hom)
  right_counit X := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
    -/
    simp only [id_obj, comp_obj, NatTrans.comp_app, NatTrans.hcomp_app, Functor.map_comp, assoc]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X) (CategoryTheory.Categor …
    -/
    slice_lhs 4 5 => rw [← F.map_comp]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X) (CategoryTheory.Categor …
    -/
    simp only [hom_inv_id_app, Functor.map_id, id_comp, ← i.hom.naturality]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X) (CategoryTheory.Categor …
    -/
    slice_lhs 2 3 => rw [T.right_counit]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (i.inv.app X) (CategoryTheory.Categor …
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.δ.app X) (CategoryTheory.CategoryS …
    -/
    simp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.δ.app X) (CategoryTheory.CategoryS …
    -/
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.δ.app X) (CategoryTheory.CategoryS …
    -/
  coassoc X := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.δ.app X) (CategoryTheory.CategoryS …
    -/
    simp only [comp_obj, NatTrans.comp_app, NatTrans.hcomp_app, Functor.map_comp, assoc,
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.δ.app X) (CategoryTheory.CategoryS …
    -/
      NatTrans.naturality_assoc, Functor.comp_map, hom_inv_id_app_assoc,
    /-
      case e_a.e_a.e_a
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T✝ : CategoryTheory.Monad C
      G : CategoryTheory.Comonad C
      F : CategoryTheory.Functor C C
      T : CategoryTheory.Comonad C
      i : CategoryTheory.Iso T.toFunctor F
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (i.hom.app (T.obj X))) (F.map  …
    -/
      NatIso.cancel_natIso_inv_left]
    /-
      🎉 no goals
    -/
    slice_lhs 3 4 => rw [← F.map_comp]
    simp only [hom_inv_id_app, Functor.map_id, id_comp, assoc]
    rw [← i.hom.naturality_assoc, ← T.coassoc_assoc]
    simp only [NatTrans.naturality_assoc]
    congr 3
    simp only [← Functor.map_comp, i.hom.naturality]


lemma map_unit_app (T : Monad C) (X : C) [IsIso T.μ] :
    T.map (T.η.app X) = T.η.app (T.obj X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Monad C
    X : C
    inst✝ : CategoryTheory.IsIso T.μ
    ⊢ Eq (T.map (T.η.app X)) (T.η.app (T.obj X))
  -/
  simp [← cancel_mono (T.μ.app _)]
  /-
    🎉 no goals
  -/


lemma isSplitMono_iff_isIso_unit (T : Monad C) (X : C) [IsIso T.μ] :
    IsSplitMono (T.η.app X) ↔ IsIso (T.η.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Monad C
    X : C
    inst✝ : CategoryTheory.IsIso T.μ
    ⊢ Iff (CategoryTheory.IsSplitMono (T.η.app X)) (CategoryTheory.IsIso (T.η.app  …
  -/
  refine ⟨fun _ ↦ ⟨retraction (T.η.app X), by simp, ?_⟩, fun _ ↦ inferInstance⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Monad C
    X : C
    inst✝ : CategoryTheory.IsIso T.μ
    x✝ : CategoryTheory.IsSplitMono (T.η.app X)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.retraction (T.η.app X …
  -/
  erw [← map_id, ← IsSplitMono.id (T.η.app X), map_comp, T.map_unit_app X, T.η.naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Monad C
    X : C
    inst✝ : CategoryTheory.IsIso T.μ
    x✝ : CategoryTheory.IsSplitMono (T.η.app X)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.η.app (T.obj X)) (T.map ⋯.some.1)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma map_counit_app (T : Comonad C) (X : C) [IsIso T.δ] :
    T.map (T.ε.app X) = T.ε.app (T.obj X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Comonad C
    X : C
    inst✝ : CategoryTheory.IsIso T.δ
    ⊢ Eq (T.map (T.ε.app X)) (T.ε.app (T.obj X))
  -/
  simp [← cancel_epi (T.δ.app _)]
  /-
    🎉 no goals
  -/


lemma isSplitEpi_iff_isIso_counit (T : Comonad C) (X : C) [IsIso T.δ] :
    IsSplitEpi (T.ε.app X) ↔ IsIso (T.ε.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Comonad C
    X : C
    inst✝ : CategoryTheory.IsIso T.δ
    ⊢ Iff (CategoryTheory.IsSplitEpi (T.ε.app X)) (CategoryTheory.IsIso (T.ε.app X))
  -/
  refine ⟨fun _ ↦ ⟨section_ (T.ε.app X), ?_, by simp⟩, fun _ ↦ inferInstance⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Comonad C
    X : C
    inst✝ : CategoryTheory.IsIso T.δ
    x✝ : CategoryTheory.IsSplitEpi (T.ε.app X)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.ε.app X) (CategoryTheory.section_  …
  -/
  erw [← map_id, ← IsSplitEpi.id (T.ε.app X), map_comp, T.map_counit_app X, T.ε.naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Comonad C
    X : C
    inst✝ : CategoryTheory.IsIso T.δ
    x✝ : CategoryTheory.IsSplitEpi (T.ε.app X)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.ε.app X) (CategoryTheory.section_  …
  -/
  rfl
  /-
    🎉 no goals
  -/


