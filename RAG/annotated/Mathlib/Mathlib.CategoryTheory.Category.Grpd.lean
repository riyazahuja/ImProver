/-- Category of groupoids -/
@[nolint checkUnivs]
def Grpd :=
  Bundled Groupoid.{v, u}


instance : Inhabited Grpd :=
  ⟨Bundled.of (SingleObj PUnit)⟩



instance str' (C : Grpd.{v, u}) : Groupoid.{v, u} C.α :=
  C.str


instance : CoeSort Grpd Type* :=
  Bundled.coeSort


/-- Construct a bundled `Grpd` from the underlying type and the typeclass `Groupoid`. -/
def of (C : Type u) [Groupoid.{v} C] : Grpd.{v, u} :=
  Bundled.of C


@[simp]
theorem coe_of (C : Type u) [Groupoid C] : (of C : Type u) = C :=
  rfl


/-- Category structure on `Grpd` -/
instance category : LargeCategory.{max v u} Grpd.{v, u} where
  Hom C D := C ⥤ D
  id C := 𝟭 C
  comp F G := F ⋙ G
  id_comp _ := rfl
  comp_id _ := rfl
              /-
                ⊢ ∀ {W X Y Z : CategoryTheory.Grpd} (f : Quiver.Hom W X) (g : Quiver.Hom X Y)  …
              -/
  assoc := by intros; rfl
                      /-
                        🎉 no goals
                      -/


/-- Functor that gets the set of objects of a groupoid. It is not
called `forget`, because it is not a faithful functor. -/
def objects : Grpd.{v, u} ⥤ Type u where
  obj := Bundled.α
  map F := F.obj


/-- Forgetting functor to `Cat` -/
def forgetToCat : Grpd.{v, u} ⥤ Cat.{v, u} where
  obj C := Cat.of C
  map := id


instance forgetToCat_full : forgetToCat.Full where map_surjective f := ⟨f, rfl⟩


instance forgetToCat_faithful : forgetToCat.Faithful where


/-- Convert arrows in the category of groupoids to functors,
which sometimes helps in applying simp lemmas -/
theorem hom_to_functor {C D E : Grpd.{v, u}} (f : C ⟶ D) (g : D ⟶ E) : f ≫ g = f ⋙ g :=
  rfl


/-- Converts identity in the category of groupoids to the functor identity -/
theorem id_to_functor {C : Grpd.{v, u}} : 𝟭 C = 𝟙 C :=
  rfl


/-- Construct the product over an indexed family of groupoids, as a fan. -/
def piLimitFan ⦃J : Type u⦄ (F : J → Grpd.{u, u}) : Limits.Fan F :=
  Limits.Fan.mk (@of (∀ j : J, F j) _) fun j => CategoryTheory.Pi.eval _ j


/-- The product fan over an indexed family of groupoids, is a limit cone. -/
def piLimitFanIsLimit ⦃J : Type u⦄ (F : J → Grpd.{u, u}) : Limits.IsLimit (piLimitFan F) :=
  Limits.mkFanLimit (piLimitFan F) (fun s => Functor.pi' fun j => s.proj j)
    (by
      /-
        J : Type u
        F : J → CategoryTheory.Grpd
        ⊢ ∀ (s : CategoryTheory.Limits.Fan F) (j : J), Eq (CategoryTheory.CategoryStru …
      -/
      intros
      /-
        J : Type u
        F : J → CategoryTheory.Grpd
        s✝ : CategoryTheory.Limits.Fan F
        j✝ : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Functor.pi' …
      -/
      dsimp only [piLimitFan]
      /-
        J : Type u
        F : J → CategoryTheory.Grpd
        s✝ : CategoryTheory.Limits.Fan F
        j✝ : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.pi' fun j =>  …
      -/
      simp [hom_to_functor])
      /-
        🎉 no goals
      -/
    (by
      /-
        J : Type u
        F : J → CategoryTheory.Grpd
        ⊢ ∀ (s : CategoryTheory.Limits.Fan F) (m : Quiver.Hom s.pt (CategoryTheory.Grp …
      -/
      intro s m w
      /-
        J : Type u
        F : J → CategoryTheory.Grpd
        s : CategoryTheory.Limits.Fan F
        m : Quiver.Hom s.pt (CategoryTheory.Grpd.piLimitFan F).pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Grpd. …
        ⊢ Eq m ((fun s => CategoryTheory.Functor.pi' fun j => s.proj j) s)
      -/
      apply Functor.pi_ext
      /-
        case h
        J : Type u
        F : J → CategoryTheory.Grpd
        s : CategoryTheory.Limits.Fan F
        m : Quiver.Hom s.pt (CategoryTheory.Grpd.piLimitFan F).pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Grpd. …
        ⊢ ∀ (i : J), Eq (CategoryTheory.Functor.comp m (CategoryTheory.Pi.eval (fun i  …
      -/
      intro j; specialize w j
      /-
        case h
        J : Type u
        F : J → CategoryTheory.Grpd
        s : CategoryTheory.Limits.Fan F
        m : Quiver.Hom s.pt (CategoryTheory.Grpd.piLimitFan F).pt
        j : J
        w : Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Grpd.piLimitFan  …
        ⊢ Eq (CategoryTheory.Functor.comp m (CategoryTheory.Pi.eval (fun i => ↑(F i))  …
      -/
      simpa)
      /-
        🎉 no goals
      -/


instance has_pi : Limits.HasProducts.{u} Grpd.{u, u} :=
                                       /-
                                         ⊢ {J : Type u} → (f : J → CategoryTheory.Grpd) → CategoryTheory.Limits.Fan f
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  Limits.hasProducts_of_limit_fans (by apply piLimitFan) (by apply piLimitFanIsLimit)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The product of a family of groupoids is isomorphic
to the product object in the category of Groupoids -/
noncomputable def piIsoPi (J : Type u) (f : J → Grpd.{u, u}) : @of (∀ j, f j) _ ≅ ∏ᶜ f :=
  Limits.IsLimit.conePointUniqueUpToIso (piLimitFanIsLimit f)
    (Limits.limit.isLimit (Discrete.functor f))


@[simp]
theorem piIsoPi_hom_π (J : Type u) (f : J → Grpd.{u, u}) (j : J) :
    (piIsoPi J f).hom ≫ Limits.Pi.π f j = CategoryTheory.Pi.eval _ j := by
  /-
    J : Type u
    f : J → CategoryTheory.Grpd
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Grpd.piIsoPi J f).hom …
  -/
  simp [piIsoPi]
  /-
    J : Type u
    f : J → CategoryTheory.Grpd
    j : J
    ⊢ Eq ((CategoryTheory.Grpd.piLimitFan f).π.app { as := j }) (CategoryTheory.Pi …
  -/
  rfl
  /-
    🎉 no goals
  -/


