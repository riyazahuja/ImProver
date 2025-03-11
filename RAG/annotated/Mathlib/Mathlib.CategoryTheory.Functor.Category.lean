/-- `Functor.category C D` gives the category structure on functors and natural transformations
between categories `C` and `D`.

Notice that if `C` and `D` are both small categories at the same universe level,
this is another small category at that level.
However if `C` and `D` are both large categories at the same universe level,
this is a small category at the next higher level.
-/
instance Functor.category : Category.{max u₁ v₂} (C ⥤ D) where
  Hom F G := NatTrans F G
  id F := NatTrans.id F
  comp α β := vcomp α β


@[ext]
theorem ext' {α β : F ⟶ G} (w : α.app = β.app) : α = β := NatTrans.ext w


@[simp]
theorem vcomp_eq_comp (α : F ⟶ G) (β : G ⟶ H) : vcomp α β = α ≫ β := rfl


theorem vcomp_app' (α : F ⟶ G) (β : G ⟶ H) (X : C) : (α ≫ β).app X = α.app X ≫ β.app X := rfl


                                                                              /-
                                                                                C : Type u₁
                                                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                D : Type u₂
                                                                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                F G : CategoryTheory.Functor C D
                                                                                α β : Quiver.Hom F G
                                                                                h : Eq α β
                                                                                X : C
                                                                                ⊢ Eq (α.app X) (β.app X)
                                                                              -/
theorem congr_app {α β : F ⟶ G} (h : α = β) (X : C) : α.app X = β.app X := by rw [h]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem id_app (F : C ⥤ D) (X : C) : (𝟙 F : F ⟶ F).app X = 𝟙 (F.obj X) := rfl


@[simp]
theorem comp_app {F G H : C ⥤ D} (α : F ⟶ G) (β : G ⟶ H) (X : C) :
    (α ≫ β).app X = α.app X ≫ β.app X := rfl


attribute [reassoc] comp_app


@[reassoc]
theorem app_naturality {F G : C ⥤ D ⥤ E} (T : F ⟶ G) (X : C) {Y Z : D} (f : Y ⟶ Z) :
    (F.obj X).map f ≫ (T.app X).app Z = (T.app X).app Y ≫ (G.obj X).map f :=
  (T.app X).naturality f


@[reassoc]
theorem naturality_app {F G : C ⥤ D ⥤ E} (T : F ⟶ G) (Z : D) {X Y : C} (f : X ⟶ Y) :
    (F.map f).app Z ≫ (T.app Y).app Z = (T.app X).app Z ≫ (G.map f).app Z :=
  congr_fun (congr_arg app (T.naturality f)) Z


@[reassoc]
theorem naturality_app_app {F G : C ⥤ D ⥤ E ⥤ E'}
    (α : F ⟶ G) {X₁ Y₁ : C} (f : X₁ ⟶ Y₁) (X₂ : D) (X₃ : E) :
    ((F.map f).app X₂).app X₃ ≫ ((α.app Y₁).app X₂).app X₃ =
      ((α.app X₁).app X₂).app X₃ ≫ ((G.map f).app X₂).app X₃ :=
  congr_app (NatTrans.naturality_app α X₂ f) X₃


/-- A natural transformation is a monomorphism if each component is. -/
theorem mono_of_mono_app (α : F ⟶ G) [∀ X : C, Mono (α.app X)] : Mono α :=
  ⟨fun g h eq => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F G : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      inst✝ : ∀ (X : C), CategoryTheory.Mono (α.app X)
      Z✝ : CategoryTheory.Functor C D
      g h : Quiver.Hom Z✝ F
      eq : Eq (CategoryTheory.CategoryStruct.comp g α) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F G : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      inst✝ : ∀ (X : C), CategoryTheory.Mono (α.app X)
      Z✝ : CategoryTheory.Functor C D
      g h : Quiver.Hom Z✝ F
      eq : Eq (CategoryTheory.CategoryStruct.comp g α) (CategoryTheory.CategoryStruc …
      X : C
      ⊢ Eq (g.app X) (h.app X)
    -/
    rw [← cancel_mono (α.app X), ← comp_app, eq, comp_app]⟩
    /-
      🎉 no goals
    -/


/-- A natural transformation is an epimorphism if each component is. -/
theorem epi_of_epi_app (α : F ⟶ G) [∀ X : C, Epi (α.app X)] : Epi α :=
  ⟨fun g h eq => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F G : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      inst✝ : ∀ (X : C), CategoryTheory.Epi (α.app X)
      Z✝ : CategoryTheory.Functor C D
      g h : Quiver.Hom G Z✝
      eq : Eq (CategoryTheory.CategoryStruct.comp α g) (CategoryTheory.CategoryStruc …
      ⊢ Eq g h
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      F G : CategoryTheory.Functor C D
      α : Quiver.Hom F G
      inst✝ : ∀ (X : C), CategoryTheory.Epi (α.app X)
      Z✝ : CategoryTheory.Functor C D
      g h : Quiver.Hom G Z✝
      eq : Eq (CategoryTheory.CategoryStruct.comp α g) (CategoryTheory.CategoryStruc …
      X : C
      ⊢ Eq (g.app X) (h.app X)
    -/
    rw [← cancel_epi (α.app X), ← comp_app, eq, comp_app]⟩
    /-
      🎉 no goals
    -/


/-- The monoid of natural transformations of the identity is commutative.-/
lemma id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    α β : Quiver.Hom (CategoryTheory.Functor.id C) (CategoryTheory.Functor.id C)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α β) (CategoryTheory.CategoryStruct.c …
  -/
  ext X
  /-
    case w.h
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    α β : Quiver.Hom (CategoryTheory.Functor.id C) (CategoryTheory.Functor.id C)
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp α β).app X) ((CategoryTheory.Categor …
  -/
  exact (α.naturality (β.app X)).symm
  /-
    🎉 no goals
  -/


/-- `hcomp α β` is the horizontal composition of natural transformations. -/
@[simps]
def hcomp {H I : D ⥤ E} (α : F ⟶ G) (β : H ⟶ I) : F ⋙ H ⟶ G ⋙ I where
  app := fun X : C => β.app (F.obj X) ≫ I.map (α.app X)
  naturality X Y f := by
    rw [Functor.comp_map, Functor.comp_map, ← assoc, naturality, assoc, ← map_comp I, naturality,
      map_comp, assoc]


/-- Notation for horizontal composition of natural transformations. -/
infixl:80 " ◫ " => hcomp


theorem hcomp_id_app {H : D ⥤ E} (α : F ⟶ G) (X : C) : (α ◫ 𝟙 H).app X = H.map (α.app X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G : CategoryTheory.Functor C D
    H : CategoryTheory.Functor D E
    α : Quiver.Hom F G
    X : C
    ⊢ Eq ((CategoryTheory.NatTrans.hcomp α (CategoryTheory.CategoryStruct.id H)).a …
  -/
  simp
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
                                                                                         F G : CategoryTheory.Functor C D
                                                                                         H : CategoryTheory.Functor E C
                                                                                         α : Quiver.Hom F G
                                                                                         X : E
                                                                                         ⊢ Eq ((CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id H) α).a …
                                                                                       -/
theorem id_hcomp_app {H : E ⥤ C} (α : F ⟶ G) (X : E) : (𝟙 H ◫ α).app X = α.app _ := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/

-- Note that we don't yet prove a `hcomp_assoc` lemma here: even stating it is painful, because we
-- need to use associativity of functor composition. (It's true without the explicit associator,
-- because functor composition is definitionally associative,
-- but relying on the definitional equality causes bad problems with elaboration later.)

theorem exchange {I J K : D ⥤ E} (α : F ⟶ G) (β : G ⟶ H) (γ : I ⟶ J) (δ : J ⟶ K) :
    (α ≫ β) ◫ (γ ≫ δ) = (α ◫ γ) ≫ β ◫ δ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G H : CategoryTheory.Functor C D
    I J K : CategoryTheory.Functor D E
    α : Quiver.Hom F G
    β : Quiver.Hom G H
    γ : Quiver.Hom I J
    δ : Quiver.Hom J K
    ⊢ Eq (CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.comp α β) ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- Flip the arguments of a bifunctor. See also `Currying.lean`. -/
@[simps]
protected def flip (F : C ⥤ D ⥤ E) : D ⥤ C ⥤ E where
  obj k :=
    { obj := fun j => (F.obj j).obj k,
      map := fun f => (F.map f).app k, }
  map f := { app := fun j => (F.obj j).map f }


variable (C D E) in
/-- The functor `(C ⥤ D ⥤ E) ⥤ D ⥤ C ⥤ E` which flips the variables. -/
@[simps]
def flipFunctor : (C ⥤ D ⥤ E) ⥤ D ⥤ C ⥤ E where
  obj F := F.flip
  map {F₁ F₂} φ :=
    { app := fun Y =>
        { app := fun X => (φ.app X).app Y
          naturality := fun X₁ X₂ f => by
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              E : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
              E' : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} E'
              F G H I : CategoryTheory.Functor C D
              F₁ F₂ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
              φ : Quiver.Hom F₁ F₂
              Y : D
              X₁ X₂ : C
              f : Quiver.Hom X₁ X₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((fun F => F.flip) F₁).obj Y).map f …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝² : CategoryTheory.Category.{v₂, u₂} D
              E : Type u₃
              inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
              E' : Type u₄
              inst✝ : CategoryTheory.Category.{v₄, u₄} E'
              F G H I : CategoryTheory.Functor C D
              F₁ F₂ : CategoryTheory.Functor C (CategoryTheory.Functor D E)
              φ : Quiver.Hom F₁ F₂
              Y : D
              X₁ X₂ : C
              f : Quiver.Hom X₁ X₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F₁.map f).app Y) ((φ.app X₂).app Y) …
            -/
            simp only [← NatTrans.comp_app, naturality] } }
            /-
              🎉 no goals
            -/


@[reassoc (attr := simp)]
theorem map_hom_inv_id_app {X Y : C} (e : X ≅ Y) (F : C ⥤ D ⥤ E) (Z : D) :
    (F.map e.hom).app Z ≫ (F.map e.inv).app Z = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    X Y : C
    e : CategoryTheory.Iso X Y
    F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    Z : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map e.hom).app Z) ((F.map e.inv). …
  -/
  simp [← NatTrans.comp_app, ← Functor.map_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem map_inv_hom_id_app {X Y : C} (e : X ≅ Y) (F : C ⥤ D ⥤ E) (Z : D) :
    (F.map e.inv).app Z ≫ (F.map e.hom).app Z = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    X Y : C
    e : CategoryTheory.Iso X Y
    F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    Z : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map e.inv).app Z) ((F.map e.hom). …
  -/
  simp [← NatTrans.comp_app, ← Functor.map_comp]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-09")] alias map_hom_inv_app := Iso.map_hom_inv_id_app

@[deprecated (since := "2024-06-09")] alias map_inv_hom_app := Iso.map_inv_hom_id_app

@[deprecated (since := "2024-06-09")] alias map_hom_inv_app_assoc := Iso.map_hom_inv_id_app_assoc

@[deprecated (since := "2024-06-09")] alias map_inv_hom_app_assoc := Iso.map_inv_hom_id_app_assoc


