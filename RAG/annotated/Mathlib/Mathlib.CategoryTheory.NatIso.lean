/-- The application of a natural isomorphism to an object. We put this definition in a different
namespace, so that we can use `α.app` -/
@[simps]
def app {F G : C ⥤ D} (α : F ≅ G) (X : C) :
    F.obj X ≅ G.obj X where
  hom := α.hom.app X
  inv := α.inv.app X
                   /-
                     C : Type u₁
                     inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝² : CategoryTheory.Category.{v₂, u₂} D
                     E : Type u₃
                     inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                     E' : Type u₄
                     inst✝ : CategoryTheory.Category.{v₄, u₄} E'
                     F G : CategoryTheory.Functor C D
                     α : CategoryTheory.Iso F G
                     X : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.hom.app X) (α.inv.app X)) (Categor …
                   -/
  hom_inv_id := by rw [← comp_app, Iso.hom_inv_id]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
                   /-
                     C : Type u₁
                     inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝² : CategoryTheory.Category.{v₂, u₂} D
                     E : Type u₃
                     inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                     E' : Type u₄
                     inst✝ : CategoryTheory.Category.{v₄, u₄} E'
                     F G : CategoryTheory.Functor C D
                     α : CategoryTheory.Iso F G
                     X : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inv.app X) (α.hom.app X)) (Categor …
                   -/
  inv_hom_id := by rw [← comp_app, Iso.inv_hom_id]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


@[reassoc (attr := simp)]
theorem hom_inv_id_app {F G : C ⥤ D} (α : F ≅ G) (X : C) :
    α.hom.app X ≫ α.inv.app X = 𝟙 (F.obj X) :=
  congr_fun (congr_arg NatTrans.app α.hom_inv_id) X


@[reassoc (attr := simp)]
theorem inv_hom_id_app {F G : C ⥤ D} (α : F ≅ G) (X : C) :
    α.inv.app X ≫ α.hom.app X = 𝟙 (G.obj X) :=
  congr_fun (congr_arg NatTrans.app α.inv_hom_id) X


@[reassoc (attr := simp)]
lemma hom_inv_id_app_app {F G : C ⥤ D ⥤ E} (e : F ≅ G) (X₁ : C) (X₂ : D) :
    (e.hom.app X₁).app X₂ ≫ (e.inv.app X₁).app X₂ = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    e : CategoryTheory.Iso F G
    X₁ : C
    X₂ : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.hom.app X₁).app X₂) ((e.inv.app X …
  -/
  rw [← NatTrans.comp_app, Iso.hom_inv_id_app, NatTrans.id_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inv_hom_id_app_app {F G : C ⥤ D ⥤ E} (e : F ≅ G) (X₁ : C) (X₂ : D) :
    (e.inv.app X₁).app X₂ ≫ (e.hom.app X₁).app X₂ = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    e : CategoryTheory.Iso F G
    X₁ : C
    X₂ : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.inv.app X₁).app X₂) ((e.hom.app X …
  -/
  rw [← NatTrans.comp_app, Iso.inv_hom_id_app, NatTrans.id_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma hom_inv_id_app_app_app {F G : C ⥤ D ⥤ E ⥤ E'} (e : F ≅ G)
    (X₁ : C) (X₂ : D) (X₃ : E) :
    ((e.hom.app X₁).app X₂).app X₃ ≫ ((e.inv.app X₁).app X₂).app X₃ = 𝟙 _ := by
  rw [← NatTrans.comp_app, ← NatTrans.comp_app, Iso.hom_inv_id_app,
    NatTrans.id_app, NatTrans.id_app]


@[reassoc (attr := simp)]
lemma inv_hom_id_app_app_app {F G : C ⥤ D ⥤ E ⥤ E'} (e : F ≅ G)
    (X₁ : C) (X₂ : D) (X₃ : E) :
    ((e.inv.app X₁).app X₂).app X₃ ≫ ((e.hom.app X₁).app X₂).app X₃ = 𝟙 _ := by
  rw [← NatTrans.comp_app, ← NatTrans.comp_app, Iso.inv_hom_id_app,
    NatTrans.id_app, NatTrans.id_app]


@[simp]
theorem trans_app {F G H : C ⥤ D} (α : F ≅ G) (β : G ≅ H) (X : C) :
    (α ≪≫ β).app X = α.app X ≪≫ β.app X :=
  rfl


theorem app_hom {F G : C ⥤ D} (α : F ≅ G) (X : C) : (α.app X).hom = α.hom.app X :=
  rfl


theorem app_inv {F G : C ⥤ D} (α : F ≅ G) (X : C) : (α.app X).inv = α.inv.app X :=
  rfl


instance hom_app_isIso (α : F ≅ G) (X : C) : IsIso (α.hom.app X) :=
  ⟨⟨α.inv.app X,
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          E' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} E'
          F G : CategoryTheory.Functor C D
          α : CategoryTheory.Iso F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.hom.app X) (α.inv.app X)) (Categor …
        -/
        /-
          🎉 no goals
        -/
    ⟨by rw [← comp_app, Iso.hom_inv_id, ← id_app], by rw [← comp_app, Iso.inv_hom_id, ← id_app]⟩⟩⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


instance inv_app_isIso (α : F ≅ G) (X : C) : IsIso (α.inv.app X) :=
  ⟨⟨α.hom.app X,
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          E' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} E'
          F G : CategoryTheory.Functor C D
          α : CategoryTheory.Iso F G
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inv.app X) (α.hom.app X)) (Categor …
        -/
        /-
          🎉 no goals
        -/
    ⟨by rw [← comp_app, Iso.inv_hom_id, ← id_app], by rw [← comp_app, Iso.hom_inv_id, ← id_app]⟩⟩⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cancel_natIso_hom_left {X : C} {Z : D} (g g' : G.obj X ⟶ Z) :
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        F G : CategoryTheory.Functor C D
                                                        α : CategoryTheory.Iso F G
                                                        X : C
                                                        Z : D
                                                        g g' : Quiver.Hom (G.obj X) Z
                                                        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (α.hom.app X) g) (CategoryTheory …
                                                      -/
    α.hom.app X ≫ g = α.hom.app X ≫ g' ↔ g = g' := by simp only [cancel_epi, refl]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cancel_natIso_inv_left {X : C} {Z : D} (g g' : F.obj X ⟶ Z) :
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        F G : CategoryTheory.Functor C D
                                                        α : CategoryTheory.Iso F G
                                                        X : C
                                                        Z : D
                                                        g g' : Quiver.Hom (F.obj X) Z
                                                        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (α.inv.app X) g) (CategoryTheory …
                                                      -/
    α.inv.app X ≫ g = α.inv.app X ≫ g' ↔ g = g' := by simp only [cancel_epi, refl]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cancel_natIso_hom_right {X : D} {Y : C} (f f' : X ⟶ F.obj Y) :
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        F G : CategoryTheory.Functor C D
                                                        α : CategoryTheory.Iso F G
                                                        X : D
                                                        Y : C
                                                        f f' : Quiver.Hom X (F.obj Y)
                                                        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (α.hom.app Y)) (CategoryTheory …
                                                      -/
    f ≫ α.hom.app Y = f' ≫ α.hom.app Y ↔ f = f' := by simp only [cancel_mono, refl]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cancel_natIso_inv_right {X : D} {Y : C} (f f' : X ⟶ G.obj Y) :
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        F G : CategoryTheory.Functor C D
                                                        α : CategoryTheory.Iso F G
                                                        X : D
                                                        Y : C
                                                        f f' : Quiver.Hom X (G.obj Y)
                                                        ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (α.inv.app Y)) (CategoryTheory …
                                                      -/
    f ≫ α.inv.app Y = f' ≫ α.inv.app Y ↔ f = f' := by simp only [cancel_mono, refl]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem cancel_natIso_hom_right_assoc {W X X' : D} {Y : C} (f : W ⟶ X) (g : X ⟶ F.obj Y)
    (f' : W ⟶ X') (g' : X' ⟶ F.obj Y) :
    f ≫ g ≫ α.hom.app Y = f' ≫ g' ≫ α.hom.app Y ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : CategoryTheory.Iso F G
    W X X' : D
    Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom X (F.obj Y)
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' (F.obj Y)
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono, refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_natIso_inv_right_assoc {W X X' : D} {Y : C} (f : W ⟶ X) (g : X ⟶ G.obj Y)
    (f' : W ⟶ X') (g' : X' ⟶ G.obj Y) :
    f ≫ g ≫ α.inv.app Y = f' ≫ g' ≫ α.inv.app Y ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : CategoryTheory.Iso F G
    W X X' : D
    Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom X (G.obj Y)
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' (G.obj Y)
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono, refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_inv_app {F G : C ⥤ D} (e : F ≅ G) (X : C) : inv (e.inv.app X) = e.hom.app X := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F G
    X : C
    ⊢ Eq (CategoryTheory.inv (e.inv.app X)) (e.hom.app X)
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


theorem naturality_1 (α : F ≅ G) (f : X ⟶ Y) : α.inv.app X ≫ F.map f ≫ α.hom.app Y = G.map f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y : C
    α : CategoryTheory.Iso F G
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.inv.app X) (CategoryTheory.Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem naturality_2 (α : F ≅ G) (f : X ⟶ Y) : α.hom.app X ≫ G.map f ≫ α.inv.app Y = F.map f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y : C
    α : CategoryTheory.Iso F G
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.hom.app X) (CategoryTheory.Categor …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem naturality_1' (α : F ⟶ G) (f : X ⟶ Y) {_ : IsIso (α.app X)} :
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        F G : CategoryTheory.Functor C D
                                                        X Y : C
                                                        α : Quiver.Hom F G
                                                        f : Quiver.Hom X Y
                                                        x✝ : CategoryTheory.IsIso (α.app X)
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (α.app X)) (Categ …
                                                      -/
    inv (α.app X) ≫ F.map f ≫ α.app Y = G.map f := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[reassoc (attr := simp)]
theorem naturality_2' (α : F ⟶ G) (f : X ⟶ Y) {_ : IsIso (α.app Y)} :
    α.app X ≫ G.map f ≫ inv (α.app Y) = F.map f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y : C
    α : Quiver.Hom F G
    f : Quiver.Hom X Y
    x✝ : CategoryTheory.IsIso (α.app Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X) (CategoryTheory.CategoryStr …
  -/
  rw [← Category.assoc, ← naturality, Category.assoc, IsIso.hom_inv_id, Category.comp_id]
  /-
    🎉 no goals
  -/


/-- The components of a natural isomorphism are isomorphisms.
-/
instance isIso_app_of_isIso (α : F ⟶ G) [IsIso α] (X) : IsIso (α.app X) :=
  ⟨⟨(inv α).app X,
      ⟨congr_fun (congr_arg NatTrans.app (IsIso.hom_inv_id α)) X,
        congr_fun (congr_arg NatTrans.app (IsIso.inv_hom_id α)) X⟩⟩⟩


@[simp]
theorem isIso_inv_app (α : F ⟶ G) {_ : IsIso α} (X) : (inv α).app X = inv (α.app X) := by
  -- Porting note: the next lemma used to be in `ext`, but that is no longer allowed.
  -- We've added an aesop apply rule;
  -- it would be nice to have a hook to run those without aesop warning it didn't close the goal.
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    x✝ : CategoryTheory.IsIso α
    X : C
    ⊢ Eq ((CategoryTheory.inv α).app X) (CategoryTheory.inv (α.app X))
  -/
  apply IsIso.eq_inv_of_hom_inv_id
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    x✝ : CategoryTheory.IsIso α
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X) ((CategoryTheory.inv α).app …
  -/
  rw [← NatTrans.comp_app]
  /-
    case hom_inv_id
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    x✝ : CategoryTheory.IsIso α
    X : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp α (CategoryTheory.inv α)).app X) (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_map_inv_app (F : C ⥤ D ⥤ E) {X Y : C} (e : X ≅ Y) (Z : D) :
    inv ((F.map e.inv).app Z) = (F.map e.hom).app Z := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F : CategoryTheory.Functor C (CategoryTheory.Functor D E)
    X Y : C
    e : CategoryTheory.Iso X Y
    Z : D
    ⊢ Eq (CategoryTheory.inv ((F.map e.inv).app Z)) ((F.map e.hom).app Z)
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- Construct a natural isomorphism between functors by giving object level isomorphisms,
and checking naturality only in the forward direction.
-/
@[simps]
def ofComponents (app : ∀ X : C, F.obj X ≅ G.obj X)
    (naturality : ∀ {X Y : C} (f : X ⟶ Y),
      F.map f ≫ (app Y).hom = (app X).hom ≫ G.map f := by aesop_cat) :
    F ≅ G where
  hom := { app := fun X => (app X).hom }
  inv :=
    { app := fun X => (app X).inv,
      naturality := fun X Y f => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          E' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} E'
          F G : CategoryTheory.Functor C D
          X✝ Y✝ : C
          app : (X : C) → CategoryTheory.Iso (F.obj X) (G.obj X)
          naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.C …
          X Y : C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (app X).inv) Y)) …
        -/
        have h := congr_arg (fun f => (app X).inv ≫ f ≫ (app Y).inv) (naturality f).symm
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          E' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} E'
          F G : CategoryTheory.Functor C D
          X✝ Y✝ : C
          app : (X : C) → CategoryTheory.Iso (F.obj X) (G.obj X)
          naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.C …
          X Y : C
          f : Quiver.Hom X Y
          h : Eq ((fun f => CategoryTheory.CategoryStruct.comp (app X).inv (CategoryTheo …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (app X).inv) Y)) …
        -/
        simp only [Iso.inv_hom_id_assoc, Iso.hom_inv_id, assoc, comp_id, cancel_mono] at h
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
          E' : Type u₄
          inst✝ : CategoryTheory.Category.{v₄, u₄} E'
          F G : CategoryTheory.Functor C D
          X✝ Y✝ : C
          app : (X : C) → CategoryTheory.Iso (F.obj X) (G.obj X)
          naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.C …
          X Y : C
          f : Quiver.Hom X Y
          h : Eq (CategoryTheory.CategoryStruct.comp (G.map f) (app Y).inv) (CategoryThe …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) ((fun X => (app X).inv) Y)) …
        -/
        exact h }
        /-
          🎉 no goals
        -/


@[simp]
theorem ofComponents.app (app' : ∀ X : C, F.obj X ≅ G.obj X) (naturality) (X) :
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          D : Type u₂
                                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                          F G : CategoryTheory.Functor C D
                                                          app' : (X : C) → CategoryTheory.Iso (F.obj X) (G.obj X)
                                                          naturality : ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
                                                          X : C
                                                          ⊢ Eq ((CategoryTheory.NatIso.ofComponents app' naturality).app X) (app' X)
                                                        -/
    (ofComponents app' naturality).app X = app' X := by aesop
                                                        /-
                                                          🎉 no goals
                                                        -/

-- Making this an instance would cause a typeclass inference loop with `isIso_app_of_isIso`.

/-- A natural transformation is an isomorphism if all its components are isomorphisms.
-/
theorem isIso_of_isIso_app (α : F ⟶ G) [∀ X : C, IsIso (α.app X)] : IsIso α :=
                                               /-
                                                 C : Type u₁
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                 D : Type u₂
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                 F G : CategoryTheory.Functor C D
                                                 α : Quiver.Hom F G
                                                 inst✝ : ∀ (X : C), CategoryTheory.IsIso (α.app X)
                                                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
                                               -/
  (ofComponents (fun X => asIso (α.app X)) (by aesop)).isIso_hom
                                               /-
                                                 🎉 no goals
                                               -/


/-- Horizontal composition of natural isomorphisms. -/
@[simps]
def hcomp {F G : C ⥤ D} {H I : D ⥤ E} (α : F ≅ G) (β : H ≅ I) : F ⋙ H ≅ G ⋙ I := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    E' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} E'
    F✝ G✝ : CategoryTheory.Functor C D
    X Y : C
    F G : CategoryTheory.Functor C D
    H I : CategoryTheory.Functor D E
    α : CategoryTheory.Iso F G
    β : CategoryTheory.Iso H I
    ⊢ CategoryTheory.Iso (F.comp H) (G.comp I)
  -/
  refine ⟨α.hom ◫ β.hom, α.inv ◫ β.inv, ?_, ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      E' : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} E'
      F✝ G✝ : CategoryTheory.Functor C D
      X Y : C
      F G : CategoryTheory.Functor C D
      H I : CategoryTheory.Functor D E
      α : CategoryTheory.Iso F G
      β : CategoryTheory.Iso H I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.hcomp α.hom  …
    -/
  · ext
    /-
      case refine_1.w.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      E' : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} E'
      F✝ G✝ : CategoryTheory.Functor C D
      X Y : C
      F G : CategoryTheory.Functor C D
      H I : CategoryTheory.Functor D E
      α : CategoryTheory.Iso F G
      β : CategoryTheory.Iso H I
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.hcomp α.hom …
    -/
    rw [← NatTrans.exchange]
    /-
      case refine_1.w.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      E' : Type u₄
      inst✝ : CategoryTheory.Category.{v₄, u₄} E'
      F✝ G✝ : CategoryTheory.Functor C D
      X Y : C
      F G : CategoryTheory.Functor C D
      H I : CategoryTheory.Functor D E
      α : CategoryTheory.Iso F G
      β : CategoryTheory.Iso H I
      x✝ : C
      ⊢ Eq ((CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.comp α.hom …
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
    E' : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} E'
    F✝ G✝ : CategoryTheory.Functor C D
    X Y : C
    F G : CategoryTheory.Functor C D
    H I : CategoryTheory.Functor D E
    α : CategoryTheory.Iso F G
    β : CategoryTheory.Iso H I
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.NatTrans.hcomp α.inv  …
  -/
  ext; rw [← NatTrans.exchange]; simp
                                 /-
                                   🎉 no goals
                                 -/


theorem isIso_map_iff {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂) {X Y : C} (f : X ⟶ Y) :
    IsIso (F₁.map f) ↔ IsIso (F₂.map f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F₁ F₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F₁ F₂
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso (F₁.map f)) (CategoryTheory.IsIso (F₂.map f))
  -/
  revert F₁ F₂
  suffices ∀ {F₁ F₂ : C ⥤ D} (_ : F₁ ≅ F₂) (_ : IsIso (F₁.map f)), IsIso (F₂.map f) by
    exact fun F₁ F₂ e => ⟨this e, this e.symm⟩
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X Y : C
    f : Quiver.Hom X Y
    ⊢ ∀ {F₁ F₂ : CategoryTheory.Functor C D}, CategoryTheory.Iso F₁ F₂ → CategoryT …
  -/
  intro F₁ F₂ e hf
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    X Y : C
    f : Quiver.Hom X Y
    F₁ F₂ : CategoryTheory.Functor C D
    e : CategoryTheory.Iso F₁ F₂
    hf : CategoryTheory.IsIso (F₁.map f)
    ⊢ CategoryTheory.IsIso (F₂.map f)
  -/
  refine IsIso.mk ⟨e.inv.app Y ≫ inv (F₁.map f) ≫ e.hom.app X, ?_, ?_⟩
    /-
      case refine_1
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X Y : C
      f : Quiver.Hom X Y
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      hf : CategoryTheory.IsIso (F₁.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F₂.map f) (CategoryTheory.CategorySt …
    -/
  · simp only [NatTrans.naturality_assoc, IsIso.hom_inv_id_assoc, Iso.inv_hom_id_app]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X Y : C
      f : Quiver.Hom X Y
      F₁ F₂ : CategoryTheory.Functor C D
      e : CategoryTheory.Iso F₁ F₂
      hf : CategoryTheory.IsIso (F₁.map f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp only [assoc, ← e.hom.naturality, IsIso.inv_hom_id_assoc, Iso.inv_hom_id_app]
    /-
      🎉 no goals
    -/


lemma NatTrans.isIso_iff_isIso_app {F G : C ⥤ D} (τ : F ⟶ G) :
    IsIso τ ↔ ∀ X, IsIso (τ.app X) :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ NatIso.isIso_of_isIso_app _⟩


/-- Constructor for a functor that is isomorphic to a given functor `F : C ⥤ D`,
while being definitionally equal on objects to a given map `obj : C → D`
such that for all `X : C`, we have an isomorphism `F.obj X ≅ obj X`. -/
@[simps obj]
def copyObj : C ⥤ D where
  obj := obj
  map f := (e _).inv ≫ F.map f ≫ (e _).hom


/-- The functor constructed with `copyObj` is isomorphic to the given functor. -/
@[simps!]
def isoCopyObj : F ≅ F.copyObj obj e :=
                            /-
                              C : Type u₁
                              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                              D : Type u₂
                              inst✝² : CategoryTheory.Category.{v₂, u₂} D
                              E : Type u₃
                              inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                              E' : Type u₄
                              inst✝ : CategoryTheory.Category.{v₄, u₄} E'
                              F : CategoryTheory.Functor C D
                              obj : C → D
                              e : (X : C) → CategoryTheory.Iso (F.obj X) (obj X)
                              ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
                            -/
  NatIso.ofComponents e (by simp [Functor.copyObj])
                            /-
                              🎉 no goals
                            -/


