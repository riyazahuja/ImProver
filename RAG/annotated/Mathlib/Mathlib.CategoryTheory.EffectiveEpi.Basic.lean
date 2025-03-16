/--
This structure encodes the data required for a morphism to be an effective epimorphism.
-/
structure EffectiveEpiStruct {X Y : C} (f : Y ⟶ X) where
  /--
  For every `W` with a morphism `e : Y ⟶ W` that coequalizes every pair of morphisms
  `g₁ g₂ : Z ⟶ Y` which `f` coequalizes, `desc e h` is a morphism `X ⟶ W`...
  -/
  desc : ∀ {W : C} (e : Y ⟶ W),
    (∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e) → (X ⟶ W)
  /-- ...factorizing `e` through `f`... -/
  fac : ∀ {W : C} (e : Y ⟶ W)
    (h : ∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e),
    f ≫ desc e h = e
  /-- ...and as such, unique. -/
  uniq : ∀ {W : C} (e : Y ⟶ W)
    (h : ∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e)
    (m : X ⟶ W), f ≫ m = e → m = desc e h


/--
A morphism `f : Y ⟶ X` is an effective epimorphism provided that `f` exhibits `X` as a colimit
of the diagram of all "relations" `R ⇉ Y`.
If `f` has a kernel pair, then this is equivalent to showing that the corresponding cofork is
a colimit.
-/
class EffectiveEpi {X Y : C} (f : Y ⟶ X) : Prop where
  /-- `f` is an effective epimorphism if there exists an `EffectiveEpiStruct` for `f`. -/
  effectiveEpi : Nonempty (EffectiveEpiStruct f)


/-- Some chosen `EffectiveEpiStruct` associated to an effective epi. -/
noncomputable
def EffectiveEpi.getStruct {X Y : C} (f : Y ⟶ X) [EffectiveEpi f] : EffectiveEpiStruct f :=
  EffectiveEpi.effectiveEpi.some


/-- Descend along an effective epi. -/
noncomputable
def EffectiveEpi.desc {X Y W : C} (f : Y ⟶ X) [EffectiveEpi f]
    (e : Y ⟶ W) (h : ∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e) :
    X ⟶ W := (EffectiveEpi.getStruct f).desc e h


@[reassoc (attr := simp)]
lemma EffectiveEpi.fac {X Y W : C} (f : Y ⟶ X) [EffectiveEpi f]
    (e : Y ⟶ W) (h : ∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e) :
    f ≫ EffectiveEpi.desc f e h = e :=
  (EffectiveEpi.getStruct f).fac e h


lemma EffectiveEpi.uniq {X Y W : C} (f : Y ⟶ X) [EffectiveEpi f]
    (e : Y ⟶ W) (h : ∀ {Z : C} (g₁ g₂ : Z ⟶ Y), g₁ ≫ f = g₂ ≫ f → g₁ ≫ e = g₂ ≫ e)
    (m : X ⟶ W) (hm : f ≫ m = e) :
    m = EffectiveEpi.desc f e h :=
  (EffectiveEpi.getStruct f).uniq e h _ hm


instance epiOfEffectiveEpi {X Y : C} (f : Y ⟶ X) [EffectiveEpi f] : Epi f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.EffectiveEpi f
    ⊢ CategoryTheory.Epi f
  -/
  constructor
  /-
    case left_cancellation
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.EffectiveEpi f
    ⊢ ∀ {Z : C} (g h : Quiver.Hom X Z), Eq (CategoryTheory.CategoryStruct.comp f g …
  -/
  intro W m₁ m₂ h
  have : m₂ = EffectiveEpi.desc f (f ≫ m₂)
    (fun {Z} g₁ g₂ h => by simp only [← Category.assoc, h]) := EffectiveEpi.uniq _ _ _ _ rfl
  /-
    case left_cancellation
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.EffectiveEpi f
    W : C
    m₁ m₂ : Quiver.Hom X W
    h : Eq (CategoryTheory.CategoryStruct.comp f m₁) (CategoryTheory.CategoryStruc …
    this : Eq m₂ (CategoryTheory.EffectiveEpi.desc f (CategoryTheory.CategoryStruc …
    ⊢ Eq m₁ m₂
  -/
  rw [this]
  /-
    case left_cancellation
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    f : Quiver.Hom Y X
    inst✝ : CategoryTheory.EffectiveEpi f
    W : C
    m₁ m₂ : Quiver.Hom X W
    h : Eq (CategoryTheory.CategoryStruct.comp f m₁) (CategoryTheory.CategoryStruc …
    this : Eq m₂ (CategoryTheory.EffectiveEpi.desc f (CategoryTheory.CategoryStruc …
    ⊢ Eq m₁ (CategoryTheory.EffectiveEpi.desc f (CategoryTheory.CategoryStruct.com …
  -/
  exact EffectiveEpi.uniq _ _ _ _ h
  /-
    🎉 no goals
  -/


/--
This structure encodes the data required for a family of morphisms to be effective epimorphic.
-/
structure EffectiveEpiFamilyStruct {B : C} {α : Type*}
    (X : α → C) (π : (a : α) → (X a ⟶ B)) where
  /--
  For every `W` with a family of morphisms `e a : Y a ⟶ W` that coequalizes every pair of morphisms
  `g₁ : Z ⟶ Y a₁`, `g₂ : Z ⟶ Y a₂` which the family `π` coequalizes, `desc e h` is a morphism
  `X ⟶ W`...
  -/
  desc : ∀ {W} (e : (a : α) → (X a ⟶ W)),
          (∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _) → (B ⟶ W)
  /-- ...factorizing the components of `e` through the components of `π`... -/
  fac : ∀ {W} (e : (a : α) → (X a ⟶ W))
          (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
            g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _)
          (a : α), π a ≫ desc e h = e a
  /-- ...and as such, unique. -/
  uniq : ∀ {W} (e : (a : α) → (X a ⟶ W))
          (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
            g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _)
          (m : B ⟶ W), (∀ (a : α), π a ≫ m = e a) → m = desc e h


/--
A family of morphisms `π a : X a ⟶ B` indexed by `α` is effective epimorphic
provided that the `π a` exhibit `B` as a colimit of the diagram of all "relations"
`R → X a₁`, `R ⟶ X a₂` for all `a₁ a₂ : α`.
-/
class EffectiveEpiFamily {B : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B)) : Prop where
  /-- `π` is an effective epimorphic family if there exists an `EffectiveEpiFamilyStruct` for `π` -/
  effectiveEpiFamily : Nonempty (EffectiveEpiFamilyStruct X π)


/-- Some chosen `EffectiveEpiFamilyStruct` associated to an effective epi family. -/
noncomputable
def EffectiveEpiFamily.getStruct {B : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π] : EffectiveEpiFamilyStruct X π :=
  EffectiveEpiFamily.effectiveEpiFamily.some


/-- Descend along an effective epi family. -/
noncomputable
def EffectiveEpiFamily.desc {B W : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π] (e : (a : α) → (X a ⟶ W))
    (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _) : B ⟶ W :=
  (EffectiveEpiFamily.getStruct X π).desc e h


@[reassoc (attr := simp)]
lemma EffectiveEpiFamily.fac {B W : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π] (e : (a : α) → (X a ⟶ W))
    (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _) (a : α) :
    π a ≫ EffectiveEpiFamily.desc X π e h = e a :=
  (EffectiveEpiFamily.getStruct X π).fac e h a


lemma EffectiveEpiFamily.uniq {B W : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π] (e : (a : α) → (X a ⟶ W))
    (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π _ = g₂ ≫ π _ → g₁ ≫ e _ = g₂ ≫ e _)
    (m : B ⟶ W) (hm : ∀ a, π a ≫ m = e a) :
    m = EffectiveEpiFamily.desc X π e h :=
  (EffectiveEpiFamily.getStruct X π).uniq e h m hm

-- TODO: Once we have "jointly epimorphic families", we could rephrase this as such a property.

lemma EffectiveEpiFamily.hom_ext {B W : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π] (m₁ m₂ : B ⟶ W) (h : ∀ a, π a ≫ m₁ = π a ≫ m₂) :
    m₁ = m₂ := by
  have : m₂ = EffectiveEpiFamily.desc X π (fun a => π a ≫ m₂)
      (fun a₁ a₂ g₁ g₂ h => by simp only [← Category.assoc, h]) := by
    apply EffectiveEpiFamily.uniq; intro; rfl
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    B W : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝ : CategoryTheory.EffectiveEpiFamily X π
    m₁ m₂ : Quiver.Hom B W
    h : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) m₁) (CategoryTheor …
    this : Eq m₂ (CategoryTheory.EffectiveEpiFamily.desc X π (fun a => CategoryThe …
    ⊢ Eq m₁ m₂
  -/
  rw [this]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    B W : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝ : CategoryTheory.EffectiveEpiFamily X π
    m₁ m₂ : Quiver.Hom B W
    h : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) m₁) (CategoryTheor …
    this : Eq m₂ (CategoryTheory.EffectiveEpiFamily.desc X π (fun a => CategoryThe …
    ⊢ Eq m₁ (CategoryTheory.EffectiveEpiFamily.desc X π (fun a => CategoryTheory.C …
  -/
  exact EffectiveEpiFamily.uniq _ _ _ _ _ h
  /-
    🎉 no goals
  -/


/--
An `EffectiveEpiFamily` consisting of a single `EffectiveEpi`
-/
noncomputable
def effectiveEpiFamilyStructSingletonOfEffectiveEpi {B X : C} (f : X ⟶ B) [EffectiveEpi f] :
    EffectiveEpiFamilyStruct (fun () ↦ X) (fun () ↦ f) where
  desc e h := EffectiveEpi.desc f (e ()) (fun g₁ g₂ hg ↦ h () () g₁ g₂ hg)
  fac e h := fun _ ↦ EffectiveEpi.fac f (e ()) (fun g₁ g₂ hg ↦ h () () g₁ g₂ hg)
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{?u.13263, u_1} C
                        B X : C
                        f : Quiver.Hom X B
                        inst✝ : CategoryTheory.EffectiveEpi f
                        W✝ : C
                        e : Unit → Quiver.Hom X W✝
                        h : ∀ {Z : C} (a₁ a₂ : Unit) (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Cate …
                        m : Quiver.Hom B W✝
                        hm : ∀ (a : Unit), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.effe …
                        ⊢ Eq m ((fun {W} e h => CategoryTheory.EffectiveEpi.desc f (e Unit.unit) ⋯) e ⋯)
                      -/
  uniq e h m hm := by apply EffectiveEpi.uniq f (e ()) (h () ()); exact hm ()
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


instance {B X : C} (f : X ⟶ B) [EffectiveEpi f] : EffectiveEpiFamily (fun () ↦ X) (fun () ↦ f) :=
  ⟨⟨effectiveEpiFamilyStructSingletonOfEffectiveEpi f⟩⟩


/--
A single element `EffectiveEpiFamily` consists of an `EffectiveEpi`
-/
noncomputable
def effectiveEpiStructOfEffectiveEpiFamilySingleton {B X : C} (f : X ⟶ B)
    [EffectiveEpiFamily (fun () ↦ X) (fun () ↦ f)] :
    EffectiveEpiStruct f where
  desc e h := EffectiveEpiFamily.desc
    (fun () ↦ X) (fun () ↦ f) (fun () ↦ e) (fun _ _ g₁ g₂ hg ↦ h g₁ g₂ hg)
  fac e h := EffectiveEpiFamily.fac
    (fun () ↦ X) (fun () ↦ f) (fun () ↦ e) (fun _ _ g₁ g₂ hg ↦ h g₁ g₂ hg) ()
  uniq e h m hm := EffectiveEpiFamily.uniq
    (fun () ↦ X) (fun () ↦ f) (fun () ↦ e) (fun _ _ g₁ g₂ hg ↦ h g₁ g₂ hg) m (fun _ ↦ hm)


instance {B X : C} (f : X ⟶ B) [EffectiveEpiFamily (fun () ↦ X) (fun () ↦ f)] :
    EffectiveEpi f :=
  ⟨⟨effectiveEpiStructOfEffectiveEpiFamilySingleton f⟩⟩


theorem effectiveEpi_iff_effectiveEpiFamily {B X : C} (f : X ⟶ B) :
    EffectiveEpi f ↔ EffectiveEpiFamily (fun () ↦ X) (fun () ↦ f) :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ inferInstance⟩


/--
A family of morphisms with the same target inducing an isomorphism from the coproduct to the target
is an `EffectiveEpiFamily`.
-/
noncomputable
def effectiveEpiFamilyStructOfIsIsoDesc {B : C} {α : Type*} (X : α → C)
    (π : (a : α) → (X a ⟶ B)) [HasCoproduct X] [IsIso (Sigma.desc π)] :
    EffectiveEpiFamilyStruct X π where
  desc e _ := (asIso (Sigma.desc π)).inv ≫ (Sigma.desc e)
  fac e h := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.32557, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝¹ : CategoryTheory.Limits.HasCoproduct X
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      ⊢ ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) ((fun {W} e x => Cat …
    -/
    intro a
    have : π a = Sigma.ι X a ≫ (asIso (Sigma.desc π)).hom := by simp only [asIso_hom,
      colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.32557, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝¹ : CategoryTheory.Limits.HasCoproduct X
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      a : α
      this : Eq (π a) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sig …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (π a) ((fun {W} e x => CategoryTheory …
    -/
    rw [this, Category.assoc]
    simp only [asIso_hom, asIso_inv, IsIso.hom_inv_id_assoc, colimit.ι_desc, Cofan.mk_pt,
      Cofan.mk_ι_app]
  uniq e h m hm := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.32557, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝¹ : CategoryTheory.Limits.HasCoproduct X
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      m : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) m) (e a)
      ⊢ Eq m ((fun {W} e x => CategoryTheory.CategoryStruct.comp (CategoryTheory.asI …
    -/
    simp only [asIso_inv, IsIso.eq_inv_comp]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.32557, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝¹ : CategoryTheory.Limits.HasCoproduct X
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      m : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) m) (e a)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc π)  …
    -/
    ext a
    simp only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt, Cofan.mk_ι_app,
      colimit.ι_desc]
    /-
      case h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.32557, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝¹ : CategoryTheory.Limits.HasCoproduct X
      inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      m : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) m) (e a)
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (π a) m) (e a)
    -/
    exact hm a
    /-
      🎉 no goals
    -/


instance {B : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B)) [HasCoproduct X]
    [IsIso (Sigma.desc π)] : EffectiveEpiFamily X π :=
  ⟨⟨effectiveEpiFamilyStructOfIsIsoDesc X π⟩⟩


/-- Any isomorphism is an effective epi. -/
noncomputable
def effectiveEpiStructOfIsIso {X Y : C} (f : X ⟶ Y) [IsIso f] : EffectiveEpiStruct f where
  desc e _ := inv f ≫ e
                /-
                  C : Type u_1
                  inst✝¹ : CategoryTheory.Category.{?u.38074, u_1} C
                  X Y : C
                  f : Quiver.Hom X Y
                  inst✝ : CategoryTheory.IsIso f
                  W✝ : C
                  x✝¹ : Quiver.Hom X W✝
                  x✝ : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.com …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((fun {W} e x => CategoryTheory.Cat …
                -/
  fac _ _ := by simp
                /-
                  🎉 no goals
                -/
                     /-
                       C : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.38074, u_1} C
                       X Y : C
                       f : Quiver.Hom X Y
                       inst✝ : CategoryTheory.IsIso f
                       W✝ : C
                       x✝² : Quiver.Hom X W✝
                       x✝¹ : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.co …
                       x✝ : Quiver.Hom Y W✝
                       h : Eq (CategoryTheory.CategoryStruct.comp f x✝) x✝²
                       ⊢ Eq x✝ ((fun {W} e x => CategoryTheory.CategoryStruct.comp (CategoryTheory.in …
                     -/
  uniq _ _ _ h := by simpa using h
                     /-
                       🎉 no goals
                     -/


instance {X Y : C} (f : X ⟶ Y) [IsIso f] : EffectiveEpi f := ⟨⟨effectiveEpiStructOfIsIso f⟩⟩


/--
Reindex the indexing type of an effective epi family struct.
-/
def EffectiveEpiFamilyStruct.reindex
    {B : C} {α α' : Type*}
    (X : α → C)
    (π : (a : α) → (X a ⟶ B))
    (e : α' ≃ α)
    (P : EffectiveEpiFamilyStruct (fun a => X (e a)) (fun a => π (e a))) :
    EffectiveEpiFamilyStruct X π where
  desc := fun f h => P.desc (fun _ => f _) (fun _ _ => h _ _)
  fac _ _ a := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.40125, u_1} C
      B : C
      α : Type u_2
      α' : Type u_3
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      e : Equiv α' α
      P : CategoryTheory.EffectiveEpiFamilyStruct (fun a => X (e a)) fun a => π (e a)
      W✝ : C
      x✝¹ : (a : α) → Quiver.Hom (X a) W✝
      x✝ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂ …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (π a) ((fun {W} f h => P.desc (fun x  …
    -/
    obtain ⟨a,rfl⟩ := e.surjective a
    /-
      case intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.40125, u_1} C
      B : C
      α : Type u_2
      α' : Type u_3
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      e : Equiv α' α
      P : CategoryTheory.EffectiveEpiFamilyStruct (fun a => X (e a)) fun a => π (e a)
      W✝ : C
      x✝¹ : (a : α) → Quiver.Hom (X a) W✝
      x✝ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂ …
      a : α'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (π (e a)) ((fun {W} f h => P.desc (fu …
    -/
    apply P.fac
    /-
      🎉 no goals
    -/
  uniq _ _ _ hm := P.uniq _ _ _ fun _ => hm _


/--
Reindex the indexing type of an effective epi family.
-/
lemma EffectiveEpiFamily.reindex
    {B : C} {α α' : Type*}
    (X : α → C)
    (π : (a : α) → (X a ⟶ B))
    (e : α' ≃ α)
    (h : EffectiveEpiFamily (fun a => X (e a)) (fun a => π (e a))) :
    EffectiveEpiFamily X π :=
  .mk <| .intro <| @EffectiveEpiFamily.getStruct _ _ _ _ _ _ h |>.reindex _ _ e


