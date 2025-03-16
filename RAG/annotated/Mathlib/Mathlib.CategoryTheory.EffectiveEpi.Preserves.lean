theorem effectiveEpiFamilyStructOfEquivalence_aux {W : D} (ε : (a : α) → e.functor.obj (X a) ⟶ W)
    (h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Z ⟶ e.functor.obj (X a₁)) (g₂ : Z ⟶ e.functor.obj (X a₂)),
      g₁ ≫ e.functor.map (π a₁) = g₂ ≫ e.functor.map (π a₂) → g₁ ≫ ε a₁ = g₂ ≫ ε a₂)
    {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂) (hg : g₁ ≫ π a₁ = g₂ ≫ π a₂) :
    g₁ ≫ (fun a ↦ e.unit.app (X a) ≫ e.inverse.map (ε a)) a₁ =
    g₂ ≫ (fun a ↦ e.unit.app (X a) ≫ e.inverse.map (ε a)) a₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    B : C
    α : Type u_3
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    W : D
    ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W
    h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun a => CategoryTheory.Category …
  -/
  have := h a₁ a₂ (e.functor.map g₁) (e.functor.map g₂)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    B : C
    α : Type u_3
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    W : D
    ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W
    h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    this : Eq (CategoryTheory.CategoryStruct.comp (e.functor.map g₁) (e.functor.ma …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun a => CategoryTheory.Category …
  -/
  simp only [← Functor.map_comp, hg] at this
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    e : CategoryTheory.Equivalence C D
    B : C
    α : Type u_3
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    W : D
    ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W
    h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    this : True → Eq (CategoryTheory.CategoryStruct.comp (e.functor.map g₁) (ε a₁) …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun a => CategoryTheory.Category …
  -/
  simpa using congrArg e.inverse.map (this (by trivial))
  /-
    🎉 no goals
  -/


/-- Equivalences preserve effective epimorphic families -/
def effectiveEpiFamilyStructOfEquivalence : EffectiveEpiFamilyStruct (fun a ↦ e.functor.obj (X a))
    (fun a ↦ e.functor.map (π a)) where
  desc ε h := (e.toAdjunction.homEquiv _ _).symm
      (EffectiveEpiFamily.desc X π (fun a ↦ e.unit.app _ ≫ e.inverse.map (ε a))
      (effectiveEpiFamilyStructOfEquivalence_aux e X π ε h))
  fac ε h a := by
    simp only [Functor.comp_obj, Adjunction.homEquiv_counit, Functor.id_obj,
      Equivalence.toAdjunction_counit]
    have := congrArg ((fun f ↦ f ≫ e.counit.app _) ∘ e.functor.map)
      (EffectiveEpiFamily.fac X π (fun a ↦ e.unit.app _ ≫ e.inverse.map (ε a))
      (effectiveEpiFamilyStructOfEquivalence_aux e X π ε h) a)
    simp only [Functor.id_obj, Functor.comp_obj, Function.comp_apply, Functor.map_comp,
        Category.assoc, Equivalence.fun_inv_map, Iso.inv_hom_id_app, Category.comp_id] at this
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.3520, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.3527, u_2} D
      e : CategoryTheory.Equivalence C D
      B : C
      α : Type u_3
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝ : CategoryTheory.EffectiveEpiFamily X π
      W✝ : D
      ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W✝
      h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
      a : α
      this : Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) (CategoryT …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) (CategoryTheory …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  uniq ε h m hm := by
    simp only [Functor.comp_obj, Adjunction.homEquiv_counit, Functor.id_obj,
      Equivalence.toAdjunction_counit]
    have := EffectiveEpiFamily.uniq X π (fun a ↦ e.unit.app _ ≫ e.inverse.map (ε a))
      (effectiveEpiFamilyStructOfEquivalence_aux e X π ε h)
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.3520, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.3527, u_2} D
      e : CategoryTheory.Equivalence C D
      B : C
      α : Type u_3
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝ : CategoryTheory.EffectiveEpiFamily X π
      W✝ : D
      ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W✝
      h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
      m : Quiver.Hom (e.functor.obj B) W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) m …
      this : ∀ (m : Quiver.Hom B (e.inverse.obj W✝)), (∀ (a : α), Eq (CategoryTheory …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp (e.functor.map (CategoryTheory.Effe …
    -/
    specialize this (e.unit.app _ ≫ e.inverse.map m) fun a ↦ ?_
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3520, u_1} C
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.3527, u_2} D
        e : CategoryTheory.Equivalence C D
        B : C
        α : Type u_3
        X : α → C
        π : (a : α) → Quiver.Hom (X a) B
        inst✝ : CategoryTheory.EffectiveEpiFamily X π
        W✝ : D
        ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W✝
        h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
        m : Quiver.Hom (e.functor.obj B) W✝
        hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) m …
        this : ∀ (m : Quiver.Hom B (e.inverse.obj W✝)), (∀ (a : α), Eq (CategoryTheory …
        a : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (π a) (CategoryTheory.CategoryStruct. …
      -/
    · rw [← congrArg e.inverse.map (hm a)]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3520, u_1} C
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.3527, u_2} D
        e : CategoryTheory.Equivalence C D
        B : C
        α : Type u_3
        X : α → C
        π : (a : α) → Quiver.Hom (X a) B
        inst✝ : CategoryTheory.EffectiveEpiFamily X π
        W✝ : D
        ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W✝
        h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
        m : Quiver.Hom (e.functor.obj B) W✝
        hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) m …
        this : ∀ (m : Quiver.Hom B (e.inverse.obj W✝)), (∀ (a : α), Eq (CategoryTheory …
        a : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (π a) (CategoryTheory.CategoryStruct. …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3520, u_1} C
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.3527, u_2} D
        e : CategoryTheory.Equivalence C D
        B : C
        α : Type u_3
        X : α → C
        π : (a : α) → Quiver.Hom (X a) B
        inst✝ : CategoryTheory.EffectiveEpiFamily X π
        W✝ : D
        ε : (a : α) → Quiver.Hom (e.functor.obj (X a)) W✝
        h : ∀ {Z : D} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (e.functor.obj (X a₁))) (g₂ : Qui …
        m : Quiver.Hom (e.functor.obj B) W✝
        hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (π a)) m …
        this : Eq (CategoryTheory.CategoryStruct.comp (e.unit.app B) (e.inverse.map m) …
        ⊢ Eq m (CategoryTheory.CategoryStruct.comp (e.functor.map (CategoryTheory.Effe …
      -/
    · simp [← this]
      /-
        🎉 no goals
      -/


instance (F : C ⥤ D) [F.IsEquivalence] :
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a ↦ F.map (π a)) :=
  ⟨⟨effectiveEpiFamilyStructOfEquivalence F.asEquivalence _ _⟩⟩


/--
A class describing the property of preserving effective epimorphisms.
-/
class PreservesEffectiveEpis (F : C ⥤ D) : Prop where
  /--
  A functor preserves effective epimorphisms if it maps effective
  epimorphisms to effective epimorphisms.
  -/
  preserves : ∀ {X Y : C} (f : X ⟶ Y) [EffectiveEpi f], EffectiveEpi (F.map f)


instance map_effectiveEpi (F : C ⥤ D) [F.PreservesEffectiveEpis] {X Y : C} (f : X ⟶ Y)
    [EffectiveEpi f] : EffectiveEpi (F.map f) :=
  PreservesEffectiveEpis.preserves f


/--
A class describing the property of preserving effective epimorphic families.
-/
class PreservesEffectiveEpiFamilies (F : C ⥤ D) : Prop where
  /--
  A functor preserves effective epimorphic families if it maps effective epimorphic families to
  effective epimorphic families.
  -/
  preserves : ∀ {α : Type u} {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B)) [EffectiveEpiFamily X π],
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a))


instance map_effectiveEpiFamily (F : C ⥤ D) [PreservesEffectiveEpiFamilies.{u} F]
    {α : Type u} {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B)) [EffectiveEpiFamily X π] :
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a)) :=
  PreservesEffectiveEpiFamilies.preserves X π


/--
A class describing the property of preserving finite effective epimorphic families.
-/
class PreservesFiniteEffectiveEpiFamilies (F : C ⥤ D) : Prop where
  /--
  A functor preserves finite effective epimorphic families if it maps finite effective epimorphic
  families to effective epimorphic families.
  -/
  preserves : ∀ {α : Type} [Finite α] {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B))
    [EffectiveEpiFamily X π],
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a))


instance map_finite_effectiveEpiFamily (F : C ⥤ D) [F.PreservesFiniteEffectiveEpiFamilies]
    {α : Type} [Finite α] {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B)) [EffectiveEpiFamily X π] :
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a)) :=
  PreservesFiniteEffectiveEpiFamilies.preserves X π


instance (F : C ⥤ D) [PreservesEffectiveEpiFamilies.{0} F] :
    PreservesFiniteEffectiveEpiFamilies F where
  preserves _ _ := inferInstance


instance (F : C ⥤ D) [PreservesFiniteEffectiveEpiFamilies F] : PreservesEffectiveEpis F where
  preserves _ := inferInstance


instance (F : C ⥤ D) [IsEquivalence F] : F.PreservesEffectiveEpiFamilies where
  preserves _ _ := inferInstance


/--
A class describing the property of reflecting effective epimorphisms.
-/
class ReflectsEffectiveEpis (F : C ⥤ D) : Prop where
  /--
  A functor reflects effective epimorphisms if morphisms that are mapped to epimorphisms are
  themselves effective epimorphisms.
  -/
  reflects : ∀ {X Y : C} (f : X ⟶ Y), EffectiveEpi (F.map f) → EffectiveEpi f


lemma effectiveEpi_of_map (F : C ⥤ D) [F.ReflectsEffectiveEpis] {X Y : C} (f : X ⟶ Y)
    (h : EffectiveEpi (F.map f)) : EffectiveEpi f :=
  ReflectsEffectiveEpis.reflects f h


/--
A class describing the property of reflecting effective epimorphic families.
-/
class ReflectsEffectiveEpiFamilies (F : C ⥤ D) : Prop where
  /--
  A functor reflects effective epimorphic families if families that are mapped to effective
  epimorphic families are themselves effective epimorphic families.
  -/
  reflects : ∀ {α : Type u} {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B)),
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a)) →
    EffectiveEpiFamily X π


lemma effectiveEpiFamily_of_map (F : C ⥤ D) [ReflectsEffectiveEpiFamilies.{u} F]
    {α : Type u} {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B))
    (h : EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a))) :
    EffectiveEpiFamily X π :=
  ReflectsEffectiveEpiFamilies.reflects X π h


/--
A class describing the property of reflecting finite effective epimorphic families.
-/
class ReflectsFiniteEffectiveEpiFamilies (F : C ⥤ D) : Prop where
  /--
  A functor reflects finite effective epimorphic families if finite families that are
  mapped to effective epimorphic families are themselves effective epimorphic families.
  -/
  reflects : ∀ {α : Type} [Finite α] {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B)),
    EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a)) →
    EffectiveEpiFamily X π


lemma finite_effectiveEpiFamily_of_map (F : C ⥤ D) [ReflectsFiniteEffectiveEpiFamilies F]
    {α : Type} [Finite α] {B : C} (X : α → C) (π : (a : α) → (X a ⟶ B))
    (h : EffectiveEpiFamily (fun a ↦ F.obj (X a)) (fun a  ↦ F.map (π a))) :
    EffectiveEpiFamily X π :=
  ReflectsFiniteEffectiveEpiFamilies.reflects X π h


instance (F : C ⥤ D) [ReflectsEffectiveEpiFamilies.{0} F] :
    ReflectsFiniteEffectiveEpiFamilies F where
  reflects _ _ h := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.ReflectsEffectiveEpiFamilies
      α✝ : Type
      inst✝ : Finite α✝
      B✝ : C
      x✝¹ : α✝ → C
      x✝ : (a : α✝) → Quiver.Hom (x✝¹ a) B✝
      h : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (x✝¹ a)) fun a => F.map  …
      ⊢ CategoryTheory.EffectiveEpiFamily x✝¹ x✝
    -/
    have := F.effectiveEpiFamily_of_map _ _ h
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝¹ : F.ReflectsEffectiveEpiFamilies
      α✝ : Type
      inst✝ : Finite α✝
      B✝ : C
      x✝¹ : α✝ → C
      x✝ : (a : α✝) → Quiver.Hom (x✝¹ a) B✝
      h : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (x✝¹ a)) fun a => F.map  …
      this : CategoryTheory.EffectiveEpiFamily x✝¹ x✝
      ⊢ CategoryTheory.EffectiveEpiFamily x✝¹ x✝
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance (F : C ⥤ D) [ReflectsFiniteEffectiveEpiFamilies F] : ReflectsEffectiveEpis F where
  reflects _ h := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.ReflectsFiniteEffectiveEpiFamilies
      X✝ Y✝ : C
      x✝ : Quiver.Hom X✝ Y✝
      h : CategoryTheory.EffectiveEpi (F.map x✝)
      ⊢ CategoryTheory.EffectiveEpi x✝
    -/
    rw [effectiveEpi_iff_effectiveEpiFamily] at h
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.ReflectsFiniteEffectiveEpiFamilies
      X✝ Y✝ : C
      x✝ : Quiver.Hom X✝ Y✝
      h : CategoryTheory.EffectiveEpiFamily (fun x => F.obj X✝) fun x => CategoryThe …
      ⊢ CategoryTheory.EffectiveEpi x✝
    -/
    have := F.finite_effectiveEpiFamily_of_map _ _ h
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.ReflectsFiniteEffectiveEpiFamilies
      X✝ Y✝ : C
      x✝ : Quiver.Hom X✝ Y✝
      h : CategoryTheory.EffectiveEpiFamily (fun x => F.obj X✝) fun x => CategoryThe …
      this : CategoryTheory.EffectiveEpiFamily (fun a => X✝) fun a => x✝
      ⊢ CategoryTheory.EffectiveEpi x✝
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance (F : C ⥤ D) [IsEquivalence F] : F.ReflectsEffectiveEpiFamilies where
  reflects {α B} X π _ := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      α : Type u_5
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      x✝ : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map ( …
      ⊢ CategoryTheory.EffectiveEpiFamily X π
    -/
    let i : (a : α) → X a ⟶ (inv F).obj (F.obj (X a)) := fun a ↦ (asEquivalence F).unit.app _
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      α : Type u_5
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      x✝ : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map ( …
      i : (a : α) → Quiver.Hom (X a) (F.inv.obj (F.obj (X a))) := fun a => F.asEquiv …
      ⊢ CategoryTheory.EffectiveEpiFamily X π
    -/
    have : EffectiveEpiFamily X (fun a ↦ (i a) ≫ (inv F).map (F.map (π a))) := inferInstance
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      α : Type u_5
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      x✝ : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map ( …
      i : (a : α) → Quiver.Hom (X a) (F.inv.obj (F.obj (X a))) := fun a => F.asEquiv …
      this : CategoryTheory.EffectiveEpiFamily X fun a => CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.EffectiveEpiFamily X π
    -/
    simp only [inv_fun_map, Iso.hom_inv_id_app_assoc, i] at this
    have : EffectiveEpiFamily X (fun a ↦ (π a ≫ (asEquivalence F).unit.app B) ≫
        (asEquivalence F).unitInv.app _) := inferInstance
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      α : Type u_5
      B : C
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      x✝ : CategoryTheory.EffectiveEpiFamily (fun a => F.obj (X a)) fun a => F.map ( …
      i : (a : α) → Quiver.Hom (X a) (F.inv.obj (F.obj (X a))) := fun a => F.asEquiv …
      this✝ : CategoryTheory.EffectiveEpiFamily X fun a => CategoryTheory.CategorySt …
      this : CategoryTheory.EffectiveEpiFamily X fun a => CategoryTheory.CategoryStr …
      ⊢ CategoryTheory.EffectiveEpiFamily X π
    -/
    simpa
    /-
      🎉 no goals
    -/


