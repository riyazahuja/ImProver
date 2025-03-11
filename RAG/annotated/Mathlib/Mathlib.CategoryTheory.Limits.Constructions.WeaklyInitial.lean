/--
If `C` has (small) products and a small weakly initial set of objects, then it has a weakly initial
object.
-/
theorem has_weakly_initial_of_weakly_initial_set_and_hasProducts [HasProducts.{v} C] {ι : Type v}
    {B : ι → C} (hB : ∀ A : C, ∃ i, Nonempty (B i ⟶ A)) : ∃ T : C, ∀ X, Nonempty (T ⟶ X) :=
  ⟨∏ᶜ B, fun X => ⟨Pi.π _ _ ≫ (hB X).choose_spec.some⟩⟩


/-- If `C` has (small) wide equalizers and a weakly initial object, then it has an initial object.

The initial object is constructed as the wide equalizer of all endomorphisms on the given weakly
initial object.
-/
theorem hasInitial_of_weakly_initial_and_hasWideEqualizers [HasWideEqualizers.{v} C] {T : C}
    (hT : ∀ X, Nonempty (T ⟶ X)) : HasInitial C := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasWideEqualizers C
    T : C
    hT : ∀ (X : C), Nonempty (Quiver.Hom T X)
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  let endos := T ⟶ T
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasWideEqualizers C
    T : C
    hT : ∀ (X : C), Nonempty (Quiver.Hom T X)
    endos : Type v := Quiver.Hom T T
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  let i := wideEqualizer.ι (id : endos → endos)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasWideEqualizers C
    T : C
    hT : ∀ (X : C), Nonempty (Quiver.Hom T X)
    endos : Type v := Quiver.Hom T T
    i : Quiver.Hom (CategoryTheory.Limits.wideEqualizer id) T := CategoryTheory.Li …
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  haveI : Nonempty endos := ⟨𝟙 _⟩
  have : ∀ X : C, Unique (wideEqualizer (id : endos → endos) ⟶ X) := by
    intro X
    refine ⟨⟨i ≫ Classical.choice (hT X)⟩, fun a => ?_⟩
    let E := equalizer a (i ≫ Classical.choice (hT _))
    let e : E ⟶ wideEqualizer id := equalizer.ι _ _
    let h : T ⟶ E := Classical.choice (hT E)
    have : ((i ≫ h) ≫ e) ≫ i = i ≫ 𝟙 _ := by
      rw [Category.assoc, Category.assoc]
      apply wideEqualizer.condition (id : endos → endos) (h ≫ e ≫ i)
    rw [Category.comp_id, cancel_mono_id i] at this
    haveI : IsSplitEpi e := IsSplitEpi.mk' ⟨i ≫ h, this⟩
    rw [← cancel_epi e]
    apply equalizer.condition
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasWideEqualizers C
    T : C
    hT : ∀ (X : C), Nonempty (Quiver.Hom T X)
    endos : Type v := Quiver.Hom T T
    i : Quiver.Hom (CategoryTheory.Limits.wideEqualizer id) T := CategoryTheory.Li …
    this✝ : Nonempty endos
    this : (X : C) → Unique (Quiver.Hom (CategoryTheory.Limits.wideEqualizer id) X)
    ⊢ CategoryTheory.Limits.HasInitial C
  -/
  exact hasInitial_of_unique (wideEqualizer (id : endos → endos))
  /-
    🎉 no goals
  -/


