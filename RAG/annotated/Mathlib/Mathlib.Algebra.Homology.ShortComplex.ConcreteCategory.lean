@[simp]
lemma ShortComplex.zero_apply
    [Limits.HasZeroMorphisms C] [(forget₂ C Ab).PreservesZeroMorphisms]
    (S : ShortComplex C) (x : (forget₂ C Ab).obj S.X₁) :
    ((forget₂ C Ab).map S.g) (((forget₂ C Ab).map S.f) x) = 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.HasForget₂ C Ab
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesZeroMorphisms
    S : CategoryTheory.ShortComplex C
    x : ↑((CategoryTheory.forget₂ C Ab).obj S.X₁)
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map S.g) (((CategoryTheory.forget₂ C Ab). …
  -/
  rw [← comp_apply, ← Functor.map_comp, S.zero, Functor.map_zero]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.ConcreteCategory C
    inst✝² : CategoryTheory.HasForget₂ C Ab
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesZeroMorphisms
    S : CategoryTheory.ShortComplex C
    x : ↑((CategoryTheory.forget₂ C Ab).obj S.X₁)
    ⊢ Eq (0 x) 0
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Preadditive.mono_iff_injective {X Y : C} (f : X ⟶ Y) :
    Mono f ↔ Function.Injective ((forget₂ C Ab).map f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ⇑((CategoryTheory.forget₂ C  …
  -/
  rw [← AddCommGrp.mono_iff_injective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (CategoryTheory.Mono ((CategoryTheory.forget₂ C  …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono f → CategoryTheory.Mono ((CategoryTheory.forget₂ C Ab).m …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget₂ C Ab).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Mono ((CategoryTheory.forget₂ C Ab).map f) → CategoryTheory.M …
    -/
  · apply Functor.mono_of_mono_map
    /-
      🎉 no goals
    -/


lemma Preadditive.mono_iff_injective' {X Y : C} (f : X ⟶ Y) :
    Mono f ↔ Function.Injective ((forget C).map f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ((CategoryTheory.forget C).m …
  -/
  simp only [mono_iff_injective, ← CategoryTheory.mono_iff_injective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono ⇑((CategoryTheory.forget₂ C Ab).map f)) (CategoryTh …
  -/
  apply (MorphismProperty.monomorphisms (Type w)).arrow_mk_iso_iff
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk ⇑((CategoryTheory.forget₂ C Ab). …
  -/
  have e : forget₂ C Ab ⋙ forget Ab ≅ forget C := eqToIso (HasForget₂.forget_comp)
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk ⇑((CategoryTheory.forget₂ C Ab). …
  -/
  exact Arrow.isoOfNatIso e (Arrow.mk f)
  /-
    🎉 no goals
  -/


lemma Preadditive.epi_iff_surjective {X Y : C} (f : X ⟶ Y) :
    Epi f ↔ Function.Surjective ((forget₂ C Ab).map f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑((CategoryTheory.forget₂ C  …
  -/
  rw [← AddCommGrp.epi_iff_surjective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (CategoryTheory.Epi ((CategoryTheory.forget₂ C Ab …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi f → CategoryTheory.Epi ((CategoryTheory.forget₂ C Ab).map …
    -/
  · intro
    /-
      case mp
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      a✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget₂ C Ab).map f)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      inst✝⁵ : CategoryTheory.ConcreteCategory C
      inst✝⁴ : CategoryTheory.HasForget₂ C Ab
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : (CategoryTheory.forget₂ C Ab).Additive
      inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Epi ((CategoryTheory.forget₂ C Ab).map f) → CategoryTheory.Ep …
    -/
  · apply Functor.epi_of_epi_map
    /-
      🎉 no goals
    -/


lemma Preadditive.epi_iff_surjective' {X Y : C} (f : X ⟶ Y) :
    Epi f ↔ Function.Surjective ((forget C).map f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ((CategoryTheory.forget C).m …
  -/
  simp only [epi_iff_surjective, ← CategoryTheory.epi_iff_surjective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi ⇑((CategoryTheory.forget₂ C Ab).map f)) (CategoryThe …
  -/
  apply (MorphismProperty.epimorphisms (Type w)).arrow_mk_iso_iff
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk ⇑((CategoryTheory.forget₂ C Ab). …
  -/
  have e : forget₂ C Ab ⋙ forget Ab ≅ forget C := eqToIso (HasForget₂.forget_comp)
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk ⇑((CategoryTheory.forget₂ C Ab). …
  -/
  exact Arrow.isoOfNatIso e (Arrow.mk f)
  /-
    🎉 no goals
  -/


lemma exact_iff_exact_map_forget₂ [S.HasHomology] :
    S.Exact ↔ (S.map (forget₂ C Ab)).Exact :=
  (S.exact_map_iff_of_faithful (forget₂ C Ab)).symm


lemma exact_iff_of_concreteCategory [S.HasHomology] :
    S.Exact ↔ ∀ (x₂ : (forget₂ C Ab).obj S.X₂) (_ : ((forget₂ C Ab).map S.g) x₂ = 0),
      ∃ (x₁ : (forget₂ C Ab).obj S.X₁), ((forget₂ C Ab).map S.f) x₁ = x₂ := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Iff S.Exact (∀ (x₂ : ↑((CategoryTheory.forget₂ C Ab).obj S.X₂)), Eq (((Categ …
  -/
  rw [S.exact_iff_exact_map_forget₂, ab_exact_iff]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Iff (∀ (x₂ : ↑(S.map (CategoryTheory.forget₂ C Ab)).X₂), Eq ((S.map (Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ShortExact.injective_f [HasZeroObject C] (hS : S.ShortExact) :
    Function.Injective ((forget₂ C Ab).map S.f) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hS : S.ShortExact
    ⊢ Function.Injective ⇑((CategoryTheory.forget₂ C Ab).map S.f)
  -/
  rw [← Preadditive.mono_iff_injective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hS : S.ShortExact
    ⊢ CategoryTheory.Mono S.f
  -/
  exact hS.mono_f
  /-
    🎉 no goals
  -/


lemma ShortExact.surjective_g [HasZeroObject C] (hS : S.ShortExact) :
    Function.Surjective ((forget₂ C Ab).map S.g) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hS : S.ShortExact
    ⊢ Function.Surjective ⇑((CategoryTheory.forget₂ C Ab).map S.g)
  -/
  rw [← Preadditive.epi_iff_surjective]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hS : S.ShortExact
    ⊢ CategoryTheory.Epi S.g
  -/
  exact hS.epi_g
  /-
    🎉 no goals
  -/


/-- Constructor for cycles of short complexes in a concrete category. -/
noncomputable def cyclesMk [S.HasHomology] (x₂ : (forget₂ C Ab).obj S.X₂)
    (hx₂ : ((forget₂ C Ab).map S.g) x₂ = 0) :
    (forget₂ C Ab).obj S.cycles :=
  (S.mapCyclesIso (forget₂ C Ab)).hom ((ShortComplex.abCyclesIso _).inv ⟨x₂, hx₂⟩)


@[simp]
lemma i_cyclesMk [S.HasHomology] (x₂ : (forget₂ C Ab).obj S.X₂)
    (hx₂ : ((forget₂ C Ab).map S.g) x₂ = 0) :
    (forget₂ C Ab).map S.iCycles (S.cyclesMk x₂ hx₂) = x₂ := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.ConcreteCategory C
    inst✝⁴ : CategoryTheory.HasForget₂ C Ab
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : (CategoryTheory.forget₂ C Ab).Additive
    inst✝¹ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    x₂ : ↑((CategoryTheory.forget₂ C Ab).obj S.X₂)
    hx₂ : Eq (((CategoryTheory.forget₂ C Ab).map S.g) x₂) 0
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map S.iCycles) (S.cyclesMk x₂ hx₂)) x₂
  -/
  dsimp [cyclesMk]
  erw [← comp_apply, S.mapCyclesIso_hom_iCycles (forget₂ C Ab),
    ← comp_apply, abCyclesIso_inv_apply_iCycles ]


/-- This lemma allows the computation of the connecting homomorphism
`D.δ` when `D : SnakeInput C` and `C` is a concrete category. -/
lemma δ_apply (x₃ : D.L₀.X₃) (x₂ : D.L₁.X₂) (x₁ : D.L₂.X₁)
    (h₂ : D.L₁.g x₂ = D.v₀₁.τ₃ x₃) (h₁ : D.L₂.f x₁ = D.v₁₂.τ₂ x₂) :
    D.δ x₃ = D.v₂₃.τ₁ x₁ := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  have := (forget₂ C Ab).preservesFiniteLimits_of_preservesHomology
  have : PreservesFiniteLimits (forget C) := by
    have : forget₂ C Ab ⋙ forget Ab = forget C := HasForget₂.forget_comp
    simpa only [← this] using comp_preservesFiniteLimits _ _
  have eq := congr_fun ((forget C).congr_map D.snd_δ)
    (Limits.Concrete.pullbackMk D.L₁.g D.v₀₁.τ₃ x₂ x₃ h₂)
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map (CategoryTheory.CategoryStruct.comp (Ca …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  have eq₁ := Concrete.pullbackMk_fst D.L₁.g D.v₀₁.τ₃ x₂ x₃ h₂
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map (CategoryTheory.CategoryStruct.comp (Ca …
    eq₁ : Eq ((CategoryTheory.Limits.pullback.fst D.L₁.g D.v₀₁.τ₃) (CategoryTheory …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  have eq₂ := Concrete.pullbackMk_snd D.L₁.g D.v₀₁.τ₃ x₂ x₃ h₂
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map (CategoryTheory.CategoryStruct.comp (Ca …
    eq₁ : Eq ((CategoryTheory.Limits.pullback.fst D.L₁.g D.v₀₁.τ₃) (CategoryTheory …
    eq₂ : Eq ((CategoryTheory.Limits.pullback.snd D.L₁.g D.v₀₁.τ₃) (CategoryTheory …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  dsimp [DFunLike.coe] at eq₁ eq₂
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map (CategoryTheory.CategoryStruct.comp (Ca …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  rw [Functor.map_comp, types_comp_apply, FunctorToTypes.map_comp_apply] at eq
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ ((CategoryTheory.forget C).map (Cat …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  rw [eq₂] at eq
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq (D.δ x₃) (D.v₂₃.τ₁ x₁)
  -/
  refine eq.trans (congr_arg ((forget C).map D.v₂₃.τ₁) ?_)
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq ((CategoryTheory.forget C).map D.φ₁ (CategoryTheory.Limits.Concrete.pullb …
  -/
  apply (Preadditive.mono_iff_injective' D.L₂.f).1 inferInstance
  /-
    case a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq ((CategoryTheory.forget C).map D.L₂.f ((CategoryTheory.forget C).map D.φ₁ …
  -/
  rw [← FunctorToTypes.map_comp_apply, φ₁_L₂_f]
  /-
    case a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq ((CategoryTheory.forget C).map D.φ₂ (CategoryTheory.Limits.Concrete.pullb …
  -/
  dsimp [φ₂]
  /-
    case a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq ((CategoryTheory.forget C).map (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  rw [Functor.map_comp, types_comp_apply, eq₁]
  /-
    case a
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : (CategoryTheory.forget C).obj D.L₀.X₃
    x₂ : (CategoryTheory.forget C).obj D.L₁.X₂
    x₁ : (CategoryTheory.forget C).obj D.L₂.X₁
    h₂ : Eq (D.L₁.g x₂) (D.v₀₁.τ₃ x₃)
    h₁ : Eq (D.L₂.f x₁) (D.v₁₂.τ₂ x₂)
    this✝ : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget₂ C  …
    this : CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.forget C)
    eq : Eq ((CategoryTheory.forget C).map D.δ x₃) ((CategoryTheory.forget C).map  …
    eq₁ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.fst D. …
    eq₂ : Eq ((CategoryTheory.forget C).map (CategoryTheory.Limits.pullback.snd D. …
    ⊢ Eq ((CategoryTheory.forget C).map D.v₁₂.τ₂ x₂) ((CategoryTheory.forget C).ma …
  -/
  exact h₁.symm
  /-
    🎉 no goals
  -/


/-- This lemma allows the computation of the connecting homomorphism
`D.δ` when `D : SnakeInput C` and `C` is a concrete category. -/
lemma δ_apply' (x₃ : (forget₂ C Ab).obj D.L₀.X₃)
    (x₂ : (forget₂ C Ab).obj D.L₁.X₂) (x₁ : (forget₂ C Ab).obj D.L₂.X₁)
    (h₂ : (forget₂ C Ab).map D.L₁.g x₂ = (forget₂ C Ab).map D.v₀₁.τ₃ x₃)
    (h₁ : (forget₂ C Ab).map D.L₂.f x₁ = (forget₂ C Ab).map D.v₁₂.τ₂ x₂) :
    (forget₂ C Ab).map D.δ x₃ = (forget₂ C Ab).map D.v₂₃.τ₁ x₁ := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
    x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
    x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
    h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
    h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map D.δ) x₃) (((CategoryTheory.forget₂ C  …
  -/
  have e : forget₂ C Ab ⋙ forget Ab ≅ forget C := eqToIso (HasForget₂.forget_comp)
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.ConcreteCategory C
    inst✝³ : CategoryTheory.HasForget₂ C Ab
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
    inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
    D : CategoryTheory.ShortComplex.SnakeInput C
    x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
    x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
    x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
    h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
    h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
    e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
    ⊢ Eq (((CategoryTheory.forget₂ C Ab).map D.δ) x₃) (((CategoryTheory.forget₂ C  …
  -/
  apply (mono_iff_injective (e.hom.app _)).1 inferInstance
  refine (congr_hom (e.hom.naturality D.δ) x₃).trans
    ((D.δ_apply (e.hom.app _ x₃) (e.hom.app _ x₂) (e.hom.app _ x₁) ?_ ?_ ).trans
    (congr_hom (e.hom.naturality D.v₂₃.τ₁).symm x₁))
  · refine ((congr_hom (e.hom.naturality D.L₁.g) x₂).symm.trans ?_).trans
      (congr_hom (e.hom.naturality D.v₀₁.τ₃) x₃)
    /-
      case a.refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ C Ab).comp …
    -/
    dsimp
    /-
      case a.refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (⇑((CategoryTheory.forget₂ C Ab).map …
    -/
    rw [comp_apply, comp_apply]
    /-
      case a.refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((e.hom.app D.L₁.X₃) (⇑((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂)) (( …
    -/
    erw [h₂]
    /-
      case a.refine_1
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((e.hom.app D.L₁.X₃) (((CategoryTheory.forget₂ C Ab).map D.v₀₁.τ₃) x₃)) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · refine ((congr_hom (e.hom.naturality D.L₂.f) x₁).symm.trans ?_).trans
      (congr_hom (e.hom.naturality D.v₁₂.τ₂) x₂)
    /-
      case a.refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.forget₂ C Ab).comp …
    -/
    dsimp
    /-
      case a.refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (⇑((CategoryTheory.forget₂ C Ab).map …
    -/
    rw [comp_apply, comp_apply]
    /-
      case a.refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((e.hom.app D.L₂.X₂) (⇑((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁)) (( …
    -/
    erw [h₁]
    /-
      case a.refine_2
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.ConcreteCategory C
      inst✝³ : CategoryTheory.HasForget₂ C Ab
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : (CategoryTheory.forget₂ C Ab).Additive
      inst✝ : (CategoryTheory.forget₂ C Ab).PreservesHomology
      D : CategoryTheory.ShortComplex.SnakeInput C
      x₃ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₀.X₃)
      x₂ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₁.X₂)
      x₁ : ↑((CategoryTheory.forget₂ C Ab).obj D.L₂.X₁)
      h₂ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₁.g) x₂) (((CategoryTheory.forg …
      h₁ : Eq (((CategoryTheory.forget₂ C Ab).map D.L₂.f) x₁) (((CategoryTheory.forg …
      e : CategoryTheory.Iso ((CategoryTheory.forget₂ C Ab).comp (CategoryTheory.for …
      ⊢ Eq ((e.hom.app D.L₂.X₂) (((CategoryTheory.forget₂ C Ab).map D.v₁₂.τ₂) x₂)) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/


