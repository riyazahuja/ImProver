/-- The pullback cone attached to a commutative square. -/
abbrev pullbackCone : PullbackCone sq.f₂₄ sq.f₃₄ :=
  PullbackCone.mk sq.f₁₂ sq.f₁₃ sq.fac


/-- The pushout cocone attached to a commutative square. -/
abbrev pushoutCocone : PushoutCocone sq.f₁₂ sq.f₁₃ :=
  PushoutCocone.mk sq.f₂₄ sq.f₃₄ sq.fac


/-- The condition that a commutative square is a pullback square. -/
protected def IsPullback : Prop :=
  IsPullback sq.f₁₂ sq.f₁₃ sq.f₂₄ sq.f₃₄


/-- The condition that a commutative square is a pushout square. -/
protected def IsPushout : Prop :=
  IsPushout sq.f₁₂ sq.f₁₃ sq.f₂₄ sq.f₃₄


lemma isPullback_iff :
    sq.IsPullback ↔ Nonempty (IsLimit sq.pullbackCone) :=
  ⟨fun h ↦ ⟨h.isLimit⟩, fun h ↦ { w := sq.fac, isLimit' := h }⟩


lemma isPushout_iff :
    sq.IsPushout ↔ Nonempty (IsColimit sq.pushoutCocone) :=
  ⟨fun h ↦ ⟨h.isColimit⟩, fun h ↦ { w := sq.fac, isColimit' := h }⟩


lemma IsPullback.mk (h : IsLimit sq.pullbackCone) : sq.IsPullback :=
  sq.isPullback_iff.2 ⟨h⟩


lemma IsPushout.mk (h : IsColimit sq.pushoutCocone) : sq.IsPushout :=
  sq.isPushout_iff.2 ⟨h⟩


/-- If a commutative square `sq` is a pullback square,
then `sq.pullbackCone` is limit. -/
noncomputable def IsPullback.isLimit (h : sq.IsPullback) :
    IsLimit sq.pullbackCone :=
  CategoryTheory.IsPullback.isLimit h


/-- If a commutative square `sq` is a pushout square,
then `sq.pushoutCocone` is colimit. -/
noncomputable def IsPushout.isColimit (h : sq.IsPushout) :
    IsColimit sq.pushoutCocone :=
  CategoryTheory.IsPushout.isColimit h


lemma IsPullback.of_iso {sq₁ sq₂ : Square C} (h : sq₁.IsPullback)
    (e : sq₁ ≅ sq₂) : sq₂.IsPullback := by
  refine CategoryTheory.IsPullback.of_iso h
    (evaluation₁.mapIso e) (evaluation₂.mapIso e)
    (evaluation₃.mapIso e) (evaluation₄.mapIso e) ?_ ?_ ?_ ?_
  /-
    case refine_1
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    sq₁ sq₂ : CategoryTheory.Square C
    h : sq₁.IsPullback
    e : CategoryTheory.Iso sq₁ sq₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp sq₁.f₁₂ (CategoryTheory.Square.evalua …
  -/
  all_goals aesop_cat
  /-
    🎉 no goals
  -/


lemma IsPullback.iff_of_iso {sq₁ sq₂ : Square C} (e : sq₁ ≅ sq₂) :
    sq₁.IsPullback ↔ sq₂.IsPullback :=
  ⟨fun h ↦ h.of_iso e, fun h ↦ h.of_iso e.symm⟩


lemma IsPushout.of_iso {sq₁ sq₂ : Square C} (h : sq₁.IsPushout)
    (e : sq₁ ≅ sq₂) : sq₂.IsPushout := by
  refine CategoryTheory.IsPushout.of_iso h
    (evaluation₁.mapIso e) (evaluation₂.mapIso e)
    (evaluation₃.mapIso e) (evaluation₄.mapIso e) ?_ ?_ ?_ ?_
  /-
    case refine_1
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    sq₁ sq₂ : CategoryTheory.Square C
    h : sq₁.IsPushout
    e : CategoryTheory.Iso sq₁ sq₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp sq₁.f₁₂ (CategoryTheory.Square.evalua …
  -/
  all_goals aesop_cat
  /-
    🎉 no goals
  -/


lemma IsPushout.iff_of_iso {sq₁ sq₂ : Square C} (e : sq₁ ≅ sq₂) :
    sq₁.IsPushout ↔ sq₂.IsPushout :=
  ⟨fun h ↦ h.of_iso e, fun h ↦ h.of_iso e.symm⟩


lemma IsPushout.op {sq : Square C} (h : sq.IsPushout) : sq.op.IsPullback :=
  CategoryTheory.IsPushout.op h.flip


lemma IsPushout.unop {sq : Square Cᵒᵖ} (h : sq.IsPushout) : sq.unop.IsPullback :=
  CategoryTheory.IsPushout.unop h.flip


lemma IsPullback.op {sq : Square C} (h : sq.IsPullback) : sq.op.IsPushout :=
  CategoryTheory.IsPullback.op h.flip


lemma IsPullback.unop {sq : Square Cᵒᵖ} (h : sq.IsPullback) : sq.unop.IsPushout :=
  CategoryTheory.IsPullback.unop h.flip


lemma flip : sq.flip.IsPullback := CategoryTheory.IsPullback.flip h


lemma mono_f₁₃ [Mono sq.f₂₄] : Mono sq.f₁₃ :=
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           sq : CategoryTheory.Square C
                                                           h : sq.IsPullback
                                                           inst✝ : CategoryTheory.Mono sq.f₂₄
                                                           ⊢ CategoryTheory.MorphismProperty.monomorphisms C sq.f₂₄
                                                         -/
  (MorphismProperty.monomorphisms C).of_isPullback h (by assumption)
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma mono_f₁₂ [Mono sq.f₃₄] : Mono sq.f₁₂ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    sq : CategoryTheory.Square C
    h : sq.IsPullback
    inst✝ : CategoryTheory.Mono sq.f₃₄
    ⊢ CategoryTheory.Mono sq.f₁₂
  -/
  have : Mono sq.flip.f₂₄ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    sq : CategoryTheory.Square C
    h : sq.IsPullback
    inst✝ : CategoryTheory.Mono sq.f₃₄
    this : CategoryTheory.Mono sq.flip.f₂₄
    ⊢ CategoryTheory.Mono sq.f₁₂
  -/
  exact h.flip.mono_f₁₃
  /-
    🎉 no goals
  -/


lemma flip : sq.flip.IsPushout := CategoryTheory.IsPushout.flip h


lemma epi_f₂₄ [Epi sq.f₁₃] : Epi sq.f₂₄ :=
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         sq : CategoryTheory.Square C
                                                         h : sq.IsPushout
                                                         inst✝ : CategoryTheory.Epi sq.f₁₃
                                                         ⊢ CategoryTheory.MorphismProperty.epimorphisms C sq.f₁₃
                                                       -/
  (MorphismProperty.epimorphisms C).of_isPushout h (by assumption)
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma epi_f₃₄ [Epi sq.f₁₂] : Epi sq.f₃₄ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    sq : CategoryTheory.Square C
    h : sq.IsPushout
    inst✝ : CategoryTheory.Epi sq.f₁₂
    ⊢ CategoryTheory.Epi sq.f₃₄
  -/
  have : Epi sq.flip.f₁₃ := by dsimp; infer_instance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    sq : CategoryTheory.Square C
    h : sq.IsPushout
    inst✝ : CategoryTheory.Epi sq.f₁₂
    this : CategoryTheory.Epi sq.flip.f₁₃
    ⊢ CategoryTheory.Epi sq.f₃₄
  -/
  exact h.flip.epi_f₂₄
  /-
    🎉 no goals
  -/


