lemma IsPullback.map (h : sq.IsPullback) (F : C ⥤ D) [PreservesLimit (cospan sq.f₂₄ sq.f₃₄) F] :
    (sq.map F).IsPullback :=
  Square.IsPullback.mk _ (isLimitPullbackConeMapOfIsLimit F sq.fac h.isLimit)


lemma IsPullback.of_map (F : C ⥤ D) [ReflectsLimit (cospan sq.f₂₄ sq.f₃₄) F]
    (h : (sq.map F).IsPullback) : sq.IsPullback :=
  CategoryTheory.IsPullback.of_map F sq.fac h


variable (sq) in
lemma IsPullback.map_iff (F : C ⥤ D) [PreservesLimit (cospan sq.f₂₄ sq.f₃₄) F]
    [ReflectsLimit (cospan sq.f₂₄ sq.f₃₄) F] :
    (sq.map F).IsPullback ↔ sq.IsPullback :=
  ⟨fun h ↦ of_map F h, fun h ↦ h.map F⟩


lemma IsPushout.map (h : sq.IsPushout) (F : C ⥤ D) [PreservesColimit (span sq.f₁₂ sq.f₁₃) F] :
    (sq.map F).IsPushout :=
  Square.IsPushout.mk _ (isColimitPushoutCoconeMapOfIsColimit F sq.fac h.isColimit)


lemma IsPushout.of_map (F : C ⥤ D) [ReflectsColimit (span sq.f₁₂ sq.f₁₃) F]
    (h : (sq.map F).IsPushout) : sq.IsPushout :=
  CategoryTheory.IsPushout.of_map F sq.fac h


variable (sq) in
lemma IsPushout.map_iff (F : C ⥤ D) [PreservesColimit (span sq.f₁₂ sq.f₁₃) F]
    [ReflectsColimit (span sq.f₁₂ sq.f₁₃) F] :
    (sq.map F).IsPushout ↔ sq.IsPushout :=
  ⟨fun h ↦ of_map F h, fun h ↦ h.map F⟩


lemma isPullback_iff_map_coyoneda_isPullback :
    sq.IsPullback ↔ ∀ (X : Cᵒᵖ), (sq.map (coyoneda.obj X)).IsPullback :=
  ⟨fun h _ ↦ h.map _, fun h ↦ IsPullback.mk _
    ((sq.pullbackCone.isLimitCoyonedaEquiv).symm (fun X ↦ (h X).isLimit))⟩


lemma isPushout_iff_op_map_yoneda_isPullback :
    sq.IsPushout ↔ ∀ (X : C), (sq.op.map (yoneda.obj X)).IsPullback :=
  ⟨fun h _ ↦ h.op.map _, fun h ↦ IsPushout.mk _
    ((sq.pushoutCocone.isColimitYonedaEquiv).symm
                                                 /-
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   sq : CategoryTheory.Square C
                                                   h : ∀ (X : C), (sq.op.map (CategoryTheory.yoneda.obj X)).IsPullback
                                                   X : C
                                                   ⊢ Eq (sq.op.map (CategoryTheory.yoneda.obj X)).pullbackCone.fst (CategoryTheor …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
      (fun X ↦ IsLimit.ofIsoLimit (h X).isLimit (PullbackCone.ext (Iso.refl _))))⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


variable (sq₁ sq₂) in
lemma IsPullback.iff_of_equiv : sq₁.IsPullback ↔ sq₂.IsPullback := by
  rw [← IsPullback.map_iff sq₁ uliftFunctor.{max u v},
      ← IsPullback.map_iff sq₂ uliftFunctor.{max u v}]
  refine iff_of_iso (Square.isoMk
    (((Equiv.trans Equiv.ulift e₁).trans Equiv.ulift.symm).toIso)
    (((Equiv.trans Equiv.ulift e₂).trans Equiv.ulift.symm).toIso)
    (((Equiv.trans Equiv.ulift e₃).trans Equiv.ulift.symm).toIso)
    (((Equiv.trans Equiv.ulift e₄).trans Equiv.ulift.symm).toIso)
    ?_ ?_ ?_ ?_)
  /-
    case refine_1
    sq₁ : CategoryTheory.Square (Type v)
    sq₂ : CategoryTheory.Square (Type u)
    e₁ : Equiv sq₁.X₁ sq₂.X₁
    e₂ : Equiv sq₁.X₂ sq₂.X₂
    e₃ : Equiv sq₁.X₃ sq₂.X₃
    e₄ : Equiv sq₁.X₄ sq₂.X₄
    comm₁₂ : Eq (Function.comp (⇑e₂) sq₁.f₁₂) (Function.comp sq₂.f₁₂ ⇑e₁)
    comm₁₃ : Eq (Function.comp (⇑e₃) sq₁.f₁₃) (Function.comp sq₂.f₁₃ ⇑e₁)
    comm₂₄ : Eq (Function.comp (⇑e₄) sq₁.f₂₄) (Function.comp sq₂.f₂₄ ⇑e₂)
    comm₃₄ : Eq (Function.comp (⇑e₄) sq₁.f₃₄) (Function.comp sq₂.f₃₄ ⇑e₃)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (sq₁.map CategoryTheory.uliftFunctor. …
  -/
  all_goals ext; apply ULift.down_injective
    /-
      case refine_1.h.a
      sq₁ : CategoryTheory.Square (Type v)
      sq₂ : CategoryTheory.Square (Type u)
      e₁ : Equiv sq₁.X₁ sq₂.X₁
      e₂ : Equiv sq₁.X₂ sq₂.X₂
      e₃ : Equiv sq₁.X₃ sq₂.X₃
      e₄ : Equiv sq₁.X₄ sq₂.X₄
      comm₁₂ : Eq (Function.comp (⇑e₂) sq₁.f₁₂) (Function.comp sq₂.f₁₂ ⇑e₁)
      comm₁₃ : Eq (Function.comp (⇑e₃) sq₁.f₁₃) (Function.comp sq₂.f₁₃ ⇑e₁)
      comm₂₄ : Eq (Function.comp (⇑e₄) sq₁.f₂₄) (Function.comp sq₂.f₂₄ ⇑e₂)
      comm₃₄ : Eq (Function.comp (⇑e₄) sq₁.f₃₄) (Function.comp sq₂.f₃₄ ⇑e₃)
      a✝ : (sq₁.map CategoryTheory.uliftFunctor.{max u v, v}).X₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (sq₁.map CategoryTheory.uliftFunctor. …
    -/
  · simpa [types_comp, uliftFunctor_map] using congrFun comm₁₂ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2.h.a
      sq₁ : CategoryTheory.Square (Type v)
      sq₂ : CategoryTheory.Square (Type u)
      e₁ : Equiv sq₁.X₁ sq₂.X₁
      e₂ : Equiv sq₁.X₂ sq₂.X₂
      e₃ : Equiv sq₁.X₃ sq₂.X₃
      e₄ : Equiv sq₁.X₄ sq₂.X₄
      comm₁₂ : Eq (Function.comp (⇑e₂) sq₁.f₁₂) (Function.comp sq₂.f₁₂ ⇑e₁)
      comm₁₃ : Eq (Function.comp (⇑e₃) sq₁.f₁₃) (Function.comp sq₂.f₁₃ ⇑e₁)
      comm₂₄ : Eq (Function.comp (⇑e₄) sq₁.f₂₄) (Function.comp sq₂.f₂₄ ⇑e₂)
      comm₃₄ : Eq (Function.comp (⇑e₄) sq₁.f₃₄) (Function.comp sq₂.f₃₄ ⇑e₃)
      a✝ : (sq₁.map CategoryTheory.uliftFunctor.{max u v, v}).X₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (sq₁.map CategoryTheory.uliftFunctor. …
    -/
  · simpa [types_comp, uliftFunctor_map] using congrFun comm₁₃ _
    /-
      🎉 no goals
    -/
    /-
      case refine_3.h.a
      sq₁ : CategoryTheory.Square (Type v)
      sq₂ : CategoryTheory.Square (Type u)
      e₁ : Equiv sq₁.X₁ sq₂.X₁
      e₂ : Equiv sq₁.X₂ sq₂.X₂
      e₃ : Equiv sq₁.X₃ sq₂.X₃
      e₄ : Equiv sq₁.X₄ sq₂.X₄
      comm₁₂ : Eq (Function.comp (⇑e₂) sq₁.f₁₂) (Function.comp sq₂.f₁₂ ⇑e₁)
      comm₁₃ : Eq (Function.comp (⇑e₃) sq₁.f₁₃) (Function.comp sq₂.f₁₃ ⇑e₁)
      comm₂₄ : Eq (Function.comp (⇑e₄) sq₁.f₂₄) (Function.comp sq₂.f₂₄ ⇑e₂)
      comm₃₄ : Eq (Function.comp (⇑e₄) sq₁.f₃₄) (Function.comp sq₂.f₃₄ ⇑e₃)
      a✝ : (sq₁.map CategoryTheory.uliftFunctor.{max u v, v}).X₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (sq₁.map CategoryTheory.uliftFunctor. …
    -/
  · simpa [types_comp, uliftFunctor_map] using congrFun comm₂₄ _
    /-
      🎉 no goals
    -/
    /-
      case refine_4.h.a
      sq₁ : CategoryTheory.Square (Type v)
      sq₂ : CategoryTheory.Square (Type u)
      e₁ : Equiv sq₁.X₁ sq₂.X₁
      e₂ : Equiv sq₁.X₂ sq₂.X₂
      e₃ : Equiv sq₁.X₃ sq₂.X₃
      e₄ : Equiv sq₁.X₄ sq₂.X₄
      comm₁₂ : Eq (Function.comp (⇑e₂) sq₁.f₁₂) (Function.comp sq₂.f₁₂ ⇑e₁)
      comm₁₃ : Eq (Function.comp (⇑e₃) sq₁.f₁₃) (Function.comp sq₂.f₁₃ ⇑e₁)
      comm₂₄ : Eq (Function.comp (⇑e₄) sq₁.f₂₄) (Function.comp sq₂.f₂₄ ⇑e₂)
      comm₃₄ : Eq (Function.comp (⇑e₄) sq₁.f₃₄) (Function.comp sq₂.f₃₄ ⇑e₃)
      a✝ : (sq₁.map CategoryTheory.uliftFunctor.{max u v, v}).X₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (sq₁.map CategoryTheory.uliftFunctor. …
    -/
  · simpa [types_comp, uliftFunctor_map] using congrFun comm₃₄ _
    /-
      🎉 no goals
    -/


lemma IsPullback.of_equiv (h₁ : sq₁.IsPullback) : sq₂.IsPullback :=
  (iff_of_equiv sq₁ sq₂ e₁ e₂ e₃ e₄ comm₁₂ comm₁₃ comm₂₄ comm₃₄).1 h₁


