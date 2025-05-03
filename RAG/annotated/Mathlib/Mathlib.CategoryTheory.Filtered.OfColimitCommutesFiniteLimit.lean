/-- A converse to `colimitLimitIsoLimitColimit`: if colimits of shape `K` commute with finite
limits, then `K` is filtered. -/
theorem isFiltered_of_nonempty_limit_colimit_to_colimit_limit
    (h : ∀ {J : Type v} [SmallCategory J] [FinCategory J] (F : J ⥤ K ⥤ Type v),
      Nonempty (limit (colimit F.flip) ⟶ colimit (limit F))) : IsFiltered K := by
  /-
    K : Type v
    inst✝ : CategoryTheory.SmallCategory K
    h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
    ⊢ CategoryTheory.IsFiltered K
  -/
  refine IsFiltered.iff_nonempty_limit.2 (fun {J} _ _ F => ?_)
  suffices Nonempty (limit (colimit (F.op ⋙ coyoneda).flip)) by
    obtain ⟨X, y, -⟩ := Types.jointly_surjective' (this.map (h (F.op ⋙ coyoneda)).some).some
    exact ⟨X, ⟨(limitObjIsoLimitCompEvaluation (F.op ⋙ coyoneda) _).hom y⟩⟩
  let _ (j : Jᵒᵖ) : Unique ((colimit (F.op ⋙ coyoneda).flip).obj j) :=
    ((colimitObjIsoColimitCompEvaluation (F.op ⋙ coyoneda).flip _ ≪≫
      Coyoneda.colimitCoyonedaIso _)).toEquiv.unique
  /-
    K : Type v
    inst✝ : CategoryTheory.SmallCategory K
    h : ∀ {J : Type v} [inst : CategoryTheory.SmallCategory J] [inst_1 : CategoryT …
    J : Type v
    x✝² : CategoryTheory.SmallCategory J
    x✝¹ : CategoryTheory.FinCategory J
    F : CategoryTheory.Functor J K
    x✝ : (j : Opposite J) → Unique ((CategoryTheory.Limits.colimit (F.op.comp Cate …
    ⊢ Nonempty (CategoryTheory.Limits.limit (CategoryTheory.Limits.colimit (F.op.c …
  -/
  exact ⟨Types.Limit.mk (colimit (F.op ⋙ coyoneda).flip) (fun j => default) (by subsingleton)⟩
  /-
    🎉 no goals
  -/


