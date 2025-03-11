instance : HasProp P (Σ (a : α), (σ a)) := HasExplicitFiniteCoproducts.hasProp (fun a ↦ of P (σ a))


/--
The comparison map from the value of a condensed set on a finite coproduct to the product of the
values on the components.
-/
def sigmaComparison : X.obj ⟨(of P ((a : α) × σ a))⟩ ⟶ ((a : α) → X.obj ⟨of P (σ a)⟩) :=
  fun x a ↦ X.map ⟨Sigma.mk a, continuous_sigmaMk⟩ x


theorem sigmaComparison_eq_comp_isos : sigmaComparison X σ =
    (X.mapIso (opCoproductIsoProduct'
      (finiteCoproduct.isColimit.{u, u} (fun a ↦ of P (σ a)))
      (productIsProduct fun x ↦ Opposite.op (of P (σ x))))).hom ≫
    (PreservesProduct.iso X fun a ↦ ⟨of P (σ a)⟩).hom ≫
    (Types.productIso.{u, max u w} fun a ↦ X.obj ⟨of P (σ a)⟩).hom := by
  /-
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    ⊢ Eq (CompHausLike.sigmaComparison X σ) (CategoryTheory.CategoryStruct.comp (X …
  -/
  ext x a
  simp only [Cofan.mk_pt, Fan.mk_pt, Functor.mapIso_hom,
    PreservesProduct.iso_hom, types_comp_apply, Types.productIso_hom_comp_eval_apply]
  /-
    case h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    ⊢ Eq (CompHausLike.sigmaComparison X σ x a) (CategoryTheory.Limits.Pi.π (fun a …
  -/
  have := congrFun (piComparison_comp_π X (fun a ↦ ⟨of P (σ a)⟩) a)
  /-
    case h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq (CompHausLike.sigmaComparison X σ x a) (CategoryTheory.Limits.Pi.π (fun a …
  -/
  simp only [types_comp_apply] at this
  /-
    case h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq (CompHausLike.sigmaComparison X σ x a) (CategoryTheory.Limits.Pi.π (fun a …
  -/
  rw [this, ← FunctorToTypes.map_comp_apply]
  /-
    case h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq (CompHausLike.sigmaComparison X σ x a) (X.map (CategoryTheory.CategoryStr …
  -/
  simp only [sigmaComparison]
  /-
    case h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq (X.map { unop := { toFun := Sigma.mk a, continuous_toFun := ⋯ } } x) (X.m …
  -/
  apply congrFun
  /-
    case h.h.h
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq (X.map { unop := { toFun := Sigma.mk a, continuous_toFun := ⋯ } }) (X.map …
  -/
  congr 2
  /-
    case h.h.h.e_a.e_unop
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq { toFun := Sigma.mk a, continuous_toFun := ⋯ } (CategoryTheory.CategorySt …
  -/
  rw [← opCoproductIsoProduct_inv_comp_ι]
  /-
    case h.h.h.e_a.e_unop
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq { toFun := Sigma.mk a, continuous_toFun := ⋯ } (CategoryTheory.CategorySt …
  -/
  simp only [coe_of, Opposite.unop_op, unop_comp, Quiver.Hom.unop_op, Category.assoc]
  /-
    case h.h.h.e_a.e_unop
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq { toFun := Sigma.mk a, continuous_toFun := ⋯ } (CategoryTheory.CategorySt …
  -/
  simp only [opCoproductIsoProduct, ← unop_comp, opCoproductIsoProduct'_comp_self]
  /-
    case h.h.h.e_a.e_unop
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq { toFun := Sigma.mk a, continuous_toFun := ⋯ } (CategoryTheory.CategorySt …
  -/
  erw [IsColimit.fac]
  /-
    case h.h.h.e_a.e_unop
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    x : X.obj { unop := CompHausLike.of P (Sigma fun a => σ a) }
    a : α
    this : ∀ (a_1 : X.obj (CategoryTheory.Limits.piObj fun a => { unop := CompHaus …
    ⊢ Eq { toFun := Sigma.mk a, continuous_toFun := ⋯ } ((CompHausLike.finiteCopro …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance isIsoSigmaComparison : IsIso <| sigmaComparison X σ := by
  /-
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    ⊢ CategoryTheory.IsIso (CompHausLike.sigmaComparison X σ)
  -/
  rw [sigmaComparison_eq_comp_isos]
  /-
    P : TopCat → Prop
    inst✝⁶ : CompHausLike.HasExplicitFiniteCoproducts P
    X : CategoryTheory.Functor (Opposite (CompHausLike P)) (Type (max u w))
    inst✝⁵ : CategoryTheory.Limits.PreservesFiniteProducts X
    α : Type u
    inst✝⁴ : Finite α
    σ : α → Type u
    inst✝³ : (a : α) → TopologicalSpace (σ a)
    inst✝² : ∀ (a : α), CompactSpace (σ a)
    inst✝¹ : ∀ (a : α), T2Space (σ a)
    inst✝ : ∀ (a : α), CompHausLike.HasProp P (σ a)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (X.mapIso (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


