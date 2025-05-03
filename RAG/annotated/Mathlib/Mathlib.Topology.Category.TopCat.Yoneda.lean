/--
A universe polymorphic "Yoneda presheaf" on `C` given by continuous maps into a topoological space
`Y`.
-/
@[simps]
def yonedaPresheaf : Cᵒᵖ ⥤ Type (max w w') where
  obj X := C(F.obj (unop X), Y)
  map f g := ContinuousMap.comp g (F.map f.unop)


/--
A universe polymorphic Yoneda presheaf on `TopCat` given by continuous maps into a topoological
space `Y`.
-/
@[simps]
def yonedaPresheaf' : TopCat.{w}ᵒᵖ ⥤ Type (max w w') where
  obj X := C((unop X).1, Y)
  map f g := ContinuousMap.comp g f.unop


theorem comp_yonedaPresheaf' : yonedaPresheaf F Y = F.op ⋙ yonedaPresheaf' Y := rfl


theorem piComparison_fac {α : Type} (X : α → TopCat) :
    piComparison (yonedaPresheaf'.{w, w'} Y) (fun x ↦ op (X x)) =
    (yonedaPresheaf' Y).map ((opCoproductIsoProduct X).inv ≫ (TopCat.sigmaIsoSigma X).inv.op) ≫
    (equivEquivIso (sigmaEquiv Y (fun x ↦ (X x).1))).inv ≫ (Types.productIso _).inv := by
  /-
    Y : Type w'
    inst✝ : TopologicalSpace Y
    α : Type
    X : α → TopCat
    ⊢ Eq (CategoryTheory.Limits.piComparison (ContinuousMap.yonedaPresheaf' Y) fun …
  -/
  rw [← Category.assoc, Iso.eq_comp_inv]
  /-
    Y : Type w'
    inst✝ : TopologicalSpace Y
    α : Type
    X : α → TopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.piComparison ( …
  -/
  ext
  simp only [yonedaPresheaf', unop_op, piComparison, types_comp_apply,
    Types.productIso_hom_comp_eval_apply, Types.pi_lift_π_apply, comp_apply, TopCat.coe_of,
    unop_comp, Quiver.Hom.unop_op, sigmaEquiv, equivEquivIso_hom, Equiv.toIso_inv,
    Equiv.coe_fn_symm_mk, comp_assoc, sigmaMk_apply, ← opCoproductIsoProduct_inv_comp_ι]
  /-
    case h.h.h
    Y : Type w'
    inst✝ : TopologicalSpace Y
    α : Type
    X : α → TopCat
    a✝¹ : (ContinuousMap.yonedaPresheaf' Y).obj (CategoryTheory.Limits.piObj fun x …
    x✝ : α
    a✝ : ↑(X x✝)
    ⊢ Eq (a✝¹ ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The universe polymorphic Yoneda presheaf on `TopCat` preserves finite products. -/
noncomputable instance : PreservesFiniteProducts (yonedaPresheaf'.{w, w'} Y) where
  preserves _ _ :=
    { preservesLimit := fun {K} =>
      have : ∀ {α : Type} (X : α → TopCat), PreservesLimit (Discrete.functor (fun x ↦ op (X x)))
          (yonedaPresheaf'.{w, w'} Y) := fun X => @PreservesProduct.of_iso_comparison _ _ _ _
                                                           /-
                                                             C : Type u
                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                             F : CategoryTheory.Functor C TopCat
                                                             Y : Type w'
                                                             inst✝ : TopologicalSpace Y
                                                             x✝¹ : Type
                                                             x✝ : Fintype x✝¹
                                                             K : CategoryTheory.Functor (CategoryTheory.Discrete x✝¹) (Opposite TopCat)
                                                             α✝ : Type
                                                             X : α✝ → TopCat
                                                             ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.piComparison (ContinuousMap.yone …
                                                           -/
          (yonedaPresheaf' Y) _ (fun x ↦ op (X x)) _ _ (by rw [piComparison_fac]; infer_instance)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
      let i : K ≅ Discrete.functor (fun i ↦ op (unop (K.obj ⟨i⟩))) := Discrete.natIsoFunctor
      preservesLimit_of_iso_diagram _ i.symm }


