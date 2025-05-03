/-- In an abelian category, the subobjects and quotient objects of an object `X` are
    order-isomorphic via taking kernels and cokernels.
    Implemented here using subobjects in the opposite category,
    since mathlib does not have a notion of quotient objects at the time of writing. -/
@[simps!]
def subobjectIsoSubobjectOp [Abelian C] (X : C) : Subobject X ≃o (Subobject (op X))ᵒᵈ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X : C
    ⊢ OrderIso (CategoryTheory.Subobject X) (OrderDual (CategoryTheory.Subobject { …
  -/
  refine OrderIso.ofHomInv (cokernelOrderHom X) (kernelOrderHom X) ?_ ?_
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ Eq ((↑(CategoryTheory.Limits.cokernelOrderHom X)).comp ↑(CategoryTheory.Limi …
    -/
  · change (cokernelOrderHom X).comp (kernelOrderHom X) = _
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ Eq ((CategoryTheory.Limits.cokernelOrderHom X).comp (CategoryTheory.Limits.k …
    -/
    refine OrderHom.ext _ _ (funext (Subobject.ind _ ?_))
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ ∀ ⦃A : Opposite C⦄ (f : Quiver.Hom A { unop := X }) [inst : CategoryTheory.M …
    -/
    intro A f hf
    dsimp only [OrderHom.comp_coe, Function.comp_apply, kernelOrderHom_coe, Subobject.lift_mk,
      cokernelOrderHom_coe, OrderHom.id_coe, id]
    refine Subobject.mk_eq_mk_of_comm _ _
        ⟨?_, ?_, Quiver.Hom.unop_inj ?_, Quiver.Hom.unop_inj ?_⟩ ?_
      /-
        case refine_1.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X : C
        A : Opposite C
        f : Quiver.Hom A { unop := X }
        hf : CategoryTheory.Mono f
        ⊢ Quiver.Hom { unop := CategoryTheory.Limits.cokernel (CategoryTheory.Limits.k …
      -/
    · exact (Abelian.epiDesc f.unop _ (cokernel.condition (kernel.ι f.unop))).op
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X : C
        A : Opposite C
        f : Quiver.Hom A { unop := X }
        hf : CategoryTheory.Mono f
        ⊢ Quiver.Hom A { unop := CategoryTheory.Limits.cokernel (CategoryTheory.Limits …
      -/
    · exact (cokernel.desc _ _ (kernel.condition f.unop)).op
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_3
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X : C
        A : Opposite C
        f : Quiver.Hom A { unop := X }
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.epiDesc f.uno …
      -/
    · rw [← cancel_epi (cokernel.π (kernel.ι f.unop))]
      simp only [unop_comp, Quiver.Hom.unop_op, unop_id_op, cokernel.π_desc_assoc,
        comp_epiDesc, Category.comp_id]
    · simp only [← cancel_epi f.unop, unop_comp, Quiver.Hom.unop_op, unop_id, comp_epiDesc_assoc,
        cokernel.π_desc, Category.comp_id]
      /-
        case refine_1.refine_5
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X : C
        A : Opposite C
        f : Quiver.Hom A { unop := X }
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := (CategoryTheory.Abelian.epiD …
      -/
    · exact Quiver.Hom.unop_inj (by simp only [unop_comp, Quiver.Hom.unop_op, comp_epiDesc])
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ Eq ((↑(CategoryTheory.Limits.kernelOrderHom X)).comp ↑(CategoryTheory.Limits …
    -/
  · change (kernelOrderHom X).comp (cokernelOrderHom X) = _
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ Eq ((CategoryTheory.Limits.kernelOrderHom X).comp (CategoryTheory.Limits.cok …
    -/
    refine OrderHom.ext _ _ (funext (Subobject.ind _ ?_))
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X : C
      ⊢ ∀ ⦃A : C⦄ (f : Quiver.Hom A X) [inst : CategoryTheory.Mono f], Eq (((Categor …
    -/
    intro A f hf
    dsimp only [OrderHom.comp_coe, Function.comp_apply, cokernelOrderHom_coe, Subobject.lift_mk,
      kernelOrderHom_coe, OrderHom.id_coe, id, unop_op, Quiver.Hom.unop_op]
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      X A : C
      f : Quiver.Hom A X
      hf : CategoryTheory.Mono f
      ⊢ Eq (CategoryTheory.Subobject.mk (CategoryTheory.Limits.kernel.ι (CategoryThe …
    -/
    refine Subobject.mk_eq_mk_of_comm _ _ ⟨?_, ?_, ?_, ?_⟩ ?_
      /-
        case refine_2.refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X A : C
        f : Quiver.Hom A X
        hf : CategoryTheory.Mono f
        ⊢ Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π f …
      -/
    · exact Abelian.monoLift f _ (kernel.condition (cokernel.π f))
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X A : C
        f : Quiver.Hom A X
        hf : CategoryTheory.Mono f
        ⊢ Quiver.Hom A (CategoryTheory.Limits.kernel (CategoryTheory.Limits.cokernel.π …
      -/
    · exact kernel.lift _ _ (cokernel.condition f)
      /-
        🎉 no goals
      -/
    · simp only [← cancel_mono (kernel.ι (cokernel.π f)), Category.assoc, image.fac, monoLift_comp,
        Category.id_comp]
      /-
        case refine_2.refine_4
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X A : C
        f : Quiver.Hom A X
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift (C …
      -/
    · simp only [← cancel_mono f, Category.assoc, monoLift_comp, image.fac, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_5
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        X A : C
        f : Quiver.Hom A X
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Abelian.monoL …
      -/
    · simp only [monoLift_comp]
      /-
        🎉 no goals
      -/


/-- A well-powered abelian category is also well-copowered. -/
instance wellPowered_opposite [Abelian C] [LocallySmall.{w} C] [WellPowered.{w} C] :
    WellPowered.{w} Cᵒᵖ where
  subobject_small X :=
    (small_congr (subobjectIsoSubobjectOp (unop X)).toEquiv).1 inferInstance


