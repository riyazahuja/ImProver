/-- When `R : Cᵒᵖ ⥤ RingCat`, `M : PresheafOfModules R`, and `X : C`, this is the
bijection `((free R).obj (yoneda.obj X) ⟶ M) ≃ M.obj (Opposite.op X)`. -/
noncomputable def freeYonedaEquiv {M : PresheafOfModules.{v} R} {X : C} :
    ((free R).obj (yoneda.obj X) ⟶ M) ≃ M.obj (Opposite.op X) :=
  freeHomEquiv.trans yonedaEquiv


lemma freeYonedaEquiv_symm_app (M : PresheafOfModules.{v} R) (X : C)
    (x : M.obj (Opposite.op X)) :
    (freeYonedaEquiv.symm x).app (Opposite.op X) (ModuleCat.freeMk (𝟙 _)) = x := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    X : C
    x : ↑(M.obj { unop := X })
    ⊢ Eq (((PresheafOfModules.freeYonedaEquiv.symm x).app { unop := X }).hom (Modu …
  -/
  dsimp [freeYonedaEquiv, freeHomEquiv, yonedaEquiv]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    X : C
    x : ↑(M.obj { unop := X })
    ⊢ Eq ((ModuleCat.freeDesc fun f => (M.presheaf.map f.op) x).hom (ModuleCat.fre …
  -/
  rw [ModuleCat.freeDesc_apply, op_id, M.presheaf.map_id]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    X : C
    x : ↑(M.obj { unop := X })
    ⊢ Eq ((CategoryTheory.CategoryStruct.id (M.presheaf.obj { unop := X })) x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma freeYonedaEquiv_comp {M N : PresheafOfModules.{v} R} {X : C}
    (m : ((free R).obj (yoneda.obj X) ⟶ M)) (φ : M ⟶ N) :
    freeYonedaEquiv (m ≫ φ) = φ.app _ (freeYonedaEquiv m) := rfl


variable (R) in
/-- The set of `PresheafOfModules.{v} R` consisting of objects of the
form `(free R).obj (yoneda.obj X)` for some `X`.  -/
def freeYoneda : Set (PresheafOfModules.{v} R) := Set.range (yoneda ⋙ free R).obj


instance : Small.{u} (freeYoneda R) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    ⊢ Small.{u, max u (v + 1)} ↑(PresheafOfModules.freeYoneda R)
  -/
  let π : C → freeYoneda R := fun X ↦ ⟨_, ⟨X, rfl⟩⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    π : C → ↑(PresheafOfModules.freeYoneda R) := fun X => ⟨(CategoryTheory.yoneda. …
    ⊢ Small.{u, max u (v + 1)} ↑(PresheafOfModules.freeYoneda R)
  -/
  have hπ : Function.Surjective π := by rintro ⟨_, ⟨X, rfl⟩⟩; exact ⟨X, rfl⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    π : C → ↑(PresheafOfModules.freeYoneda R) := fun X => ⟨(CategoryTheory.yoneda. …
    hπ : Function.Surjective π
    ⊢ Small.{u, max u (v + 1)} ↑(PresheafOfModules.freeYoneda R)
  -/
  exact small_of_surjective hπ
  /-
    🎉 no goals
  -/


lemma isSeparating : IsSeparating (freeYoneda R) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    ⊢ CategoryTheory.IsSeparating (PresheafOfModules.freeYoneda R)
  -/
  intro M N f₁ f₂ h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M N : PresheafOfModules R
    f₁ f₂ : Quiver.Hom M N
    h : ∀ (G : PresheafOfModules R), Membership.mem (PresheafOfModules.freeYoneda  …
    ⊢ Eq f₁ f₂
  -/
  ext ⟨X⟩ m
  /-
    case h.op.hf.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M N : PresheafOfModules R
    f₁ f₂ : Quiver.Hom M N
    h : ∀ (G : PresheafOfModules R), Membership.mem (PresheafOfModules.freeYoneda  …
    X : C
    m : ↑(M.obj { unop := X })
    ⊢ Eq ((f₁.app { unop := X }).hom m) ((f₂.app { unop := X }).hom m)
  -/
  obtain ⟨g, rfl⟩ := freeYonedaEquiv.surjective m
  /-
    case h.op.hf.h.intro
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M N : PresheafOfModules R
    f₁ f₂ : Quiver.Hom M N
    h : ∀ (G : PresheafOfModules R), Membership.mem (PresheafOfModules.freeYoneda  …
    X : C
    g : Quiver.Hom ((PresheafOfModules.free R).obj (CategoryTheory.yoneda.obj X)) M
    ⊢ Eq ((f₁.app { unop := X }).hom (PresheafOfModules.freeYonedaEquiv g)) ((f₂.a …
  -/
  exact congr_arg freeYonedaEquiv (h _ ⟨X, rfl⟩ g)
  /-
    🎉 no goals
  -/


lemma isDetecting : IsDetecting (freeYoneda R) :=
  (isSeparating R).isDetecting


instance wellPowered {C₀ : Type u} [SmallCategory C₀] (R₀ : C₀ᵒᵖ ⥤ RingCat.{u}) :
    WellPowered.{u} (PresheafOfModules.{u} R₀) :=
  wellPowered_of_isDetecting (freeYoneda.isDetecting R₀)


/-- The type of elements of a presheaf of modules. A term of this type is a pair
`⟨X, a⟩` with `X : Cᵒᵖ` and `a : M.obj X`. -/
abbrev Elements {C : Type u₁} [Category.{v₁} C] {R : Cᵒᵖ ⥤ RingCat.{u}}
  (M : PresheafOfModules.{v} R) := ((toPresheaf R).obj M ⋙ forget Ab).Elements


/-- Given a presheaf of modules `M`, this is a constructor for the type `M.Elements`. -/
abbrev elementsMk {C : Type u₁} [Category.{v₁} C] {R : Cᵒᵖ ⥤ RingCat.{u}}
    (M : PresheafOfModules.{v} R) (X : Cᵒᵖ) (x : M.obj X) : M.Elements :=
  Functor.elementsMk _ X x


/-- Given an element `m : M.Elements` of a presheaf of modules `M`, this is the
free presheaf of modules on the Yoneda presheaf of types corresponding to the
underlying object of `m`. -/
noncomputable abbrev freeYoneda (m : M.Elements) :
    PresheafOfModules.{v} R := (free R).obj (yoneda.obj m.1.unop)


/-- Given an element `m : M.Elements` of a presheaf of modules `M`, this is
the canonical morphism `m.freeYoneda ⟶ M`. -/
noncomputable abbrev fromFreeYoneda (m : M.Elements) :
    m.freeYoneda ⟶ M :=
  freeYonedaEquiv.symm m.2


lemma fromFreeYoneda_app_apply (m : M.Elements) :
    m.fromFreeYoneda.app m.1 (ModuleCat.freeMk (𝟙 _)) = m.2 := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    m : M.Elements
    ⊢ Eq ((m.fromFreeYoneda.app m.fst).hom (ModuleCat.freeMk (CategoryTheory.Categ …
  -/
  apply freeYonedaEquiv_symm_app
  /-
    🎉 no goals
  -/


/-- Given a presheaf of modules `M`, this is the coproduct of
all free Yoneda presheaves `m.freeYoneda` for all `m : M.Elements`. -/
noncomputable abbrev freeYonedaCoproduct : PresheafOfModules.{u} R :=
  ∐ (Elements.freeYoneda (M := M))


/-- Given an element `m : M.Elements` of a presheaf of modules `M`, this is the
canonical inclusion `m.freeYoneda ⟶ M.freeYonedaCoproduct`. -/
noncomputable abbrev ιFreeYonedaCoproduct (m : M.Elements) :
    m.freeYoneda ⟶ M.freeYonedaCoproduct :=
  Sigma.ι _ m


/-- Given a presheaf of modules `M`, this is the
canonical morphism `M.freeYonedaCoproduct ⟶ M`. -/
noncomputable def fromFreeYonedaCoproduct :
    M.freeYonedaCoproduct ⟶ M :=
  Sigma.desc Elements.fromFreeYoneda


/-- Given an element `m` of a presheaf of modules `M`, this is the associated
canonical section of the presheaf `M.freeYonedaCoproduct` over the object `m.1`. -/
noncomputable def freeYonedaCoproductMk (m : M.Elements) :
    M.freeYonedaCoproduct.obj m.1 :=
  (M.ιFreeYonedaCoproduct m).app _ (ModuleCat.freeMk (𝟙 _))


@[reassoc (attr := simp)]
lemma ι_fromFreeYonedaCoproduct (m : M.Elements) :
    M.ιFreeYonedaCoproduct m ≫ M.fromFreeYonedaCoproduct = m.fromFreeYoneda := by
  /-
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    m : M.Elements
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (M.ιFreeYonedaCoproduct m) M.fromFree …
  -/
  apply Sigma.ι_desc
  /-
    🎉 no goals
  -/


lemma ι_fromFreeYonedaCoproduct_apply (m : M.Elements) (X : Cᵒᵖ) (x : m.freeYoneda.obj X) :
    M.fromFreeYonedaCoproduct.app X ((M.ιFreeYonedaCoproduct m).app X x) =
      m.fromFreeYoneda.app X x :=
  congr_fun ((evaluation R X ⋙ forget _).congr_map (M.ι_fromFreeYonedaCoproduct m)) x


@[simp]
lemma fromFreeYonedaCoproduct_app_mk (m : M.Elements) :
    M.fromFreeYonedaCoproduct.app _ (M.freeYonedaCoproductMk m) = m.2 := by
  /-
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    m : M.Elements
    ⊢ Eq ((M.fromFreeYonedaCoproduct.app m.fst).hom (M.freeYonedaCoproductMk m)) m …
  -/
  dsimp [freeYonedaCoproductMk]
  /-
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    m : M.Elements
    ⊢ Eq ((M.fromFreeYonedaCoproduct.app m.fst).hom (((M.ιFreeYonedaCoproduct m).a …
  -/
  erw [M.ι_fromFreeYonedaCoproduct_apply m]
  /-
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    m : M.Elements
    ⊢ Eq ((m.fromFreeYoneda.app m.fst).hom (ModuleCat.freeMk (CategoryTheory.Categ …
  -/
  rw [m.fromFreeYoneda_app_apply]
  /-
    🎉 no goals
  -/


instance : Epi M.fromFreeYonedaCoproduct :=
  epi_of_surjective (fun X m ↦ ⟨M.freeYonedaCoproductMk (M.elementsMk X m),
    M.fromFreeYonedaCoproduct_app_mk (M.elementsMk X m)⟩)


/-- Given a presheaf of modules `M`, this is a morphism between coproducts
of free presheaves of modules on Yoneda presheaves which gives a presentation
of the module `M`, see `isColimitFreeYonedaCoproductsCokernelCofork`. -/
noncomputable def toFreeYonedaCoproduct :
    (kernel M.fromFreeYonedaCoproduct).freeYonedaCoproduct ⟶ M.freeYonedaCoproduct :=
  (kernel M.fromFreeYonedaCoproduct).fromFreeYonedaCoproduct ≫ kernel.ι _


@[reassoc (attr := simp)]
lemma toFreeYonedaCoproduct_fromFreeYonedaCoproduct :
    M.toFreeYonedaCoproduct ≫ M.fromFreeYonedaCoproduct = 0 := by
  /-
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp M.toFreeYonedaCoproduct M.fromFreeYon …
  -/
  simp [toFreeYonedaCoproduct]
  /-
    🎉 no goals
  -/


/-- (Colimit) cofork which gives a presentation of a presheaf of modules `M` using
coproducts of free presheaves of modules on Yoneda presheaves. -/
noncomputable abbrev freeYonedaCoproductsCokernelCofork :
    CokernelCofork M.toFreeYonedaCoproduct :=
  CokernelCofork.ofπ _ M.toFreeYonedaCoproduct_fromFreeYonedaCoproduct


/-- If `M` is a presheaf of modules, the cokernel cofork
`M.freeYonedaCoproductsCokernelCofork` is a colimit, which means that
`M` can be expressed as a cokernel of the morphism `M.toFreeYonedaCoproduct`
between coproducts of free presheaves of modules on Yoneda presheaves. -/
noncomputable def isColimitFreeYonedaCoproductsCokernelCofork :
    IsColimit M.freeYonedaCoproductsCokernelCofork := by
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    R✝ : CategoryTheory.Functor (Opposite C✝) RingCat
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    ⊢ CategoryTheory.Limits.IsColimit M.freeYonedaCoproductsCokernelCofork
  -/
  let S := ShortComplex.mk _ _ M.toFreeYonedaCoproduct_fromFreeYonedaCoproduct
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    R✝ : CategoryTheory.Functor (Opposite C✝) RingCat
    C : Type u
    inst✝ : CategoryTheory.SmallCategory C
    R : CategoryTheory.Functor (Opposite C) RingCat
    M : PresheafOfModules R
    S : CategoryTheory.ShortComplex (PresheafOfModules R) := CategoryTheory.ShortC …
    ⊢ CategoryTheory.Limits.IsColimit M.freeYonedaCoproductsCokernelCofork
  -/
  let T := ShortComplex.mk _ _ (kernel.condition M.fromFreeYonedaCoproduct)
  let φ : S ⟶ T :=
    { τ₁ := fromFreeYonedaCoproduct _
      τ₂ := 𝟙 _
      τ₃ := 𝟙 _ }
  exact ((ShortComplex.exact_iff_of_epi_of_isIso_of_mono φ).2
    (T.exact_of_f_is_kernel (kernelIsKernel _))).gIsCokernel


