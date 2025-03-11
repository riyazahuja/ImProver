/-- The (finite) Galois group `Gal(L / k)` associated to a
`L : FiniteGaloisIntermediateField k K` `L`. -/
def FiniteGaloisIntermediateField.finGaloisGroup (L : FiniteGaloisIntermediateField k K) :
    FiniteGrp :=
  letI := AlgEquiv.fintype k L
  FiniteGrp.of <| L ≃ₐ[k] L


/-- For `FiniteGaloisIntermediateField` s `L₁` and `L₂` with `L₂ ≤ L₁`
  the restriction homomorphism from `Gal(L₁/k)` to `Gal(L₂/k)` -/
noncomputable def finGaloisGroupMap {L₁ L₂ : (FiniteGaloisIntermediateField k K)ᵒᵖ}
    (le : L₁ ⟶ L₂) : L₁.unop.finGaloisGroup ⟶ L₂.unop.finGaloisGroup :=
  haveI : Normal k L₂.unop := IsGalois.to_normal
  letI : Algebra L₂.unop L₁.unop := RingHom.toAlgebra (Subsemiring.inclusion <| leOfHom le.1)
  haveI : IsScalarTower k L₂.unop L₁.unop := IsScalarTower.of_algebraMap_eq (congrFun rfl)
  FiniteGrp.ofHom (AlgEquiv.restrictNormalHom L₂.unop)


@[simp]
lemma map_id (L : (FiniteGaloisIntermediateField k K)ᵒᵖ) :
    (finGaloisGroupMap (𝟙 L)) = 𝟙 L.unop.finGaloisGroup :=
  AlgEquiv.restrictNormalHom_id _ _


@[simp]
lemma map_comp {L₁ L₂ L₃ : (FiniteGaloisIntermediateField k K)ᵒᵖ} (f : L₁ ⟶ L₂) (g : L₂ ⟶ L₃) :
    finGaloisGroupMap (f ≫ g) = finGaloisGroupMap f ≫ finGaloisGroupMap g := by
  iterate 2
    induction L₁ with | _ L₁ => ?_
    induction L₂ with | _ L₂ => ?_
    induction L₃ with | _ L₃ => ?_
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  letI : Algebra L₃ L₂ := RingHom.toAlgebra (Subsemiring.inclusion g.unop.le)
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Member …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  letI : Algebra L₂ L₁ := RingHom.toAlgebra (Subsemiring.inclusion f.unop.le)
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Membe …
    this : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Member …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  letI : Algebra L₃ L₁ := RingHom.toAlgebra (Subsemiring.inclusion (g.unop.le.trans f.unop.le))
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝¹ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝ : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Membe …
    this : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Member …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  haveI : IsScalarTower k L₂ L₁ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝² : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝¹ : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Memb …
    this✝ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Membe …
    this : IsScalarTower k (Subtype fun x => Membership.mem L₂ x) (Subtype fun x = …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  haveI : IsScalarTower k L₃ L₁ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝³ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝² : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Memb …
    this✝¹ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝ : IsScalarTower k (Subtype fun x => Membership.mem L₂ x) (Subtype fun x  …
    this : IsScalarTower k (Subtype fun x => Membership.mem L₃ x) (Subtype fun x = …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  haveI : IsScalarTower k L₃ L₂ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝⁴ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝³ : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Memb …
    this✝² : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝¹ : IsScalarTower k (Subtype fun x => Membership.mem L₂ x) (Subtype fun x …
    this✝ : IsScalarTower k (Subtype fun x => Membership.mem L₃ x) (Subtype fun x  …
    this : IsScalarTower k (Subtype fun x => Membership.mem L₃ x) (Subtype fun x = …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  haveI : IsScalarTower L₃ L₂ L₁ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case h.h.h.mk.mk.mk
    k : Type u_1
    K : Type u_2
    inst✝² : Field k
    inst✝¹ : Field K
    inst✝ : Algebra k K
    L₁ : IntermediateField k K
    finiteDimensional✝² : FiniteDimensional k (Subtype fun x => Membership.mem L₁ x)
    isGalois✝² : IsGalois k (Subtype fun x => Membership.mem L₁ x)
    L₂ : IntermediateField k K
    finiteDimensional✝¹ : FiniteDimensional k (Subtype fun x => Membership.mem L₂ x)
    isGalois✝¹ : IsGalois k (Subtype fun x => Membership.mem L₂ x)
    f : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₁ } { unop := Finit …
    L₃ : IntermediateField k K
    finiteDimensional✝ : FiniteDimensional k (Subtype fun x => Membership.mem L₃ x)
    isGalois✝ : IsGalois k (Subtype fun x => Membership.mem L₃ x)
    g : Quiver.Hom { unop := FiniteGaloisIntermediateField.mk L₂ } { unop := Finit …
    this✝⁵ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝⁴ : Algebra (Subtype fun x => Membership.mem L₂ x) (Subtype fun x => Memb …
    this✝³ : Algebra (Subtype fun x => Membership.mem L₃ x) (Subtype fun x => Memb …
    this✝² : IsScalarTower k (Subtype fun x => Membership.mem L₂ x) (Subtype fun x …
    this✝¹ : IsScalarTower k (Subtype fun x => Membership.mem L₃ x) (Subtype fun x …
    this✝ : IsScalarTower k (Subtype fun x => Membership.mem L₃ x) (Subtype fun x  …
    this : IsScalarTower (Subtype fun x => Membership.mem L₃ x) (Subtype fun x =>  …
    ⊢ Eq (finGaloisGroupMap (CategoryTheory.CategoryStruct.comp f g)) (CategoryThe …
  -/
  apply IsScalarTower.AlgEquiv.restrictNormalHom_comp k L₃ L₂ L₁
  /-
    🎉 no goals
  -/


variable (k K) in
/-- The functor from `FiniteGaloisIntermediateField` (ordered by reverse inclusion) to `FiniteGrp`,
mapping each intermediate field `K/L/k` to `Gal (L/k)`.-/
noncomputable def finGaloisGroupFunctor : (FiniteGaloisIntermediateField k K)ᵒᵖ ⥤ FiniteGrp where
  obj L := L.unop.finGaloisGroup
  map := finGaloisGroupMap
  map_id := finGaloisGroupMap.map_id
  map_comp := finGaloisGroupMap.map_comp


