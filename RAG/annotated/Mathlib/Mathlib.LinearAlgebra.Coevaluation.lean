/-- The coevaluation map is a linear map from a field `K` to a finite dimensional
  vector space `V`. -/
def coevaluation : K →ₗ[K] V ⊗[K] Module.Dual K V :=
  let bV := Basis.ofVectorSpace K V
  (Basis.singleton Unit K).constr K fun _ =>
    ∑ i : Basis.ofVectorSpaceIndex K V, bV i ⊗ₜ[K] bV.coord i


theorem coevaluation_apply_one :
    (coevaluation K V) (1 : K) =
      let bV := Basis.ofVectorSpace K V
      ∑ i : Basis.ofVectorSpaceIndex K V, bV i ⊗ₜ[K] bV.coord i := by
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    ⊢ Eq ((coevaluation K V) 1)
        (let bV := Basis.ofVectorSpace K V;
        Finset.univ.sum fun i => TensorProduct.tmul K (bV i) (bV.coord i))
  -/
  simp only [coevaluation, id]
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    ⊢ Eq ((((Basis.singleton Unit K).constr K) fun x => Finset.univ.sum fun i => T …
  -/
  rw [(Basis.singleton Unit K).constr_apply_fintype K]
  simp only [Fintype.univ_punit, Finset.sum_const, one_smul, Basis.singleton_repr,
    Basis.equivFun_apply, Basis.coe_ofVectorSpace, one_nsmul, Finset.card_singleton]


/-- This lemma corresponds to one of the coherence laws for duals in rigid categories, see
  `CategoryTheory.Monoidal.Rigid`. -/
theorem contractLeft_assoc_coevaluation :
    (contractLeft K V).rTensor _ ∘ₗ
        (TensorProduct.assoc K _ _ _).symm.toLinearMap ∘ₗ
          (coevaluation K V).lTensor (Module.Dual K V) =
      (TensorProduct.lid K _).symm.toLinearMap ∘ₗ (TensorProduct.rid K _).toLinearMap := by
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)).comp ((↑(Tensor …
  -/
  letI := Classical.decEq (Basis.ofVectorSpaceIndex K V)
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)).comp ((↑(Tensor …
  -/
  apply TensorProduct.ext
  /-
    case H
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    ⊢ Eq ((TensorProduct.mk K (Module.Dual K V) K).compr₂ ((LinearMap.rTensor (Mod …
  -/
  apply (Basis.ofVectorSpace K V).dualBasis.ext; intro j; apply LinearMap.ext_ring
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((((TensorProduct.mk K (Module.Dual K V) K).compr₂ ((LinearMap.rTensor (M …
  -/
  rw [LinearMap.compr₂_apply, LinearMap.compr₂_apply, TensorProduct.mk_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)).comp ((↑(Tenso …
  -/
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_toLinearMap]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)) ((TensorProduct …
  -/
  rw [rid_tmul, one_smul, lid_symm_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)) ((TensorProduct …
  -/
  simp only [LinearEquiv.coe_toLinearMap, LinearMap.lTensor_tmul, coevaluation_apply_one]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)) ((TensorProduct …
  -/
  rw [TensorProduct.tmul_sum, map_sum]; simp only [assoc_symm_tmul]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.rTensor (Module.Dual K V) (contractLeft K V)) (Finset.univ.su …
  -/
  rw [map_sum]; simp only [LinearMap.rTensor_tmul, contractLeft_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (Finset.univ.sum fun x => TensorProduct.tmul K (((Basis.ofVectorSpace K V …
  -/
  simp only [Basis.coe_dualBasis, Basis.coord_apply, Basis.repr_self_apply, TensorProduct.ite_tmul]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (Finset.univ.sum fun x => ite (Eq x j) (TensorProduct.tmul K 1 ((Basis.of …
  -/
  rw [Finset.sum_ite_eq']; simp only [Finset.mem_univ, if_true]
                           /-
                             🎉 no goals
                           -/


/-- This lemma corresponds to one of the coherence laws for duals in rigid categories, see
  `CategoryTheory.Monoidal.Rigid`. -/
theorem contractLeft_assoc_coevaluation' :
    (contractLeft K V).lTensor _ ∘ₗ
        (TensorProduct.assoc K _ _ _).toLinearMap ∘ₗ (coevaluation K V).rTensor V =
      (TensorProduct.rid K _).symm.toLinearMap ∘ₗ (TensorProduct.lid K _).toLinearMap := by
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)).comp ((↑(TensorProduct.assoc K  …
  -/
  letI := Classical.decEq (Basis.ofVectorSpaceIndex K V)
  /-
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)).comp ((↑(TensorProduct.assoc K  …
  -/
  apply TensorProduct.ext
  /-
    case H
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    ⊢ Eq ((TensorProduct.mk K K V).compr₂ ((LinearMap.lTensor V (contractLeft K V) …
  -/
  apply LinearMap.ext_ring; apply (Basis.ofVectorSpace K V).ext; intro j
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((((TensorProduct.mk K K V).compr₂ ((LinearMap.lTensor V (contractLeft K  …
  -/
  rw [LinearMap.compr₂_apply, LinearMap.compr₂_apply, TensorProduct.mk_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (((LinearMap.lTensor V (contractLeft K V)).comp ((↑(TensorProduct.assoc K …
  -/
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_toLinearMap]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)) ((TensorProduct.assoc K V (Modu …
  -/
  rw [lid_tmul, one_smul, rid_symm_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)) ((TensorProduct.assoc K V (Modu …
  -/
  simp only [LinearEquiv.coe_toLinearMap, LinearMap.rTensor_tmul, coevaluation_apply_one]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)) ((TensorProduct.assoc K V (Modu …
  -/
  rw [TensorProduct.sum_tmul, map_sum]; simp only [assoc_tmul]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq ((LinearMap.lTensor V (contractLeft K V)) (Finset.univ.sum fun x => Tenso …
  -/
  rw [map_sum]; simp only [LinearMap.lTensor_tmul, contractLeft_apply]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (Finset.univ.sum fun x => TensorProduct.tmul K ((Basis.ofVectorSpace K V) …
  -/
  simp only [Basis.coord_apply, Basis.repr_self_apply, TensorProduct.tmul_ite]
  /-
    case H.h
    K : Type u
    inst✝³ : Field K
    V : Type v
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    this : DecidableEq ↑(Basis.ofVectorSpaceIndex K V) := Classical.decEq ↑(Basis. …
    j : ↑(Basis.ofVectorSpaceIndex K V)
    ⊢ Eq (Finset.univ.sum fun x => ite (Eq j x) (TensorProduct.tmul K ((Basis.ofVe …
  -/
  rw [Finset.sum_ite_eq]; simp only [Finset.mem_univ, if_true]
                          /-
                            🎉 no goals
                          -/


