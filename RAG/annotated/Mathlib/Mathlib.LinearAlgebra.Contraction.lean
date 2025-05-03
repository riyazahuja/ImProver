/-- The natural left-handed pairing between a module and its dual. -/
def contractLeft : Module.Dual R M ⊗[R] M →ₗ[R] R :=
  (uncurry _ _ _ _).toFun LinearMap.id

-- Porting note: doesn't like implicit ring in the tensor product

/-- The natural right-handed pairing between a module and its dual. -/
def contractRight : M ⊗[R] Module.Dual R M →ₗ[R] R :=
  (uncurry _ _ _ _).toFun (LinearMap.flip LinearMap.id)

-- Porting note: doesn't like implicit ring in the tensor product

/-- The natural map associating a linear map to the tensor product of two modules. -/
def dualTensorHom : Module.Dual R M ⊗[R] N →ₗ[R] M →ₗ[R] N :=
  let M' := Module.Dual R M
  (uncurry R M' N (M →ₗ[R] N) : _ → M' ⊗ N →ₗ[R] M →ₗ[R] N) LinearMap.smulRightₗ


@[simp]
theorem contractLeft_apply (f : Module.Dual R M) (m : M) : contractLeft R M (f ⊗ₜ m) = f m :=
  rfl


@[simp]
theorem contractRight_apply (f : Module.Dual R M) (m : M) : contractRight R M (m ⊗ₜ f) = f m :=
  rfl


@[simp]
theorem dualTensorHom_apply (f : Module.Dual R M) (m : M) (n : N) :
    dualTensorHom R M N (f ⊗ₜ n) m = f m • n :=
  rfl


@[simp]
theorem transpose_dualTensorHom (f : Module.Dual R M) (m : M) :
    Dual.transpose (R := R) (dualTensorHom R M M (f ⊗ₜ m)) =
    dualTensorHom R _ _ (Dual.eval R M m ⊗ₜ f) := by
  /-
    R : Type u
    M : Type v₁
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.Dual R M
    m : M
    ⊢ Eq (Module.Dual.transpose ((dualTensorHom R M M) (TensorProduct.tmul R f m)) …
  -/
  ext f' m'
  simp only [Dual.transpose_apply, coe_comp, Function.comp_apply, dualTensorHom_apply,
    LinearMap.map_smulₛₗ, RingHom.id_apply, Algebra.id.smul_eq_mul, Dual.eval_apply,
    LinearMap.smul_apply]
  /-
    case h.h
    R : Type u
    M : Type v₁
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f : Module.Dual R M
    m : M
    f' : Module.Dual R M
    m' : M
    ⊢ Eq (HMul.hMul (f m') (f' m)) (HMul.hMul (f' m) (f m'))
  -/
  exact mul_comm _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem dualTensorHom_prodMap_zero (f : Module.Dual R M) (p : P) :
    ((dualTensorHom R M P) (f ⊗ₜ[R] p)).prodMap (0 : N →ₗ[R] Q) =
      dualTensorHom R (M × N) (P × Q) ((f ∘ₗ fst R M N) ⊗ₜ inl R P Q p) := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    Q : Type v₄
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    f : Module.Dual R M
    p : P
    ⊢ Eq (((dualTensorHom R M P) (TensorProduct.tmul R f p)).prodMap 0) ((dualTens …
  -/
  ext <;>
    simp only [coe_comp, coe_inl, Function.comp_apply, prodMap_apply, dualTensorHom_apply,
      fst_apply, Prod.smul_mk, LinearMap.zero_apply, smul_zero]


@[simp]
theorem zero_prodMap_dualTensorHom (g : Module.Dual R N) (q : Q) :
    (0 : M →ₗ[R] P).prodMap ((dualTensorHom R N Q) (g ⊗ₜ[R] q)) =
      dualTensorHom R (M × N) (P × Q) ((g ∘ₗ snd R M N) ⊗ₜ inr R P Q q) := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    Q : Type v₄
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : Module.Dual R N
    q : Q
    ⊢ Eq (LinearMap.prodMap 0 ((dualTensorHom R N Q) (TensorProduct.tmul R g q)))  …
  -/
  ext <;>
    simp only [coe_comp, coe_inr, Function.comp_apply, prodMap_apply, dualTensorHom_apply,
      snd_apply, Prod.smul_mk, LinearMap.zero_apply, smul_zero]


theorem map_dualTensorHom (f : Module.Dual R M) (p : P) (g : Module.Dual R N) (q : Q) :
    TensorProduct.map (dualTensorHom R M P (f ⊗ₜ[R] p)) (dualTensorHom R N Q (g ⊗ₜ[R] q)) =
      dualTensorHom R (M ⊗[R] N) (P ⊗[R] Q) (dualDistrib R M N (f ⊗ₜ g) ⊗ₜ[R] p ⊗ₜ[R] q) := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    Q : Type v₄
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid P
    inst✝⁴ : AddCommMonoid Q
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    f : Module.Dual R M
    p : P
    g : Module.Dual R N
    q : Q
    ⊢ Eq (TensorProduct.map ((dualTensorHom R M P) (TensorProduct.tmul R f p)) ((d …
  -/
  ext m n
  simp only [compr₂_apply, mk_apply, map_tmul, dualTensorHom_apply, dualDistrib_apply, ←
    smul_tmul_smul]


@[simp]
theorem comp_dualTensorHom (f : Module.Dual R M) (n : N) (g : Module.Dual R N) (p : P) :
    dualTensorHom R N P (g ⊗ₜ[R] p) ∘ₗ dualTensorHom R M N (f ⊗ₜ[R] n) =
      g n • dualTensorHom R M P (f ⊗ₜ p) := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : Module.Dual R M
    n : N
    g : Module.Dual R N
    p : P
    ⊢ Eq (((dualTensorHom R N P) (TensorProduct.tmul R g p)).comp ((dualTensorHom  …
  -/
  ext m
  simp only [coe_comp, Function.comp_apply, dualTensorHom_apply, LinearMap.map_smul,
    RingHom.id_apply, LinearMap.smul_apply]
  /-
    case h
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : Module.Dual R M
    n : N
    g : Module.Dual R N
    p : P
    m : M
    ⊢ Eq (HSMul.hSMul (f m) (HSMul.hSMul (g n) p)) (HSMul.hSMul (g n) (HSMul.hSMul …
  -/
  rw [smul_comm]
  /-
    🎉 no goals
  -/


/-- As a matrix, `dualTensorHom` evaluated on a basis element of `M* ⊗ N` is a matrix with a
single one and zeros elsewhere -/
theorem toMatrix_dualTensorHom {m : Type*} {n : Type*} [Fintype m] [Finite n] [DecidableEq m]
    [DecidableEq n] (bM : Basis m R M) (bN : Basis n R N) (j : m) (i : n) :
    toMatrix bM bN (dualTensorHom R M N (bM.coord j ⊗ₜ bN i)) = stdBasisMatrix i j 1 := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    m : Type u_1
    n : Type u_2
    inst✝³ : Fintype m
    inst✝² : Finite n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    bM : Basis m R M
    bN : Basis n R N
    j : m
    i : n
    ⊢ Eq ((LinearMap.toMatrix bM bN) ((dualTensorHom R M N) (TensorProduct.tmul R  …
  -/
  ext i' j'
  /-
    case a
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    m : Type u_1
    n : Type u_2
    inst✝³ : Fintype m
    inst✝² : Finite n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    bM : Basis m R M
    bN : Basis n R N
    j : m
    i i' : n
    j' : m
    ⊢ Eq ((LinearMap.toMatrix bM bN) ((dualTensorHom R M N) (TensorProduct.tmul R  …
  -/
  by_cases hij : i = i' ∧ j = j' <;>
    /-
      case pos
      R : Type u
      M : Type v₁
      N : Type v₂
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      m : Type u_1
      n : Type u_2
      inst✝³ : Fintype m
      inst✝² : Finite n
      inst✝¹ : DecidableEq m
      inst✝ : DecidableEq n
      bM : Basis m R M
      bN : Basis n R N
      j : m
      i i' : n
      j' : m
      hij : And (Eq i i') (Eq j j')
      ⊢ Eq ((LinearMap.toMatrix bM bN) ((dualTensorHom R M N) (TensorProduct.tmul R  …
    -/
    /-
      🎉 no goals
    -/
    simp [LinearMap.toMatrix_apply, Finsupp.single_eq_pi_single, hij]
  /-
    case neg
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    m : Type u_1
    n : Type u_2
    inst✝³ : Fintype m
    inst✝² : Finite n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    bM : Basis m R M
    bN : Basis n R N
    j : m
    i i' : n
    j' : m
    hij : Not (And (Eq i i') (Eq j j'))
    ⊢ Eq (Pi.single i (Pi.single j' 1 j) i') 0
  -/
  rw [and_iff_not_or_not, Classical.not_not] at hij
  /-
    case neg
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    m : Type u_1
    n : Type u_2
    inst✝³ : Fintype m
    inst✝² : Finite n
    inst✝¹ : DecidableEq m
    inst✝ : DecidableEq n
    bM : Basis m R M
    bN : Basis n R N
    j : m
    i i' : n
    j' : m
    hij : Or (Not (Eq i i')) (Not (Eq j j'))
    ⊢ Eq (Pi.single i (Pi.single j' 1 j) i') 0
  -/
                              /-
                                🎉 no goals
                              -/
  cases' hij with hij hij <;> simp [hij]
                              /-
                                🎉 no goals
                              -/


/-- If `M` is free, the natural linear map $M^* ⊗ N → Hom(M, N)$ is an equivalence. This function
provides this equivalence in return for a basis of `M`. -/
-- @[simps! apply] -- Porting note: removed and created manually; malformed
noncomputable def dualTensorHomEquivOfBasis : Module.Dual R M ⊗[R] N ≃ₗ[R] M →ₗ[R] N :=
  LinearEquiv.ofLinear (dualTensorHom R M N)
    (∑ i, TensorProduct.mk R _ N (b.dualBasis i) ∘ₗ (LinearMap.applyₗ (R := R) (b i)))
    (by
      /-
        ι : Type w
        R : Type u
        M : Type v₁
        N : Type v₂
        P : Type v₃
        Q : Type v₄
        inst✝¹⁰ : CommRing R
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : AddCommGroup N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : AddCommGroup Q
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : Module R Q
        inst✝¹ : DecidableEq ι
        inst✝ : Fintype ι
        b : Basis ι R M
        ⊢ Eq ((dualTensorHom R M N).comp (Finset.univ.sum fun i => ((TensorProduct.mk  …
      -/
      ext f m
      simp only [applyₗ_apply_apply, coeFn_sum, dualTensorHom_apply, mk_apply, id_coe, _root_.id,
        Fintype.sum_apply, Function.comp_apply, Basis.coe_dualBasis, coe_comp, Basis.coord_apply, ←
        f.map_smul, _root_.map_sum (dualTensorHom R M N), ← _root_.map_sum f, b.sum_repr])
    (by
      /-
        ι : Type w
        R : Type u
        M : Type v₁
        N : Type v₂
        P : Type v₃
        Q : Type v₄
        inst✝¹⁰ : CommRing R
        inst✝⁹ : AddCommGroup M
        inst✝⁸ : AddCommGroup N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : AddCommGroup Q
        inst✝⁵ : Module R M
        inst✝⁴ : Module R N
        inst✝³ : Module R P
        inst✝² : Module R Q
        inst✝¹ : DecidableEq ι
        inst✝ : Fintype ι
        b : Basis ι R M
        ⊢ Eq ((Finset.univ.sum fun i => ((TensorProduct.mk R (Module.Dual R M) N) (b.d …
      -/
      ext f m
      simp only [applyₗ_apply_apply, coeFn_sum, dualTensorHom_apply, mk_apply, id_coe, _root_.id,
        Fintype.sum_apply, Function.comp_apply, Basis.coe_dualBasis, coe_comp, compr₂_apply,
        tmul_smul, smul_tmul', ← sum_tmul, Basis.sum_dual_apply_smul_coord])


@[simp]
theorem dualTensorHomEquivOfBasis_apply (x : Module.Dual R M ⊗[R] N) :
    (dualTensorHomEquivOfBasis (N := N) b :
    Module.Dual R M ⊗[R] N → (M →ₗ[R] N)) x = (dualTensorHom R M N) x := by
  /-
    ι : Type w
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    x : TensorProduct R (Module.Dual R M) N
    ⊢ Eq ((dualTensorHomEquivOfBasis b) x) ((dualTensorHom R M N) x)
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem dualTensorHomEquivOfBasis_toLinearMap :
    (dualTensorHomEquivOfBasis b : Module.Dual R M ⊗[R] N ≃ₗ[R] M →ₗ[R] N).toLinearMap =
      dualTensorHom R M N :=
  rfl

-- Porting note: should N be explicit in dualTensorHomEquivOfBasis?

@[simp]
theorem dualTensorHomEquivOfBasis_symm_cancel_left (x : Module.Dual R M ⊗[R] N) :
    (dualTensorHomEquivOfBasis (N := N) b).symm (dualTensorHom R M N x) = x := by
  rw [← dualTensorHomEquivOfBasis_apply b,
    LinearEquiv.symm_apply_apply <| dualTensorHomEquivOfBasis (N := N) b]


@[simp]
theorem dualTensorHomEquivOfBasis_symm_cancel_right (x : M →ₗ[R] N) :
    dualTensorHom R M N ((dualTensorHomEquivOfBasis (N := N) b).symm x) = x := by
  /-
    ι : Type w
    R : Type u
    M : Type v₁
    N : Type v₂
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    b : Basis ι R M
    x : LinearMap (RingHom.id R) M N
    ⊢ Eq ((dualTensorHom R M N) ((dualTensorHomEquivOfBasis b).symm x)) x
  -/
  rw [← dualTensorHomEquivOfBasis_apply b, LinearEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- If `M` is finite free, the natural map $M^* ⊗ N → Hom(M, N)$ is an
equivalence. -/
@[simp]
noncomputable def dualTensorHomEquiv : Module.Dual R M ⊗[R] N ≃ₗ[R] M →ₗ[R] N :=
  dualTensorHomEquivOfBasis (Module.Free.chooseBasis R M)


/-- When `M` is a finite free module, the map `lTensorHomToHomLTensor` is an equivalence. Note
that `lTensorHomEquivHomLTensor` is not defined directly in terms of
`lTensorHomToHomLTensor`, but the equivalence between the two is given by
`lTensorHomEquivHomLTensor_toLinearMap` and `lTensorHomEquivHomLTensor_apply`. -/
noncomputable def lTensorHomEquivHomLTensor : P ⊗[R] (M →ₗ[R] Q) ≃ₗ[R] M →ₗ[R] P ⊗[R] Q :=
  congr (LinearEquiv.refl R P) (dualTensorHomEquiv R M Q).symm ≪≫ₗ
      TensorProduct.leftComm R P _ Q ≪≫ₗ
    dualTensorHomEquiv R M _


/-- When `M` is a finite free module, the map `rTensorHomToHomRTensor` is an equivalence. Note
that `rTensorHomEquivHomRTensor` is not defined directly in terms of
`rTensorHomToHomRTensor`, but the equivalence between the two is given by
`rTensorHomEquivHomRTensor_toLinearMap` and `rTensorHomEquivHomRTensor_apply`. -/
noncomputable def rTensorHomEquivHomRTensor : (M →ₗ[R] P) ⊗[R] Q ≃ₗ[R] M →ₗ[R] P ⊗[R] Q :=
  congr (dualTensorHomEquiv R M P).symm (LinearEquiv.refl R Q) ≪≫ₗ TensorProduct.assoc R _ P Q ≪≫ₗ
    dualTensorHomEquiv R M _


@[simp]
theorem lTensorHomEquivHomLTensor_toLinearMap :
    (lTensorHomEquivHomLTensor R M P Q).toLinearMap = lTensorHomToHomLTensor R M P Q := by
  classical -- Porting note: missing decidable for choosing basis
  let e := congr (LinearEquiv.refl R P) (dualTensorHomEquiv R M Q)
  have h : Function.Surjective e.toLinearMap := e.surjective
  refine (cancel_right h).1 ?_
  ext f q m
  simp only [e, lTensorHomEquivHomLTensor, dualTensorHomEquiv, LinearEquiv.comp_coe, compr₂_apply,
    mk_apply, LinearEquiv.coe_coe, LinearEquiv.trans_apply, congr_tmul, LinearEquiv.refl_apply,
    dualTensorHomEquivOfBasis_apply, dualTensorHomEquivOfBasis_symm_cancel_left, leftComm_tmul,
    dualTensorHom_apply, coe_comp, Function.comp_apply, lTensorHomToHomLTensor_apply, tmul_smul]


@[simp]
theorem rTensorHomEquivHomRTensor_toLinearMap :
    (rTensorHomEquivHomRTensor R M P Q).toLinearMap = rTensorHomToHomRTensor R M P Q := by
  classical -- Porting note: missing decidable for choosing basis
  let e := congr (dualTensorHomEquiv R M P) (LinearEquiv.refl R Q)
  have h : Function.Surjective e.toLinearMap := e.surjective
  refine (cancel_right h).1 ?_
  ext f p q m
  simp only [e, rTensorHomEquivHomRTensor, dualTensorHomEquiv, compr₂_apply, mk_apply, coe_comp,
    LinearEquiv.coe_toLinearMap, Function.comp_apply, map_tmul, LinearEquiv.coe_coe,
    dualTensorHomEquivOfBasis_apply, LinearEquiv.trans_apply, congr_tmul,
    dualTensorHomEquivOfBasis_symm_cancel_left, LinearEquiv.refl_apply, assoc_tmul,
    dualTensorHom_apply, rTensorHomToHomRTensor_apply, smul_tmul']


@[simp]
theorem lTensorHomEquivHomLTensor_apply (x : P ⊗[R] (M →ₗ[R] Q)) :
    lTensorHomEquivHomLTensor R M P Q x = lTensorHomToHomLTensor R M P Q x := by
  /-
    R : Type u
    M : Type v₁
    P : Type v₃
    Q : Type v₄
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup P
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R M
    inst✝³ : Module R P
    inst✝² : Module R Q
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    x : TensorProduct R P (LinearMap (RingHom.id R) M Q)
    ⊢ Eq ((lTensorHomEquivHomLTensor R M P Q) x) ((TensorProduct.lTensorHomToHomLT …
  -/
  rw [← LinearEquiv.coe_toLinearMap, lTensorHomEquivHomLTensor_toLinearMap]
  /-
    🎉 no goals
  -/


@[simp]
theorem rTensorHomEquivHomRTensor_apply (x : (M →ₗ[R] P) ⊗[R] Q) :
    rTensorHomEquivHomRTensor R M P Q x = rTensorHomToHomRTensor R M P Q x := by
  /-
    R : Type u
    M : Type v₁
    P : Type v₃
    Q : Type v₄
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup P
    inst✝⁵ : AddCommGroup Q
    inst✝⁴ : Module R M
    inst✝³ : Module R P
    inst✝² : Module R Q
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    x : TensorProduct R (LinearMap (RingHom.id R) M P) Q
    ⊢ Eq ((rTensorHomEquivHomRTensor R M P Q) x) ((TensorProduct.rTensorHomToHomRT …
  -/
  rw [← LinearEquiv.coe_toLinearMap, rTensorHomEquivHomRTensor_toLinearMap]
  /-
    🎉 no goals
  -/


/-- When `M` and `N` are free `R` modules, the map `homTensorHomMap` is an equivalence. Note that
`homTensorHomEquiv` is not defined directly in terms of `homTensorHomMap`, but the equivalence
between the two is given by `homTensorHomEquiv_toLinearMap` and `homTensorHomEquiv_apply`.
-/
noncomputable def homTensorHomEquiv : (M →ₗ[R] P) ⊗[R] (N →ₗ[R] Q) ≃ₗ[R] M ⊗[R] N →ₗ[R] P ⊗[R] Q :=
  rTensorHomEquivHomRTensor R M P _ ≪≫ₗ
      (LinearEquiv.refl R M).arrowCongr (lTensorHomEquivHomLTensor R N _ Q) ≪≫ₗ
    lift.equiv R M N _


@[simp]
theorem homTensorHomEquiv_toLinearMap :
    (homTensorHomEquiv R M N P Q).toLinearMap = homTensorHomMap R M N P Q := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    Q : Type v₄
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : AddCommGroup Q
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module R P
    inst✝⁴ : Module R Q
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    ⊢ Eq (↑(homTensorHomEquiv R M N P Q)) (TensorProduct.homTensorHomMap R M N P Q)
  -/
  ext m n
  simp only [homTensorHomEquiv, compr₂_apply, mk_apply, LinearEquiv.coe_toLinearMap,
    LinearEquiv.trans_apply, lift.equiv_apply, LinearEquiv.arrowCongr_apply, LinearEquiv.refl_symm,
    LinearEquiv.refl_apply, rTensorHomEquivHomRTensor_apply, lTensorHomEquivHomLTensor_apply,
    lTensorHomToHomLTensor_apply, rTensorHomToHomRTensor_apply, homTensorHomMap_apply,
    map_tmul]


@[simp]
theorem homTensorHomEquiv_apply (x : (M →ₗ[R] P) ⊗[R] (N →ₗ[R] Q)) :
    homTensorHomEquiv R M N P Q x = homTensorHomMap R M N P Q x := by
  /-
    R : Type u
    M : Type v₁
    N : Type v₂
    P : Type v₃
    Q : Type v₄
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : AddCommGroup Q
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module R P
    inst✝⁴ : Module R Q
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    x : TensorProduct R (LinearMap (RingHom.id R) M P) (LinearMap (RingHom.id R) N …
    ⊢ Eq ((homTensorHomEquiv R M N P Q) x) ((TensorProduct.homTensorHomMap R M N P …
  -/
  rw [← LinearEquiv.coe_toLinearMap, homTensorHomEquiv_toLinearMap]
  /-
    🎉 no goals
  -/


