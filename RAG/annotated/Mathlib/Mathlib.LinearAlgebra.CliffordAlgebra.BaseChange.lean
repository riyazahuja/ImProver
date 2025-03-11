/-- Auxiliary construction: note this is really just a heterobasic `CliffordAlgebra.map`. -/
-- `noncomputable` is a performance workaround for https://github.com/leanprover-community/mathlib4/issues/7103
noncomputable def ofBaseChangeAux (Q : QuadraticForm R V) :
    CliffordAlgebra Q →ₐ[R] CliffordAlgebra (Q.baseChange A) :=
  CliffordAlgebra.lift Q <| by
    /-
      R : Type u_1
      A : Type u_2
      V : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : AddCommGroup V
      inst✝² : Algebra R A
      inst✝¹ : Module R V
      inst✝ : Invertible 2
      Q : QuadraticForm R V
      ⊢ Subtype fun f => ∀ (m : V), Eq (HMul.hMul (f m) (f m)) ((algebraMap R (Cliff …
    -/
    refine ⟨(ι (Q.baseChange A)).restrictScalars R ∘ₗ TensorProduct.mk R A V 1, fun v => ?_⟩
    /-
      R : Type u_1
      A : Type u_2
      V : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : AddCommGroup V
      inst✝² : Algebra R A
      inst✝¹ : Module R V
      inst✝ : Invertible 2
      Q : QuadraticForm R V
      v : V
      ⊢ Eq (HMul.hMul (((↑R (CliffordAlgebra.ι (QuadraticForm.baseChange A Q))).comp …
    -/
    refine (CliffordAlgebra.ι_sq_scalar (Q.baseChange A) (1 ⊗ₜ v)).trans ?_
    rw [QuadraticForm.baseChange_tmul, one_mul, ← Algebra.algebraMap_eq_smul_one,
      ← IsScalarTower.algebraMap_apply]


@[simp] theorem ofBaseChangeAux_ι (Q : QuadraticForm R V) (v : V) :
    ofBaseChangeAux A Q (ι Q v) = ι (Q.baseChange A) (1 ⊗ₜ v) :=
  CliffordAlgebra.lift_ι_apply _ _ v


/-- Convert from the base-changed clifford algebra to the clifford algebra over a base-changed
module. -/
-- `noncomputable` is a performance workaround for https://github.com/leanprover-community/mathlib4/issues/7103
noncomputable def ofBaseChange (Q : QuadraticForm R V) :
    A ⊗[R] CliffordAlgebra Q →ₐ[A] CliffordAlgebra (Q.baseChange A) :=
  Algebra.TensorProduct.lift (Algebra.ofId _ _) (ofBaseChangeAux A Q)
    fun _a _x => Algebra.commutes _ _


@[simp] theorem ofBaseChange_tmul_ι (Q : QuadraticForm R V) (z : A) (v : V) :
    ofBaseChange A Q (z ⊗ₜ ι Q v) = ι (Q.baseChange A) (z ⊗ₜ v) := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    z : A
    v : V
    ⊢ Eq ((CliffordAlgebra.ofBaseChange A Q) (TensorProduct.tmul R z ((CliffordAlg …
  -/
  show algebraMap _ _ z * ofBaseChangeAux A Q (ι Q v) = ι (Q.baseChange A) (z ⊗ₜ[R] v)
  rw [ofBaseChangeAux_ι, ← Algebra.smul_def, ← map_smul, TensorProduct.smul_tmul', smul_eq_mul,
    mul_one]


@[simp] theorem ofBaseChange_tmul_one (Q : QuadraticForm R V) (z : A) :
    ofBaseChange A Q (z ⊗ₜ 1) = algebraMap _ _ z := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    z : A
    ⊢ Eq ((CliffordAlgebra.ofBaseChange A Q) (TensorProduct.tmul R z 1)) ((algebra …
  -/
  show algebraMap _ _ z * ofBaseChangeAux A Q 1 = _
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    z : A
    ⊢ Eq (HMul.hMul ((algebraMap A (CliffordAlgebra (QuadraticForm.baseChange A Q) …
  -/
  rw [map_one, mul_one]
  /-
    🎉 no goals
  -/


/-- Convert from the clifford algebra over a base-changed module to the base-changed clifford
algebra. -/
-- `noncomputable` is a performance workaround for https://github.com/leanprover-community/mathlib4/issues/7103
noncomputable def toBaseChange (Q : QuadraticForm R V) :
    CliffordAlgebra (Q.baseChange A) →ₐ[A] A ⊗[R] CliffordAlgebra Q :=
  CliffordAlgebra.lift _ <| by
    /-
      R : Type u_1
      A : Type u_2
      V : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : AddCommGroup V
      inst✝² : Algebra R A
      inst✝¹ : Module R V
      inst✝ : Invertible 2
      Q : QuadraticForm R V
      ⊢ Subtype fun f => ∀ (m : TensorProduct R A V), Eq (HMul.hMul (f m) (f m)) ((a …
    -/
    refine ⟨TensorProduct.AlgebraTensorModule.map (LinearMap.id : A →ₗ[A] A) (ι Q), ?_⟩
    /-
      R : Type u_1
      A : Type u_2
      V : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : AddCommGroup V
      inst✝² : Algebra R A
      inst✝¹ : Module R V
      inst✝ : Invertible 2
      Q : QuadraticForm R V
      ⊢ ∀ (m : TensorProduct R A V), Eq (HMul.hMul ((TensorProduct.AlgebraTensorModu …
    -/
    letI : Invertible (2 : A) := (Invertible.map (algebraMap R A) 2).copy 2 (map_ofNat _ _).symm
    letI : Invertible (2 : A ⊗[R] CliffordAlgebra Q) :=
      (Invertible.map (algebraMap R _) 2).copy 2 (map_ofNat _ _).symm
    suffices hpure_tensor : ∀ v w, (1 * 1) ⊗ₜ[R] (ι Q v * ι Q w) + (1 * 1) ⊗ₜ[R] (ι Q w * ι Q v) =
        QuadraticMap.polarBilin (Q.baseChange A) (1 ⊗ₜ[R] v) (1 ⊗ₜ[R] w) ⊗ₜ[R] 1 by
      -- the crux is that by converting to a statement about linear maps instead of quadratic forms,
      -- we then have access to all the partially-applied `ext` lemmas.
      rw [CliffordAlgebra.forall_mul_self_eq_iff (isUnit_of_invertible _)]
      refine TensorProduct.AlgebraTensorModule.curry_injective ?_
      ext v w
      dsimp
      exact hpure_tensor v w
    /-
      R : Type u_1
      A : Type u_2
      V : Type u_3
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing A
      inst✝³ : AddCommGroup V
      inst✝² : Algebra R A
      inst✝¹ : Module R V
      inst✝ : Invertible 2
      Q : QuadraticForm R V
      this✝ : Invertible 2 := (Invertible.map (algebraMap R A) 2).copy 2 ⋯
      this : Invertible 2 := (Invertible.map (algebraMap R (TensorProduct R A (Cliff …
      ⊢ ∀ (v w : V), Eq (HAdd.hAdd (TensorProduct.tmul R (HMul.hMul 1 1) (HMul.hMul  …
    -/
    intros v w
    rw [← TensorProduct.tmul_add, CliffordAlgebra.ι_mul_ι_add_swap,
      QuadraticForm.polarBilin_baseChange, LinearMap.BilinForm.baseChange_tmul, one_mul,
      TensorProduct.smul_tmul, Algebra.algebraMap_eq_smul_one, QuadraticMap.polarBilin_apply_apply]


@[simp] theorem toBaseChange_ι (Q : QuadraticForm R V) (z : A) (v : V) :
    toBaseChange A Q (ι (Q.baseChange A) (z ⊗ₜ v)) = z ⊗ₜ ι Q v :=
  CliffordAlgebra.lift_ι_apply _ _ _


theorem toBaseChange_comp_involute (Q : QuadraticForm R V) :
    (toBaseChange A Q).comp (involute : CliffordAlgebra (Q.baseChange A) →ₐ[A] _) =
      (Algebra.TensorProduct.map (AlgHom.id _ _) involute).comp (toBaseChange A Q) := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    ⊢ Eq ((CliffordAlgebra.toBaseChange A Q).comp CliffordAlgebra.involute) ((Alge …
  -/
  ext v
  show toBaseChange A Q (involute (ι (Q.baseChange A) (1 ⊗ₜ[R] v)))
    = (Algebra.TensorProduct.map (AlgHom.id _ _) involute :
        A ⊗[R] CliffordAlgebra Q →ₐ[A] _)
      (toBaseChange A Q (ι (Q.baseChange A) (1 ⊗ₜ[R] v)))
  rw [toBaseChange_ι, involute_ι, map_neg (toBaseChange A Q), toBaseChange_ι,
    Algebra.TensorProduct.map_tmul, AlgHom.id_apply, involute_ι, TensorProduct.tmul_neg]


/-- The involution acts only on the right of the tensor product. -/
theorem toBaseChange_involute (Q : QuadraticForm R V) (x : CliffordAlgebra (Q.baseChange A)) :
    toBaseChange A Q (involute x) =
      TensorProduct.map LinearMap.id (involute.toLinearMap) (toBaseChange A Q x) :=
  DFunLike.congr_fun (toBaseChange_comp_involute A Q) x


/-- Auxiliary theorem used to prove `toBaseChange_reverse` without needing induction. -/
theorem toBaseChange_comp_reverseOp (Q : QuadraticForm R V) :
    (toBaseChange A Q).op.comp reverseOp =
      ((Algebra.TensorProduct.opAlgEquiv R A A (CliffordAlgebra Q)).toAlgHom.comp <|
        (Algebra.TensorProduct.map
          (AlgEquiv.toOpposite A A).toAlgHom (reverseOp (Q := Q))).comp
        (toBaseChange A Q)) := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    ⊢ Eq ((AlgHom.op (CliffordAlgebra.toBaseChange A Q)).comp CliffordAlgebra.reve …
  -/
  ext v
  show op (toBaseChange A Q (reverse (ι (Q.baseChange A) (1 ⊗ₜ[R] v)))) =
    Algebra.TensorProduct.opAlgEquiv R A A (CliffordAlgebra Q)
      (Algebra.TensorProduct.map (AlgEquiv.toOpposite A A).toAlgHom (reverseOp (Q := Q))
        (toBaseChange A Q (ι (Q.baseChange A) (1 ⊗ₜ[R] v))))
  rw [toBaseChange_ι, reverse_ι, toBaseChange_ι, Algebra.TensorProduct.map_tmul,
    Algebra.TensorProduct.opAlgEquiv_tmul, reverseOp_ι]
  /-
    case a.a.h.h
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    v : V
    ⊢ Eq (MulOpposite.op (TensorProduct.tmul R 1 ((CliffordAlgebra.ι Q) v))) (MulO …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `reverse` acts only on the right of the tensor product. -/
theorem toBaseChange_reverse (Q : QuadraticForm R V) (x : CliffordAlgebra (Q.baseChange A)) :
    toBaseChange A Q (reverse x) =
      TensorProduct.map LinearMap.id reverse (toBaseChange A Q x) := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    ⊢ Eq ((CliffordAlgebra.toBaseChange A Q) (CliffordAlgebra.reverse x)) ((Tensor …
  -/
  have := DFunLike.congr_fun (toBaseChange_comp_reverseOp A Q) x
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    this : Eq (((AlgHom.op (CliffordAlgebra.toBaseChange A Q)).comp CliffordAlgebr …
    ⊢ Eq ((CliffordAlgebra.toBaseChange A Q) (CliffordAlgebra.reverse x)) ((Tensor …
  -/
  refine (congr_arg unop this).trans ?_; clear this
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    ⊢ Eq (MulOpposite.unop (((↑(Algebra.TensorProduct.opAlgEquiv R A A (CliffordAl …
  -/
  refine (LinearMap.congr_fun (TensorProduct.AlgebraTensorModule.map_comp _ _ _ _).symm _).trans ?_
  rw [reverse, ← AlgEquiv.toLinearMap, ← AlgEquiv.toLinearEquiv_toLinearMap,
    AlgEquiv.toLinearEquiv_toOpposite]
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    ⊢ Eq ((TensorProduct.AlgebraTensorModule.map ((↑(MulOpposite.opLinearEquiv A). …
  -/
  dsimp
  -- `simp` fails here due to a timeout looking for a `Subsingleton` instance!?
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    ⊢ Eq ((TensorProduct.AlgebraTensorModule.map (↑((MulOpposite.opLinearEquiv A). …
  -/
  rw [LinearEquiv.self_trans_symm]
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : CliffordAlgebra (QuadraticForm.baseChange A Q)
    ⊢ Eq ((TensorProduct.AlgebraTensorModule.map (↑(LinearEquiv.refl A A)) ((↑(Mul …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toBaseChange_comp_ofBaseChange (Q : QuadraticForm R V) :
    (toBaseChange A Q).comp (ofBaseChange A Q) = AlgHom.id _ _ := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    ⊢ Eq ((CliffordAlgebra.toBaseChange A Q).comp (CliffordAlgebra.ofBaseChange A  …
  -/
  ext v
  /-
    case hb.a.h
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    v : V
    ⊢ Eq ((((AlgHom.restrictScalars R ((CliffordAlgebra.toBaseChange A Q).comp (Cl …
  -/
  change toBaseChange A Q (ofBaseChange A Q (1 ⊗ₜ[R] ι Q v)) = 1 ⊗ₜ[R] ι Q v
  /-
    case hb.a.h
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    v : V
    ⊢ Eq ((CliffordAlgebra.toBaseChange A Q) ((CliffordAlgebra.ofBaseChange A Q) ( …
  -/
  rw [ofBaseChange_tmul_ι, toBaseChange_ι]
  /-
    🎉 no goals
  -/


@[simp] theorem toBaseChange_ofBaseChange (Q : QuadraticForm R V) (x : A ⊗[R] CliffordAlgebra Q) :
    toBaseChange A Q (ofBaseChange A Q x) = x :=
  AlgHom.congr_fun (toBaseChange_comp_ofBaseChange A Q : _) x


theorem ofBaseChange_comp_toBaseChange (Q : QuadraticForm R V) :
    (ofBaseChange A Q).comp (toBaseChange A Q) = AlgHom.id _ _ := by
  /-
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    ⊢ Eq ((CliffordAlgebra.ofBaseChange A Q).comp (CliffordAlgebra.toBaseChange A  …
  -/
  ext x
  show ofBaseChange A Q (toBaseChange A Q (ι (Q.baseChange A) (1 ⊗ₜ[R] x)))
    = ι (Q.baseChange A) (1 ⊗ₜ[R] x)
  /-
    case a.a.h.h
    R : Type u_1
    A : Type u_2
    V : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : AddCommGroup V
    inst✝² : Algebra R A
    inst✝¹ : Module R V
    inst✝ : Invertible 2
    Q : QuadraticForm R V
    x : V
    ⊢ Eq ((CliffordAlgebra.ofBaseChange A Q) ((CliffordAlgebra.toBaseChange A Q) ( …
  -/
  rw [toBaseChange_ι, ofBaseChange_tmul_ι]
  /-
    🎉 no goals
  -/


@[simp] theorem ofBaseChange_toBaseChange
    (Q : QuadraticForm R V) (x : CliffordAlgebra (Q.baseChange A)) :
    ofBaseChange A Q (toBaseChange A Q x) = x :=
  AlgHom.congr_fun (ofBaseChange_comp_toBaseChange A Q : _) x


/-- Base-changing the vector space of a clifford algebra is isomorphic as an A-algebra to
base-changing the clifford algebra itself; <|Cℓ(A ⊗_R V, Q_A) ≅ A ⊗_R Cℓ(V, Q)<|.

This is `CliffordAlgebra.toBaseChange` and `CliffordAlgebra.ofBaseChange` as an equivalence. -/
@[simps!]
-- `noncomputable` is a performance workaround for https://github.com/leanprover-community/mathlib4/issues/7103
noncomputable def equivBaseChange (Q : QuadraticForm R V) :
    CliffordAlgebra (Q.baseChange A) ≃ₐ[A] A ⊗[R] CliffordAlgebra Q :=
  AlgEquiv.ofAlgHom (toBaseChange A Q) (ofBaseChange A Q)
    (toBaseChange_comp_ofBaseChange A Q)
    (ofBaseChange_comp_toBaseChange A Q)


