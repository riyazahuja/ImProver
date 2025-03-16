instance (priority := 90) [SMul R ℝ] [SMul S ℝ] [SMulCommClass R S ℝ] : SMulCommClass R S ℂ where
                        /-
                          R : Type u_1
                          S : Type u_2
                          inst✝² : SMul R Real
                          inst✝¹ : SMul S Real
                          inst✝ : SMulCommClass R S Real
                          r : R
                          s : S
                          x : Complex
                          ⊢ Eq (HSMul.hSMul r (HSMul.hSMul s x)) (HSMul.hSMul s (HSMul.hSMul r x))
                        -/
                                /-
                                  🎉 no goals
                                -/
  smul_comm r s x := by ext <;> simp [smul_re, smul_im, smul_comm]
                                /-
                                  🎉 no goals
                                -/

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 90) [SMul R S] [SMul R ℝ] [SMul S ℝ] [IsScalarTower R S ℝ] :
    IsScalarTower R S ℂ where
                         /-
                           R : Type u_1
                           S : Type u_2
                           inst✝³ : SMul R S
                           inst✝² : SMul R Real
                           inst✝¹ : SMul S Real
                           inst✝ : IsScalarTower R S Real
                           r : R
                           s : S
                           x : Complex
                           ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
                         -/
                                 /-
                                   🎉 no goals
                                 -/
  smul_assoc r s x := by ext <;> simp [smul_re, smul_im, smul_assoc]
                                 /-
                                   🎉 no goals
                                 -/

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 90) [SMul R ℝ] [SMul Rᵐᵒᵖ ℝ] [IsCentralScalar R ℝ] :
    IsCentralScalar R ℂ where
                            /-
                              R : Type u_1
                              S : Type u_2
                              inst✝² : SMul R Real
                              inst✝¹ : SMul (MulOpposite R) Real
                              inst✝ : IsCentralScalar R Real
                              r : R
                              x : Complex
                              ⊢ Eq (HSMul.hSMul (MulOpposite.op r) x) (HSMul.hSMul r x)
                            -/
                                    /-
                                      🎉 no goals
                                    -/
  op_smul_eq_smul r x := by ext <;> simp [smul_re, smul_im, op_smul_eq_smul]
                                    /-
                                      🎉 no goals
                                    -/

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 90) mulAction [Monoid R] [MulAction R ℝ] : MulAction R ℂ where
                   /-
                     R : Type u_1
                     S : Type u_2
                     inst✝¹ : Monoid R
                     inst✝ : MulAction R Real
                     x : Complex
                     ⊢ Eq (HSMul.hSMul 1 x) x
                   -/
                           /-
                             🎉 no goals
                           -/
  one_smul x := by ext <;> simp [smul_re, smul_im, one_smul]
                           /-
                             🎉 no goals
                           -/
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝¹ : Monoid R
                         inst✝ : MulAction R Real
                         r s : R
                         x : Complex
                         ⊢ Eq (HSMul.hSMul (HMul.hMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
                       -/
                               /-
                                 🎉 no goals
                               -/
  mul_smul r s x := by ext <;> simp [smul_re, smul_im, mul_smul]
                               /-
                                 🎉 no goals
                               -/

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 90) distribSMul [DistribSMul R ℝ] : DistribSMul R ℂ where
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝ : DistribSMul R Real
                         r : R
                         x y : Complex
                         ⊢ Eq (HSMul.hSMul r (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝ : DistribSMul R Real
                      r : R
                      ⊢ Eq (HSMul.hSMul r 0) 0
                    -/
                            /-
                              🎉 no goals
                            -/
                               /-
                                 🎉 no goals
                               -/
                            /-
                              🎉 no goals
                            -/
  smul_add r x y := by ext <;> simp [smul_re, smul_im, smul_add]
                               /-
                                 🎉 no goals
                               -/
  smul_zero r := by ext <;> simp [smul_re, smul_im, smul_zero]

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 90) [Semiring R] [DistribMulAction R ℝ] : DistribMulAction R ℂ :=
  { Complex.distribSMul, Complex.mulAction with }

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 100) instModule [Semiring R] [Module R ℝ] : Module R ℂ where
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝¹ : Semiring R
                         inst✝ : Module R Real
                         r s : R
                         x : Complex
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
                               /-
                                 🎉 no goals
                               -/
  add_smul r s x := by ext <;> simp [smul_re, smul_im, add_smul]
                               /-
                                 🎉 no goals
                               -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝¹ : Semiring R
                      inst✝ : Module R Real
                      r : Complex
                      ⊢ Eq (HSMul.hSMul 0 r) 0
                    -/
                            /-
                              🎉 no goals
                            -/
  zero_smul r := by ext <;> simp [smul_re, smul_im, zero_smul]
                            /-
                              🎉 no goals
                            -/

-- priority manually adjusted in https://github.com/leanprover-community/mathlib4/pull/11980

instance (priority := 95) instAlgebraOfReal [CommSemiring R] [Algebra R ℝ] : Algebra R ℂ :=
  { Complex.ofRealHom.comp (algebraMap R ℝ) with
    smul := (· • ·)
                               /-
                                 R : Type u_1
                                 S : Type u_2
                                 inst✝¹ : CommSemiring R
                                 inst✝ : Algebra R Real
                                 r : R
                                 x : Complex
                                 ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (__src✝ r) x)
                               -/
                                      /-
                                        R : Type u_1
                                        S : Type u_2
                                        inst✝¹ : CommSemiring R
                                        inst✝ : Algebra R Real
                                        r : R
                                        x✝ : Complex
                                        xr xi : Real
                                        ⊢ Eq (HMul.hMul (__src✝ r) { re := xr, im := xi }) (HMul.hMul { re := xr, im : …
                                      -/
                                              /-
                                                🎉 no goals
                                              -/
                                       /-
                                         🎉 no goals
                                       -/
                                              /-
                                                🎉 no goals
                                              -/
    smul_def' := fun r x => by ext <;> simp [smul_re, smul_im, Algebra.smul_def]
                                       /-
                                         🎉 no goals
                                       -/
    commutes' := fun r ⟨xr, xi⟩ => by ext <;> simp [smul_re, smul_im, Algebra.commutes] }


instance : StarModule ℝ ℂ :=
                 /-
                   R : Type u_1
                   S : Type u_2
                   r : Real
                   x : Complex
                   ⊢ Eq (Star.star (HSMul.hSMul r x)) (HSMul.hSMul (Star.star r) (Star.star x))
                 -/
  ⟨fun r x => by simp only [star_def, star_trivial, real_smul, map_mul, conj_ofReal]⟩
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem coe_algebraMap : (algebraMap ℝ ℂ : ℝ → ℂ) = ((↑) : ℝ → ℂ) :=
  rfl


/-- We need this lemma since `Complex.coe_algebraMap` diverts the simp-normal form away from
`AlgHom.commutes`. -/
@[simp]
theorem _root_.AlgHom.map_coe_real_complex (f : ℂ →ₐ[ℝ] A) (x : ℝ) : f x = algebraMap ℝ A x :=
  f.commutes x


/-- Two `ℝ`-algebra homomorphisms from `ℂ` are equal if they agree on `Complex.I`. -/
@[ext]
theorem algHom_ext ⦃f g : ℂ →ₐ[ℝ] A⦄ (h : f I = g I) : f = g := by
  /-
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra Real A
    f g : AlgHom Real Complex A
    h : Eq (f Complex.I) (g Complex.I)
    ⊢ Eq f g
  -/
  ext ⟨x, y⟩
  /-
    case H.mk
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra Real A
    f g : AlgHom Real Complex A
    h : Eq (f Complex.I) (g Complex.I)
    x y : Real
    ⊢ Eq (f { re := x, im := y }) (g { re := x, im := y })
  -/
  simp only [mk_eq_add_mul_I, map_add, AlgHom.map_coe_real_complex, map_mul, h]
  /-
    🎉 no goals
  -/


/-- `ℂ` has a basis over `ℝ` given by `1` and `I`. -/
noncomputable def basisOneI : Basis (Fin 2) ℝ ℂ :=
  Basis.ofEquivFun
    { toFun := fun z => ![z.re, z.im]
      invFun := fun c => c 0 + c 1 • I
                              /-
                                R : Type u_1
                                S : Type u_2
                                z : Complex
                                ⊢ Eq ((fun c => HAdd.hAdd (↑(c 0)) (HSMul.hSMul (c 1) Complex.I)) ({ toFun :=  …
                              -/
      left_inv := fun z => by simp
                              /-
                                🎉 no goals
                              -/
      right_inv := fun c => by
                                 /-
                                   R : Type u_1
                                   S : Type u_2
                                   z z' : Complex
                                   ⊢ Eq ((fun z => Matrix.vecCons z.re (Matrix.vecCons z.im Matrix.vecEmpty)) (HA …
                                 -/
        /-
          R : Type u_1
          S : Type u_2
          c : Fin 2 → Real
          ⊢ Eq ({ toFun := fun z => Matrix.vecCons z.re (Matrix.vecCons z.im Matrix.vecE …
        -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   R : Type u_1
                                   S : Type u_2
                                   c : Real
                                   z : Complex
                                   ⊢ Eq ({ toFun := fun z => Matrix.vecCons z.re (Matrix.vecCons z.im Matrix.vecE …
                                 -/
        ext i
                                 /-
                                   🎉 no goals
                                 -/
        /-
          case h
          R : Type u_1
          S : Type u_2
          c : Fin 2 → Real
          i : Fin 2
          ⊢ Eq ({ toFun := fun z => Matrix.vecCons z.re (Matrix.vecCons z.im Matrix.vecE …
        -/
                        /-
                          🎉 no goals
                        -/
        fin_cases i <;> simp
                        /-
                          🎉 no goals
                        -/
      map_add' := fun z z' => by simp
      map_smul' := fun c z => by simp }


@[simp]
theorem coe_basisOneI_repr (z : ℂ) : ⇑(basisOneI.repr z) = ![z.re, z.im] :=
  rfl


@[simp]
theorem coe_basisOneI : ⇑basisOneI = ![1, I] :=
  funext fun i =>
    Basis.apply_eq_iff.mpr <|
      Finsupp.ext fun j => by
        /-
          i j : Fin 2
          ⊢ Eq ((Complex.basisOneI.repr (Matrix.vecCons 1 (Matrix.vecCons Complex.I Matr …
        -/
        fin_cases i <;> fin_cases j <;>
          -- Porting note: removed `only`, consider squeezing again
          simp [coe_basisOneI_repr, Finsupp.single_eq_of_ne, Matrix.cons_val_zero,
            Matrix.cons_val_one, Matrix.head_cons, Fin.one_eq_zero_iff, Ne, not_false_iff, I_re,
            Nat.succ_succ_ne_one, one_im, I_im, one_re, Finsupp.single_eq_same, Fin.zero_eq_one_iff]


instance (priority := 900) Module.complexToReal (E : Type*) [AddCommGroup E] [Module ℂ E] :
    Module ℝ E :=
  RestrictScalars.module ℝ ℂ E

/- Register as an instance (with low priority) the fact that a complex algebra is also a real
algebra. -/

instance (priority := 900) Algebra.complexToReal {A : Type*} [Semiring A] [Algebra ℂ A] :
    Algebra ℝ A :=
  RestrictScalars.algebra ℝ ℂ A

-- try to make sure we're not introducing diamonds but we will need
-- `reducible_and_instances` which currently fails https://github.com/leanprover-community/mathlib4/issues/10906

@[simp, norm_cast]
theorem Complex.coe_smul {E : Type*} [AddCommGroup E] [Module ℂ E] (x : ℝ) (y : E) :
    (x : ℂ) • y = x • y :=
  rfl


/-- The scalar action of `ℝ` on a `ℂ`-module `E` induced by `Module.complexToReal` commutes with
another scalar action of `M` on `E` whenever the action of `ℂ` commutes with the action of `M`. -/
instance (priority := 900) SMulCommClass.complexToReal {M E : Type*} [AddCommGroup E] [Module ℂ E]
    [SMul M E] [SMulCommClass ℂ M E] : SMulCommClass ℝ M E where
  smul_comm r _ _ := (smul_comm (r : ℂ) _ _ : _)


/-- The scalar action of `ℝ` on a `ℂ`-module `E` induced by `Module.complexToReal` associates with
another scalar action of `M` on `E` whenever the action of `ℂ` associates with the action of `M`. -/
instance IsScalarTower.complexToReal {M E : Type*} [AddCommGroup M] [Module ℂ M] [AddCommGroup E]
    [Module ℂ E] [SMul M E] [IsScalarTower ℂ M E] : IsScalarTower ℝ M E where
  smul_assoc r _ _ := (smul_assoc (r : ℂ) _ _ : _)

-- check that the following instance is implied by the one above.

instance (priority := 900) StarModule.complexToReal {E : Type*} [AddCommGroup E] [Star E]
    [Module ℂ E] [StarModule ℂ E] : StarModule ℝ E :=
                 /-
                   E : Type u_1
                   inst✝³ : AddCommGroup E
                   inst✝² : Star E
                   inst✝¹ : Module Complex E
                   inst✝ : StarModule Complex E
                   r : Real
                   a : E
                   ⊢ Eq (Star.star (HSMul.hSMul r a)) (HSMul.hSMul (Star.star r) (Star.star a))
                 -/
  ⟨fun r a => by rw [← smul_one_smul ℂ r a, star_smul, star_smul, star_one, smul_one_smul]⟩
                 /-
                   🎉 no goals
                 -/


/-- Linear map version of the real part function, from `ℂ` to `ℝ`. -/
def reLm : ℂ →ₗ[ℝ] ℝ where
  toFun x := x.re
  map_add' := add_re
                  /-
                    ⊢ ∀ (m : Real) (x : Complex), Eq ({ toFun := fun x => x.re, map_add' := Comple …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem reLm_coe : ⇑reLm = re :=
  rfl


/-- Linear map version of the imaginary part function, from `ℂ` to `ℝ`. -/
def imLm : ℂ →ₗ[ℝ] ℝ where
  toFun x := x.im
  map_add' := add_im
                  /-
                    ⊢ ∀ (m : Real) (x : Complex), Eq ({ toFun := fun x => x.im, map_add' := Comple …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem imLm_coe : ⇑imLm = im :=
  rfl


/-- `ℝ`-algebra morphism version of the canonical embedding of `ℝ` in `ℂ`. -/
def ofRealAm : ℝ →ₐ[ℝ] ℂ :=
  Algebra.ofId ℝ ℂ


@[simp]
theorem ofRealAm_coe : ⇑ofRealAm = ((↑) : ℝ → ℂ) :=
  rfl


/-- `ℝ`-algebra isomorphism version of the complex conjugation function from `ℂ` to `ℂ` -/
def conjAe : ℂ ≃ₐ[ℝ] ℂ :=
  { conj with
    invFun := conj
    left_inv := star_star
    right_inv := star_star
    commutes' := conj_ofReal }


@[simp]
theorem conjAe_coe : ⇑conjAe = conj :=
  rfl


/-- The matrix representation of `conjAe`. -/
@[simp]
theorem toMatrix_conjAe :
    LinearMap.toMatrix basisOneI basisOneI conjAe.toLinearMap = !![1, 0; 0, -1] := by
  /-
    ⊢ Eq ((LinearMap.toMatrix Complex.basisOneI Complex.basisOneI) Complex.conjAe. …
  -/
  ext i j
  -- Porting note: replaced non-terminal `simp [LinearMap.toMatrix_apply]`
  /-
    case a
    i j : Fin 2
    ⊢ Eq ((LinearMap.toMatrix Complex.basisOneI Complex.basisOneI) Complex.conjAe. …
  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  fin_cases i <;> fin_cases j <;> simp [LinearMap.toMatrix_apply]
                                  /-
                                    🎉 no goals
                                  -/


/-- The identity and the complex conjugation are the only two `ℝ`-algebra homomorphisms of `ℂ`. -/
theorem real_algHom_eq_id_or_conj (f : ℂ →ₐ[ℝ] ℂ) : f = AlgHom.id ℝ ℂ ∨ f = conjAe := by
  refine
      (eq_or_eq_neg_of_sq_eq_sq (f I) I <| by rw [← map_pow, I_sq, map_neg, map_one]).imp ?_ ?_ <;>
    /-
      case refine_1
      f : AlgHom Real Complex Complex
      ⊢ Eq (f Complex.I) Complex.I → Eq f (AlgHom.id Real Complex)
    -/
    refine fun h => algHom_ext ?_
  /-
    case refine_1
    f : AlgHom Real Complex Complex
    h : Eq (f Complex.I) Complex.I
    ⊢ Eq (f Complex.I) ((AlgHom.id Real Complex) Complex.I)
  -/
  exacts [h, conj_I.symm ▸ h]
  /-
    🎉 no goals
  -/


/-- The natural `LinearEquiv` from `ℂ` to `ℝ × ℝ`. -/
@[simps! (config := { simpRhs := true }) apply symm_apply_re symm_apply_im]
def equivRealProdLm : ℂ ≃ₗ[ℝ] ℝ × ℝ :=
  { equivRealProdAddHom with
    -- Porting note: `simp` has issues with `Prod.smul_def`
                               /-
                                 r : Real
                                 c : Complex
                                 ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r c)) (HSMul …
                               -/
    map_smul' := fun r c => by simp [equivRealProdAddHom, Prod.smul_def, smul_eq_mul] }
                               /-
                                 🎉 no goals
                               -/


theorem equivRealProdLm_symm_apply (p : ℝ × ℝ) :
    Complex.equivRealProdLm.symm p = p.1 + p.2 * Complex.I := Complex.equivRealProd_symm_apply p

/-- There is an alg_hom from `ℂ` to any `ℝ`-algebra with an element that squares to `-1`.

See `Complex.lift` for this as an equiv. -/
def liftAux (I' : A) (hf : I' * I' = -1) : ℂ →ₐ[ℝ] A :=
  AlgHom.ofLinearMap
    ((Algebra.linearMap ℝ A).comp reLm + (LinearMap.toSpanSingleton _ _ I').comp imLm)
                                                 /-
                                                   A : Type u_1
                                                   inst✝¹ : Ring A
                                                   inst✝ : Algebra Real A
                                                   I' : A
                                                   hf : Eq (HMul.hMul I' I') (-1)
                                                   ⊢ Eq (HAdd.hAdd ((algebraMap Real A) 1) (HSMul.hSMul 0 I')) 1
                                                 -/
    (show algebraMap ℝ A 1 + (0 : ℝ) • I' = 1 by rw [RingHom.map_one, zero_smul, add_zero])
                                                 /-
                                                   🎉 no goals
                                                 -/
    fun ⟨x₁, y₁⟩ ⟨x₂, y₂⟩ =>
    show
      algebraMap ℝ A (x₁ * x₂ - y₁ * y₂) + (x₁ * y₂ + y₁ * x₂) • I' =
        (algebraMap ℝ A x₁ + y₁ • I') * (algebraMap ℝ A x₂ + y₂ • I') by
      /-
        A : Type u_1
        inst✝¹ : Ring A
        inst✝ : Algebra Real A
        I' : A
        hf : Eq (HMul.hMul I' I') (-1)
        x✝¹ x✝ : Complex
        x₁ y₁ x₂ y₂ : Real
        ⊢ Eq (HAdd.hAdd ((algebraMap Real A) (HSub.hSub (HMul.hMul x₁ x₂) (HMul.hMul y …
      -/
      rw [add_mul, mul_add, mul_add, add_comm _ (y₁ • I' * y₂ • I'), add_add_add_comm]
      /-
        A : Type u_1
        inst✝¹ : Ring A
        inst✝ : Algebra Real A
        I' : A
        hf : Eq (HMul.hMul I' I') (-1)
        x✝¹ x✝ : Complex
        x₁ y₁ x₂ y₂ : Real
        ⊢ Eq (HAdd.hAdd ((algebraMap Real A) (HSub.hSub (HMul.hMul x₁ x₂) (HMul.hMul y …
      -/
      congr 1
      -- equate "real" and "imaginary" parts
        /-
          case e_a
          A : Type u_1
          inst✝¹ : Ring A
          inst✝ : Algebra Real A
          I' : A
          hf : Eq (HMul.hMul I' I') (-1)
          x✝¹ x✝ : Complex
          x₁ y₁ x₂ y₂ : Real
          ⊢ Eq ((algebraMap Real A) (HSub.hSub (HMul.hMul x₁ x₂) (HMul.hMul y₁ y₂))) (HA …
        -/
      · let inst : SMulCommClass ℝ A A := by infer_instance  -- Porting note: added
        rw [smul_mul_smul_comm, hf, smul_neg, ← Algebra.algebraMap_eq_smul_one, ← sub_eq_add_neg, ←
          RingHom.map_mul, ← RingHom.map_sub]
      · rw [Algebra.smul_def, Algebra.smul_def, Algebra.smul_def, ← Algebra.right_comm _ x₂, ←
          mul_assoc, ← add_mul, ← RingHom.map_mul, ← RingHom.map_mul, ← RingHom.map_add]


@[simp]
theorem liftAux_apply (I' : A) (hI') (z : ℂ) : liftAux I' hI' z = algebraMap ℝ A z.re + z.im • I' :=
  rfl


                                                                     /-
                                                                       A : Type u_1
                                                                       inst✝¹ : Ring A
                                                                       inst✝ : Algebra Real A
                                                                       I' : A
                                                                       hI' : Eq (HMul.hMul I' I') (-1)
                                                                       ⊢ Eq ((Complex.liftAux I' hI') Complex.I) I'
                                                                     -/
theorem liftAux_apply_I (I' : A) (hI') : liftAux I' hI' I = I' := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- A universal property of the complex numbers, providing a unique `ℂ →ₐ[ℝ] A` for every element
of `A` which squares to `-1`.

This can be used to embed the complex numbers in the `Quaternion`s.

This isomorphism is named to match the very similar `Zsqrtd.lift`. -/
@[simps (config := { simpRhs := true })]
def lift : { I' : A // I' * I' = -1 } ≃ (ℂ →ₐ[ℝ] A) where
  toFun I' := liftAux I' I'.prop
                       /-
                         A : Type u_1
                         inst✝¹ : Ring A
                         inst✝ : Algebra Real A
                         F : AlgHom Real Complex A
                         ⊢ Eq (HMul.hMul (F Complex.I) (F Complex.I)) (-1)
                       -/
  invFun F := ⟨F I, by rw [← map_mul, I_mul_I, map_neg, map_one]⟩
                       /-
                         🎉 no goals
                       -/
  left_inv I' := Subtype.ext <| liftAux_apply_I (I' : A) I'.prop
  right_inv _ := algHom_ext <| liftAux_apply_I _ _

-- When applied to `Complex.I` itself, `lift` is the identity.

@[simp]
theorem liftAux_I : liftAux I I_mul_I = AlgHom.id ℝ ℂ :=
  algHom_ext <| liftAux_apply_I _ _

-- When applied to `-Complex.I`, `lift` is conjugation, `conj`.

@[simp]
theorem liftAux_neg_I : liftAux (-I) ((neg_mul_neg _ _).trans I_mul_I) = conjAe :=
  algHom_ext <| (liftAux_apply_I _ _).trans conj_I.symm


/-- Create a `selfAdjoint` element from a `skewAdjoint` element by multiplying by the scalar
`-Complex.I`. -/
@[simps]
def skewAdjoint.negISMul : skewAdjoint A →ₗ[ℝ] selfAdjoint A where
  toFun a :=
    ⟨-I • ↑a, by
      simp only [neg_smul, neg_mem_iff, selfAdjoint.mem_iff, star_smul, star_def, conj_I,
        star_val_eq, smul_neg, neg_neg]⟩
  map_add' a b := by
    /-
      A : Type u_1
      inst✝³ : AddCommGroup A
      inst✝² : Module Complex A
      inst✝¹ : StarAddMonoid A
      inst✝ : StarModule Complex A
      a b : Subtype fun x => Membership.mem (skewAdjoint A) x
      ⊢ Eq ((fun a => ⟨HSMul.hSMul (Neg.neg Complex.I) ↑a, ⋯⟩) (HAdd.hAdd a b)) (HAd …
    -/
    ext
    /-
      case a
      A : Type u_1
      inst✝³ : AddCommGroup A
      inst✝² : Module Complex A
      inst✝¹ : StarAddMonoid A
      inst✝ : StarModule Complex A
      a b : Subtype fun x => Membership.mem (skewAdjoint A) x
      ⊢ Eq ↑((fun a => ⟨HSMul.hSMul (Neg.neg Complex.I) ↑a, ⋯⟩) (HAdd.hAdd a b)) ↑(H …
    -/
    simp only [AddSubgroup.coe_add, smul_add, AddMemClass.mk_add_mk]
    /-
      🎉 no goals
    -/
  map_smul' a b := by
    /-
      A : Type u_1
      inst✝³ : AddCommGroup A
      inst✝² : Module Complex A
      inst✝¹ : StarAddMonoid A
      inst✝ : StarModule Complex A
      a : Real
      b : Subtype fun x => Membership.mem (skewAdjoint A) x
      ⊢ Eq ({ toFun := fun a => ⟨HSMul.hSMul (Neg.neg Complex.I) ↑a, ⋯⟩, map_add' := …
    -/
    ext
    simp only [neg_smul, skewAdjoint.val_smul, AddSubgroup.coe_mk, RingHom.id_apply,
      selfAdjoint.val_smul, smul_neg, neg_inj]
    /-
      case a
      A : Type u_1
      inst✝³ : AddCommGroup A
      inst✝² : Module Complex A
      inst✝¹ : StarAddMonoid A
      inst✝ : StarModule Complex A
      a : Real
      b : Subtype fun x => Membership.mem (skewAdjoint A) x
      ⊢ Eq (HSMul.hSMul Complex.I (HSMul.hSMul a ↑b)) (HSMul.hSMul a (HSMul.hSMul Co …
    -/
    rw [smul_comm]
    /-
      🎉 no goals
    -/


theorem skewAdjoint.I_smul_neg_I (a : skewAdjoint A) : I • (skewAdjoint.negISMul a : A) = a := by
  simp only [smul_smul, skewAdjoint.negISMul_apply_coe, neg_smul, smul_neg, I_mul_I, one_smul,
    neg_neg]


/-- The real part `ℜ a` of an element `a` of a star module over `ℂ`, as a linear map. This is just
`selfAdjointPart ℝ`, but we provide it as a separate definition in order to link it with lemmas
concerning the `imaginaryPart`, which doesn't exist in star modules over other rings. -/
noncomputable def realPart : A →ₗ[ℝ] selfAdjoint A :=
  selfAdjointPart ℝ


/-- The imaginary part `ℑ a` of an element `a` of a star module over `ℂ`, as a linear map into the
self adjoint elements. In a general star module, we have a decomposition into the `selfAdjoint`
and `skewAdjoint` parts, but in a star module over `ℂ` we have
`realPart_add_I_smul_imaginaryPart`, which allows us to decompose into a linear combination of
`selfAdjoint`s. -/
noncomputable def imaginaryPart : A →ₗ[ℝ] selfAdjoint A :=
  skewAdjoint.negISMul.comp (skewAdjointPart ℝ)


@[inherit_doc]
scoped[ComplexStarModule] notation "ℜ" => realPart

@[inherit_doc]
scoped[ComplexStarModule] notation "ℑ" => imaginaryPart


theorem realPart_apply_coe (a : A) : (ℜ a : A) = (2 : ℝ)⁻¹ • (a + star a) := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (↑(realPart a)) (HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a (Star.star a)))
  -/
  unfold realPart
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (↑((selfAdjointPart Real) a)) (HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a (Star …
  -/
  simp only [selfAdjointPart_apply_coe, invOf_eq_inv]
  /-
    🎉 no goals
  -/


theorem imaginaryPart_apply_coe (a : A) : (ℑ a : A) = -I • (2 : ℝ)⁻¹ • (a - star a) := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (↑(imaginaryPart a)) (HSMul.hSMul (Neg.neg Complex.I) (HSMul.hSMul (Inv.i …
  -/
  unfold imaginaryPart
  simp only [LinearMap.coe_comp, Function.comp_apply, skewAdjoint.negISMul_apply_coe,
    skewAdjointPart_apply_coe, invOf_eq_inv, neg_smul]


/-- The standard decomposition of `ℜ a + Complex.I • ℑ a = a` of an element of a star module over
`ℂ` into a linear combination of self adjoint elements. -/
theorem realPart_add_I_smul_imaginaryPart (a : A) : (ℜ a : A) + I • (ℑ a : A) = a := by
  simpa only [smul_smul, realPart_apply_coe, imaginaryPart_apply_coe, neg_smul, I_mul_I, one_smul,
    neg_sub, add_add_sub_cancel, smul_sub, smul_add, neg_sub_neg, invOf_eq_inv] using
    invOf_two_smul_add_invOf_two_smul ℝ a


@[simp]
theorem realPart_I_smul (a : A) : ℜ (I • a) = -ℑ a := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (realPart (HSMul.hSMul Complex.I a)) (Neg.neg (imaginaryPart a))
  -/
  ext
  -- Porting note: was
  -- simp [smul_comm I, smul_sub, sub_eq_add_neg, add_comm]
  rw [realPart_apply_coe, NegMemClass.coe_neg, imaginaryPart_apply_coe, neg_smul, neg_neg,
    smul_comm I, star_smul, star_def, conj_I, smul_sub, neg_smul, sub_eq_add_neg]


@[simp]
theorem imaginaryPart_I_smul (a : A) : ℑ (I • a) = ℜ a := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (imaginaryPart (HSMul.hSMul Complex.I a)) (realPart a)
  -/
  ext
  -- Porting note: was
  -- simp [smul_comm I, smul_smul I]
  /-
    case a
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq ↑(imaginaryPart (HSMul.hSMul Complex.I a)) ↑(realPart a)
  -/
  rw [realPart_apply_coe, imaginaryPart_apply_coe, smul_comm]
  /-
    case a
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    a : A
    ⊢ Eq (HSMul.hSMul (Inv.inv 2) (HSMul.hSMul (Neg.neg Complex.I) (HSub.hSub (HSM …
  -/
  simp [← smul_assoc]
  /-
    🎉 no goals
  -/


theorem realPart_smul (z : ℂ) (a : A) : ℜ (z • a) = z.re • ℜ a - z.im • ℑ a := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    z : Complex
    a : A
    ⊢ Eq (realPart (HSMul.hSMul z a)) (HSub.hSub (HSMul.hSMul z.re (realPart a)) ( …
  -/
  have := by congrm (ℜ ($((re_add_im z).symm) • a))
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    z : Complex
    a : A
    this : Eq (realPart (HSMul.hSMul z a)) (realPart (HSMul.hSMul (HAdd.hAdd (↑z.r …
    ⊢ Eq (realPart (HSMul.hSMul z a)) (HSub.hSub (HSMul.hSMul z.re (realPart a)) ( …
  -/
  simpa [-re_add_im, add_smul, ← smul_smul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem imaginaryPart_smul (z : ℂ) (a : A) : ℑ (z • a) = z.re • ℑ a + z.im • ℜ a := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    z : Complex
    a : A
    ⊢ Eq (imaginaryPart (HSMul.hSMul z a)) (HAdd.hAdd (HSMul.hSMul z.re (imaginary …
  -/
  have := by congrm (ℑ ($((re_add_im z).symm) • a))
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    z : Complex
    a : A
    this : Eq (imaginaryPart (HSMul.hSMul z a)) (imaginaryPart (HSMul.hSMul (HAdd. …
    ⊢ Eq (imaginaryPart (HSMul.hSMul z a)) (HAdd.hAdd (HSMul.hSMul z.re (imaginary …
  -/
  simpa [-re_add_im, add_smul, ← smul_smul]
  /-
    🎉 no goals
  -/


lemma skewAdjointPart_eq_I_smul_imaginaryPart (x : A) :
    (skewAdjointPart ℝ x : A) = I • (imaginaryPart x : A) := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    x : A
    ⊢ Eq (↑((skewAdjointPart Real) x)) (HSMul.hSMul Complex.I ↑(imaginaryPart x))
  -/
  simp [imaginaryPart_apply_coe, smul_smul]
  /-
    🎉 no goals
  -/


lemma imaginaryPart_eq_neg_I_smul_skewAdjointPart (x : A) :
    (imaginaryPart x : A) = -I • (skewAdjointPart ℝ x : A) :=
  rfl


lemma IsSelfAdjoint.coe_realPart {x : A} (hx : IsSelfAdjoint x) :
    (ℜ x : A) = x :=
  hx.coe_selfAdjointPart_apply ℝ


nonrec lemma IsSelfAdjoint.imaginaryPart {x : A} (hx : IsSelfAdjoint x) :
    ℑ x = 0 := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    x : A
    hx : IsSelfAdjoint x
    ⊢ Eq (_root_.imaginaryPart x) 0
  -/
  rw [imaginaryPart, LinearMap.comp_apply, hx.skewAdjointPart_apply _, map_zero]
  /-
    🎉 no goals
  -/


lemma realPart_comp_subtype_selfAdjoint :
    realPart.comp (selfAdjoint.submodule ℝ A).subtype = LinearMap.id :=
  selfAdjointPart_comp_subtype_selfAdjoint ℝ


lemma imaginaryPart_comp_subtype_selfAdjoint :
    imaginaryPart.comp (selfAdjoint.submodule ℝ A).subtype = 0 := by
  rw [imaginaryPart, LinearMap.comp_assoc, skewAdjointPart_comp_subtype_selfAdjoint,
    LinearMap.comp_zero]


@[simp]
lemma imaginaryPart_realPart {x : A} : ℑ (ℜ x : A) = 0 :=
  (ℜ x).property.imaginaryPart


@[simp]
lemma imaginaryPart_imaginaryPart {x : A} : ℑ (ℑ x : A) = 0 :=
  (ℑ x).property.imaginaryPart


@[simp]
lemma realPart_idem {x : A} : ℜ (ℜ x : A) = ℜ x :=
  Subtype.ext <| (ℜ x).property.coe_realPart


@[simp]
lemma realPart_imaginaryPart {x : A} : ℜ (ℑ x : A) = ℑ x :=
  Subtype.ext <| (ℑ x).property.coe_realPart


lemma realPart_surjective : Function.Surjective (realPart (A := A)) :=
  fun x ↦ ⟨(x : A), Subtype.ext x.property.coe_realPart⟩


lemma imaginaryPart_surjective : Function.Surjective (imaginaryPart (A := A)) :=
  fun x ↦
                                    /-
                                      A : Type u_1
                                      inst✝³ : AddCommGroup A
                                      inst✝² : Module Complex A
                                      inst✝¹ : StarAddMonoid A
                                      inst✝ : StarModule Complex A
                                      x : Subtype fun x => Membership.mem (selfAdjoint A) x
                                      ⊢ Eq ↑(imaginaryPart (HSMul.hSMul Complex.I ↑x)) ↑x
                                    -/
    ⟨I • (x : A), Subtype.ext <| by simp only [imaginaryPart_I_smul, x.property.coe_realPart]⟩
                                    /-
                                      🎉 no goals
                                    -/


lemma span_selfAdjoint : span ℂ (selfAdjoint A : Set A) = ⊤ := by
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    ⊢ Eq (Submodule.span Complex ↑(selfAdjoint A)) Top.top
  -/
  refine eq_top_iff'.mpr fun x ↦ ?_
  /-
    A : Type u_1
    inst✝³ : AddCommGroup A
    inst✝² : Module Complex A
    inst✝¹ : StarAddMonoid A
    inst✝ : StarModule Complex A
    x : A
    ⊢ Membership.mem (Submodule.span Complex ↑(selfAdjoint A)) x
  -/
  rw [← realPart_add_I_smul_imaginaryPart x]
  exact add_mem (subset_span (ℜ x).property) <|
    SMulMemClass.smul_mem _ <| subset_span (ℑ x).property


/-- The natural `ℝ`-linear equivalence between `selfAdjoint ℂ` and `ℝ`. -/
@[simps apply symm_apply]
def Complex.selfAdjointEquiv : selfAdjoint ℂ ≃ₗ[ℝ] ℝ where
  toFun := fun z ↦ (z : ℂ).re
  invFun := fun x ↦ ⟨x, conj_ofReal x⟩
  left_inv := fun z ↦ Subtype.ext <| conj_eq_iff_re.mp z.property.star_eq
  right_inv := fun _ ↦ rfl
                 /-
                   A : Type u_1
                   inst✝³ : AddCommGroup A
                   inst✝² : Module Complex A
                   inst✝¹ : StarAddMonoid A
                   inst✝ : StarModule Complex A
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem (selfAdjoint Complex) x), Eq ((fun  …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    A : Type u_1
                    inst✝³ : AddCommGroup A
                    inst✝² : Module Complex A
                    inst✝¹ : StarAddMonoid A
                    inst✝ : StarModule Complex A
                    ⊢ ∀ (m : Real) (x : Subtype fun x => Membership.mem (selfAdjoint Complex) x),  …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


lemma Complex.coe_selfAdjointEquiv (z : selfAdjoint ℂ) :
    (selfAdjointEquiv z : ℂ) = z := by
  simpa [selfAdjointEquiv_symm_apply]
    using (congr_arg Subtype.val <| Complex.selfAdjointEquiv.left_inv z)


@[simp]
lemma realPart_ofReal (r : ℝ) : (ℜ (r : ℂ) : ℂ) = r := by
  /-
    r : Real
    ⊢ Eq ↑(realPart ↑r) ↑r
  -/
  rw [realPart_apply_coe, star_def, conj_ofReal, ← two_smul ℝ (r : ℂ)]
  /-
    r : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv 2) (HSMul.hSMul 2 ↑r)) ↑r
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma imaginaryPart_ofReal (r : ℝ) : ℑ (r : ℂ) = 0 := by
  /-
    r : Real
    ⊢ Eq (imaginaryPart ↑r) 0
  -/
  ext1; simp [imaginaryPart_apply_coe, conj_ofReal]
        /-
          🎉 no goals
        -/


lemma Complex.coe_realPart (z : ℂ) : (ℜ z : ℂ) = z.re := calc
                                               /-
                                                 z : Complex
                                                 ⊢ Eq ↑(realPart z) ↑(realPart (HAdd.hAdd (↑z.re) (HMul.hMul (↑z.im) Complex.I)))
                                               -/
  (ℜ z : ℂ) = (↑(ℜ (↑z.re + ↑z.im * I))) := by congrm (ℜ $((re_add_im z).symm))
                                               /-
                                                 🎉 no goals
                                               -/
  _         = z.re                       := by
    /-
      z : Complex
      ⊢ Eq ↑(realPart (HAdd.hAdd (↑z.re) (HMul.hMul (↑z.im) Complex.I))) ↑z.re
    -/
    rw [map_add, AddSubmonoid.coe_add, mul_comm, ← smul_eq_mul, realPart_I_smul]
    /-
      z : Complex
      ⊢ Eq (HAdd.hAdd ↑(realPart ↑z.re) ↑(Neg.neg (imaginaryPart ↑z.im))) ↑z.re
    -/
    simp [conj_ofReal, ← two_mul]
    /-
      🎉 no goals
    -/


lemma star_mul_self_add_self_mul_star {A : Type*} [NonUnitalRing A] [StarRing A]
    [Module ℂ A] [IsScalarTower ℂ A A] [SMulCommClass ℂ A A] [StarModule ℂ A] (a : A) :
    star a * a + a * star a = 2 • (ℜ a * ℜ a + ℑ a * ℑ a) :=
  have a_eq := (realPart_add_I_smul_imaginaryPart a).symm
  calc
    star a * a + a * star a = _ :=
      congr((star $(a_eq)) * $(a_eq) + $(a_eq) * (star $(a_eq)))
    _ = 2 • (ℜ a * ℜ a + ℑ a * ℑ a) := by
      simp [mul_add, add_mul, smul_smul, two_smul, mul_smul_comm,
        smul_mul_assoc]
      /-
        A : Type u_2
        inst✝⁵ : NonUnitalRing A
        inst✝⁴ : StarRing A
        inst✝³ : Module Complex A
        inst✝² : IsScalarTower Complex A A
        inst✝¹ : SMulCommClass Complex A A
        inst✝ : StarModule Complex A
        a : A
        a_eq : Eq a (HAdd.hAdd (↑(realPart a)) (HSMul.hSMul Complex.I ↑(imaginaryPart  …
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HMul.hMul ↑(realPart a) ↑(realPart a))  …
      -/
      /-
        🎉 no goals
      -/
      abel
      /-
        🎉 no goals
      -/


