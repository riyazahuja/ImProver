variable (A) in
/-- Given an `R`-algebra `A` and an `R`-basis of `M`, this is an `R`-linear isomorphism
`A ⊗[R] M ≃ (ι →₀ A)` (which is in fact `A`-linear). -/
noncomputable def basisAux : A ⊗[R] M ≃ₗ[R] ι →₀ A :=
  _root_.TensorProduct.congr (Finsupp.LinearEquiv.finsuppUnique R A PUnit.{uι+1}).symm b.repr ≪≫ₗ
    (finsuppTensorFinsupp R R A R PUnit ι).trans
      (Finsupp.lcongr (Equiv.uniqueProd ι PUnit) (_root_.TensorProduct.rid R A))


theorem basisAux_tmul (a : A) (m : M) :
    basisAux A b (a ⊗ₜ m) = a • Finsupp.mapRange (algebraMap R A) (map_zero _) (b.repr m) := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    a : A
    m : M
    ⊢ Eq ((Algebra.TensorProduct.basisAux A b) (TensorProduct.tmul R a m)) (HSMul. …
  -/
  ext
  /-
    case h
    R : Type u_1
    A : Type u_2
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    a : A
    m : M
    a✝ : ι
    ⊢ Eq (((Algebra.TensorProduct.basisAux A b) (TensorProduct.tmul R a m)) a✝) (( …
  -/
  simp [basisAux, ← Algebra.commutes, Algebra.smul_def]
  /-
    🎉 no goals
  -/


theorem basisAux_map_smul (a : A) (x : A ⊗[R] M) : basisAux A b (a • x) = a • basisAux A b x :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     M : Type uM
                                     ι : Type uι
                                     inst✝⁴ : CommSemiring R
                                     inst✝³ : Semiring A
                                     inst✝² : Algebra R A
                                     inst✝¹ : AddCommMonoid M
                                     inst✝ : Module R M
                                     b : Basis ι R M
                                     a : A
                                     x : TensorProduct R A M
                                     ⊢ Eq ((Algebra.TensorProduct.basisAux A b) (HSMul.hSMul a 0)) (HSMul.hSMul a ( …
                                   -/
  TensorProduct.induction_on x (by simp)
                                   /-
                                     🎉 no goals
                                   -/
                   /-
                     R : Type u_1
                     A : Type u_2
                     M : Type uM
                     ι : Type uι
                     inst✝⁴ : CommSemiring R
                     inst✝³ : Semiring A
                     inst✝² : Algebra R A
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     b : Basis ι R M
                     a : A
                     x✝ : TensorProduct R A M
                     x : A
                     y : M
                     ⊢ Eq ((Algebra.TensorProduct.basisAux A b) (HSMul.hSMul a (TensorProduct.tmul  …
                   -/
    (fun x y => by simp only [TensorProduct.smul_tmul', basisAux_tmul, smul_assoc])
                   /-
                     🎉 no goals
                   -/
                        /-
                          R : Type u_1
                          A : Type u_2
                          M : Type uM
                          ι : Type uι
                          inst✝⁴ : CommSemiring R
                          inst✝³ : Semiring A
                          inst✝² : Algebra R A
                          inst✝¹ : AddCommMonoid M
                          inst✝ : Module R M
                          b : Basis ι R M
                          a : A
                          x✝ x y : TensorProduct R A M
                          hx : Eq ((Algebra.TensorProduct.basisAux A b) (HSMul.hSMul a x)) (HSMul.hSMul  …
                          hy : Eq ((Algebra.TensorProduct.basisAux A b) (HSMul.hSMul a y)) (HSMul.hSMul  …
                          ⊢ Eq ((Algebra.TensorProduct.basisAux A b) (HSMul.hSMul a (HAdd.hAdd x y))) (H …
                        -/
    fun x y hx hy => by simp [hx, hy]
                        /-
                          🎉 no goals
                        -/


variable (A) in
/-- Given a `R`-algebra `A`, this is the `A`-basis of `A ⊗[R] M` induced by a `R`-basis of `M`. -/
noncomputable def basis : Basis ι A (A ⊗[R] M) where
  repr := { basisAux A b with map_smul' := basisAux_map_smul b }


@[simp]
theorem basis_repr_tmul (a : A) (m : M) :
    (basis A b).repr (a ⊗ₜ m) = a • Finsupp.mapRange (algebraMap R A) (map_zero _) (b.repr m) :=
  basisAux_tmul b a m -- Porting note: Lean 3 had _ _ _


theorem basis_repr_symm_apply (a : A) (i : ι) :
    (basis A b).repr.symm (Finsupp.single i a) = a ⊗ₜ b.repr.symm (Finsupp.single i 1) := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    a : A
    i : ι
    ⊢ Eq ((Algebra.TensorProduct.basis A b).repr.symm (Finsupp.single i a)) (Tenso …
  -/
  rw [basis, LinearEquiv.coe_symm_mk] -- Porting note: `coe_symm_mk` isn't firing in `simp`
  /-
    R : Type u_1
    A : Type u_2
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    a : A
    i : ι
    ⊢ Eq ((Algebra.TensorProduct.basisAux A b).invFun (Finsupp.single i a)) (Tenso …
  -/
  simp [Equiv.uniqueProd_symm_apply, basisAux]
  /-
    🎉 no goals
  -/


@[simp]
theorem basis_apply (i : ι) : basis A b i = 1 ⊗ₜ b i := basis_repr_symm_apply b 1 i


theorem basis_repr_symm_apply' (a : A) (i : ι) : a • basis A b i = a ⊗ₜ b i := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type uM
    ι : Type uι
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    a : A
    i : ι
    ⊢ Eq (HSMul.hSMul a ((Algebra.TensorProduct.basis A b) i)) (TensorProduct.tmul …
  -/
  simpa using basis_repr_symm_apply b a i
  /-
    🎉 no goals
  -/


lemma _root_.Basis.baseChange_linearMap (b : Basis ι R M) (b' : Basis ι' R N) (ij : ι × ι') :
    baseChange A (b'.linearMap b ij) = (basis A b').linearMap (basis A b) ij := by
  /-
    R : Type u_1
    M : Type uM
    ι : Type uι
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : Fintype ι
    ι' : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι'
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    A : Type u_5
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    b : Basis ι R M
    b' : Basis ι' R N
    ij : Prod ι ι'
    ⊢ Eq (LinearMap.baseChange A ((b'.linearMap b) ij)) (((Algebra.TensorProduct.b …
  -/
  apply (basis A b').ext
  /-
    R : Type u_1
    M : Type uM
    ι : Type uι
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : Fintype ι
    ι' : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι'
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    A : Type u_5
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    b : Basis ι R M
    b' : Basis ι' R N
    ij : Prod ι ι'
    ⊢ ∀ (i : ι'), Eq ((LinearMap.baseChange A ((b'.linearMap b) ij)) ((Algebra.Ten …
  -/
  intro k
  /-
    R : Type u_1
    M : Type uM
    ι : Type uι
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : Fintype ι
    ι' : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι'
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    A : Type u_5
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    b : Basis ι R M
    b' : Basis ι' R N
    ij : Prod ι ι'
    k : ι'
    ⊢ Eq ((LinearMap.baseChange A ((b'.linearMap b) ij)) ((Algebra.TensorProduct.b …
  -/
  conv_lhs => simp only [basis_apply, baseChange_tmul]
  /-
    R : Type u_1
    M : Type uM
    ι : Type uι
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : Fintype ι
    ι' : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι'
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    A : Type u_5
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    b : Basis ι R M
    b' : Basis ι' R N
    ij : Prod ι ι'
    k : ι'
    ⊢ Eq (TensorProduct.tmul R 1 (((b'.linearMap b) ij) (b' k))) ((((Algebra.Tenso …
  -/
  simp_rw [Basis.linearMap_apply_apply, basis_apply]
  /-
    R : Type u_1
    M : Type uM
    ι : Type uι
    inst✝⁹ : CommSemiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : Fintype ι
    ι' : Type u_3
    N : Type u_4
    inst✝⁵ : Fintype ι'
    inst✝⁴ : DecidableEq ι'
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    A : Type u_5
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    b : Basis ι R M
    b' : Basis ι' R N
    ij : Prod ι ι'
    k : ι'
    ⊢ Eq (TensorProduct.tmul R 1 (ite (Eq ij.2 k) (b ij.1) 0)) (ite (Eq ij.2 k) (T …
  -/
            /-
              🎉 no goals
            -/
  split <;> simp only [TensorProduct.tmul_zero]
            /-
              🎉 no goals
            -/


lemma _root_.Basis.baseChange_end (b : Basis ι R M) (ij : ι × ι) :
    baseChange A (b.end ij) = (basis A b).end ij :=
  b.baseChange_linearMap A b ij


instance instFree (R A M : Type*)
    [CommSemiring R] [AddCommMonoid M] [Module R M] [Module.Free R M]
    [CommSemiring A] [Algebra R A] :
    Module.Free A (A ⊗[R] M) :=
  Module.Free.of_basis <| Algebra.TensorProduct.basis A (Module.Free.chooseBasis R M)


@[simp]
lemma toMatrix_baseChange (f : M₁ →ₗ[R] M₂) (b₁ : Basis ι R M₁) (b₂ : Basis ι₂ R M₂) :
    toMatrix (basis A b₁) (basis A b₂) (f.baseChange A) =
    (toMatrix b₁ b₂ f).map (algebraMap R A) := by
  /-
    R : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    ι : Type u_4
    ι₂ : Type u_5
    A : Type u_6
    inst✝⁹ : Fintype ι
    inst✝⁸ : Finite ι₂
    inst✝⁷ : DecidableEq ι
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M₁
    inst✝² : Module R M₁
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : LinearMap (RingHom.id R) M₁ M₂
    b₁ : Basis ι R M₁
    b₂ : Basis ι₂ R M₂
    ⊢ Eq ((LinearMap.toMatrix (Algebra.TensorProduct.basis A b₁) (Algebra.TensorPr …
  -/
  ext; simp [toMatrix_apply]
       /-
         🎉 no goals
       -/


