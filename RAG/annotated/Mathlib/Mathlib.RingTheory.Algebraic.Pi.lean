/-- This is not an instance as it forms a diamond with `Pi.instSMul`.

See the `instance_diamonds` test for details. -/
def Polynomial.hasSMulPi [Semiring R] [SMul R S] : SMul R[X] (R → S) :=
  ⟨fun p f x => eval x p • f x⟩


/-- This is not an instance as it forms a diamond with `Pi.instSMul`.

See the `instance_diamonds` test for details. -/
noncomputable def Polynomial.hasSMulPi' [CommSemiring R] [Semiring S] [Algebra R S]
    [SMul S T] : SMul R[X] (S → T) :=
  ⟨fun p f x => aeval x p • f x⟩


@[simp]
theorem polynomial_smul_apply [Semiring R] [SMul R S] (p : R[X]) (f : R → S) (x : R) :
    (p • f) x = eval x p • f x :=
  rfl


@[simp]
theorem polynomial_smul_apply' [CommSemiring R] [Semiring S] [Algebra R S] [SMul S T]
    (p : R[X]) (f : S → T) (x : S) : (p • f) x = aeval x p • f x :=
  rfl


/-- This is not an instance for the same reasons as `Polynomial.hasSMulPi'`. -/
noncomputable def Polynomial.algebraPi : Algebra R[X] (S → T) :=
  { Polynomial.hasSMulPi' R S T with
    toFun := fun p z => algebraMap S T (aeval z p)
    map_one' := by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        ⊢ Eq ((fun p z => (algebraMap S T) ((Polynomial.aeval z) p)) 1) 1
      -/
      funext z
      /-
        case h
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        z : S
        ⊢ Eq ((fun p z => (algebraMap S T) ((Polynomial.aeval z) p)) 1 z) (1 z)
      -/
      simp only [Polynomial.aeval_one, Pi.one_apply, map_one]
      /-
        🎉 no goals
      -/
    map_mul' := fun f g => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        f g : Polynomial R
        ⊢ Eq ({ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_one …
      -/
      funext z
      /-
        case h
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        f g : Polynomial R
        z : S
        ⊢ Eq ({ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_one …
      -/
      simp only [Pi.mul_apply, map_mul]
      /-
        🎉 no goals
      -/
    map_zero' := by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        ⊢ Eq ((↑{ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_o …
      -/
      funext z
      /-
        case h
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        z : S
        ⊢ Eq ((↑{ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_o …
      -/
      simp only [Polynomial.aeval_zero, Pi.zero_apply, map_zero]
      /-
        🎉 no goals
      -/
    map_add' := fun f g => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        f g : Polynomial R
        ⊢ Eq ((↑{ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_o …
      -/
      funext z
      /-
        case h
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        f g : Polynomial R
        z : S
        ⊢ Eq ((↑{ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z) p), map_o …
      -/
      simp only [Polynomial.aeval_add, Pi.add_apply, map_add]
      /-
        🎉 no goals
      -/
    commutes' := fun p f => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        p : Polynomial R
        f : S → T
        ⊢ Eq (HMul.hMul ({ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z)  …
      -/
      funext z
      /-
        case h
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        p : Polynomial R
        f : S → T
        z : S
        ⊢ Eq (HMul.hMul ({ toFun := fun p z => (algebraMap S T) ((Polynomial.aeval z)  …
      -/
      exact mul_comm _ _
      /-
        🎉 no goals
      -/
    smul_def' := fun p f => by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S
        inst✝² : CommSemiring T
        inst✝¹ : Algebra R S
        inst✝ : Algebra S T
        p : Polynomial R
        f : S → T
        ⊢ Eq (HSMul.hSMul p f) (HMul.hMul ({ toFun := fun p z => (algebraMap S T) ((Po …
      -/
      funext z
      simp only [polynomial_smul_apply', Algebra.algebraMap_eq_smul_one, RingHom.coe_mk,
        MonoidHom.coe_mk, OneHom.coe_mk, Pi.mul_apply, Algebra.smul_mul_assoc, one_mul] }


@[simp]
theorem Polynomial.algebraMap_pi_eq_aeval :
    (algebraMap R[X] (S → T) : R[X] → S → T) = fun p z => algebraMap _ _ (aeval z p) :=
  rfl


@[simp]
theorem Polynomial.algebraMap_pi_self_eq_eval :
    (algebraMap R[X] (R → R) : R[X] → R → R) = fun p z => eval z p :=
  rfl


