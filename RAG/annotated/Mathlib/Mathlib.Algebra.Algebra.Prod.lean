instance algebra : Algebra R (A × B) :=
  { Prod.instModule,
    RingHom.prod (algebraMap R A) (algebraMap R B) with
    commutes' := by
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        ⊢ ∀ (r : R) (x : Prod A B), Eq (HMul.hMul (__src✝ r) x) (HMul.hMul x (__src✝ r))
      -/
      rintro r ⟨a, b⟩
      /-
        case mk
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        r : R
        a : A
        b : B
        ⊢ Eq (HMul.hMul (__src✝ r) { fst := a, snd := b }) (HMul.hMul { fst := a, snd  …
      -/
      dsimp
      /-
        case mk
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        r : R
        a : A
        b : B
        ⊢ Eq { fst := HMul.hMul ((algebraMap R A) r) a, snd := HMul.hMul ((algebraMap  …
      -/
      rw [commutes r a, commutes r b]
      /-
        🎉 no goals
      -/
    smul_def' := by
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        ⊢ ∀ (r : R) (x : Prod A B), Eq (HSMul.hSMul r x) (HMul.hMul (__src✝ r) x)
      -/
      rintro r ⟨a, b⟩
      /-
        case mk
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        r : R
        a : A
        b : B
        ⊢ Eq (HSMul.hSMul r { fst := a, snd := b }) (HMul.hMul (__src✝ r) { fst := a,  …
      -/
      dsimp
      /-
        case mk
        R : Type u_1
        A : Type u_2
        B : Type u_3
        C : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Algebra R A
        inst✝³ : Semiring B
        inst✝² : Algebra R B
        inst✝¹ : Semiring C
        inst✝ : Algebra R C
        r : R
        a : A
        b : B
        ⊢ Eq { fst := HSMul.hSMul r a, snd := HSMul.hSMul r b } { fst := HMul.hMul ((a …
      -/
      rw [Algebra.smul_def r a, Algebra.smul_def r b] }
      /-
        🎉 no goals
      -/


@[simp]
theorem algebraMap_apply (r : R) : algebraMap R (A × B) r = (algebraMap R A r, algebraMap R B r) :=
  rfl


/-- First projection as `AlgHom`. -/
def fst : A × B →ₐ[R] A :=
  { RingHom.fst A B with commutes' := fun _r => rfl }


/-- Second projection as `AlgHom`. -/
def snd : A × B →ₐ[R] B :=
  { RingHom.snd A B with commutes' := fun _r => rfl }


/-- The `Pi.prod` of two morphisms is a morphism. -/
@[simps!]
def prod (f : A →ₐ[R] B) (g : A →ₐ[R] C) : A →ₐ[R] B × C :=
  { f.toRingHom.prod g.toRingHom with
    commutes' := fun r => by
      simp only [toRingHom_eq_coe, RingHom.toFun_eq_coe, RingHom.prod_apply, coe_toRingHom,
        commutes, Prod.algebraMap_apply] }


theorem coe_prod (f : A →ₐ[R] B) (g : A →ₐ[R] C) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[simp]
                                                                                         /-
                                                                                           R : Type u_1
                                                                                           A : Type u_2
                                                                                           B : Type u_3
                                                                                           C : Type u_4
                                                                                           inst✝⁶ : CommSemiring R
                                                                                           inst✝⁵ : Semiring A
                                                                                           inst✝⁴ : Algebra R A
                                                                                           inst✝³ : Semiring B
                                                                                           inst✝² : Algebra R B
                                                                                           inst✝¹ : Semiring C
                                                                                           inst✝ : Algebra R C
                                                                                           f : AlgHom R A B
                                                                                           g : AlgHom R A C
                                                                                           ⊢ Eq ((AlgHom.fst R B C).comp (f.prod g)) f
                                                                                         -/
theorem fst_prod (f : A →ₐ[R] B) (g : A →ₐ[R] C) : (fst R B C).comp (prod f g) = f := by ext; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp]
                                                                                         /-
                                                                                           R : Type u_1
                                                                                           A : Type u_2
                                                                                           B : Type u_3
                                                                                           C : Type u_4
                                                                                           inst✝⁶ : CommSemiring R
                                                                                           inst✝⁵ : Semiring A
                                                                                           inst✝⁴ : Algebra R A
                                                                                           inst✝³ : Semiring B
                                                                                           inst✝² : Algebra R B
                                                                                           inst✝¹ : Semiring C
                                                                                           inst✝ : Algebra R C
                                                                                           f : AlgHom R A B
                                                                                           g : AlgHom R A C
                                                                                           ⊢ Eq ((AlgHom.snd R B C).comp (f.prod g)) g
                                                                                         -/
theorem snd_prod (f : A →ₐ[R] B) (g : A →ₐ[R] C) : (snd R B C).comp (prod f g) = g := by ext; rfl
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


@[simp]
theorem prod_fst_snd : prod (fst R A B) (snd R A B) = 1 :=
  DFunLike.coe_injective Pi.prod_fst_snd


/-- Taking the product of two maps with the same domain is equivalent to taking the product of
their codomains. -/
@[simps]
def prodEquiv : (A →ₐ[R] B) × (A →ₐ[R] C) ≃ (A →ₐ[R] B × C) where
  toFun f := f.1.prod f.2
  invFun f := ((fst _ _ _).comp f, (snd _ _ _).comp f)
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     C : Type u_4
                     inst✝⁶ : CommSemiring R
                     inst✝⁵ : Semiring A
                     inst✝⁴ : Algebra R A
                     inst✝³ : Semiring B
                     inst✝² : Algebra R B
                     inst✝¹ : Semiring C
                     inst✝ : Algebra R C
                     f : Prod (AlgHom R A B) (AlgHom R A C)
                     ⊢ Eq ((fun f => { fst := (AlgHom.fst R B C).comp f, snd := (AlgHom.snd R B C). …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      C : Type u_4
                      inst✝⁶ : CommSemiring R
                      inst✝⁵ : Semiring A
                      inst✝⁴ : Algebra R A
                      inst✝³ : Semiring B
                      inst✝² : Algebra R B
                      inst✝¹ : Semiring C
                      inst✝ : Algebra R C
                      f : AlgHom R A (Prod B C)
                      ⊢ Eq ((fun f => f.1.prod f.2) ((fun f => { fst := (AlgHom.fst R B C).comp f, s …
                    -/
                            /-
                              🎉 no goals
                            -/
  right_inv f := by ext <;> rfl
                            /-
                              🎉 no goals
                            -/


/-- `Prod.map` of two algebra homomorphisms. -/
def prodMap {D : Type*} [Semiring D] [Algebra R D] (f : A →ₐ[R] B) (g : C →ₐ[R] D) :
    A × C →ₐ[R] B × D :=
  { toRingHom := f.toRingHom.prodMap g.toRingHom
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               C : Type u_4
                               inst✝⁸ : CommSemiring R
                               inst✝⁷ : Semiring A
                               inst✝⁶ : Algebra R A
                               inst✝⁵ : Semiring B
                               inst✝⁴ : Algebra R B
                               inst✝³ : Semiring C
                               inst✝² : Algebra R C
                               D : Type u_5
                               inst✝¹ : Semiring D
                               inst✝ : Algebra R D
                               f : AlgHom R A B
                               g : AlgHom R C D
                               r : R
                               ⊢ Eq ((↑↑(f.prodMap g.toRingHom)).toFun ((algebraMap R (Prod A C)) r)) ((algeb …
                             -/
    commutes' := fun r => by simp [commutes] }
                             /-
                               🎉 no goals
                             -/


