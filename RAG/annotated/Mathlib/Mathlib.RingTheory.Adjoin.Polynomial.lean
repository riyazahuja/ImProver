@[simp]
theorem adjoin_X : Algebra.adjoin R ({X} : Set R[X]) = ⊤ := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    ⊢ Eq (Algebra.adjoin R (Singleton.singleton Polynomial.X)) Top.top
  -/
  refine top_unique fun p _hp => ?_
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Polynomial R
    _hp : Membership.mem Top.top p
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton Polynomial.X)) p
  -/
  set S := Algebra.adjoin R ({X} : Set R[X])
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Polynomial R
    _hp : Membership.mem Top.top p
    S : Subalgebra R (Polynomial R) := Algebra.adjoin R (Singleton.singleton Polyn …
    ⊢ Membership.mem S p
  -/
  rw [← sum_monomial_eq p]; simp only [← smul_X_eq_monomial, Sum]
  /-
    R : Type u
    inst✝ : CommSemiring R
    p : Polynomial R
    _hp : Membership.mem Top.top p
    S : Subalgebra R (Polynomial R) := Algebra.adjoin R (Singleton.singleton Polyn …
    ⊢ Membership.mem S (p.sum fun n a => HSMul.hSMul a (HPow.hPow Polynomial.X n))
  -/
  exact S.sum_mem fun n _hn => S.smul_mem (S.pow_mem (Algebra.subset_adjoin rfl) _) _
  /-
    🎉 no goals
  -/


theorem _root_.Algebra.adjoin_singleton_eq_range_aeval (x : A) :
    Algebra.adjoin R {x} = (Polynomial.aeval x).range := by
  /-
    R : Type u
    A : Type z
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    ⊢ Eq (Algebra.adjoin R (Singleton.singleton x)) (Polynomial.aeval x).range
  -/
  rw [← Algebra.map_top, ← adjoin_X, AlgHom.map_adjoin, Set.image_singleton, aeval_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem aeval_mem_adjoin_singleton :
    aeval x p ∈ Algebra.adjoin R {x} := by
  /-
    R : Type u
    A : Type z
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    p : Polynomial R
    x : A
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) ((Polynomial.aeval …
  -/
  simpa only [Algebra.adjoin_singleton_eq_range_aeval] using Set.mem_range_self p
  /-
    🎉 no goals
  -/


instance instCommSemiringAdjoinSingleton :
    CommSemiring <| Algebra.adjoin R {x} :=
  { mul_comm := fun ⟨p, hp⟩ ⟨q, hq⟩ ↦ by
      /-
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : CommSemiring A'
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p✝ q✝ : Polynomial R
        x : A
        x✝¹ x✝ : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.single …
        p : A
        hp : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) p
        q : A
        hq : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) q
        ⊢ Eq (HMul.hMul ⟨p, hp⟩ ⟨q, hq⟩) (HMul.hMul ⟨q, hq⟩ ⟨p, hp⟩)
      -/
      obtain ⟨p', rfl⟩ := Algebra.adjoin_singleton_eq_range_aeval R x ▸ hp
      /-
        case intro
        R : Type u
        S : Type v
        T : Type w
        A : Type z
        A' : Type u_1
        B : Type u_2
        a b : R
        n : Nat
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : CommSemiring A'
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        p q✝ : Polynomial R
        x : A
        x✝¹ x✝ : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.single …
        q : A
        hq : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) q
        p' : Polynomial R
        hp : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) ((Polynomial.ae …
        ⊢ Eq (HMul.hMul ⟨(Polynomial.aeval x).toRingHom p', hp⟩ ⟨q, hq⟩) (HMul.hMul ⟨q …
      -/
      obtain ⟨q', rfl⟩ := Algebra.adjoin_singleton_eq_range_aeval R x ▸ hq
      simp only [AlgHom.toRingHom_eq_coe, RingHom.coe_coe, MulMemClass.mk_mul_mk, ← map_mul,
        mul_comm p' q'] }


instance instCommRingAdjoinSingleton {R A : Type*} [CommRing R] [Ring A] [Algebra R A] (x : A) :
    CommRing <| Algebra.adjoin R {x} :=
  { mul_comm := mul_comm }


