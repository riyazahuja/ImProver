theorem quotient_mk_comp_C_injective [Field K] (I : Ideal (MvPolynomial σ K)) (hI : I ≠ ⊤) :
    Function.Injective ((Ideal.Quotient.mk I).comp MvPolynomial.C) := by
  /-
    σ : Type u
    K : Type v
    inst✝ : Field K
    I : Ideal (MvPolynomial σ K)
    hI : Ne I Top.top
    ⊢ Function.Injective ⇑((Ideal.Quotient.mk I).comp MvPolynomial.C)
  -/
  refine (injective_iff_map_eq_zero _).2 fun x hx => ?_
  /-
    σ : Type u
    K : Type v
    inst✝ : Field K
    I : Ideal (MvPolynomial σ K)
    hI : Ne I Top.top
    x : K
    hx : Eq (((Ideal.Quotient.mk I).comp MvPolynomial.C) x) 0
    ⊢ Eq x 0
  -/
  rw [RingHom.comp_apply, Ideal.Quotient.eq_zero_iff_mem] at hx
  /-
    σ : Type u
    K : Type v
    inst✝ : Field K
    I : Ideal (MvPolynomial σ K)
    hI : Ne I Top.top
    x : K
    hx : Membership.mem I (MvPolynomial.C x)
    ⊢ Eq x 0
  -/
  refine _root_.by_contradiction fun hx0 => absurd (I.eq_top_iff_one.2 ?_) hI
  /-
    σ : Type u
    K : Type v
    inst✝ : Field K
    I : Ideal (MvPolynomial σ K)
    hI : Ne I Top.top
    x : K
    hx : Membership.mem I (MvPolynomial.C x)
    hx0 : Not (Eq x 0)
    ⊢ Membership.mem I 1
  -/
  have := I.mul_mem_left (MvPolynomial.C x⁻¹) hx
  /-
    σ : Type u
    K : Type v
    inst✝ : Field K
    I : Ideal (MvPolynomial σ K)
    hI : Ne I Top.top
    x : K
    hx : Membership.mem I (MvPolynomial.C x)
    hx0 : Not (Eq x 0)
    this : Membership.mem I (HMul.hMul (MvPolynomial.C (Inv.inv x)) (MvPolynomial. …
    ⊢ Membership.mem I 1
  -/
  rwa [← MvPolynomial.C.map_mul, inv_mul_cancel₀ hx0, MvPolynomial.C_1] at this
  /-
    🎉 no goals
  -/


theorem rank_eq_lift : Module.rank K (MvPolynomial σ K) = lift.{v} #(σ →₀ ℕ) := by
  /-
    σ : Type u
    K : Type v
    inst✝¹ : CommRing K
    inst✝ : Nontrivial K
    ⊢ Eq (Module.rank K (MvPolynomial σ K)) (Cardinal.lift.{v, u} (Cardinal.mk (Fi …
  -/
  rw [← Cardinal.lift_inj, ← (basisMonomials σ K).mk_eq_rank, lift_lift, lift_umax.{u,v}]
  /-
    🎉 no goals
  -/


theorem rank_eq {σ : Type v} : Module.rank K (MvPolynomial σ K) = #(σ →₀ ℕ) := by
  /-
    K : Type v
    inst✝¹ : CommRing K
    inst✝ : Nontrivial K
    σ : Type v
    ⊢ Eq (Module.rank K (MvPolynomial σ K)) (Cardinal.mk (Finsupp σ Nat))
  -/
  rw [← Cardinal.lift_inj, ← (basisMonomials σ K).mk_eq_rank]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-07")] alias rank_mvPolynomial := rank_eq


theorem finrank_eq_zero [Nonempty σ] : Module.finrank K (MvPolynomial σ K) = 0 :=
  (basisMonomials σ K).linearIndependent.finrank_eq_zero_of_infinite


omit [Nontrivial K] in
theorem finrank_eq_one [IsEmpty σ] : Module.finrank K (MvPolynomial σ K) = 1 :=
  Module.rank_eq_one_iff_finrank_eq_one.mp <| by
    /-
      σ : Type u
      K : Type v
      inst✝¹ : CommRing K
      inst✝ : IsEmpty σ
      ⊢ Eq (Module.rank K (MvPolynomial σ K)) 1
    -/
                                           /-
                                             🎉 no goals
                                           -/
    cases subsingleton_or_nontrivial K <;> simp [rank_eq_lift]
                                           /-
                                             🎉 no goals
                                           -/


