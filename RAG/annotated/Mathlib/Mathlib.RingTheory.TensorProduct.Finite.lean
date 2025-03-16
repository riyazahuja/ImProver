theorem Subalgebra.finite_sup {K L : Type*} [CommSemiring K] [CommSemiring L] [Algebra K L]
    (E1 E2 : Subalgebra K L) [Module.Finite K E1] [Module.Finite K E2] :
    Module.Finite K ↥(E1 ⊔ E2) := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommSemiring K
    inst✝³ : CommSemiring L
    inst✝² : Algebra K L
    E1 E2 : Subalgebra K L
    inst✝¹ : Module.Finite K (Subtype fun x => Membership.mem E1 x)
    inst✝ : Module.Finite K (Subtype fun x => Membership.mem E2 x)
    ⊢ Module.Finite K (Subtype fun x => Membership.mem (Max.max E1 E2) x)
  -/
  rw [← E1.range_val, ← E2.range_val, ← Algebra.TensorProduct.productMap_range]
  /-
    K : Type u_1
    L : Type u_2
    inst✝⁴ : CommSemiring K
    inst✝³ : CommSemiring L
    inst✝² : Algebra K L
    E1 E2 : Subalgebra K L
    inst✝¹ : Module.Finite K (Subtype fun x => Membership.mem E1 x)
    inst✝ : Module.Finite K (Subtype fun x => Membership.mem E2 x)
    ⊢ Module.Finite K (Subtype fun x => Membership.mem (Algebra.TensorProduct.prod …
  -/
  exact Module.Finite.range (Algebra.TensorProduct.productMap E1.val E2.val).toLinearMap
  /-
    🎉 no goals
  -/


open TensorProduct in
lemma RingHom.surjective_of_tmul_eq_tmul_of_finite {R S}
    [CommRing R] [CommRing S] [Algebra R S] [Module.Finite R S]
    (h₁ : ∀ s : S, s ⊗ₜ[R] 1 = 1 ⊗ₜ s) : Function.Surjective (algebraMap R S) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Module.Finite R S
    h₁ : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
    ⊢ Function.Surjective ⇑(algebraMap R S)
  -/
  let R' := LinearMap.range (Algebra.ofId R S).toLinearMap
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Module.Finite R S
    h₁ : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
    R' : Submodule R S := LinearMap.range (Algebra.ofId R S).toLinearMap
    ⊢ Function.Surjective ⇑(algebraMap R S)
  -/
  cases' subsingleton_or_nontrivial (S ⧸ R') with h
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : Module.Finite R S
      h₁ : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
      R' : Submodule R S := LinearMap.range (Algebra.ofId R S).toLinearMap
      h : Subsingleton (HasQuotient.Quotient S R')
      ⊢ Function.Surjective ⇑(algebraMap R S)
    -/
  · rwa [Submodule.subsingleton_quotient_iff_eq_top, LinearMap.range_eq_top] at h
    /-
      🎉 no goals
    -/
  have : Subsingleton ((S ⧸ R') ⊗[R] (S ⧸ R')) := by
    refine subsingleton_of_forall_eq 0 fun y ↦ ?_
    induction y with
    | zero => rfl
    | add a b e₁ e₂ => rwa [e₁, zero_add]
    | tmul x y =>
      obtain ⟨x, rfl⟩ := R'.mkQ_surjective x
      obtain ⟨y, rfl⟩ := R'.mkQ_surjective y
      obtain ⟨s, hs⟩ : ∃ s, 1 ⊗ₜ[R] s = x ⊗ₜ[R] y := by
        use x * y
        trans x ⊗ₜ 1 * 1 ⊗ₜ y
        · simp [h₁]
        · simp
      have : R'.mkQ 1 = 0 := (Submodule.Quotient.mk_eq_zero R').mpr ⟨1, map_one (algebraMap R S)⟩
      rw [← map_tmul R'.mkQ R'.mkQ, ← hs, map_tmul, this, zero_tmul]
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Module.Finite R S
    h₁ : ∀ (s : S), Eq (TensorProduct.tmul R s 1) (TensorProduct.tmul R 1 s)
    R' : Submodule R S := LinearMap.range (Algebra.ofId R S).toLinearMap
    h✝ : Nontrivial (HasQuotient.Quotient S R')
    this : Subsingleton (TensorProduct R (HasQuotient.Quotient S R') (HasQuotient. …
    ⊢ Function.Surjective ⇑(algebraMap R S)
  -/
  cases false_of_nontrivial_of_subsingleton ((S ⧸ R') ⊗[R] (S ⧸ R'))
  /-
    🎉 no goals
  -/

