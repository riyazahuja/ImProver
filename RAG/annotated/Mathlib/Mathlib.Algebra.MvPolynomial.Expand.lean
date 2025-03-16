/-- Expand the polynomial by a factor of p, so `∑ aₙ xⁿ` becomes `∑ aₙ xⁿᵖ`.

See also `Polynomial.expand`. -/
noncomputable def expand (p : ℕ) : MvPolynomial σ R →ₐ[R] MvPolynomial σ R :=
  { (eval₂Hom C fun i ↦ X i ^ p : MvPolynomial σ R →+* MvPolynomial σ R) with
    commutes' := fun _ ↦ eval₂Hom_C _ _ _ }


theorem expand_C (p : ℕ) (r : R) : expand p (C r : MvPolynomial σ R) = C r :=
  eval₂Hom_C _ _ _


@[simp]
theorem expand_X (p : ℕ) (i : σ) : expand p (X i : MvPolynomial σ R) = X i ^ p :=
  eval₂Hom_X' _ _ _


@[simp]
theorem expand_monomial (p : ℕ) (d : σ →₀ ℕ) (r : R) :
    expand p (monomial d r) = C r * ∏ i ∈ d.support, (X i ^ p) ^ d i :=
  bind₁_monomial _ _ _


theorem expand_one_apply (f : MvPolynomial σ R) : expand 1 f = f := by
  simp only [expand, pow_one, eval₂Hom_eq_bind₂, bind₂_C_left, RingHom.toMonoidHom_eq_coe,
    RingHom.coe_monoidHom_id, AlgHom.coe_mk, RingHom.coe_mk, MonoidHom.id_apply, RingHom.id_apply]


@[simp]
theorem expand_one : expand 1 = AlgHom.id R (MvPolynomial σ R) := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.expand 1) (AlgHom.id R (MvPolynomial σ R))
  -/
  ext1 f
  /-
    case hf
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    f : σ
    ⊢ Eq ((MvPolynomial.expand 1) (MvPolynomial.X f)) ((AlgHom.id R (MvPolynomial  …
  -/
  rw [expand_one_apply, AlgHom.id_apply]
  /-
    🎉 no goals
  -/


theorem expand_comp_bind₁ (p : ℕ) (f : σ → MvPolynomial τ R) :
    (expand p).comp (bind₁ f) = bind₁ fun i ↦ expand p (f i) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    p : Nat
    f : σ → MvPolynomial τ R
    ⊢ Eq ((MvPolynomial.expand p).comp (MvPolynomial.bind₁ f)) (MvPolynomial.bind₁ …
  -/
  apply algHom_ext
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    p : Nat
    f : σ → MvPolynomial τ R
    ⊢ ∀ (i : σ), Eq (((MvPolynomial.expand p).comp (MvPolynomial.bind₁ f)) (MvPoly …
  -/
  intro i
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    p : Nat
    f : σ → MvPolynomial τ R
    i : σ
    ⊢ Eq (((MvPolynomial.expand p).comp (MvPolynomial.bind₁ f)) (MvPolynomial.X i) …
  -/
  simp only [AlgHom.comp_apply, bind₁_X_right]
  /-
    🎉 no goals
  -/


theorem expand_bind₁ (p : ℕ) (f : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    expand p (bind₁ f φ) = bind₁ (fun i ↦ expand p (f i)) φ := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    p : Nat
    f : σ → MvPolynomial τ R
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.expand p) ((MvPolynomial.bind₁ f) φ)) ((MvPolynomial.bind₁ …
  -/
  rw [← AlgHom.comp_apply, expand_comp_bind₁]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_expand (f : R →+* S) (p : ℕ) (φ : MvPolynomial σ R) :
                                                  /-
                                                    σ : Type u_1
                                                    R : Type u_3
                                                    S : Type u_4
                                                    inst✝¹ : CommSemiring R
                                                    inst✝ : CommSemiring S
                                                    f : RingHom R S
                                                    p : Nat
                                                    φ : MvPolynomial σ R
                                                    ⊢ Eq ((MvPolynomial.map f) ((MvPolynomial.expand p) φ)) ((MvPolynomial.expand  …
                                                  -/
    map f (expand p φ) = expand p (map f φ) := by simp [expand, map_bind₁]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem rename_expand (f : σ → τ) (p : ℕ) (φ : MvPolynomial σ R) :
    rename f (expand p φ) = expand p (rename f φ) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    f : σ → τ
    p : Nat
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.rename f) ((MvPolynomial.expand p) φ)) ((MvPolynomial.expa …
  -/
  simp [expand, bind₁_rename, rename_bind₁, Function.comp_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem rename_comp_expand (f : σ → τ) (p : ℕ) :
    (rename f).comp (expand p) =
      (expand p).comp (rename f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    f : σ → τ
    p : Nat
    ⊢ Eq ((MvPolynomial.rename f).comp (MvPolynomial.expand p)) ((MvPolynomial.exp …
  -/
  ext1 φ
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    f : σ → τ
    p : Nat
    φ : σ
    ⊢ Eq (((MvPolynomial.rename f).comp (MvPolynomial.expand p)) (MvPolynomial.X φ …
  -/
  simp only [rename_expand, AlgHom.comp_apply]
  /-
    🎉 no goals
  -/


