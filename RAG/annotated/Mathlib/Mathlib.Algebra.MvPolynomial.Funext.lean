private theorem funext_fin {n : ℕ} {p : MvPolynomial (Fin n) R}
    (h : ∀ x : Fin n → R, eval x p = 0) : p = 0 := by
  induction n with
  | zero =>
    apply (MvPolynomial.isEmptyRingEquiv R (Fin 0)).injective
    rw [RingEquiv.map_zero]
    convert h finZeroElim
  | succ n ih =>
    apply (finSuccEquiv R n).injective
    simp only [map_zero]
    refine Polynomial.funext fun q => ?_
    rw [Polynomial.eval_zero]
    apply ih fun x => ?_
    calc _ = _ := eval_polynomial_eval_finSuccEquiv p _
         _ = 0 := h _


/-- Two multivariate polynomials over an infinite integral domain are equal
if they are equal upon evaluating them on an arbitrary assignment of the variables. -/
theorem funext {σ : Type*} {p q : MvPolynomial σ R} (h : ∀ x : σ → R, eval x p = eval x q) :
    p = q := by
  suffices ∀ p, (∀ x : σ → R, eval x p = 0) → p = 0 by
    rw [← sub_eq_zero, this (p - q)]
    simp only [h, RingHom.map_sub, forall_const, sub_self]
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    p q : MvPolynomial σ R
    h : ∀ (x : σ → R), Eq ((MvPolynomial.eval x) p) ((MvPolynomial.eval x) q)
    ⊢ ∀ (p : MvPolynomial σ R), (∀ (x : σ → R), Eq ((MvPolynomial.eval x) p) 0) →  …
  -/
  clear h p q
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    ⊢ ∀ (p : MvPolynomial σ R), (∀ (x : σ → R), Eq ((MvPolynomial.eval x) p) 0) →  …
  -/
  intro p h
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    p : MvPolynomial σ R
    h : ∀ (x : σ → R), Eq ((MvPolynomial.eval x) p) 0
    ⊢ Eq p 0
  -/
  obtain ⟨n, f, hf, p, rfl⟩ := exists_fin_rename p
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    n : Nat
    f : Fin n → σ
    hf : Function.Injective f
    p : MvPolynomial (Fin n) R
    h : ∀ (x : σ → R), Eq ((MvPolynomial.eval x) ((MvPolynomial.rename f) p)) 0
    ⊢ Eq ((MvPolynomial.rename f) p) 0
  -/
  suffices p = 0 by rw [this, map_zero]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    n : Nat
    f : Fin n → σ
    hf : Function.Injective f
    p : MvPolynomial (Fin n) R
    h : ∀ (x : σ → R), Eq ((MvPolynomial.eval x) ((MvPolynomial.rename f) p)) 0
    ⊢ Eq p 0
  -/
  apply funext_fin
  /-
    case intro.intro.intro.intro.h
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Infinite R
    σ : Type u_2
    n : Nat
    f : Fin n → σ
    hf : Function.Injective f
    p : MvPolynomial (Fin n) R
    h : ∀ (x : σ → R), Eq ((MvPolynomial.eval x) ((MvPolynomial.rename f) p)) 0
    ⊢ ∀ (x : Fin n → R), Eq ((MvPolynomial.eval x) p) 0
  -/
  intro x
  classical
    convert h (Function.extend f x 0)
    simp only [eval, eval₂Hom_rename, Function.extend_comp hf]


theorem funext_iff {σ : Type*} {p q : MvPolynomial σ R} :
    p = q ↔ ∀ x : σ → R, eval x p = eval x q :=
      /-
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : Infinite R
        σ : Type u_2
        p q : MvPolynomial σ R
        ⊢ Eq p q → ∀ (x : σ → R), Eq ((MvPolynomial.eval x) p) ((MvPolynomial.eval x) q)
      -/
  ⟨by rintro rfl; simp only [forall_const, eq_self_iff_true], funext⟩
                  /-
                    🎉 no goals
                  -/


