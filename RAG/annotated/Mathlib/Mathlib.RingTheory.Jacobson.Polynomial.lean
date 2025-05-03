theorem jacobson_bot_polynomial_le_sInf_map_maximal :
    jacobson (⊥ : Ideal R[X]) ≤ sInf (map (C : R →+* R[X]) '' { J : Ideal R | J.IsMaximal }) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ LE.le Bot.bot.jacobson (InfSet.sInf (Set.image (Ideal.map Polynomial.C) (set …
  -/
  refine le_sInf fun J => exists_imp.2 fun j hj => ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    J : Ideal (Polynomial R)
    j : Ideal R
    hj : And (Membership.mem (setOf fun J => J.IsMaximal) j) (Eq (Ideal.map Polyno …
    ⊢ LE.le Bot.bot.jacobson J
  -/
  haveI : j.IsMaximal := hj.1
  /-
    R : Type u_1
    inst✝ : CommRing R
    J : Ideal (Polynomial R)
    j : Ideal R
    hj : And (Membership.mem (setOf fun J => J.IsMaximal) j) (Eq (Ideal.map Polyno …
    this : j.IsMaximal
    ⊢ LE.le Bot.bot.jacobson J
  -/
  refine Trans.trans (jacobson_mono bot_le) (le_of_eq ?_ : J.jacobson ≤ J)
  suffices t : (⊥ : Ideal (Polynomial (R ⧸ j))).jacobson = ⊥ by
    rw [← hj.2, jacobson_eq_iff_jacobson_quotient_eq_bot]
    replace t := congr_arg (map (polynomialQuotientEquivQuotientPolynomial j).toRingHom) t
    rwa [map_jacobson_of_bijective _, map_bot] at t
    exact RingEquiv.bijective (polynomialQuotientEquivQuotientPolynomial j)
  /-
    R : Type u_1
    inst✝ : CommRing R
    J : Ideal (Polynomial R)
    j : Ideal R
    hj : And (Membership.mem (setOf fun J => J.IsMaximal) j) (Eq (Ideal.map Polyno …
    this : j.IsMaximal
    ⊢ Eq Bot.bot.jacobson Bot.bot
  -/
  refine eq_bot_iff.2 fun f hf => ?_
  /-
    R : Type u_1
    inst✝ : CommRing R
    J : Ideal (Polynomial R)
    j : Ideal R
    hj : And (Membership.mem (setOf fun J => J.IsMaximal) j) (Eq (Ideal.map Polyno …
    this : j.IsMaximal
    f : Polynomial (HasQuotient.Quotient R j)
    hf : Membership.mem Bot.bot.jacobson f
    ⊢ Membership.mem Bot.bot f
  -/
  have r1 : (X : (R ⧸ j)[X]) ≠ 0 := ne_of_apply_ne (coeff · 1) <| by simp
  /-
    R : Type u_1
    inst✝ : CommRing R
    J : Ideal (Polynomial R)
    j : Ideal R
    hj : And (Membership.mem (setOf fun J => J.IsMaximal) j) (Eq (Ideal.map Polyno …
    this : j.IsMaximal
    f : Polynomial (HasQuotient.Quotient R j)
    hf : Membership.mem Bot.bot.jacobson f
    r1 : Ne Polynomial.X 0
    ⊢ Membership.mem Bot.bot f
  -/
  simpa [r1] using eq_C_of_degree_eq_zero (degree_eq_zero_of_isUnit ((mem_jacobson_bot.1 hf) X))
  /-
    🎉 no goals
  -/


theorem jacobson_bot_polynomial_of_jacobson_bot (h : jacobson (⊥ : Ideal R) = ⊥) :
    jacobson (⊥ : Ideal R[X]) = ⊥ := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Eq Bot.bot.jacobson Bot.bot
    ⊢ Eq Bot.bot.jacobson Bot.bot
  -/
  refine eq_bot_iff.2 (le_trans jacobson_bot_polynomial_le_sInf_map_maximal ?_)
  refine fun f hf => (Submodule.mem_bot R[X]).2 <| Polynomial.ext fun n =>
    Trans.trans (?_ : coeff f n = 0) (coeff_zero n).symm
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Eq Bot.bot.jacobson Bot.bot
    f : Polynomial R
    hf : Membership.mem (InfSet.sInf (Set.image (Ideal.map Polynomial.C) (setOf fu …
    n : Nat
    ⊢ Eq (f.coeff n) 0
  -/
  suffices f.coeff n ∈ Ideal.jacobson ⊥ by rwa [h, Submodule.mem_bot] at this
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : Eq Bot.bot.jacobson Bot.bot
    f : Polynomial R
    hf : Membership.mem (InfSet.sInf (Set.image (Ideal.map Polynomial.C) (setOf fu …
    n : Nat
    ⊢ Membership.mem Bot.bot.jacobson (f.coeff n)
  -/
  exact mem_sInf.2 fun j hj => (mem_map_C_iff.1 ((mem_sInf.1 hf) ⟨j, ⟨hj.2, rfl⟩⟩)) n
  /-
    🎉 no goals
  -/


