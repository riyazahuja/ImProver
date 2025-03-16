/-- The characteristic polynomial of `f : M →ₗ[R] M`. -/
def charpoly : R[X] :=
  (toMatrix (chooseBasis R M) (chooseBasis R M) f).charpoly


theorem charpoly_def : f.charpoly = (toMatrix (chooseBasis R M) (chooseBasis R M) f).charpoly :=
  rfl


theorem charpoly_monic : f.charpoly.Monic :=
  Matrix.charpoly_monic _


open Module in
lemma charpoly_natDegree [Nontrivial R] [StrongRankCondition R] :
    natDegree (charpoly f) = finrank R M := by
  /-
    R : Type u
    M : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    inst✝¹ : Nontrivial R
    inst✝ : StrongRankCondition R
    ⊢ Eq f.charpoly.natDegree (Module.finrank R M)
  -/
  rw [charpoly, Matrix.charpoly_natDegree_eq_dim, finrank_eq_card_chooseBasisIndex]
  /-
    🎉 no goals
  -/


/-- The **Cayley-Hamilton Theorem**, that the characteristic polynomial of a linear map, applied
to the linear map itself, is zero.

See `Matrix.aeval_self_charpoly` for the equivalent statement about matrices. -/
theorem aeval_self_charpoly : aeval f f.charpoly = 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq ((Polynomial.aeval f) f.charpoly) 0
  -/
  apply (LinearEquiv.map_eq_zero_iff (algEquivMatrix (chooseBasis R M)).toLinearEquiv).1
  rw [AlgEquiv.toLinearEquiv_apply, ← AlgEquiv.coe_algHom, ← Polynomial.aeval_algHom_apply _ _ _,
    charpoly_def]
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq ((Polynomial.aeval (↑(algEquivMatrix (Module.Free.chooseBasis R M)) f)) ( …
  -/
  exact Matrix.aeval_self_charpoly _
  /-
    🎉 no goals
  -/


theorem isIntegral : IsIntegral R f :=
  ⟨f.charpoly, ⟨charpoly_monic f, aeval_self_charpoly f⟩⟩


theorem minpoly_dvd_charpoly {K : Type u} {M : Type v} [Field K] [AddCommGroup M] [Module K M]
    [FiniteDimensional K M] (f : M →ₗ[K] M) : minpoly K f ∣ f.charpoly :=
  minpoly.dvd _ _ (aeval_self_charpoly f)


/-- Any endomorphism polynomial `p` is equivalent under evaluation to `p %ₘ f.charpoly`; that is,
`p` is equivalent to a polynomial with degree less than the dimension of the module. -/
theorem aeval_eq_aeval_mod_charpoly (p : R[X]) : aeval f p = aeval f (p %ₘ f.charpoly) :=
  (aeval_modByMonic_eq_self_of_root f.charpoly_monic f.aeval_self_charpoly).symm


/-- Any endomorphism power can be computed as the sum of endomorphism powers less than the
dimension of the module. -/
theorem pow_eq_aeval_mod_charpoly (k : ℕ) : f ^ k = aeval f (X ^ k %ₘ f.charpoly) := by
  /-
    R : Type u
    M : Type v
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    k : Nat
    ⊢ Eq (HPow.hPow f k) ((Polynomial.aeval f) ((HPow.hPow Polynomial.X k).modByMo …
  -/
  rw [← aeval_eq_aeval_mod_charpoly, map_pow, aeval_X]
  /-
    🎉 no goals
  -/


theorem minpoly_coeff_zero_of_injective [Nontrivial R] (hf : Function.Injective f) :
    (minpoly R f).coeff 0 ≠ 0 := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    inst✝ : Nontrivial R
    hf : Function.Injective ⇑f
    ⊢ Ne ((minpoly R f).coeff 0) 0
  -/
  intro h
  /-
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    inst✝ : Nontrivial R
    hf : Function.Injective ⇑f
    h : Eq ((minpoly R f).coeff 0) 0
    ⊢ False
  -/
  obtain ⟨P, hP⟩ := X_dvd_iff.2 h
  have hdegP : P.degree < (minpoly R f).degree := by
    rw [hP, mul_comm]
    refine degree_lt_degree_mul_X fun h => ?_
    rw [h, mul_zero] at hP
    exact minpoly.ne_zero (isIntegral f) hP
  have hPmonic : P.Monic := by
    suffices (minpoly R f).Monic by
      rwa [Monic.def, hP, mul_comm, leadingCoeff_mul_X, ← Monic.def] at this
    exact minpoly.monic (isIntegral f)
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    inst✝ : Nontrivial R
    hf : Function.Injective ⇑f
    h : Eq ((minpoly R f).coeff 0) 0
    P : Polynomial R
    hP : Eq (minpoly R f) (HMul.hMul Polynomial.X P)
    hdegP : LT.lt P.degree (minpoly R f).degree
    hPmonic : P.Monic
    ⊢ False
  -/
  have hzero : aeval f (minpoly R f) = 0 := minpoly.aeval _ _
  simp only [hP, mul_eq_comp, LinearMap.ext_iff, hf, aeval_X, map_eq_zero_iff, coe_comp,
    _root_.map_mul, zero_apply, Function.comp_apply] at hzero
  /-
    case intro
    R : Type u
    M : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Free R M
    inst✝¹ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    inst✝ : Nontrivial R
    hf : Function.Injective ⇑f
    h : Eq ((minpoly R f).coeff 0) 0
    P : Polynomial R
    hP : Eq (minpoly R f) (HMul.hMul Polynomial.X P)
    hdegP : LT.lt P.degree (minpoly R f).degree
    hPmonic : P.Monic
    hzero : ∀ (x : M), Eq (((Polynomial.aeval f) P) x) 0
    ⊢ False
  -/
  exact not_le.2 hdegP (minpoly.min _ _ hPmonic (LinearMap.ext hzero))
  /-
    🎉 no goals
  -/


