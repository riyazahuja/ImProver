@[simp]
theorem minpoly_toLin' : minpoly R (toLin' M) = minpoly R M :=
  minpoly.algEquiv_eq (toLinAlgEquiv' : Matrix n n R ≃ₐ[R] _) M


@[simp]
theorem minpoly_toLin (b : Basis n R N) (M : Matrix n n R) :
    minpoly R (toLin b b M) = minpoly R M :=
  minpoly.algEquiv_eq (toLinAlgEquiv b : Matrix n n R ≃ₐ[R] _) M


theorem isIntegral : IsIntegral R M :=
  ⟨M.charpoly, ⟨charpoly_monic M, aeval_self_charpoly M⟩⟩


theorem minpoly_dvd_charpoly {K : Type*} [Field K] (M : Matrix n n K) : minpoly K M ∣ M.charpoly :=
  minpoly.dvd _ _ (aeval_self_charpoly M)


@[simp]
theorem minpoly_toMatrix' (f : (n → R) →ₗ[R] n → R) : minpoly R (toMatrix' f) = minpoly R f :=
  minpoly.algEquiv_eq (toMatrixAlgEquiv' : _ ≃ₐ[R] Matrix n n R) f


@[simp]
theorem minpoly_toMatrix (b : Basis n R N) (f : N →ₗ[R] N) :
    minpoly R (toMatrix b b f) = minpoly R f :=
  minpoly.algEquiv_eq (toMatrixAlgEquiv b : _ ≃ₐ[R] Matrix n n R) f


/-- The characteristic polynomial of the map `fun x => a * x` is the minimal polynomial of `a`.

In combination with `det_eq_sign_charpoly_coeff` or `trace_eq_neg_charpoly_coeff`
and a bit of rewriting, this will allow us to conclude the
field norm resp. trace of `x` is the product resp. sum of `x`'s conjugates.
-/
theorem charpoly_leftMulMatrix {S : Type*} [Ring S] [Algebra R S] (h : PowerBasis R S) :
    (leftMulMatrix h.basis h.gen).charpoly = minpoly R h.gen := by
  /-
    R : Type u
    inst✝² : CommRing R
    S : Type u_1
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : PowerBasis R S
    ⊢ Eq ((Algebra.leftMulMatrix h.basis) h.gen).charpoly (minpoly R h.gen)
  -/
  cases subsingleton_or_nontrivial R; · subsingleton
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    R : Type u
    inst✝² : CommRing R
    S : Type u_1
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : PowerBasis R S
    h✝ : Nontrivial R
    ⊢ Eq ((Algebra.leftMulMatrix h.basis) h.gen).charpoly (minpoly R h.gen)
  -/
  apply minpoly.unique' R h.gen (charpoly_monic _)
  · apply (injective_iff_map_eq_zero (G := S) (leftMulMatrix _)).mp
      (leftMulMatrix_injective h.basis)
    /-
      case inr.hp.a
      R : Type u
      inst✝² : CommRing R
      S : Type u_1
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      h : PowerBasis R S
      h✝ : Nontrivial R
      ⊢ Eq ((Algebra.leftMulMatrix h.basis) ((Polynomial.aeval h.gen) ((Algebra.left …
    -/
    rw [← Polynomial.aeval_algHom_apply, aeval_self_charpoly]
    /-
      🎉 no goals
    -/
  /-
    case inr.hl
    R : Type u
    inst✝² : CommRing R
    S : Type u_1
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : PowerBasis R S
    h✝ : Nontrivial R
    ⊢ ∀ (q : Polynomial R), LT.lt q.degree ((Algebra.leftMulMatrix h.basis) h.gen) …
  -/
  refine fun q hq => or_iff_not_imp_left.2 fun h0 => ?_
  /-
    case inr.hl
    R : Type u
    inst✝² : CommRing R
    S : Type u_1
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : PowerBasis R S
    h✝ : Nontrivial R
    q : Polynomial R
    hq : LT.lt q.degree ((Algebra.leftMulMatrix h.basis) h.gen).charpoly.degree
    h0 : Not (Eq q 0)
    ⊢ Ne ((Polynomial.aeval h.gen) q) 0
  -/
  rw [Matrix.charpoly_degree_eq_dim, Fintype.card_fin] at hq
  /-
    case inr.hl
    R : Type u
    inst✝² : CommRing R
    S : Type u_1
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : PowerBasis R S
    h✝ : Nontrivial R
    q : Polynomial R
    hq : LT.lt q.degree ↑h.dim
    h0 : Not (Eq q 0)
    ⊢ Ne ((Polynomial.aeval h.gen) q) 0
  -/
  contrapose! hq; exact h.dim_le_degree_of_root h0 hq
                  /-
                    🎉 no goals
                  -/


