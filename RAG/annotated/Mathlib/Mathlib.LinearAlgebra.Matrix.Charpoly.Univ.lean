/-- The universal characteristic polynomial for `n × n`-matrices,
is the charactistic polynomial of `Matrix.mvPolynomialX n n ℤ` with entries `Xᵢⱼ`.

Its `i`-th coefficient is a homogeneous polynomial of degree `n - i`,
see `Matrix.charpoly.univ_coeff_isHomogeneous`.

By evaluating the coefficients at the entries of a matrix `M`,
one obtains the characteristic polynomial of `M`,
see `Matrix.charpoly.univ_map_eval₂Hom`. -/
noncomputable
abbrev univ : Polynomial (MvPolynomial (n × n) R) :=
  charpoly <| mvPolynomialX n n R


open MvPolynomial RingHomClass in
@[simp]
lemma univ_map_eval₂Hom (M : n × n → S) :
    (univ R n).map (eval₂Hom f M) = charpoly (Matrix.of M.curry) := by
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    f : RingHom R S
    M : Prod n n → S
    ⊢ Eq (Polynomial.map (MvPolynomial.eval₂Hom f M) (Matrix.charpoly.univ R n)) ( …
  -/
  rw [univ, ← charpoly_map, coe_eval₂Hom, ← mvPolynomialX_map_eval₂ f (Matrix.of M.curry)]
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    f : RingHom R S
    M : Prod n n → S
    ⊢ Eq ((Matrix.mvPolynomialX n n R).map (MvPolynomial.eval₂ f M)).charpoly ((Ma …
  -/
  simp only [of_apply, Function.curry_apply, Prod.mk.eta]
  /-
    🎉 no goals
  -/


lemma univ_map_map :
    (univ R n).map (MvPolynomial.map f) = univ S n := by
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    f : RingHom R S
    ⊢ Eq (Polynomial.map (MvPolynomial.map f) (Matrix.charpoly.univ R n)) (Matrix. …
  -/
  rw [MvPolynomial.map, univ_map_eval₂Hom]; rfl
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
lemma univ_coeff_eval₂Hom (M : n × n → S) (i : ℕ) :
    MvPolynomial.eval₂Hom f M ((univ R n).coeff i) =
      (charpoly (Matrix.of M.curry)).coeff i := by
  /-
    R : Type u_1
    S : Type u_2
    n : Type u_3
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    f : RingHom R S
    M : Prod n n → S
    i : Nat
    ⊢ Eq ((MvPolynomial.eval₂Hom f M) ((Matrix.charpoly.univ R n).coeff i)) ((Matr …
  -/
  rw [← univ_map_eval₂Hom n f M, Polynomial.coeff_map]
  /-
    🎉 no goals
  -/


lemma univ_monic : (univ R n).Monic := charpoly_monic (mvPolynomialX n n R)


lemma univ_natDegree [Nontrivial R] : (univ R n).natDegree = Fintype.card n :=
  charpoly_natDegree_eq_dim (mvPolynomialX n n R)


@[simp]
lemma univ_coeff_card : (univ R n).coeff (Fintype.card n) = 1 := by
  suffices Polynomial.coeff (univ ℤ n) (Fintype.card n) = 1 by
    rw [← univ_map_map n (Int.castRingHom R), Polynomial.coeff_map, this, _root_.map_one]
  /-
    R : Type u_1
    n : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq ((Matrix.charpoly.univ Int n).coeff (Fintype.card n)) 1
  -/
  rw [← univ_natDegree ℤ n]
  /-
    R : Type u_1
    n : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    ⊢ Eq ((Matrix.charpoly.univ Int n).coeff (Matrix.charpoly.univ Int n).natDegre …
  -/
  exact (univ_monic ℤ n).leadingCoeff
  /-
    🎉 no goals
  -/


open MvPolynomial in
lemma optionEquivLeft_symm_univ_isHomogeneous :
    ((optionEquivLeft R (n × n)).symm (univ R n)).IsHomogeneous (Fintype.card n) := by
  have aux : Fintype.card n = 0 + ∑ i : n, 1 := by
    simp only [zero_add, Finset.sum_const, smul_eq_mul, mul_one, Fintype.card]
  simp only [aux, univ, charpoly, charmatrix, scalar_apply, RingHom.mapMatrix_apply, det_apply',
    sub_apply, map_apply, of_apply, map_sum, _root_.map_mul, map_intCast, map_prod, map_sub,
    optionEquivLeft_symm_apply, Polynomial.aevalTower_C, rename_X, diagonal, mvPolynomialX]
  /-
    R : Type u_1
    n : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
    ⊢ (Finset.univ.sum fun x => HMul.hMul (↑↑(Equiv.Perm.sign x)) (Finset.univ.pro …
  -/
  apply IsHomogeneous.sum
  /-
    case h
    R : Type u_1
    n : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
    ⊢ ∀ (i : Equiv.Perm n), Membership.mem Finset.univ i → (HMul.hMul (↑↑(Equiv.Pe …
  -/
  rintro i -
  /-
    case h
    R : Type u_1
    n : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
    i : Equiv.Perm n
    ⊢ (HMul.hMul (↑↑(Equiv.Perm.sign i)) (Finset.univ.prod fun x => HSub.hSub ((Po …
  -/
  apply IsHomogeneous.mul
    /-
      case h.hφ
      R : Type u_1
      n : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
      i : Equiv.Perm n
      ⊢ (↑↑(Equiv.Perm.sign i)).IsHomogeneous 0
    -/
  · apply isHomogeneous_C
    /-
      🎉 no goals
    -/
    /-
      case h.hψ
      R : Type u_1
      n : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
      i : Equiv.Perm n
      ⊢ (Finset.univ.prod fun x => HSub.hSub ((Polynomial.aevalTower (MvPolynomial.r …
    -/
  · apply IsHomogeneous.prod
    /-
      case h.hψ.h
      R : Type u_1
      n : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
      i : Equiv.Perm n
      ⊢ ∀ (i_1 : n), Membership.mem Finset.univ i_1 → (HSub.hSub ((Polynomial.aevalT …
    -/
    rintro j -
    /-
      case h.hψ.h
      R : Type u_1
      n : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Fintype n
      inst✝ : DecidableEq n
      aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
      i : Equiv.Perm n
      j : n
      ⊢ (HSub.hSub ((Polynomial.aevalTower (MvPolynomial.rename Option.some) (MvPoly …
    -/
    by_cases h : i j = j
      /-
        case pos
        R : Type u_1
        n : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
        i : Equiv.Perm n
        j : n
        h : Eq (i j) j
        ⊢ (HSub.hSub ((Polynomial.aevalTower (MvPolynomial.rename Option.some) (MvPoly …
      -/
    · simp only [h, ↓reduceIte, Polynomial.aevalTower_X, IsHomogeneous.sub, isHomogeneous_X]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        n : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Fintype n
        inst✝ : DecidableEq n
        aux : Eq (Fintype.card n) (HAdd.hAdd 0 (Finset.univ.sum fun i => 1))
        i : Equiv.Perm n
        j : n
        h : Not (Eq (i j) j)
        ⊢ (HSub.hSub ((Polynomial.aevalTower (MvPolynomial.rename Option.some) (MvPoly …
      -/
    · simp only [h, ↓reduceIte, map_zero, zero_sub, (isHomogeneous_X _ _).neg]
      /-
        🎉 no goals
      -/


lemma univ_coeff_isHomogeneous (i j : ℕ) (h : i + j = Fintype.card n) :
    ((univ R n).coeff i).IsHomogeneous j :=
  (optionEquivLeft_symm_univ_isHomogeneous R n).coeff_isHomogeneous_of_optionEquivLeft_symm _ _ h


