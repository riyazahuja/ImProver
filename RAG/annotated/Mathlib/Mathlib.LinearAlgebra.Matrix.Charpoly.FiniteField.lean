@[simp]
theorem FiniteField.Matrix.charpoly_pow_card {K : Type*} [Field K] [Fintype K] (M : Matrix n n K) :
    (M ^ Fintype.card K).charpoly = M.charpoly := by
  /-
    n : Type u_1
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Fintype K
    M : Matrix n n K
    ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
  -/
  cases (isEmpty_or_nonempty n).symm
    /-
      case inl
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
    -/
  · cases' CharP.exists K with p hp; letI := hp
    /-
      case inl.intro
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp : CharP K p
      this : CharP K p := hp
      ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
    -/
    rcases FiniteField.card K p with ⟨⟨k, kpos⟩, ⟨hp, hk⟩⟩
    /-
      case inl.intro.intro.mk.intro
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p ↑⟨k, kpos⟩)
      ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
    -/
    haveI : Fact p.Prime := ⟨hp⟩
    /-
      case inl.intro.intro.mk.intro
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p ↑⟨k, kpos⟩)
      this : Fact (Nat.Prime p)
      ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
    -/
    dsimp at hk; rw [hk]
    /-
      case inl.intro.intro.mk.intro
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq (HPow.hPow M (HPow.hPow p k)).charpoly M.charpoly
    -/
    apply (frobenius_inj K[X] p).iterate k
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq (Nat.iterate (⇑(frobenius (Polynomial K) p)) k (HPow.hPow M (HPow.hPow p  …
    -/
    repeat' rw [iterate_frobenius (R := K[X])]; rw [← hk]
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq (HPow.hPow (HPow.hPow M (Fintype.card K)).charpoly (Fintype.card K)) (HPo …
    -/
    rw [← FiniteField.expand_card]
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq ((Polynomial.expand K (Fintype.card K)) (HPow.hPow M (Fintype.card K)).ch …
    -/
    unfold charpoly
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq ((Polynomial.expand K (Fintype.card K)) (HPow.hPow M (Fintype.card K)).ch …
    -/
    rw [AlgHom.map_det, ← coe_detMonoidHom, ← (detMonoidHom : Matrix n n K[X] →* K[X]).map_pow]
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq (Matrix.detMonoidHom ((Polynomial.expand K (Fintype.card K)).mapMatrix (H …
    -/
    apply congr_arg det
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq ((Polynomial.expand K (Fintype.card K)).mapMatrix (HPow.hPow M (Fintype.c …
    -/
    refine matPolyEquiv.injective ?_
    /-
      case inl.intro.intro.mk.intro.a
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : Nonempty n
      p : Nat
      hp✝ : CharP K p
      this✝ : CharP K p := hp✝
      k : Nat
      kpos : LT.lt 0 k
      hp : Nat.Prime p
      hk : Eq (Fintype.card K) (HPow.hPow p k)
      this : Fact (Nat.Prime p)
      ⊢ Eq (matPolyEquiv ((Polynomial.expand K (Fintype.card K)).mapMatrix (HPow.hPo …
    -/
    rw [map_pow, matPolyEquiv_charmatrix, hk, sub_pow_char_pow_of_commute, ← C_pow]
      /-
        case inl.intro.intro.mk.intro.a
        n : Type u_1
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Fintype K
        M : Matrix n n K
        h✝ : Nonempty n
        p : Nat
        hp✝ : CharP K p
        this✝ : CharP K p := hp✝
        k : Nat
        kpos : LT.lt 0 k
        hp : Nat.Prime p
        hk : Eq (Fintype.card K) (HPow.hPow p k)
        this : Fact (Nat.Prime p)
        ⊢ Eq (matPolyEquiv ((Polynomial.expand K (HPow.hPow p k)).mapMatrix (HPow.hPow …
      -/
    · exact (id (matPolyEquiv_eq_X_pow_sub_C (p ^ k) M) : _)
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.intro.mk.intro.a.h
        n : Type u_1
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Fintype K
        M : Matrix n n K
        h✝ : Nonempty n
        p : Nat
        hp✝ : CharP K p
        this✝ : CharP K p := hp✝
        k : Nat
        kpos : LT.lt 0 k
        hp : Nat.Prime p
        hk : Eq (Fintype.card K) (HPow.hPow p k)
        this : Fact (Nat.Prime p)
        ⊢ Commute Polynomial.X (Polynomial.C M)
      -/
    · exact (C M).commute_X
      /-
        🎉 no goals
      -/
    /-
      case inr
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : IsEmpty n
      ⊢ Eq (HPow.hPow M (Fintype.card K)).charpoly M.charpoly
    -/
  · exact congr_arg _ (Subsingleton.elim _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem ZMod.charpoly_pow_card {p : ℕ} [Fact p.Prime] (M : Matrix n n (ZMod p)) :
    (M ^ p).charpoly = M.charpoly := by
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    M : Matrix n n (ZMod p)
    ⊢ Eq (HPow.hPow M p).charpoly M.charpoly
  -/
  have h := FiniteField.Matrix.charpoly_pow_card M
  /-
    n : Type u_1
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    M : Matrix n n (ZMod p)
    h : Eq (HPow.hPow M (Fintype.card (ZMod p))).charpoly M.charpoly
    ⊢ Eq (HPow.hPow M p).charpoly M.charpoly
  -/
  rwa [ZMod.card] at h
  /-
    🎉 no goals
  -/


theorem FiniteField.trace_pow_card {K : Type*} [Field K] [Fintype K] (M : Matrix n n K) :
    trace (M ^ Fintype.card K) = trace M ^ Fintype.card K := by
  /-
    n : Type u_1
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Fintype K
    M : Matrix n n K
    ⊢ Eq (HPow.hPow M (Fintype.card K)).trace (HPow.hPow M.trace (Fintype.card K))
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      n : Type u_1
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Fintype K
      M : Matrix n n K
      h✝ : IsEmpty n
      ⊢ Eq (HPow.hPow M (Fintype.card K)).trace (HPow.hPow M.trace (Fintype.card K))
    -/
  · simp [Matrix.trace]
    /-
      🎉 no goals
    -/
  rw [Matrix.trace_eq_neg_charpoly_coeff, Matrix.trace_eq_neg_charpoly_coeff,
    FiniteField.Matrix.charpoly_pow_card, FiniteField.pow_card]


theorem ZMod.trace_pow_card {p : ℕ} [Fact p.Prime] (M : Matrix n n (ZMod p)) :
                                      /-
                                        n : Type u_1
                                        inst✝² : DecidableEq n
                                        inst✝¹ : Fintype n
                                        p : Nat
                                        inst✝ : Fact (Nat.Prime p)
                                        M : Matrix n n (ZMod p)
                                        ⊢ Eq (HPow.hPow M p).trace (HPow.hPow M.trace p)
                                      -/
    trace (M ^ p) = trace M ^ p := by have h := FiniteField.trace_pow_card M; rwa [ZMod.card] at h
                                                                              /-
                                                                                🎉 no goals
                                                                              -/

