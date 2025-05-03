local notation "𝕎" => WittVector p


/-- The composition of Frobenius and Verschiebung is multiplication by `p`. -/
theorem frobenius_verschiebung (x : 𝕎 R) : frobenius (verschiebung x) = x * p := by
  have : IsPoly p fun {R} [CommRing R] x ↦ frobenius (verschiebung x) :=
    IsPoly.comp (hg := frobenius_isPoly p) (hf := verschiebung_isPoly)
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    this : WittVector.IsPoly p fun {R} [CommRing R] x => WittVector.frobenius (Wit …
    ⊢ Eq (WittVector.frobenius (WittVector.verschiebung x)) (HMul.hMul x ↑p)
  -/
  have : IsPoly p fun {R} [CommRing R] x ↦ x * p := mulN_isPoly p p
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    this✝ : WittVector.IsPoly p fun {R} [CommRing R] x => WittVector.frobenius (Wi …
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HMul.hMul x ↑p
    ⊢ Eq (WittVector.frobenius (WittVector.verschiebung x)) (HMul.hMul x ↑p)
  -/
  ghost_calc x
  /-
    case refine_3
    p : Nat
    hp : Fact (Nat.Prime p)
    this✝ : WittVector.IsPoly p fun {R} [CommRing R] x => WittVector.frobenius (Wi …
    this : WittVector.IsPoly p fun {R} [CommRing R] x => HMul.hMul x ↑p
    R : Type u_1
    R._inst : CommRing R
    x : WittVector p R
    ⊢ ∀ (n : Nat), Eq ((WittVector.ghostComponent n) (WittVector.frobenius (WittVe …
  -/
  ghost_simp [mul_comm]
  /-
    🎉 no goals
  -/


/-- Verschiebung is the same as multiplication by `p` on the ring of Witt vectors of `ZMod p`. -/
theorem verschiebung_zmod (x : 𝕎 (ZMod p)) : verschiebung x = x * p := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    x : WittVector p (ZMod p)
    ⊢ Eq (WittVector.verschiebung x) (HMul.hMul x ↑p)
  -/
  rw [← frobenius_verschiebung, frobenius_zmodp]
  /-
    🎉 no goals
  -/


theorem coeff_p_pow [CharP R p] (i : ℕ) : ((p : 𝕎 R) ^ i).coeff i = 1 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    ⊢ Eq ((HPow.hPow (↑p) i).coeff i) 1
  -/
  induction' i with i h
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      ⊢ Eq ((HPow.hPow (↑p) 0).coeff 0) 1
    -/
  · simp only [one_coeff_zero, Ne, pow_zero]
    /-
      🎉 no goals
    -/
  · rw [pow_succ, ← frobenius_verschiebung, coeff_frobenius_charP,
      verschiebung_coeff_succ, h, one_pow]


theorem coeff_p_pow_eq_zero [CharP R p] {i j : ℕ} (hj : j ≠ i) : ((p : 𝕎 R) ^ i).coeff j = 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i j : Nat
    hj : Ne j i
    ⊢ Eq ((HPow.hPow (↑p) i).coeff j) 0
  -/
  induction' i with i hi generalizing j
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      j : Nat
      hj : Ne j 0
      ⊢ Eq ((HPow.hPow (↑p) 0).coeff j) 0
    -/
  · rw [pow_zero, one_coeff_eq_of_pos]
    /-
      case zero.hn
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      j : Nat
      hj : Ne j 0
      ⊢ LT.lt 0 j
    -/
    exact Nat.pos_of_ne_zero hj
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      i : Nat
      hi : ∀ {j : Nat}, Ne j i → Eq ((HPow.hPow (↑p) i).coeff j) 0
      j : Nat
      hj : Ne j (HAdd.hAdd i 1)
      ⊢ Eq ((HPow.hPow (↑p) (HAdd.hAdd i 1)).coeff j) 0
    -/
  · rw [pow_succ, ← frobenius_verschiebung, coeff_frobenius_charP]
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      i : Nat
      hi : ∀ {j : Nat}, Ne j i → Eq ((HPow.hPow (↑p) i).coeff j) 0
      j : Nat
      hj : Ne j (HAdd.hAdd i 1)
      ⊢ Eq (HPow.hPow ((WittVector.verschiebung (HPow.hPow (↑p) i)).coeff j) p) 0
    -/
    cases j
      /-
        case succ.zero
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝¹ : CommRing R
        inst✝ : CharP R p
        i : Nat
        hi : ∀ {j : Nat}, Ne j i → Eq ((HPow.hPow (↑p) i).coeff j) 0
        hj : Ne 0 (HAdd.hAdd i 1)
        ⊢ Eq (HPow.hPow ((WittVector.verschiebung (HPow.hPow (↑p) i)).coeff 0) p) 0
      -/
    · rw [verschiebung_coeff_zero, zero_pow hp.out.ne_zero]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        p : Nat
        R : Type u_1
        hp : Fact (Nat.Prime p)
        inst✝¹ : CommRing R
        inst✝ : CharP R p
        i : Nat
        hi : ∀ {j : Nat}, Ne j i → Eq ((HPow.hPow (↑p) i).coeff j) 0
        n✝ : Nat
        hj : Ne (HAdd.hAdd n✝ 1) (HAdd.hAdd i 1)
        ⊢ Eq (HPow.hPow ((WittVector.verschiebung (HPow.hPow (↑p) i)).coeff (HAdd.hAdd …
      -/
    · rw [verschiebung_coeff_succ, hi (ne_of_apply_ne _ hj), zero_pow hp.out.ne_zero]
      /-
        🎉 no goals
      -/


theorem coeff_p [CharP R p] (i : ℕ) : (p : 𝕎 R).coeff i = if i = 1 then 1 else 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    i : Nat
    ⊢ Eq ((↑p).coeff i) (ite (Eq i 1) 1 0)
  -/
  split_ifs with hi
    /-
      case pos
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      i : Nat
      hi : Eq i 1
      ⊢ Eq ((↑p).coeff i) 1
    -/
  · simpa only [hi, pow_one] using coeff_p_pow p R 1
    /-
      🎉 no goals
    -/
    /-
      case neg
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      i : Nat
      hi : Not (Eq i 1)
      ⊢ Eq ((↑p).coeff i) 0
    -/
  · simpa only [pow_one] using coeff_p_pow_eq_zero p R hi
    /-
      🎉 no goals
    -/


@[simp]
theorem coeff_p_zero [CharP R p] : (p : 𝕎 R).coeff 0 = 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    ⊢ Eq ((↑p).coeff 0) 0
  -/
  rw [coeff_p, if_neg]
  /-
    case hnc
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    ⊢ Not (Eq 0 1)
  -/
  exact zero_ne_one
  /-
    🎉 no goals
  -/


@[simp]
                                                              /-
                                                                p : Nat
                                                                R : Type u_1
                                                                hp : Fact (Nat.Prime p)
                                                                inst✝¹ : CommRing R
                                                                inst✝ : CharP R p
                                                                ⊢ Eq ((↑p).coeff 1) 1
                                                              -/
theorem coeff_p_one [CharP R p] : (p : 𝕎 R).coeff 1 = 1 := by rw [coeff_p, if_pos rfl]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem p_nonzero [Nontrivial R] [CharP R p] : (p : 𝕎 R) ≠ 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : CharP R p
    ⊢ Ne (↑p) 0
  -/
  intro h
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : CharP R p
    h : Eq (↑p) 0
    ⊢ False
  -/
  simpa only [h, zero_coeff, zero_ne_one] using coeff_p_one p R
  /-
    🎉 no goals
  -/


theorem FractionRing.p_nonzero [Nontrivial R] [CharP R p] : (p : FractionRing (𝕎 R)) ≠ 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : CharP R p
    ⊢ Ne (↑p) 0
  -/
  simpa using (IsFractionRing.injective (𝕎 R) (FractionRing (𝕎 R))).ne (WittVector.p_nonzero _ _)
  /-
    🎉 no goals
  -/


/-- The “projection formula” for Frobenius and Verschiebung. -/
theorem verschiebung_mul_frobenius (x y : 𝕎 R) :
    verschiebung (x * frobenius y) = verschiebung x * y := by
  have : IsPoly₂ p fun {R} [Rcr : CommRing R] x y ↦ verschiebung (x * frobenius y) :=
    IsPoly.comp₂ (hg := verschiebung_isPoly)
      (hf := IsPoly₂.comp (hh := mulIsPoly₂) (hf := idIsPolyI' p) (hg := frobenius_isPoly p))
  have : IsPoly₂ p fun {R} [CommRing R] x y ↦ verschiebung x * y :=
    IsPoly₂.comp (hh := mulIsPoly₂) (hf := verschiebung_isPoly) (hg := idIsPolyI' p)
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    this✝ : WittVector.IsPoly₂ p fun {R} [CommRing R] x y => WittVector.verschiebu …
    this : WittVector.IsPoly₂ p fun {R} [CommRing R] x y => HMul.hMul (WittVector. …
    ⊢ Eq (WittVector.verschiebung (HMul.hMul x (WittVector.frobenius y))) (HMul.hM …
  -/
  ghost_calc x y
  /-
    case refine_3
    p : Nat
    hp : Fact (Nat.Prime p)
    this✝ : WittVector.IsPoly₂ p fun {R} [CommRing R] x y => WittVector.verschiebu …
    this : WittVector.IsPoly₂ p fun {R} [CommRing R] x y => HMul.hMul (WittVector. …
    R : Type u_1
    R._inst : CommRing R
    x y : WittVector p R
    ⊢ ∀ (n : Nat), Eq ((WittVector.ghostComponent n) (WittVector.verschiebung (HMu …
  -/
                /-
                  🎉 no goals
                -/
  rintro ⟨⟩ <;> ghost_simp [mul_assoc]
                /-
                  🎉 no goals
                -/


theorem mul_charP_coeff_zero [CharP R p] (x : 𝕎 R) : (x * p).coeff 0 = 0 := by
  rw [← frobenius_verschiebung, coeff_frobenius_charP, verschiebung_coeff_zero,
    zero_pow hp.out.ne_zero]


theorem mul_charP_coeff_succ [CharP R p] (x : 𝕎 R) (i : ℕ) :
    (x * p).coeff (i + 1) = x.coeff i ^ p := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    i : Nat
    ⊢ Eq ((HMul.hMul x ↑p).coeff (HAdd.hAdd i 1)) (HPow.hPow (x.coeff i) p)
  -/
  rw [← frobenius_verschiebung, coeff_frobenius_charP, verschiebung_coeff_succ]
  /-
    🎉 no goals
  -/


theorem verschiebung_frobenius [CharP R p] (x : 𝕎 R) : verschiebung (frobenius x) = x * p := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    ⊢ Eq (WittVector.verschiebung (WittVector.frobenius x)) (HMul.hMul x ↑p)
  -/
  ext ⟨i⟩
    /-
      case h.zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      ⊢ Eq ((WittVector.verschiebung (WittVector.frobenius x)).coeff 0) ((HMul.hMul  …
    -/
  · rw [mul_charP_coeff_zero, verschiebung_coeff_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      n✝ : Nat
      ⊢ Eq ((WittVector.verschiebung (WittVector.frobenius x)).coeff (HAdd.hAdd n✝ 1 …
    -/
  · rw [mul_charP_coeff_succ, verschiebung_coeff_succ, coeff_frobenius_charP]
    /-
      🎉 no goals
    -/


theorem verschiebung_frobenius_comm [CharP R p] :
    Function.Commute (verschiebung : 𝕎 R → 𝕎 R) frobenius := fun x => by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    ⊢ Eq (WittVector.verschiebung (WittVector.frobenius x)) (WittVector.frobenius  …
  -/
  rw [verschiebung_frobenius, frobenius_verschiebung]
  /-
    🎉 no goals
  -/


theorem iterate_verschiebung_coeff (x : 𝕎 R) (n k : ℕ) :
    (verschiebung^[n] x).coeff (k + n) = x.coeff k := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    n k : Nat
    ⊢ Eq ((Nat.iterate (⇑WittVector.verschiebung) n x).coeff (HAdd.hAdd k n)) (x.c …
  -/
  induction' n with k ih
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k : Nat
      ⊢ Eq ((Nat.iterate (⇑WittVector.verschiebung) 0 x).coeff (HAdd.hAdd k 0)) (x.c …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k✝ k : Nat
      ih : Eq ((Nat.iterate (⇑WittVector.verschiebung) k x).coeff (HAdd.hAdd k✝ k))  …
      ⊢ Eq ((Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd k 1) x).coeff (HAdd.h …
    -/
  · rw [iterate_succ_apply', Nat.add_succ, verschiebung_coeff_succ]
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      k✝ k : Nat
      ih : Eq ((Nat.iterate (⇑WittVector.verschiebung) k x).coeff (HAdd.hAdd k✝ k))  …
      ⊢ Eq ((Nat.iterate (⇑WittVector.verschiebung) k x).coeff (HAdd.hAdd k✝ k)) (x. …
    -/
    exact ih
    /-
      🎉 no goals
    -/


theorem iterate_verschiebung_mul_left (x y : 𝕎 R) (i : ℕ) :
    verschiebung^[i] x * y = verschiebung^[i] (x * frobenius^[i] y) := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    i : Nat
    ⊢ Eq (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) i x) y) (Nat.iterate ( …
  -/
  induction' i with i ih generalizing y
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x y : WittVector p R
      ⊢ Eq (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) 0 x) y) (Nat.iterate ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝ : CommRing R
      x : WittVector p R
      i : Nat
      ih : ∀ (y : WittVector p R), Eq (HMul.hMul (Nat.iterate (⇑WittVector.verschieb …
      y : WittVector p R
      ⊢ Eq (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd i 1) x) y)  …
    -/
  · rw [iterate_succ_apply', ← verschiebung_mul_frobenius, ih, iterate_succ_apply']; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem iterate_verschiebung_mul (x y : 𝕎 R) (i j : ℕ) :
    verschiebung^[i] x * verschiebung^[j] y =
      verschiebung^[i + j] (frobenius^[j] x * frobenius^[i] y) := by
  calc
    _ = verschiebung^[i] (x * frobenius^[i] (verschiebung^[j] y)) := ?_
    _ = verschiebung^[i] (x * verschiebung^[j] (frobenius^[i] y)) := ?_
    _ = verschiebung^[i] (verschiebung^[j] (frobenius^[i] y) * x) := ?_
    _ = verschiebung^[i] (verschiebung^[j] (frobenius^[i] y * frobenius^[j] x)) := ?_
    _ = verschiebung^[i + j] (frobenius^[i] y * frobenius^[j] x) := ?_
    _ = _ := ?_
    /-
      case calc_1
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) i x) (Nat.iterate (⇑Wi …
    -/
  · apply iterate_verschiebung_mul_left
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (Nat.iterate (⇑WittVector.verschiebung) i (HMul.hMul x (Nat.iterate (⇑Wit …
    -/
  · rw [verschiebung_frobenius_comm.iterate_iterate]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (Nat.iterate (⇑WittVector.verschiebung) i (HMul.hMul x (Nat.iterate (⇑Wit …
    -/
  · rw [mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case calc_4
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (Nat.iterate (⇑WittVector.verschiebung) i (HMul.hMul (Nat.iterate (⇑WittV …
    -/
  · rw [iterate_verschiebung_mul_left]
    /-
      🎉 no goals
    -/
    /-
      case calc_5
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (Nat.iterate (⇑WittVector.verschiebung) i (Nat.iterate (⇑WittVector.versc …
    -/
  · rw [iterate_add_apply]
    /-
      🎉 no goals
    -/
    /-
      case calc_6
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd i j) (HMul.hMul (Nat.i …
    -/
  · rw [mul_comm]
    /-
      🎉 no goals
    -/

-- Porting note: `ring_nf` doesn't handle powers yet; needed to add `Nat.pow_succ` rewrite

theorem iterate_frobenius_coeff (x : 𝕎 R) (i k : ℕ) :
    (frobenius^[i] x).coeff k = x.coeff k ^ p ^ i := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝¹ : CommRing R
    inst✝ : CharP R p
    x : WittVector p R
    i k : Nat
    ⊢ Eq ((Nat.iterate (⇑WittVector.frobenius) i x).coeff k) (HPow.hPow (x.coeff k …
  -/
  induction' i with i ih
    /-
      case zero
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      k : Nat
      ⊢ Eq ((Nat.iterate (⇑WittVector.frobenius) 0 x).coeff k) (HPow.hPow (x.coeff k …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      k i : Nat
      ih : Eq ((Nat.iterate (⇑WittVector.frobenius) i x).coeff k) (HPow.hPow (x.coef …
      ⊢ Eq ((Nat.iterate (⇑WittVector.frobenius) (HAdd.hAdd i 1) x).coeff k) (HPow.h …
    -/
  · rw [iterate_succ_apply', coeff_frobenius_charP, ih, Nat.pow_succ]
    /-
      case succ
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x : WittVector p R
      k i : Nat
      ih : Eq ((Nat.iterate (⇑WittVector.frobenius) i x).coeff k) (HPow.hPow (x.coef …
      ⊢ Eq (HPow.hPow (HPow.hPow (x.coeff k) (HPow.hPow p i)) p) (HPow.hPow (x.coeff …
    -/
    ring_nf
    /-
      🎉 no goals
    -/


/-- This is a slightly specialized form of [Hazewinkel, *Witt Vectors*][Haze09] 6.2 equation 5. -/
theorem iterate_verschiebung_mul_coeff (x y : 𝕎 R) (i j : ℕ) :
    (verschiebung^[i] x * verschiebung^[j] y).coeff (i + j) =
      x.coeff 0 ^ p ^ j * y.coeff 0 ^ p ^ i := by
  calc
    _ = (verschiebung^[i + j] (frobenius^[j] x * frobenius^[i] y)).coeff (i + j) := ?_
    _ = (frobenius^[j] x * frobenius^[i] y).coeff 0 := ?_
    _ = (frobenius^[j] x).coeff 0 * (frobenius^[i] y).coeff 0 := ?_
    _ = _ := ?_
    /-
      case calc_1
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq ((HMul.hMul (Nat.iterate (⇑WittVector.verschiebung) i x) (Nat.iterate (⇑W …
    -/
  · rw [iterate_verschiebung_mul]
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq ((Nat.iterate (⇑WittVector.verschiebung) (HAdd.hAdd i j) (HMul.hMul (Nat. …
    -/
  · convert iterate_verschiebung_coeff (p := p) (R := R) _ _ _ using 2
    /-
      case h.e'_2.h.e'_4
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (HAdd.hAdd i j) (HAdd.hAdd 0 (HAdd.hAdd i j))
    -/
    rw [zero_add]
    /-
      🎉 no goals
    -/
    /-
      case calc_3
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq ((HMul.hMul (Nat.iterate (⇑WittVector.frobenius) j x) (Nat.iterate (⇑Witt …
    -/
  · apply mul_coeff_zero
    /-
      🎉 no goals
    -/
    /-
      case calc_4
      p : Nat
      R : Type u_1
      hp : Fact (Nat.Prime p)
      inst✝¹ : CommRing R
      inst✝ : CharP R p
      x y : WittVector p R
      i j : Nat
      ⊢ Eq (HMul.hMul ((Nat.iterate (⇑WittVector.frobenius) j x).coeff 0) ((Nat.iter …
    -/
  · simp only [iterate_frobenius_coeff]
    /-
      🎉 no goals
    -/


