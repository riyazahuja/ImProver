/-- For any positive `k : ℕ` there exists an arbitrarily large prime `p` such that
`p ≡ 1 [MOD k]`. -/
theorem exists_prime_gt_modEq_one {k : ℕ} (n : ℕ) (hk0 : k ≠ 0) :
    ∃ p : ℕ, Nat.Prime p ∧ n < p ∧ p ≡ 1 [MOD k] := by
  /-
    k n : Nat
    hk0 : Ne k 0
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
  -/
  rcases (one_le_iff_ne_zero.2 hk0).eq_or_lt with (rfl | hk1)
    /-
      case inl
      n : Nat
      hk0 : Ne 1 0
      ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (Nat.ModEq 1 p 1))
    -/
  · rcases exists_infinite_primes (n + 1) with ⟨p, hnp, hp⟩
    /-
      case inl.intro.intro
      n : Nat
      hk0 : Ne 1 0
      p : Nat
      hnp : LE.le (HAdd.hAdd n 1) p
      hp : Nat.Prime p
      ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (Nat.ModEq 1 p 1))
    -/
    exact ⟨p, hp, hnp, modEq_one⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    k n : Nat
    hk0 : Ne k 0
    hk1 : LT.lt 1 k
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
  -/
  let b := k * (n !)
  have hgt : 1 < (eval (↑b) (cyclotomic k ℤ)).natAbs := by
    rcases le_iff_exists_add'.1 hk1.le with ⟨k, rfl⟩
    have hb : 2 ≤ b := le_mul_of_le_of_one_le hk1 n.factorial_pos
    calc
      1 ≤ b - 1 := le_tsub_of_add_le_left hb
      _ < (eval (b : ℤ) (cyclotomic (k + 1) ℤ)).natAbs :=
        sub_one_lt_natAbs_cyclotomic_eval hk1 (succ_le_iff.1 hb).ne'
  /-
    case inr
    k n : Nat
    hk0 : Ne k 0
    hk1 : LT.lt 1 k
    b : Nat := HMul.hMul k n.factorial
    hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
  -/
  let p := minFac (eval (↑b) (cyclotomic k ℤ)).natAbs
  /-
    case inr
    k n : Nat
    hk0 : Ne k 0
    hk1 : LT.lt 1 k
    b : Nat := HMul.hMul k n.factorial
    hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
    p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
  -/
  haveI hprime : Fact p.Prime := ⟨minFac_prime (ne_of_lt hgt).symm⟩
  have hroot : IsRoot (cyclotomic k (ZMod p)) (castRingHom (ZMod p) b) := by
    have : ((b : ℤ) : ZMod p) = ↑(Int.castRingHom (ZMod p) b) := by simp
    rw [IsRoot.def, ← map_cyclotomic_int k (ZMod p), eval_map, coe_castRingHom,
      ← Int.cast_natCast, this, eval₂_hom, Int.coe_castRingHom, ZMod.intCast_zmod_eq_zero_iff_dvd]
    apply Int.dvd_natAbs.1
    exact mod_cast minFac_dvd (eval (↑b) (cyclotomic k ℤ)).natAbs
  have hpb : ¬p ∣ b :=
    hprime.1.coprime_iff_not_dvd.1 (coprime_of_root_cyclotomic hk0.bot_lt hroot).symm
  /-
    case inr
    k n : Nat
    hk0 : Ne k 0
    hk1 : LT.lt 1 k
    b : Nat := HMul.hMul k n.factorial
    hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
    p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
    hprime : Fact (Nat.Prime p)
    hroot : (Polynomial.cyclotomic k (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) b)
    hpb : Not (Dvd.dvd p b)
    ⊢ Exists fun p => And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
  -/
  refine ⟨p, hprime.1, not_le.1 fun habs => ?_, ?_⟩
    /-
      case inr.refine_1
      k n : Nat
      hk0 : Ne k 0
      hk1 : LT.lt 1 k
      b : Nat := HMul.hMul k n.factorial
      hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
      p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
      hprime : Fact (Nat.Prime p)
      hroot : (Polynomial.cyclotomic k (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) b)
      hpb : Not (Dvd.dvd p b)
      habs : LE.le p n
      ⊢ False
    -/
  · exact hpb (dvd_mul_of_dvd_right (dvd_factorial (minFac_pos _) habs) _)
    /-
      🎉 no goals
    -/
  · have hdiv : orderOf (b : ZMod p) ∣ p - 1 :=
      ZMod.orderOf_dvd_card_sub_one (mt (CharP.cast_eq_zero_iff _ _ _).1 hpb)
    haveI : NeZero (k : ZMod p) :=
      NeZero.of_not_dvd (ZMod p) fun hpk => hpb (dvd_mul_of_dvd_left hpk _)
    /-
      case inr.refine_2
      k n : Nat
      hk0 : Ne k 0
      hk1 : LT.lt 1 k
      b : Nat := HMul.hMul k n.factorial
      hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
      p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
      hprime : Fact (Nat.Prime p)
      hroot : (Polynomial.cyclotomic k (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) b)
      hpb : Not (Dvd.dvd p b)
      hdiv : Dvd.dvd (orderOf ↑b) (HSub.hSub p 1)
      this : NeZero ↑k
      ⊢ k.ModEq p 1
    -/
    have : k = orderOf (b : ZMod p) := (isRoot_cyclotomic_iff.mp hroot).eq_orderOf
    /-
      case inr.refine_2
      k n : Nat
      hk0 : Ne k 0
      hk1 : LT.lt 1 k
      b : Nat := HMul.hMul k n.factorial
      hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
      p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
      hprime : Fact (Nat.Prime p)
      hroot : (Polynomial.cyclotomic k (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) b)
      hpb : Not (Dvd.dvd p b)
      hdiv : Dvd.dvd (orderOf ↑b) (HSub.hSub p 1)
      this✝ : NeZero ↑k
      this : Eq k (orderOf ↑b)
      ⊢ k.ModEq p 1
    -/
    rw [← this] at hdiv
    /-
      case inr.refine_2
      k n : Nat
      hk0 : Ne k 0
      hk1 : LT.lt 1 k
      b : Nat := HMul.hMul k n.factorial
      hgt : LT.lt 1 (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs
      p : Nat := (Polynomial.eval (↑b) (Polynomial.cyclotomic k Int)).natAbs.minFac
      hprime : Fact (Nat.Prime p)
      hroot : (Polynomial.cyclotomic k (ZMod p)).IsRoot ((Nat.castRingHom (ZMod p)) b)
      hpb : Not (Dvd.dvd p b)
      hdiv : Dvd.dvd k (HSub.hSub p 1)
      this✝ : NeZero ↑k
      this : Eq k (orderOf ↑b)
      ⊢ k.ModEq p 1
    -/
    exact ((modEq_iff_dvd' hprime.1.pos).2 hdiv).symm
    /-
      🎉 no goals
    -/


theorem frequently_atTop_modEq_one {k : ℕ} (hk0 : k ≠ 0) :
    ∃ᶠ p in atTop, Nat.Prime p ∧ p ≡ 1 [MOD k] := by
  /-
    k : Nat
    hk0 : Ne k 0
    ⊢ Filter.Frequently (fun p => And (Nat.Prime p) (k.ModEq p 1)) Filter.atTop
  -/
  refine frequently_atTop.2 fun n => ?_
  /-
    k : Nat
    hk0 : Ne k 0
    n : Nat
    ⊢ Exists fun b => And (GE.ge b n) (And (Nat.Prime b) (k.ModEq b 1))
  -/
  obtain ⟨p, hp⟩ := exists_prime_gt_modEq_one n hk0
  /-
    case intro
    k : Nat
    hk0 : Ne k 0
    n p : Nat
    hp : And (Nat.Prime p) (And (LT.lt n p) (k.ModEq p 1))
    ⊢ Exists fun b => And (GE.ge b n) (And (Nat.Prime b) (k.ModEq b 1))
  -/
  exact ⟨p, ⟨hp.2.1.le, hp.1, hp.2.2⟩⟩
  /-
    🎉 no goals
  -/


/-- For any positive `k : ℕ` there are infinitely many primes `p` such that `p ≡ 1 [MOD k]`. -/
theorem infinite_setOf_prime_modEq_one {k : ℕ} (hk0 : k ≠ 0) :
    Set.Infinite {p : ℕ | Nat.Prime p ∧ p ≡ 1 [MOD k]} :=
  frequently_atTop_iff_infinite.1 (frequently_atTop_modEq_one hk0)


