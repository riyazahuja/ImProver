private lemma cube_of_castHom_ne_zero {n : ZMod 9} :
                           /-
                             n : ZMod 9
                             ⊢ Dvd.dvd 3 9
                           -/
    castHom (show 3 ∣ 9 by norm_num) (ZMod 3) n ≠ 0 → n ^ 3 = 1 ∨ n ^ 3 = 8 := by
                           /-
                             🎉 no goals
                           -/
  /-
    n : ZMod 9
    ⊢ Ne ((ZMod.castHom ⋯ (ZMod 3)) n) 0 → Or (Eq (HPow.hPow n 3) 1) (Eq (HPow.hPo …
  -/
  revert n; decide
            /-
              🎉 no goals
            -/


private lemma cube_of_not_dvd {n : ℤ} (h : ¬ 3 ∣ n) :
    (n : ZMod 9) ^ 3 = 1 ∨ (n : ZMod 9) ^ 3 = 8 := by
  /-
    n : Int
    h : Not (Dvd.dvd 3 n)
    ⊢ Or (Eq (HPow.hPow (↑n) 3) 1) (Eq (HPow.hPow (↑n) 3) 8)
  -/
  apply cube_of_castHom_ne_zero
  /-
    case a
    n : Int
    h : Not (Dvd.dvd 3 n)
    ⊢ Ne ((ZMod.castHom ⋯ (ZMod 3)) ↑n) 0
  -/
  rwa [map_intCast, Ne, ZMod.intCast_zmod_eq_zero_iff_dvd]
  /-
    🎉 no goals
  -/


/-- If `a b c : ℤ` are such that `¬ 3 ∣ a * b * c`, then `a ^ 3 + b ^ 3 ≠ c ^ 3`. -/
theorem fermatLastTheoremThree_case_1 {a b c : ℤ} (hdvd : ¬ 3 ∣ a * b * c) :
    a ^ 3 + b ^ 3 ≠ c ^ 3 := by
  /-
    a b c : Int
    hdvd : Not (Dvd.dvd 3 (HMul.hMul (HMul.hMul a b) c))
    ⊢ Ne (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
  -/
  simp_rw [Int.prime_three.dvd_mul, not_or] at hdvd
  /-
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ⊢ Ne (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
  -/
  apply mt (congrArg (Int.cast : ℤ → ZMod 9))
  /-
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ⊢ Not (Eq ↑(HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) ↑(HPow.hPow c 3))
  -/
  simp_rw [Int.cast_add, Int.cast_pow]
  /-
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HPow.hPow (↑c) 3))
  -/
  rcases cube_of_not_dvd hdvd.1.1 with ha | ha <;>
  /-
    case inl
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ha : Eq (HPow.hPow (↑a) 3) 1
    ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HPow.hPow (↑c) 3))
  -/
  rcases cube_of_not_dvd hdvd.1.2 with hb | hb <;>
  /-
    case inl.inl
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ha : Eq (HPow.hPow (↑a) 3) 1
    hb : Eq (HPow.hPow (↑b) 3) 1
    ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HPow.hPow (↑c) 3))
  -/
  rcases cube_of_not_dvd hdvd.2 with hc | hc <;>
  /-
    case inl.inl.inl
    a b c : Int
    hdvd : And (And (Not (Dvd.dvd 3 a)) (Not (Dvd.dvd 3 b))) (Not (Dvd.dvd 3 c))
    ha : Eq (HPow.hPow (↑a) 3) 1
    hb : Eq (HPow.hPow (↑b) 3) 1
    hc : Eq (HPow.hPow (↑c) 3) 1
    ⊢ Not (Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HPow.hPow (↑c) 3))
  -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  rw [ha, hb, hc] <;> decide
                      /-
                        🎉 no goals
                      -/


private lemma three_dvd_b_of_dvd_a_of_gcd_eq_one_of_case2 {a b c : ℤ} (ha : a ≠ 0)
    (Hgcd : Finset.gcd {a, b, c} id = 1) (h3a : 3 ∣ a) (HF : a ^ 3 + b ^ 3 + c ^ 3 = 0)
    (H : ∀ a b c : ℤ, c ≠ 0 → ¬ 3 ∣ a → ¬ 3 ∣ b  → 3 ∣ c → IsCoprime a b → a ^ 3 + b ^ 3 ≠ c ^ 3) :
    3 ∣ b := by
  have hbc : IsCoprime (-b) (-c) := by
    refine IsCoprime.neg_neg ?_
    rw [add_comm (a ^ 3), add_assoc, add_comm (a ^ 3), ← add_assoc] at HF
    refine isCoprime_of_gcd_eq_one_of_FLT ?_ HF
    convert Hgcd using 2
    rw [Finset.pair_comm, Finset.insert_comm]
  /-
    a b c : Int
    ha : Ne a 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    h3a : Dvd.dvd 3 a
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    hbc : IsCoprime (Neg.neg b) (Neg.neg c)
    ⊢ Dvd.dvd 3 b
  -/
  by_contra! h3b
  /-
    a b c : Int
    ha : Ne a 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    h3a : Dvd.dvd 3 a
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    hbc : IsCoprime (Neg.neg b) (Neg.neg c)
    h3b : Not (Dvd.dvd 3 b)
    ⊢ False
  -/
  by_cases h3c : 3 ∣ c
    /-
      case pos
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Dvd.dvd 3 c
      ⊢ False
    -/
  · apply h3b
    /-
      case pos
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Dvd.dvd 3 c
      ⊢ Dvd.dvd 3 b
    -/
    rw [add_assoc, add_comm (b ^ 3), ← add_assoc] at HF
    /-
      case pos
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow c 3)) (HPow.hPow b 3) …
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Dvd.dvd 3 c
      ⊢ Dvd.dvd 3 b
    -/
    exact dvd_c_of_prime_of_dvd_a_of_dvd_b_of_FLT Int.prime_three h3a h3c HF
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Not (Dvd.dvd 3 c)
      ⊢ False
    -/
  · refine H (-b) (-c) a ha (by simp [h3b]) (by simp [h3c]) h3a hbc ?_
    /-
      case neg
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Not (Dvd.dvd 3 c)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Neg.neg b) 3) (HPow.hPow (Neg.neg c) 3)) (HPow.hPo …
    -/
    rw [add_eq_zero_iff_eq_neg, ← (show Odd 3 by decide).neg_pow] at HF
    /-
      case neg
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg.neg c) 3)
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Not (Dvd.dvd 3 c)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Neg.neg b) 3) (HPow.hPow (Neg.neg c) 3)) (HPow.hPo …
    -/
    rw [← HF]
    /-
      case neg
      a b c : Int
      ha : Ne a 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      h3a : Dvd.dvd 3 a
      HF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg.neg c) 3)
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      hbc : IsCoprime (Neg.neg b) (Neg.neg c)
      h3b : Not (Dvd.dvd 3 b)
      h3c : Not (Dvd.dvd 3 c)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (Neg.neg b) 3) (HAdd.hAdd (HPow.hPow a 3) (HPow.hPo …
    -/
    ring
    /-
      🎉 no goals
    -/


open Finset in
private lemma fermatLastTheoremThree_of_dvd_a_of_gcd_eq_one_of_case2 {a b c : ℤ} (ha : a ≠ 0)
    (h3a : 3 ∣ a) (Hgcd : Finset.gcd {a, b, c} id = 1)
    (H : ∀ a b c : ℤ, c ≠ 0 → ¬ 3 ∣ a → ¬ 3 ∣ b  → 3 ∣ c → IsCoprime a b → a ^ 3 + b ^ 3 ≠ c ^ 3) :
    a ^ 3 + b ^ 3 + c ^ 3 ≠ 0 := by
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    ⊢ Ne (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)) 0
  -/
  intro HF
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    ⊢ False
  -/
  apply (show ¬(3 ∣ (1 : ℤ)) by decide)
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    ⊢ Dvd.dvd 3 1
  -/
  rw [← Hgcd]
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    ⊢ Dvd.dvd 3 ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id)
  -/
  refine dvd_gcd (fun x hx ↦ ?_)
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    x : Int
    hx : Membership.mem (Insert.insert a (Insert.insert b (Singleton.singleton c)) …
    ⊢ Dvd.dvd 3 (id x)
  -/
  simp only [mem_insert, mem_singleton] at hx
  have h3b : 3 ∣ b := by
    refine three_dvd_b_of_dvd_a_of_gcd_eq_one_of_case2 ha ?_ h3a HF H
    simp only [← Hgcd, gcd_insert, gcd_singleton, id_eq, ← Int.abs_eq_normalize, abs_neg]
  /-
    a b c : Int
    ha : Ne a 0
    h3a : Dvd.dvd 3 a
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
    x : Int
    hx : Or (Eq x a) (Or (Eq x b) (Eq x c))
    h3b : Dvd.dvd 3 b
    ⊢ Dvd.dvd 3 (id x)
  -/
  rcases hx with hx | hx | hx
    /-
      case inl
      a b c : Int
      ha : Ne a 0
      h3a : Dvd.dvd 3 a
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      x : Int
      h3b : Dvd.dvd 3 b
      hx : Eq x a
      ⊢ Dvd.dvd 3 (id x)
    -/
  · exact hx ▸ h3a
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      a b c : Int
      ha : Ne a 0
      h3a : Dvd.dvd 3 a
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      x : Int
      h3b : Dvd.dvd 3 b
      hx : Eq x b
      ⊢ Dvd.dvd 3 (id x)
    -/
  · exact hx ▸ h3b
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b c : Int
      ha : Ne a 0
      h3a : Dvd.dvd 3 a
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      HF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3) …
      x : Int
      h3b : Dvd.dvd 3 b
      hx : Eq x c
      ⊢ Dvd.dvd 3 (id x)
    -/
  · simpa [hx] using dvd_c_of_prime_of_dvd_a_of_dvd_b_of_FLT Int.prime_three h3a h3b HF
    /-
      🎉 no goals
    -/


open Finset Int in
/--
  To prove Fermat's Last Theorem for `n = 3`, it is enough to show that for all `a`, `b`, `c`
  in `ℤ` such that `c ≠ 0`, `¬ 3 ∣ a`, `¬ 3 ∣ b`, `a` and `b` are coprime and `3 ∣ c`, we have
  `a ^ 3 + b ^ 3 ≠ c ^ 3`.
-/
theorem fermatLastTheoremThree_of_three_dvd_only_c
    (H : ∀ a b c : ℤ, c ≠ 0 → ¬ 3 ∣ a → ¬ 3 ∣ b  → 3 ∣ c → IsCoprime a b → a ^ 3 + b ^ 3 ≠ c ^ 3) :
    FermatLastTheoremFor 3 := by
  /-
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    ⊢ FermatLastTheoremFor 3
  -/
  rw [fermatLastTheoremFor_iff_int]
  /-
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    ⊢ FermatLastTheoremWith Int 3
  -/
  refine fermatLastTheoremWith_of_fermatLastTheoremWith_coprime (fun a b c ha hb hc Hgcd hF ↦?_)
  /-
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    hF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
    ⊢ False
  -/
  by_cases h1 : 3 ∣ a * b * c
  /-
    case pos
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    hF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
    h1 : Dvd.dvd 3 (HMul.hMul (HMul.hMul a b) c)
    ⊢ False
  -/
  swap
    /-
      case neg
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      h1 : Not (Dvd.dvd 3 (HMul.hMul (HMul.hMul a b) c))
      ⊢ False
    -/
  · exact fermatLastTheoremThree_case_1 h1 hF
    /-
      🎉 no goals
    -/
  /-
    case pos
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    hF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
    h1 : Dvd.dvd 3 (HMul.hMul (HMul.hMul a b) c)
    ⊢ False
  -/
  rw [(prime_three).dvd_mul, (prime_three).dvd_mul] at h1
  /-
    case pos
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    hF : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
    h1 : Or (Or (Dvd.dvd 3 a) (Dvd.dvd 3 b)) (Dvd.dvd 3 c)
    ⊢ False
  -/
  rw [← sub_eq_zero, sub_eq_add_neg, ← (show Odd 3 by decide).neg_pow] at hF
  /-
    case pos
    H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
    a b c : Int
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
    hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg …
    h1 : Or (Or (Dvd.dvd 3 a) (Dvd.dvd 3 b)) (Dvd.dvd 3 c)
    ⊢ False
  -/
  rcases h1 with (h3a | h3b) | h3c
    /-
      case pos.inl.inl
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg …
      h3a : Dvd.dvd 3 a
      ⊢ False
    -/
  · refine fermatLastTheoremThree_of_dvd_a_of_gcd_eq_one_of_case2 ha h3a ?_ H hF
    /-
      case pos.inl.inl
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg …
      h3a : Dvd.dvd 3 a
      ⊢ Eq ((Insert.insert a (Insert.insert b (Singleton.singleton (Neg.neg c)))).gc …
    -/
    simp only [← Hgcd, insert_comm, gcd_insert, gcd_singleton, id_eq, ← abs_eq_normalize, abs_neg]
    /-
      🎉 no goals
    -/
    /-
      case pos.inl.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg …
      h3b : Dvd.dvd 3 b
      ⊢ False
    -/
  · rw [add_comm (a ^ 3)] at hF
    /-
      case pos.inl.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow b 3) (HPow.hPow a 3)) (HPow.hPow (Neg …
      h3b : Dvd.dvd 3 b
      ⊢ False
    -/
    refine fermatLastTheoremThree_of_dvd_a_of_gcd_eq_one_of_case2 hb h3b ?_ H hF
    /-
      case pos.inl.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow b 3) (HPow.hPow a 3)) (HPow.hPow (Neg …
      h3b : Dvd.dvd 3 b
      ⊢ Eq ((Insert.insert b (Insert.insert a (Singleton.singleton (Neg.neg c)))).gc …
    -/
    simp only [← Hgcd, insert_comm, gcd_insert, gcd_singleton, id_eq, ← abs_eq_normalize, abs_neg]
    /-
      🎉 no goals
    -/
    /-
      case pos.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow (Neg …
      h3c : Dvd.dvd 3 c
      ⊢ False
    -/
  · rw [add_comm _ ((-c) ^ 3), ← add_assoc] at hF
    refine fermatLastTheoremThree_of_dvd_a_of_gcd_eq_one_of_case2 (neg_ne_zero.2 hc) (by simp [h3c])
      ?_ H hF
    /-
      case pos.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (Neg.neg c) 3) (HPow.hPow a 3)) (HPow …
      h3c : Dvd.dvd 3 c
      ⊢ Eq ((Insert.insert (Neg.neg c) (Insert.insert a (Singleton.singleton b))).gc …
    -/
    rw [Finset.insert_comm (-c), Finset.pair_comm (-c) b]
    /-
      case pos.inr
      H : ∀ (a b c : Int), Ne c 0 → Not (Dvd.dvd 3 a) → Not (Dvd.dvd 3 b) → Dvd.dvd  …
      a b c : Int
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      Hgcd : Eq ((Insert.insert a (Insert.insert b (Singleton.singleton c))).gcd id) 1
      hF : Eq (HAdd.hAdd (HAdd.hAdd (HPow.hPow (Neg.neg c) 3) (HPow.hPow a 3)) (HPow …
      h3c : Dvd.dvd 3 c
      ⊢ Eq ((Insert.insert a (Insert.insert b (Singleton.singleton (Neg.neg c)))).gc …
    -/
    simp only [← Hgcd, insert_comm, gcd_insert, gcd_singleton, id_eq, ← abs_eq_normalize, abs_neg]
    /-
      🎉 no goals
    -/


                                                                                  /-
                                                                                    K : Type u_1
                                                                                    inst✝ : Field K
                                                                                    ζ : K
                                                                                    hζ : IsPrimitiveRoot ζ ↑3
                                                                                    ⊢ LT.lt 0 ↑3
                                                                                  -/
local notation3 "η" => (IsPrimitiveRoot.isUnit (hζ.toInteger_isPrimitiveRoot) (by decide)).unit
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/

local notation3 "λ" => hζ.toInteger - 1


/-- `FermatLastTheoremForThreeGen` is the statement that `a ^ 3 + b ^ 3 = u * c ^ 3` has no
nontrivial solutions in `𝓞 K` for all `u : (𝓞 K)ˣ` such that `¬ λ ∣ a`, `¬ λ ∣ b` and `λ ∣ c`.
The reason to consider `FermatLastTheoremForThreeGen` is to make a descent argument working. -/
def FermatLastTheoremForThreeGen : Prop :=
  ∀ a b c : 𝓞 K, ∀ u : (𝓞 K)ˣ, c ≠ 0 → ¬ λ ∣ a → ¬ λ ∣ b  → λ ∣ c → IsCoprime a b →
    a ^ 3 + b ^ 3 ≠ u * c ^ 3


/-- To prove `FermatLastTheoremFor 3`, it is enough to prove `FermatLastTheoremForThreeGen`. -/
lemma FermatLastTheoremForThree_of_FermatLastTheoremThreeGen
    [NumberField K] [IsCyclotomicExtension {3} ℚ K] :
    FermatLastTheoremForThreeGen hζ → FermatLastTheoremFor 3 := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ FermatLastTheoremForThreeGen hζ → FermatLastTheoremFor 3
  -/
  intro H
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    H : FermatLastTheoremForThreeGen hζ
    ⊢ FermatLastTheoremFor 3
  -/
  refine fermatLastTheoremThree_of_three_dvd_only_c (fun a b c hc ha hb ⟨x, hx⟩ hcoprime h ↦ ?_)
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    H : FermatLastTheoremForThreeGen hζ
    a b c : Int
    hc : Ne c 0
    ha : Not (Dvd.dvd 3 a)
    hb : Not (Dvd.dvd 3 b)
    x✝ : Dvd.dvd 3 c
    hcoprime : IsCoprime a b
    h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
    x : Int
    hx : Eq c (HMul.hMul 3 x)
    ⊢ False
  -/
  refine H a b c 1 (by simp [hc]) (fun hdvd ↦ ha ?_) (fun hdvd ↦ hb ?_) ?_ ?_ ?_
  · rwa [← Ideal.norm_dvd_iff (hζ.prime_norm_toInteger_sub_one_of_prime_ne_two' (by decide)),
      hζ.norm_toInteger_sub_one_of_prime_ne_two' (by decide)] at hdvd
  · rwa [← Ideal.norm_dvd_iff (hζ.prime_norm_toInteger_sub_one_of_prime_ne_two' (by decide)),
      hζ.norm_toInteger_sub_one_of_prime_ne_two' (by decide)] at hdvd
    /-
      case refine_3
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      H : FermatLastTheoremForThreeGen hζ
      a b c : Int
      hc : Ne c 0
      ha : Not (Dvd.dvd 3 a)
      hb : Not (Dvd.dvd 3 b)
      x✝ : Dvd.dvd 3 c
      hcoprime : IsCoprime a b
      h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      x : Int
      hx : Eq c (HMul.hMul 3 x)
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) ↑c
    -/
  · exact dvd_trans hζ.toInteger_sub_one_dvd_prime' ⟨x, by simp [hx]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      H : FermatLastTheoremForThreeGen hζ
      a b c : Int
      hc : Ne c 0
      ha : Not (Dvd.dvd 3 a)
      hb : Not (Dvd.dvd 3 b)
      x✝ : Dvd.dvd 3 c
      hcoprime : IsCoprime a b
      h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      x : Int
      hx : Eq c (HMul.hMul 3 x)
      ⊢ IsCoprime ↑a ↑b
    -/
  · rw [show a = algebraMap _ (𝓞 K) a by simp, show b = algebraMap _ (𝓞 K) b by simp]
    /-
      case refine_4
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      H : FermatLastTheoremForThreeGen hζ
      a b c : Int
      hc : Ne c 0
      ha : Not (Dvd.dvd 3 a)
      hb : Not (Dvd.dvd 3 b)
      x✝ : Dvd.dvd 3 c
      hcoprime : IsCoprime a b
      h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      x : Int
      hx : Eq c (HMul.hMul 3 x)
      ⊢ IsCoprime ((algebraMap Int (NumberField.RingOfIntegers K)) a) ((algebraMap I …
    -/
    exact hcoprime.map _
    /-
      🎉 no goals
    -/
    /-
      case refine_5
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      H : FermatLastTheoremForThreeGen hζ
      a b c : Int
      hc : Ne c 0
      ha : Not (Dvd.dvd 3 a)
      hb : Not (Dvd.dvd 3 b)
      x✝ : Dvd.dvd 3 c
      hcoprime : IsCoprime a b
      h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      x : Int
      hx : Eq c (HMul.hMul 3 x)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HMul.hMul (↑1) (HPow.h …
    -/
  · simp only [Units.val_one, one_mul]
    /-
      case refine_5
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      H : FermatLastTheoremForThreeGen hζ
      a b c : Int
      hc : Ne c 0
      ha : Not (Dvd.dvd 3 a)
      hb : Not (Dvd.dvd 3 b)
      x✝ : Dvd.dvd 3 c
      hcoprime : IsCoprime a b
      h : Eq (HAdd.hAdd (HPow.hPow a 3) (HPow.hPow b 3)) (HPow.hPow c 3)
      x : Int
      hx : Eq c (HMul.hMul 3 x)
      ⊢ Eq (HAdd.hAdd (HPow.hPow (↑a) 3) (HPow.hPow (↑b) 3)) (HPow.hPow (↑c) 3)
    -/
    exact_mod_cast h
    /-
      🎉 no goals
    -/


/-- `Solution'` is a tuple given by a solution to `a ^ 3 + b ^ 3 = u * c ^ 3`,
where `a`, `b`, `c` and `u` are as in `FermatLastTheoremForThreeGen`.
See `Solution` for the actual structure on which we will do the descent. -/
structure Solution' where
  a : 𝓞 K
  b : 𝓞 K
  c : 𝓞 K
  u : (𝓞 K)ˣ
  ha : ¬ λ ∣ a
  hb : ¬ λ ∣ b
  hc : c ≠ 0
  coprime : IsCoprime a b
  hcdvd : λ ∣ c
  H : a ^ 3 + b ^ 3 = u * c ^ 3

/-- `Solution` is the same as `Solution'` with the additional assumption that `λ ^ 2 ∣ a + b`. -/
structure Solution extends Solution' hζ where
  hab : λ ^ 2 ∣ a + b


/-- For any `S' : Solution'`, the multiplicity of `λ` in `S'.c` is finite. -/
lemma Solution'.multiplicity_lambda_c_finite :
    FiniteMultiplicity (hζ.toInteger - 1) S'.c :=
  .of_not_isUnit hζ.zeta_sub_one_prime'.not_unit S'.hc


/-- Given `S' : Solution'`, `S'.multiplicity` is the multiplicity of `λ` in `S'.c`, as a natural
number. -/
noncomputable def Solution'.multiplicity :=
  _root_.multiplicity (hζ.toInteger - 1) S'.c


/-- Given `S : Solution`, `S.multiplicity` is the multiplicity of `λ` in `S.c`, as a natural
number. -/
noncomputable def Solution.multiplicity := S.toSolution'.multiplicity


/-- We say that `S : Solution` is minimal if for all `S₁ : Solution`, the multiplicity of `λ` in
`S.c` is less or equal than the multiplicity in `S₁.c`. -/
def Solution.isMinimal : Prop := ∀ (S₁ : Solution hζ), S.multiplicity ≤ S₁.multiplicity


omit [NumberField K] [IsCyclotomicExtension {3} ℚ K] in
include S in
/-- If there is a solution then there is a minimal one. -/
lemma Solution.exists_minimal : ∃ (S₁ : Solution hζ), S₁.isMinimal := by
  classical
  let T := {n | ∃ (S' : Solution hζ), S'.multiplicity = n}
  rcases Nat.find_spec (⟨S.multiplicity, ⟨S, rfl⟩⟩ : T.Nonempty) with ⟨S₁, hS₁⟩
  exact ⟨S₁, fun S'' ↦ hS₁ ▸ Nat.find_min' _ ⟨S'', rfl⟩⟩


/-- Given `S' : Solution'`, then `S'.a` and `S'.b` are both congruent to `1` modulo `λ ^ 4` or are
both congruent to `-1`. -/
lemma a_cube_b_cube_congr_one_or_neg_one :
    λ ^ 4 ∣ S'.a ^ 3 - 1 ∧ λ ^ 4 ∣ S'.b ^ 3 + 1 ∨ λ ^ 4 ∣ S'.a ^ 3 + 1 ∧ λ ^ 4 ∣ S'.b ^ 3 - 1 := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Or (And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.h …
  -/
  obtain ⟨z, hz⟩ := S'.hcdvd
  rcases lambda_pow_four_dvd_cube_sub_one_or_add_one_of_lambda_not_dvd hζ S'.ha with
    ⟨x, hx⟩ | ⟨x, hx⟩ <;>
  rcases lambda_pow_four_dvd_cube_sub_one_or_add_one_of_lambda_not_dvd hζ S'.hb with
    ⟨y, hy⟩ | ⟨y, hy⟩
    /-
      case intro.inl.intro.inl.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Or (And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.h …
    -/
  · exfalso
    /-
      case intro.inl.intro.inl.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ False
    -/
    replace hζ : IsPrimitiveRoot ζ ((3 : ℕ+) ^ 1) := by rwa [pow_one]
    /-
      case intro.inl.intro.inl.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ✝ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ✝
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ✝.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      hζ : IsPrimitiveRoot ζ (HPow.hPow (↑3) 1)
      ⊢ False
    -/
    refine hζ.toInteger_sub_one_not_dvd_two (by decide) ⟨S'.u * λ ^ 2 * z ^ 3 - λ ^ 3 * (x + y), ?_⟩
    /-
      case intro.inl.intro.inl.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ✝ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ✝
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ✝.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      hζ : IsPrimitiveRoot ζ (HPow.hPow (↑3) 1)
      ⊢ Eq 2 (HMul.hMul (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul (HMul.hMul  …
    -/
    symm
    calc _ = S'.u * (λ * z) ^ 3 - λ ^ 4 * x - λ ^ 4 * y := by ring
    _ = (S'.a ^ 3 + S'.b ^ 3) - (S'.a ^ 3 - 1) - (S'.b ^ 3 - 1) := by rw [← hx, ← hy, ← hz, ← S'.H]
    _ = 2 := by ring
    /-
      case intro.inl.intro.inr.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Or (And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.h …
    -/
  · left
    /-
      case intro.inl.intro.inr.intro.h
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.hPow  …
    -/
    exact ⟨⟨x, hx⟩, ⟨y, hy⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.intro.inl.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Or (And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.h …
    -/
  · right
    /-
      case intro.inr.intro.inl.intro.h
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HAdd.hAdd (HPow.hPow  …
    -/
    exact ⟨⟨x, hx⟩, ⟨y, hy⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.intro.inr.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Or (And (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.h …
    -/
  · exfalso
    /-
      case intro.inr.intro.inr.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ False
    -/
    replace hζ : IsPrimitiveRoot ζ ((3 : ℕ+) ^ 1) := by rwa [pow_one]
    /-
      case intro.inr.intro.inr.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ✝ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ✝
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ✝.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      hζ : IsPrimitiveRoot ζ (HPow.hPow (↑3) 1)
      ⊢ False
    -/
    refine hζ.toInteger_sub_one_not_dvd_two (by decide) ⟨λ ^ 3 * (x + y) - S'.u * λ ^ 2 * z ^ 3, ?_⟩
    /-
      case intro.inr.intro.inr.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ✝ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ✝
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      z : NumberField.RingOfIntegers K
      hz : Eq S'.c (HMul.hMul (HSub.hSub hζ✝.toInteger 1) z)
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ✝. …
      hζ : IsPrimitiveRoot ζ (HPow.hPow (↑3) 1)
      ⊢ Eq 2 (HMul.hMul (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul (HPow.hPow  …
    -/
    symm
    calc _ =  λ ^ 4 * x + λ ^ 4 * y - S'.u * (λ * z) ^ 3 := by ring
    _ = (S'.a ^ 3 + 1) + (S'.b ^ 3 + 1) - (S'.a ^ 3 + S'.b ^ 3) := by rw [← hx, ← hy, ← hz, ← S'.H]
    _ = 2 := by ring


/-- Given `S' : Solution'`, we have that `λ ^ 4` divides `S'.c ^ 3`. -/
lemma lambda_pow_four_dvd_c_cube : λ ^ 4 ∣ S'.c ^ 3 := by
  rcases a_cube_b_cube_congr_one_or_neg_one S' with
    ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ | ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ <;>
    /-
      case inl.intro.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HPow.hPow S'.c 3)
    -/
    /-
      case inl.intro.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      x : NumberField.RingOfIntegers K
      hx : Eq (HSub.hSub (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HAdd.hAdd (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Eq (HPow.hPow S'.c 3) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (H …
    -/
    /-
      case inr.intro.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      x : NumberField.RingOfIntegers K
      hx : Eq (HAdd.hAdd (HPow.hPow S'.a 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      y : NumberField.RingOfIntegers K
      hy : Eq (HSub.hSub (HPow.hPow S'.b 3) 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.t …
      ⊢ Eq (HPow.hPow S'.c 3) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (H …
    -/
    symm
    calc _ = S'.u⁻¹ * (λ ^ 4 * x + λ ^ 4 * y) := by ring
    _ = S'.u⁻¹ * (S'.a ^ 3 + S'.b ^ 3) := by rw [← hx, ← hy]; ring
    _ = S'.u⁻¹ * (S'.u * S'.c ^ 3) := by rw [S'.H]
    _ = S'.c ^ 3 := by simp


/-- Given `S' : Solution'`, we have that `λ ^ 2` divides `S'.c`. -/
lemma lambda_sq_dvd_c : λ ^ 2 ∣ S'.c := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) S'.c
  -/
  have hm := S'.multiplicity_lambda_c_finite
  suffices 2 ≤ multiplicity (hζ.toInteger - 1) S'.c by
    obtain ⟨x, hx⟩ := pow_multiplicity_dvd (hζ.toInteger - 1) S'.c
    refine ⟨λ ^ (multiplicity (hζ.toInteger - 1) S'.c - 2) * x, ?_⟩
    rw [← mul_assoc, ← pow_add]
    convert hx using 3
    simp [this]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    hm : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) S'.c
    ⊢ LE.le 2 (multiplicity (HSub.hSub hζ.toInteger 1) S'.c)
  -/
  have := lambda_pow_four_dvd_c_cube S'
  rw [pow_dvd_iff_le_emultiplicity, emultiplicity_pow hζ.zeta_sub_one_prime',
    hm.emultiplicity_eq_multiplicity] at this
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    hm : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) S'.c
    this : LE.le (↑4) (HMul.hMul ↑3 ↑(multiplicity (HSub.hSub hζ.toInteger 1) S'.c))
    ⊢ LE.le 2 (multiplicity (HSub.hSub hζ.toInteger 1) S'.c)
  -/
  norm_cast at this
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    hm : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) S'.c
    this : LE.le 4 (HMul.hMul 3 (multiplicity (HSub.hSub hζ.toInteger 1) S'.c))
    ⊢ LE.le 2 (multiplicity (HSub.hSub hζ.toInteger 1) S'.c)
  -/
  omega
  /-
    🎉 no goals
  -/


/-- Given `S' : Solution'`, we have that `2 ≤ S'.multiplicity`. -/
lemma Solution'.two_le_multiplicity : 2 ≤ S'.multiplicity := by
  simpa [Solution'.multiplicity] using
    S'.multiplicity_lambda_c_finite.le_multiplicity_of_pow_dvd (lambda_sq_dvd_c S')


/-- Given `S : Solution`, we have that `2 ≤ S.multiplicity`. -/
lemma Solution.two_le_multiplicity : 2 ≤ S.multiplicity :=
  S.toSolution'.two_le_multiplicity


/-- Given `S' : Solution'`, the key factorization of `S'.a ^ 3 + S'.b ^ 3`. -/
lemma a_cube_add_b_cube_eq_mul :
    S'.a ^ 3 + S'.b ^ 3 = (S'.a + S'.b) * (S'.a + η * S'.b) * (S'.a + η ^ 2 * S'.b) := by
  /-
    K : Type u_1
    inst✝ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    ⊢ Eq (HAdd.hAdd (HPow.hPow S'.a 3) (HPow.hPow S'.b 3)) (HMul.hMul (HMul.hMul ( …
  -/
  symm
  calc _ = S'.a^3+S'.a^2*S'.b*(η^2+η+1)+S'.a*S'.b^2*(η^2+η+η^3)+η^3*S'.b^3 := by ring
  _ = S'.a^3+S'.a^2*S'.b*(η^2+η+1)+S'.a*S'.b^2*(η^2+η+1)+S'.b^3 := by
    simp [hζ.toInteger_cube_eq_one]
  _ = S'.a ^ 3 + S'.b ^ 3 := by rw [eta_sq]; ring


/-- Given `S' : Solution'`, we have that `λ ^ 2` divides one amongst `S'.a + S'.b`,
`S'.a + η * S'.b` and `S'.a + η ^ 2 * S'.b`. -/
lemma lambda_sq_dvd_or_dvd_or_dvd :
    λ ^ 2 ∣ S'.a + S'.b ∨ λ ^ 2 ∣ S'.a + η * S'.b ∨ λ ^ 2 ∣ S'.a + η ^ 2 * S'.b := by
  /-
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    ⊢ Or (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a S'.b))  …
  -/
  by_contra! h
  /-
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h : And (Not (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a …
    ⊢ False
  -/
  rcases h with ⟨h1, h2, h3⟩
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : Not (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a S'. …
    h2 : Not (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HM …
    h3 : Not (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HM …
    ⊢ False
  -/
  rw [← emultiplicity_lt_iff_not_dvd] at h1 h2 h3
  have h1' : FiniteMultiplicity (hζ.toInteger - 1) (S'.a + S'.b) :=
    finiteMultiplicity_iff_emultiplicity_ne_top.2 (fun ht ↦ by simp [ht] at h1)
  have h2' : FiniteMultiplicity (hζ.toInteger - 1) (S'.a + η * S'.b) := by
    refine finiteMultiplicity_iff_emultiplicity_ne_top.2 (fun ht ↦ ?_)
    rw [coe_eta] at ht
    simp [ht] at h2
  have h3' : FiniteMultiplicity (hζ.toInteger - 1) (S'.a + η ^ 2 * S'.b) := by
    refine finiteMultiplicity_iff_emultiplicity_ne_top.2 (fun ht ↦ ?_)
    rw [coe_eta] at ht
    simp [ht] at h3
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)) ↑2
    h2 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMu …
    h3 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMu …
    h1' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)
    h2' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    ⊢ False
  -/
  rw [h1'.emultiplicity_eq_multiplicity, Nat.cast_lt] at h1
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)) 2
    h2 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMu …
    h3 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMu …
    h1' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)
    h2' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    ⊢ False
  -/
  rw [h2'.emultiplicity_eq_multiplicity, Nat.cast_lt] at h2
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)) 2
    h2 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3 : LT.lt (emultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMu …
    h1' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)
    h2' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    ⊢ False
  -/
  rw [h3'.emultiplicity_eq_multiplicity, Nat.cast_lt] at h3
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)) 2
    h2 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h1' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)
    h2' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    ⊢ False
  -/
  have := (pow_dvd_pow_of_dvd (lambda_sq_dvd_c S') 3).mul_left S'.u
  rw [← pow_mul, ← S'.H, a_cube_add_b_cube_eq_mul, pow_dvd_iff_le_emultiplicity,
    emultiplicity_mul hζ.zeta_sub_one_prime', emultiplicity_mul hζ.zeta_sub_one_prime',
      h1'.emultiplicity_eq_multiplicity, h2'.emultiplicity_eq_multiplicity,
      h3'.emultiplicity_eq_multiplicity, ← Nat.cast_add, ← Nat.cast_add, Nat.cast_le] at this
  /-
    case intro.intro
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    h1 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)) 2
    h2 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3 : LT.lt (multiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h1' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a S'.b)
    h2' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    h3' : FiniteMultiplicity (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S'.a (HMul.hMul …
    this : LE.le (HMul.hMul 2 3) (HAdd.hAdd (HAdd.hAdd (multiplicity (HSub.hSub hζ …
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


open Units in
/-- Given `S' : Solution'`, we may assume that `λ ^ 2` divides `S'.a + S'.b ∨ λ ^ 2` (see also the
result below). -/
lemma ex_cube_add_cube_eq_and_isCoprime_and_not_dvd_and_dvd :
    ∃ (a' b' : 𝓞 K), a' ^ 3 + b' ^ 3 = S'.u * S'.c ^ 3 ∧ IsCoprime a' b' ∧ ¬ λ ∣ a' ∧
      ¬ λ ∣ b' ∧ λ ^ 2 ∣ a' + b' := by
  /-
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    ⊢ Exists fun a' => Exists fun b' => And (Eq (HAdd.hAdd (HPow.hPow a' 3) (HPow. …
  -/
  rcases lambda_sq_dvd_or_dvd_or_dvd S' with h | h | h
    /-
      case inl
      K : Type u_1
      inst✝³ : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝² : NumberField K
      inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      inst✝ : DecidableRel fun a b => Dvd.dvd a b
      h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a S'.b)
      ⊢ Exists fun a' => Exists fun b' => And (Eq (HAdd.hAdd (HPow.hPow a' 3) (HPow. …
    -/
  · exact ⟨S'.a, S'.b, S'.H, S'.coprime, S'.ha, S'.hb, h⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      K : Type u_1
      inst✝³ : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝² : NumberField K
      inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      inst✝ : DecidableRel fun a b => Dvd.dvd a b
      h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HMul.hMu …
      ⊢ Exists fun a' => Exists fun b' => And (Eq (HAdd.hAdd (HPow.hPow a' 3) (HPow. …
    -/
  · refine ⟨S'.a, η * S'.b, ?_, ?_, S'.ha, fun ⟨x, hx⟩ ↦ S'.hb ⟨η ^ 2 * x, ?_⟩, h⟩
      /-
        case inr.inl.refine_1
        K : Type u_1
        inst✝³ : Field K
        ζ : K
        hζ : IsPrimitiveRoot ζ ↑3
        S' : FermatLastTheoremForThreeGen.Solution' hζ
        inst✝² : NumberField K
        inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
        inst✝ : DecidableRel fun a b => Dvd.dvd a b
        h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HMul.hMu …
        ⊢ Eq (HAdd.hAdd (HPow.hPow S'.a 3) (HPow.hPow (HMul.hMul (↑⋯.unit) S'.b) 3)) ( …
      -/
    · simp [mul_pow, ← val_pow_eq_pow_val, hζ.toInteger_cube_eq_one, val_one, one_mul, S'.H]
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.refine_2
        K : Type u_1
        inst✝³ : Field K
        ζ : K
        hζ : IsPrimitiveRoot ζ ↑3
        S' : FermatLastTheoremForThreeGen.Solution' hζ
        inst✝² : NumberField K
        inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
        inst✝ : DecidableRel fun a b => Dvd.dvd a b
        h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HMul.hMu …
        ⊢ IsCoprime S'.a (HMul.hMul (↑⋯.unit) S'.b)
      -/
    · refine (isCoprime_mul_unit_left_right (Units.isUnit η) _ _).2 S'.coprime
      /-
        🎉 no goals
      -/
    · rw [mul_comm _ x, ← mul_assoc, ← hx, mul_comm _ S'.b, mul_assoc, ← pow_succ', coe_eta,
        hζ.toInteger_cube_eq_one, mul_one]
    /-
      case inr.inr
      K : Type u_1
      inst✝³ : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝² : NumberField K
      inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      inst✝ : DecidableRel fun a b => Dvd.dvd a b
      h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HMul.hMu …
      ⊢ Exists fun a' => Exists fun b' => And (Eq (HAdd.hAdd (HPow.hPow a' 3) (HPow. …
    -/
  · refine ⟨S'.a, η ^ 2 * S'.b, ?_, ?_, S'.ha, fun ⟨x, hx⟩ ↦ S'.hb ⟨η * x, ?_⟩, h⟩
    · rw [mul_pow, ← pow_mul, mul_comm 2, pow_mul, coe_eta, hζ.toInteger_cube_eq_one, one_pow,
        one_mul, S'.H]
      /-
        case inr.inr.refine_2
        K : Type u_1
        inst✝³ : Field K
        ζ : K
        hζ : IsPrimitiveRoot ζ ↑3
        S' : FermatLastTheoremForThreeGen.Solution' hζ
        inst✝² : NumberField K
        inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
        inst✝ : DecidableRel fun a b => Dvd.dvd a b
        h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S'.a (HMul.hMu …
        ⊢ IsCoprime S'.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S'.b)
      -/
    · exact (isCoprime_mul_unit_left_right ((Units.isUnit η).pow _) _ _).2 S'.coprime
      /-
        🎉 no goals
      -/
    · rw [mul_comm _ x, ← mul_assoc, ← hx, mul_comm _ S'.b, mul_assoc, ← pow_succ, coe_eta,
        hζ.toInteger_cube_eq_one, mul_one]


/-- Given `S' : Solution'`, then there is `S₁ : Solution` such that
`S₁.multiplicity = S'.multiplicity`. -/
lemma exists_Solution_of_Solution' : ∃ (S₁ : Solution hζ), S₁.multiplicity = S'.multiplicity := by
  /-
    K : Type u_1
    inst✝³ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S' : FermatLastTheoremForThreeGen.Solution' hζ
    inst✝² : NumberField K
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    inst✝ : DecidableRel fun a b => Dvd.dvd a b
    ⊢ Exists fun S₁ => Eq S₁.multiplicity S'.multiplicity
  -/
  obtain ⟨a, b, H, coprime, ha, hb, hab⟩ := ex_cube_add_cube_eq_and_isCoprime_and_not_dvd_and_dvd S'
  exact ⟨
  { a := a
    b := b
    c := S'.c
    u := S'.u
    ha := ha
    hb := hb
    hc := S'.hc
    coprime := coprime
    hcdvd := S'.hcdvd
    H := H
    hab := hab }, rfl⟩


                                                                    /-
                                                                      K : Type u_1
                                                                      inst✝ : Field K
                                                                      ζ : K
                                                                      hζ : IsPrimitiveRoot ζ ↑3
                                                                      S : FermatLastTheoremForThreeGen.Solution hζ
                                                                      ⊢ Eq (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b)) (HAdd.hAdd (HAdd.hAdd S.a S.b)  …
                                                                    -/
lemma a_add_eta_mul_b : S.a + η * S.b = (S.a + S.b) + λ * S.b := by rw [coe_eta]; ring
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- Given `(S : Solution)`, we have that `λ ∣ (S.a + η * S.b)`. -/
lemma lambda_dvd_a_add_eta_mul_b : λ ∣ (S.a + η * S.b) :=
                                                             /-
                                                               K : Type u_1
                                                               inst✝ : Field K
                                                               ζ : K
                                                               hζ : IsPrimitiveRoot ζ ↑3
                                                               S : FermatLastTheoremForThreeGen.Solution hζ
                                                               ⊢ Ne 2 0
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  a_add_eta_mul_b S ▸ dvd_add (dvd_trans (dvd_pow_self _ (by decide)) S.hab) ⟨S.b, by rw [mul_comm]⟩
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- Given `(S : Solution)`, we have that `λ ∣ (S.a + η ^ 2 * S.b)`. -/
lemma lambda_dvd_a_add_eta_sq_mul_b : λ ∣ (S.a + η ^ 2 * S.b) := by
  /-
    K : Type u_1
    inst✝ : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯. …
  -/
  rw [show S.a + η ^ 2 * S.b = (S.a + S.b) + λ ^ 2 * S.b + 2 * λ * S.b by rw [coe_eta]; ring]
  exact dvd_add (dvd_add (dvd_trans (dvd_pow_self _ (by decide)) S.hab) ⟨λ * S.b, by ring⟩)
    ⟨2 * S.b, by ring⟩


/-- Given `(S : Solution)`, we have that `λ ^ 2` does not divide `S.a + η * S.b`. -/
lemma lambda_sq_not_dvd_a_add_eta_mul_b : ¬ λ ^ 2 ∣ (S.a + η * S.b) := by
  simp_rw [a_add_eta_mul_b, dvd_add_right S.hab, pow_two, mul_dvd_mul_iff_left
    hζ.zeta_sub_one_prime'.ne_zero, S.hb, not_false_eq_true]


/-- Given `(S : Solution)`, we have that `λ ^ 2` does not divide `S.a + η ^ 2 * S.b`. -/
lemma lambda_sq_not_dvd_a_add_eta_sq_mul_b : ¬ λ ^ 2 ∣ (S.a + η ^ 2 * S.b) := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Not (Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S.a (HMul.h …
  -/
  intro ⟨k, hk⟩
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k : NumberField.RingOfIntegers K
    hk : Eq (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b)) (HMul.hMul (HP …
    ⊢ False
  -/
  rcases S.hab with ⟨k', hk'⟩
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k : NumberField.RingOfIntegers K
    hk : Eq (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b)) (HMul.hMul (HP …
    k' : NumberField.RingOfIntegers K
    hk' : Eq (HAdd.hAdd S.a S.b) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1)  …
    ⊢ False
  -/
  refine S.hb ⟨(k - k') * (-η), ?_⟩
  rw [show S.a + η ^ 2 * S.b = S.a + S.b - S.b + η ^ 2 * S.b by ring, hk',
    show λ ^ 2 * k' - S.b + η ^ 2 * S.b = λ * (S.b * (η +1) + λ * k') by rw [coe_eta]; ring,
    pow_two, mul_assoc] at hk
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k k' : NumberField.RingOfIntegers K
    hk : Eq (HMul.hMul (HSub.hSub hζ.toInteger 1) (HAdd.hAdd (HMul.hMul S.b (HAdd. …
    hk' : Eq (HAdd.hAdd S.a S.b) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1)  …
    ⊢ Eq S.b (HMul.hMul (HSub.hSub hζ.toInteger 1) (HMul.hMul (HSub.hSub k k') (Ne …
  -/
  simp only [mul_eq_mul_left_iff, hζ.zeta_sub_one_prime'.ne_zero, or_false] at hk
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k k' : NumberField.RingOfIntegers K
    hk' : Eq (HAdd.hAdd S.a S.b) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1)  …
    hk : Eq (HAdd.hAdd (HMul.hMul S.b (HAdd.hAdd (↑⋯.unit) 1)) (HMul.hMul (HSub.hS …
    ⊢ Eq S.b (HMul.hMul (HSub.hSub hζ.toInteger 1) (HMul.hMul (HSub.hSub k k') (Ne …
  -/
  apply_fun (· * -↑η) at hk
  rw [show (S.b * (η + 1) + λ * k') * -η = (- S.b) * (η ^ 2 + η + 1 - 1) - η * λ * k' by ring,
    eta_sq, show -S.b * (-↑η - 1 + ↑η + 1 - 1) = S.b by ring, sub_eq_iff_eq_add] at hk
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k k' : NumberField.RingOfIntegers K
    hk' : Eq (HAdd.hAdd S.a S.b) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1)  …
    hk : Eq S.b (HAdd.hAdd (HMul.hMul (HMul.hMul (HSub.hSub hζ.toInteger 1) k) (Ne …
    ⊢ Eq S.b (HMul.hMul (HSub.hSub hζ.toInteger 1) (HMul.hMul (HSub.hSub k k') (Ne …
  -/
  rw [hk]
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k k' : NumberField.RingOfIntegers K
    hk' : Eq (HAdd.hAdd S.a S.b) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1)  …
    hk : Eq S.b (HAdd.hAdd (HMul.hMul (HMul.hMul (HSub.hSub hζ.toInteger 1) k) (Ne …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HSub.hSub hζ.toInteger 1) k) (Neg.neg ↑ …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If `p : 𝓞 K` is a prime that divides both `S.a + S.b` and `S.a + η * S.b`, then `p`
is associated with `λ`. -/
lemma associated_of_dvd_a_add_b_of_dvd_a_add_eta_mul_b {p : 𝓞 K} (hp : Prime p)
    (hpab : p ∣ S.a + S.b) (hpaηb : p ∣ S.a + η * S.b) : Associated p λ := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd S.a S.b)
    hpaηb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b))
    ⊢ Associated p (HSub.hSub hζ.toInteger 1)
  -/
  suffices p_lam : p ∣ λ from hp.associated_of_dvd hζ.zeta_sub_one_prime' p_lam
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd S.a S.b)
    hpaηb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a, ← one_mul S.b] at hpab
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a] at hpaηb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  have := dvd_mul_sub_mul_mul_gcd_of_dvd hpab hpaηb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    this : Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul 1 ↑⋯.unit) (HMul.hMul 1 1))  …
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rwa [one_mul, one_mul, coe_eta, IsUnit.dvd_mul_right <| (gcd_isUnit_iff _ _).2 S.coprime] at this
  /-
    🎉 no goals
  -/


/-- If `p : 𝓞 K` is a prime that divides both `S.a + S.b` and `S.a + η ^ 2 * S.b`, then `p`
is associated with `λ`. -/
lemma associated_of_dvd_a_add_b_of_dvd_a_add_eta_sq_mul_b {p : 𝓞 K} (hp : Prime p)
    (hpab : p ∣ (S.a + S.b)) (hpaηsqb : p ∣ (S.a + η ^ 2 * S.b)) : Associated p λ := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd S.a S.b)
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Associated p (HSub.hSub hζ.toInteger 1)
  -/
  suffices p_lam : p ∣ λ from hp.associated_of_dvd hζ.zeta_sub_one_prime' p_lam
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd S.a S.b)
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a, ← one_mul S.b] at hpab
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a] at hpaηsqb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  have := dvd_mul_sub_mul_mul_gcd_of_dvd hpab hpaηsqb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul 1 (HPow.hPow (↑⋯.unit) 2)) ( …
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [one_mul, mul_one, IsUnit.dvd_mul_right <| (gcd_isUnit_iff _ _).2 S.coprime, ← dvd_neg] at this
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (Neg.neg (HSub.hSub (HPow.hPow (↑⋯.unit) 2) 1))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  convert dvd_mul_of_dvd_left this η using 1
  /-
    case h.e'_4
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (Neg.neg (HSub.hSub (HPow.hPow (↑⋯.unit) 2) 1))
    ⊢ Eq (HSub.hSub hζ.toInteger 1) (HMul.hMul (Neg.neg (HSub.hSub (HPow.hPow (↑⋯. …
  -/
  rw [eta_sq, neg_sub, sub_mul, sub_mul, neg_mul, ← pow_two, eta_sq, coe_eta]
  /-
    case h.e'_4
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpab : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul 1 S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (Neg.neg (HSub.hSub (HPow.hPow (↑⋯.unit) 2) 1))
    ⊢ Eq (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 1 hζ.toInteger) (HSub.hS …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- If `p : 𝓞 K` is a prime that divides both `S.a + η * S.b` and `S.a + η ^ 2 * S.b`, then `p`
is associated with `λ`. -/
lemma associated_of_dvd_a_add_eta_mul_b_of_dvd_a_add_eta_sq_mul_b {p : 𝓞 K} (hp : Prime p)
    (hpaηb : p ∣ S.a + η * S.b) (hpaηsqb : p ∣ S.a + η ^ 2 * S.b) : Associated p λ := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Associated p (HSub.hSub hζ.toInteger 1)
  -/
  suffices p_lam : p ∣ λ from hp.associated_of_dvd hζ.zeta_sub_one_prime' p_lam
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a] at hpaηb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd S.a (HMul.hMul (HPow.hPow (↑⋯.unit) 2) S.b))
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [← one_mul S.a] at hpaηsqb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  have := dvd_mul_sub_mul_mul_gcd_of_dvd hpaηb hpaηsqb
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (HMul.hMul (HSub.hSub (HMul.hMul 1 (HPow.hPow (↑⋯.unit) 2)) ( …
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  rw [one_mul, mul_one, IsUnit.dvd_mul_right <| (gcd_isUnit_iff _ _).2 S.coprime] at this
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (HSub.hSub (HPow.hPow (↑⋯.unit) 2) ↑⋯.unit)
    ⊢ Dvd.dvd p (HSub.hSub hζ.toInteger 1)
  -/
  convert (dvd_mul_of_dvd_left (dvd_mul_of_dvd_left this η) η) using 1
  /-
    case h.e'_4
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    p : NumberField.RingOfIntegers K
    hp : Prime p
    hpaηb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (↑⋯.unit) S.b))
    hpaηsqb : Dvd.dvd p (HAdd.hAdd (HMul.hMul 1 S.a) (HMul.hMul (HPow.hPow (↑⋯.uni …
    this : Dvd.dvd p (HSub.hSub (HPow.hPow (↑⋯.unit) 2) ↑⋯.unit)
    ⊢ Eq (HSub.hSub hζ.toInteger 1) (HMul.hMul (HMul.hMul (HSub.hSub (HPow.hPow (↑ …
  -/
  symm
  calc _ = (-η.1 - 1 - η) * (-η - 1) := by rw [eta_sq, mul_assoc, ← pow_two, eta_sq]
  _ = 2 * η.1 ^ 2 + 3 * η + 1 := by ring
  _ = λ := by rw [eta_sq, coe_eta]; ring


/-- Given `S : Solution`, we let `S.y` be any element such that `S.a + η * S.b = λ * S.y` -/
private noncomputable def y := (lambda_dvd_a_add_eta_mul_b S).choose

private lemma y_spec : S.a + η * S.b = λ * S.y :=
  (lambda_dvd_a_add_eta_mul_b S).choose_spec


/-- Given `S : Solution`, we let `S.z` be any element such that `S.a + η ^ 2 * S.b = λ * S.z` -/
private noncomputable def z := (lambda_dvd_a_add_eta_sq_mul_b S).choose

private lemma z_spec : S.a + η ^ 2 * S.b = λ * S.z :=
  (lambda_dvd_a_add_eta_sq_mul_b S).choose_spec


private lemma lambda_not_dvd_y : ¬ λ ∣ S.y := fun h ↦ by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
    ⊢ False
  -/
  replace h := mul_dvd_mul_left ((η : 𝓞 K) - 1) h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HSub.hSub (↑⋯.unit) 1) (HSub.hSub hζ.toInteger 1)) (HM …
    ⊢ False
  -/
  rw [coe_eta, ← y_spec, ← pow_two] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S.a (HMul.hMul …
    ⊢ False
  -/
  exact lambda_sq_not_dvd_a_add_eta_mul_b _ h
  /-
    🎉 no goals
  -/


private lemma lambda_not_dvd_z : ¬ λ ∣ S.z := fun h ↦ by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
    ⊢ False
  -/
  replace h := mul_dvd_mul_left ((η : 𝓞 K) - 1) h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HSub.hSub (↑⋯.unit) 1) (HSub.hSub hζ.toInteger 1)) (HM …
    ⊢ False
  -/
  rw [coe_eta, ← z_spec, ← pow_two] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HAdd.hAdd S.a (HMul.hMul …
    ⊢ False
  -/
  exact lambda_sq_not_dvd_a_add_eta_sq_mul_b _ h
  /-
    🎉 no goals
  -/


/-- We have that `λ ^ (3*S.multiplicity-2)` divides `S.a + S.b`. -/
private lemma lambda_pow_dvd_a_add_b : λ ^ (3 * S.multiplicity - 2) ∣ S.a + S.b := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  have h : λ ^ S.multiplicity ∣ S.c := pow_multiplicity_dvd _ _
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) S.multiplicity) S.c
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  replace h : (λ ^ multiplicity S) ^ 3 ∣ S.u * S.c ^ 3 := by simp [h]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HPow.hPow (HSub.hSub hζ.toInteger 1) S.multiplicity) 3 …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  rw [← S.H, a_cube_add_b_cube_eq_mul, ← pow_mul, mul_comm, y_spec, z_spec] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HMul.hMul 3 S.multiplicity) …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  apply hζ.zeta_sub_one_prime'.pow_dvd_of_dvd_mul_left _ S.lambda_not_dvd_z
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HMul.hMul 3 S.multiplicity) …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  apply hζ.zeta_sub_one_prime'.pow_dvd_of_dvd_mul_left _ S.lambda_not_dvd_y
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HMul.hMul 3 S.multiplicity) …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  have := S.two_le_multiplicity
  rw [show 3 * multiplicity S = 3 * multiplicity S - 2 + 1 + 1 by omega, pow_succ, pow_succ,
    show (S.a + S.b) * (λ * y S) * (λ * z S) = (S.a + S.b) * y S * z S * λ * λ by ring] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub. …
    this : LE.le 2 S.multiplicity
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  simp only [mul_dvd_mul_iff_right hζ.zeta_sub_one_prime'.ne_zero] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    this : LE.le 2 S.multiplicity
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mu …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul.hMul 3 S.mult …
  -/
  rwa [show (S.a + S.b) * y S * z S = y S * (z S * (S.a + S.b)) by ring] at h
  /-
    🎉 no goals
  -/


/-- Given `S : Solution`, we let `S.x` be any element such that
`S.a + S.b = λ ^ (3*S.multiplicity-2) * S.x` -/
private noncomputable def x := (lambda_pow_dvd_a_add_b S).choose

private lemma x_spec : S.a + S.b = λ ^ (3 * S.multiplicity - 2) * S.x :=
  (lambda_pow_dvd_a_add_b S).choose_spec


/-- Given `S : Solution`, we let `S.w` be any element such that `S.c = λ ^ S.multiplicity * S.w` -/
private noncomputable def w :=
  (pow_multiplicity_dvd (hζ.toInteger - 1) S.c).choose


omit [NumberField K] [IsCyclotomicExtension {3} ℚ K] in
private lemma w_spec : S.c = λ ^ S.multiplicity * S.w :=
  (pow_multiplicity_dvd (hζ.toInteger - 1) S.c).choose_spec


private lemma lambda_not_dvd_w : ¬ λ ∣ S.w := fun h ↦ by
  refine S.toSolution'.multiplicity_lambda_c_finite.not_pow_dvd_of_multiplicity_lt
    (lt_add_one S.multiplicity) ?_
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HAdd.hAdd S.multiplicity 1))  …
  -/
  rw [pow_succ', mul_comm]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) S.multiplicity) (HS …
  -/
  exact S.w_spec ▸ (mul_dvd_mul_left (λ ^ S.multiplicity) h)
  /-
    🎉 no goals
  -/


private lemma lambda_not_dvd_x : ¬ λ ∣ S.x := fun h ↦ by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
    ⊢ False
  -/
  replace h := mul_dvd_mul_left (λ ^ (3 * S.multiplicity - 2)) h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub (HMul. …
    ⊢ False
  -/
  rw [mul_comm, ← x_spec] at h
  replace h :=
    mul_dvd_mul (mul_dvd_mul h S.lambda_dvd_a_add_eta_mul_b) S.lambda_dvd_a_add_eta_sq_mul_b
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HMul.hMul (HMul.hMul (HSub.hSub hζ.toInteger 1) (HPow. …
    ⊢ False
  -/
  simp only [← a_cube_add_b_cube_eq_mul, S.H, w_spec, Units.isUnit, IsUnit.dvd_mul_left] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HMul.hMul (HMul.hMul (HSub.hSub hζ.toInteger 1) (HPow. …
    ⊢ False
  -/
  rw [← pow_succ', mul_comm, ← mul_assoc, ← pow_succ'] at h
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HAdd.hAdd (HAdd. …
    ⊢ False
  -/
  have := S.two_le_multiplicity
  rw [show 3 * multiplicity S - 2 + 1 + 1 = 3 * multiplicity S by omega, mul_pow, ← pow_mul,
    mul_comm _ 3, mul_dvd_mul_iff_left _] at h
    /-
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (HPow.hPow (FermatLastTheoremForThreeGe …
      this : LE.le 2 S.multiplicity
      ⊢ False
    -/
  · exact lambda_not_dvd_w _ <| hζ.zeta_sub_one_prime'.dvd_of_dvd_pow h
    /-
      🎉 no goals
    -/
    /-
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      h : Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HMul.hMul 3 S.mu …
      this : LE.le 2 S.multiplicity
      ⊢ Ne (HPow.hPow (HSub.hSub hζ.toInteger 1) (HMul.hMul 3 S.multiplicity)) 0
    -/
  · simp [hζ.zeta_sub_one_prime'.ne_zero]
    /-
      🎉 no goals
    -/


private lemma isCoprime_helper {r s t w : 𝓞 K} (hr : ¬ λ ∣ r) (hs : ¬ λ ∣ s)
    (Hp : ∀ {p}, Prime p → p ∣ t → p ∣ w → Associated p λ) (H₁ : ∀ {q}, q ∣ r → q ∣ t)
    (H₂ : ∀ {q}, q ∣ s → q ∣ w) : IsCoprime r s := by
  refine isCoprime_of_prime_dvd (not_and.2 (fun _ hz ↦ hs (by simp [hz])))
    (fun p hp p_dvd_r p_dvd_s ↦ hr ?_)
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    r s t w : NumberField.RingOfIntegers K
    hr : Not (Dvd.dvd (HSub.hSub hζ.toInteger 1) r)
    hs : Not (Dvd.dvd (HSub.hSub hζ.toInteger 1) s)
    Hp : ∀ {p : NumberField.RingOfIntegers K}, Prime p → Dvd.dvd p t → Dvd.dvd p w …
    H₁ : ∀ {q : NumberField.RingOfIntegers K}, Dvd.dvd q r → Dvd.dvd q t
    H₂ : ∀ {q : NumberField.RingOfIntegers K}, Dvd.dvd q s → Dvd.dvd q w
    p : NumberField.RingOfIntegers K
    hp : Prime p
    p_dvd_r : Dvd.dvd p r
    p_dvd_s : Dvd.dvd p s
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) r
  -/
  rwa [← Associated.dvd_iff_dvd_left <| Hp hp (H₁ p_dvd_r) (H₂ p_dvd_s)]
  /-
    🎉 no goals
  -/


private lemma isCoprime_x_y [DecidableRel fun (a b : 𝓞 K) ↦ a ∣ b] : IsCoprime S.x S.y :=
  isCoprime_helper (lambda_not_dvd_x S) (lambda_not_dvd_y S)
    (associated_of_dvd_a_add_b_of_dvd_a_add_eta_mul_b S) (fun hq ↦ x_spec S ▸ hq.mul_left _)
      (fun hq ↦ y_spec S ▸ hq.mul_left _)


private lemma isCoprime_x_z [DecidableRel fun (a b : 𝓞 K) ↦ a ∣ b] : IsCoprime S.x S.z :=
  isCoprime_helper (lambda_not_dvd_x S) (lambda_not_dvd_z S)
    (associated_of_dvd_a_add_b_of_dvd_a_add_eta_sq_mul_b S) (fun hq ↦ x_spec S ▸ hq.mul_left _)
      (fun hq ↦ z_spec S ▸ hq.mul_left _)


private lemma isCoprime_y_z : IsCoprime S.y S.z :=
  isCoprime_helper (lambda_not_dvd_y S) (lambda_not_dvd_z S)
    (associated_of_dvd_a_add_eta_mul_b_of_dvd_a_add_eta_sq_mul_b S)
    (fun hq ↦ y_spec S ▸ hq.mul_left _) (fun hq ↦ z_spec S ▸ hq.mul_left _)


private lemma x_mul_y_mul_z_eq_u_mul_w_cube : S.x * S.y * S.z = S.u * S.w ^ 3 := by
  suffices hh : λ ^ (3 * S.multiplicity - 2) * S.x * λ * S.y * λ * S.z =
      S.u * λ ^ (3 * S.multiplicity) * S.w ^ 3 by
    rw [show λ ^ (3 * multiplicity S - 2) * x S * λ * y S * λ * z S =
      λ ^ (3 * multiplicity S - 2) * λ * λ * x S * y S * z S by ring] at hh
    have := S.two_le_multiplicity
    rw [mul_comm _ (λ ^ (3 * multiplicity S)), ← pow_succ, ← pow_succ,
      show 3 * multiplicity S - 2 + 1 + 1 = 3 * multiplicity S by omega, mul_assoc, mul_assoc,
      mul_assoc] at hh
    simp only [mul_eq_mul_left_iff, pow_eq_zero_iff', hζ.zeta_sub_one_prime'.ne_zero, ne_eq,
      mul_eq_zero, OfNat.ofNat_ne_zero, false_or, false_and, or_false] at hh
    convert hh using 1
    ring
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HSub.h …
  -/
  simp only [← x_spec, mul_assoc, ← y_spec, ← z_spec]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HMul.hMul (HAdd.hAdd S.a S.b) (HMul.hMul (HAdd.hAdd S.a (HMul.hMul (↑⋯.u …
  -/
  rw [mul_comm 3, pow_mul, ← mul_pow, ← w_spec, ← S.H, a_cube_add_b_cube_eq_mul]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HMul.hMul (HAdd.hAdd S.a S.b) (HMul.hMul (HAdd.hAdd S.a (HMul.hMul (↑⋯.u …
  -/
  ring
  /-
    🎉 no goals
  -/


private lemma exists_cube_associated :
    (∃ X, Associated (X ^ 3) S.x) ∧ (∃ Y, Associated (Y ^ 3) S.y) ∧
      ∃ Z, Associated (Z ^ 3) S.z := by classical
  have h₁ := S.isCoprime_x_z.mul_left S.isCoprime_y_z
  have h₂ : Associated (S.w ^ 3) (S.x * S.y * S.z) :=
    ⟨S.u, by rw [x_mul_y_mul_z_eq_u_mul_w_cube S, mul_comm]⟩
  obtain ⟨T, h₃⟩ := exists_associated_pow_of_associated_pow_mul h₁ h₂
  exact ⟨exists_associated_pow_of_associated_pow_mul S.isCoprime_x_y h₃,
    exists_associated_pow_of_associated_pow_mul S.isCoprime_x_y.symm (mul_comm _ S.x ▸ h₃),
    exists_associated_pow_of_associated_pow_mul h₁.symm (mul_comm _ S.z ▸ h₂)⟩


/-- Given `S : Solution`, we let `S.u₁` and `S.X` be any elements such that
`S.X ^ 3 * S.u₁ = S.x` -/
private noncomputable def X := (exists_cube_associated S).1.choose

private noncomputable def u₁ := (exists_cube_associated S).1.choose_spec.choose

private lemma X_u₁_spec : S.X ^ 3 * S.u₁ = S.x :=
  (exists_cube_associated S).1.choose_spec.choose_spec


/-- Given `S : Solution`, we let `S.u₂` and `S.Y` be any elements such that
`S.Y ^ 3 * S.u₂ = S.y` -/
private noncomputable def Y := (exists_cube_associated S).2.1.choose

private noncomputable def u₂ := (exists_cube_associated S).2.1.choose_spec.choose

private lemma Y_u₂_spec : S.Y ^ 3 * S.u₂ = S.y :=
  (exists_cube_associated S).2.1.choose_spec.choose_spec


/-- Given `S : Solution`, we let `S.u₃` and `S.Z` be any elements such that
`S.Z ^ 3 * S.u₃ = S.z` -/
private noncomputable def Z := (exists_cube_associated S).2.2.choose

private noncomputable def u₃ :=(exists_cube_associated S).2.2.choose_spec.choose

private lemma Z_u₃_spec : S.Z ^ 3 * S.u₃ = S.z :=
  (exists_cube_associated S).2.2.choose_spec.choose_spec


private lemma X_ne_zero : S.X ≠ 0 :=
                                   /-
                                     K : Type u_1
                                     inst✝² : Field K
                                     ζ : K
                                     hζ : IsPrimitiveRoot ζ ↑3
                                     S : FermatLastTheoremForThreeGen.Solution hζ
                                     inst✝¹ : NumberField K
                                     inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                     h : Eq (FermatLastTheoremForThreeGen.Solution.X S) 0
                                     ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution.x S)
                                   -/
  fun h ↦ lambda_not_dvd_x S <| by simp [← X_u₁_spec, h]
                                   /-
                                     🎉 no goals
                                   -/


private lemma lambda_not_dvd_X : ¬ λ ∣ S.X :=
                                                                                 /-
                                                                                   K : Type u_1
                                                                                   inst✝² : Field K
                                                                                   ζ : K
                                                                                   hζ : IsPrimitiveRoot ζ ↑3
                                                                                   S : FermatLastTheoremForThreeGen.Solution hζ
                                                                                   inst✝¹ : NumberField K
                                                                                   inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                                                                   h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
                                                                                   ⊢ Ne 3 0
                                                                                 -/
  fun h ↦ lambda_not_dvd_x S <| X_u₁_spec S ▸ dvd_mul_of_dvd_left (dvd_pow h (by decide)) _
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


private lemma lambda_not_dvd_Y : ¬ λ ∣ S.Y :=
                                                                                 /-
                                                                                   K : Type u_1
                                                                                   inst✝² : Field K
                                                                                   ζ : K
                                                                                   hζ : IsPrimitiveRoot ζ ↑3
                                                                                   S : FermatLastTheoremForThreeGen.Solution hζ
                                                                                   inst✝¹ : NumberField K
                                                                                   inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                                                                   h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
                                                                                   ⊢ Ne 3 0
                                                                                 -/
  fun h ↦ lambda_not_dvd_y S <| Y_u₂_spec S ▸ dvd_mul_of_dvd_left (dvd_pow h (by decide)) _
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


private lemma lambda_not_dvd_Z : ¬ λ ∣ S.Z :=
                                                                                 /-
                                                                                   K : Type u_1
                                                                                   inst✝² : Field K
                                                                                   ζ : K
                                                                                   hζ : IsPrimitiveRoot ζ ↑3
                                                                                   S : FermatLastTheoremForThreeGen.Solution hζ
                                                                                   inst✝¹ : NumberField K
                                                                                   inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                                                                   h : Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution. …
                                                                                   ⊢ Ne 3 0
                                                                                 -/
  fun h ↦ lambda_not_dvd_z S <| Z_u₃_spec S ▸ dvd_mul_of_dvd_left (dvd_pow h (by decide)) _
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


private lemma isCoprime_Y_Z : IsCoprime S.Y S.Z := by
  rw [← IsCoprime.pow_iff (m := 3) (n := 3) (by decide) (by decide),
    ← isCoprime_mul_unit_right_left S.u₂.isUnit, ← isCoprime_mul_unit_right_right S.u₃.isUnit,
    Y_u₂_spec, Z_u₃_spec]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ IsCoprime (FermatLastTheoremForThreeGen.Solution.y S) (FermatLastTheoremForT …
  -/
  exact isCoprime_y_z S
  /-
    🎉 no goals
  -/


private lemma formula1 : S.X^3*S.u₁*λ^(3*S.multiplicity-2)+S.Y^3*S.u₂*λ*η+S.Z^3*S.u₃*λ*η^2 = 0 := by
  rw [X_u₁_spec, Y_u₂_spec, Z_u₃_spec, mul_comm S.x, ← x_spec, mul_comm S.y, ← y_spec, mul_comm S.z,
    ← z_spec, eta_sq]
  calc _ = S.a+S.b+η^2*S.b-S.a+η^2*S.b+2*η*S.b+S.b := by ring
  _ = 0 := by rw [eta_sq]; ring


/-- Let `u₄ := η * S.u₃ * S.u₂⁻¹` -/
private noncomputable def u₄ := η * S.u₃ * S.u₂⁻¹

private lemma u₄_def : S.u₄ = η * S.u₃ * S.u₂⁻¹ := rfl

/-- Let `u₅ := -η ^ 2 * S.u₁ * S.u₂⁻¹` -/
private noncomputable def u₅ := -η ^ 2 * S.u₁ * S.u₂⁻¹

private lemma u₅_def : S.u₅ = -η ^ 2 * S.u₁ * S.u₂⁻¹ := rfl


private lemma formula2 :
    S.Y ^ 3 + S.u₄ * S.Z ^ 3 = S.u₅ * (λ ^ (S.multiplicity - 1) * S.X) ^ 3 := by
  rw [u₅_def, neg_mul, neg_mul, Units.val_neg, neg_mul, eq_neg_iff_add_eq_zero, add_assoc,
    add_comm (S.u₄ * S.Z ^ 3), ← add_assoc, add_comm (S.Y ^ 3)]
  apply mul_right_cancel₀ <| mul_ne_zero
    (mul_ne_zero hζ.zeta_sub_one_prime'.ne_zero S.u₂.isUnit.ne_zero) (Units.isUnit η).ne_zero
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HMul.hMul (HAdd.hAdd (HAdd.hAdd (HMul.hMul (↑(HMul.hMul (HMul.hMul (HPow …
  -/
  simp only [zero_mul, add_mul]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (↑(HMul.hMul (HMul.hMul (HPow …
  -/
  rw [← formula1 S]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (↑(HMul.hMul (HMul.hMul (HPow …
  -/
  congrm ?_ + ?_ + ?_
    /-
      case refine_1
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      ⊢ Eq (HMul.hMul (HMul.hMul (↑(HMul.hMul (HMul.hMul (HPow.hPow ⋯.unit 2) (Ferma …
    -/
  · have : (S.multiplicity-1)*3+1 = 3*S.multiplicity-2 := by have := S.two_le_multiplicity; omega
    calc _ = S.X^3 *(S.u₂*S.u₂⁻¹)*(η^3*S.u₁)*(λ^((S.multiplicity-1)*3)*λ):= by push_cast; ring
    _ = S.X^3*S.u₁*λ^(3*S.multiplicity-2) := by simp [hζ.toInteger_cube_eq_one, ← pow_succ, this]
    /-
      case refine_2
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      ⊢ Eq (HMul.hMul (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) (HMu …
    -/
  · ring
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      ⊢ Eq (HMul.hMul (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₄ S)) (HP …
    -/
  · field_simp [u₄_def]
    /-
      case refine_3
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (FermatLastTheorem …
    -/
    ring
    /-
      🎉 no goals
    -/


private lemma lambda_sq_div_u₅_mul : λ ^ 2 ∣ S.u₅ * (λ ^ (S.multiplicity - 1) * S.X) ^ 3 := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HMul.hMul (↑(FermatLastThe …
  -/
  use λ^(3*S.multiplicity-5)*S.u₅*(S.X^3)
  /-
    case h
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow (HM …
  -/
  have : 3*(S.multiplicity-1) = 2+(3*S.multiplicity-5) := by have := S.two_le_multiplicity; omega
  calc _ = λ^(3*(S.multiplicity-1))*S.u₅*S.X^3 := by ring
  _ = λ^2*λ^(3*S.multiplicity-5)*S.u₅*S.X^3 := by rw [this, pow_add]
  _ = λ^2*(λ^(3*S.multiplicity-5)*S.u₅*S.X^3) := by ring


private lemma u₄_eq_one_or_neg_one : S.u₄ = 1 ∨ S.u₄ = -1 := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Or (Eq (FermatLastTheoremForThreeGen.Solution.u₄ S) 1) (Eq (FermatLastTheore …
  -/
  have : λ^2 ∣ λ^4  := ⟨λ^2, by ring⟩
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
    ⊢ Or (Eq (FermatLastTheoremForThreeGen.Solution.u₄ S) 1) (Eq (FermatLastTheore …
  -/
  have h := S.lambda_sq_div_u₅_mul
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HMul.hMul (↑(FermatLastT …
    ⊢ Or (Eq (FermatLastTheoremForThreeGen.Solution.u₄ S) 1) (Eq (FermatLastTheore …
  -/
  apply IsCyclotomicExtension.Rat.Three.eq_one_or_neg_one_of_unit_of_congruent hζ
  /-
    case hcong
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HMul.hMul (↑(FermatLastT …
    ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
  -/
  rcases h with ⟨X, hX⟩
  rcases lambda_pow_four_dvd_cube_sub_one_or_add_one_of_lambda_not_dvd hζ S.lambda_not_dvd_Y with
    HY | HY <;> rcases lambda_pow_four_dvd_cube_sub_one_or_add_one_of_lambda_not_dvd
                                             /-
                                               case hcong.intro.inl.inl
                                               K : Type u_1
                                               inst✝² : Field K
                                               ζ : K
                                               hζ : IsPrimitiveRoot ζ ↑3
                                               S : FermatLastTheoremForThreeGen.Solution hζ
                                               inst✝¹ : NumberField K
                                               inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                               this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
                                               X : NumberField.RingOfIntegers K
                                               hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
                                               HY : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.hPow (F …
                                               HZ : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 4) (HSub.hSub (HPow.hPow (F …
                                               ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
                                             -/
      hζ S.lambda_not_dvd_Z with HZ | HZ <;> replace HY := this.trans HY <;> replace HZ :=
                        /-
                          case hcong.intro.inl.inl
                          K : Type u_1
                          inst✝² : Field K
                          ζ : K
                          hζ : IsPrimitiveRoot ζ ↑3
                          S : FermatLastTheoremForThreeGen.Solution hζ
                          inst✝¹ : NumberField K
                          inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                          this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
                          X : NumberField.RingOfIntegers K
                          hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
                          HY : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub (HPow.hPow (F …
                          HZ : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub (HPow.hPow (F …
                          ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
                        -/
      this.trans HZ <;> rcases HY with ⟨Y, hY⟩ <;> rcases HZ with ⟨Z, hZ⟩
    /-
      case hcong.intro.inl.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
    -/
  · refine ⟨-1, X-Y-S.u₄*Z, ?_⟩
    /-
      case hcong.intro.inl.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑(-1)) (HMul.hMu …
    -/
    rw [show λ^2*(X-Y-S.u₄*Z)=λ^2*X-λ^2*Y-S.u₄*(λ^2*Z) by ring, ← hX, ← hY, ← hZ, ← formula2]
    /-
      case hcong.intro.inl.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑(-1)) (HSub.hSu …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case hcong.intro.inl.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
    -/
  · refine ⟨1, -X+Y+S.u₄*Z, ?_⟩
    /-
      case hcong.intro.inl.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑1) (HMul.hMul ( …
    -/
    rw [show λ^2*(-X+Y+S.u₄*Z)=-(λ^2*X-λ^2*Y-S.u₄*(λ^2*Z)) by ring, ← hX, ← hY, ← hZ, ← formula2]
    /-
      case hcong.intro.inl.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑1) (Neg.neg (HS …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case hcong.intro.inr.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
    -/
  · refine ⟨1, X-Y-S.u₄*Z, ?_⟩
    /-
      case hcong.intro.inr.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑1) (HMul.hMul ( …
    -/
    rw [show λ^2*(X-Y-S.u₄*Z)=λ^2*X-λ^2*Y-S.u₄*(λ^2*Z) by ring, ← hX, ← hY, ← hZ, ← formula2]
    /-
      case hcong.intro.inr.inl.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HSub.hSub (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑1) (HSub.hSub ( …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case hcong.intro.inr.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Exists fun n => Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HSub.hSub  …
    -/
  · refine ⟨-1, -X+Y+S.u₄*Z, ?_⟩
    /-
      case hcong.intro.inr.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑(-1)) (HMul.hMu …
    -/
    rw [show λ^2*(-X+Y+S.u₄*Z)=-(λ^2*X-λ^2*Y-S.u₄*(λ^2*Z)) by ring, ← hX, ← hY, ← hZ, ← formula2]
    /-
      case hcong.intro.inr.inr.intro.intro
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      this : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) 2) (HPow.hPow (HSub.hSub  …
      X : NumberField.RingOfIntegers K
      hX : Eq (HMul.hMul (↑(FermatLastTheoremForThreeGen.Solution.u₅ S)) (HPow.hPow  …
      Y : NumberField.RingOfIntegers K
      hY : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) 1 …
      Z : NumberField.RingOfIntegers K
      hZ : Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Z S) 3) 1 …
      ⊢ Eq (HSub.hSub ↑(FermatLastTheoremForThreeGen.Solution.u₄ S) ↑(-1)) (Neg.neg  …
    -/
    ring
    /-
      🎉 no goals
    -/


private lemma u₄_sq : S.u₄ ^ 2 = 1 := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ Eq (HPow.hPow (FermatLastTheoremForThreeGen.Solution.u₄ S) 2) 1
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  rcases S.u₄_eq_one_or_neg_one with h | h <;> simp [h]
                                               /-
                                                 🎉 no goals
                                               -/


/-- Given `S : Solution`, we have that
`S.Y ^ 3 + (S.u₄ * S.Z) ^ 3 = S.u₅ * (λ ^ (S.multiplicity - 1) * S.X) ^ 3`. -/
private lemma formula3 :
    S.Y ^ 3 + (S.u₄ * S.Z) ^ 3 = S.u₅ * (λ ^ (S.multiplicity - 1) * S.X) ^ 3 :=
                                                        /-
                                                          K : Type u_1
                                                          inst✝² : Field K
                                                          ζ : K
                                                          hζ : IsPrimitiveRoot ζ ↑3
                                                          S : FermatLastTheoremForThreeGen.Solution hζ
                                                          inst✝¹ : NumberField K
                                                          inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                                          ⊢ Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) (HPo …
                                                        -/
  calc S.Y^3+(S.u₄*S.Z)^3=S.Y^3+S.u₄^2*S.u₄*S.Z^3 := by ring
                                                        /-
                                                          🎉 no goals
                                                        -/
                             /-
                               K : Type u_1
                               inst✝² : Field K
                               ζ : K
                               hζ : IsPrimitiveRoot ζ ↑3
                               S : FermatLastTheoremForThreeGen.Solution hζ
                               inst✝¹ : NumberField K
                               inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                               ⊢ Eq (HAdd.hAdd (HPow.hPow (FermatLastTheoremForThreeGen.Solution.Y S) 3) (HMu …
                             -/
  _ = S.Y^3+S.u₄*S.Z^3 := by simp [← Units.val_pow_eq_pow_val, S.u₄_sq]
                             /-
                               🎉 no goals
                             -/
  _ = S.u₅*(λ^(S.multiplicity-1)*S.X)^3 := S.formula2


/-- Given `S : Solution`, we construct `S₁ : Solution'`, with smaller multiplicity of `λ` in
  `c` (see `Solution'_descent_multiplicity_lt` below.). -/
noncomputable def Solution'_descent : Solution' hζ where
  a := S.Y
  b := S.u₄ * S.Z
  c := λ ^ (S.multiplicity - 1) * S.X
  u := S.u₅
  ha := S.lambda_not_dvd_Y
  hb := fun h ↦ S.lambda_not_dvd_Z <| Units.dvd_mul_left.1 h
                                  /-
                                    K : Type u_1
                                    inst✝² : Field K
                                    ζ : K
                                    hζ : IsPrimitiveRoot ζ ↑3
                                    S : FermatLastTheoremForThreeGen.Solution hζ
                                    S' : FermatLastTheoremForThreeGen.Solution' hζ
                                    inst✝¹ : NumberField K
                                    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
                                    h : Eq (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub S.multiplic …
                                    ⊢ Eq (FermatLastTheoremForThreeGen.Solution.X S) 0
                                  -/
  hc := fun h ↦ S.X_ne_zero <| by simpa [hζ.zeta_sub_one_prime'.ne_zero] using h
                                  /-
                                    🎉 no goals
                                  -/
  coprime := (isCoprime_mul_unit_left_right S.u₄.isUnit _ _).2 S.isCoprime_Y_Z
  hcdvd := by
    /-
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInt …
    -/
    refine dvd_mul_of_dvd_left (dvd_pow_self _ (fun h ↦ ?_)) _
    /-
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      h : Eq (HSub.hSub S.multiplicity 1) 0
      ⊢ False
    -/
    rw [Nat.sub_eq_iff_eq_add (le_trans (by norm_num) S.two_le_multiplicity), zero_add] at h
    /-
      K : Type u_1
      inst✝² : Field K
      ζ : K
      hζ : IsPrimitiveRoot ζ ↑3
      S : FermatLastTheoremForThreeGen.Solution hζ
      S' : FermatLastTheoremForThreeGen.Solution' hζ
      inst✝¹ : NumberField K
      inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
      h : Eq S.multiplicity 1
      ⊢ False
    -/
    simpa [h] using S.two_le_multiplicity
    /-
      🎉 no goals
    -/
  H := formula3 S


/-- We have that `S.Solution'_descent.multiplicity = S.multiplicity - 1`. -/
lemma Solution'_descent_multiplicity : S.Solution'_descent.multiplicity = S.multiplicity - 1 := by
  refine multiplicity_eq_of_dvd_of_not_dvd
    (by simp [Solution'_descent]) (fun h ↦ S.lambda_not_dvd_X ?_)
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    h : Dvd.dvd (HPow.hPow (HSub.hSub hζ.toInteger 1) (HAdd.hAdd (HSub.hSub S.mult …
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution.X S)
  -/
  obtain ⟨k, hk : λ^(S.multiplicity-1)*S.X=λ^(S.multiplicity-1+1)*k⟩ := h
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k : NumberField.RingOfIntegers K
    hk : Eq (HMul.hMul (HPow.hPow (HSub.hSub hζ.toInteger 1) (HSub.hSub S.multipli …
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution.X S)
  -/
  rw [pow_succ, mul_assoc] at hk
  simp only [mul_eq_mul_left_iff, pow_eq_zero_iff', hζ.zeta_sub_one_prime'.ne_zero, ne_eq,
    false_and, or_false] at hk
  /-
    case intro
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    k : NumberField.RingOfIntegers K
    hk : Eq (FermatLastTheoremForThreeGen.Solution.X S) (HMul.hMul (HSub.hSub hζ.t …
    ⊢ Dvd.dvd (HSub.hSub hζ.toInteger 1) (FermatLastTheoremForThreeGen.Solution.X S)
  -/
  simp [hk]
  /-
    🎉 no goals
  -/


/-- We have that `S.Solution'_descent.multiplicity < S.multiplicity`. -/
lemma Solution'_descent_multiplicity_lt :
    (Solution'_descent S).multiplicity < S.multiplicity := by
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ LT.lt S.Solution'_descent.multiplicity S.multiplicity
  -/
  rw [Solution'_descent_multiplicity S, Nat.sub_one]
  /-
    K : Type u_1
    inst✝² : Field K
    ζ : K
    hζ : IsPrimitiveRoot ζ ↑3
    S : FermatLastTheoremForThreeGen.Solution hζ
    inst✝¹ : NumberField K
    inst✝ : IsCyclotomicExtension (Singleton.singleton 3) Rat K
    ⊢ LT.lt S.multiplicity.pred S.multiplicity
  -/
  exact Nat.pred_lt <| by have := S.two_le_multiplicity; omega
  /-
    🎉 no goals
  -/


/-- Given any `S : Solution`, there is another `S₁ : Solution` such that
  `S₁.multiplicity < S.multiplicity` -/
theorem exists_Solution_multiplicity_lt :
    ∃ S₁ : Solution hζ, S₁.multiplicity < S.multiplicity := by classical
  obtain ⟨S', hS'⟩ := exists_Solution_of_Solution' (Solution'_descent S)
  exact ⟨S', hS' ▸ Solution'_descent_multiplicity_lt S⟩


/-- Fermat's Last Theorem for `n = 3`: if `a b c : ℕ` are all non-zero then
`a ^ 3 + b ^ 3 ≠ c ^ 3`. -/
theorem fermatLastTheoremThree : FermatLastTheoremFor 3 := by
  classical
  let K := CyclotomicField 3 ℚ
  let hζ := IsCyclotomicExtension.zeta_spec 3 ℚ K
  have : NumberField K := IsCyclotomicExtension.numberField {3} ℚ _
  apply FermatLastTheoremForThree_of_FermatLastTheoremThreeGen hζ
  intro a b c u hc ha hb hcdvd coprime H
  let S' : FermatLastTheoremForThreeGen.Solution' hζ :=
  { a := a
    b := b
    c := c
    u := u
    ha := ha
    hb := hb
    hc := hc
    coprime := coprime
    hcdvd := hcdvd
    H := H }
  obtain ⟨S, -⟩ := FermatLastTheoremForThreeGen.exists_Solution_of_Solution' S'
  obtain ⟨Smin, hSmin⟩ := S.exists_minimal
  obtain ⟨Sfin, hSfin⟩ := Smin.exists_Solution_multiplicity_lt
  linarith [hSmin Sfin]

