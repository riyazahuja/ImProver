lemma coprime_mul_succ {n m a} (h : n ≤ m) (ha : m - n ∣ a) : Coprime (n * a + 1) (m * a + 1) :=
  Nat.coprime_of_dvd fun p pp hn hm => by
    have : p ∣ (m - n) * a := by
      simpa [Nat.succ_sub_succ, ← Nat.mul_sub_right_distrib] using
        Nat.dvd_sub (Nat.succ_le_succ <| Nat.mul_le_mul_right a h) hm hn
    have : p ∣ a := by
      rcases (Nat.Prime.dvd_mul pp).mp this with (hp | hp)
      · exact Nat.dvd_trans hp ha
      · exact hp
    /-
      n m a : Nat
      h : LE.le n m
      ha : Dvd.dvd (HSub.hSub m n) a
      p : Nat
      pp : Nat.Prime p
      hn : Dvd.dvd p (HAdd.hAdd (HMul.hMul n a) 1)
      hm : Dvd.dvd p (HAdd.hAdd (HMul.hMul m a) 1)
      this✝ : Dvd.dvd p (HMul.hMul (HSub.hSub m n) a)
      this : Dvd.dvd p a
      ⊢ False
    -/
    apply pp.ne_one
    /-
      n m a : Nat
      h : LE.le n m
      ha : Dvd.dvd (HSub.hSub m n) a
      p : Nat
      pp : Nat.Prime p
      hn : Dvd.dvd p (HAdd.hAdd (HMul.hMul n a) 1)
      hm : Dvd.dvd p (HAdd.hAdd (HMul.hMul m a) 1)
      this✝ : Dvd.dvd p (HMul.hMul (HSub.hSub m n) a)
      this : Dvd.dvd p a
      ⊢ Eq p 1
    -/
    simpa [Nat.add_sub_cancel_left] using Nat.dvd_sub (le_add_right _ _) hn (this.mul_left n)
    /-
      🎉 no goals
    -/


private def supOfSeq (a : Fin m → ℕ) : ℕ := max m (Finset.sup .univ a) + 1


private def coprimes (a : Fin m → ℕ) : Fin m → ℕ := fun i => (i + 1) * (supOfSeq a)! + 1


lemma coprimes_lt (a : Fin m → ℕ) (i) : a i < coprimes a i := by
  have h₁ : a i < supOfSeq a :=
    Nat.lt_add_one_iff.mpr (le_max_of_le_right <| Finset.le_sup (by simp))
  have h₂ : supOfSeq a ≤ (i + 1) * (supOfSeq a)! + 1 :=
    le_trans (self_le_factorial _) (le_trans (Nat.le_mul_of_pos_left (supOfSeq a)! (succ_pos i))
      (le_add_right _ _))
  /-
    m : Nat
    a : Fin m → Nat
    i : Fin m
    h₁ : LT.lt (a i) (Nat.supOfSeq a)
    h₂ : LE.le (Nat.supOfSeq a) (HAdd.hAdd (HMul.hMul (HAdd.hAdd (↑i) 1) (Nat.supO …
    ⊢ LT.lt (a i) (Nat.coprimes a i)
  -/
  simpa only [coprimes] using lt_of_lt_of_le h₁ h₂
  /-
    🎉 no goals
  -/


private lemma pairwise_coprime_coprimes (a : Fin m → ℕ) : Pairwise (Coprime on coprimes a) := by
  /-
    m : Nat
    a : Fin m → Nat
    ⊢ Pairwise (Function.onFun Nat.Coprime (Nat.coprimes a))
  -/
  intro i j hij
  /-
    m : Nat
    a : Fin m → Nat
    i j : Fin m
    hij : Ne i j
    ⊢ Function.onFun Nat.Coprime (Nat.coprimes a) i j
  -/
  wlog ltij : i < j
    /-
      case inr
      m : Nat
      a : Fin m → Nat
      i j : Fin m
      hij : Ne i j
      this : ∀ {m : Nat} (a : Fin m → Nat) ⦃i j : Fin m⦄, Ne i j → LT.lt i j → Funct …
      ltij : Not (LT.lt i j)
      ⊢ Function.onFun Nat.Coprime (Nat.coprimes a) i j
    -/
  · exact (this a hij.symm (lt_of_le_of_ne (Fin.not_lt.mp ltij) hij.symm)).symm
    /-
      🎉 no goals
    -/
  /-
    m✝ m : Nat
    a : Fin m → Nat
    i j : Fin m
    hij : Ne i j
    ltij : LT.lt i j
    ⊢ Function.onFun Nat.Coprime (Nat.coprimes a) i j
  -/
  unfold Function.onFun coprimes
  /-
    m✝ m : Nat
    a : Fin m → Nat
    i j : Fin m
    hij : Ne i j
    ltij : LT.lt i j
    ⊢ (HAdd.hAdd (HMul.hMul (HAdd.hAdd (↑i) 1) (Nat.supOfSeq a).factorial) 1).Copr …
  -/
  have hja : j < supOfSeq a := lt_of_lt_of_le j.prop (le_step (le_max_left _ _))
  exact coprime_mul_succ
    (Nat.succ_le_succ <| le_of_lt ltij)
    (Nat.dvd_factorial (by omega)
      (by simpa only [Nat.succ_sub_succ] using le_of_lt (lt_of_le_of_lt (sub_le j i) hja)))


/-- Gödel's Beta Function. This is similar to `(Encodable.decodeList)[i]`, but it is easier to
prove that it is arithmetically definable. -/
def beta (n i : ℕ) : ℕ := n.unpair.1 % ((i + 1) * n.unpair.2 + 1)


/-- Inverse of Gödel's Beta Function. This is similar to `Encodable.encodeList`, but it is easier
to prove that it is arithmetically definable. -/
def unbeta (l : List ℕ) : ℕ :=
  (chineseRemainderOfFinset (l[·]) (coprimes (l[·])) Finset.univ
        /-
          m : Nat
          l : List Nat
          ⊢ ∀ (i : Fin l.length), Membership.mem Finset.univ i → Ne (Nat.coprimes (fun x …
        -/
    (by simp [coprimes])
        /-
          🎉 no goals
        -/
        /-
          m : Nat
          l : List Nat
          ⊢ (↑Finset.univ).Pairwise (Function.onFun Nat.Coprime (Nat.coprimes fun x => G …
        -/
    (by simpa using Set.pairwise_univ.mpr (pairwise_coprime_coprimes _)) : ℕ).pair
        /-
          🎉 no goals
        -/
  (supOfSeq (l[·]))!


/-- **Gödel's Beta Function Lemma** -/
lemma beta_unbeta_coe (l : List ℕ) (i : Fin l.length) : beta (unbeta l) i = l[i] := by
  simpa [beta, unbeta, coprimes] using mod_eq_of_modEq
    ((chineseRemainderOfFinset (l[·]) (coprimes (l[·])) Finset.univ
      (by simp [coprimes])
      (by simpa using Set.pairwise_univ.mpr (pairwise_coprime_coprimes _))).prop i (by simp))
    (coprimes_lt _ _)


