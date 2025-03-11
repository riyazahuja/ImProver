instance addCommSemigroup (n : ℕ) : AddCommSemigroup (Fin n) where
                  /-
                    n✝ n : Nat
                    ⊢ ∀ (a b c : Fin n), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd  …
                  -/
  add_assoc := by simp [Fin.ext_iff, add_def, Nat.add_assoc]
                  /-
                    🎉 no goals
                  -/
                 /-
                   n✝ n : Nat
                   ⊢ ∀ (a b : Fin n), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                 -/
  add_comm := by simp [Fin.ext_iff, add_def, Nat.add_comm]
                 /-
                   🎉 no goals
                 -/


instance (n) : AddCommSemigroup (Fin n) where
                  /-
                    n✝ n : Nat
                    ⊢ ∀ (a b c : Fin n), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd  …
                  -/
  add_assoc := by simp [Fin.ext_iff, add_def, Nat.add_assoc]
                  /-
                    🎉 no goals
                  -/
                 /-
                   n✝ n : Nat
                   ⊢ ∀ (a b : Fin n), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                 -/
  add_comm := by simp [Fin.ext_iff, add_def, add_comm]
                 /-
                   🎉 no goals
                 -/


instance addCommMonoid (n : ℕ) [NeZero n] : AddCommMonoid (Fin n) where
  zero_add := Fin.zero_add
  add_zero := Fin.add_zero
  nsmul := nsmulRec
  __ := Fin.addCommSemigroup n


instance instAddMonoidWithOne (n) [NeZero n] : AddMonoidWithOne (Fin n) where
  __ := inferInstanceAs (AddCommMonoid (Fin n))
  natCast i := Fin.ofNat' n i
  natCast_zero := rfl
  natCast_succ _ := Fin.ext (add_mod _ _ _)


instance addCommGroup (n : ℕ) [NeZero n] : AddCommGroup (Fin n) where
  __ := addCommMonoid n
  __ := neg n
  neg_add_cancel := fun ⟨a, ha⟩ ↦
    Fin.ext <| (Nat.mod_add_mod _ _ _).trans <| by
      /-
        n✝ n : Nat
        inst✝ : NeZero n
        x✝ : Fin n
        a : Nat
        ha : LT.lt a n
        ⊢ Eq (HMod.hMod (HAdd.hAdd (HSub.hSub n ↑⟨a, ha⟩) a) n) ↑0
      -/
      rw [Fin.val_zero', Nat.sub_add_cancel, Nat.mod_self]
      /-
        n✝ n : Nat
        inst✝ : NeZero n
        x✝ : Fin n
        a : Nat
        ha : LT.lt a n
        ⊢ LE.le (↑⟨a, ha⟩) n
      -/
      exact le_of_lt ha
                  /-
                    n✝ n : Nat
                    inst✝ : NeZero n
                    x✝¹ x✝ : Fin n
                    a : Nat
                    ha : LT.lt a n
                    b : Nat
                    hb : LT.lt b n
                    ⊢ Eq ↑(HSub.hSub ⟨a, ha⟩ ⟨b, hb⟩) ↑(HAdd.hAdd ⟨a, ha⟩ (Neg.neg ⟨b, hb⟩))
                  -/
      /-
        🎉 no goals
      -/
                  /-
                    🎉 no goals
                  -/
  sub := Fin.sub
  sub_eq_add_neg := fun ⟨a, ha⟩ ⟨b, hb⟩ ↦
    Fin.ext <| by simp [Fin.sub_def, Fin.neg_def, Fin.add_def, Nat.add_comm]
  zsmul := zsmulRec


/-- Note this is more general than `Fin.addCommGroup` as it applies (vacuously) to `Fin 0` too. -/
instance instInvolutiveNeg (n : ℕ) : InvolutiveNeg (Fin n) where
  neg_neg := Nat.casesOn n finZeroElim fun _i ↦ neg_neg


/-- Note this is more general than `Fin.addCommGroup` as it applies (vacuously) to `Fin 0` too. -/
instance instIsCancelAdd (n : ℕ) : IsCancelAdd (Fin n) where
  add_left_cancel := Nat.casesOn n finZeroElim fun _i _ _ _ ↦ add_left_cancel
  add_right_cancel := Nat.casesOn n finZeroElim fun _i _ _ _ ↦ add_right_cancel


/-- Note this is more general than `Fin.addCommGroup` as it applies (vacuously) to `Fin 0` too. -/
instance instAddLeftCancelSemigroup (n : ℕ) : AddLeftCancelSemigroup (Fin n) :=
  { Fin.addCommSemigroup n, Fin.instIsCancelAdd n with }


/-- Note this is more general than `Fin.addCommGroup` as it applies (vacuously) to `Fin 0` too. -/
instance instAddRightCancelSemigroup (n : ℕ) : AddRightCancelSemigroup (Fin n) :=
  { Fin.addCommSemigroup n, Fin.instIsCancelAdd n with }


lemma coe_sub_one (a : Fin (n + 1)) : ↑(a - 1) = if a = 0 then n else a - 1 := by
  /-
    n : Nat
    a : Fin (HAdd.hAdd n 1)
    ⊢ Eq (↑(HSub.hSub a 1)) (ite (Eq a 0) n (HSub.hSub (↑a) 1))
  -/
  cases n
    /-
      case zero
      a : Fin (HAdd.hAdd 0 1)
      ⊢ Eq (↑(HSub.hSub a 1)) (ite (Eq a 0) 0 (HSub.hSub (↑a) 1))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq (↑(HSub.hSub a 1)) (ite (Eq a 0) (HAdd.hAdd n✝ 1) (HSub.hSub (↑a) 1))
  -/
  split_ifs with h
    /-
      case pos
      n✝ : Nat
      a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
      h : Eq a 0
      ⊢ Eq (↑(HSub.hSub a 1)) (HAdd.hAdd n✝ 1)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n✝ : Nat
    a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : Not (Eq a 0)
    ⊢ Eq (↑(HSub.hSub a 1)) (HSub.hSub (↑a) 1)
  -/
  rw [sub_eq_add_neg, val_add_eq_ite, coe_neg_one, if_pos, Nat.add_comm, Nat.add_sub_add_left]
  /-
    case neg.hc
    n✝ : Nat
    a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : Not (Eq a 0)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (↑a) (HAdd.hAdd n✝ 1))
  -/
  conv_rhs => rw [Nat.add_comm]
  /-
    case neg.hc
    n✝ : Nat
    a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : Not (Eq a 0)
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n✝ 1) 1) (HAdd.hAdd (HAdd.hAdd n✝ 1) ↑a)
  -/
  rw [Nat.add_le_add_iff_left, Nat.one_le_iff_ne_zero]
  /-
    case neg.hc
    n✝ : Nat
    a : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    h : Not (Eq a 0)
    ⊢ Ne (↑a) 0
  -/
  rwa [Fin.ext_iff] at h
  /-
    🎉 no goals
  -/


@[simp]
lemma lt_sub_iff {n : ℕ} {a b : Fin n} : a < a - b ↔ a < b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LT.lt a (HSub.hSub a b)) (LT.lt a b)
  -/
  cases' n with n
    /-
      case zero
      a b : Fin 0
      ⊢ Iff (LT.lt a (HSub.hSub a b)) (LT.lt a b)
    -/
  · exact a.elim0
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ⊢ Iff (LT.lt a (HSub.hSub a b)) (LT.lt a b)
  -/
  constructor
    /-
      case succ.mp
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      ⊢ LT.lt a (HSub.hSub a b) → LT.lt a b
    -/
  · contrapose!
    /-
      case succ.mp
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      ⊢ Not (LT.lt a b) → Not (LT.lt a (HSub.hSub a b))
    -/
    intro h
    /-
      case succ.mp
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : Not (LT.lt a b)
      ⊢ Not (LT.lt a (HSub.hSub a b))
    -/
    obtain ⟨l, hl⟩ := Nat.exists_eq_add_of_le (Fin.not_lt.mp h)
    simpa only [Fin.not_lt, le_iff_val_le_val, sub_def, hl, ← Nat.add_assoc, Nat.add_mod_left,
      Nat.mod_eq_of_lt, Nat.sub_add_cancel b.is_lt.le] using
        (le_trans (mod_le _ _) (le_add_left _ _))
    /-
      case succ.mpr
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      ⊢ LT.lt a b → LT.lt a (HSub.hSub a b)
    -/
  · intro h
    /-
      case succ.mpr
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LT.lt a b
      ⊢ LT.lt a (HSub.hSub a b)
    -/
    rw [lt_iff_val_lt_val, sub_def]
    /-
      case succ.mpr
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LT.lt a b
      ⊢ LT.lt ↑a ↑⟨HMod.hMod (HAdd.hAdd (HSub.hSub (HAdd.hAdd n 1) ↑b) ↑a) (HAdd.hAd …
    -/
    simp only
    /-
      case succ.mpr
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LT.lt a b
      ⊢ LT.lt (↑a) (HMod.hMod (HAdd.hAdd (HSub.hSub (HAdd.hAdd n 1) ↑b) ↑a) (HAdd.hA …
    -/
    obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt b.is_lt
    have : n + 1 - b = k + 1 := by
      simp_rw [hk, Nat.add_assoc, Nat.add_sub_cancel_left]
      -- simp_rw because, otherwise, rw tries to rewrite inside `b : Fin (n + 1)`
    /-
      case succ.mpr.intro
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LT.lt a b
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HAdd.hAdd (HAdd.hAdd (↑b) k) 1)
      this : Eq (HSub.hSub (HAdd.hAdd n 1) ↑b) (HAdd.hAdd k 1)
      ⊢ LT.lt (↑a) (HMod.hMod (HAdd.hAdd (HSub.hSub (HAdd.hAdd n 1) ↑b) ↑a) (HAdd.hA …
    -/
    rw [this, Nat.mod_eq_of_lt (hk.ge.trans_lt' ?_), Nat.lt_add_left_iff_pos] <;>
    /-
      case succ.mpr.intro
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LT.lt a b
      k : Nat
      hk : Eq (HAdd.hAdd n 1) (HAdd.hAdd (HAdd.hAdd (↑b) k) 1)
      this : Eq (HSub.hSub (HAdd.hAdd n 1) ↑b) (HAdd.hAdd k 1)
      ⊢ LT.lt 0 (HAdd.hAdd k 1)
    -/
    /-
      🎉 no goals
    -/
    omega
    /-
      🎉 no goals
    -/


@[simp]
lemma sub_le_iff {n : ℕ} {a b : Fin n} : a - b ≤ a ↔ b ≤ a := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LE.le (HSub.hSub a b) a) (LE.le b a)
  -/
  rw [← not_iff_not, Fin.not_le, Fin.not_le, lt_sub_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma lt_one_iff {n : ℕ} (x : Fin (n + 2)) : x < 1 ↔ x = 0 := by
  /-
    n : Nat
    x : Fin (HAdd.hAdd n 2)
    ⊢ Iff (LT.lt x 1) (Eq x 0)
  -/
  simp [lt_iff_val_lt_val, Fin.ext_iff]
  /-
    🎉 no goals
  -/


lemma lt_sub_one_iff {k : Fin (n + 2)} : k < k - 1 ↔ k = 0 := by
  /-
    n : Nat
    k : Fin (HAdd.hAdd n 2)
    ⊢ Iff (LT.lt k (HSub.hSub k 1)) (Eq k 0)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] lemma le_sub_one_iff {k : Fin (n + 1)} : k ≤ k - 1 ↔ k = 0 := by
  /-
    n : Nat
    k : Fin (HAdd.hAdd n 1)
    ⊢ Iff (LE.le k (HSub.hSub k 1)) (Eq k 0)
  -/
  cases n
    /-
      case zero
      k : Fin (HAdd.hAdd 0 1)
      ⊢ Iff (LE.le k (HSub.hSub k 1)) (Eq k 0)
    -/
  · simp [fin_one_eq_zero k]
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    k : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Iff (LE.le k (HSub.hSub k 1)) (Eq k 0)
  -/
  simp only [le_def]
  rw [← lt_sub_one_iff, le_iff_lt_or_eq, val_fin_lt, val_inj, lt_sub_one_iff, or_iff_left_iff_imp,
    eq_comm, sub_eq_iff_eq_add]
  /-
    case succ
    n✝ : Nat
    k : Fin (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq k (HAdd.hAdd k 1) → Eq k 0
  -/
  simp
  /-
    🎉 no goals
  -/


lemma sub_one_lt_iff {k : Fin (n + 1)} : k - 1 < k ↔ 0 < k :=
                      /-
                        n : Nat
                        k : Fin (HAdd.hAdd n 1)
                        ⊢ Iff (Not (LT.lt (HSub.hSub k 1) k)) (Not (LT.lt 0 k))
                      -/
  not_iff_not.1 <| by simp only [lt_def, not_lt, val_fin_le, le_sub_one_iff, le_zero_iff]
                      /-
                        🎉 no goals
                      -/


                                                       /-
                                                         n : Nat
                                                         ⊢ Eq (Neg.neg (Fin.last n)) 1
                                                       -/
@[simp] lemma neg_last (n : ℕ) : -Fin.last n = 1 := by simp [neg_eq_iff_add_eq_zero]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma neg_natCast_eq_one (n : ℕ) : -(n : Fin (n + 1)) = 1 := by
  /-
    n : Nat
    ⊢ Eq (Neg.neg ↑n) 1
  -/
  simp only [natCast_eq_last, neg_last]
  /-
    🎉 no goals
  -/


lemma rev_add (a b : Fin n) : rev (a + b) = rev a - b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (HAdd.hAdd a b).rev (HSub.hSub a.rev b)
  -/
  cases' n
    /-
      case zero
      a b : Fin 0
      ⊢ Eq (HAdd.hAdd a b).rev (HSub.hSub a.rev b)
    -/
  · exact a.elim0
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    a b : Fin (HAdd.hAdd n✝ 1)
    ⊢ Eq (HAdd.hAdd a b).rev (HSub.hSub a.rev b)
  -/
  rw [← last_sub, ← last_sub, sub_add_eq_sub_sub]
  /-
    🎉 no goals
  -/


lemma rev_sub (a b : Fin n) : rev (a - b) = rev a + b := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Eq (HSub.hSub a b).rev (HAdd.hAdd a.rev b)
  -/
  rw [rev_eq_iff, rev_add, rev_rev]
  /-
    🎉 no goals
  -/


lemma add_lt_left_iff {n : ℕ} {a b : Fin n} : a + b < a ↔ rev b < a := by
  /-
    n : Nat
    a b : Fin n
    ⊢ Iff (LT.lt (HAdd.hAdd a b) a) (LT.lt b.rev a)
  -/
  rw [← rev_lt_rev, Iff.comm, ← rev_lt_rev, rev_add, lt_sub_iff, rev_rev]
  /-
    🎉 no goals
  -/


