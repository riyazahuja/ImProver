@[simp]
theorem natCast_mem_center [NonAssocSemiring M] (n : ℕ) : (n : M) ∈ Set.center M where
               /-
                 M : Type u_1
                 inst✝ : NonAssocSemiring M
                 n : Nat
                 x✝ : M
                 ⊢ Eq (HMul.hMul (↑n) x✝) (HMul.hMul x✝ ↑n)
               -/
  comm _ := by rw [Nat.commute_cast]
               /-
                 🎉 no goals
               -/
  left_assoc _ _ := by
    induction n with
    | zero => rw [Nat.cast_zero, zero_mul, zero_mul, zero_mul]
    | succ n ihn => rw [Nat.cast_succ, add_mul, one_mul, ihn, add_mul, add_mul, one_mul]
  mid_assoc _ _ := by
    induction n with
    | zero => rw [Nat.cast_zero, zero_mul, mul_zero, zero_mul]
    | succ n ihn => rw [Nat.cast_succ, add_mul, mul_add, add_mul, ihn, mul_add, one_mul, mul_one]
  right_assoc _ _ := by
    induction n with
    | zero => rw [Nat.cast_zero, mul_zero, mul_zero, mul_zero]
    | succ n ihn => rw [Nat.cast_succ, mul_add, ihn, mul_add, mul_add, mul_one, mul_one]


@[simp]
theorem ofNat_mem_center [NonAssocSemiring M] (n : ℕ) [n.AtLeastTwo] :
    ofNat(n) ∈ Set.center M :=
  natCast_mem_center M n


@[simp]
theorem intCast_mem_center [NonAssocRing M] (n : ℤ) : (n : M) ∈ Set.center M where
               /-
                 M : Type u_1
                 inst✝ : NonAssocRing M
                 n : Int
                 x✝ : M
                 ⊢ Eq (HMul.hMul (↑n) x✝) (HMul.hMul x✝ ↑n)
               -/
  comm _ := by rw [Int.commute_cast]
               /-
                 🎉 no goals
               -/
  left_assoc _ _ := match n with
                    /-
                      M : Type u_1
                      inst✝ : NonAssocRing M
                      n✝ : Int
                      x✝¹ x✝ : M
                      n : Nat
                      ⊢ Eq (HMul.hMul (↑↑n) (HMul.hMul x✝¹ x✝)) (HMul.hMul (HMul.hMul (↑↑n) x✝¹) x✝)
                    -/
    | (n : ℕ) => by rw [Int.cast_natCast, (natCast_mem_center _ n).left_assoc _ _]
                    /-
                      🎉 no goals
                    -/
    | Int.negSucc n => by
      rw [Int.cast_negSucc, Nat.cast_add, Nat.cast_one, neg_add_rev, add_mul, add_mul, add_mul,
        neg_mul, one_mul, neg_mul 1, one_mul, ← neg_mul, add_right_inj, neg_mul,
        (natCast_mem_center _ n).left_assoc _ _, neg_mul, neg_mul]
  mid_assoc _ _ := match n with
                    /-
                      M : Type u_1
                      inst✝ : NonAssocRing M
                      n✝ : Int
                      x✝¹ x✝ : M
                      n : Nat
                      ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ ↑↑n) x✝) (HMul.hMul x✝¹ (HMul.hMul (↑↑n) x✝))
                    -/
    | (n : ℕ) => by rw [Int.cast_natCast, (natCast_mem_center _ n).mid_assoc _ _]
                    /-
                      🎉 no goals
                    -/
    | Int.negSucc n => by
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ ↑(Int.negSucc n)) x✝) (HMul.hMul x✝¹ (HMul.hMul …
        -/
        simp only [Int.cast_negSucc, Nat.cast_add, Nat.cast_one, neg_add_rev]
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ (HAdd.hAdd (-1) (Neg.neg ↑n))) x✝) (HMul.hMul x …
        -/
        rw [add_mul, mul_add, add_mul, mul_add, neg_mul, one_mul]
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul x✝¹ (-1)) x✝) (HMul.hMul (HMul.hMul x✝¹  …
        -/
        rw [neg_mul, mul_neg, mul_one, mul_neg, neg_mul, neg_mul]
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul x✝¹ x✝)) (Neg.neg (HMul.hMul (HMul.hMul x✝ …
        -/
        rw [(natCast_mem_center _ n).mid_assoc _ _]
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HAdd.hAdd (Neg.neg (HMul.hMul x✝¹ x✝)) (Neg.neg (HMul.hMul x✝¹ (HMul.hMu …
        -/
        simp only [mul_neg]
        /-
          🎉 no goals
        -/
  right_assoc _ _ := match n with
                    /-
                      M : Type u_1
                      inst✝ : NonAssocRing M
                      n✝ : Int
                      x✝¹ x✝ : M
                      n : Nat
                      ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) ↑↑n) (HMul.hMul x✝¹ (HMul.hMul x✝ ↑↑n))
                    -/
    | (n : ℕ) => by rw [Int.cast_natCast, (natCast_mem_center _ n).right_assoc _ _]
                    /-
                      🎉 no goals
                    -/
    | Int.negSucc n => by
        /-
          M : Type u_1
          inst✝ : NonAssocRing M
          n✝ : Int
          x✝¹ x✝ : M
          n : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) ↑(Int.negSucc n)) (HMul.hMul x✝¹ (HMul.hMul …
        -/
        simp only [Int.cast_negSucc, Nat.cast_add, Nat.cast_one, neg_add_rev]
        rw [mul_add, mul_add, mul_add, mul_neg, mul_one, mul_neg, mul_neg, mul_one, mul_neg,
          add_right_inj, (natCast_mem_center _ n).right_assoc _ _, mul_neg, mul_neg]


@[simp]
theorem add_mem_center [Distrib M] {a b : M} (ha : a ∈ Set.center M) (hb : b ∈ Set.center M) :
    a + b ∈ Set.center M  where
               /-
                 M : Type u_1
                 inst✝ : Distrib M
                 a b : M
                 ha : Membership.mem (Set.center M) a
                 hb : Membership.mem (Set.center M) b
                 x✝ : M
                 ⊢ Eq (HMul.hMul (HAdd.hAdd a b) x✝) (HMul.hMul x✝ (HAdd.hAdd a b))
               -/
  comm _ := by rw [add_mul, mul_add, ha.comm, hb.comm]
               /-
                 🎉 no goals
               -/
                       /-
                         M : Type u_1
                         inst✝ : Distrib M
                         a b : M
                         ha : Membership.mem (Set.center M) a
                         hb : Membership.mem (Set.center M) b
                         x✝¹ x✝ : M
                         ⊢ Eq (HMul.hMul (HAdd.hAdd a b) (HMul.hMul x✝¹ x✝)) (HMul.hMul (HMul.hMul (HAd …
                       -/
  left_assoc _ _ := by rw [add_mul, ha.left_assoc, hb.left_assoc, ← add_mul, ← add_mul]
                       /-
                         🎉 no goals
                       -/
                      /-
                        M : Type u_1
                        inst✝ : Distrib M
                        a b : M
                        ha : Membership.mem (Set.center M) a
                        hb : Membership.mem (Set.center M) b
                        x✝¹ x✝ : M
                        ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ (HAdd.hAdd a b)) x✝) (HMul.hMul x✝¹ (HMul.hMul  …
                      -/
  mid_assoc _ _ := by rw [mul_add, add_mul, ha.mid_assoc, hb.mid_assoc, ← mul_add, ← add_mul]
                      /-
                        🎉 no goals
                      -/
                        /-
                          M : Type u_1
                          inst✝ : Distrib M
                          a b : M
                          ha : Membership.mem (Set.center M) a
                          hb : Membership.mem (Set.center M) b
                          x✝¹ x✝ : M
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) (HAdd.hAdd a b)) (HMul.hMul x✝¹ (HMul.hMul  …
                        -/
  right_assoc _ _ := by rw [mul_add, ha.right_assoc, hb.right_assoc, ← mul_add, ← mul_add]
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem neg_mem_center [NonUnitalNonAssocRing M] {a : M} (ha : a ∈ Set.center M) :
    -a ∈ Set.center M where
               /-
                 M : Type u_1
                 inst✝ : NonUnitalNonAssocRing M
                 a : M
                 ha : Membership.mem (Set.center M) a
                 x✝ : M
                 ⊢ Eq (HMul.hMul (Neg.neg a) x✝) (HMul.hMul x✝ (Neg.neg a))
               -/
  comm _ := by rw [← neg_mul_comm, ← ha.comm, neg_mul_comm]
               /-
                 🎉 no goals
               -/
                       /-
                         M : Type u_1
                         inst✝ : NonUnitalNonAssocRing M
                         a : M
                         ha : Membership.mem (Set.center M) a
                         x✝¹ x✝ : M
                         ⊢ Eq (HMul.hMul (Neg.neg a) (HMul.hMul x✝¹ x✝)) (HMul.hMul (HMul.hMul (Neg.neg …
                       -/
  left_assoc _ _ := by rw [neg_mul, ha.left_assoc, neg_mul, neg_mul]
                       /-
                         🎉 no goals
                       -/
                      /-
                        M : Type u_1
                        inst✝ : NonUnitalNonAssocRing M
                        a : M
                        ha : Membership.mem (Set.center M) a
                        x✝¹ x✝ : M
                        ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ (Neg.neg a)) x✝) (HMul.hMul x✝¹ (HMul.hMul (Neg …
                      -/
  mid_assoc _ _ := by rw [← neg_mul_comm, ha.mid_assoc, neg_mul_comm, neg_mul]
                      /-
                        🎉 no goals
                      -/
                        /-
                          M : Type u_1
                          inst✝ : NonUnitalNonAssocRing M
                          a : M
                          ha : Membership.mem (Set.center M) a
                          x✝¹ x✝ : M
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) (Neg.neg a)) (HMul.hMul x✝¹ (HMul.hMul x✝ ( …
                        -/
  right_assoc _ _ := by rw [mul_neg, ha.right_assoc, mul_neg, mul_neg]
                        /-
                          🎉 no goals
                        -/


