/-- Distance (absolute value of difference) between natural numbers. -/
def dist (n m : ℕ) :=
  n - m + (m - n)


                                                        /-
                                                          n m : Nat
                                                          ⊢ Eq (n.dist m) (m.dist n)
                                                        -/
theorem dist_comm (n m : ℕ) : dist n m = dist m n := by simp [dist, add_comm]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
                                               /-
                                                 n : Nat
                                                 ⊢ Eq (n.dist n) 0
                                               -/
theorem dist_self (n : ℕ) : dist n n = 0 := by simp [dist, tsub_self]
                                               /-
                                                 🎉 no goals
                                               -/


theorem eq_of_dist_eq_zero {n m : ℕ} (h : dist n m = 0) : n = m :=
  have : n - m = 0 := Nat.eq_zero_of_add_eq_zero_right h
  have : n ≤ m := tsub_eq_zero_iff_le.mp this
  have : m - n = 0 := Nat.eq_zero_of_add_eq_zero_left h
  have : m ≤ n := tsub_eq_zero_iff_le.mp this
  le_antisymm ‹n ≤ m› ‹m ≤ n›


                                                                /-
                                                                  n m : Nat
                                                                  h : Eq n m
                                                                  ⊢ Eq (n.dist m) 0
                                                                -/
theorem dist_eq_zero {n m : ℕ} (h : n = m) : dist n m = 0 := by rw [h, dist_self]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem dist_eq_sub_of_le {n m : ℕ} (h : n ≤ m) : dist n m = m - n := by
  /-
    n m : Nat
    h : LE.le n m
    ⊢ Eq (n.dist m) (HSub.hSub m n)
  -/
  rw [dist, tsub_eq_zero_iff_le.mpr h, zero_add]
  /-
    🎉 no goals
  -/


theorem dist_eq_sub_of_le_right {n m : ℕ} (h : m ≤ n) : dist n m = n - m := by
  /-
    n m : Nat
    h : LE.le m n
    ⊢ Eq (n.dist m) (HSub.hSub n m)
  -/
  rw [dist_comm]; apply dist_eq_sub_of_le h
                  /-
                    🎉 no goals
                  -/


theorem dist_tri_left (n m : ℕ) : m ≤ dist n m + n :=
  le_trans le_tsub_add (add_le_add_right (Nat.le_add_left _ _) _)


                                                          /-
                                                            n m : Nat
                                                            ⊢ LE.le m (HAdd.hAdd n (n.dist m))
                                                          -/
theorem dist_tri_right (n m : ℕ) : m ≤ n + dist n m := by rw [add_comm]; apply dist_tri_left
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                                          /-
                                                            n m : Nat
                                                            ⊢ LE.le n (HAdd.hAdd (n.dist m) m)
                                                          -/
theorem dist_tri_left' (n m : ℕ) : n ≤ dist n m + m := by rw [dist_comm]; apply dist_tri_left
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                           /-
                                                             n m : Nat
                                                             ⊢ LE.le n (HAdd.hAdd m (n.dist m))
                                                           -/
theorem dist_tri_right' (n m : ℕ) : n ≤ m + dist n m := by rw [dist_comm]; apply dist_tri_right
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem dist_zero_right (n : ℕ) : dist n 0 = n :=
  Eq.trans (dist_eq_sub_of_le_right (zero_le n)) (tsub_zero n)


theorem dist_zero_left (n : ℕ) : dist 0 n = n :=
  Eq.trans (dist_eq_sub_of_le (zero_le n)) (tsub_zero n)


theorem dist_add_add_right (n k m : ℕ) : dist (n + k) (m + k) = dist n m :=
  calc
    dist (n + k) (m + k) = n + k - (m + k) + (m + k - (n + k)) := rfl
                                        /-
                                          n k m : Nat
                                          ⊢ Eq (HAdd.hAdd (HSub.hSub (HAdd.hAdd n k) (HAdd.hAdd m k)) (HSub.hSub (HAdd.h …
                                        -/
    _ = n - m + (m + k - (n + k)) := by rw [@add_tsub_add_eq_tsub_right]
                                        /-
                                          🎉 no goals
                                        -/
                              /-
                                n k m : Nat
                                ⊢ Eq (HAdd.hAdd (HSub.hSub n m) (HSub.hSub (HAdd.hAdd m k) (HAdd.hAdd n k))) ( …
                              -/
    _ = n - m + (m - n) := by rw [@add_tsub_add_eq_tsub_right]
                              /-
                                🎉 no goals
                              -/


theorem dist_add_add_left (k n m : ℕ) : dist (k + n) (k + m) = dist n m := by
  /-
    k n m : Nat
    ⊢ Eq ((HAdd.hAdd k n).dist (HAdd.hAdd k m)) (n.dist m)
  -/
  rw [add_comm k n, add_comm k m]; apply dist_add_add_right
                                   /-
                                     🎉 no goals
                                   -/


theorem dist_eq_intro {n m k l : ℕ} (h : n + m = k + l) : dist n k = dist l m :=
  calc
                                          /-
                                            n m k l : Nat
                                            h : Eq (HAdd.hAdd n m) (HAdd.hAdd k l)
                                            ⊢ Eq (n.dist k) ((HAdd.hAdd n m).dist (HAdd.hAdd k m))
                                          -/
    dist n k = dist (n + m) (k + m) := by rw [dist_add_add_right]
                                          /-
                                            🎉 no goals
                                          -/
                                   /-
                                     n m k l : Nat
                                     h : Eq (HAdd.hAdd n m) (HAdd.hAdd k l)
                                     ⊢ Eq ((HAdd.hAdd n m).dist (HAdd.hAdd k m)) ((HAdd.hAdd k l).dist (HAdd.hAdd k …
                                   -/
    _ = dist (k + l) (k + m) := by rw [h]
                                   /-
                                     🎉 no goals
                                   -/
                       /-
                         n m k l : Nat
                         h : Eq (HAdd.hAdd n m) (HAdd.hAdd k l)
                         ⊢ Eq ((HAdd.hAdd k l).dist (HAdd.hAdd k m)) (l.dist m)
                       -/
    _ = dist l m := by rw [dist_add_add_left]
                       /-
                         🎉 no goals
                       -/


theorem dist.triangle_inequality (n m k : ℕ) : dist n k ≤ dist n m + dist m k := by
  have : dist n m + dist m k = n - m + (m - k) + (k - m + (m - n)) := by
    simp [dist, add_comm, add_left_comm, add_assoc]
  /-
    n m k : Nat
    this : Eq (HAdd.hAdd (n.dist m) (m.dist k)) (HAdd.hAdd (HAdd.hAdd (HSub.hSub n …
    ⊢ LE.le (n.dist k) (HAdd.hAdd (n.dist m) (m.dist k))
  -/
  rw [this, dist]
  /-
    n m k : Nat
    this : Eq (HAdd.hAdd (n.dist m) (m.dist k)) (HAdd.hAdd (HAdd.hAdd (HSub.hSub n …
    ⊢ LE.le (HAdd.hAdd (HSub.hSub n k) (HSub.hSub k n)) (HAdd.hAdd (HAdd.hAdd (HSu …
  -/
  exact add_le_add tsub_le_tsub_add_tsub tsub_le_tsub_add_tsub
  /-
    🎉 no goals
  -/


theorem dist_mul_right (n k m : ℕ) : dist (n * k) (m * k) = dist n m * k := by
  /-
    n k m : Nat
    ⊢ Eq ((HMul.hMul n k).dist (HMul.hMul m k)) (HMul.hMul (n.dist m) k)
  -/
  rw [dist, dist, right_distrib, tsub_mul n, tsub_mul m]
  /-
    🎉 no goals
  -/


theorem dist_mul_left (k n m : ℕ) : dist (k * n) (k * m) = k * dist n m := by
  /-
    k n m : Nat
    ⊢ Eq ((HMul.hMul k n).dist (HMul.hMul k m)) (HMul.hMul k (n.dist m))
  -/
  rw [mul_comm k n, mul_comm k m, dist_mul_right, mul_comm]
  /-
    🎉 no goals
  -/


theorem dist_eq_max_sub_min {i j : ℕ} : dist i j = (max i j) - min i j :=
  Or.elim (lt_or_ge i j)
      /-
        i j : Nat
        ⊢ LT.lt i j → Eq (i.dist j) (HSub.hSub (Max.max i j) (Min.min i j))
      -/
  (by intro h; rw [max_eq_right_of_lt h, min_eq_left_of_lt h, dist_eq_sub_of_le (Nat.le_of_lt h)])
               /-
                 🎉 no goals
               -/
      /-
        i j : Nat
        ⊢ GE.ge i j → Eq (i.dist j) (HSub.hSub (Max.max i j) (Min.min i j))
      -/
  (by intro h; rw [max_eq_left h, min_eq_right h, dist_eq_sub_of_le_right h])
               /-
                 🎉 no goals
               -/


theorem dist_succ_succ {i j : Nat} : dist (succ i) (succ j) = dist i j := by
  /-
    i j : Nat
    ⊢ Eq (i.succ.dist j.succ) (i.dist j)
  -/
  simp [dist, succ_sub_succ]
  /-
    🎉 no goals
  -/


theorem dist_pos_of_ne {i j : Nat} : i ≠ j → 0 < dist i j := fun hne =>
  ltByCases i j
                         /-
                           i j : Nat
                           hne : Ne i j
                           h : LT.lt i j
                           ⊢ LT.lt 0 (i.dist j)
                         -/
    (fun h : i < j => by rw [dist_eq_sub_of_le (le_of_lt h)]; apply tsub_pos_of_lt h)
                                                              /-
                                                                🎉 no goals
                                                              -/
                         /-
                           i j : Nat
                           hne : Ne i j
                           h : Eq i j
                           ⊢ LT.lt 0 (i.dist j)
                         -/
    (fun h : i = j => by contradiction) fun h : i > j => by
                         /-
                           🎉 no goals
                         -/
    /-
      i j : Nat
      hne : Ne i j
      h : GT.gt i j
      ⊢ LT.lt 0 (i.dist j)
    -/
    rw [dist_eq_sub_of_le_right (le_of_lt h)]; apply tsub_pos_of_lt h
                                               /-
                                                 🎉 no goals
                                               -/


