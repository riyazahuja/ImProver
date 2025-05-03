theorem gcd_greatest {a b d : ℕ} (hda : d ∣ a) (hdb : d ∣ b) (hd : ∀ e : ℕ, e ∣ a → e ∣ b → e ∣ d) :
    d = a.gcd b :=
  (dvd_antisymm (hd _ (gcd_dvd_left a b) (gcd_dvd_right a b)) (dvd_gcd hda hdb)).symm


@[simp]
theorem gcd_add_mul_right_right (m n k : ℕ) : gcd m (n + k * m) = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq (m.gcd (HAdd.hAdd n (HMul.hMul k m))) (m.gcd n)
  -/
  simp [gcd_rec m (n + k * m), gcd_rec m n]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_add_mul_left_right (m n k : ℕ) : gcd m (n + m * k) = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq (m.gcd (HAdd.hAdd n (HMul.hMul m k))) (m.gcd n)
  -/
  simp [gcd_rec m (n + m * k), gcd_rec m n]
  /-
    🎉 no goals
  -/


@[simp]
                                                                                /-
                                                                                  m n k : Nat
                                                                                  ⊢ Eq (m.gcd (HAdd.hAdd (HMul.hMul k m) n)) (m.gcd n)
                                                                                -/
theorem gcd_mul_right_add_right (m n k : ℕ) : gcd m (k * m + n) = gcd m n := by simp [add_comm _ n]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
                                                                               /-
                                                                                 m n k : Nat
                                                                                 ⊢ Eq (m.gcd (HAdd.hAdd (HMul.hMul m k) n)) (m.gcd n)
                                                                               -/
theorem gcd_mul_left_add_right (m n k : ℕ) : gcd m (m * k + n) = gcd m n := by simp [add_comm _ n]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem gcd_add_mul_right_left (m n k : ℕ) : gcd (m + k * n) n = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq ((HAdd.hAdd m (HMul.hMul k n)).gcd n) (m.gcd n)
  -/
  rw [gcd_comm, gcd_add_mul_right_right, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_add_mul_left_left (m n k : ℕ) : gcd (m + n * k) n = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq ((HAdd.hAdd m (HMul.hMul n k)).gcd n) (m.gcd n)
  -/
  rw [gcd_comm, gcd_add_mul_left_right, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_mul_right_add_left (m n k : ℕ) : gcd (k * n + m) n = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq ((HAdd.hAdd (HMul.hMul k n) m).gcd n) (m.gcd n)
  -/
  rw [gcd_comm, gcd_mul_right_add_right, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_mul_left_add_left (m n k : ℕ) : gcd (n * k + m) n = gcd m n := by
  /-
    m n k : Nat
    ⊢ Eq ((HAdd.hAdd (HMul.hMul n k) m).gcd n) (m.gcd n)
  -/
  rw [gcd_comm, gcd_mul_left_add_right, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_add_self_right (m n : ℕ) : gcd m (n + m) = gcd m n :=
               /-
                 m n : Nat
                 ⊢ Eq (m.gcd (HAdd.hAdd n m)) (m.gcd (HAdd.hAdd n (HMul.hMul 1 m)))
               -/
  Eq.trans (by rw [one_mul]) (gcd_add_mul_right_right m n 1)
               /-
                 🎉 no goals
               -/


@[simp]
theorem gcd_add_self_left (m n : ℕ) : gcd (m + n) n = gcd m n := by
  /-
    m n : Nat
    ⊢ Eq ((HAdd.hAdd m n).gcd n) (m.gcd n)
  -/
  rw [gcd_comm, gcd_add_self_right, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                                    /-
                                                                      m n : Nat
                                                                      ⊢ Eq ((HAdd.hAdd m n).gcd m) (n.gcd m)
                                                                    -/
theorem gcd_self_add_left (m n : ℕ) : gcd (m + n) m = gcd n m := by rw [add_comm, gcd_add_self_left]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem gcd_self_add_right (m n : ℕ) : gcd m (m + n) = gcd m n := by
  /-
    m n : Nat
    ⊢ Eq (m.gcd (HAdd.hAdd m n)) (m.gcd n)
  -/
  rw [add_comm, gcd_add_self_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_sub_self_left {m n : ℕ} (h : m ≤ n) : gcd (n - m) m = gcd n m := by
  calc
    gcd (n - m) m = gcd (n - m + m) m := by rw [← gcd_add_self_left (n - m) m]
                _ = gcd n m := by rw [Nat.sub_add_cancel h]


@[simp]
theorem gcd_sub_self_right {m n : ℕ} (h : m ≤ n) : gcd m (n - m) = gcd m n := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Eq (m.gcd (HSub.hSub n m)) (m.gcd n)
  -/
  rw [gcd_comm, gcd_sub_self_left h, gcd_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_self_sub_left {m n : ℕ} (h : m ≤ n) : gcd (n - m) n = gcd m n := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Eq ((HSub.hSub n m).gcd n) (m.gcd n)
  -/
  have := Nat.sub_add_cancel h
  /-
    m n : Nat
    h : LE.le m n
    this : Eq (HAdd.hAdd (HSub.hSub n m) m) n
    ⊢ Eq ((HSub.hSub n m).gcd n) (m.gcd n)
  -/
  rw [gcd_comm m n, ← this, gcd_add_self_left (n - m) m]
  have : gcd (n - m) n = gcd (n - m) m := by
    nth_rw 2 [← Nat.add_sub_cancel' h]
    rw [gcd_add_self_right, gcd_comm]
  /-
    m n : Nat
    h : LE.le m n
    this✝ : Eq (HAdd.hAdd (HSub.hSub n m) m) n
    this : Eq ((HSub.hSub n m).gcd n) ((HSub.hSub n m).gcd m)
    ⊢ Eq ((HSub.hSub (HAdd.hAdd (HSub.hSub n m) m) m).gcd (HAdd.hAdd (HSub.hSub n  …
  -/
  convert this
  /-
    🎉 no goals
  -/


@[simp]
theorem gcd_self_sub_right {m n : ℕ} (h : m ≤ n) : gcd n (n - m) = gcd n m := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Eq (n.gcd (HSub.hSub n m)) (n.gcd m)
  -/
  rw [gcd_comm, gcd_self_sub_left h, gcd_comm]
  /-
    🎉 no goals
  -/


theorem lcm_dvd_mul (m n : ℕ) : lcm m n ∣ m * n :=
  lcm_dvd (dvd_mul_right _ _) (dvd_mul_left _ _)


theorem lcm_dvd_iff {m n k : ℕ} : lcm m n ∣ k ↔ m ∣ k ∧ n ∣ k :=
  ⟨fun h => ⟨(dvd_lcm_left _ _).trans h, (dvd_lcm_right _ _).trans h⟩, and_imp.2 lcm_dvd⟩


theorem lcm_pos {m n : ℕ} : 0 < m → 0 < n → 0 < m.lcm n := by
  /-
    m n : Nat
    ⊢ LT.lt 0 m → LT.lt 0 n → LT.lt 0 (m.lcm n)
  -/
  simp_rw [Nat.pos_iff_ne_zero]
  /-
    m n : Nat
    ⊢ Ne m 0 → Ne n 0 → Ne (m.lcm n) 0
  -/
  exact lcm_ne_zero
  /-
    🎉 no goals
  -/


theorem lcm_mul_left {m n k : ℕ} : (m * n).lcm (m * k) = m * n.lcm k := by
  /-
    m n k : Nat
    ⊢ Eq ((HMul.hMul m n).lcm (HMul.hMul m k)) (HMul.hMul m (n.lcm k))
  -/
  apply dvd_antisymm
    /-
      case a
      m n k : Nat
      ⊢ Dvd.dvd ((HMul.hMul m n).lcm (HMul.hMul m k)) (HMul.hMul m (n.lcm k))
    -/
  · exact lcm_dvd (mul_dvd_mul_left m (dvd_lcm_left n k)) (mul_dvd_mul_left m (dvd_lcm_right n k))
    /-
      🎉 no goals
    -/
    /-
      case a
      m n k : Nat
      ⊢ Dvd.dvd (HMul.hMul m (n.lcm k)) ((HMul.hMul m n).lcm (HMul.hMul m k))
    -/
  · have h : m ∣ lcm (m * n) (m * k) := (dvd_mul_right m n).trans (dvd_lcm_left (m * n) (m * k))
    rw [← dvd_div_iff_mul_dvd h, lcm_dvd_iff, dvd_div_iff_mul_dvd h, dvd_div_iff_mul_dvd h,
      ← lcm_dvd_iff]


theorem lcm_mul_right {m n k : ℕ} : (m * n).lcm (k * n) = m.lcm k * n := by
 /-
   m n k : Nat
   ⊢ Eq ((HMul.hMul m n).lcm (HMul.hMul k n)) (HMul.hMul (m.lcm k) n)
 -/
 rw [mul_comm, mul_comm k n, lcm_mul_left, mul_comm]
 /-
   🎉 no goals
 -/


theorem Coprime.lcm_eq_mul {m n : ℕ} (h : Coprime m n) : lcm m n = m * n := by
  /-
    m n : Nat
    h : m.Coprime n
    ⊢ Eq (m.lcm n) (HMul.hMul m n)
  -/
  rw [← one_mul (lcm m n), ← h.gcd_eq_one, gcd_mul_lcm]
  /-
    🎉 no goals
  -/


theorem Coprime.symmetric : Symmetric Coprime := fun _ _ => Coprime.symm


theorem Coprime.dvd_mul_right {m n k : ℕ} (H : Coprime k n) : k ∣ m * n ↔ k ∣ m :=
  ⟨H.dvd_of_dvd_mul_right, fun h => dvd_mul_of_dvd_left h n⟩


theorem Coprime.dvd_mul_left {m n k : ℕ} (H : Coprime k m) : k ∣ m * n ↔ k ∣ n :=
  ⟨H.dvd_of_dvd_mul_left, fun h => dvd_mul_of_dvd_right h m⟩


@[simp]
theorem coprime_add_self_right {m n : ℕ} : Coprime m (n + m) ↔ Coprime m n := by
  /-
    m n : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd n m)) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_self_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_self_add_right {m n : ℕ} : Coprime m (m + n) ↔ Coprime m n := by
  /-
    m n : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd m n)) (m.Coprime n)
  -/
  rw [add_comm, coprime_add_self_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_add_self_left {m n : ℕ} : Coprime (m + n) n ↔ Coprime m n := by
  /-
    m n : Nat
    ⊢ Iff ((HAdd.hAdd m n).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_self_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_self_add_left {m n : ℕ} : Coprime (m + n) m ↔ Coprime n m := by
  /-
    m n : Nat
    ⊢ Iff ((HAdd.hAdd m n).Coprime m) (n.Coprime m)
  -/
  rw [Coprime, Coprime, gcd_self_add_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_add_mul_right_right (m n k : ℕ) : Coprime m (n + k * m) ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd n (HMul.hMul k m))) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_mul_right_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_add_mul_left_right (m n k : ℕ) : Coprime m (n + m * k) ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd n (HMul.hMul m k))) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_mul_left_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_mul_right_add_right (m n k : ℕ) : Coprime m (k * m + n) ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd (HMul.hMul k m) n)) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_mul_right_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_mul_left_add_right (m n k : ℕ) : Coprime m (m * k + n) ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff (m.Coprime (HAdd.hAdd (HMul.hMul m k) n)) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_mul_left_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_add_mul_right_left (m n k : ℕ) : Coprime (m + k * n) n ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff ((HAdd.hAdd m (HMul.hMul k n)).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_mul_right_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_add_mul_left_left (m n k : ℕ) : Coprime (m + n * k) n ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff ((HAdd.hAdd m (HMul.hMul n k)).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_add_mul_left_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_mul_right_add_left (m n k : ℕ) : Coprime (k * n + m) n ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff ((HAdd.hAdd (HMul.hMul k n) m).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_mul_right_add_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_mul_left_add_left (m n k : ℕ) : Coprime (n * k + m) n ↔ Coprime m n := by
  /-
    m n k : Nat
    ⊢ Iff ((HAdd.hAdd (HMul.hMul n k) m).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_mul_left_add_left]
  /-
    🎉 no goals
  -/


lemma add_coprime_iff_left (h : c ∣ b) : Coprime (a + b) c ↔ Coprime a c := by
  /-
    a b c : Nat
    h : Dvd.dvd c b
    ⊢ Iff ((HAdd.hAdd a b).Coprime c) (a.Coprime c)
  -/
  obtain ⟨n, rfl⟩ := h; simp
                        /-
                          🎉 no goals
                        -/


lemma add_coprime_iff_right (h : c ∣ a) : Coprime (a + b) c ↔ Coprime b c := by
  /-
    a b c : Nat
    h : Dvd.dvd c a
    ⊢ Iff ((HAdd.hAdd a b).Coprime c) (b.Coprime c)
  -/
  obtain ⟨n, rfl⟩ := h; simp
                        /-
                          🎉 no goals
                        -/


lemma coprime_add_iff_left (h : a ∣ c) : Coprime a (b + c) ↔ Coprime a b := by
  /-
    a b c : Nat
    h : Dvd.dvd a c
    ⊢ Iff (a.Coprime (HAdd.hAdd b c)) (a.Coprime b)
  -/
  obtain ⟨n, rfl⟩ := h; simp
                        /-
                          🎉 no goals
                        -/


lemma coprime_add_iff_right (h : a ∣ b) : Coprime a (b + c) ↔ Coprime a c := by
  /-
    a b c : Nat
    h : Dvd.dvd a b
    ⊢ Iff (a.Coprime (HAdd.hAdd b c)) (a.Coprime c)
  -/
  obtain ⟨n, rfl⟩ := h; simp
                        /-
                          🎉 no goals
                        -/

-- TODO: Replace `Nat.Coprime.coprime_dvd_left`

lemma Coprime.of_dvd_left (ha : a₁ ∣ a₂) (h : Coprime a₂ b) : Coprime a₁ b := h.coprime_dvd_left ha

-- TODO: Replace `Nat.Coprime.coprime_dvd_right`

lemma Coprime.of_dvd_right (hb : b₁ ∣ b₂) (h : Coprime a b₂) : Coprime a b₁ :=
  h.coprime_dvd_right hb


lemma Coprime.of_dvd (ha : a₁ ∣ a₂) (hb : b₁ ∣ b₂) (h : Coprime a₂ b₂) : Coprime a₁ b₁ :=
  (h.of_dvd_left ha).of_dvd_right hb


@[simp]
theorem coprime_sub_self_left {m n : ℕ} (h : m ≤ n) : Coprime (n - m) m ↔ Coprime n m := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Iff ((HSub.hSub n m).Coprime m) (n.Coprime m)
  -/
  rw [Coprime, Coprime, gcd_sub_self_left h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_sub_self_right {m n : ℕ} (h : m ≤ n) : Coprime m (n - m) ↔ Coprime m n := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Iff (m.Coprime (HSub.hSub n m)) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_sub_self_right h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_self_sub_left {m n : ℕ} (h : m ≤ n) : Coprime (n - m) n ↔ Coprime m n := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Iff ((HSub.hSub n m).Coprime n) (m.Coprime n)
  -/
  rw [Coprime, Coprime, gcd_self_sub_left h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_self_sub_right {m n : ℕ} (h : m ≤ n) : Coprime n (n - m) ↔ Coprime n m := by
  /-
    m n : Nat
    h : LE.le m n
    ⊢ Iff (n.Coprime (HSub.hSub n m)) (n.Coprime m)
  -/
  rw [Coprime, Coprime, gcd_self_sub_right h]
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_pow_left_iff {n : ℕ} (hn : 0 < n) (a b : ℕ) :
    Nat.Coprime (a ^ n) b ↔ Nat.Coprime a b := by
  /-
    n : Nat
    hn : LT.lt 0 n
    a b : Nat
    ⊢ Iff ((HPow.hPow a n).Coprime b) (a.Coprime b)
  -/
  obtain ⟨n, rfl⟩ := exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
  /-
    case intro
    a b n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Iff ((HPow.hPow a n.succ).Coprime b) (a.Coprime b)
  -/
  rw [Nat.pow_succ, Nat.coprime_mul_iff_left]
  /-
    case intro
    a b n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Iff (And ((HPow.hPow a n).Coprime b) (a.Coprime b)) (a.Coprime b)
  -/
  exact ⟨And.right, fun hab => ⟨hab.pow_left _, hab⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem coprime_pow_right_iff {n : ℕ} (hn : 0 < n) (a b : ℕ) :
    Nat.Coprime a (b ^ n) ↔ Nat.Coprime a b := by
  /-
    n : Nat
    hn : LT.lt 0 n
    a b : Nat
    ⊢ Iff (a.Coprime (HPow.hPow b n)) (a.Coprime b)
  -/
  rw [Nat.coprime_comm, coprime_pow_left_iff hn, Nat.coprime_comm]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     ⊢ Not (Nat.Coprime 0 0)
                                                   -/
theorem not_coprime_zero_zero : ¬Coprime 0 0 := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                                /-
                                                                  n : Nat
                                                                  ⊢ Iff (Nat.Coprime 1 n) True
                                                                -/
theorem coprime_one_left_iff (n : ℕ) : Coprime 1 n ↔ True := by simp [Coprime]
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                                 /-
                                                                   n : Nat
                                                                   ⊢ Iff (n.Coprime 1) True
                                                                 -/
theorem coprime_one_right_iff (n : ℕ) : Coprime n 1 ↔ True := by simp [Coprime]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem gcd_mul_of_coprime_of_dvd {a b c : ℕ} (hac : Coprime a c) (b_dvd_c : b ∣ c) :
    gcd (a * b) c = b := by
  /-
    a b c : Nat
    hac : a.Coprime c
    b_dvd_c : Dvd.dvd b c
    ⊢ Eq ((HMul.hMul a b).gcd c) b
  -/
  rcases exists_eq_mul_left_of_dvd b_dvd_c with ⟨d, rfl⟩
  /-
    case intro
    a b d : Nat
    hac : a.Coprime (HMul.hMul d b)
    b_dvd_c : Dvd.dvd b (HMul.hMul d b)
    ⊢ Eq ((HMul.hMul a b).gcd (HMul.hMul d b)) b
  -/
  rw [gcd_mul_right]
  /-
    case intro
    a b d : Nat
    hac : a.Coprime (HMul.hMul d b)
    b_dvd_c : Dvd.dvd b (HMul.hMul d b)
    ⊢ Eq (HMul.hMul (a.gcd d) b) b
  -/
  convert one_mul b
  /-
    case h.e'_2.h.e'_5
    a b d : Nat
    hac : a.Coprime (HMul.hMul d b)
    b_dvd_c : Dvd.dvd b (HMul.hMul d b)
    ⊢ Eq (a.gcd d) 1
  -/
  exact Coprime.coprime_mul_right_right hac
  /-
    🎉 no goals
  -/


theorem Coprime.eq_of_mul_eq_zero {m n : ℕ} (h : m.Coprime n) (hmn : m * n = 0) :
    m = 0 ∧ n = 1 ∨ m = 1 ∧ n = 0 :=
  (Nat.mul_eq_zero.mp hmn).imp (fun hm => ⟨hm, n.coprime_zero_left.mp <| hm ▸ h⟩) fun hn =>
    let eq := hn ▸ h.symm
    ⟨m.coprime_zero_left.mp <| eq, hn⟩


/-- Represent a divisor of `m * n` as a product of a divisor of `m` and a divisor of `n`.

See `exists_dvd_and_dvd_of_dvd_mul` for the more general but less constructive version for other
`GCDMonoid`s. -/
def prodDvdAndDvdOfDvdProd {m n k : ℕ} (H : k ∣ m * n) :
    { d : { m' // m' ∣ m } × { n' // n' ∣ n } // k = d.1 * d.2 } := by
  cases h0 : gcd k m with
  | zero =>
    obtain rfl : k = 0 := eq_zero_of_gcd_eq_zero_left h0
    obtain rfl : m = 0 := eq_zero_of_gcd_eq_zero_right h0
    exact ⟨⟨⟨0, dvd_refl 0⟩, ⟨n, dvd_refl n⟩⟩, (zero_mul n).symm⟩
  | succ tmp =>
    have hpos : 0 < gcd k m := h0.symm ▸ Nat.zero_lt_succ _; clear h0 tmp
    have hd : gcd k m * (k / gcd k m) = k := Nat.mul_div_cancel' (gcd_dvd_left k m)
    refine ⟨⟨⟨gcd k m, gcd_dvd_right k m⟩, ⟨k / gcd k m, ?_⟩⟩, hd.symm⟩
    apply Nat.dvd_of_mul_dvd_mul_left hpos
    rw [hd, ← gcd_mul_right]
    exact dvd_gcd (dvd_mul_right _ _) H


theorem dvd_mul {x m n : ℕ} : x ∣ m * n ↔ ∃ y z, y ∣ m ∧ z ∣ n ∧ y * z = x := by
  /-
    x m n : Nat
    ⊢ Iff (Dvd.dvd x (HMul.hMul m n)) (Exists fun y => Exists fun z => And (Dvd.dv …
  -/
  constructor
    /-
      case mp
      x m n : Nat
      ⊢ Dvd.dvd x (HMul.hMul m n) → Exists fun y => Exists fun z => And (Dvd.dvd y m …
    -/
  · intro h
    /-
      case mp
      x m n : Nat
      h : Dvd.dvd x (HMul.hMul m n)
      ⊢ Exists fun y => Exists fun z => And (Dvd.dvd y m) (And (Dvd.dvd z n) (Eq (HM …
    -/
    obtain ⟨⟨⟨y, hy⟩, ⟨z, hz⟩⟩, rfl⟩ := prod_dvd_and_dvd_of_dvd_prod h
    /-
      case mp.mk.mk.mk.mk
      m n y : Nat
      hy : Dvd.dvd y m
      z : Nat
      hz : Dvd.dvd z n
      h : Dvd.dvd (HMul.hMul ↑{ fst := ⟨y, hy⟩, snd := ⟨z, hz⟩ }.1 ↑{ fst := ⟨y, hy⟩ …
      ⊢ Exists fun y_1 => Exists fun z_1 => And (Dvd.dvd y_1 m) (And (Dvd.dvd z_1 n) …
    -/
    exact ⟨y, z, hy, hz, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x m n : Nat
      ⊢ (Exists fun y => Exists fun z => And (Dvd.dvd y m) (And (Dvd.dvd z n) (Eq (H …
    -/
  · rintro ⟨y, z, hy, hz, rfl⟩
    /-
      case mpr.intro.intro.intro.intro
      m n y z : Nat
      hy : Dvd.dvd y m
      hz : Dvd.dvd z n
      ⊢ Dvd.dvd (HMul.hMul y z) (HMul.hMul m n)
    -/
    exact mul_dvd_mul hy hz
    /-
      🎉 no goals
    -/


theorem pow_dvd_pow_iff {a b n : ℕ} (n0 : n ≠ 0) : a ^ n ∣ b ^ n ↔ a ∣ b := by
  /-
    a b n : Nat
    n0 : Ne n 0
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)) (Dvd.dvd a b)
  -/
  refine ⟨fun h => ?_, fun h => pow_dvd_pow_of_dvd h _⟩
  /-
    a b n : Nat
    n0 : Ne n 0
    h : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    ⊢ Dvd.dvd a b
  -/
  rcases Nat.eq_zero_or_pos (gcd a b) with g0 | g0
    /-
      case inl
      a b n : Nat
      n0 : Ne n 0
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
      g0 : Eq (a.gcd b) 0
      ⊢ Dvd.dvd a b
    -/
  · simp [eq_zero_of_gcd_eq_zero_right g0]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b n : Nat
    n0 : Ne n 0
    h : Dvd.dvd (HPow.hPow a n) (HPow.hPow b n)
    g0 : GT.gt (a.gcd b) 0
    ⊢ Dvd.dvd a b
  -/
  rcases exists_coprime' g0 with ⟨g, a', b', g0', co, rfl, rfl⟩
  /-
    case inr.intro.intro.intro.intro.intro.intro
    n : Nat
    n0 : Ne n 0
    g a' b' : Nat
    g0' : LT.lt 0 g
    co : a'.Coprime b'
    h : Dvd.dvd (HPow.hPow (HMul.hMul a' g) n) (HPow.hPow (HMul.hMul b' g) n)
    g0 : GT.gt ((HMul.hMul a' g).gcd (HMul.hMul b' g)) 0
    ⊢ Dvd.dvd (HMul.hMul a' g) (HMul.hMul b' g)
  -/
  rw [mul_pow, mul_pow] at h
  /-
    case inr.intro.intro.intro.intro.intro.intro
    n : Nat
    n0 : Ne n 0
    g a' b' : Nat
    g0' : LT.lt 0 g
    co : a'.Coprime b'
    h : Dvd.dvd (HMul.hMul (HPow.hPow a' n) (HPow.hPow g n)) (HMul.hMul (HPow.hPow …
    g0 : GT.gt ((HMul.hMul a' g).gcd (HMul.hMul b' g)) 0
    ⊢ Dvd.dvd (HMul.hMul a' g) (HMul.hMul b' g)
  -/
  replace h := Nat.dvd_of_mul_dvd_mul_right (Nat.pow_pos g0') h
  /-
    case inr.intro.intro.intro.intro.intro.intro
    n : Nat
    n0 : Ne n 0
    g a' b' : Nat
    g0' : LT.lt 0 g
    co : a'.Coprime b'
    g0 : GT.gt ((HMul.hMul a' g).gcd (HMul.hMul b' g)) 0
    h : Dvd.dvd (HPow.hPow a' n) (HPow.hPow b' n)
    ⊢ Dvd.dvd (HMul.hMul a' g) (HMul.hMul b' g)
  -/
  have := pow_dvd_pow a' <| Nat.pos_of_ne_zero n0
  /-
    case inr.intro.intro.intro.intro.intro.intro
    n : Nat
    n0 : Ne n 0
    g a' b' : Nat
    g0' : LT.lt 0 g
    co : a'.Coprime b'
    g0 : GT.gt ((HMul.hMul a' g).gcd (HMul.hMul b' g)) 0
    h : Dvd.dvd (HPow.hPow a' n) (HPow.hPow b' n)
    this : Dvd.dvd (HPow.hPow a' (Nat.succ 0)) (HPow.hPow a' n)
    ⊢ Dvd.dvd (HMul.hMul a' g) (HMul.hMul b' g)
  -/
  rw [pow_one, (co.pow n n).eq_one_of_dvd h] at this
  /-
    case inr.intro.intro.intro.intro.intro.intro
    n : Nat
    n0 : Ne n 0
    g a' b' : Nat
    g0' : LT.lt 0 g
    co : a'.Coprime b'
    g0 : GT.gt ((HMul.hMul a' g).gcd (HMul.hMul b' g)) 0
    h : Dvd.dvd (HPow.hPow a' n) (HPow.hPow b' n)
    this : Dvd.dvd a' 1
    ⊢ Dvd.dvd (HMul.hMul a' g) (HMul.hMul b' g)
  -/
  simp [eq_one_of_dvd_one this]
  /-
    🎉 no goals
  -/


theorem coprime_iff_isRelPrime {m n : ℕ} : m.Coprime n ↔ IsRelPrime m n := by
  /-
    m n : Nat
    ⊢ Iff (m.Coprime n) (IsRelPrime m n)
  -/
  simp_rw [coprime_iff_gcd_eq_one, IsRelPrime, ← and_imp, ← dvd_gcd_iff, isUnit_iff_dvd_one]
  /-
    m n : Nat
    ⊢ Iff (Eq (m.gcd n) 1) (∀ ⦃d : Nat⦄, Dvd.dvd d (m.gcd n) → Dvd.dvd d 1)
  -/
  exact ⟨fun h _ ↦ (h ▸ ·), (dvd_one.mp <| · dvd_rfl)⟩
  /-
    🎉 no goals
  -/


/-- If `k:ℕ` divides coprime `a` and `b` then `k = 1` -/
theorem eq_one_of_dvd_coprimes {a b k : ℕ} (h_ab_coprime : Coprime a b) (hka : k ∣ a)
    (hkb : k ∣ b) : k = 1 :=
  dvd_one.mp (isUnit_iff_dvd_one.mp <| coprime_iff_isRelPrime.mp h_ab_coprime hka hkb)


theorem Coprime.mul_add_mul_ne_mul {m n a b : ℕ} (cop : Coprime m n) (ha : a ≠ 0) (hb : b ≠ 0) :
    a * m + b * n ≠ m * n := by
  /-
    m n a b : Nat
    cop : m.Coprime n
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Ne (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b n)) (HMul.hMul m n)
  -/
  intro h
  obtain ⟨x, rfl⟩ : n ∣ a :=
    cop.symm.dvd_of_dvd_mul_right
      ((Nat.dvd_add_iff_left (Nat.dvd_mul_left n b)).mpr
        ((congr_arg _ h).mpr (Nat.dvd_mul_left n m)))
  obtain ⟨y, rfl⟩ : m ∣ b :=
    cop.dvd_of_dvd_mul_right
      ((Nat.dvd_add_iff_right (Nat.dvd_mul_left m (n * x))).mpr
        ((congr_arg _ h).mpr (Nat.dvd_mul_right m n)))
  /-
    case intro.intro
    m n : Nat
    cop : m.Coprime n
    x : Nat
    ha : Ne (HMul.hMul n x) 0
    y : Nat
    hb : Ne (HMul.hMul m y) 0
    h : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul n x) m) (HMul.hMul (HMul.hMul m y) n)) …
    ⊢ False
  -/
  rw [mul_comm, mul_ne_zero_iff, ← one_le_iff_ne_zero] at ha hb
  /-
    case intro.intro
    m n : Nat
    cop : m.Coprime n
    x : Nat
    ha : And (LE.le 1 x) (Ne n 0)
    y : Nat
    hb : And (LE.le 1 y) (Ne m 0)
    h : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul n x) m) (HMul.hMul (HMul.hMul m y) n)) …
    ⊢ False
  -/
  refine mul_ne_zero hb.2 ha.2 (eq_zero_of_mul_eq_self_left (ne_of_gt (add_le_add ha.1 hb.1)) ?_)
  /-
    case intro.intro
    m n : Nat
    cop : m.Coprime n
    x : Nat
    ha : And (LE.le 1 x) (Ne n 0)
    y : Nat
    hb : And (LE.le 1 y) (Ne m 0)
    h : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul n x) m) (HMul.hMul (HMul.hMul m y) n)) …
    ⊢ Eq (HMul.hMul (HAdd.hAdd x y) (HMul.hMul m n)) (HMul.hMul m n)
  -/
  rw [← mul_assoc, ← h, Nat.add_mul, Nat.add_mul, mul_comm _ n, ← mul_assoc, mul_comm y]
  /-
    🎉 no goals
  -/


theorem dvd_gcd_mul_iff_dvd_mul : x ∣ gcd x n * m ↔ x ∣ n * m := by
  /-
    x n m : Nat
    ⊢ Iff (Dvd.dvd x (HMul.hMul (x.gcd n) m)) (Dvd.dvd x (HMul.hMul n m))
  -/
  refine ⟨(·.trans <| mul_dvd_mul_right (x.gcd_dvd_right n) m), fun ⟨y, hy⟩ ↦ ?_⟩
  /-
    x n m : Nat
    x✝ : Dvd.dvd x (HMul.hMul n m)
    y : Nat
    hy : Eq (HMul.hMul n m) (HMul.hMul x y)
    ⊢ Dvd.dvd x (HMul.hMul (x.gcd n) m)
  -/
  rw [← gcd_mul_right, hy, gcd_mul_left]
  /-
    x n m : Nat
    x✝ : Dvd.dvd x (HMul.hMul n m)
    y : Nat
    hy : Eq (HMul.hMul n m) (HMul.hMul x y)
    ⊢ Dvd.dvd x (HMul.hMul x (m.gcd y))
  -/
  exact dvd_mul_right x (gcd m y)
  /-
    🎉 no goals
  -/


theorem dvd_mul_gcd_iff_dvd_mul : x ∣ n * gcd x m ↔ x ∣ n * m := by
  /-
    x n m : Nat
    ⊢ Iff (Dvd.dvd x (HMul.hMul n (x.gcd m))) (Dvd.dvd x (HMul.hMul n m))
  -/
  rw [mul_comm, dvd_gcd_mul_iff_dvd_mul, mul_comm]
  /-
    🎉 no goals
  -/


theorem dvd_gcd_mul_gcd_iff_dvd_mul : x ∣ gcd x n * gcd x m ↔ x ∣ n * m := by
  /-
    x n m : Nat
    ⊢ Iff (Dvd.dvd x (HMul.hMul (x.gcd n) (x.gcd m))) (Dvd.dvd x (HMul.hMul n m))
  -/
  rw [dvd_gcd_mul_iff_dvd_mul, dvd_mul_gcd_iff_dvd_mul]
  /-
    🎉 no goals
  -/


theorem gcd_mul_gcd_eq_iff_dvd_mul_of_coprime (hcop : Coprime n m) :
    gcd x n * gcd x m = x ↔ x ∣ n * m := by
  /-
    x n m : Nat
    hcop : n.Coprime m
    ⊢ Iff (Eq (HMul.hMul (x.gcd n) (x.gcd m)) x) (Dvd.dvd x (HMul.hMul n m))
  -/
  refine ⟨fun h ↦ ?_, (dvd_antisymm ?_ <| dvd_gcd_mul_gcd_iff_dvd_mul.mpr ·)⟩
  /-
    case refine_1
    x n m : Nat
    hcop : n.Coprime m
    h : Eq (HMul.hMul (x.gcd n) (x.gcd m)) x
    ⊢ Dvd.dvd x (HMul.hMul n m)
  -/
                                       /-
                                         🎉 no goals
                                       -/
  refine h ▸ Nat.mul_dvd_mul ?_ ?_ <;> exact x.gcd_dvd_right _
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case refine_2
    x n m : Nat
    hcop : n.Coprime m
    x✝ : Dvd.dvd x (HMul.hMul n m)
    ⊢ Dvd.dvd (HMul.hMul (x.gcd n) (x.gcd m)) x
  -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  refine (hcop.gcd_both x x).mul_dvd_of_dvd_of_dvd ?_ ?_ <;> exact x.gcd_dvd_left _
                                                             /-
                                                               🎉 no goals
                                                             -/


