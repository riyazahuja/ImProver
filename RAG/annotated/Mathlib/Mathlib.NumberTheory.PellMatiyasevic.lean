/-- The property of being a solution to the Pell equation, expressed
  as a property of elements of `ℤ√d`. -/
def IsPell : ℤ√d → Prop
  | ⟨x, y⟩ => x * x - d * y * y = 1


theorem isPell_norm : ∀ {b : ℤ√d}, IsPell b ↔ b * star b = 1
                 /-
                   d x y : Int
                   ⊢ Iff (Pell.IsPell { re := x, im := y }) (Eq (HMul.hMul { re := x, im := y } ( …
                 -/
  | ⟨x, y⟩ => by simp [Zsqrtd.ext_iff, IsPell, mul_comm]; ring_nf
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem isPell_iff_mem_unitary : ∀ {b : ℤ√d}, IsPell b ↔ b ∈ unitary (ℤ√d)
                 /-
                   d x y : Int
                   ⊢ Iff (Pell.IsPell { re := x, im := y }) (Membership.mem (unitary (Zsqrtd d))  …
                 -/
  | ⟨x, y⟩ => by rw [unitary.mem_iff, isPell_norm, mul_comm (star _), and_self_iff]
                 /-
                   🎉 no goals
                 -/


theorem isPell_mul {b c : ℤ√d} (hb : IsPell b) (hc : IsPell c) : IsPell (b * c) :=
  isPell_norm.2 (by simp [mul_comm, mul_left_comm c, mul_assoc,
    star_mul, isPell_norm.1 hb, isPell_norm.1 hc])


theorem isPell_star : ∀ {b : ℤ√d}, IsPell b ↔ IsPell (star b)
                 /-
                   d x y : Int
                   ⊢ Iff (Pell.IsPell { re := x, im := y }) (Pell.IsPell (Star.star { re := x, im …
                 -/
  | ⟨x, y⟩ => by simp [IsPell, Zsqrtd.star_mk]
                 /-
                   🎉 no goals
                 -/


private def d (_a1 : 1 < a) :=
  a * a - 1


@[simp]
theorem d_pos : 0 < d a1 :=
                                                  /-
                                                    a : Nat
                                                    a1 : LT.lt 1 a
                                                    ⊢ LT.lt 0 1
                                                  -/
  tsub_pos_of_lt (mul_lt_mul a1 (le_of_lt a1) (by decide) (Nat.zero_le _) : 1 * 1 < a * a)
                                                  /-
                                                    🎉 no goals
                                                  -/

-- TODO(lint): Fix double namespace issue

/-- The Pell sequences, i.e. the sequence of integer solutions to `x ^ 2 - d * y ^ 2 = 1`, where
`d = a ^ 2 - 1`, defined together in mutual recursion. -/
--@[nolint dup_namespace]
def pell : ℕ → ℕ × ℕ
  -- Porting note: used pattern matching because `Nat.recOn` is noncomputable
  | 0 => (1, 0)
  | n+1 => ((pell n).1 * a + d a1 * (pell n).2, (pell n).1 + (pell n).2 * a)


/-- The Pell `x` sequence. -/
def xn (n : ℕ) : ℕ :=
  (pell a1 n).1


/-- The Pell `y` sequence. -/
def yn (n : ℕ) : ℕ :=
  (pell a1 n).2


@[simp]
theorem pell_val (n : ℕ) : pell a1 n = (xn a1 n, yn a1 n) :=
  show pell a1 n = ((pell a1 n).1, (pell a1 n).2) from
    match pell a1 n with
    | (_, _) => rfl


@[simp]
theorem xn_zero : xn a1 0 = 1 :=
  rfl


@[simp]
theorem yn_zero : yn a1 0 = 0 :=
  rfl


@[simp]
theorem xn_succ (n : ℕ) : xn a1 (n + 1) = xn a1 n * a + d a1 * yn a1 n :=
  rfl


@[simp]
theorem yn_succ (n : ℕ) : yn a1 (n + 1) = xn a1 n + yn a1 n * a :=
  rfl


                                   /-
                                     a : Nat
                                     a1 : LT.lt 1 a
                                     ⊢ Eq (Pell.xn a1 1) a
                                   -/
theorem xn_one : xn a1 1 = a := by simp
                                   /-
                                     🎉 no goals
                                   -/


                                   /-
                                     a : Nat
                                     a1 : LT.lt 1 a
                                     ⊢ Eq (Pell.yn a1 1) 1
                                   -/
theorem yn_one : yn a1 1 = 1 := by simp
                                   /-
                                     🎉 no goals
                                   -/


/-- The Pell `x` sequence, considered as an integer sequence. -/
def xz (n : ℕ) : ℤ :=
  xn a1 n


/-- The Pell `y` sequence, considered as an integer sequence. -/
def yz (n : ℕ) : ℤ :=
  yn a1 n


/-- The element `a` such that `d = a ^ 2 - 1`, considered as an integer. -/
def az (a : ℕ) : ℤ :=
  a


include a1 in
theorem asq_pos : 0 < a * a :=
  le_trans (le_of_lt a1)
        /-
          a : Nat
          a1 : LT.lt 1 a
          ⊢ LE.le a (HMul.hMul a a)
        -/
    (by have := @Nat.mul_le_mul_left 1 a a (le_of_lt a1); rwa [mul_one] at this)
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem dz_val : ↑(d a1) = az a * az a - 1 :=
  have : 1 ≤ a * a := asq_pos a1
     /-
       a : Nat
       a1 : LT.lt 1 a
       this : LE.le 1 (HMul.hMul a a)
       ⊢ Eq (↑(Pell.d a1)) (HSub.hSub (HMul.hMul (Pell.az a) (Pell.az a)) 1)
     -/
  by rw [Pell.d, Int.ofNat_sub this]; rfl
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem xz_succ (n : ℕ) : (xz a1 (n + 1)) = xz a1 n * az a + d a1 * yz a1 n :=
  rfl


@[simp]
theorem yz_succ (n : ℕ) : yz a1 (n + 1) = xz a1 n + yz a1 n * az a :=
  rfl


/-- The Pell sequence can also be viewed as an element of `ℤ√d` -/
def pellZd (n : ℕ) : ℤ√(d a1) :=
  ⟨xn a1 n, yn a1 n⟩


@[simp]
theorem pellZd_re (n : ℕ) : (pellZd a1 n).re = xn a1 n :=
  rfl


@[simp]
theorem pellZd_im (n : ℕ) : (pellZd a1 n).im = yn a1 n :=
  rfl


theorem isPell_nat {x y : ℕ} : IsPell (⟨x, y⟩ : ℤ√(d a1)) ↔ x * x - d a1 * y * y = 1 :=
  ⟨fun h =>
    (Nat.cast_inj (R := ℤ)).1
          /-
            a : Nat
            a1 : LT.lt 1 a
            x y : Nat
            h : Pell.IsPell { re := ↑x, im := ↑y }
            ⊢ Eq ↑(HSub.hSub (HMul.hMul x x) (HMul.hMul (HMul.hMul (Pell.d a1) y) y)) ↑1
          -/
      (by rw [Int.ofNat_sub (Int.le_of_ofNat_le_ofNat <| Int.le.intro_sub _ h)]; exact h),
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    fun h =>
    show ((x * x : ℕ) - (d a1 * y * y : ℕ) : ℤ) = 1 by
      /-
        a : Nat
        a1 : LT.lt 1 a
        x y : Nat
        h : Eq (HSub.hSub (HMul.hMul x x) (HMul.hMul (HMul.hMul (Pell.d a1) y) y)) 1
        ⊢ Eq (HSub.hSub ↑(HMul.hMul x x) ↑(HMul.hMul (HMul.hMul (Pell.d a1) y) y)) 1
      -/
      rw [← Int.ofNat_sub <| le_of_lt <| Nat.lt_of_sub_eq_succ h, h]; rfl⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
                                                                             /-
                                                                               a : Nat
                                                                               a1 : LT.lt 1 a
                                                                               n : Nat
                                                                               ⊢ Eq (Pell.pellZd a1 (HAdd.hAdd n 1)) (HMul.hMul (Pell.pellZd a1 n) { re := ↑a …
                                                                             -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
theorem pellZd_succ (n : ℕ) : pellZd a1 (n + 1) = pellZd a1 n * ⟨a, 1⟩ := by ext <;> simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem isPell_one : IsPell (⟨a, 1⟩ : ℤ√(d a1)) :=
                                         /-
                                           a : Nat
                                           a1 : LT.lt 1 a
                                           ⊢ Eq (HSub.hSub (HMul.hMul (Pell.az a) (Pell.az a)) (HMul.hMul (HMul.hMul (↑(P …
                                         -/
  show az a * az a - d a1 * 1 * 1 = 1 by simp [dz_val]
                                         /-
                                           🎉 no goals
                                         -/


theorem isPell_pellZd : ∀ n : ℕ, IsPell (pellZd a1 n)
  | 0 => rfl
  | n + 1 => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      ⊢ Pell.IsPell (Pell.pellZd a1 (HAdd.hAdd n 1))
    -/
    let o := isPell_one a1
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      o : Pell.IsPell { re := ↑a, im := 1 } := Pell.isPell_one a1
      ⊢ Pell.IsPell (Pell.pellZd a1 (HAdd.hAdd n 1))
    -/
    simpa using Pell.isPell_mul (isPell_pellZd n) o
    /-
      🎉 no goals
    -/


@[simp]
theorem pell_eqz (n : ℕ) : xz a1 n * xz a1 n - d a1 * yz a1 n * yz a1 n = 1 :=
  isPell_pellZd a1 n


@[simp]
theorem pell_eq (n : ℕ) : xn a1 n * xn a1 n - d a1 * yn a1 n * yn a1 n = 1 :=
  let pn := pell_eqz a1 n
  have h : (↑(xn a1 n * xn a1 n) : ℤ) - ↑(d a1 * yn a1 n * yn a1 n) = 1 := by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      pn : Eq (HSub.hSub (HMul.hMul (Pell.xz a1 n) (Pell.xz a1 n)) (HMul.hMul (HMul. …
      ⊢ Eq (HSub.hSub ↑(HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) ↑(HMul.hMul (HMul.h …
    -/
    repeat' rw [Int.ofNat_mul]; exact pn
    /-
      🎉 no goals
    -/
  have hl : d a1 * yn a1 n * yn a1 n ≤ xn a1 n * xn a1 n :=
    Nat.cast_le.1 <| Int.le.intro _ <| add_eq_of_eq_sub' <| Eq.symm h
                                /-
                                  a : Nat
                                  a1 : LT.lt 1 a
                                  n : Nat
                                  pn : Eq (HSub.hSub (HMul.hMul (Pell.xz a1 n) (Pell.xz a1 n)) (HMul.hMul (HMul. …
                                  h : Eq (HSub.hSub ↑(HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) ↑(HMul.hMul (HMul …
                                  hl : LE.le (HMul.hMul (HMul.hMul (Pell.d a1) (Pell.yn a1 n)) (Pell.yn a1 n)) ( …
                                  ⊢ Eq ↑(HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul.hM …
                                -/
  (Nat.cast_inj (R := ℤ)).1 (by rw [Int.ofNat_sub hl]; exact h)
                                                       /-
                                                         🎉 no goals
                                                       -/


instance dnsq : Zsqrtd.Nonsquare (d a1) :=
  ⟨fun n h =>
                                   /-
                                     a : Nat
                                     a1 : LT.lt 1 a
                                     n : Nat
                                     h : Eq (Pell.d a1) (HMul.hMul n n)
                                     ⊢ Eq (HAdd.hAdd (HMul.hMul n n) 1) (HMul.hMul a a)
                                   -/
    have : n * n + 1 = a * a := by rw [← h]; exact Nat.succ_pred_eq_of_pos (asq_pos a1)
                                             /-
                                               🎉 no goals
                                             -/
                                                          /-
                                                            a : Nat
                                                            a1 : LT.lt 1 a
                                                            n : Nat
                                                            h : Eq (Pell.d a1) (HMul.hMul n n)
                                                            this : Eq (HAdd.hAdd (HMul.hMul n n) 1) (HMul.hMul a a)
                                                            ⊢ LT.lt (HMul.hMul n n) (HMul.hMul a a)
                                                          -/
    have na : n < a := Nat.mul_self_lt_mul_self_iff.1 (by rw [← this]; exact Nat.lt_succ_self _)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                               /-
                                                 a : Nat
                                                 a1 : LT.lt 1 a
                                                 n : Nat
                                                 h : Eq (Pell.d a1) (HMul.hMul n n)
                                                 this : Eq (HAdd.hAdd (HMul.hMul n n) 1) (HMul.hMul a a)
                                                 na : LT.lt n a
                                                 ⊢ LE.le (HMul.hMul (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (HAdd.hAdd (HMul.hMul n n) …
                                               -/
    have : (n + 1) * (n + 1) ≤ n * n + 1 := by rw [this]; exact Nat.mul_self_le_mul_self na
                                                          /-
                                                            🎉 no goals
                                                          -/
    have : n + n ≤ 0 :=
                                                      /-
                                                        a : Nat
                                                        a1 : LT.lt 1 a
                                                        n : Nat
                                                        h : Eq (Pell.d a1) (HMul.hMul n n)
                                                        this✝ : Eq (HAdd.hAdd (HMul.hMul n n) 1) (HMul.hMul a a)
                                                        na : LT.lt n a
                                                        this : LE.le (HMul.hMul (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (HAdd.hAdd (HMul.hMul …
                                                        ⊢ LE.le (HAdd.hAdd (HAdd.hAdd n n) (HAdd.hAdd (HMul.hMul n n) 1)) (HAdd.hAdd 0 …
                                                      -/
      @Nat.le_of_add_le_add_right _ (n * n + 1) _ (by ring_nf at this ⊢; assumption)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    Nat.ne_of_gt (d_pos a1) <| by
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        h : Eq (Pell.d a1) (HMul.hMul n n)
        this✝¹ : Eq (HAdd.hAdd (HMul.hMul n n) 1) (HMul.hMul a a)
        na : LT.lt n a
        this✝ : LE.le (HMul.hMul (HAdd.hAdd n 1) (HAdd.hAdd n 1)) (HAdd.hAdd (HMul.hMu …
        this : LE.le (HAdd.hAdd n n) 0
        ⊢ Eq (Pell.d a1) 0
      -/
      rwa [Nat.eq_zero_of_le_zero ((Nat.le_add_left _ _).trans this)] at h⟩
      /-
        🎉 no goals
      -/


theorem xn_ge_a_pow : ∀ n : ℕ, a ^ n ≤ xn a1 n
  | 0 => le_refl 1
  | n + 1 => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      ⊢ LE.le (HPow.hPow a (HAdd.hAdd n 1)) (Pell.xn a1 (HAdd.hAdd n 1))
    -/
    simp only [_root_.pow_succ, xn_succ]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      ⊢ LE.le (HMul.hMul (HPow.hPow a n) a) (HAdd.hAdd (HMul.hMul (Pell.xn a1 n) a)  …
    -/
    exact le_trans (Nat.mul_le_mul_right _ (xn_ge_a_pow n)) (Nat.le_add_right _ _)
    /-
      🎉 no goals
    -/


include a1 in
theorem n_lt_a_pow : ∀ n : ℕ, n < a ^ n
  | 0 => Nat.le_refl 1
  | n + 1 => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      ⊢ LT.lt (HAdd.hAdd n 1) (HPow.hPow a (HAdd.hAdd n 1))
    -/
    have IH := n_lt_a_pow n
    have : a ^ n + a ^ n ≤ a ^ n * a := by
      rw [← mul_two]
      exact Nat.mul_le_mul_left _ a1
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      IH : LT.lt n (HPow.hPow a n)
      this : LE.le (HAdd.hAdd (HPow.hPow a n) (HPow.hPow a n)) (HMul.hMul (HPow.hPow …
      ⊢ LT.lt (HAdd.hAdd n 1) (HPow.hPow a (HAdd.hAdd n 1))
    -/
    simp only [_root_.pow_succ, gt_iff_lt]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      IH : LT.lt n (HPow.hPow a n)
      this : LE.le (HAdd.hAdd (HPow.hPow a n) (HPow.hPow a n)) (HMul.hMul (HPow.hPow …
      ⊢ LT.lt (HAdd.hAdd n 1) (HMul.hMul (HPow.hPow a n) a)
    -/
    refine lt_of_lt_of_le ?_ this
    /-
      a : Nat
      a1 : LT.lt 1 a
      n : Nat
      IH : LT.lt n (HPow.hPow a n)
      this : LE.le (HAdd.hAdd (HPow.hPow a n) (HPow.hPow a n)) (HMul.hMul (HPow.hPow …
      ⊢ LT.lt (HAdd.hAdd n 1) (HAdd.hAdd (HPow.hPow a n) (HPow.hPow a n))
    -/
    exact add_lt_add_of_lt_of_le IH (lt_of_le_of_lt (Nat.zero_le _) IH)
    /-
      🎉 no goals
    -/


theorem n_lt_xn (n) : n < xn a1 n :=
  lt_of_lt_of_le (n_lt_a_pow a1 n) (xn_ge_a_pow a1 n)


theorem x_pos (n) : 0 < xn a1 n :=
  lt_of_le_of_lt (Nat.zero_le n) (n_lt_xn a1 n)


theorem eq_pell_lem : ∀ (n) (b : ℤ√(d a1)), 1 ≤ b → IsPell b →
    b ≤ pellZd a1 n → ∃ n, b = pellZd a1 n
  | 0, _ => fun h1 _ hl => ⟨0, @Zsqrtd.le_antisymm _ (dnsq a1) _ _ hl h1⟩
  | n + 1, b => fun h1 hp h =>
    have a1p : (0 : ℤ√(d a1)) ≤ ⟨a, 1⟩ := trivial
                                                                  /-
                                                                    a : Nat
                                                                    a1 : LT.lt 1 a
                                                                    n : Nat
                                                                    b : Zsqrtd ↑(Pell.d a1)
                                                                    h1 : LE.le 1 b
                                                                    hp : Pell.IsPell b
                                                                    h : LE.le b (Pell.pellZd a1 (HAdd.hAdd n 1))
                                                                    a1p : LE.le 0 { re := ↑a, im := 1 }
                                                                    ⊢ LE.le (HMul.hMul (HMul.hMul (Pell.d a1) (HAdd.hAdd 0 1)) (HAdd.hAdd 0 1)) (H …
                                                                  -/
    have am1p : (0 : ℤ√(d a1)) ≤ ⟨a, -1⟩ := show (_ : Nat) ≤ _ by simp; exact Nat.pred_le _
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    have a1m : (⟨a, 1⟩ * ⟨a, -1⟩ : ℤ√(d a1)) = 1 := isPell_norm.1 (isPell_one a1)
    if ha : (⟨↑a, 1⟩ : ℤ√(d a1)) ≤ b then
      let ⟨m, e⟩ :=
                                        /-
                                          a : Nat
                                          a1 : LT.lt 1 a
                                          n : Nat
                                          b : Zsqrtd ↑(Pell.d a1)
                                          h1 : LE.le 1 b
                                          hp : Pell.IsPell b
                                          h : LE.le b (Pell.pellZd a1 (HAdd.hAdd n 1))
                                          a1p : LE.le 0 { re := ↑a, im := 1 }
                                          am1p : LE.le 0 { re := ↑a, im := -1 }
                                          a1m : Eq (HMul.hMul { re := ↑a, im := 1 } { re := ↑a, im := -1 }) 1
                                          ha : LE.le { re := ↑a, im := 1 } b
                                          ⊢ LE.le 1 (HMul.hMul b { re := ↑a, im := -1 })
                                        -/
        eq_pell_lem n (b * ⟨a, -1⟩) (by rw [← a1m]; exact mul_le_mul_of_nonneg_right ha am1p)
                                                    /-
                                                      🎉 no goals
                                                    -/
          (isPell_mul hp (isPell_star.1 (isPell_one a1)))
          (by
            /-
              a : Nat
              a1 : LT.lt 1 a
              n : Nat
              b : Zsqrtd ↑(Pell.d a1)
              h1 : LE.le 1 b
              hp : Pell.IsPell b
              h : LE.le b (Pell.pellZd a1 (HAdd.hAdd n 1))
              a1p : LE.le 0 { re := ↑a, im := 1 }
              am1p : LE.le 0 { re := ↑a, im := -1 }
              a1m : Eq (HMul.hMul { re := ↑a, im := 1 } { re := ↑a, im := -1 }) 1
              ha : LE.le { re := ↑a, im := 1 } b
              ⊢ LE.le (HMul.hMul b { re := ↑a, im := -1 }) (Pell.pellZd a1 n)
            -/
            have t := mul_le_mul_of_nonneg_right h am1p
            /-
              a : Nat
              a1 : LT.lt 1 a
              n : Nat
              b : Zsqrtd ↑(Pell.d a1)
              h1 : LE.le 1 b
              hp : Pell.IsPell b
              h : LE.le b (Pell.pellZd a1 (HAdd.hAdd n 1))
              a1p : LE.le 0 { re := ↑a, im := 1 }
              am1p : LE.le 0 { re := ↑a, im := -1 }
              a1m : Eq (HMul.hMul { re := ↑a, im := 1 } { re := ↑a, im := -1 }) 1
              ha : LE.le { re := ↑a, im := 1 } b
              t : LE.le (HMul.hMul b { re := ↑a, im := -1 }) (HMul.hMul (Pell.pellZd a1 (HAd …
              ⊢ LE.le (HMul.hMul b { re := ↑a, im := -1 }) (Pell.pellZd a1 n)
            -/
            rwa [pellZd_succ, mul_assoc, a1m, mul_one] at t)
            /-
              🎉 no goals
            -/
      ⟨m + 1, by
        rw [show b = b * ⟨a, -1⟩ * ⟨a, 1⟩ by rw [mul_assoc, Eq.trans (mul_comm _ _) a1m]; simp,
          pellZd_succ, e]⟩
    else
      suffices ¬1 < b from ⟨0, show b = 1 from (Or.resolve_left (lt_or_eq_of_le h1) this).symm⟩
      fun h1l => by
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        b : Zsqrtd ↑(Pell.d a1)
        h1 : LE.le 1 b
        hp : Pell.IsPell b
        h : LE.le b (Pell.pellZd a1 (HAdd.hAdd n 1))
        a1p : LE.le 0 { re := ↑a, im := 1 }
        am1p : LE.le 0 { re := ↑a, im := -1 }
        a1m : Eq (HMul.hMul { re := ↑a, im := 1 } { re := ↑a, im := -1 }) 1
        ha : Not (LE.le { re := ↑a, im := 1 } b)
        h1l : LT.lt 1 b
        ⊢ False
      -/
      cases' b with x y
      exact by
        have bm : (_ * ⟨_, _⟩ : ℤ√d a1) = 1 := Pell.isPell_norm.1 hp
        have y0l : (0 : ℤ√d a1) < ⟨x - x, y - -y⟩ :=
          sub_lt_sub h1l fun hn : (1 : ℤ√d a1) ≤ ⟨x, -y⟩ => by
            have t := mul_le_mul_of_nonneg_left hn (le_trans zero_le_one h1)
            rw [bm, mul_one] at t
            exact h1l t
        have yl2 : (⟨_, _⟩ : ℤ√_) < ⟨_, _⟩ :=
          show (⟨x, y⟩ - ⟨x, -y⟩ : ℤ√d a1) < ⟨a, 1⟩ - ⟨a, -1⟩ from
            sub_lt_sub ha fun hn : (⟨x, -y⟩ : ℤ√d a1) ≤ ⟨a, -1⟩ => by
              have t := mul_le_mul_of_nonneg_right
                      (mul_le_mul_of_nonneg_left hn (le_trans zero_le_one h1)) a1p
              rw [bm, one_mul, mul_assoc, Eq.trans (mul_comm _ _) a1m, mul_one] at t
              exact ha t
        simp only [sub_self, sub_neg_eq_add] at y0l; simp only [Zsqrtd.neg_re, add_neg_cancel,
          Zsqrtd.neg_im, neg_neg] at yl2
        exact
          match y, y0l, (yl2 : (⟨_, _⟩ : ℤ√_) < ⟨_, _⟩) with
          | 0, y0l, _ => y0l (le_refl 0)
          | (y + 1 : ℕ), _, yl2 =>
            yl2
              (Zsqrtd.le_of_le_le (by simp [sub_eq_add_neg])
                (let t := Int.ofNat_le_ofNat_of_le (Nat.succ_pos y)
                add_le_add t t))
          | Int.negSucc _, y0l, _ => y0l trivial


theorem eq_pellZd (b : ℤ√(d a1)) (b1 : 1 ≤ b) (hp : IsPell b) : ∃ n, b = pellZd a1 n :=
  let ⟨n, h⟩ := @Zsqrtd.le_arch (d a1) b
  eq_pell_lem a1 n b b1 hp <|
    h.trans <| by
      /-
        a : Nat
        a1 : LT.lt 1 a
        b : Zsqrtd ↑(Pell.d a1)
        b1 : LE.le 1 b
        hp : Pell.IsPell b
        n : Nat
        h : LE.le b ↑n
        ⊢ LE.le (↑n) (Pell.pellZd a1 n)
      -/
      rw [Zsqrtd.natCast_val]
      exact
        Zsqrtd.le_of_le_le (Int.ofNat_le_ofNat_of_le <| le_of_lt <| n_lt_xn _ _)
          (Int.ofNat_zero_le _)


/-- Every solution to **Pell's equation** is recursively obtained from the initial solution
`(1,0)` using the recursion `pell`. -/
theorem eq_pell {x y : ℕ} (hp : x * x - d a1 * y * y = 1) : ∃ n, x = xn a1 n ∧ y = yn a1 n :=
  have : (1 : ℤ√(d a1)) ≤ ⟨x, y⟩ :=
    match x, hp with
                                /-
                                  a : Nat
                                  a1 : LT.lt 1 a
                                  x y : Nat
                                  hp✝ : Eq (HSub.hSub (HMul.hMul x x) (HMul.hMul (HMul.hMul (Pell.d a1) y) y)) 1
                                  hp : Eq (HSub.hSub 0 (HMul.hMul (HMul.hMul (Pell.d a1) y) y)) 1
                                  ⊢ LE.le 1 { re := ↑0, im := ↑y }
                                -/
    | 0, (hp : 0 - _ = 1) => by rw [zero_tsub] at hp; contradiction
                                                      /-
                                                        🎉 no goals
                                                      -/
    | x + 1, _hp =>
      Zsqrtd.le_of_le_le (Int.ofNat_le_ofNat_of_le <| Nat.succ_pos x) (Int.ofNat_zero_le _)
  let ⟨m, e⟩ := eq_pellZd a1 ⟨x, y⟩ this ((isPell_nat a1).2 hp)
  ⟨m,
    match x, y, e with
    | _, _, rfl => ⟨rfl, rfl⟩⟩


theorem pellZd_add (m) : ∀ n, pellZd a1 (m + n) = pellZd a1 m * pellZd a1 n
  | 0 => (mul_one _).symm
                /-
                  a : Nat
                  a1 : LT.lt 1 a
                  m n : Nat
                  ⊢ Eq (Pell.pellZd a1 (HAdd.hAdd m (HAdd.hAdd n 1))) (HMul.hMul (Pell.pellZd a1 …
                -/
  | n + 1 => by rw [← add_assoc, pellZd_succ, pellZd_succ, pellZd_add _ n, ← mul_assoc]
                /-
                  🎉 no goals
                -/


theorem xn_add (m n) : xn a1 (m + n) = xn a1 m * xn a1 n + d a1 * yn a1 m * yn a1 n := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    ⊢ Eq (Pell.xn a1 (HAdd.hAdd m n)) (HAdd.hAdd (HMul.hMul (Pell.xn a1 m) (Pell.x …
  -/
  injection pellZd_add a1 m n with h _
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    im_eq✝ : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    ⊢ Eq (Pell.xn a1 (HAdd.hAdd m n)) (HAdd.hAdd (HMul.hMul (Pell.xn a1 m) (Pell.x …
  -/
  zify
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    im_eq✝ : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    ⊢ Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul ↑(Pell.xn a1 m) ↑(P …
  -/
  rw [h]
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    im_eq✝ : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m).re (Pell.pellZd a1 n).re) (HMul. …
  -/
  simp [pellZd]
  /-
    🎉 no goals
  -/


theorem yn_add (m n) : yn a1 (m + n) = xn a1 m * yn a1 n + yn a1 m * xn a1 n := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    ⊢ Eq (Pell.yn a1 (HAdd.hAdd m n)) (HAdd.hAdd (HMul.hMul (Pell.xn a1 m) (Pell.y …
  -/
  injection pellZd_add a1 m n with _ h
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    re_eq✝ : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    h : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    ⊢ Eq (Pell.yn a1 (HAdd.hAdd m n)) (HAdd.hAdd (HMul.hMul (Pell.xn a1 m) (Pell.y …
  -/
  zify
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    re_eq✝ : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    h : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    ⊢ Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul ↑(Pell.xn a1 m) ↑(P …
  -/
  rw [h]
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    re_eq✝ : Eq (↑(Pell.xn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd …
    h : Eq (↑(Pell.yn a1 (HAdd.hAdd m n))) (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Pell.pellZd a1 m).re (Pell.pellZd a1 n).im) (HMul. …
  -/
  simp [pellZd]
  /-
    🎉 no goals
  -/


theorem pellZd_sub {m n} (h : n ≤ m) : pellZd a1 (m - n) = pellZd a1 m * star (pellZd a1 n) := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    ⊢ Eq (Pell.pellZd a1 (HSub.hSub m n)) (HMul.hMul (Pell.pellZd a1 m) (Star.star …
  -/
  let t := pellZd_add a1 n (m - n)
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    t : Eq (Pell.pellZd a1 (HAdd.hAdd n (HSub.hSub m n))) (HMul.hMul (Pell.pellZd  …
    ⊢ Eq (Pell.pellZd a1 (HSub.hSub m n)) (HMul.hMul (Pell.pellZd a1 m) (Star.star …
  -/
  rw [add_tsub_cancel_of_le h] at t
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    t : Eq (Pell.pellZd a1 m) (HMul.hMul (Pell.pellZd a1 n) (Pell.pellZd a1 (HSub. …
    ⊢ Eq (Pell.pellZd a1 (HSub.hSub m n)) (HMul.hMul (Pell.pellZd a1 m) (Star.star …
  -/
  rw [t, mul_comm (pellZd _ n) _, mul_assoc, isPell_norm.1 (isPell_pellZd _ _), mul_one]
  /-
    🎉 no goals
  -/


theorem xz_sub {m n} (h : n ≤ m) :
    xz a1 (m - n) = xz a1 m * xz a1 n - d a1 * yz a1 m * yz a1 n := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    ⊢ Eq (Pell.xz a1 (HSub.hSub m n)) (HSub.hSub (HMul.hMul (Pell.xz a1 m) (Pell.x …
  -/
  rw [sub_eq_add_neg, ← mul_neg]
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    ⊢ Eq (Pell.xz a1 (HSub.hSub m n)) (HAdd.hAdd (HMul.hMul (Pell.xz a1 m) (Pell.x …
  -/
  exact congr_arg Zsqrtd.re (pellZd_sub a1 h)
  /-
    🎉 no goals
  -/


theorem yz_sub {m n} (h : n ≤ m) : yz a1 (m - n) = xz a1 n * yz a1 m - xz a1 m * yz a1 n := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    ⊢ Eq (Pell.yz a1 (HSub.hSub m n)) (HSub.hSub (HMul.hMul (Pell.xz a1 n) (Pell.y …
  -/
  rw [sub_eq_add_neg, ← mul_neg, mul_comm, add_comm]
  /-
    a : Nat
    a1 : LT.lt 1 a
    m n : Nat
    h : LE.le n m
    ⊢ Eq (Pell.yz a1 (HSub.hSub m n)) (HAdd.hAdd (HMul.hMul (Pell.xz a1 m) (Neg.ne …
  -/
  exact congr_arg Zsqrtd.im (pellZd_sub a1 h)
  /-
    🎉 no goals
  -/


theorem xy_coprime (n) : (xn a1 n).Coprime (yn a1 n) :=
  Nat.coprime_of_dvd' fun k _ kx ky => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      x✝ : Nat.Prime k
      kx : Dvd.dvd k (Pell.xn a1 n)
      ky : Dvd.dvd k (Pell.yn a1 n)
      ⊢ Dvd.dvd k 1
    -/
    let p := pell_eq a1 n
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      x✝ : Nat.Prime k
      kx : Dvd.dvd k (Pell.xn a1 n)
      ky : Dvd.dvd k (Pell.yn a1 n)
      p : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul.h …
      ⊢ Dvd.dvd k 1
    -/
    rw [← p]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      x✝ : Nat.Prime k
      kx : Dvd.dvd k (Pell.xn a1 n)
      ky : Dvd.dvd k (Pell.yn a1 n)
      p : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul.h …
      ⊢ Dvd.dvd k (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (H …
    -/
    exact Nat.dvd_sub (le_of_lt <| Nat.lt_of_sub_eq_succ p) (kx.mul_left _) (ky.mul_left _)
    /-
      🎉 no goals
    -/


theorem strictMono_y : StrictMono (yn a1)
  | _, 0, h => absurd h <| Nat.not_lt_zero _
  | m, n + 1, h => by
    have : yn a1 m ≤ yn a1 n :=
      Or.elim (lt_or_eq_of_le <| Nat.le_of_succ_le_succ h) (fun hl => le_of_lt <| strictMono_y hl)
        fun e => by rw [e]
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.yn a1 m) (Pell.yn a1 n)
      ⊢ LT.lt (Pell.yn a1 m) (Pell.yn a1 (HAdd.hAdd n 1))
    -/
    simp only [yn_succ, gt_iff_lt]; refine lt_of_le_of_lt ?_ (Nat.lt_add_of_pos_left <| x_pos a1 n)
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.yn a1 m) (Pell.yn a1 n)
      ⊢ LE.le (Pell.yn a1 m) (HMul.hMul (Pell.yn a1 n) a)
    -/
    rw [← mul_one (yn a1 m)]
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.yn a1 m) (Pell.yn a1 n)
      ⊢ LE.le (HMul.hMul (Pell.yn a1 m) 1) (HMul.hMul (Pell.yn a1 n) a)
    -/
    exact mul_le_mul this (le_of_lt a1) (Nat.zero_le _) (Nat.zero_le _)
    /-
      🎉 no goals
    -/


theorem strictMono_x : StrictMono (xn a1)
  | _, 0, h => absurd h <| Nat.not_lt_zero _
  | m, n + 1, h => by
    have : xn a1 m ≤ xn a1 n :=
      Or.elim (lt_or_eq_of_le <| Nat.le_of_succ_le_succ h) (fun hl => le_of_lt <| strictMono_x hl)
        fun e => by rw [e]
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.xn a1 m) (Pell.xn a1 n)
      ⊢ LT.lt (Pell.xn a1 m) (Pell.xn a1 (HAdd.hAdd n 1))
    -/
    simp only [xn_succ, gt_iff_lt]
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.xn a1 m) (Pell.xn a1 n)
      ⊢ LT.lt (Pell.xn a1 m) (HAdd.hAdd (HMul.hMul (Pell.xn a1 n) a) (HMul.hMul (Pel …
    -/
    refine lt_of_lt_of_le (lt_of_le_of_lt this ?_) (Nat.le_add_right _ _)
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.xn a1 m) (Pell.xn a1 n)
      ⊢ LT.lt (Pell.xn a1 n) (HMul.hMul (Pell.xn a1 n) a)
    -/
    have t := Nat.mul_lt_mul_of_pos_left a1 (x_pos a1 n)
    /-
      a : Nat
      a1 : LT.lt 1 a
      m n : Nat
      h : LT.lt m (HAdd.hAdd n 1)
      this : LE.le (Pell.xn a1 m) (Pell.xn a1 n)
      t : LT.lt (HMul.hMul (Pell.xn a1 n) 1) (HMul.hMul (Pell.xn a1 n) a)
      ⊢ LT.lt (Pell.xn a1 n) (HMul.hMul (Pell.xn a1 n) a)
    -/
    rwa [mul_one] at t
    /-
      🎉 no goals
    -/


theorem yn_ge_n : ∀ n, n ≤ yn a1 n
  | 0 => Nat.zero_le _
  | n + 1 =>
    show n < yn a1 (n + 1) from lt_of_le_of_lt (yn_ge_n n) (strictMono_y a1 <| Nat.lt_succ_self n)


theorem y_mul_dvd (n) : ∀ k, yn a1 n ∣ yn a1 (n * k)
  | 0 => dvd_zero _
  | k + 1 => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      ⊢ Dvd.dvd (Pell.yn a1 n) (Pell.yn a1 (HMul.hMul n (HAdd.hAdd k 1)))
    -/
    rw [Nat.mul_succ, yn_add]; exact dvd_add (dvd_mul_left _ _) ((y_mul_dvd _ k).mul_right _)
                               /-
                                 🎉 no goals
                               -/


theorem y_dvd_iff (m n) : yn a1 m ∣ yn a1 n ↔ m ∣ n :=
  ⟨fun h =>
    Nat.dvd_of_mod_eq_zero <|
      (Nat.eq_zero_or_pos _).resolve_right fun hp => by
        have co : Nat.Coprime (yn a1 m) (xn a1 (m * (n / m))) :=
          Nat.Coprime.symm <| (xy_coprime a1 _).coprime_dvd_right (y_mul_dvd a1 m (n / m))
        have m0 : 0 < m :=
          m.eq_zero_or_pos.resolve_left fun e => by
            rw [e, Nat.mod_zero] at hp;rw [e] at h
            exact _root_.ne_of_lt (strictMono_y a1 hp) (eq_zero_of_zero_dvd h).symm
        /-
          a : Nat
          a1 : LT.lt 1 a
          m n : Nat
          h : Dvd.dvd (Pell.yn a1 m) (Pell.yn a1 n)
          hp : GT.gt (HMod.hMod n m) 0
          co : (Pell.yn a1 m).Coprime (Pell.xn a1 (HMul.hMul m (HDiv.hDiv n m)))
          m0 : LT.lt 0 m
          ⊢ False
        -/
        rw [← Nat.mod_add_div n m, yn_add] at h
        exact
          not_le_of_gt (strictMono_y _ <| Nat.mod_lt n m0)
            (Nat.le_of_dvd (strictMono_y _ hp) <|
              co.dvd_of_dvd_mul_right <|
                (Nat.dvd_add_iff_right <| (y_mul_dvd _ _ _).mul_left _).2 h),
                     /-
                       a : Nat
                       a1 : LT.lt 1 a
                       m n : Nat
                       x✝ : Dvd.dvd m n
                       k : Nat
                       e : Eq n (HMul.hMul m k)
                       ⊢ Dvd.dvd (Pell.yn a1 m) (Pell.yn a1 n)
                     -/
    fun ⟨k, e⟩ => by rw [e]; apply y_mul_dvd⟩
                             /-
                               🎉 no goals
                             -/


theorem xy_modEq_yn (n) :
    ∀ k, xn a1 (n * k) ≡ xn a1 n ^ k [MOD yn a1 n ^ 2] ∧ yn a1 (n * k) ≡
        k * xn a1 n ^ (k - 1) * yn a1 n [MOD yn a1 n ^ 3]
            /-
              a : Nat
              a1 : LT.lt 1 a
              n : Nat
              ⊢ And ((HPow.hPow (Pell.yn a1 n) 2).ModEq (Pell.xn a1 (HMul.hMul n 0)) (HPow.h …
            -/
                            /-
                              🎉 no goals
                            -/
  | 0 => by constructor <;> simpa using Nat.ModEq.refl _
                            /-
                              🎉 no goals
                            -/
  | k + 1 => by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      ⊢ And ((HPow.hPow (Pell.yn a1 n) 2).ModEq (Pell.xn a1 (HMul.hMul n (HAdd.hAdd  …
    -/
    let ⟨hx, hy⟩ := xy_modEq_yn n k
    have L : xn a1 (n * k) * xn a1 n + d a1 * yn a1 (n * k) * yn a1 n ≡
        xn a1 n ^ k * xn a1 n + 0 [MOD yn a1 n ^ 2] :=
      (hx.mul_right _).add <|
        modEq_zero_iff_dvd.2 <| by
          rw [_root_.pow_succ]
          exact
            mul_dvd_mul_right
              (dvd_mul_of_dvd_right
                (modEq_zero_iff_dvd.1 <|
                  (hy.of_dvd <| by simp [_root_.pow_succ]).trans <|
                    modEq_zero_iff_dvd.2 <| by simp)
                _) _
    have R : xn a1 (n * k) * yn a1 n + yn a1 (n * k) * xn a1 n ≡
        xn a1 n ^ k * yn a1 n + k * xn a1 n ^ k * yn a1 n [MOD yn a1 n ^ 3] :=
      ModEq.add
          (by
            rw [_root_.pow_succ]
            exact hx.mul_right' _) <| by
        have : k * xn a1 n ^ (k - 1) * yn a1 n * xn a1 n = k * xn a1 n ^ k * yn a1 n := by
          cases' k with k <;> simp [_root_.pow_succ]; ring_nf
        rw [← this]
        exact hy.mul_right _
    rw [add_tsub_cancel_right, Nat.mul_succ, xn_add, yn_add, pow_succ (xn _ n), Nat.succ_mul,
      add_comm (k * xn _ n ^ k) (xn _ n ^ k), right_distrib]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n k : Nat
      hx : (HPow.hPow (Pell.yn a1 n) 2).ModEq (Pell.xn a1 (HMul.hMul n k)) (HPow.hPo …
      hy : (HPow.hPow (Pell.yn a1 n) 3).ModEq (Pell.yn a1 (HMul.hMul n k)) (HMul.hMu …
      L : (HPow.hPow (Pell.yn a1 n) 2).ModEq (HAdd.hAdd (HMul.hMul (Pell.xn a1 (HMul …
      R : (HPow.hPow (Pell.yn a1 n) 3).ModEq (HAdd.hAdd (HMul.hMul (Pell.xn a1 (HMul …
      ⊢ And ((HPow.hPow (Pell.yn a1 n) 2).ModEq (HAdd.hAdd (HMul.hMul (Pell.xn a1 (H …
    -/
    exact ⟨L, R⟩
    /-
      🎉 no goals
    -/


theorem ysq_dvd_yy (n) : yn a1 n * yn a1 n ∣ yn a1 (n * yn a1 n) :=
  modEq_zero_iff_dvd.1 <|
                                                     /-
                                                       a : Nat
                                                       a1 : LT.lt 1 a
                                                       n : Nat
                                                       ⊢ Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (HPow.hPow (Pell.yn a1 n) 3)
                                                     -/
    ((xy_modEq_yn a1 n (yn a1 n)).right.of_dvd <| by simp [_root_.pow_succ]).trans
                                                     /-
                                                       🎉 no goals
                                                     -/
                                  /-
                                    a : Nat
                                    a1 : LT.lt 1 a
                                    n : Nat
                                    ⊢ Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (HMul.hMul (HMul.hMul (Pel …
                                  -/
      (modEq_zero_iff_dvd.2 <| by simp [mul_dvd_mul_left, mul_assoc])
                                  /-
                                    🎉 no goals
                                  -/


theorem dvd_of_ysq_dvd {n t} (h : yn a1 n * yn a1 n ∣ yn a1 t) : yn a1 n ∣ t :=
  have nt : n ∣ t := (y_dvd_iff a1 n t).1 <| dvd_of_mul_left_dvd h
                                      /-
                                        a : Nat
                                        a1 : LT.lt 1 a
                                        n t : Nat
                                        h : Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (Pell.yn a1 t)
                                        nt : Dvd.dvd n t
                                        n0 : Eq n 0
                                        ⊢ Dvd.dvd (Pell.yn a1 n) t
                                      -/
  n.eq_zero_or_pos.elim (fun n0 => by rwa [n0] at nt ⊢) fun n0l : 0 < n => by
                                      /-
                                        🎉 no goals
                                      -/
    /-
      a : Nat
      a1 : LT.lt 1 a
      n t : Nat
      h : Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (Pell.yn a1 t)
      nt : Dvd.dvd n t
      n0l : LT.lt 0 n
      ⊢ Dvd.dvd (Pell.yn a1 n) t
    -/
    let ⟨k, ke⟩ := nt
    have : yn a1 n ∣ k * xn a1 n ^ (k - 1) :=
      Nat.dvd_of_mul_dvd_mul_right (strictMono_y a1 n0l) <|
        modEq_zero_iff_dvd.1 <| by
          have xm := (xy_modEq_yn a1 n k).right; rw [← ke] at xm
          exact (xm.of_dvd <| by simp [_root_.pow_succ]).symm.trans h.modEq_zero_nat
    /-
      a : Nat
      a1 : LT.lt 1 a
      n t : Nat
      h : Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (Pell.yn a1 t)
      nt : Dvd.dvd n t
      n0l : LT.lt 0 n
      k : Nat
      ke : Eq t (HMul.hMul n k)
      this : Dvd.dvd (Pell.yn a1 n) (HMul.hMul k (HPow.hPow (Pell.xn a1 n) (HSub.hSu …
      ⊢ Dvd.dvd (Pell.yn a1 n) t
    -/
    rw [ke]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n t : Nat
      h : Dvd.dvd (HMul.hMul (Pell.yn a1 n) (Pell.yn a1 n)) (Pell.yn a1 t)
      nt : Dvd.dvd n t
      n0l : LT.lt 0 n
      k : Nat
      ke : Eq t (HMul.hMul n k)
      this : Dvd.dvd (Pell.yn a1 n) (HMul.hMul k (HPow.hPow (Pell.xn a1 n) (HSub.hSu …
      ⊢ Dvd.dvd (Pell.yn a1 n) (HMul.hMul n k)
    -/
    exact dvd_mul_of_dvd_right (((xy_coprime _ _).pow_left _).symm.dvd_of_dvd_mul_right this) _
    /-
      🎉 no goals
    -/


theorem pellZd_succ_succ (n) :
    pellZd a1 (n + 2) + pellZd a1 n = (2 * a : ℕ) * pellZd a1 (n + 1) := by
  have : (1 : ℤ√(d a1)) + ⟨a, 1⟩ * ⟨a, 1⟩ = ⟨a, 1⟩ * (2 * a) := by
    rw [Zsqrtd.natCast_val]
    change (⟨_, _⟩ : ℤ√(d a1)) = ⟨_, _⟩
    rw [dz_val]
    dsimp [az]
    ext <;> dsimp <;> ring_nf
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    this : Eq (HAdd.hAdd 1 (HMul.hMul { re := ↑a, im := 1 } { re := ↑a, im := 1 }) …
    ⊢ Eq (HAdd.hAdd (Pell.pellZd a1 (HAdd.hAdd n 2)) (Pell.pellZd a1 n)) (HMul.hMu …
  -/
  simpa [mul_add, mul_comm, mul_left_comm, add_comm] using congr_arg (· * pellZd a1 n) this
  /-
    🎉 no goals
  -/


theorem xy_succ_succ (n) :
    xn a1 (n + 2) + xn a1 n =
      2 * a * xn a1 (n + 1) ∧ yn a1 (n + 2) + yn a1 n = 2 * a * yn a1 (n + 1) := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    ⊢ And (Eq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd n 2)) (Pell.xn a1 n)) (HMul.hMul ( …
  -/
  have := pellZd_succ_succ a1 n; unfold pellZd at this
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    this : Eq (HAdd.hAdd { re := ↑(Pell.xn a1 (HAdd.hAdd n 2)), im := ↑(Pell.yn a1 …
    ⊢ And (Eq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd n 2)) (Pell.xn a1 n)) (HMul.hMul ( …
  -/
  erw [Zsqrtd.smul_val (2 * a : ℕ)] at this
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    this : Eq (HAdd.hAdd { re := ↑(Pell.xn a1 (HAdd.hAdd n 2)), im := ↑(Pell.yn a1 …
    ⊢ And (Eq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd n 2)) (Pell.xn a1 n)) (HMul.hMul ( …
  -/
  injection this with h₁ h₂
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    h₁ : Eq (HAdd.hAdd { re := ↑(Pell.xn a1 (HAdd.hAdd n 2)), im := ↑(Pell.yn a1 ( …
    h₂ : Eq (HAdd.hAdd { re := ↑(Pell.xn a1 (HAdd.hAdd n 2)), im := ↑(Pell.yn a1 ( …
    ⊢ And (Eq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd n 2)) (Pell.xn a1 n)) (HMul.hMul ( …
  -/
  constructor <;> apply Int.ofNat.inj <;> [simpa using h₁; simpa using h₂]
  /-
    🎉 no goals
  -/


theorem xn_succ_succ (n) : xn a1 (n + 2) + xn a1 n = 2 * a * xn a1 (n + 1) :=
  (xy_succ_succ a1 n).1


theorem yn_succ_succ (n) : yn a1 (n + 2) + yn a1 n = 2 * a * yn a1 (n + 1) :=
  (xy_succ_succ a1 n).2


theorem xz_succ_succ (n) : xz a1 (n + 2) = (2 * a : ℕ) * xz a1 (n + 1) - xz a1 n :=
                         /-
                           a : Nat
                           a1 : LT.lt 1 a
                           n : Nat
                           ⊢ Eq (HAdd.hAdd (Pell.xz a1 (HAdd.hAdd n 2)) (Pell.xz a1 n)) (HMul.hMul (↑(HMu …
                         -/
  eq_sub_of_add_eq <| by delta xz; rw [← Int.ofNat_add, ← Int.ofNat_mul, xn_succ_succ]
                                   /-
                                     🎉 no goals
                                   -/


theorem yz_succ_succ (n) : yz a1 (n + 2) = (2 * a : ℕ) * yz a1 (n + 1) - yz a1 n :=
                         /-
                           a : Nat
                           a1 : LT.lt 1 a
                           n : Nat
                           ⊢ Eq (HAdd.hAdd (Pell.yz a1 (HAdd.hAdd n 2)) (Pell.yz a1 n)) (HMul.hMul (↑(HMu …
                         -/
  eq_sub_of_add_eq <| by delta yz; rw [← Int.ofNat_add, ← Int.ofNat_mul, yn_succ_succ]
                                   /-
                                     🎉 no goals
                                   -/


theorem yn_modEq_a_sub_one : ∀ n, yn a1 n ≡ n [MOD a - 1]
            /-
              a : Nat
              a1 : LT.lt 1 a
              ⊢ (HSub.hSub a 1).ModEq (Pell.yn a1 0) 0
            -/
  | 0 => by simp [Nat.ModEq.refl]
            /-
              🎉 no goals
            -/
            /-
              a : Nat
              a1 : LT.lt 1 a
              ⊢ (HSub.hSub a 1).ModEq (Pell.yn a1 1) 1
            -/
  | 1 => by simp [Nat.ModEq.refl]
            /-
              🎉 no goals
            -/
  | n + 2 =>
    (yn_modEq_a_sub_one n).add_right_cancel <| by
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        ⊢ (HSub.hSub a 1).ModEq (HAdd.hAdd (Pell.yn a1 (HAdd.hAdd n 2)) (Pell.yn a1 n) …
      -/
      rw [yn_succ_succ, (by ring : n + 2 + n = 2 * (n + 1))]
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        ⊢ (HSub.hSub a 1).ModEq (HMul.hMul (HMul.hMul 2 a) (Pell.yn a1 (HAdd.hAdd n 1) …
      -/
      exact ((modEq_sub a1.le).mul_left 2).mul (yn_modEq_a_sub_one (n + 1))
      /-
        🎉 no goals
      -/


theorem yn_modEq_two : ∀ n, yn a1 n ≡ n [MOD 2]
            /-
              a : Nat
              a1 : LT.lt 1 a
              ⊢ Nat.ModEq 2 (Pell.yn a1 0) 0
            -/
  | 0 => by rfl
            /-
              🎉 no goals
            -/
            /-
              a : Nat
              a1 : LT.lt 1 a
              ⊢ Nat.ModEq 2 (Pell.yn a1 1) 1
            -/
  | 1 => by simp; rfl
                  /-
                    🎉 no goals
                  -/
  | n + 2 =>
    (yn_modEq_two n).add_right_cancel <| by
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        ⊢ Nat.ModEq 2 (HAdd.hAdd (Pell.yn a1 (HAdd.hAdd n 2)) (Pell.yn a1 n)) (HAdd.hA …
      -/
      rw [yn_succ_succ, mul_assoc, (by ring : n + 2 + n = 2 * (n + 1))]
      /-
        a : Nat
        a1 : LT.lt 1 a
        n : Nat
        ⊢ Nat.ModEq 2 (HMul.hMul 2 (HMul.hMul a (Pell.yn a1 (HAdd.hAdd n 1)))) (HMul.h …
      -/
      exact (dvd_mul_right 2 _).modEq_zero_nat.trans (dvd_mul_right 2 _).zero_modEq_nat
      /-
        🎉 no goals
      -/


theorem x_sub_y_dvd_pow_lem (y2 y1 y0 yn1 yn0 xn1 xn0 ay a2 : ℤ) :
    (a2 * yn1 - yn0) * ay + y2 - (a2 * xn1 - xn0) =
      y2 - a2 * y1 + y0 + a2 * (yn1 * ay + y1 - xn1) - (yn0 * ay + y0 - xn0) := by
  /-
    y2 y1 y0 yn1 yn0 xn1 xn0 ay a2 : Int
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HSub.hSub (HMul.hMul a2 yn1) yn0) ay) y …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem x_sub_y_dvd_pow (y : ℕ) :
    ∀ n, (2 * a * y - y * y - 1 : ℤ) ∣ yz a1 n * (a - y) + ↑(y ^ n) - xz a1 n
            /-
              a : Nat
              a1 : LT.lt 1 a
              y : Nat
              ⊢ Dvd.dvd (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑y) (HMul.hMul ↑y  …
            -/
  | 0 => by simp [xz, yz, Int.ofNat_zero, Int.ofNat_one]
            /-
              🎉 no goals
            -/
            /-
              a : Nat
              a1 : LT.lt 1 a
              y : Nat
              ⊢ Dvd.dvd (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑y) (HMul.hMul ↑y  …
            -/
  | 1 => by simp [xz, yz, Int.ofNat_zero, Int.ofNat_one]
            /-
              🎉 no goals
            -/
  | n + 2 => by
    have : (2 * a * y - y * y - 1 : ℤ) ∣ ↑(y ^ (n + 2)) - ↑(2 * a) * ↑(y ^ (n + 1)) + ↑(y ^ n) :=
      ⟨-↑(y ^ n), by
        simp [_root_.pow_succ, mul_add, Int.ofNat_mul, show ((2 : ℕ) : ℤ) = 2 from rfl, mul_comm,
          mul_left_comm]
        ring⟩
    /-
      a : Nat
      a1 : LT.lt 1 a
      y n : Nat
      this : Dvd.dvd (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑y) (HMul.hMu …
      ⊢ Dvd.dvd (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑y) (HMul.hMul ↑y  …
    -/
    rw [xz_succ_succ, yz_succ_succ, x_sub_y_dvd_pow_lem ↑(y ^ (n + 2)) ↑(y ^ (n + 1)) ↑(y ^ n)]
    exact _root_.dvd_sub (dvd_add this <| (x_sub_y_dvd_pow _ (n + 1)).mul_left _)
      (x_sub_y_dvd_pow _ n)


theorem xn_modEq_x2n_add_lem (n j) : xn a1 n ∣ d a1 * yn a1 n * (yn a1 n * xn a1 j) + xn a1 j := by
  have h1 : d a1 * yn a1 n * (yn a1 n * xn a1 j) + xn a1 j =
      (d a1 * yn a1 n * yn a1 n + 1) * xn a1 j := by
    simp [add_mul, mul_assoc]
  have h2 : d a1 * yn a1 n * yn a1 n + 1 = xn a1 n * xn a1 n := by
    zify at *
    apply add_eq_of_eq_sub' (Eq.symm (pell_eqz a1 n))
  /-
    a : Nat
    a1 : LT.lt 1 a
    n j : Nat
    h1 : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (Pell.d a1) (Pell.yn a1 n)) (HMul.hMu …
    h2 : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (Pell.d a1) (Pell.yn a1 n)) (Pell.yn  …
    ⊢ Dvd.dvd (Pell.xn a1 n) (HAdd.hAdd (HMul.hMul (HMul.hMul (Pell.d a1) (Pell.yn …
  -/
  rw [h2] at h1; rw [h1, mul_assoc]; exact dvd_mul_right _ _
                                     /-
                                       🎉 no goals
                                     -/


theorem xn_modEq_x2n_add (n j) : xn a1 (2 * n + j) + xn a1 j ≡ 0 [MOD xn a1 n] := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    n j : Nat
    ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd (HMul.hMul 2 n) j)) ( …
  -/
  rw [two_mul, add_assoc, xn_add, add_assoc, ← zero_add 0]
  /-
    a : Nat
    a1 : LT.lt 1 a
    n j : Nat
    ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 (HAdd. …
  -/
  refine (dvd_mul_right (xn a1 n) (xn a1 (n + j))).modEq_zero_nat.add ?_
  /-
    a : Nat
    a1 : LT.lt 1 a
    n j : Nat
    ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (HMul.hMul (HMul.hMul (Pell.d a1) (Pell.yn a …
  -/
  rw [yn_add, left_distrib, add_assoc, ← zero_add 0]
  exact
    ((dvd_mul_right _ _).mul_left _).modEq_zero_nat.add (xn_modEq_x2n_add_lem _ _ _).modEq_zero_nat


theorem xn_modEq_x2n_sub_lem {n j} (h : j ≤ n) : xn a1 (2 * n - j) + xn a1 j ≡ 0 [MOD xn a1 n] := by
  have h1 : xz a1 n ∣ d a1 * yz a1 n * yz a1 (n - j) + xz a1 j := by
    rw [yz_sub _ h, mul_sub_left_distrib, sub_add_eq_add_sub]
    exact
      dvd_sub
        (by
          delta xz; delta yz
          rw [mul_comm (xn _ _ : ℤ)]
          exact mod_cast (xn_modEq_x2n_add_lem _ n j))
        ((dvd_mul_right _ _).mul_left _)
  /-
    a : Nat
    a1 : LT.lt 1 a
    n j : Nat
    h : LE.le j n
    h1 : Dvd.dvd (Pell.xz a1 n) (HAdd.hAdd (HMul.hMul (HMul.hMul (↑(Pell.d a1)) (P …
    ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) j)) ( …
  -/
  rw [two_mul, add_tsub_assoc_of_le h, xn_add, add_assoc, ← zero_add 0]
  exact
    (dvd_mul_right _ _).modEq_zero_nat.add
      (Int.natCast_dvd_natCast.1 <| by simpa [xz, yz] using h1).modEq_zero_nat


theorem xn_modEq_x2n_sub {n j} (h : j ≤ 2 * n) : xn a1 (2 * n - j) + xn a1 j ≡ 0 [MOD xn a1 n] :=
  (le_total j n).elim (xn_modEq_x2n_sub_lem a1) fun jn => by
    have : 2 * n - j + j ≤ n + j := by
      rw [tsub_add_cancel_of_le h, two_mul]; exact Nat.add_le_add_left jn _
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      h : LE.le j (HMul.hMul 2 n)
      jn : LE.le n j
      this : LE.le (HAdd.hAdd (HSub.hSub (HMul.hMul 2 n) j) j) (HAdd.hAdd n j)
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) j)) ( …
    -/
    let t := xn_modEq_x2n_sub_lem a1 (Nat.le_of_add_le_add_right this)
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      h : LE.le j (HMul.hMul 2 n)
      jn : LE.le n j
      this : LE.le (HAdd.hAdd (HSub.hSub (HMul.hMul 2 n) j) j) (HAdd.hAdd n j)
      t : (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) (HS …
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) j)) ( …
    -/
    rwa [tsub_tsub_cancel_of_le h, add_comm] at t
    /-
      🎉 no goals
    -/


theorem xn_modEq_x4n_add (n j) : xn a1 (4 * n + j) ≡ xn a1 j [MOD xn a1 n] :=
  ModEq.add_right_cancel' (xn a1 (2 * n + j)) <| by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd (HMul.hMul 4 n) j)) ( …
    -/
    refine @ModEq.trans _ _ 0 _ ?_ (by rw [add_comm]; exact (xn_modEq_x2n_add _ _ _).symm)
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd (HMul.hMul 4 n) j)) ( …
    -/
    rw [show 4 * n = 2 * n + 2 * n from right_distrib 2 2 n, add_assoc]
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd (HMul.hMul 2 n) (HAdd …
    -/
    apply xn_modEq_x2n_add
    /-
      🎉 no goals
    -/


theorem xn_modEq_x4n_sub {n j} (h : j ≤ 2 * n) : xn a1 (4 * n - j) ≡ xn a1 j [MOD xn a1 n] :=
                                        /-
                                          a : Nat
                                          a1 : LT.lt 1 a
                                          n j : Nat
                                          h : LE.le j (HMul.hMul 2 n)
                                          ⊢ LE.le (HMul.hMul 2 n) (HMul.hMul 2 n)
                                        -/
  have h' : j ≤ 2 * n := le_trans h (by rw [Nat.succ_mul])
                                        /-
                                          🎉 no goals
                                        -/
  ModEq.add_right_cancel' (xn a1 (2 * n - j)) <| by
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      h h' : LE.le j (HMul.hMul 2 n)
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 4 n) j)) ( …
    -/
    refine @ModEq.trans _ _ 0 _ ?_ (by rw [add_comm]; exact (xn_modEq_x2n_sub _ h).symm)
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      h h' : LE.le j (HMul.hMul 2 n)
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 4 n) j)) ( …
    -/
    rw [show 4 * n = 2 * n + 2 * n from right_distrib 2 2 n, add_tsub_assoc_of_le h']
    /-
      a : Nat
      a1 : LT.lt 1 a
      n j : Nat
      h h' : LE.le j (HMul.hMul 2 n)
      ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd (HMul.hMul 2 n) (HSub …
    -/
    apply xn_modEq_x2n_add
    /-
      🎉 no goals
    -/


theorem eq_of_xn_modEq_lem1 {i n} : ∀ {j}, i < j → j < n → xn a1 i % xn a1 n < xn a1 j % xn a1 n
  | 0, ij, _ => absurd ij (Nat.not_lt_zero _)
  | j + 1, ij, jn => by
    suffices xn a1 j % xn a1 n < xn a1 (j + 1) % xn a1 n from
      (lt_or_eq_of_le (Nat.le_of_succ_le_succ ij)).elim
        (fun h => lt_trans (eq_of_xn_modEq_lem1 h (le_of_lt jn)) this) fun h => by
        rw [h]; exact this
    rw [Nat.mod_eq_of_lt (strictMono_x _ (Nat.lt_of_succ_lt jn)),
        Nat.mod_eq_of_lt (strictMono_x _ jn)]
    /-
      a : Nat
      a1 : LT.lt 1 a
      i n j : Nat
      ij : LT.lt i (HAdd.hAdd j 1)
      jn : LT.lt (HAdd.hAdd j 1) n
      ⊢ LT.lt (Pell.xn a1 j) (Pell.xn a1 (HAdd.hAdd j 1))
    -/
    exact strictMono_x _ (Nat.lt_succ_self _)
    /-
      🎉 no goals
    -/


theorem eq_of_xn_modEq_lem2 {n} (h : 2 * xn a1 n = xn a1 (n + 1)) : a = 2 ∧ n = 0 := by
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    h : Eq (HMul.hMul 2 (Pell.xn a1 n)) (Pell.xn a1 (HAdd.hAdd n 1))
    ⊢ And (Eq a 2) (Eq n 0)
  -/
  rw [xn_succ, mul_comm] at h
  have : n = 0 :=
    n.eq_zero_or_pos.resolve_right fun np =>
      _root_.ne_of_lt
        (lt_of_le_of_lt (Nat.mul_le_mul_left _ a1)
          (Nat.lt_add_of_pos_right <| mul_pos (d_pos a1) (strictMono_y a1 np)))
        h
  /-
    a : Nat
    a1 : LT.lt 1 a
    n : Nat
    h : Eq (HMul.hMul (Pell.xn a1 n) 2) (HAdd.hAdd (HMul.hMul (Pell.xn a1 n) a) (H …
    this : Eq n 0
    ⊢ And (Eq a 2) (Eq n 0)
  -/
  cases this; simp at h; exact ⟨h.symm, rfl⟩
                         /-
                           🎉 no goals
                         -/


theorem eq_of_xn_modEq_lem3 {i n} (npos : 0 < n) :
    ∀ {j}, i < j → j ≤ 2 * n → j ≠ n → ¬(a = 2 ∧ n = 1 ∧ i = 0 ∧ j = 2) →
        xn a1 i % xn a1 n < xn a1 j % xn a1 n
  | 0, ij, _, _, _ => absurd ij (Nat.not_lt_zero _)
  | j + 1, ij, j2n, jnn, ntriv =>
    have lem2 : ∀ k > n, k ≤ 2 * n → (↑(xn a1 k % xn a1 n) : ℤ) =
        xn a1 n - xn a1 (2 * n - k) := fun k kn k2n => by
      let k2nl :=
        lt_of_add_lt_add_right <|
          show 2 * n - k + k < n + k by
            rw [tsub_add_cancel_of_le]
            · rw [two_mul]
              exact add_lt_add_left kn n
            exact k2n
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        ⊢ Eq (↑(HMod.hMod (Pell.xn a1 k) (Pell.xn a1 n))) (HSub.hSub ↑(Pell.xn a1 n) ↑ …
      -/
      have xle : xn a1 (2 * n - k) ≤ xn a1 n := le_of_lt <| strictMono_x a1 k2nl
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        ⊢ Eq (↑(HMod.hMod (Pell.xn a1 k) (Pell.xn a1 n))) (HSub.hSub ↑(Pell.xn a1 n) ↑ …
      -/
      suffices xn a1 k % xn a1 n = xn a1 n - xn a1 (2 * n - k) by rw [this, Int.ofNat_sub xle]
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        ⊢ Eq (HMod.hMod (Pell.xn a1 k) (Pell.xn a1 n)) (HSub.hSub (Pell.xn a1 n) (Pell …
      -/
      rw [← Nat.mod_eq_of_lt (Nat.sub_lt (x_pos a1 n) (x_pos a1 (2 * n - k)))]
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        ⊢ Eq (HMod.hMod (Pell.xn a1 k) (Pell.xn a1 n)) (HMod.hMod (HSub.hSub (Pell.xn  …
      -/
      apply ModEq.add_right_cancel' (xn a1 (2 * n - k))
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 k) (Pell.xn a1 (HSub.hSub (HMul. …
      -/
      rw [tsub_add_cancel_of_le xle]
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 k) (Pell.xn a1 (HSub.hSub (HMul. …
      -/
      have t := xn_modEq_x2n_sub_lem a1 k2nl.le
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        t : (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) (HS …
        ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 k) (Pell.xn a1 (HSub.hSub (HMul. …
      -/
      rw [tsub_tsub_cancel_of_le k2n] at t
      /-
        a : Nat
        a1 : LT.lt 1 a
        i n : Nat
        npos : LT.lt 0 n
        j : Nat
        ij : LT.lt i (HAdd.hAdd j 1)
        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
        jnn : Ne (HAdd.hAdd j 1) n
        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
        k : Nat
        kn : GT.gt k n
        k2n : LE.le k (HMul.hMul 2 n)
        k2nl : LT.lt (HSub.hSub (HMul.hMul 2 n) k) n := lt_of_add_lt_add_right (letFun …
        xle : LE.le (Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) k)) (Pell.xn a1 n)
        t : (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 k) (Pell.xn a1 (HSub.hSub (HMu …
        ⊢ (Pell.xn a1 n).ModEq (HAdd.hAdd (Pell.xn a1 k) (Pell.xn a1 (HSub.hSub (HMul. …
      -/
      exact t.trans dvd_rfl.zero_modEq_nat
      /-
        🎉 no goals
      -/
    (lt_trichotomy j n).elim (fun jn : j < n => eq_of_xn_modEq_lem1 _ ij (lt_of_le_of_ne jn jnn))
      fun o =>
      o.elim
        (fun jn : j = n => by
          /-
            a : Nat
            a1 : LT.lt 1 a
            i n : Nat
            npos : LT.lt 0 n
            j : Nat
            ij : LT.lt i (HAdd.hAdd j 1)
            j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
            jnn : Ne (HAdd.hAdd j 1) n
            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
            lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
            o : Or (Eq j n) (LT.lt n j)
            jn : Eq j n
            ⊢ LT.lt (HMod.hMod (Pell.xn a1 i) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 (HAdd …
          -/
          cases jn
          /-
            case refl
            a : Nat
            a1 : LT.lt 1 a
            i n : Nat
            npos : LT.lt 0 n
            lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
            ij : LT.lt i (HAdd.hAdd n 1)
            j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
            jnn : Ne (HAdd.hAdd n 1) n
            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
            o : Or (Eq n n) (LT.lt n n)
            ⊢ LT.lt (HMod.hMod (Pell.xn a1 i) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 (HAdd …
          -/
          apply Int.lt_of_ofNat_lt_ofNat
          rw [lem2 (n + 1) (Nat.lt_succ_self _) j2n,
            show 2 * n - (n + 1) = n - 1 by
              rw [two_mul, tsub_add_eq_tsub_tsub, add_tsub_cancel_right]]
          /-
            case refl.a
            a : Nat
            a1 : LT.lt 1 a
            i n : Nat
            npos : LT.lt 0 n
            lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
            ij : LT.lt i (HAdd.hAdd n 1)
            j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
            jnn : Ne (HAdd.hAdd n 1) n
            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
            o : Or (Eq n n) (LT.lt n n)
            ⊢ LT.lt (↑(HMod.hMod (Pell.xn a1 i) (Pell.xn a1 n))) (HSub.hSub ↑(Pell.xn a1 n …
          -/
          refine lt_sub_left_of_add_lt (Int.ofNat_lt_ofNat_of_lt ?_)
          /-
            case refl.a
            a : Nat
            a1 : LT.lt 1 a
            i n : Nat
            npos : LT.lt 0 n
            lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
            ij : LT.lt i (HAdd.hAdd n 1)
            j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
            jnn : Ne (HAdd.hAdd n 1) n
            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
            o : Or (Eq n n) (LT.lt n n)
            ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (HMod.hMod (Pell.xn a1 i) (Pel …
          -/
          rcases lt_or_eq_of_le <| Nat.le_of_succ_le_succ ij with lin | ein
            /-
              case refl.a.inl
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              ij : LT.lt i (HAdd.hAdd n 1)
              j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd n 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
              o : Or (Eq n n) (LT.lt n n)
              lin : LT.lt i n
              ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (HMod.hMod (Pell.xn a1 i) (Pel …
            -/
          · rw [Nat.mod_eq_of_lt (strictMono_x _ lin)]
            have ll : xn a1 (n - 1) + xn a1 (n - 1) ≤ xn a1 n := by
              rw [← two_mul, mul_comm,
                show xn a1 n = xn a1 (n - 1 + 1) by rw [tsub_add_cancel_of_le (succ_le_of_lt npos)],
                xn_succ]
              exact le_trans (Nat.mul_le_mul_left _ a1) (Nat.le_add_right _ _)
            /-
              case refl.a.inl
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              ij : LT.lt i (HAdd.hAdd n 1)
              j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd n 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
              o : Or (Eq n n) (LT.lt n n)
              lin : LT.lt i n
              ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
              ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 i)) (Pell.xn a1 n)
            -/
            have npm : (n - 1).succ = n := Nat.succ_pred_eq_of_pos npos
            have il : i ≤ n - 1 := by
              apply Nat.le_of_succ_le_succ
              rw [npm]
              exact lin
            /-
              case refl.a.inl
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              ij : LT.lt i (HAdd.hAdd n 1)
              j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd n 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
              o : Or (Eq n n) (LT.lt n n)
              lin : LT.lt i n
              ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
              npm : Eq (HSub.hSub n 1).succ n
              il : LE.le i (HSub.hSub n 1)
              ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 i)) (Pell.xn a1 n)
            -/
            rcases lt_or_eq_of_le il with ill | ile
              /-
                case refl.a.inl.inl
                a : Nat
                a1 : LT.lt 1 a
                i n : Nat
                npos : LT.lt 0 n
                lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                ij : LT.lt i (HAdd.hAdd n 1)
                j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
                jnn : Ne (HAdd.hAdd n 1) n
                ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
                o : Or (Eq n n) (LT.lt n n)
                lin : LT.lt i n
                ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
                npm : Eq (HSub.hSub n 1).succ n
                il : LE.le i (HSub.hSub n 1)
                ill : LT.lt i (HSub.hSub n 1)
                ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 i)) (Pell.xn a1 n)
              -/
            · exact lt_of_lt_of_le (Nat.add_lt_add_left (strictMono_x a1 ill) _) ll
              /-
                🎉 no goals
              -/
              /-
                case refl.a.inl.inr
                a : Nat
                a1 : LT.lt 1 a
                i n : Nat
                npos : LT.lt 0 n
                lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                ij : LT.lt i (HAdd.hAdd n 1)
                j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
                jnn : Ne (HAdd.hAdd n 1) n
                ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
                o : Or (Eq n n) (LT.lt n n)
                lin : LT.lt i n
                ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
                npm : Eq (HSub.hSub n 1).succ n
                il : LE.le i (HSub.hSub n 1)
                ile : Eq i (HSub.hSub n 1)
                ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 i)) (Pell.xn a1 n)
              -/
            · rw [ile]
              /-
                case refl.a.inl.inr
                a : Nat
                a1 : LT.lt 1 a
                i n : Nat
                npos : LT.lt 0 n
                lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                ij : LT.lt i (HAdd.hAdd n 1)
                j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
                jnn : Ne (HAdd.hAdd n 1) n
                ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
                o : Or (Eq n n) (LT.lt n n)
                lin : LT.lt i n
                ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
                npm : Eq (HSub.hSub n 1).succ n
                il : LE.le i (HSub.hSub n 1)
                ile : Eq i (HSub.hSub n 1)
                ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1)))  …
              -/
              apply lt_of_le_of_ne ll
              /-
                case refl.a.inl.inr
                a : Nat
                a1 : LT.lt 1 a
                i n : Nat
                npos : LT.lt 0 n
                lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                ij : LT.lt i (HAdd.hAdd n 1)
                j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
                jnn : Ne (HAdd.hAdd n 1) n
                ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
                o : Or (Eq n n) (LT.lt n n)
                lin : LT.lt i n
                ll : LE.le (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1) …
                npm : Eq (HSub.hSub n 1).succ n
                il : LE.le i (HSub.hSub n 1)
                ile : Eq i (HSub.hSub n 1)
                ⊢ Ne (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 (HSub.hSub n 1))) (Pe …
              -/
              rw [← two_mul]
              exact fun e =>
                ntriv <| by
                  let ⟨a2, s1⟩ :=
                    @eq_of_xn_modEq_lem2 _ a1 (n - 1)
                      (by rwa [tsub_add_cancel_of_le (succ_le_of_lt npos)])
                  have n1 : n = 1 := le_antisymm (tsub_eq_zero_iff_le.mp s1) npos
                  rw [ile, a2, n1]; exact ⟨rfl, rfl, rfl, rfl⟩
            /-
              case refl.a.inr
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              ij : LT.lt i (HAdd.hAdd n 1)
              j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd n 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
              o : Or (Eq n n) (LT.lt n n)
              ein : Eq i n
              ⊢ LT.lt (HAdd.hAdd (Pell.xn a1 (HSub.hSub n 1)) (HMod.hMod (Pell.xn a1 i) (Pel …
            -/
          · rw [ein, Nat.mod_self, add_zero]
            /-
              case refl.a.inr
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              ij : LT.lt i (HAdd.hAdd n 1)
              j2n : LE.le (HAdd.hAdd n 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd n 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd n 1) 2))))
              o : Or (Eq n n) (LT.lt n n)
              ein : Eq i n
              ⊢ LT.lt (Pell.xn a1 (HSub.hSub n 1)) (Pell.xn a1 n)
            -/
            exact strictMono_x _ (Nat.pred_lt npos.ne'))
            /-
              🎉 no goals
            -/
        fun jn : j > n =>
        have lem1 : j ≠ n → xn a1 j % xn a1 n < xn a1 (j + 1) % xn a1 n →
            xn a1 i % xn a1 n < xn a1 (j + 1) % xn a1 n :=
          fun jn s =>
          (lt_or_eq_of_le (Nat.le_of_succ_le_succ ij)).elim
            (fun h =>
              lt_trans
                (eq_of_xn_modEq_lem3 npos h (le_of_lt (Nat.lt_of_succ_le j2n)) jn
                    fun ⟨_, n1, _, j2⟩ => by
                      /-
                        a : Nat
                        a1 : LT.lt 1 a
                        i n : Nat
                        npos : LT.lt 0 n
                        j : Nat
                        ij : LT.lt i (HAdd.hAdd j 1)
                        j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
                        jnn : Ne (HAdd.hAdd j 1) n
                        ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
                        lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                        o : Or (Eq j n) (LT.lt n j)
                        jn✝ : GT.gt j n
                        jn : Ne j n
                        s : LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 (HA …
                        h : LT.lt i j
                        x✝ : And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq j 2)))
                        left✝¹ : Eq a 2
                        n1 : Eq n 1
                        left✝ : Eq i 0
                        j2 : Eq j 2
                        ⊢ False
                      -/
                      rw [n1, j2] at j2n; exact absurd j2n (by decide))
                                          /-
                                            🎉 no goals
                                          -/
                s)
                        /-
                          a : Nat
                          a1 : LT.lt 1 a
                          i n : Nat
                          npos : LT.lt 0 n
                          j : Nat
                          ij : LT.lt i (HAdd.hAdd j 1)
                          j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
                          jnn : Ne (HAdd.hAdd j 1) n
                          ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
                          lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
                          o : Or (Eq j n) (LT.lt n j)
                          jn✝ : GT.gt j n
                          jn : Ne j n
                          s : LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 (HA …
                          h : Eq i j
                          ⊢ LT.lt (HMod.hMod (Pell.xn a1 i) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 (HAdd …
                        -/
            fun h => by rw [h]; exact s
                                /-
                                  🎉 no goals
                                -/
        lem1 (_root_.ne_of_gt jn) <|
          Int.lt_of_ofNat_lt_ofNat <| by
            /-
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              j : Nat
              ij : LT.lt i (HAdd.hAdd j 1)
              j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd j 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              o : Or (Eq j n) (LT.lt n j)
              jn : GT.gt j n
              lem1 : Ne j n → LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pe …
              ⊢ LT.lt ↑(HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) ↑(HMod.hMod (Pell.xn a1 (HA …
            -/
            rw [lem2 j jn (le_of_lt j2n), lem2 (j + 1) (Nat.le_succ_of_le jn) j2n]
            /-
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              j : Nat
              ij : LT.lt i (HAdd.hAdd j 1)
              j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd j 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              o : Or (Eq j n) (LT.lt n j)
              jn : GT.gt j n
              lem1 : Ne j n → LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pe …
              ⊢ LT.lt (HSub.hSub ↑(Pell.xn a1 n) ↑(Pell.xn a1 (HSub.hSub (HMul.hMul 2 n) j)) …
            -/
            refine sub_lt_sub_left (Int.ofNat_lt_ofNat_of_lt <| strictMono_x _ ?_) _
            /-
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              j : Nat
              ij : LT.lt i (HAdd.hAdd j 1)
              j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd j 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              o : Or (Eq j n) (LT.lt n j)
              jn : GT.gt j n
              lem1 : Ne j n → LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pe …
              ⊢ LT.lt (HSub.hSub (HMul.hMul 2 n) (HAdd.hAdd j 1)) (HSub.hSub (HMul.hMul 2 n) …
            -/
            rw [Nat.sub_succ]
            /-
              a : Nat
              a1 : LT.lt 1 a
              i n : Nat
              npos : LT.lt 0 n
              j : Nat
              ij : LT.lt i (HAdd.hAdd j 1)
              j2n : LE.le (HAdd.hAdd j 1) (HMul.hMul 2 n)
              jnn : Ne (HAdd.hAdd j 1) n
              ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq (HAdd.hAdd j 1) 2))))
              lem2 : ∀ (k : Nat), GT.gt k n → LE.le k (HMul.hMul 2 n) → Eq (↑(HMod.hMod (Pel …
              o : Or (Eq j n) (LT.lt n j)
              jn : GT.gt j n
              lem1 : Ne j n → LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pe …
              ⊢ LT.lt (HSub.hSub (HMul.hMul 2 n) j).pred (HSub.hSub (HMul.hMul 2 n) j)
            -/
            exact Nat.pred_lt (_root_.ne_of_gt <| tsub_pos_of_lt j2n)
            /-
              🎉 no goals
            -/


theorem eq_of_xn_modEq_le {i j n} (ij : i ≤ j) (j2n : j ≤ 2 * n)
    (h : xn a1 i ≡ xn a1 j [MOD xn a1 n])
    (ntriv : ¬(a = 2 ∧ n = 1 ∧ i = 0 ∧ j = 2)) : i = j :=
                          /-
                            a : Nat
                            a1 : LT.lt 1 a
                            i j n : Nat
                            ij : LE.le i j
                            j2n : LE.le j (HMul.hMul 2 n)
                            h : (Pell.xn a1 n).ModEq (Pell.xn a1 i) (Pell.xn a1 j)
                            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq j 2))))
                            npos : Eq n 0
                            ⊢ Eq i j
                          -/
  if npos : n = 0 then by simp_all
                          /-
                            🎉 no goals
                          -/
  else
    (lt_or_eq_of_le ij).resolve_left fun ij' =>
      if jn : j = n then by
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ij : LE.le i j
          j2n : LE.le j (HMul.hMul 2 n)
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 i) (Pell.xn a1 j)
          ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq j 2))))
          npos : Not (Eq n 0)
          ij' : LT.lt i j
          jn : Eq j n
          ⊢ False
        -/
        refine _root_.ne_of_gt ?_ h
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ij : LE.le i j
          j2n : LE.le j (HMul.hMul 2 n)
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 i) (Pell.xn a1 j)
          ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq j 2))))
          npos : Not (Eq n 0)
          ij' : LT.lt i j
          jn : Eq j n
          ⊢ LT.lt (HMod.hMod (Pell.xn a1 j) (Pell.xn a1 n)) (HMod.hMod (Pell.xn a1 i) (P …
        -/
        rw [jn, Nat.mod_self]
        have x0 : 0 < xn a1 0 % xn a1 n := by
          rw [Nat.mod_eq_of_lt (strictMono_x a1 (Nat.pos_of_ne_zero npos))]
          exact Nat.succ_pos _
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ij : LE.le i j
          j2n : LE.le j (HMul.hMul 2 n)
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 i) (Pell.xn a1 j)
          ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq i 0) (Eq j 2))))
          npos : Not (Eq n 0)
          ij' : LT.lt i j
          jn : Eq j n
          x0 : LT.lt 0 (HMod.hMod (Pell.xn a1 0) (Pell.xn a1 n))
          ⊢ LT.lt 0 (HMod.hMod (Pell.xn a1 i) (Pell.xn a1 n))
        -/
        cases' i with i
          /-
            case zero
            a : Nat
            a1 : LT.lt 1 a
            j n : Nat
            j2n : LE.le j (HMul.hMul 2 n)
            npos : Not (Eq n 0)
            jn : Eq j n
            x0 : LT.lt 0 (HMod.hMod (Pell.xn a1 0) (Pell.xn a1 n))
            ij : LE.le 0 j
            h : (Pell.xn a1 n).ModEq (Pell.xn a1 0) (Pell.xn a1 j)
            ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq 0 0) (Eq j 2))))
            ij' : LT.lt 0 j
            ⊢ LT.lt 0 (HMod.hMod (Pell.xn a1 0) (Pell.xn a1 n))
          -/
        · exact x0
          /-
            🎉 no goals
          -/
        /-
          case succ
          a : Nat
          a1 : LT.lt 1 a
          j n : Nat
          j2n : LE.le j (HMul.hMul 2 n)
          npos : Not (Eq n 0)
          jn : Eq j n
          x0 : LT.lt 0 (HMod.hMod (Pell.xn a1 0) (Pell.xn a1 n))
          i : Nat
          ij : LE.le (HAdd.hAdd i 1) j
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd i 1)) (Pell.xn a1 j)
          ntriv : Not (And (Eq a 2) (And (Eq n 1) (And (Eq (HAdd.hAdd i 1) 0) (Eq j 2))))
          ij' : LT.lt (HAdd.hAdd i 1) j
          ⊢ LT.lt 0 (HMod.hMod (Pell.xn a1 (HAdd.hAdd i 1)) (Pell.xn a1 n))
        -/
        rw [jn] at ij'
        exact
          x0.trans
            (eq_of_xn_modEq_lem3 _ (Nat.pos_of_ne_zero npos) (Nat.succ_pos _) (le_trans ij j2n)
              (_root_.ne_of_lt ij') fun ⟨_, n1, _, i2⟩ => by
              rw [n1, i2] at ij'; exact absurd ij' (by decide))
      else _root_.ne_of_lt (eq_of_xn_modEq_lem3 a1 (Nat.pos_of_ne_zero npos) ij' j2n jn ntriv) h


theorem eq_of_xn_modEq {i j n} (i2n : i ≤ 2 * n) (j2n : j ≤ 2 * n)
    (h : xn a1 i ≡ xn a1 j [MOD xn a1 n])
    (ntriv : a = 2 → n = 1 → (i = 0 → j ≠ 2) ∧ (i = 2 → j ≠ 0)) : i = j :=
  (le_total i j).elim
    (fun ij => eq_of_xn_modEq_le a1 ij j2n h fun ⟨a2, n1, i0, j2⟩ => (ntriv a2 n1).left i0 j2)
    fun ij =>
    (eq_of_xn_modEq_le a1 ij i2n h.symm fun ⟨a2, n1, j0, i2⟩ => (ntriv a2 n1).right i2 j0).symm


theorem eq_of_xn_modEq' {i j n} (ipos : 0 < i) (hin : i ≤ n) (j4n : j ≤ 4 * n)
    (h : xn a1 j ≡ xn a1 i [MOD xn a1 n]) : j = i ∨ j + i = 4 * n :=
                             /-
                               a : Nat
                               a1 : LT.lt 1 a
                               i j n : Nat
                               ipos : LT.lt 0 i
                               hin : LE.le i n
                               j4n : LE.le j (HMul.hMul 4 n)
                               h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                               ⊢ LE.le i (HMul.hMul 2 n)
                             -/
  have i2n : i ≤ 2 * n := by apply le_trans hin; rw [two_mul]; apply Nat.le_add_left
                                                               /-
                                                                 🎉 no goals
                                                               -/
  (le_or_gt j (2 * n)).imp
    (fun j2n : j ≤ 2 * n =>
      eq_of_xn_modEq a1 j2n i2n h fun _ n1 =>
                        /-
                          a : Nat
                          a1 : LT.lt 1 a
                          i j n : Nat
                          ipos : LT.lt 0 i
                          hin : LE.le i n
                          j4n : LE.le j (HMul.hMul 4 n)
                          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                          i2n : LE.le i (HMul.hMul 2 n)
                          j2n : LE.le j (HMul.hMul 2 n)
                          x✝¹ : Eq a 2
                          n1 : Eq n 1
                          x✝ : Eq j 0
                          i2 : Eq i 2
                          ⊢ False
                        -/
        ⟨fun _ i2 => by rw [n1, i2] at hin; exact absurd hin (by decide), fun _ i0 =>
                                            /-
                                              🎉 no goals
                                            -/
          _root_.ne_of_gt ipos i0⟩)
    fun j2n : 2 * n < j =>
                              /-
                                a : Nat
                                a1 : LT.lt 1 a
                                i j n : Nat
                                ipos : LT.lt 0 i
                                hin : LE.le i n
                                j4n : LE.le j (HMul.hMul 4 n)
                                h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                                i2n : LE.le i (HMul.hMul 2 n)
                                j2n : LT.lt (HMul.hMul 2 n) j
                                this : Eq i (HSub.hSub (HMul.hMul 4 n) j)
                                ⊢ Eq (HAdd.hAdd j i) (HMul.hMul 4 n)
                              -/
                                        /-
                                          a : Nat
                                          a1 : LT.lt 1 a
                                          i j n : Nat
                                          ipos : LT.lt 0 i
                                          hin : LE.le i n
                                          j4n : LE.le j (HMul.hMul 4 n)
                                          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                                          i2n : LE.le i (HMul.hMul 2 n)
                                          j2n : LT.lt (HMul.hMul 2 n) j
                                          ⊢ LE.le (HSub.hSub (HMul.hMul 4 n) j) (HMul.hMul 2 n)
                                        -/
    suffices i = 4 * n - j by rw [this, add_tsub_cancel_of_le j4n]
                                        /-
                                          🎉 no goals
                                        -/
                              /-
                                🎉 no goals
                              -/
    have j42n : 4 * n - j ≤ 2 * n := by omega
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ipos : LT.lt 0 i
          hin : LE.le i n
          j4n : LE.le j (HMul.hMul 4 n)
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
          i2n : LE.le i (HMul.hMul 2 n)
          j2n : LT.lt (HMul.hMul 2 n) j
          j42n : LE.le (HSub.hSub (HMul.hMul 4 n) j) (HMul.hMul 2 n)
          ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 (HSub.hSub (HMul.hMul 4 n) j))
        -/
    eq_of_xn_modEq a1 i2n j42n
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ipos : LT.lt 0 i
          hin : LE.le i n
          j4n : LE.le j (HMul.hMul 4 n)
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
          i2n : LE.le i (HMul.hMul 2 n)
          j2n : LT.lt (HMul.hMul 2 n) j
          j42n : LE.le (HSub.hSub (HMul.hMul 4 n) j) (HMul.hMul 2 n)
          t : (Pell.xn a1 n).ModEq (Pell.xn a1 (HSub.hSub (HMul.hMul 4 n) (HSub.hSub (HM …
          ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 (HSub.hSub (HMul.hMul 4 n) j))
        -/
      (h.symm.trans <| by
        /-
          🎉 no goals
        -/
          /-
            a : Nat
            a1 : LT.lt 1 a
            i j n : Nat
            ipos : LT.lt 0 i
            hin : LE.le i n
            j4n : LE.le j (HMul.hMul 4 n)
            h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
            i2n : LE.le i (HMul.hMul 2 n)
            j2n : LT.lt (HMul.hMul 2 n) j
            j42n : LE.le (HSub.hSub (HMul.hMul 4 n) j) (HMul.hMul 2 n)
            ⊢ Eq a 2 → Eq n 1 → And (Eq i 0 → Ne (HSub.hSub (HMul.hMul 4 n) j) 2) (Eq i 2  …
          -/
        let t := xn_modEq_x4n_sub a1 j42n
          /-
            🎉 no goals
          -/
        rwa [tsub_tsub_cancel_of_le j4n] at t)
      (by omega)


theorem modEq_of_xn_modEq {i j n} (ipos : 0 < i) (hin : i ≤ n)
    (h : xn a1 j ≡ xn a1 i [MOD xn a1 n]) :
    j ≡ i [MOD 4 * n] ∨ j + i ≡ 0 [MOD 4 * n] :=
  let j' := j % (4 * n)
                                     /-
                                       a : Nat
                                       a1 : LT.lt 1 a
                                       i j n : Nat
                                       ipos : LT.lt 0 i
                                       hin : LE.le i n
                                       h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                                       j' : Nat := HMod.hMod j (HMul.hMul 4 n)
                                       ⊢ LT.lt 0 4
                                     -/
  have n4 : 0 < 4 * n := mul_pos (by decide) (ipos.trans_le hin)
                                     /-
                                       🎉 no goals
                                     -/
  have jl : j' < 4 * n := Nat.mod_lt _ n4
                                     /-
                                       a : Nat
                                       a1 : LT.lt 1 a
                                       i j n : Nat
                                       ipos : LT.lt 0 i
                                       hin : LE.le i n
                                       h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                                       j' : Nat := HMod.hMod j (HMul.hMul 4 n)
                                       n4 : LT.lt 0 (HMul.hMul 4 n)
                                       jl : LT.lt j' (HMul.hMul 4 n)
                                       ⊢ (HMul.hMul 4 n).ModEq j j'
                                     -/
  have jj : j ≡ j' [MOD 4 * n] := by delta ModEq; rw [Nat.mod_eq_of_lt jl]
                                                  /-
                                                    🎉 no goals
                                                  -/
  have : ∀ j q, xn a1 (j + 4 * n * q) ≡ xn a1 j [MOD xn a1 n] := by
    /-
      a : Nat
      a1 : LT.lt 1 a
      i j n : Nat
      ipos : LT.lt 0 i
      hin : LE.le i n
      h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
      j' : Nat := HMod.hMod j (HMul.hMul 4 n)
      n4 : LT.lt 0 (HMul.hMul 4 n)
      jl : LT.lt j' (HMul.hMul 4 n)
      jj : (HMul.hMul 4 n).ModEq j j'
      ⊢ ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul (HMu …
    -/
    intro j q; induction' q with q IH
      /-
        case zero
        a : Nat
        a1 : LT.lt 1 a
        i j✝ n : Nat
        ipos : LT.lt 0 i
        hin : LE.le i n
        h : (Pell.xn a1 n).ModEq (Pell.xn a1 j✝) (Pell.xn a1 i)
        j' : Nat := HMod.hMod j✝ (HMul.hMul 4 n)
        n4 : LT.lt 0 (HMul.hMul 4 n)
        jl : LT.lt j' (HMul.hMul 4 n)
        jj : (HMul.hMul 4 n).ModEq j✝ j'
        j : Nat
        ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul (HMul.hMul 4 n) 0)) …
      -/
    · simp [ModEq.refl]
      /-
        🎉 no goals
      -/
    /-
      case succ
      a : Nat
      a1 : LT.lt 1 a
      i j✝ n : Nat
      ipos : LT.lt 0 i
      hin : LE.le i n
      h : (Pell.xn a1 n).ModEq (Pell.xn a1 j✝) (Pell.xn a1 i)
      j' : Nat := HMod.hMod j✝ (HMul.hMul 4 n)
      n4 : LT.lt 0 (HMul.hMul 4 n)
      jl : LT.lt j' (HMul.hMul 4 n)
      jj : (HMul.hMul 4 n).ModEq j✝ j'
      j q : Nat
      IH : (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul (HMul.hMul 4 n)  …
      ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul (HMul.hMul 4 n) (HA …
    -/
    rw [Nat.mul_succ, ← add_assoc, add_comm]
    /-
      case succ
      a : Nat
      a1 : LT.lt 1 a
      i j✝ n : Nat
      ipos : LT.lt 0 i
      hin : LE.le i n
      h : (Pell.xn a1 n).ModEq (Pell.xn a1 j✝) (Pell.xn a1 i)
      j' : Nat := HMod.hMod j✝ (HMul.hMul 4 n)
      n4 : LT.lt 0 (HMul.hMul 4 n)
      jl : LT.lt j' (HMul.hMul 4 n)
      jj : (HMul.hMul 4 n).ModEq j✝ j'
      j q : Nat
      IH : (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul (HMul.hMul 4 n)  …
      ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd (HMul.hMul 4 n) (HAdd.hAdd j (HM …
    -/
    exact (xn_modEq_x4n_add _ _ _).trans IH
    /-
      🎉 no goals
    -/
                                /-
                                  a : Nat
                                  a1 : LT.lt 1 a
                                  i j n : Nat
                                  ipos : LT.lt 0 i
                                  hin : LE.le i n
                                  h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
                                  j' : Nat := HMod.hMod j (HMul.hMul 4 n)
                                  n4 : LT.lt 0 (HMul.hMul 4 n)
                                  jl : LT.lt j' (HMul.hMul 4 n)
                                  jj : (HMul.hMul 4 n).ModEq j j'
                                  this : ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul …
                                  ji : Eq j' i
                                  ⊢ (HMul.hMul 4 n).ModEq j i
                                -/
  Or.imp (fun ji : j' = i => by rwa [← ji])
                                /-
                                  🎉 no goals
                                -/
    (fun ji : j' + i = 4 * n =>
      (jj.add_right _).trans <| by
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ipos : LT.lt 0 i
          hin : LE.le i n
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
          j' : Nat := HMod.hMod j (HMul.hMul 4 n)
          n4 : LT.lt 0 (HMul.hMul 4 n)
          jl : LT.lt j' (HMul.hMul 4 n)
          jj : (HMul.hMul 4 n).ModEq j j'
          this : ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul …
          ji : Eq (HAdd.hAdd j' i) (HMul.hMul 4 n)
          ⊢ (HMul.hMul 4 n).ModEq (HAdd.hAdd j' i) 0
        -/
        rw [ji]
        /-
          a : Nat
          a1 : LT.lt 1 a
          i j n : Nat
          ipos : LT.lt 0 i
          hin : LE.le i n
          h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
          j' : Nat := HMod.hMod j (HMul.hMul 4 n)
          n4 : LT.lt 0 (HMul.hMul 4 n)
          jl : LT.lt j' (HMul.hMul 4 n)
          jj : (HMul.hMul 4 n).ModEq j j'
          this : ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul …
          ji : Eq (HAdd.hAdd j' i) (HMul.hMul 4 n)
          ⊢ (HMul.hMul 4 n).ModEq (HMul.hMul 4 n) 0
        -/
        exact dvd_rfl.modEq_zero_nat)
        /-
          🎉 no goals
        -/
    (eq_of_xn_modEq' a1 ipos hin jl.le <|
      (h.symm.trans <| by
          /-
            a : Nat
            a1 : LT.lt 1 a
            i j n : Nat
            ipos : LT.lt 0 i
            hin : LE.le i n
            h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
            j' : Nat := HMod.hMod j (HMul.hMul 4 n)
            n4 : LT.lt 0 (HMul.hMul 4 n)
            jl : LT.lt j' (HMul.hMul 4 n)
            jj : (HMul.hMul 4 n).ModEq j j'
            this : ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul …
            ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 j')
          -/
          rw [← Nat.mod_add_div j (4 * n)]
          /-
            a : Nat
            a1 : LT.lt 1 a
            i j n : Nat
            ipos : LT.lt 0 i
            hin : LE.le i n
            h : (Pell.xn a1 n).ModEq (Pell.xn a1 j) (Pell.xn a1 i)
            j' : Nat := HMod.hMod j (HMul.hMul 4 n)
            n4 : LT.lt 0 (HMul.hMul 4 n)
            jl : LT.lt j' (HMul.hMul 4 n)
            jj : (HMul.hMul 4 n).ModEq j j'
            this : ∀ (j q : Nat), (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd j (HMul.hMul …
            ⊢ (Pell.xn a1 n).ModEq (Pell.xn a1 (HAdd.hAdd (HMod.hMod j (HMul.hMul 4 n)) (H …
          -/
          exact this j' _).symm)
          /-
            🎉 no goals
          -/


theorem xy_modEq_of_modEq {a b c} (a1 : 1 < a) (b1 : 1 < b) (h : a ≡ b [MOD c]) :
    ∀ n, xn a1 n ≡ xn b1 n [MOD c] ∧ yn a1 n ≡ yn b1 n [MOD c]
            /-
              a b c : Nat
              a1 : LT.lt 1 a
              b1 : LT.lt 1 b
              h : c.ModEq a b
              ⊢ And (c.ModEq (Pell.xn a1 0) (Pell.xn b1 0)) (c.ModEq (Pell.yn a1 0) (Pell.yn …
            -/
                            /-
                              🎉 no goals
                            -/
  | 0 => by constructor <;> rfl
                            /-
                              🎉 no goals
                            -/
            /-
              a b c : Nat
              a1 : LT.lt 1 a
              b1 : LT.lt 1 b
              h : c.ModEq a b
              ⊢ And (c.ModEq (Pell.xn a1 1) (Pell.xn b1 1)) (c.ModEq (Pell.yn a1 1) (Pell.yn …
            -/
  | 1 => by simpa using ⟨h, ModEq.refl 1⟩
            /-
              🎉 no goals
            -/
  | n + 2 =>
    ⟨(xy_modEq_of_modEq a1 b1 h n).left.add_right_cancel <| by
        /-
          a b c : Nat
          a1 : LT.lt 1 a
          b1 : LT.lt 1 b
          h : c.ModEq a b
          n : Nat
          ⊢ c.ModEq (HAdd.hAdd (Pell.xn a1 (HAdd.hAdd n 2)) (Pell.xn a1 n)) (HAdd.hAdd ( …
        -/
        rw [xn_succ_succ a1, xn_succ_succ b1]
        /-
          a b c : Nat
          a1 : LT.lt 1 a
          b1 : LT.lt 1 b
          h : c.ModEq a b
          n : Nat
          ⊢ c.ModEq (HMul.hMul (HMul.hMul 2 a) (Pell.xn a1 (HAdd.hAdd n 1))) (HMul.hMul  …
        -/
        exact (h.mul_left _).mul (xy_modEq_of_modEq _ _ h (n + 1)).left,
        /-
          🎉 no goals
        -/
      (xy_modEq_of_modEq a1 b1 h n).right.add_right_cancel <| by
        /-
          a b c : Nat
          a1 : LT.lt 1 a
          b1 : LT.lt 1 b
          h : c.ModEq a b
          n : Nat
          ⊢ c.ModEq (HAdd.hAdd (Pell.yn a1 (HAdd.hAdd n 2)) (Pell.yn a1 n)) (HAdd.hAdd ( …
        -/
        rw [yn_succ_succ a1, yn_succ_succ b1]
        /-
          a b c : Nat
          a1 : LT.lt 1 a
          b1 : LT.lt 1 b
          h : c.ModEq a b
          n : Nat
          ⊢ c.ModEq (HMul.hMul (HMul.hMul 2 a) (Pell.yn a1 (HAdd.hAdd n 1))) (HMul.hMul  …
        -/
        exact (h.mul_left _).mul (xy_modEq_of_modEq _ _ h (n + 1)).right⟩
        /-
          🎉 no goals
        -/


theorem matiyasevic {a k x y} :
    (∃ a1 : 1 < a, xn a1 k = x ∧ yn a1 k = y) ↔
      1 < a ∧ k ≤ y ∧ (x = 1 ∧ y = 0 ∨
        ∃ u v s t b : ℕ,
          x * x - (a * a - 1) * y * y = 1 ∧ u * u - (a * a - 1) * v * v = 1 ∧
          s * s - (b * b - 1) * t * t = 1 ∧ 1 < b ∧ b ≡ 1 [MOD 4 * y] ∧
          b ≡ a [MOD u] ∧ 0 < v ∧ y * y ∣ v ∧ s ≡ x [MOD u] ∧ t ≡ k [MOD 4 * y]) :=
  ⟨fun ⟨a1, hx, hy⟩ => by
    /-
      a k x y : Nat
      x✝ : Exists fun a1 => And (Eq (Pell.xn a1 k) x) (Eq (Pell.yn a1 k) y)
      a1 : LT.lt 1 a
      hx : Eq (Pell.xn a1 k) x
      hy : Eq (Pell.yn a1 k) y
      ⊢ And (LT.lt 1 a) (And (LE.le k y) (Or (And (Eq x 1) (Eq y 0)) (Exists fun u = …
    -/
    rw [← hx, ← hy]
    refine ⟨a1,
        (Nat.eq_zero_or_pos k).elim (fun k0 => by rw [k0]; exact ⟨le_rfl, Or.inl ⟨rfl, rfl⟩⟩)
          fun kpos => ?_⟩
    exact
      let x := xn a1 k
      let y := yn a1 k
      let m := 2 * (k * y)
      let u := xn a1 m
      let v := yn a1 m
      have ky : k ≤ y := yn_ge_n a1 k
      have yv : y * y ∣ v := (ysq_dvd_yy a1 k).trans <| (y_dvd_iff _ _ _).2 <| dvd_mul_left _ _
      have uco : Nat.Coprime u (4 * y) :=
        have : 2 ∣ v :=
          modEq_zero_iff_dvd.1 <| (yn_modEq_two _ _).trans (dvd_mul_right _ _).modEq_zero_nat
        have : Nat.Coprime u 2 := (xy_coprime a1 m).coprime_dvd_right this
        (this.mul_right this).mul_right <|
          (xy_coprime _ _).coprime_dvd_right (dvd_of_mul_left_dvd yv)
      let ⟨b, ba, bm1⟩ := chineseRemainder uco a 1
      have m1 : 1 < m :=
        have : 0 < k * y := mul_pos kpos (strictMono_y a1 kpos)
        Nat.mul_le_mul_left 2 this
      have vp : 0 < v := strictMono_y a1 (lt_trans zero_lt_one m1)
      have b1 : 1 < b :=
        have : xn a1 1 < u := strictMono_x a1 m1
        have : a < u := by simpa using this
        lt_of_lt_of_le a1 <| by
          delta ModEq at ba; rw [Nat.mod_eq_of_lt this] at ba; rw [← ba]
          apply Nat.mod_le
      let s := xn b1 k
      let t := yn b1 k
      have sx : s ≡ x [MOD u] := (xy_modEq_of_modEq b1 a1 ba k).left
      have tk : t ≡ k [MOD 4 * y] :=
        have : 4 * y ∣ b - 1 :=
          Int.natCast_dvd_natCast.1 <| by rw [Int.ofNat_sub (le_of_lt b1)]; exact bm1.symm.dvd
        (yn_modEq_a_sub_one _ _).of_dvd this
      ⟨ky,
        Or.inr
          ⟨u, v, s, t, b, pell_eq _ _, pell_eq _ _, pell_eq _ _, b1, bm1, ba, vp, yv, sx, tk⟩⟩,
    fun ⟨a1, ky, o⟩ =>
    ⟨a1,
      match o with
      | Or.inl ⟨x1, y0⟩ => by
        /-
          a k x y : Nat
          x✝ : And (LT.lt 1 a) (And (LE.le k y) (Or (And (Eq x 1) (Eq y 0)) (Exists fun  …
          a1 : LT.lt 1 a
          ky : LE.le k y
          o : Or (And (Eq x 1) (Eq y 0)) (Exists fun u => Exists fun v => Exists fun s = …
          x1 : Eq x 1
          y0 : Eq y 0
          ⊢ And (Eq (Pell.xn a1 k) x) (Eq (Pell.yn a1 k) y)
        -/
        rw [y0] at ky; rw [Nat.eq_zero_of_le_zero ky, x1, y0]; exact ⟨rfl, rfl⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/
      | Or.inr ⟨u, v, s, t, b, xy, uv, st, b1, rem⟩ =>
        match x, y, eq_pell a1 xy, u, v, eq_pell a1 uv, s, t, eq_pell b1 st, rem, ky with
        | _, _, ⟨i, rfl, rfl⟩, _, _, ⟨n, rfl, rfl⟩, _, _, ⟨j, rfl, rfl⟩,
          ⟨(bm1 : b ≡ 1 [MOD 4 * yn a1 i]), (ba : b ≡ a [MOD xn a1 n]), (vp : 0 < yn a1 n),
            (yv : yn a1 i * yn a1 i ∣ yn a1 n), (sx : xn b1 j ≡ xn a1 i [MOD xn a1 n]),
            (tk : yn b1 j ≡ k [MOD 4 * yn a1 i])⟩,
          (ky : k ≤ yn a1 i) =>
          (Nat.eq_zero_or_pos i).elim
            (fun i0 => by
              /-
                a k x y : Nat
                a1 : LT.lt 1 a
                ky✝ : LE.le k y
                u v s t b : Nat
                b1 : LT.lt 1 b
                rem : And ((HMul.hMul 4 y).ModEq b 1) (And (u.ModEq b a) (And (LT.lt 0 v) (And …
                i n j : Nat
                bm1 : (HMul.hMul 4 (Pell.yn a1 i)).ModEq b 1
                ba : (Pell.xn a1 n).ModEq b a
                vp : LT.lt 0 (Pell.yn a1 n)
                yv : Dvd.dvd (HMul.hMul (Pell.yn a1 i) (Pell.yn a1 i)) (Pell.yn a1 n)
                sx : (Pell.xn a1 n).ModEq (Pell.xn b1 j) (Pell.xn a1 i)
                tk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq (Pell.yn b1 j) k
                ky : LE.le k (Pell.yn a1 i)
                x✝ : And (LT.lt 1 a) (And (LE.le k (Pell.yn a1 i)) (Or (And (Eq (Pell.xn a1 i) …
                o : Or (And (Eq (Pell.xn a1 i) 1) (Eq (Pell.yn a1 i) 0)) (Exists fun u => Exis …
                xy : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 i) (Pell.xn a1 i)) (HMul.hMul (HMul. …
                uv : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul. …
                st : Eq (HSub.hSub (HMul.hMul (Pell.xn b1 j) (Pell.xn b1 j)) (HMul.hMul (HMul. …
                i0 : Eq i 0
                ⊢ And (Eq (Pell.xn a1 k) (Pell.xn a1 i)) (Eq (Pell.yn a1 k) (Pell.yn a1 i))
              -/
              simp only [i0, yn_zero, nonpos_iff_eq_zero] at ky; rw [i0, ky]; exact ⟨rfl, rfl⟩)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
            fun ipos => by
            /-
              a k x y : Nat
              a1 : LT.lt 1 a
              ky✝ : LE.le k y
              u v s t b : Nat
              b1 : LT.lt 1 b
              rem : And ((HMul.hMul 4 y).ModEq b 1) (And (u.ModEq b a) (And (LT.lt 0 v) (And …
              i n j : Nat
              bm1 : (HMul.hMul 4 (Pell.yn a1 i)).ModEq b 1
              ba : (Pell.xn a1 n).ModEq b a
              vp : LT.lt 0 (Pell.yn a1 n)
              yv : Dvd.dvd (HMul.hMul (Pell.yn a1 i) (Pell.yn a1 i)) (Pell.yn a1 n)
              sx : (Pell.xn a1 n).ModEq (Pell.xn b1 j) (Pell.xn a1 i)
              tk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq (Pell.yn b1 j) k
              ky : LE.le k (Pell.yn a1 i)
              x✝ : And (LT.lt 1 a) (And (LE.le k (Pell.yn a1 i)) (Or (And (Eq (Pell.xn a1 i) …
              o : Or (And (Eq (Pell.xn a1 i) 1) (Eq (Pell.yn a1 i) 0)) (Exists fun u => Exis …
              xy : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 i) (Pell.xn a1 i)) (HMul.hMul (HMul. …
              uv : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul. …
              st : Eq (HSub.hSub (HMul.hMul (Pell.xn b1 j) (Pell.xn b1 j)) (HMul.hMul (HMul. …
              ipos : GT.gt i 0
              ⊢ And (Eq (Pell.xn a1 k) (Pell.xn a1 i)) (Eq (Pell.yn a1 k) (Pell.yn a1 i))
            -/
            suffices i = k by rw [this]; exact ⟨rfl, rfl⟩
            /-
              a k x y : Nat
              a1 : LT.lt 1 a
              ky✝ : LE.le k y
              u v s t b : Nat
              b1 : LT.lt 1 b
              rem : And ((HMul.hMul 4 y).ModEq b 1) (And (u.ModEq b a) (And (LT.lt 0 v) (And …
              i n j : Nat
              bm1 : (HMul.hMul 4 (Pell.yn a1 i)).ModEq b 1
              ba : (Pell.xn a1 n).ModEq b a
              vp : LT.lt 0 (Pell.yn a1 n)
              yv : Dvd.dvd (HMul.hMul (Pell.yn a1 i) (Pell.yn a1 i)) (Pell.yn a1 n)
              sx : (Pell.xn a1 n).ModEq (Pell.xn b1 j) (Pell.xn a1 i)
              tk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq (Pell.yn b1 j) k
              ky : LE.le k (Pell.yn a1 i)
              x✝ : And (LT.lt 1 a) (And (LE.le k (Pell.yn a1 i)) (Or (And (Eq (Pell.xn a1 i) …
              o : Or (And (Eq (Pell.xn a1 i) 1) (Eq (Pell.yn a1 i) 0)) (Exists fun u => Exis …
              xy : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 i) (Pell.xn a1 i)) (HMul.hMul (HMul. …
              uv : Eq (HSub.hSub (HMul.hMul (Pell.xn a1 n) (Pell.xn a1 n)) (HMul.hMul (HMul. …
              st : Eq (HSub.hSub (HMul.hMul (Pell.xn b1 j) (Pell.xn b1 j)) (HMul.hMul (HMul. …
              ipos : GT.gt i 0
              ⊢ Eq i k
            -/
            clear o rem xy uv st
            have iln : i ≤ n :=
              le_of_not_gt fun hin =>
                not_lt_of_ge (Nat.le_of_dvd vp (dvd_of_mul_left_dvd yv)) (strictMono_y a1 hin)
            /-
              a k x y : Nat
              a1 : LT.lt 1 a
              ky✝ : LE.le k y
              u v s t b : Nat
              b1 : LT.lt 1 b
              i n j : Nat
              bm1 : (HMul.hMul 4 (Pell.yn a1 i)).ModEq b 1
              ba : (Pell.xn a1 n).ModEq b a
              vp : LT.lt 0 (Pell.yn a1 n)
              yv : Dvd.dvd (HMul.hMul (Pell.yn a1 i) (Pell.yn a1 i)) (Pell.yn a1 n)
              sx : (Pell.xn a1 n).ModEq (Pell.xn b1 j) (Pell.xn a1 i)
              tk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq (Pell.yn b1 j) k
              ky : LE.le k (Pell.yn a1 i)
              x✝ : And (LT.lt 1 a) (And (LE.le k (Pell.yn a1 i)) (Or (And (Eq (Pell.xn a1 i) …
              ipos : GT.gt i 0
              iln : LE.le i n
              ⊢ Eq i k
            -/
            have yd : 4 * yn a1 i ∣ 4 * n := mul_dvd_mul_left _ <| dvd_of_ysq_dvd a1 yv
            have jk : j ≡ k [MOD 4 * yn a1 i] :=
              have : 4 * yn a1 i ∣ b - 1 :=
                Int.natCast_dvd_natCast.1 <| by rw [Int.ofNat_sub (le_of_lt b1)]; exact bm1.symm.dvd
              ((yn_modEq_a_sub_one b1 _).of_dvd this).symm.trans tk
            have ki : k + i < 4 * yn a1 i :=
              lt_of_le_of_lt (_root_.add_le_add ky (yn_ge_n a1 i)) <| by
                rw [← two_mul]
                exact Nat.mul_lt_mul_of_pos_right (by decide) (strictMono_y a1 ipos)
            have ji : j ≡ i [MOD 4 * n] :=
              have : xn a1 j ≡ xn a1 i [MOD xn a1 n] :=
                (xy_modEq_of_modEq b1 a1 ba j).left.symm.trans sx
              (modEq_of_xn_modEq a1 ipos iln this).resolve_right
                fun ji : j + i ≡ 0 [MOD 4 * n] =>
                not_le_of_gt ki <|
                  Nat.le_of_dvd (lt_of_lt_of_le ipos <| Nat.le_add_left _ _) <|
                    modEq_zero_iff_dvd.1 <| (jk.symm.add_right i).trans <| ji.of_dvd yd
            /-
              a k x y : Nat
              a1 : LT.lt 1 a
              ky✝ : LE.le k y
              u v s t b : Nat
              b1 : LT.lt 1 b
              i n j : Nat
              bm1 : (HMul.hMul 4 (Pell.yn a1 i)).ModEq b 1
              ba : (Pell.xn a1 n).ModEq b a
              vp : LT.lt 0 (Pell.yn a1 n)
              yv : Dvd.dvd (HMul.hMul (Pell.yn a1 i) (Pell.yn a1 i)) (Pell.yn a1 n)
              sx : (Pell.xn a1 n).ModEq (Pell.xn b1 j) (Pell.xn a1 i)
              tk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq (Pell.yn b1 j) k
              ky : LE.le k (Pell.yn a1 i)
              x✝ : And (LT.lt 1 a) (And (LE.le k (Pell.yn a1 i)) (Or (And (Eq (Pell.xn a1 i) …
              ipos : GT.gt i 0
              iln : LE.le i n
              yd : Dvd.dvd (HMul.hMul 4 (Pell.yn a1 i)) (HMul.hMul 4 n)
              jk : (HMul.hMul 4 (Pell.yn a1 i)).ModEq j k
              ki : LT.lt (HAdd.hAdd k i) (HMul.hMul 4 (Pell.yn a1 i))
              ji : (HMul.hMul 4 n).ModEq j i
              ⊢ Eq i k
            -/
            have : i % (4 * yn a1 i) = k % (4 * yn a1 i) := (ji.of_dvd yd).symm.trans jk
            rwa [Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.le_add_left _ _) ki),
              Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.le_add_right _ _) ki)] at this⟩⟩


theorem eq_pow_of_pell_lem {a y k : ℕ} (hy0 : y ≠ 0) (hk0 : k ≠ 0) (hyk : y ^ k < a) :
    (↑(y ^ k) : ℤ) < 2 * a * y - y * y - 1 :=
  have hya : y < a := (Nat.le_self_pow hk0 _).trans_lt hyk
  calc
    (↑(y ^ k) : ℤ) < a := Nat.cast_lt.2 hyk
    _ ≤ (a : ℤ) ^ 2 - (a - 1 : ℤ) ^ 2 - 1 := by
      rw [sub_sq, mul_one, one_pow, sub_add, sub_sub_cancel, two_mul, sub_sub, ← add_sub,
        le_add_iff_nonneg_right, sub_nonneg, Int.add_one_le_iff]
      /-
        a y k : Nat
        hy0 : Ne y 0
        hk0 : Ne k 0
        hyk : LT.lt (HPow.hPow y k) a
        hya : LT.lt y a
        ⊢ LT.lt 1 ↑a
      -/
      norm_cast
      /-
        a y k : Nat
        hy0 : Ne y 0
        hk0 : Ne k 0
        hyk : LT.lt (HPow.hPow y k) a
        hya : LT.lt y a
        ⊢ LT.lt 1 a
      -/
      exact lt_of_le_of_lt (Nat.succ_le_of_lt (Nat.pos_of_ne_zero hy0)) hya
      /-
        🎉 no goals
      -/
    _ ≤ (a : ℤ) ^ 2 - (a - y : ℤ) ^ 2 - 1 := by
      /-
        a y k : Nat
        hy0 : Ne y 0
        hk0 : Ne k 0
        hyk : LT.lt (HPow.hPow y k) a
        hya : LT.lt y a
        ⊢ LE.le (HSub.hSub (HSub.hSub (HPow.hPow (↑a) 2) (HPow.hPow (HSub.hSub (↑a) 1) …
      -/
      have := hya.le
      /-
        a y k : Nat
        hy0 : Ne y 0
        hk0 : Ne k 0
        hyk : LT.lt (HPow.hPow y k) a
        hya : LT.lt y a
        this : LE.le y a
        ⊢ LE.le (HSub.hSub (HSub.hSub (HPow.hPow (↑a) 2) (HPow.hPow (HSub.hSub (↑a) 1) …
      -/
                               /-
                                 🎉 no goals
                               -/
      gcongr <;> norm_cast <;> omega
                               /-
                                 🎉 no goals
                               -/
                                    /-
                                      a y k : Nat
                                      hy0 : Ne y 0
                                      hk0 : Ne k 0
                                      hyk : LT.lt (HPow.hPow y k) a
                                      hya : LT.lt y a
                                      ⊢ Eq (HSub.hSub (HSub.hSub (HPow.hPow (↑a) 2) (HPow.hPow (HSub.hSub ↑a ↑y) 2)) …
                                    -/
    _ = 2 * a * y - y * y - 1 := by ring
                                    /-
                                      🎉 no goals
                                    -/


theorem eq_pow_of_pell {m n k} :
    n ^ k = m ↔ k = 0 ∧ m = 1 ∨0 < k ∧ (n = 0 ∧ m = 0 ∨
      0 < n ∧ ∃ (w a t z : ℕ) (a1 : 1 < a), xn a1 k ≡ yn a1 k * (a - n) + m [MOD t] ∧
      2 * a * n = t + (n * n + 1) ∧ m < t ∧
      n ≤ w ∧ k ≤ w ∧ a * a - ((w + 1) * (w + 1) - 1) * (w * z) * (w * z) = 1) := by
  /-
    m n k : Nat
    ⊢ Iff (Eq (HPow.hPow n k) m) (Or (And (Eq k 0) (Eq m 1)) (And (LT.lt 0 k) (Or  …
  -/
  constructor
    /-
      case mp
      m n k : Nat
      ⊢ Eq (HPow.hPow n k) m → Or (And (Eq k 0) (Eq m 1)) (And (LT.lt 0 k) (Or (And  …
    -/
  · rintro rfl
    /-
      case mp
      n k : Nat
      ⊢ Or (And (Eq k 0) (Eq (HPow.hPow n k) 1)) (And (LT.lt 0 k) (Or (And (Eq n 0)  …
    -/
    refine k.eq_zero_or_pos.imp (fun k0 : k = 0 => k0.symm ▸ ⟨rfl, rfl⟩) fun hk => ⟨hk, ?_⟩
    refine n.eq_zero_or_pos.imp (fun n0 : n = 0 ↦ n0.symm ▸ ⟨rfl, zero_pow hk.ne'⟩)
      fun hn ↦ ⟨hn, ?_⟩
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    set w := max n k
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have nw : n ≤ w := le_max_left _ _
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have kw : k ≤ w := le_max_right _ _
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have wpos : 0 < w := hn.trans_le nw
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have w1 : 1 < w + 1 := Nat.succ_lt_succ wpos
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    set a := xn w1 w
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have a1 : 1 < a := strictMono_x w1 wpos
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have na : n ≤ a := nw.trans (n_lt_xn w1 w).le
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      na : LE.le n a
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    set x := xn a1 k
    /-
      case mp
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      na : LE.le n a
      x : Nat := Pell.xn a1 k
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    set y := yn a1 k
    obtain ⟨z, ze⟩ : w ∣ yn w1 w :=
      modEq_zero_iff_dvd.1 ((yn_modEq_a_sub_one w1 w).trans dvd_rfl.modEq_zero_nat)
    have nt : (↑(n ^ k) : ℤ) < 2 * a * n - n * n - 1 := by
      refine eq_pow_of_pell_lem hn.ne' hk.ne' ?_
      calc
        n ^ k ≤ n ^ w := Nat.pow_le_pow_of_le_right hn kw
        _ < (w + 1) ^ w := Nat.pow_lt_pow_left (Nat.lt_succ_of_le nw) wpos.ne'
        _ ≤ a := xn_ge_a_pow w1 w
    /-
      case mp.intro
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      na : LE.le n a
      x : Nat := Pell.xn a1 k
      y : Nat := Pell.yn a1 k
      z : Nat
      ze : Eq (Pell.yn w1 w) (HMul.hMul w z)
      nt : LT.lt (↑(HPow.hPow n k)) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a …
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    lift (2 * a * n - n * n - 1 : ℤ) to ℕ using (Nat.cast_nonneg _).trans nt.le with t te
    have tm : x ≡ y * (a - n) + n ^ k [MOD t] := by
      apply modEq_of_dvd
      rw [Int.ofNat_add, Int.ofNat_mul, Int.ofNat_sub na, te]
      exact x_sub_y_dvd_pow a1 n k
    have ta : 2 * a * n = t + (n * n + 1) := by
      zify
      omega
    /-
      case mp.intro.intro
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      na : LE.le n a
      x : Nat := Pell.xn a1 k
      y : Nat := Pell.yn a1 k
      z : Nat
      ze : Eq (Pell.yn w1 w) (HMul.hMul w z)
      t : Nat
      te : Eq (↑t) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑n) (HMul.hMul  …
      nt✝ nt : LT.lt ↑(HPow.hPow n k) ↑t
      tm : t.ModEq x (HAdd.hAdd (HMul.hMul y (HSub.hSub a n)) (HPow.hPow n k))
      ta : Eq (HMul.hMul (HMul.hMul 2 a) n) (HAdd.hAdd t (HAdd.hAdd (HMul.hMul n n)  …
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    have zp : a * a - ((w + 1) * (w + 1) - 1) * (w * z) * (w * z) = 1 := ze ▸ pell_eq w1 w
    /-
      case mp.intro.intro
      n k : Nat
      hk : LT.lt 0 k
      hn : LT.lt 0 n
      w : Nat := Max.max n k
      nw : LE.le n w
      kw : LE.le k w
      wpos : LT.lt 0 w
      w1 : LT.lt 1 (HAdd.hAdd w 1)
      a : Nat := Pell.xn w1 w
      a1 : LT.lt 1 a
      na : LE.le n a
      x : Nat := Pell.xn a1 k
      y : Nat := Pell.yn a1 k
      z : Nat
      ze : Eq (Pell.yn w1 w) (HMul.hMul w z)
      t : Nat
      te : Eq (↑t) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑a) ↑n) (HMul.hMul  …
      nt✝ nt : LT.lt ↑(HPow.hPow n k) ↑t
      tm : t.ModEq x (HAdd.hAdd (HMul.hMul y (HSub.hSub a n)) (HPow.hPow n k))
      ta : Eq (HMul.hMul (HMul.hMul 2 a) n) (HAdd.hAdd t (HAdd.hAdd (HMul.hMul n n)  …
      zp : Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul …
      ⊢ Exists fun w => Exists fun a => Exists fun t => Exists fun z => Exists fun a …
    -/
    exact ⟨w, a, t, z, a1, tm, ta, Nat.cast_lt.1 nt, nw, kw, zp⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m n k : Nat
      ⊢ Or (And (Eq k 0) (Eq m 1)) (And (LT.lt 0 k) (Or (And (Eq n 0) (Eq m 0)) (And …
    -/
  · rintro (⟨rfl, rfl⟩ | ⟨hk0, ⟨rfl, rfl⟩ | ⟨hn0, w, a, t, z, a1, tm, ta, mt, nw, kw, zp⟩⟩)
      /-
        case mpr.inl.intro
        n : Nat
        ⊢ Eq (HPow.hPow n 0) 1
      -/
    · exact _root_.pow_zero n
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.inl.intro
        k : Nat
        hk0 : LT.lt 0 k
        ⊢ Eq (HPow.hPow 0 k) 0
      -/
    · exact zero_pow hk0.ne'
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w a t z : Nat
      a1 : LT.lt 1 a
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub a  …
      ta : Eq (HMul.hMul (HMul.hMul 2 a) n) (HAdd.hAdd t (HAdd.hAdd (HMul.hMul n n)  …
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      zp : Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul …
      ⊢ Eq (HPow.hPow n k) m
    -/
    have hw0 : 0 < w := hn0.trans_le nw
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w a t z : Nat
      a1 : LT.lt 1 a
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub a  …
      ta : Eq (HMul.hMul (HMul.hMul 2 a) n) (HAdd.hAdd t (HAdd.hAdd (HMul.hMul n n)  …
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      zp : Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul …
      hw0 : LT.lt 0 w
      ⊢ Eq (HPow.hPow n k) m
    -/
    have hw1 : 1 < w + 1 := Nat.succ_lt_succ hw0
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w a t z : Nat
      a1 : LT.lt 1 a
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub a  …
      ta : Eq (HMul.hMul (HMul.hMul 2 a) n) (HAdd.hAdd t (HAdd.hAdd (HMul.hMul n n)  …
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      zp : Eq (HSub.hSub (HMul.hMul a a) (HMul.hMul (HMul.hMul (HSub.hSub (HMul.hMul …
      hw0 : LT.lt 0 w
      hw1 : LT.lt 1 (HAdd.hAdd w 1)
      ⊢ Eq (HPow.hPow n k) m
    -/
    rcases eq_pell hw1 zp with ⟨j, rfl, yj⟩
    have hj0 : 0 < j := by
      apply Nat.pos_of_ne_zero
      rintro rfl
      exact lt_irrefl 1 a1
    have wj : w ≤ j :=
      Nat.le_of_dvd hj0
        (modEq_zero_iff_dvd.1 <|
          (yn_modEq_a_sub_one hw1 j).symm.trans <| modEq_zero_iff_dvd.2 ⟨z, yj.symm⟩)
    have hnka : n ^ k < xn hw1 j := calc
      n ^ k ≤ n ^ j := Nat.pow_le_pow_of_le_right hn0 (le_trans kw wj)
      _ < (w + 1) ^ j := Nat.pow_lt_pow_left (Nat.lt_succ_of_le nw) hj0.ne'
      _ ≤ xn hw1 j := xn_ge_a_pow hw1 j
    have nt : (↑(n ^ k) : ℤ) < 2 * xn hw1 j * n - n * n - 1 :=
      eq_pow_of_pell_lem hn0.ne' hk0.ne' hnka
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w t z : Nat
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      hw0 : LT.lt 0 w
      hw1 : LT.lt 1 (HAdd.hAdd w 1)
      j : Nat
      yj : Eq (HMul.hMul w z) (Pell.yn hw1 j)
      a1 : LT.lt 1 (Pell.xn hw1 j)
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub (P …
      ta : Eq (HMul.hMul (HMul.hMul 2 (Pell.xn hw1 j)) n) (HAdd.hAdd t (HAdd.hAdd (H …
      zp : Eq (HSub.hSub (HMul.hMul (Pell.xn hw1 j) (Pell.xn hw1 j)) (HMul.hMul (HMu …
      hj0 : LT.lt 0 j
      wj : LE.le w j
      hnka : LT.lt (HPow.hPow n k) (Pell.xn hw1 j)
      nt : LT.lt (↑(HPow.hPow n k)) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑( …
      ⊢ Eq (HPow.hPow n k) m
    -/
    have na : n ≤ xn hw1 j := (Nat.le_self_pow hk0.ne' _).trans hnka.le
    have te : (t : ℤ) = 2 * xn hw1 j * n - n * n - 1 := by
      rw [sub_sub, eq_sub_iff_add_eq]
      exact mod_cast ta.symm
    have : xn a1 k ≡ yn a1 k * (xn hw1 j - n) + n ^ k [MOD t] := by
      apply modEq_of_dvd
      rw [te, Nat.cast_add, Nat.cast_mul, Int.ofNat_sub na]
      exact x_sub_y_dvd_pow a1 n k
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w t z : Nat
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      hw0 : LT.lt 0 w
      hw1 : LT.lt 1 (HAdd.hAdd w 1)
      j : Nat
      yj : Eq (HMul.hMul w z) (Pell.yn hw1 j)
      a1 : LT.lt 1 (Pell.xn hw1 j)
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub (P …
      ta : Eq (HMul.hMul (HMul.hMul 2 (Pell.xn hw1 j)) n) (HAdd.hAdd t (HAdd.hAdd (H …
      zp : Eq (HSub.hSub (HMul.hMul (Pell.xn hw1 j) (Pell.xn hw1 j)) (HMul.hMul (HMu …
      hj0 : LT.lt 0 j
      wj : LE.le w j
      hnka : LT.lt (HPow.hPow n k) (Pell.xn hw1 j)
      nt : LT.lt (↑(HPow.hPow n k)) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑( …
      na : LE.le n (Pell.xn hw1 j)
      te : Eq (↑t) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑(Pell.xn hw1 j)) ↑ …
      this : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub  …
      ⊢ Eq (HPow.hPow n k) m
    -/
    have : n ^ k % t = m % t := (this.symm.trans tm).add_left_cancel' _
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w t z : Nat
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      hw0 : LT.lt 0 w
      hw1 : LT.lt 1 (HAdd.hAdd w 1)
      j : Nat
      yj : Eq (HMul.hMul w z) (Pell.yn hw1 j)
      a1 : LT.lt 1 (Pell.xn hw1 j)
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub (P …
      ta : Eq (HMul.hMul (HMul.hMul 2 (Pell.xn hw1 j)) n) (HAdd.hAdd t (HAdd.hAdd (H …
      zp : Eq (HSub.hSub (HMul.hMul (Pell.xn hw1 j) (Pell.xn hw1 j)) (HMul.hMul (HMu …
      hj0 : LT.lt 0 j
      wj : LE.le w j
      hnka : LT.lt (HPow.hPow n k) (Pell.xn hw1 j)
      nt : LT.lt (↑(HPow.hPow n k)) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑( …
      na : LE.le n (Pell.xn hw1 j)
      te : Eq (↑t) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑(Pell.xn hw1 j)) ↑ …
      this✝ : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub …
      this : Eq (HMod.hMod (HPow.hPow n k) t) (HMod.hMod m t)
      ⊢ Eq (HPow.hPow n k) m
    -/
    rw [← te] at nt
    /-
      case mpr.inr.intro.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      m n k : Nat
      hk0 : LT.lt 0 k
      hn0 : LT.lt 0 n
      w t z : Nat
      mt : LT.lt m t
      nw : LE.le n w
      kw : LE.le k w
      hw0 : LT.lt 0 w
      hw1 : LT.lt 1 (HAdd.hAdd w 1)
      j : Nat
      yj : Eq (HMul.hMul w z) (Pell.yn hw1 j)
      a1 : LT.lt 1 (Pell.xn hw1 j)
      tm : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub (P …
      ta : Eq (HMul.hMul (HMul.hMul 2 (Pell.xn hw1 j)) n) (HAdd.hAdd t (HAdd.hAdd (H …
      zp : Eq (HSub.hSub (HMul.hMul (Pell.xn hw1 j) (Pell.xn hw1 j)) (HMul.hMul (HMu …
      hj0 : LT.lt 0 j
      wj : LE.le w j
      hnka : LT.lt (HPow.hPow n k) (Pell.xn hw1 j)
      nt : LT.lt ↑(HPow.hPow n k) ↑t
      na : LE.le n (Pell.xn hw1 j)
      te : Eq (↑t) (HSub.hSub (HSub.hSub (HMul.hMul (HMul.hMul 2 ↑(Pell.xn hw1 j)) ↑ …
      this✝ : t.ModEq (Pell.xn a1 k) (HAdd.hAdd (HMul.hMul (Pell.yn a1 k) (HSub.hSub …
      this : Eq (HMod.hMod (HPow.hPow n k) t) (HMod.hMod m t)
      ⊢ Eq (HPow.hPow n k) m
    -/
    rwa [Nat.mod_eq_of_lt (Nat.cast_lt.1 nt), Nat.mod_eq_of_lt mt] at this
    /-
      🎉 no goals
    -/


