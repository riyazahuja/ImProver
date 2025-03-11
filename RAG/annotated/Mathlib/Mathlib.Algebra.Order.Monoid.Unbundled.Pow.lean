@[to_additive Left.nsmul_nonneg]
theorem one_le_pow_of_le (ha : 1 ≤ a) : ∀ n : ℕ, 1 ≤ a ^ n
            /-
              M : Type u_3
              inst✝² : Monoid M
              inst✝¹ : Preorder M
              inst✝ : MulLeftMono M
              a : M
              ha : LE.le 1 a
              ⊢ LE.le 1 (HPow.hPow a 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      M : Type u_3
      inst✝² : Monoid M
      inst✝¹ : Preorder M
      inst✝ : MulLeftMono M
      a : M
      ha : LE.le 1 a
      k : Nat
      ⊢ LE.le 1 (HPow.hPow a (HAdd.hAdd k 1))
    -/
    rw [pow_succ]
    /-
      M : Type u_3
      inst✝² : Monoid M
      inst✝¹ : Preorder M
      inst✝ : MulLeftMono M
      a : M
      ha : LE.le 1 a
      k : Nat
      ⊢ LE.le 1 (HMul.hMul (HPow.hPow a k) a)
    -/
    exact one_le_mul (one_le_pow_of_le ha k) ha
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-21")] alias pow_nonneg := nsmul_nonneg


@[to_additive nsmul_nonpos]
theorem pow_le_one_of_le (ha : a ≤ 1) (n : ℕ) : a ^ n ≤ 1 := one_le_pow_of_le (M := Mᵒᵈ) ha n


@[deprecated (since := "2024-09-21")] alias pow_nonpos := nsmul_nonpos


@[to_additive nsmul_neg]
theorem pow_lt_one_of_lt {a : M} {n : ℕ} (h : a < 1) (hn : n ≠ 0) : a ^ n < 1 := by
  /-
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    a : M
    n : Nat
    h : LT.lt a 1
    hn : Ne n 0
    ⊢ LT.lt (HPow.hPow a n) 1
  -/
  rcases Nat.exists_eq_succ_of_ne_zero hn with ⟨k, rfl⟩
  /-
    case intro
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    a : M
    h : LT.lt a 1
    k : Nat
    hn : Ne k.succ 0
    ⊢ LT.lt (HPow.hPow a k.succ) 1
  -/
  rw [pow_succ']
  /-
    case intro
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulLeftMono M
    a : M
    h : LT.lt a 1
    k : Nat
    hn : Ne k.succ 0
    ⊢ LT.lt (HMul.hMul a (HPow.hPow a k)) 1
  -/
  exact mul_lt_one_of_lt_of_le h (pow_le_one_of_le h.le _)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-21")] alias pow_neg := nsmul_neg


@[to_additive nsmul_nonneg] alias one_le_pow_of_one_le' := Left.one_le_pow_of_le

@[to_additive nsmul_nonpos] alias pow_le_one' := Left.pow_le_one_of_le

@[to_additive nsmul_neg] alias pow_lt_one' := Left.pow_lt_one_of_lt


@[to_additive nsmul_left_monotone]
theorem pow_right_monotone {a : M} (ha : 1 ≤ a) : Monotone fun n : ℕ ↦ a ^ n :=
                                     /-
                                       M : Type u_3
                                       inst✝² : Monoid M
                                       inst✝¹ : Preorder M
                                       inst✝ : MulLeftMono M
                                       a : M
                                       ha : LE.le 1 a
                                       n : Nat
                                       ⊢ LE.le (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
                                     -/
  monotone_nat_of_le_succ fun n ↦ by rw [pow_succ]; exact le_mul_of_one_le_right' ha
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive (attr := gcongr) nsmul_le_nsmul_left]
theorem pow_le_pow_right' {a : M} {n m : ℕ} (ha : 1 ≤ a) (h : n ≤ m) : a ^ n ≤ a ^ m :=
  pow_right_monotone ha h


@[to_additive nsmul_le_nsmul_left_of_nonpos]
theorem pow_le_pow_right_of_le_one' {a : M} {n m : ℕ} (ha : a ≤ 1) (h : n ≤ m) : a ^ m ≤ a ^ n :=
  pow_le_pow_right' (M := Mᵒᵈ) ha h


@[to_additive nsmul_pos]
theorem one_lt_pow' {a : M} (ha : 1 < a) {k : ℕ} (hk : k ≠ 0) : 1 < a ^ k :=
  pow_lt_one' (M := Mᵒᵈ) ha hk


@[to_additive nsmul_left_strictMono]
theorem pow_right_strictMono' (ha : 1 < a) : StrictMono ((a ^ ·) : ℕ → M) :=
                                       /-
                                         M : Type u_3
                                         inst✝² : Monoid M
                                         inst✝¹ : Preorder M
                                         inst✝ : MulLeftStrictMono M
                                         a : M
                                         ha : LT.lt 1 a
                                         n : Nat
                                         ⊢ LT.lt (HPow.hPow a n) (HPow.hPow a (HAdd.hAdd n 1))
                                       -/
  strictMono_nat_of_lt_succ fun n ↦ by rw [pow_succ]; exact lt_mul_of_one_lt_right' (a ^ n) ha
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive (attr := gcongr) nsmul_lt_nsmul_left]
theorem pow_lt_pow_right' (ha : 1 < a) (h : n < m) : a ^ n < a ^ m :=
  pow_right_strictMono' ha h


@[to_additive Right.nsmul_nonneg]
theorem Right.one_le_pow_of_le (hx : 1 ≤ x) : ∀ {n : ℕ}, 1 ≤ x ^ n
  | 0 => (pow_zero _).ge
  | n + 1 => by
    /-
      M : Type u_3
      inst✝² : Monoid M
      inst✝¹ : Preorder M
      inst✝ : MulRightMono M
      x : M
      hx : LE.le 1 x
      n : Nat
      ⊢ LE.le 1 (HPow.hPow x (HAdd.hAdd n 1))
    -/
    rw [pow_succ]
    /-
      M : Type u_3
      inst✝² : Monoid M
      inst✝¹ : Preorder M
      inst✝ : MulRightMono M
      x : M
      hx : LE.le 1 x
      n : Nat
      ⊢ LE.le 1 (HMul.hMul (HPow.hPow x n) x)
    -/
    exact Right.one_le_mul (Right.one_le_pow_of_le hx) hx
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-21")] alias Right.pow_nonneg := Right.nsmul_nonneg


@[to_additive Right.nsmul_nonpos]
theorem Right.pow_le_one_of_le (hx : x ≤ 1) {n : ℕ} : x ^ n ≤ 1 :=
  Right.one_le_pow_of_le (M := Mᵒᵈ) hx


@[deprecated (since := "2024-09-21")] alias Right.pow_nonpos := Right.nsmul_nonpos


@[to_additive Right.nsmul_neg]
theorem Right.pow_lt_one_of_lt {n : ℕ} {x : M} (hn : 0 < n) (h : x < 1) : x ^ n < 1 := by
  /-
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulRightMono M
    n : Nat
    x : M
    hn : LT.lt 0 n
    h : LT.lt x 1
    ⊢ LT.lt (HPow.hPow x n) 1
  -/
  rcases Nat.exists_eq_succ_of_ne_zero hn.ne' with ⟨k, rfl⟩
  /-
    case intro
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulRightMono M
    x : M
    h : LT.lt x 1
    k : Nat
    hn : LT.lt 0 k.succ
    ⊢ LT.lt (HPow.hPow x k.succ) 1
  -/
  rw [pow_succ]
  /-
    case intro
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Preorder M
    inst✝ : MulRightMono M
    x : M
    h : LT.lt x 1
    k : Nat
    hn : LT.lt 0 k.succ
    ⊢ LT.lt (HMul.hMul (HPow.hPow x k) x) 1
  -/
  exact mul_lt_one_of_le_of_lt (pow_le_one_of_le h.le) h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-21")] alias Right.pow_neg := Right.nsmul_neg


/-- This lemma is useful in non-cancellative monoids, like sets under pointwise operations. -/
@[to_additive
"This lemma is useful in non-cancellative monoids, like sets under pointwise operations."]
lemma pow_le_pow_mul_of_sq_le_mul [MulLeftMono M] {a b : M} (hab : a ^ 2 ≤ b * a) :
    ∀ {n}, n ≠ 0 → a ^ n ≤ b ^ (n - 1) * a
               /-
                 M : Type u_3
                 inst✝³ : Monoid M
                 inst✝² : Preorder M
                 inst✝¹ : MulRightMono M
                 inst✝ : MulLeftMono M
                 a b : M
                 hab : LE.le (HPow.hPow a 2) (HMul.hMul b a)
                 x✝ : Ne 1 0
                 ⊢ LE.le (HPow.hPow a 1) (HMul.hMul (HPow.hPow b (HSub.hSub 1 1)) a)
               -/
  | 1, _ => by simp
               /-
                 🎉 no goals
               -/
  | n + 2, _ => by
    calc
      a ^ (n + 2) = a ^ (n + 1) * a := by rw [pow_succ]
      _ ≤ b ^ n * a * a := mul_le_mul_right' (pow_le_pow_mul_of_sq_le_mul hab (by omega)) _
      _ = b ^ n * a ^ 2 := by rw [mul_assoc, sq]
      _ ≤ b ^ n * (b * a) := mul_le_mul_left' hab _
      _ = b ^ (n + 1) * a := by rw [← mul_assoc, ← pow_succ]


@[to_additive StrictMono.const_nsmul]
theorem StrictMono.pow_const (hf : StrictMono f) : ∀ {n : ℕ}, n ≠ 0 → StrictMono (f · ^ n)
  | 0, hn => (hn rfl).elim
               /-
                 β : Type u_1
                 M : Type u_3
                 inst✝⁴ : Monoid M
                 inst✝³ : Preorder M
                 inst✝² : Preorder β
                 inst✝¹ : MulLeftStrictMono M
                 inst✝ : MulRightStrictMono M
                 f : β → M
                 hf : StrictMono f
                 x✝ : Ne 1 0
                 ⊢ StrictMono fun x => HPow.hPow (f x) 1
               -/
  | 1, _ => by simpa
               /-
                 🎉 no goals
               -/
  | Nat.succ <| Nat.succ n, _ => by
    /-
      β : Type u_1
      M : Type u_3
      inst✝⁴ : Monoid M
      inst✝³ : Preorder M
      inst✝² : Preorder β
      inst✝¹ : MulLeftStrictMono M
      inst✝ : MulRightStrictMono M
      f : β → M
      hf : StrictMono f
      n : Nat
      x✝ : Ne n.succ.succ 0
      ⊢ StrictMono fun x => HPow.hPow (f x) n.succ.succ
    -/
    simpa only [pow_succ] using (hf.pow_const n.succ_ne_zero).mul' hf
    /-
      🎉 no goals
    -/


/-- See also `pow_left_strictMonoOn₀`. -/
@[to_additive nsmul_right_strictMono]  -- Porting note: nolint to_additive_doc
theorem pow_left_strictMono (hn : n ≠ 0) : StrictMono (· ^ n : M → M) := strictMono_id.pow_const hn


@[to_additive (attr := mono, gcongr) nsmul_lt_nsmul_right]
lemma pow_lt_pow_left' (hn : n ≠ 0) {a b : M} (hab : a < b) : a ^ n < b ^ n :=
  pow_left_strictMono hn hab


@[to_additive (attr := mono, gcongr) nsmul_le_nsmul_right]
theorem pow_le_pow_left' {a b : M} (hab : a ≤ b) : ∀ i : ℕ, a ^ i ≤ b ^ i
            /-
              M : Type u_3
              inst✝³ : Monoid M
              inst✝² : Preorder M
              inst✝¹ : MulLeftMono M
              inst✝ : MulRightMono M
              a b : M
              hab : LE.le a b
              ⊢ LE.le (HPow.hPow a 0) (HPow.hPow b 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : Preorder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      a b : M
      hab : LE.le a b
      k : Nat
      ⊢ LE.le (HPow.hPow a (HAdd.hAdd k 1)) (HPow.hPow b (HAdd.hAdd k 1))
    -/
    rw [pow_succ, pow_succ]
    /-
      M : Type u_3
      inst✝³ : Monoid M
      inst✝² : Preorder M
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      a b : M
      hab : LE.le a b
      k : Nat
      ⊢ LE.le (HMul.hMul (HPow.hPow a k) a) (HMul.hMul (HPow.hPow b k) b)
    -/
    exact mul_le_mul' (pow_le_pow_left' hab k) hab
    /-
      🎉 no goals
    -/


@[to_additive Monotone.const_nsmul]
theorem Monotone.pow_const {f : β → M} (hf : Monotone f) : ∀ n : ℕ, Monotone fun a => f a ^ n
            /-
              β : Type u_1
              M : Type u_3
              inst✝⁴ : Monoid M
              inst✝³ : Preorder M
              inst✝² : Preorder β
              inst✝¹ : MulLeftMono M
              inst✝ : MulRightMono M
              f : β → M
              hf : Monotone f
              ⊢ Monotone fun a => HPow.hPow (f a) 0
            -/
  | 0 => by simpa using monotone_const
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      β : Type u_1
      M : Type u_3
      inst✝⁴ : Monoid M
      inst✝³ : Preorder M
      inst✝² : Preorder β
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      f : β → M
      hf : Monotone f
      n : Nat
      ⊢ Monotone fun a => HPow.hPow (f a) (HAdd.hAdd n 1)
    -/
    simp_rw [pow_succ]
    /-
      β : Type u_1
      M : Type u_3
      inst✝⁴ : Monoid M
      inst✝³ : Preorder M
      inst✝² : Preorder β
      inst✝¹ : MulLeftMono M
      inst✝ : MulRightMono M
      f : β → M
      hf : Monotone f
      n : Nat
      ⊢ Monotone fun a => HMul.hMul (HPow.hPow (f a) n) (f a)
    -/
    exact (Monotone.pow_const hf _).mul' hf
    /-
      🎉 no goals
    -/


@[to_additive nsmul_right_mono]
theorem pow_left_mono (n : ℕ) : Monotone fun a : M => a ^ n := monotone_id.pow_const _


@[to_additive (attr := gcongr)]
lemma pow_le_pow {a b : M} (hab : a ≤ b) (ht : 1 ≤ b) {m n : ℕ} (hmn : m ≤ n) : a ^ m ≤ b ^ n :=
  (pow_le_pow_left' hab _).trans (pow_le_pow_right' ht hmn)


@[to_additive nsmul_nonneg_iff]
theorem one_le_pow_iff {x : M} {n : ℕ} (hn : n ≠ 0) : 1 ≤ x ^ n ↔ 1 ≤ x :=
  ⟨le_imp_le_of_lt_imp_lt fun h => pow_lt_one' h hn, fun h => one_le_pow_of_one_le' h n⟩


@[to_additive]
theorem pow_le_one_iff {x : M} {n : ℕ} (hn : n ≠ 0) : x ^ n ≤ 1 ↔ x ≤ 1 :=
  one_le_pow_iff (M := Mᵒᵈ) hn


@[to_additive nsmul_pos_iff]
theorem one_lt_pow_iff {x : M} {n : ℕ} (hn : n ≠ 0) : 1 < x ^ n ↔ 1 < x :=
  lt_iff_lt_of_le_iff_le (pow_le_one_iff hn)


@[to_additive]
theorem pow_lt_one_iff {x : M} {n : ℕ} (hn : n ≠ 0) : x ^ n < 1 ↔ x < 1 :=
  lt_iff_lt_of_le_iff_le (one_le_pow_iff hn)


@[to_additive]
theorem pow_eq_one_iff {x : M} {n : ℕ} (hn : n ≠ 0) : x ^ n = 1 ↔ x = 1 := by
  /-
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : LinearOrder M
    inst✝ : MulLeftMono M
    x : M
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq (HPow.hPow x n) 1) (Eq x 1)
  -/
  simp only [le_antisymm_iff]
  /-
    M : Type u_3
    inst✝² : Monoid M
    inst✝¹ : LinearOrder M
    inst✝ : MulLeftMono M
    x : M
    n : Nat
    hn : Ne n 0
    ⊢ Iff (And (LE.le (HPow.hPow x n) 1) (LE.le 1 (HPow.hPow x n))) (And (LE.le x  …
  -/
  rw [pow_le_one_iff hn, one_le_pow_iff hn]
  /-
    🎉 no goals
  -/


@[to_additive nsmul_le_nsmul_iff_left]
theorem pow_le_pow_iff_right' (ha : 1 < a) : a ^ m ≤ a ^ n ↔ m ≤ n :=
  (pow_right_strictMono' ha).le_iff_le


@[to_additive nsmul_lt_nsmul_iff_left]
theorem pow_lt_pow_iff_right' (ha : 1 < a) : a ^ m < a ^ n ↔ m < n :=
  (pow_right_strictMono' ha).lt_iff_lt


@[to_additive lt_of_nsmul_lt_nsmul_right]
theorem lt_of_pow_lt_pow_left' {a b : M} (n : ℕ) : a ^ n < b ^ n → a < b :=
  (pow_left_mono _).reflect_lt


@[to_additive min_lt_of_add_lt_two_nsmul]
theorem min_lt_of_mul_lt_sq {a b c : M} (h : a * b < c ^ 2) : min a b < c := by
  /-
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    a b c : M
    h : LT.lt (HMul.hMul a b) (HPow.hPow c 2)
    ⊢ LT.lt (Min.min a b) c
  -/
  simpa using min_lt_max_of_mul_lt_mul (h.trans_eq <| pow_two _)
  /-
    🎉 no goals
  -/


@[to_additive lt_max_of_two_nsmul_lt_add]
theorem lt_max_of_sq_lt_mul {a b c : M} (h : a ^ 2 < b * c) : a < max b c := by
  /-
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftMono M
    inst✝ : MulRightMono M
    a b c : M
    h : LT.lt (HPow.hPow a 2) (HMul.hMul b c)
    ⊢ LT.lt a (Max.max b c)
  -/
  simpa using min_lt_max_of_mul_lt_mul ((pow_two _).symm.trans_lt h)
  /-
    🎉 no goals
  -/


@[to_additive le_of_nsmul_le_nsmul_right]
theorem le_of_pow_le_pow_left' {a b : M} {n : ℕ} (hn : n ≠ 0) : a ^ n ≤ b ^ n → a ≤ b :=
  (pow_left_strictMono hn).le_iff_le.1


@[to_additive min_le_of_add_le_two_nsmul]
theorem min_le_of_mul_le_sq {a b c : M} (h : a * b ≤ c ^ 2) : min a b ≤ c := by
  /-
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftStrictMono M
    inst✝ : MulRightStrictMono M
    a b c : M
    h : LE.le (HMul.hMul a b) (HPow.hPow c 2)
    ⊢ LE.le (Min.min a b) c
  -/
  simpa using min_le_max_of_mul_le_mul (h.trans_eq <| pow_two _)
  /-
    🎉 no goals
  -/


@[to_additive le_max_of_two_nsmul_le_add]
theorem le_max_of_sq_le_mul {a b c : M} (h : a ^ 2 ≤ b * c) : a ≤ max b c := by
  /-
    M : Type u_3
    inst✝³ : Monoid M
    inst✝² : LinearOrder M
    inst✝¹ : MulLeftStrictMono M
    inst✝ : MulRightStrictMono M
    a b c : M
    h : LE.le (HPow.hPow a 2) (HMul.hMul b c)
    ⊢ LE.le a (Max.max b c)
  -/
  simpa using min_le_max_of_mul_le_mul ((pow_two _).symm.trans_le h)
  /-
    🎉 no goals
  -/


@[to_additive Left.nsmul_neg_iff]
theorem Left.pow_lt_one_iff' [MulLeftStrictMono M] {n : ℕ} {x : M} (hn : 0 < n) :
    x ^ n < 1 ↔ x < 1 :=
  haveI := mulLeftMono_of_mulLeftStrictMono M
  pow_lt_one_iff hn.ne'


theorem Left.pow_lt_one_iff [MulLeftStrictMono M] {n : ℕ} {x : M} (hn : 0 < n) :
    x ^ n < 1 ↔ x < 1 := Left.pow_lt_one_iff' hn


@[to_additive]
theorem Right.pow_lt_one_iff [MulRightStrictMono M] {n : ℕ} {x : M}
    (hn : 0 < n) : x ^ n < 1 ↔ x < 1 :=
  haveI := mulRightMono_of_mulRightStrictMono M
  ⟨fun H => not_le.mp fun k => H.not_le <| Right.one_le_pow_of_le k, Right.pow_lt_one_of_lt hn⟩


@[to_additive zsmul_nonneg]
theorem one_le_zpow {x : G} (H : 1 ≤ x) {n : ℤ} (hn : 0 ≤ n) : 1 ≤ x ^ n := by
  /-
    G : Type u_2
    inst✝² : DivInvMonoid G
    inst✝¹ : Preorder G
    inst✝ : MulLeftMono G
    x : G
    H : LE.le 1 x
    n : Int
    hn : LE.le 0 n
    ⊢ LE.le 1 (HPow.hPow x n)
  -/
  lift n to ℕ using hn
  /-
    case intro
    G : Type u_2
    inst✝² : DivInvMonoid G
    inst✝¹ : Preorder G
    inst✝ : MulLeftMono G
    x : G
    H : LE.le 1 x
    n : Nat
    ⊢ LE.le 1 (HPow.hPow x ↑n)
  -/
  rw [zpow_natCast]
  /-
    case intro
    G : Type u_2
    inst✝² : DivInvMonoid G
    inst✝¹ : Preorder G
    inst✝ : MulLeftMono G
    x : G
    H : LE.le 1 x
    n : Nat
    ⊢ LE.le 1 (HPow.hPow x n)
  -/
  apply one_le_pow_of_one_le' H
  /-
    🎉 no goals
  -/


