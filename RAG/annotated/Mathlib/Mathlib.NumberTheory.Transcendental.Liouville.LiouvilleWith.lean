/-- We say that a real number `x` is a Liouville number with exponent `p : ℝ` if there exists a real
number `C` such that for infinitely many denominators `n` there exists a numerator `m` such that
`x ≠ m / n` and `|x - m / n| < C / n ^ p`.

A number is a Liouville number in the sense of `Liouville` if it is `LiouvilleWith` any real
exponent. -/
def LiouvilleWith (p x : ℝ) : Prop :=
  ∃ C, ∃ᶠ n : ℕ in atTop, ∃ m : ℤ, x ≠ m / n ∧ |x - m / n| < C / n ^ p


/-- For `p = 1` (hence, for any `p ≤ 1`), the condition `LiouvilleWith p x` is trivial. -/
theorem liouvilleWith_one (x : ℝ) : LiouvilleWith 1 x := by
  /-
    x : Real
    ⊢ LiouvilleWith 1 x
  -/
  use 2
  /-
    case h
    x : Real
    ⊢ Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT …
  -/
  refine ((eventually_gt_atTop 0).mono fun n hn => ?_).frequently
  /-
    case h
    x : Real
    n : Nat
    hn : LT.lt 0 n
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  have hn' : (0 : ℝ) < n := by simpa
  have : x < ↑(⌊x * ↑n⌋ + 1) / ↑n := by
    rw [lt_div_iff₀ hn', Int.cast_add, Int.cast_one]
    exact Int.lt_floor_add_one _
  /-
    case h
    x : Real
    n : Nat
    hn : LT.lt 0 n
    hn' : LT.lt 0 ↑n
    this : LT.lt x (HDiv.hDiv ↑(HAdd.hAdd (Int.floor (HMul.hMul x ↑n)) 1) ↑n)
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  refine ⟨⌊x * n⌋ + 1, this.ne, ?_⟩
  rw [abs_sub_comm, abs_of_pos (sub_pos.2 this), rpow_one, sub_lt_iff_lt_add',
    add_div_eq_mul_add_div _ _ hn'.ne']
  /-
    case h
    x : Real
    n : Nat
    hn : LT.lt 0 n
    hn' : LT.lt 0 ↑n
    this : LT.lt x (HDiv.hDiv ↑(HAdd.hAdd (Int.floor (HMul.hMul x ↑n)) 1) ↑n)
    ⊢ LT.lt (HDiv.hDiv ↑(HAdd.hAdd (Int.floor (HMul.hMul x ↑n)) 1) ↑n) (HDiv.hDiv  …
  -/
  gcongr
  calc _ ≤ x * n + 1 := by push_cast; gcongr; apply Int.floor_le
    _ < x * n + 2 := by linarith


/-- The constant `C` provided by the definition of `LiouvilleWith` can be made positive.
We also add `1 ≤ n` to the list of assumptions about the denominator. While it is equivalent to
the original statement, the case `n = 0` breaks many arguments. -/
theorem exists_pos (h : LiouvilleWith p x) :
    ∃ (C : ℝ) (_h₀ : 0 < C),
      ∃ᶠ n : ℕ in atTop, 1 ≤ n ∧ ∃ m : ℤ, x ≠ m / n ∧ |x - m / n| < C / n ^ p := by
  /-
    p x : Real
    h : LiouvilleWith p x
    ⊢ Exists fun C => Exists fun _h₀ => Filter.Frequently (fun n => And (LE.le 1 n …
  -/
  rcases h with ⟨C, hC⟩
  /-
    case intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    ⊢ Exists fun C => Exists fun _h₀ => Filter.Frequently (fun n => And (LE.le 1 n …
  -/
  refine ⟨max C 1, zero_lt_one.trans_le <| le_max_right _ _, ?_⟩
  /-
    case intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    ⊢ Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (HDiv …
  -/
  refine ((eventually_ge_atTop 1).and_frequently hC).mono ?_
  /-
    case intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    ⊢ ∀ (x_1 : Nat), And (LE.le 1 x_1) (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑x …
  -/
  rintro n ⟨hle, m, hne, hlt⟩
  /-
    case intro.intro.intro.intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    hle : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ And (LE.le 1 n) (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (H …
  -/
  refine ⟨hle, m, hne, hlt.trans_le ?_⟩
  /-
    case intro.intro.intro.intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    hle : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ LE.le (HDiv.hDiv C (HPow.hPow (↑n) p)) (HDiv.hDiv (Max.max C 1) (HPow.hPow ( …
  -/
  gcongr
  /-
    case intro.intro.intro.intro.hab
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    hle : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ LE.le C (Max.max C 1)
  -/
  apply le_max_left
  /-
    🎉 no goals
  -/


/-- If a number is Liouville with exponent `p`, then it is Liouville with any smaller exponent. -/
theorem mono (h : LiouvilleWith p x) (hle : q ≤ p) : LiouvilleWith q x := by
  /-
    p q x : Real
    h : LiouvilleWith p x
    hle : LE.le q p
    ⊢ LiouvilleWith q x
  -/
  rcases h.exists_pos with ⟨C, hC₀, hC⟩
  /-
    case intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hle : LE.le q p
    C : Real
    hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    ⊢ LiouvilleWith q x
  -/
  refine ⟨C, hC.mono ?_⟩; rintro n ⟨hn, m, hne, hlt⟩
  /-
    case intro.intro.intro.intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hle : LE.le q p
    C : Real
    hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  refine ⟨m, hne, hlt.trans_le <| ?_⟩
  /-
    case intro.intro.intro.intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hle : LE.le q p
    C : Real
    hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ LE.le (HDiv.hDiv C (HPow.hPow (↑n) p)) (HDiv.hDiv C (HPow.hPow (↑n) q))
  -/
  gcongr
  /-
    case intro.intro.intro.intro.intro.h.hx
    p q x : Real
    h : LiouvilleWith p x
    hle : LE.le q p
    C : Real
    hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ LE.le 1 ↑n
  -/
  exact_mod_cast hn
  /-
    🎉 no goals
  -/


/-- If `x` satisfies Liouville condition with exponent `p` and `q < p`, then `x`
satisfies Liouville condition with exponent `q` and constant `1`. -/
theorem frequently_lt_rpow_neg (h : LiouvilleWith p x) (hlt : q < p) :
    ∃ᶠ n : ℕ in atTop, ∃ m : ℤ, x ≠ m / n ∧ |x - m / n| < n ^ (-q) := by
  /-
    p q x : Real
    h : LiouvilleWith p x
    hlt : LT.lt q p
    ⊢ Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT …
  -/
  rcases h.exists_pos with ⟨C, _hC₀, hC⟩
  have : ∀ᶠ n : ℕ in atTop, C < n ^ (p - q) := by
    simpa only [(· ∘ ·), neg_sub, one_div] using
      ((tendsto_rpow_atTop (sub_pos.2 hlt)).comp tendsto_natCast_atTop_atTop).eventually
        (eventually_gt_atTop C)
  /-
    case intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hlt : LT.lt q p
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    this : Filter.Eventually (fun n => LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))) F …
    ⊢ Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT …
  -/
  refine (this.and_frequently hC).mono ?_
  /-
    case intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hlt : LT.lt q p
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    this : Filter.Eventually (fun n => LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))) F …
    ⊢ ∀ (x_1 : Nat), And (LT.lt C (HPow.hPow (↑x_1) (HSub.hSub p q))) (And (LE.le  …
  -/
  rintro n ⟨hnC, hn, m, hne, hlt⟩
  /-
    case intro.intro.intro.intro.intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hlt✝ : LT.lt q p
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    this : Filter.Eventually (fun n => LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))) F …
    n : Nat
    hnC : LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  replace hn : (0 : ℝ) < n := Nat.cast_pos.2 hn
  /-
    case intro.intro.intro.intro.intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hlt✝ : LT.lt q p
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    this : Filter.Eventually (fun n => LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))) F …
    n : Nat
    hnC : LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    hn : LT.lt 0 ↑n
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  refine ⟨m, hne, hlt.trans <| (div_lt_iff₀ <| rpow_pos_of_pos hn _).2 ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    p q x : Real
    h : LiouvilleWith p x
    hlt✝ : LT.lt q p
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    this : Filter.Eventually (fun n => LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))) F …
    n : Nat
    hnC : LT.lt C (HPow.hPow (↑n) (HSub.hSub p q))
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    hn : LT.lt 0 ↑n
    ⊢ LT.lt C (HMul.hMul (HPow.hPow (↑n) (Neg.neg q)) (HPow.hPow (↑n) p))
  -/
  rwa [mul_comm, ← rpow_add hn, ← sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- The product of a Liouville number and a nonzero rational number is again a Liouville number. -/
theorem mul_rat (h : LiouvilleWith p x) (hr : r ≠ 0) : LiouvilleWith p (x * r) := by
  /-
    p x : Real
    r : Rat
    h : LiouvilleWith p x
    hr : Ne r 0
    ⊢ LiouvilleWith p (HMul.hMul x ↑r)
  -/
  rcases h.exists_pos with ⟨C, _hC₀, hC⟩
  /-
    case intro.intro
    p x : Real
    r : Rat
    h : LiouvilleWith p x
    hr : Ne r 0
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    ⊢ LiouvilleWith p (HMul.hMul x ↑r)
  -/
  refine ⟨r.den ^ p * (|r| * C), (tendsto_id.nsmul_atTop r.pos).frequently (hC.mono ?_)⟩
  /-
    case intro.intro
    p x : Real
    r : Rat
    h : LiouvilleWith p x
    hr : Ne r 0
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    ⊢ ∀ (x_1 : Nat), And (LE.le 1 x_1) (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑x …
  -/
  rintro n ⟨_hn, m, hne, hlt⟩
  have A : (↑(r.num * m) : ℝ) / ↑(r.den • id n) = m / n * r := by
    simp [← div_mul_div_comm, ← r.cast_def, mul_comm]
  /-
    case intro.intro.intro.intro.intro
    p x : Real
    r : Rat
    h : LiouvilleWith p x
    hr : Ne r 0
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    _hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
    ⊢ Exists fun m => And (Ne (HMul.hMul x ↑r) (HDiv.hDiv ↑m ↑(HSMul.hSMul r.den ( …
  -/
  refine ⟨r.num * m, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      p x : Real
      r : Rat
      h : LiouvilleWith p x
      hr : Ne r 0
      C : Real
      _hC₀ : LT.lt 0 C
      hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
      n : Nat
      _hn : LE.le 1 n
      m : Int
      hne : Ne x (HDiv.hDiv ↑m ↑n)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
      A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
      ⊢ Ne (HMul.hMul x ↑r) (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id  …
    -/
  · rw [A]; simp [hne, hr]
            /-
              🎉 no goals
            -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      p x : Real
      r : Rat
      h : LiouvilleWith p x
      hr : Ne r 0
      C : Real
      _hC₀ : LT.lt 0 C
      hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
      n : Nat
      _hn : LE.le 1 n
      m : Int
      hne : Ne x (HDiv.hDiv ↑m ↑n)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
      A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
      ⊢ LT.lt (abs (HSub.hSub (HMul.hMul x ↑r) (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSM …
    -/
  · rw [A, ← sub_mul, abs_mul]
    /-
      case intro.intro.intro.intro.intro.refine_2
      p x : Real
      r : Rat
      h : LiouvilleWith p x
      hr : Ne r 0
      C : Real
      _hC₀ : LT.lt 0 C
      hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
      n : Nat
      _hn : LE.le 1 n
      m : Int
      hne : Ne x (HDiv.hDiv ↑m ↑n)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
      A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
      ⊢ LT.lt (HMul.hMul (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (abs ↑r)) (HDiv.hDiv  …
    -/
    simp only [smul_eq_mul, id, Nat.cast_mul]
    calc _ < C / ↑n ^ p * |↑r| := by gcongr
      _ = ↑r.den ^ p * (↑|r| * C) / (↑r.den * ↑n) ^ p := ?_
    /-
      case intro.intro.intro.intro.intro.refine_2
      p x : Real
      r : Rat
      h : LiouvilleWith p x
      hr : Ne r 0
      C : Real
      _hC₀ : LT.lt 0 C
      hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
      n : Nat
      _hn : LE.le 1 n
      m : Int
      hne : Ne x (HDiv.hDiv ↑m ↑n)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
      A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
      ⊢ Eq (HMul.hMul (HDiv.hDiv C (HPow.hPow (↑n) p)) (abs ↑r)) (HDiv.hDiv (HMul.hM …
    -/
    rw [mul_rpow, mul_div_mul_left, mul_comm, mul_div_assoc]
      /-
        case intro.intro.intro.intro.intro.refine_2
        p x : Real
        r : Rat
        h : LiouvilleWith p x
        hr : Ne r 0
        C : Real
        _hC₀ : LT.lt 0 C
        hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
        n : Nat
        _hn : LE.le 1 n
        m : Int
        hne : Ne x (HDiv.hDiv ↑m ↑n)
        hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
        A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
        ⊢ Eq (HMul.hMul (abs ↑r) (HDiv.hDiv C (HPow.hPow (↑n) p))) (HMul.hMul (↑(abs r …
      -/
    · simp only [Rat.cast_abs, le_refl]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.refine_2.hc
      p x : Real
      r : Rat
      h : LiouvilleWith p x
      hr : Ne r 0
      C : Real
      _hC₀ : LT.lt 0 C
      hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
      n : Nat
      _hn : LE.le 1 n
      m : Int
      hne : Ne x (HDiv.hDiv ↑m ↑n)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
      A : Eq (HDiv.hDiv ↑(HMul.hMul r.num m) ↑(HSMul.hSMul r.den (id n))) (HMul.hMul …
      ⊢ Ne (HPow.hPow (↑r.den) p) 0
    -/
    all_goals positivity
    /-
      🎉 no goals
    -/


/-- The product `x * r`, `r : ℚ`, `r ≠ 0`, is a Liouville number with exponent `p` if and only if
`x` satisfies the same condition. -/
theorem mul_rat_iff (hr : r ≠ 0) : LiouvilleWith p (x * r) ↔ LiouvilleWith p x :=
  ⟨fun h => by
    simpa only [mul_assoc, ← Rat.cast_mul, mul_inv_cancel₀ hr, Rat.cast_one, mul_one] using
      h.mul_rat (inv_ne_zero hr),
    fun h => h.mul_rat hr⟩


/-- The product `r * x`, `r : ℚ`, `r ≠ 0`, is a Liouville number with exponent `p` if and only if
`x` satisfies the same condition. -/
theorem rat_mul_iff (hr : r ≠ 0) : LiouvilleWith p (r * x) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    r : Rat
    hr : Ne r 0
    ⊢ Iff (LiouvilleWith p (HMul.hMul (↑r) x)) (LiouvilleWith p x)
  -/
  rw [mul_comm, mul_rat_iff hr]
  /-
    🎉 no goals
  -/


theorem rat_mul (h : LiouvilleWith p x) (hr : r ≠ 0) : LiouvilleWith p (r * x) :=
  (rat_mul_iff hr).2 h


theorem mul_int_iff (hm : m ≠ 0) : LiouvilleWith p (x * m) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    m : Int
    hm : Ne m 0
    ⊢ Iff (LiouvilleWith p (HMul.hMul x ↑m)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_intCast, mul_rat_iff (Int.cast_ne_zero.2 hm)]
  /-
    🎉 no goals
  -/


theorem mul_int (h : LiouvilleWith p x) (hm : m ≠ 0) : LiouvilleWith p (x * m) :=
  (mul_int_iff hm).2 h


theorem int_mul_iff (hm : m ≠ 0) : LiouvilleWith p (m * x) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    m : Int
    hm : Ne m 0
    ⊢ Iff (LiouvilleWith p (HMul.hMul (↑m) x)) (LiouvilleWith p x)
  -/
  rw [mul_comm, mul_int_iff hm]
  /-
    🎉 no goals
  -/


theorem int_mul (h : LiouvilleWith p x) (hm : m ≠ 0) : LiouvilleWith p (m * x) :=
  (int_mul_iff hm).2 h


theorem mul_nat_iff (hn : n ≠ 0) : LiouvilleWith p (x * n) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LiouvilleWith p (HMul.hMul x ↑n)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_natCast, mul_rat_iff (Nat.cast_ne_zero.2 hn)]
  /-
    🎉 no goals
  -/


theorem mul_nat (h : LiouvilleWith p x) (hn : n ≠ 0) : LiouvilleWith p (x * n) :=
  (mul_nat_iff hn).2 h


theorem nat_mul_iff (hn : n ≠ 0) : LiouvilleWith p (n * x) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LiouvilleWith p (HMul.hMul (↑n) x)) (LiouvilleWith p x)
  -/
  rw [mul_comm, mul_nat_iff hn]
  /-
    🎉 no goals
  -/


theorem nat_mul (h : LiouvilleWith p x) (hn : n ≠ 0) : LiouvilleWith p (n * x) := by
  /-
    p x : Real
    n : Nat
    h : LiouvilleWith p x
    hn : Ne n 0
    ⊢ LiouvilleWith p (HMul.hMul (↑n) x)
  -/
  rw [mul_comm]; exact h.mul_nat hn
                 /-
                   🎉 no goals
                 -/


theorem add_rat (h : LiouvilleWith p x) (r : ℚ) : LiouvilleWith p (x + r) := by
  /-
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    ⊢ LiouvilleWith p (HAdd.hAdd x ↑r)
  -/
  rcases h.exists_pos with ⟨C, _hC₀, hC⟩
  /-
    case intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    ⊢ LiouvilleWith p (HAdd.hAdd x ↑r)
  -/
  refine ⟨r.den ^ p * C, (tendsto_id.nsmul_atTop r.pos).frequently (hC.mono ?_)⟩
  /-
    case intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    ⊢ ∀ (x_1 : Nat), And (LE.le 1 x_1) (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑x …
  -/
  rintro n ⟨hn, m, hne, hlt⟩
  have : (↑(r.den * m + r.num * n : ℤ) / ↑(r.den • id n) : ℝ) = m / n + r := by
    rw [Algebra.id.smul_eq_mul, id]
    nth_rewrite 4 [← Rat.num_div_den r]
    push_cast
    rw [add_div, mul_div_mul_left _ _ (by positivity), mul_div_mul_right _ _ (by positivity)]
  /-
    case intro.intro.intro.intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    this : Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul (↑r.den) m) (HMul.hMul r.num ↑n))  …
    ⊢ Exists fun m => And (Ne (HAdd.hAdd x ↑r) (HDiv.hDiv ↑m ↑(HSMul.hSMul r.den ( …
  -/
  refine ⟨r.den * m + r.num * n, ?_⟩; rw [this, add_sub_add_right_eq_sub]
  /-
    case intro.intro.intro.intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    this : Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul (↑r.den) m) (HMul.hMul r.num ↑n))  …
    ⊢ And (Ne (HAdd.hAdd x ↑r) (HAdd.hAdd (HDiv.hDiv ↑m ↑n) ↑r)) (LT.lt (abs (HSub …
  -/
  refine ⟨by simpa, hlt.trans_le (le_of_eq ?_)⟩
  /-
    case intro.intro.intro.intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    this : Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul (↑r.den) m) (HMul.hMul r.num ↑n))  …
    ⊢ Eq (HDiv.hDiv C (HPow.hPow (↑n) p)) (HDiv.hDiv (HMul.hMul (HPow.hPow (↑r.den …
  -/
  have : (r.den ^ p : ℝ) ≠ 0 := by positivity
  /-
    case intro.intro.intro.intro.intro
    p x : Real
    h : LiouvilleWith p x
    r : Rat
    C : Real
    _hC₀ : LT.lt 0 C
    hC : Filter.Frequently (fun n => And (LE.le 1 n) (Exists fun m => And (Ne x (H …
    n : Nat
    hn : LE.le 1 n
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    this✝ : Eq (HDiv.hDiv ↑(HAdd.hAdd (HMul.hMul (↑r.den) m) (HMul.hMul r.num ↑n)) …
    this : Ne (HPow.hPow (↑r.den) p) 0
    ⊢ Eq (HDiv.hDiv C (HPow.hPow (↑n) p)) (HDiv.hDiv (HMul.hMul (HPow.hPow (↑r.den …
  -/
  simp [mul_rpow, Nat.cast_nonneg, mul_div_mul_left, this]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_rat_iff : LiouvilleWith p (x + r) ↔ LiouvilleWith p x :=
               /-
                 p x : Real
                 r : Rat
                 h : LiouvilleWith p (HAdd.hAdd x ↑r)
                 ⊢ LiouvilleWith p x
               -/
  ⟨fun h => by simpa using h.add_rat (-r), fun h => h.add_rat r⟩
               /-
                 🎉 no goals
               -/


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          r : Rat
                                                                          ⊢ Iff (LiouvilleWith p (HAdd.hAdd (↑r) x)) (LiouvilleWith p x)
                                                                        -/
theorem rat_add_iff : LiouvilleWith p (r + x) ↔ LiouvilleWith p x := by rw [add_comm, add_rat_iff]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem rat_add (h : LiouvilleWith p x) (r : ℚ) : LiouvilleWith p (r + x) :=
  add_comm x r ▸ h.add_rat r


@[simp]
theorem add_int_iff : LiouvilleWith p (x + m) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    m : Int
    ⊢ Iff (LiouvilleWith p (HAdd.hAdd x ↑m)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_intCast m, add_rat_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          m : Int
                                                                          ⊢ Iff (LiouvilleWith p (HAdd.hAdd (↑m) x)) (LiouvilleWith p x)
                                                                        -/
theorem int_add_iff : LiouvilleWith p (m + x) ↔ LiouvilleWith p x := by rw [add_comm, add_int_iff]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem add_nat_iff : LiouvilleWith p (x + n) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    n : Nat
    ⊢ Iff (LiouvilleWith p (HAdd.hAdd x ↑n)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_natCast n, add_rat_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          n : Nat
                                                                          ⊢ Iff (LiouvilleWith p (HAdd.hAdd (↑n) x)) (LiouvilleWith p x)
                                                                        -/
theorem nat_add_iff : LiouvilleWith p (n + x) ↔ LiouvilleWith p x := by rw [add_comm, add_nat_iff]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem add_int (h : LiouvilleWith p x) (m : ℤ) : LiouvilleWith p (x + m) :=
  add_int_iff.2 h


theorem int_add (h : LiouvilleWith p x) (m : ℤ) : LiouvilleWith p (m + x) :=
  int_add_iff.2 h


theorem add_nat (h : LiouvilleWith p x) (n : ℕ) : LiouvilleWith p (x + n) :=
  h.add_int n


theorem nat_add (h : LiouvilleWith p x) (n : ℕ) : LiouvilleWith p (n + x) :=
  h.int_add n


protected theorem neg (h : LiouvilleWith p x) : LiouvilleWith p (-x) := by
  /-
    p x : Real
    h : LiouvilleWith p x
    ⊢ LiouvilleWith p (Neg.neg x)
  -/
  rcases h with ⟨C, hC⟩
  /-
    case intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    ⊢ LiouvilleWith p (Neg.neg x)
  -/
  refine ⟨C, hC.mono ?_⟩
  /-
    case intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    ⊢ ∀ (x_1 : Nat), (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑x_1)) (LT.lt (abs ( …
  -/
  rintro n ⟨m, hne, hlt⟩
  /-
    case intro.intro.intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ Exists fun m => And (Ne (Neg.neg x) (HDiv.hDiv ↑m ↑n)) (LT.lt (abs (HSub.hSu …
  -/
  refine ⟨-m, by simp [neg_div, hne], ?_⟩
  /-
    case intro.intro.intro
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ LT.lt (abs (HSub.hSub (Neg.neg x) (HDiv.hDiv ↑(Neg.neg m) ↑n))) (HDiv.hDiv C …
  -/
  convert hlt using 1
  /-
    case h.e'_3
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ Eq (abs (HSub.hSub (Neg.neg x) (HDiv.hDiv ↑(Neg.neg m) ↑n))) (abs (HSub.hSub …
  -/
  rw [abs_sub_comm]
  /-
    case h.e'_3
    p x C : Real
    hC : Filter.Frequently (fun n => Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑n))  …
    n : Nat
    m : Int
    hne : Ne x (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑m ↑n))) (HDiv.hDiv C (HPow.hPow (↑n) …
    ⊢ Eq (abs (HSub.hSub (HDiv.hDiv ↑(Neg.neg m) ↑n) (Neg.neg x))) (abs (HSub.hSub …
  -/
  congr! 1; push_cast; ring
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem neg_iff : LiouvilleWith p (-x) ↔ LiouvilleWith p x :=
  ⟨fun h => neg_neg x ▸ h.neg, LiouvilleWith.neg⟩


@[simp]
theorem sub_rat_iff : LiouvilleWith p (x - r) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    r : Rat
    ⊢ Iff (LiouvilleWith p (HSub.hSub x ↑r)) (LiouvilleWith p x)
  -/
  rw [sub_eq_add_neg, ← Rat.cast_neg, add_rat_iff]
  /-
    🎉 no goals
  -/


theorem sub_rat (h : LiouvilleWith p x) (r : ℚ) : LiouvilleWith p (x - r) :=
  sub_rat_iff.2 h


@[simp]
theorem sub_int_iff : LiouvilleWith p (x - m) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    m : Int
    ⊢ Iff (LiouvilleWith p (HSub.hSub x ↑m)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_intCast, sub_rat_iff]
  /-
    🎉 no goals
  -/


theorem sub_int (h : LiouvilleWith p x) (m : ℤ) : LiouvilleWith p (x - m) :=
  sub_int_iff.2 h


@[simp]
theorem sub_nat_iff : LiouvilleWith p (x - n) ↔ LiouvilleWith p x := by
  /-
    p x : Real
    n : Nat
    ⊢ Iff (LiouvilleWith p (HSub.hSub x ↑n)) (LiouvilleWith p x)
  -/
  rw [← Rat.cast_natCast, sub_rat_iff]
  /-
    🎉 no goals
  -/


theorem sub_nat (h : LiouvilleWith p x) (n : ℕ) : LiouvilleWith p (x - n) :=
  sub_nat_iff.2 h


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          r : Rat
                                                                          ⊢ Iff (LiouvilleWith p (HSub.hSub (↑r) x)) (LiouvilleWith p x)
                                                                        -/
theorem rat_sub_iff : LiouvilleWith p (r - x) ↔ LiouvilleWith p x := by simp [sub_eq_add_neg]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem rat_sub (h : LiouvilleWith p x) (r : ℚ) : LiouvilleWith p (r - x) :=
  rat_sub_iff.2 h


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          m : Int
                                                                          ⊢ Iff (LiouvilleWith p (HSub.hSub (↑m) x)) (LiouvilleWith p x)
                                                                        -/
theorem int_sub_iff : LiouvilleWith p (m - x) ↔ LiouvilleWith p x := by simp [sub_eq_add_neg]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem int_sub (h : LiouvilleWith p x) (m : ℤ) : LiouvilleWith p (m - x) :=
  int_sub_iff.2 h


@[simp]
                                                                        /-
                                                                          p x : Real
                                                                          n : Nat
                                                                          ⊢ Iff (LiouvilleWith p (HSub.hSub (↑n) x)) (LiouvilleWith p x)
                                                                        -/
theorem nat_sub_iff : LiouvilleWith p (n - x) ↔ LiouvilleWith p x := by simp [sub_eq_add_neg]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem nat_sub (h : LiouvilleWith p x) (n : ℕ) : LiouvilleWith p (n - x) :=
  nat_sub_iff.2 h


theorem ne_cast_int (h : LiouvilleWith p x) (hp : 1 < p) (m : ℤ) : x ≠ m := by
  /-
    p x : Real
    h : LiouvilleWith p x
    hp : LT.lt 1 p
    m : Int
    ⊢ Ne x ↑m
  -/
  rintro rfl; rename' m => M
  rcases ((eventually_gt_atTop 0).and_frequently (h.frequently_lt_rpow_neg hp)).exists with
    ⟨n : ℕ, hn : 0 < n, m : ℤ, hne : (M : ℝ) ≠ m / n, hlt : |(M - m / n : ℝ)| < n ^ (-1 : ℝ)⟩
  /-
    case intro.intro.intro.intro
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    ⊢ False
  -/
  refine hlt.not_le ?_
  /-
    case intro.intro.intro.intro
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    ⊢ LE.le (HPow.hPow (↑n) (-1)) (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n)))
  -/
  have hn' : (0 : ℝ) < n := by simpa
  /-
    case intro.intro.intro.intro
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ LE.le (HPow.hPow (↑n) (-1)) (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n)))
  -/
  rw [rpow_neg_one, ← one_div, sub_div' _ _ _ hn'.ne', abs_div, Nat.abs_cast]
  /-
    case intro.intro.intro.intro
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ LE.le (HDiv.hDiv 1 ↑n) (HDiv.hDiv (abs (HSub.hSub (HMul.hMul ↑M ↑n) ↑m)) ↑n)
  -/
  gcongr
  /-
    case intro.intro.intro.intro.hab
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ LE.le 1 (abs (HSub.hSub (HMul.hMul ↑M ↑n) ↑m))
  -/
  norm_cast
  /-
    case intro.intro.intro.intro.hab
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ LE.le 1 (abs (HSub.hSub (HMul.hMul M ↑n) m))
  -/
  rw [← zero_add (1 : ℤ), Int.add_one_le_iff, abs_pos, sub_ne_zero]
  /-
    case intro.intro.intro.intro.hab
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Ne (↑M) (HDiv.hDiv ↑m ↑n)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ Ne (HMul.hMul M ↑n) m
  -/
  rw [Ne, eq_div_iff hn'.ne'] at hne
  /-
    case intro.intro.intro.intro.hab
    p : Real
    hp : LT.lt 1 p
    M : Int
    h : LiouvilleWith p ↑M
    n : Nat
    hn : LT.lt 0 n
    m : Int
    hne : Not (Eq (HMul.hMul ↑M ↑n) ↑m)
    hlt : LT.lt (abs (HSub.hSub (↑M) (HDiv.hDiv ↑m ↑n))) (HPow.hPow (↑n) (-1))
    hn' : LT.lt 0 ↑n
    ⊢ Ne (HMul.hMul M ↑n) m
  -/
  exact mod_cast hne
  /-
    🎉 no goals
  -/


/-- A number satisfying the Liouville condition with exponent `p > 1` is an irrational number. -/
protected theorem irrational (h : LiouvilleWith p x) (hp : 1 < p) : Irrational x := by
  /-
    p x : Real
    h : LiouvilleWith p x
    hp : LT.lt 1 p
    ⊢ Irrational x
  -/
  rintro ⟨r, rfl⟩
  /-
    case intro
    p : Real
    hp : LT.lt 1 p
    r : Rat
    h : LiouvilleWith p ↑r
    ⊢ False
  -/
  rcases eq_or_ne r 0 with (rfl | h0)
    /-
      case intro.inl
      p : Real
      hp : LT.lt 1 p
      h : LiouvilleWith p ↑0
      ⊢ False
    -/
  · refine h.ne_cast_int hp 0 ?_; rw [Rat.cast_zero, Int.cast_zero]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case intro.inr
      p : Real
      hp : LT.lt 1 p
      r : Rat
      h : LiouvilleWith p ↑r
      h0 : Ne r 0
      ⊢ False
    -/
  · refine (h.mul_rat (inv_ne_zero h0)).ne_cast_int hp 1 ?_
    /-
      case intro.inr
      p : Real
      hp : LT.lt 1 p
      r : Rat
      h : LiouvilleWith p ↑r
      h0 : Ne r 0
      ⊢ Eq (HMul.hMul ↑r ↑(Inv.inv r)) ↑1
    -/
    rw [Rat.cast_inv, mul_inv_cancel₀]
    /-
      case intro.inr
      p : Real
      hp : LT.lt 1 p
      r : Rat
      h : LiouvilleWith p ↑r
      h0 : Ne r 0
      ⊢ Eq 1 ↑1
    -/
    exacts [Int.cast_one.symm, Rat.cast_ne_zero.mpr h0]
    /-
      🎉 no goals
    -/


/-- If `x` is a Liouville number, then for any `n`, for infinitely many denominators `b` there
exists a numerator `a` such that `x ≠ a / b` and `|x - a / b| < 1 / b ^ n`. -/
theorem frequently_exists_num (hx : Liouville x) (n : ℕ) :
    ∃ᶠ b : ℕ in atTop, ∃ a : ℤ, x ≠ a / b ∧ |x - a / b| < 1 / (b : ℝ) ^ n := by
  /-
    x : Real
    hx : Liouville x
    n : Nat
    ⊢ Filter.Frequently (fun b => Exists fun a => And (Ne x (HDiv.hDiv ↑a ↑b)) (LT …
  -/
  refine Classical.not_not.1 fun H => ?_
  simp only [Liouville, not_forall, not_exists, not_frequently, not_and, not_lt,
    eventually_atTop] at H
  /-
    x : Real
    hx : Liouville x
    n : Nat
    H : Exists fun a => ∀ (b : Nat), GE.ge b a → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x …
    ⊢ False
  -/
  rcases H with ⟨N, hN⟩
  have : ∀ b > (1 : ℕ), ∀ᶠ m : ℕ in atTop, ∀ a : ℤ, 1 / (b : ℝ) ^ m ≤ |x - a / b| := by
    intro b hb
    replace hb : (1 : ℝ) < b := Nat.one_lt_cast.2 hb
    have H : Tendsto (fun m => 1 / (b : ℝ) ^ m : ℕ → ℝ) atTop (𝓝 0) := by
      simp only [one_div]
      exact tendsto_inv_atTop_zero.comp (tendsto_pow_atTop_atTop_of_one_lt hb)
    refine (H.eventually (hx.irrational.eventually_forall_le_dist_cast_div b)).mono ?_
    exact fun m hm a => hm a
  have : ∀ᶠ m : ℕ in atTop, ∀ b < N, 1 < b → ∀ a : ℤ, 1 / (b : ℝ) ^ m ≤ |x - a / b| :=
    (finite_lt_nat N).eventually_all.2 fun b _hb => eventually_imp_distrib_left.2 (this b)
  /-
    case intro
    x : Real
    hx : Liouville x
    n N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
    this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
    this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
    ⊢ False
  -/
  rcases (this.and (eventually_ge_atTop n)).exists with ⟨m, hm, hnm⟩
  /-
    case intro.intro.intro
    x : Real
    hx : Liouville x
    n N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
    this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
    this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
    m : Nat
    hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
    hnm : LE.le n m
    ⊢ False
  -/
  rcases hx m with ⟨a, b, hb, hne, hlt⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    x : Real
    hx : Liouville x
    n N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
    this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
    this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
    m : Nat
    hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
    hnm : LE.le n m
    a b : Int
    hb : LT.lt 1 b
    hne : Ne x (HDiv.hDiv ↑a ↑b)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ False
  -/
  lift b to ℕ using zero_le_one.trans hb.le; norm_cast at hb; push_cast at hne hlt
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    x : Real
    hx : Liouville x
    n N : Nat
    hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
    this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
    this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
    m : Nat
    hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
    hnm : LE.le n m
    a : Int
    b : Nat
    hb : LT.lt 1 b
    hne : Ne x (HDiv.hDiv ↑a ↑b)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ False
  -/
  rcases le_or_lt N b with h | h
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl
      x : Real
      hx : Liouville x
      n N : Nat
      hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
      this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
      this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
      m : Nat
      hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
      hnm : LE.le n m
      a : Int
      b : Nat
      hb : LT.lt 1 b
      hne : Ne x (HDiv.hDiv ↑a ↑b)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
      h : LE.le N b
      ⊢ False
    -/
  · refine (hN b h a hne).not_lt (hlt.trans_le ?_)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl
      x : Real
      hx : Liouville x
      n N : Nat
      hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
      this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
      this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
      m : Nat
      hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
      hnm : LE.le n m
      a : Int
      b : Nat
      hb : LT.lt 1 b
      hne : Ne x (HDiv.hDiv ↑a ↑b)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
      h : LE.le N b
      ⊢ LE.le (HDiv.hDiv 1 (HPow.hPow (↑b) m)) (HDiv.hDiv 1 (HPow.hPow (↑b) n))
    -/
    gcongr
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inl.h.ha
      x : Real
      hx : Liouville x
      n N : Nat
      hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
      this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
      this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
      m : Nat
      hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
      hnm : LE.le n m
      a : Int
      b : Nat
      hb : LT.lt 1 b
      hne : Ne x (HDiv.hDiv ↑a ↑b)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
      h : LE.le N b
      ⊢ LE.le 1 ↑b
    -/
    exact_mod_cast hb.le
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.inr
      x : Real
      hx : Liouville x
      n N : Nat
      hN : ∀ (b : Nat), GE.ge b N → ∀ (x_1 : Int), Ne x (HDiv.hDiv ↑x_1 ↑b) → LE.le  …
      this✝ : ∀ (b : Nat), GT.gt b 1 → Filter.Eventually (fun m => ∀ (a : Int), LE.l …
      this : Filter.Eventually (fun m => ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : …
      m : Nat
      hm : ∀ (b : Nat), LT.lt b N → LT.lt 1 b → ∀ (a : Int), LE.le (HDiv.hDiv 1 (HPo …
      hnm : LE.le n m
      a : Int
      b : Nat
      hb : LT.lt 1 b
      hne : Ne x (HDiv.hDiv ↑a ↑b)
      hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
      h : LT.lt b N
      ⊢ False
    -/
  · exact (hm b h hb _).not_lt hlt
    /-
      🎉 no goals
    -/


/-- A Liouville number is a Liouville number with any real exponent. -/
protected theorem liouvilleWith (hx : Liouville x) (p : ℝ) : LiouvilleWith p x := by
  /-
    x : Real
    hx : Liouville x
    p : Real
    ⊢ LiouvilleWith p x
  -/
  suffices LiouvilleWith ⌈p⌉₊ x from this.mono (Nat.le_ceil p)
  /-
    x : Real
    hx : Liouville x
    p : Real
    ⊢ LiouvilleWith (↑(Nat.ceil p)) x
  -/
  refine ⟨1, ((eventually_gt_atTop 1).and_frequently (hx.frequently_exists_num ⌈p⌉₊)).mono ?_⟩
  /-
    x : Real
    hx : Liouville x
    p : Real
    ⊢ ∀ (x_1 : Nat), And (LT.lt 1 x_1) (Exists fun a => And (Ne x (HDiv.hDiv ↑a ↑x …
  -/
  rintro b ⟨_hb, a, hne, hlt⟩
  /-
    case intro.intro.intro
    x : Real
    hx : Liouville x
    p : Real
    b : Nat
    _hb : LT.lt 1 b
    a : Int
    hne : Ne x (HDiv.hDiv ↑a ↑b)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑b)) (LT.lt (abs (HSub.hSub x (HDiv. …
  -/
  refine ⟨a, hne, ?_⟩
  /-
    case intro.intro.intro
    x : Real
    hx : Liouville x
    p : Real
    b : Nat
    _hb : LT.lt 1 b
    a : Int
    hne : Ne x (HDiv.hDiv ↑a ↑b)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow ↑b ↑(Nat …
  -/
  rwa [rpow_natCast]
  /-
    🎉 no goals
  -/


/-- A number satisfies the Liouville condition with any exponent if and only if it is a Liouville
number. -/
theorem forall_liouvilleWith_iff {x : ℝ} : (∀ p, LiouvilleWith p x) ↔ Liouville x := by
  /-
    x : Real
    ⊢ Iff (∀ (p : Real), LiouvilleWith p x) (Liouville x)
  -/
  refine ⟨fun H n => ?_, Liouville.liouvilleWith⟩
  rcases ((eventually_gt_atTop 1).and_frequently
    ((H (n + 1)).frequently_lt_rpow_neg (lt_add_one (n : ℝ)))).exists
    with ⟨b, hb, a, hne, hlt⟩
  /-
    case intro.intro.intro.intro
    x : Real
    H : ∀ (p : Real), LiouvilleWith p x
    n b : Nat
    hb : LT.lt 1 b
    a : Int
    hne : Ne x (HDiv.hDiv ↑a ↑b)
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HPow.hPow (↑b) (Neg.neg ↑n))
    ⊢ Exists fun a => Exists fun b => And (LT.lt 1 b) (And (Ne x (HDiv.hDiv ↑a ↑b) …
  -/
  exact ⟨a, b, mod_cast hb, hne, by simpa [rpow_neg] using hlt⟩
  /-
    🎉 no goals
  -/

