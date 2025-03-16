/-- The body of the sum in `cantorFunction`.
`cantorFunctionAux c f n = c ^ n` if `f n = true`;
`cantorFunctionAux c f n = 0` if `f n = false`. -/
def cantorFunctionAux (c : ℝ) (f : ℕ → Bool) (n : ℕ) : ℝ :=
  cond (f n) (c ^ n) 0


@[simp]
theorem cantorFunctionAux_true (h : f n = true) : cantorFunctionAux c f n = c ^ n := by
  /-
    c : Real
    f : Nat → Bool
    n : Nat
    h : Eq (f n) Bool.true
    ⊢ Eq (Cardinal.cantorFunctionAux c f n) (HPow.hPow c n)
  -/
  simp [cantorFunctionAux, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem cantorFunctionAux_false (h : f n = false) : cantorFunctionAux c f n = 0 := by
  /-
    c : Real
    f : Nat → Bool
    n : Nat
    h : Eq (f n) Bool.false
    ⊢ Eq (Cardinal.cantorFunctionAux c f n) 0
  -/
  simp [cantorFunctionAux, h]
  /-
    🎉 no goals
  -/


theorem cantorFunctionAux_nonneg (h : 0 ≤ c) : 0 ≤ cantorFunctionAux c f n := by
  /-
    c : Real
    f : Nat → Bool
    n : Nat
    h : LE.le 0 c
    ⊢ LE.le 0 (Cardinal.cantorFunctionAux c f n)
  -/
  cases h' : f n
    /-
      case false
      c : Real
      f : Nat → Bool
      n : Nat
      h : LE.le 0 c
      h' : Eq (f n) Bool.false
      ⊢ LE.le 0 (Cardinal.cantorFunctionAux c f n)
    -/
  · simp [h']
    /-
      🎉 no goals
    -/
    /-
      case true
      c : Real
      f : Nat → Bool
      n : Nat
      h : LE.le 0 c
      h' : Eq (f n) Bool.true
      ⊢ LE.le 0 (Cardinal.cantorFunctionAux c f n)
    -/
  · simpa [h'] using pow_nonneg h _
    /-
      🎉 no goals
    -/


theorem cantorFunctionAux_eq (h : f n = g n) :
                                                            /-
                                                              c : Real
                                                              f g : Nat → Bool
                                                              n : Nat
                                                              h : Eq (f n) (g n)
                                                              ⊢ Eq (Cardinal.cantorFunctionAux c f n) (Cardinal.cantorFunctionAux c g n)
                                                            -/
    cantorFunctionAux c f n = cantorFunctionAux c g n := by simp [cantorFunctionAux, h]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem cantorFunctionAux_zero (f : ℕ → Bool) : cantorFunctionAux c f 0 = cond (f 0) 1 0 := by
  /-
    c : Real
    f : Nat → Bool
    ⊢ Eq (Cardinal.cantorFunctionAux c f 0) (cond (f 0) 1 0)
  -/
                    /-
                      🎉 no goals
                    -/
  cases h : f 0 <;> simp [h]
                    /-
                      🎉 no goals
                    -/


theorem cantorFunctionAux_succ (f : ℕ → Bool) :
    (fun n => cantorFunctionAux c f (n + 1)) = fun n =>
      c * cantorFunctionAux c (fun n => f (n + 1)) n := by
  /-
    c : Real
    f : Nat → Bool
    ⊢ Eq (fun n => Cardinal.cantorFunctionAux c f (HAdd.hAdd n 1)) fun n => HMul.h …
  -/
  ext n
  /-
    case h
    c : Real
    f : Nat → Bool
    n : Nat
    ⊢ Eq (Cardinal.cantorFunctionAux c f (HAdd.hAdd n 1)) (HMul.hMul c (Cardinal.c …
  -/
                          /-
                            🎉 no goals
                          -/
  cases h : f (n + 1) <;> simp [h, _root_.pow_succ']
                          /-
                            🎉 no goals
                          -/


theorem summable_cantor_function (f : ℕ → Bool) (h1 : 0 ≤ c) (h2 : c < 1) :
    Summable (cantorFunctionAux c f) := by
  /-
    c : Real
    f : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    ⊢ Summable (Cardinal.cantorFunctionAux c f)
  -/
  apply (summable_geometric_of_lt_one h1 h2).summable_of_eq_zero_or_self
  /-
    c : Real
    f : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    ⊢ ∀ (b : Nat), Or (Eq (Cardinal.cantorFunctionAux c f b) 0) (Eq (Cardinal.cant …
  -/
                             /-
                               🎉 no goals
                             -/
  intro n; cases h : f n <;> simp [h]
                             /-
                               🎉 no goals
                             -/


/-- `cantorFunction c (f : ℕ → Bool)` is `Σ n, f n * c ^ n`, where `true` is interpreted as `1` and
`false` is interpreted as `0`. It is implemented using `cantorFunctionAux`. -/
def cantorFunction (c : ℝ) (f : ℕ → Bool) : ℝ :=
  ∑' n, cantorFunctionAux c f n


theorem cantorFunction_le (h1 : 0 ≤ c) (h2 : c < 1) (h3 : ∀ n, f n → g n) :
    cantorFunction c f ≤ cantorFunction c g := by
  /-
    c : Real
    f g : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    h3 : ∀ (n : Nat), Eq (f n) Bool.true → Eq (g n) Bool.true
    ⊢ LE.le (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
  -/
  apply tsum_le_tsum _ (summable_cantor_function f h1 h2) (summable_cantor_function g h1 h2)
  /-
    c : Real
    f g : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    h3 : ∀ (n : Nat), Eq (f n) Bool.true → Eq (g n) Bool.true
    ⊢ ∀ (i : Nat), LE.le (Cardinal.cantorFunctionAux c f i) (Cardinal.cantorFuncti …
  -/
  intro n; cases h : f n
    /-
      case false
      c : Real
      f g : Nat → Bool
      h1 : LE.le 0 c
      h2 : LT.lt c 1
      h3 : ∀ (n : Nat), Eq (f n) Bool.true → Eq (g n) Bool.true
      n : Nat
      h : Eq (f n) Bool.false
      ⊢ LE.le (Cardinal.cantorFunctionAux c f n) (Cardinal.cantorFunctionAux c g n)
    -/
  · simp [h, cantorFunctionAux_nonneg h1]
    /-
      🎉 no goals
    -/
  /-
    case true
    c : Real
    f g : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    h3 : ∀ (n : Nat), Eq (f n) Bool.true → Eq (g n) Bool.true
    n : Nat
    h : Eq (f n) Bool.true
    ⊢ LE.le (Cardinal.cantorFunctionAux c f n) (Cardinal.cantorFunctionAux c g n)
  -/
  replace h3 : g n = true := h3 n h; simp [h, h3]
                                     /-
                                       🎉 no goals
                                     -/


theorem cantorFunction_succ (f : ℕ → Bool) (h1 : 0 ≤ c) (h2 : c < 1) :
    cantorFunction c f = cond (f 0) 1 0 + c * cantorFunction c fun n => f (n + 1) := by
  /-
    c : Real
    f : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    ⊢ Eq (Cardinal.cantorFunction c f) (HAdd.hAdd (cond (f 0) 1 0) (HMul.hMul c (C …
  -/
  rw [cantorFunction, tsum_eq_zero_add (summable_cantor_function f h1 h2)]
  /-
    c : Real
    f : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    ⊢ Eq (HAdd.hAdd (Cardinal.cantorFunctionAux c f 0) (tsum fun b => Cardinal.can …
  -/
  rw [cantorFunctionAux_succ, tsum_mul_left, cantorFunctionAux, _root_.pow_zero]
  /-
    c : Real
    f : Nat → Bool
    h1 : LE.le 0 c
    h2 : LT.lt c 1
    ⊢ Eq (HAdd.hAdd (cond (f 0) 1 0) (HMul.hMul c (tsum fun x => Cardinal.cantorFu …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `cantorFunction c` is strictly increasing with if `0 < c < 1/2`, if we endow `ℕ → Bool` with a
lexicographic order. The lexicographic order doesn't exist for these infinitary products, so we
explicitly write out what it means. -/
theorem increasing_cantorFunction (h1 : 0 < c) (h2 : c < 1 / 2) {n : ℕ} {f g : ℕ → Bool}
    (hn : ∀ k < n, f k = g k) (fn : f n = false) (gn : g n = true) :
    cantorFunction c f < cantorFunction c g := by
  have h3 : c < 1 := by
    apply h2.trans
    norm_num
  /-
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    n : Nat
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k n → Eq (f k) (g k)
    fn : Eq (f n) Bool.false
    gn : Eq (g n) Bool.true
    h3 : LT.lt c 1
    ⊢ LT.lt (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
  -/
  induction' n with n ih generalizing f g
    /-
      case zero
      c : Real
      h1 : LT.lt 0 c
      h2 : LT.lt c (1 / 2)
      h3 : LT.lt c 1
      f g : Nat → Bool
      hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
      fn : Eq (f 0) Bool.false
      gn : Eq (g 0) Bool.true
      ⊢ LT.lt (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
    -/
  · let f_max : ℕ → Bool := fun n => Nat.rec false (fun _ _ => true) n
    have hf_max : ∀ n, f n → f_max n := by
      intro n hn
      cases n
      · rw [fn] at hn
        contradiction
      apply rfl
    /-
      case zero
      c : Real
      h1 : LT.lt 0 c
      h2 : LT.lt c (1 / 2)
      h3 : LT.lt c 1
      f g : Nat → Bool
      hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
      fn : Eq (f 0) Bool.false
      gn : Eq (g 0) Bool.true
      f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
      hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
      ⊢ LT.lt (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
    -/
    let g_min : ℕ → Bool := fun n => Nat.rec true (fun _ _ => false) n
    have hg_min : ∀ n, g_min n → g n := by
      intro n hn
      cases n
      · rw [gn]
      simp at hn
    /-
      case zero
      c : Real
      h1 : LT.lt 0 c
      h2 : LT.lt c (1 / 2)
      h3 : LT.lt c 1
      f g : Nat → Bool
      hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
      fn : Eq (f 0) Bool.false
      gn : Eq (g 0) Bool.true
      f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
      hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
      g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
      hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
      ⊢ LT.lt (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
    -/
    apply (cantorFunction_le (le_of_lt h1) h3 hf_max).trans_lt
    /-
      case zero
      c : Real
      h1 : LT.lt 0 c
      h2 : LT.lt c (1 / 2)
      h3 : LT.lt c 1
      f g : Nat → Bool
      hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
      fn : Eq (f 0) Bool.false
      gn : Eq (g 0) Bool.true
      f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
      hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
      g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
      hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
      ⊢ LT.lt (Cardinal.cantorFunction c f_max) (Cardinal.cantorFunction c g)
    -/
    refine lt_of_lt_of_le ?_ (cantorFunction_le (le_of_lt h1) h3 hg_min)
    have : c / (1 - c) < 1 := by
      rw [div_lt_one, lt_sub_iff_add_lt]
      · convert _root_.add_lt_add h2 h2
        norm_num
      rwa [sub_pos]
    /-
      case zero
      c : Real
      h1 : LT.lt 0 c
      h2 : LT.lt c (1 / 2)
      h3 : LT.lt c 1
      f g : Nat → Bool
      hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
      fn : Eq (f 0) Bool.false
      gn : Eq (g 0) Bool.true
      f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
      hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
      g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
      hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
      this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
      ⊢ LT.lt (Cardinal.cantorFunction c f_max) (Cardinal.cantorFunction c g_min)
    -/
    convert this
    · rw [cantorFunction_succ _ (le_of_lt h1) h3, div_eq_mul_inv, ←
        tsum_geometric_of_lt_one (le_of_lt h1) h3]
      /-
        case h.e'_3
        c : Real
        h1 : LT.lt 0 c
        h2 : LT.lt c (1 / 2)
        h3 : LT.lt c 1
        f g : Nat → Bool
        hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
        fn : Eq (f 0) Bool.false
        gn : Eq (g 0) Bool.true
        f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
        hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
        g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
        hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
        this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
        ⊢ Eq (HAdd.hAdd (cond (f_max 0) 1 0) (HMul.hMul c (Cardinal.cantorFunction c f …
      -/
      apply zero_add
      /-
        🎉 no goals
      -/
      /-
        case h.e'_4
        c : Real
        h1 : LT.lt 0 c
        h2 : LT.lt c (1 / 2)
        h3 : LT.lt c 1
        f g : Nat → Bool
        hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
        fn : Eq (f 0) Bool.false
        gn : Eq (g 0) Bool.true
        f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
        hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
        g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
        hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
        this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
        ⊢ Eq (Cardinal.cantorFunction c g_min) 1
      -/
    · refine (tsum_eq_single 0 ?_).trans ?_
        /-
          case h.e'_4.refine_1
          c : Real
          h1 : LT.lt 0 c
          h2 : LT.lt c (1 / 2)
          h3 : LT.lt c 1
          f g : Nat → Bool
          hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
          fn : Eq (f 0) Bool.false
          gn : Eq (g 0) Bool.true
          f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
          hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
          g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
          hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
          this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
          ⊢ ∀ (b' : Nat), Ne b' 0 → Eq (Cardinal.cantorFunctionAux c g_min b') 0
        -/
      · intro n hn
        /-
          case h.e'_4.refine_1
          c : Real
          h1 : LT.lt 0 c
          h2 : LT.lt c (1 / 2)
          h3 : LT.lt c 1
          f g : Nat → Bool
          hn✝ : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
          fn : Eq (f 0) Bool.false
          gn : Eq (g 0) Bool.true
          f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
          hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
          g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
          hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
          this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
          n : Nat
          hn : Ne n 0
          ⊢ Eq (Cardinal.cantorFunctionAux c g_min n) 0
        -/
        cases n
          /-
            case h.e'_4.refine_1.zero
            c : Real
            h1 : LT.lt 0 c
            h2 : LT.lt c (1 / 2)
            h3 : LT.lt c 1
            f g : Nat → Bool
            hn✝ : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
            fn : Eq (f 0) Bool.false
            gn : Eq (g 0) Bool.true
            f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
            hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
            g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
            hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
            this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
            hn : Ne 0 0
            ⊢ Eq (Cardinal.cantorFunctionAux c g_min 0) 0
          -/
        · contradiction
          /-
            🎉 no goals
          -/
        /-
          case h.e'_4.refine_1.succ
          c : Real
          h1 : LT.lt 0 c
          h2 : LT.lt c (1 / 2)
          h3 : LT.lt c 1
          f g : Nat → Bool
          hn✝ : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
          fn : Eq (f 0) Bool.false
          gn : Eq (g 0) Bool.true
          f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
          hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
          g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
          hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
          this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
          n✝ : Nat
          hn : Ne (HAdd.hAdd n✝ 1) 0
          ⊢ Eq (Cardinal.cantorFunctionAux c g_min (HAdd.hAdd n✝ 1)) 0
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case h.e'_4.refine_2
          c : Real
          h1 : LT.lt 0 c
          h2 : LT.lt c (1 / 2)
          h3 : LT.lt c 1
          f g : Nat → Bool
          hn : ∀ (k : Nat), LT.lt k 0 → Eq (f k) (g k)
          fn : Eq (f 0) Bool.false
          gn : Eq (g 0) Bool.true
          f_max : Nat → Bool := fun n => Nat.rec Bool.false (fun x x => Bool.true) n
          hf_max : ∀ (n : Nat), Eq (f n) Bool.true → Eq (f_max n) Bool.true
          g_min : Nat → Bool := fun n => Nat.rec Bool.true (fun x x => Bool.false) n
          hg_min : ∀ (n : Nat), Eq (g_min n) Bool.true → Eq (g n) Bool.true
          this : LT.lt (HDiv.hDiv c (HSub.hSub 1 c)) 1
          ⊢ Eq (Cardinal.cantorFunctionAux c g_min 0) 1
        -/
      · exact cantorFunctionAux_zero _
        /-
          🎉 no goals
        -/
  /-
    case succ
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    h3 : LT.lt c 1
    n : Nat
    ih : ∀ {f g : Nat → Bool}, (∀ (k : Nat), LT.lt k n → Eq (f k) (g k)) → Eq (f n …
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (f k) (g k)
    fn : Eq (f (HAdd.hAdd n 1)) Bool.false
    gn : Eq (g (HAdd.hAdd n 1)) Bool.true
    ⊢ LT.lt (Cardinal.cantorFunction c f) (Cardinal.cantorFunction c g)
  -/
  rw [cantorFunction_succ f (le_of_lt h1) h3, cantorFunction_succ g (le_of_lt h1) h3]
  /-
    case succ
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    h3 : LT.lt c 1
    n : Nat
    ih : ∀ {f g : Nat → Bool}, (∀ (k : Nat), LT.lt k n → Eq (f k) (g k)) → Eq (f n …
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (f k) (g k)
    fn : Eq (f (HAdd.hAdd n 1)) Bool.false
    gn : Eq (g (HAdd.hAdd n 1)) Bool.true
    ⊢ LT.lt (HAdd.hAdd (cond (f 0) 1 0) (HMul.hMul c (Cardinal.cantorFunction c fu …
  -/
  rw [hn 0 <| zero_lt_succ n]
  /-
    case succ
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    h3 : LT.lt c 1
    n : Nat
    ih : ∀ {f g : Nat → Bool}, (∀ (k : Nat), LT.lt k n → Eq (f k) (g k)) → Eq (f n …
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (f k) (g k)
    fn : Eq (f (HAdd.hAdd n 1)) Bool.false
    gn : Eq (g (HAdd.hAdd n 1)) Bool.true
    ⊢ LT.lt (HAdd.hAdd (cond (g 0) 1 0) (HMul.hMul c (Cardinal.cantorFunction c fu …
  -/
  apply add_lt_add_left
  /-
    case succ.bc
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    h3 : LT.lt c 1
    n : Nat
    ih : ∀ {f g : Nat → Bool}, (∀ (k : Nat), LT.lt k n → Eq (f k) (g k)) → Eq (f n …
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (f k) (g k)
    fn : Eq (f (HAdd.hAdd n 1)) Bool.false
    gn : Eq (g (HAdd.hAdd n 1)) Bool.true
    ⊢ LT.lt (HMul.hMul c (Cardinal.cantorFunction c fun n => f (HAdd.hAdd n 1))) ( …
  -/
  rw [mul_lt_mul_left h1]
  /-
    case succ.bc
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    h3 : LT.lt c 1
    n : Nat
    ih : ∀ {f g : Nat → Bool}, (∀ (k : Nat), LT.lt k n → Eq (f k) (g k)) → Eq (f n …
    f g : Nat → Bool
    hn : ∀ (k : Nat), LT.lt k (HAdd.hAdd n 1) → Eq (f k) (g k)
    fn : Eq (f (HAdd.hAdd n 1)) Bool.false
    gn : Eq (g (HAdd.hAdd n 1)) Bool.true
    ⊢ LT.lt (Cardinal.cantorFunction c fun n => f (HAdd.hAdd n 1)) (Cardinal.canto …
  -/
  exact ih (fun k hk => hn _ <| Nat.succ_lt_succ hk) fn gn
  /-
    🎉 no goals
  -/


/-- `cantorFunction c` is injective if `0 < c < 1/2`. -/
theorem cantorFunction_injective (h1 : 0 < c) (h2 : c < 1 / 2) :
    Function.Injective (cantorFunction c) := by
  /-
    c : Real
    h1 : LT.lt 0 c
    h2 : LT.lt c (1 / 2)
    ⊢ Function.Injective (Cardinal.cantorFunction c)
  -/
  intro f g hfg
  classical
    by_contra h
    revert hfg
    have : ∃ n, f n ≠ g n := by
      rw [← not_forall]
      intro h'
      apply h
      ext
      apply h'
    let n := Nat.find this
    have hn : ∀ k : ℕ, k < n → f k = g k := by
      intro k hk
      apply of_not_not
      exact Nat.find_min this hk
    cases fn : f n
    · apply _root_.ne_of_lt
      refine increasing_cantorFunction h1 h2 hn fn ?_
      apply Bool.eq_true_of_not_eq_false
      rw [← fn]
      apply Ne.symm
      exact Nat.find_spec this
    · apply _root_.ne_of_gt
      refine increasing_cantorFunction h1 h2 (fun k hk => (hn k hk).symm) ?_ fn
      apply Bool.eq_false_of_not_eq_true
      rw [← fn]
      apply Ne.symm
      exact Nat.find_spec this


/-- The cardinality of the reals, as a type. -/
theorem mk_real : #ℝ = 𝔠 := by
  /-
    ⊢ Eq (Cardinal.mk Real) Cardinal.continuum
  -/
  apply le_antisymm
    /-
      case a
      ⊢ LE.le (Cardinal.mk Real) Cardinal.continuum
    -/
  · rw [Real.equivCauchy.cardinal_eq]
    /-
      case a
      ⊢ LE.le (Cardinal.mk (CauSeq.Completion.Cauchy abs)) Cardinal.continuum
    -/
    apply mk_quotient_le.trans
    /-
      case a
      ⊢ LE.le (Cardinal.mk (CauSeq Rat abs)) Cardinal.continuum
    -/
    apply (mk_subtype_le _).trans_eq
    /-
      case a
      ⊢ Eq (Cardinal.mk (Nat → Rat)) Cardinal.continuum
    -/
    rw [← power_def, mk_nat, mkRat, aleph0_power_aleph0]
    /-
      🎉 no goals
    -/
    /-
      case a
      ⊢ LE.le Cardinal.continuum (Cardinal.mk Real)
    -/
  · convert mk_le_of_injective (cantorFunction_injective _ _)
      /-
        case h.e'_3
        ⊢ Eq Cardinal.continuum (Cardinal.mk (Nat → Bool))
      -/
    · rw [← power_def, mk_bool, mk_nat, two_power_aleph0]
      /-
        🎉 no goals
      -/
      /-
        case a.convert_1
        ⊢ Real
      -/
    · exact 1 / 3
      /-
        🎉 no goals
      -/
      /-
        case a.convert_2
        ⊢ LT.lt 0 (1 / 3)
      -/
    · norm_num
      /-
        🎉 no goals
      -/
      /-
        case a.convert_3
        ⊢ LT.lt (1 / 3) (1 / 2)
      -/
    · norm_num
      /-
        🎉 no goals
      -/


/-- The cardinality of the reals, as a set. -/
                                                     /-
                                                       ⊢ Eq (Cardinal.mk ↑Set.univ) Cardinal.continuum
                                                     -/
theorem mk_univ_real : #(Set.univ : Set ℝ) = 𝔠 := by rw [mk_univ, mk_real]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- **Non-Denumerability of the Continuum**: The reals are not countable. -/
instance : Uncountable ℝ := by
  /-
    c : Real
    f g : Nat → Bool
    n : Nat
    ⊢ Uncountable Real
  -/
  rw [← aleph0_lt_mk_iff, mk_real]
  /-
    c : Real
    f g : Nat → Bool
    n : Nat
    ⊢ LT.lt Cardinal.aleph0 Cardinal.continuum
  -/
  exact aleph0_lt_continuum
  /-
    🎉 no goals
  -/


theorem not_countable_real : ¬(Set.univ : Set ℝ).Countable :=
  not_countable_univ


/-- The cardinality of the interval (a, ∞). -/
theorem mk_Ioi_real (a : ℝ) : #(Ioi a) = 𝔠 := by
  /-
    a : Real
    ⊢ Eq (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
  -/
  refine le_antisymm (mk_real ▸ mk_set_le _) ?_
  /-
    a : Real
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioi a))
  -/
  rw [← not_lt]
  /-
    a : Real
    ⊢ Not (LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum)
  -/
  intro h
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    ⊢ False
  -/
  refine _root_.ne_of_lt ?_ mk_univ_real
  have hu : Iio a ∪ {a} ∪ Ioi a = Set.univ := by
    convert @Iic_union_Ioi ℝ _ _
    exact Iio_union_right
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    ⊢ LT.lt (Cardinal.mk ↑Set.univ) Cardinal.continuum
  -/
  rw [← hu]
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    ⊢ LT.lt (Cardinal.mk ↑(Union.union (Union.union (Set.Iio a) (Singleton.singlet …
  -/
  refine lt_of_le_of_lt (mk_union_le _ _) ?_
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    ⊢ LT.lt (HAdd.hAdd (Cardinal.mk ↑(Union.union (Set.Iio a) (Singleton.singleton …
  -/
  refine lt_of_le_of_lt (add_le_add_right (mk_union_le _ _) _) ?_
  have h2 : (fun x => a + a - x) '' Ioi a = Iio a := by
    convert @image_const_sub_Ioi ℝ _ _ _
    simp
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Ioi a)) (Set.Ii …
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (Cardinal.mk ↑(Set.Iio a)) (Cardinal.mk ↑(Single …
  -/
  rw [← h2]
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Ioi a)) (Set.Ii …
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (Cardinal.mk ↑(Set.image (fun x => HSub.hSub (HA …
  -/
  refine add_lt_of_lt (cantor _).le ?_ h
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Ioi a)) (Set.Ii …
    ⊢ LT.lt (HAdd.hAdd (Cardinal.mk ↑(Set.image (fun x => HSub.hSub (HAdd.hAdd a a …
  -/
  refine add_lt_of_lt (cantor _).le (mk_image_le.trans_lt h) ?_
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Ioi a)) (Set.Ii …
    ⊢ LT.lt (Cardinal.mk ↑(Singleton.singleton a)) Cardinal.continuum
  -/
  rw [mk_singleton]
  /-
    a : Real
    h : LT.lt (Cardinal.mk ↑(Set.Ioi a)) Cardinal.continuum
    hu : Eq (Union.union (Union.union (Set.Iio a) (Singleton.singleton a)) (Set.Io …
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Ioi a)) (Set.Ii …
    ⊢ LT.lt 1 Cardinal.continuum
  -/
  exact one_lt_aleph0.trans (cantor _)
  /-
    🎉 no goals
  -/


/-- The cardinality of the interval [a, ∞). -/
theorem mk_Ici_real (a : ℝ) : #(Ici a) = 𝔠 :=
  le_antisymm (mk_real ▸ mk_set_le _) (mk_Ioi_real a ▸ mk_le_mk_of_subset Ioi_subset_Ici_self)


/-- The cardinality of the interval (-∞, a). -/
theorem mk_Iio_real (a : ℝ) : #(Iio a) = 𝔠 := by
  /-
    a : Real
    ⊢ Eq (Cardinal.mk ↑(Set.Iio a)) Cardinal.continuum
  -/
  refine le_antisymm (mk_real ▸ mk_set_le _) ?_
  have h2 : (fun x => a + a - x) '' Iio a = Ioi a := by
    simp only [image_const_sub_Iio, add_sub_cancel_right]
  /-
    a : Real
    h2 : Eq (Set.image (fun x => HSub.hSub (HAdd.hAdd a a) x) (Set.Iio a)) (Set.Io …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Iio a))
  -/
  exact mk_Ioi_real a ▸ h2 ▸ mk_image_le
  /-
    🎉 no goals
  -/


/-- The cardinality of the interval (-∞, a]. -/
theorem mk_Iic_real (a : ℝ) : #(Iic a) = 𝔠 :=
  le_antisymm (mk_real ▸ mk_set_le _) (mk_Iio_real a ▸ mk_le_mk_of_subset Iio_subset_Iic_self)


/-- The cardinality of the interval (a, b). -/
theorem mk_Ioo_real {a b : ℝ} (h : a < b) : #(Ioo a b) = 𝔠 := by
  /-
    a b : Real
    h : LT.lt a b
    ⊢ Eq (Cardinal.mk ↑(Set.Ioo a b)) Cardinal.continuum
  -/
  refine le_antisymm (mk_real ▸ mk_set_le _) ?_
  /-
    a b : Real
    h : LT.lt a b
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioo a b))
  -/
  have h1 : #((fun x => x - a) '' Ioo a b) ≤ #(Ioo a b) := mk_image_le
  /-
    a b : Real
    h : LT.lt a b
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioo a b))
  -/
  refine le_trans ?_ h1
  /-
    a b : Real
    h : LT.lt a b
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) ( …
  -/
  rw [image_sub_const_Ioo, sub_self]
  /-
    a b : Real
    h : LT.lt a b
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioo 0 (HSub.hSub b a)))
  -/
  replace h := sub_pos_of_lt h
  /-
    a b : Real
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    h : LT.lt 0 (HSub.hSub b a)
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioo 0 (HSub.hSub b a)))
  -/
  have h2 : #(Inv.inv '' Ioo 0 (b - a)) ≤ #(Ioo 0 (b - a)) := mk_image_le
  /-
    a b : Real
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    h : LT.lt 0 (HSub.hSub b a)
    h2 : LE.le (Cardinal.mk ↑(Set.image Inv.inv (Set.Ioo 0 (HSub.hSub b a)))) (Car …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.Ioo 0 (HSub.hSub b a)))
  -/
  refine le_trans ?_ h2
  /-
    a b : Real
    h1 : LE.le (Cardinal.mk ↑(Set.image (fun x => HSub.hSub x a) (Set.Ioo a b))) ( …
    h : LT.lt 0 (HSub.hSub b a)
    h2 : LE.le (Cardinal.mk ↑(Set.image Inv.inv (Set.Ioo 0 (HSub.hSub b a)))) (Car …
    ⊢ LE.le Cardinal.continuum (Cardinal.mk ↑(Set.image Inv.inv (Set.Ioo 0 (HSub.h …
  -/
  rw [image_inv_eq_inv, inv_Ioo_0_left h, mk_Ioi_real]
  /-
    🎉 no goals
  -/


/-- The cardinality of the interval [a, b). -/
theorem mk_Ico_real {a b : ℝ} (h : a < b) : #(Ico a b) = 𝔠 :=
  le_antisymm (mk_real ▸ mk_set_le _) (mk_Ioo_real h ▸ mk_le_mk_of_subset Ioo_subset_Ico_self)


/-- The cardinality of the interval [a, b]. -/
theorem mk_Icc_real {a b : ℝ} (h : a < b) : #(Icc a b) = 𝔠 :=
  le_antisymm (mk_real ▸ mk_set_le _) (mk_Ioo_real h ▸ mk_le_mk_of_subset Ioo_subset_Icc_self)


/-- The cardinality of the interval (a, b]. -/
theorem mk_Ioc_real {a b : ℝ} (h : a < b) : #(Ioc a b) = 𝔠 :=
  le_antisymm (mk_real ▸ mk_set_le _) (mk_Ioo_real h ▸ mk_le_mk_of_subset Ioo_subset_Ioc_self)


