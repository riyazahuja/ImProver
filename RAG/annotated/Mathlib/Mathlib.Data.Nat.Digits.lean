/-- (Impl.) An auxiliary definition for `digits`, to help get the desired definitional unfolding. -/
def digitsAux0 : ℕ → List ℕ
  | 0 => []
  | n + 1 => [n + 1]


/-- (Impl.) An auxiliary definition for `digits`, to help get the desired definitional unfolding. -/
def digitsAux1 (n : ℕ) : List ℕ :=
  List.replicate n 1


/-- (Impl.) An auxiliary definition for `digits`, to help get the desired definitional unfolding. -/
def digitsAux (b : ℕ) (h : 2 ≤ b) : ℕ → List ℕ
  | 0 => []
  | n + 1 =>
    ((n + 1) % b) :: digitsAux b h ((n + 1) / b)
/-
  b : Nat
  h : LE.le 2 b
  n : Nat
  ⊢ LT.lt (HDiv.hDiv (HAdd.hAdd n 1) b) n.succ
-/
decreasing_by exact Nat.div_lt_self (Nat.succ_pos _) h
/-
  🎉 no goals
-/


@[simp]
                                                                        /-
                                                                          b : Nat
                                                                          h : LE.le 2 b
                                                                          ⊢ Eq (b.digitsAux h 0) List.nil
                                                                        -/
theorem digitsAux_zero (b : ℕ) (h : 2 ≤ b) : digitsAux b h 0 = [] := by rw [digitsAux]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem digitsAux_def (b : ℕ) (h : 2 ≤ b) (n : ℕ) (w : 0 < n) :
    digitsAux b h n = (n % b) :: digitsAux b h (n / b) := by
  /-
    b : Nat
    h : LE.le 2 b
    n : Nat
    w : LT.lt 0 n
    ⊢ Eq (b.digitsAux h n) (List.cons (HMod.hMod n b) (b.digitsAux h (HDiv.hDiv n  …
  -/
  cases n
    /-
      case zero
      b : Nat
      h : LE.le 2 b
      w : LT.lt 0 0
      ⊢ Eq (b.digitsAux h 0) (List.cons (HMod.hMod 0 b) (b.digitsAux h (HDiv.hDiv 0  …
    -/
  · cases w
    /-
      🎉 no goals
    -/
    /-
      case succ
      b : Nat
      h : LE.le 2 b
      n✝ : Nat
      w : LT.lt 0 (HAdd.hAdd n✝ 1)
      ⊢ Eq (b.digitsAux h (HAdd.hAdd n✝ 1)) (List.cons (HMod.hMod (HAdd.hAdd n✝ 1) b …
    -/
  · rw [digitsAux]
    /-
      🎉 no goals
    -/


/-- `digits b n` gives the digits, in little-endian order,
of a natural number `n` in a specified base `b`.

In any base, we have `ofDigits b L = L.foldr (fun x y ↦ x + b * y) 0`.
* For any `2 ≤ b`, we have `l < b` for any `l ∈ digits b n`,
  and the last digit is not zero.
  This uniquely specifies the behaviour of `digits b`.
* For `b = 1`, we define `digits 1 n = List.replicate n 1`.
* For `b = 0`, we define `digits 0 n = [n]`, except `digits 0 0 = []`.

Note this differs from the existing `Nat.toDigits` in core, which is used for printing numerals.
In particular, `Nat.toDigits b 0 = ['0']`, while `digits b 0 = []`.
-/
def digits : ℕ → ℕ → List ℕ
  | 0 => digitsAux0
  | 1 => digitsAux1
                                   /-
                                     n b : Nat
                                     ⊢ LE.le 2 (HAdd.hAdd b 2)
                                   -/
  | b + 2 => digitsAux (b + 2) (by norm_num)
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem digits_zero (b : ℕ) : digits b 0 = [] := by
  /-
    b : Nat
    ⊢ Eq (b.digits 0) List.nil
  -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  rcases b with (_ | ⟨_ | ⟨_⟩⟩) <;> simp [digits, digitsAux0, digitsAux1]
                                    /-
                                      🎉 no goals
                                    -/


theorem digits_zero_zero : digits 0 0 = [] :=
  rfl


@[simp]
theorem digits_zero_succ (n : ℕ) : digits 0 n.succ = [n + 1] :=
  rfl


theorem digits_zero_succ' : ∀ {n : ℕ}, n ≠ 0 → digits 0 n = [n]
  | 0, h => (h rfl).elim
  | _ + 1, _ => rfl


@[simp]
theorem digits_one (n : ℕ) : digits 1 n = List.replicate n 1 :=
  rfl

-- @[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this

theorem digits_one_succ (n : ℕ) : digits 1 (n + 1) = 1 :: digits 1 n :=
  rfl


theorem digits_add_two_add_one (b n : ℕ) :
    digits (b + 2) (n + 1) = ((n + 1) % (b + 2)) :: digits (b + 2) ((n + 1) / (b + 2)) := by
  /-
    b n : Nat
    ⊢ Eq ((HAdd.hAdd b 2).digits (HAdd.hAdd n 1)) (List.cons (HMod.hMod (HAdd.hAdd …
  -/
  simp [digits, digitsAux_def]
  /-
    🎉 no goals
  -/


@[simp]
lemma digits_of_two_le_of_pos {b : ℕ} (hb : 2 ≤ b) (hn : 0 < n) :
    Nat.digits b n = n % b :: Nat.digits b (n / b) := by
  /-
    n b : Nat
    hb : LE.le 2 b
    hn : LT.lt 0 n
    ⊢ Eq (b.digits n) (List.cons (HMod.hMod n b) (b.digits (HDiv.hDiv n b)))
  -/
  rw [Nat.eq_add_of_sub_eq hb rfl, Nat.eq_add_of_sub_eq hn rfl, Nat.digits_add_two_add_one]
  /-
    🎉 no goals
  -/


theorem digits_def' :
    ∀ {b : ℕ} (_ : 1 < b) {n : ℕ} (_ : 0 < n), digits b n = (n % b) :: digits b (n / b)
                         /-
                           n✝ : Nat
                           h : LT.lt 1 0
                           ⊢ Not (LT.lt 1 0)
                         -/
  | 0, h => absurd h (by decide)
                         /-
                           🎉 no goals
                         -/
                         /-
                           n✝ : Nat
                           h : LT.lt 1 1
                           ⊢ Not (LT.lt 1 1)
                         -/
  | 1, h => absurd h (by decide)
                         /-
                           🎉 no goals
                         -/
                                    /-
                                      n✝ b : Nat
                                      x✝ : LT.lt 1 (HAdd.hAdd b 2)
                                      ⊢ LE.le 2 (HAdd.hAdd b 2)
                                    -/
  | b + 2, _ => digitsAux_def _ (by simp) _
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem digits_of_lt (b x : ℕ) (hx : x ≠ 0) (hxb : x < b) : digits b x = [x] := by
  /-
    b x : Nat
    hx : Ne x 0
    hxb : LT.lt x b
    ⊢ Eq (b.digits x) (List.cons x List.nil)
  -/
  rcases exists_eq_succ_of_ne_zero hx with ⟨x, rfl⟩
  /-
    case intro
    b x : Nat
    hx : Ne x.succ 0
    hxb : LT.lt x.succ b
    ⊢ Eq (b.digits x.succ) (List.cons x.succ List.nil)
  -/
  rcases Nat.exists_eq_add_of_le' ((Nat.le_add_left 1 x).trans_lt hxb) with ⟨b, rfl⟩
  /-
    case intro.intro
    x : Nat
    hx : Ne x.succ 0
    b : Nat
    hxb : LT.lt x.succ (HAdd.hAdd b (Nat.succ 1))
    ⊢ Eq ((HAdd.hAdd b (Nat.succ 1)).digits x.succ) (List.cons x.succ List.nil)
  -/
  rw [digits_add_two_add_one, div_eq_of_lt hxb, digits_zero, mod_eq_of_lt hxb]
  /-
    🎉 no goals
  -/


theorem digits_add (b : ℕ) (h : 1 < b) (x y : ℕ) (hxb : x < b) (hxy : x ≠ 0 ∨ y ≠ 0) :
    digits b (x + b * y) = x :: digits b y := by
  /-
    b : Nat
    h : LT.lt 1 b
    x y : Nat
    hxb : LT.lt x b
    hxy : Or (Ne x 0) (Ne y 0)
    ⊢ Eq (b.digits (HAdd.hAdd x (HMul.hMul b y))) (List.cons x (b.digits y))
  -/
  rcases Nat.exists_eq_add_of_le' h with ⟨b, rfl : _ = _ + 2⟩
  /-
    case intro
    x y : Nat
    hxy : Or (Ne x 0) (Ne y 0)
    b : Nat
    h : LT.lt 1 (HAdd.hAdd b 2)
    hxb : LT.lt x (HAdd.hAdd b 2)
    ⊢ Eq ((HAdd.hAdd b 2).digits (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) y))) (Lis …
  -/
  cases y
    /-
      case intro.zero
      x b : Nat
      h : LT.lt 1 (HAdd.hAdd b 2)
      hxb : LT.lt x (HAdd.hAdd b 2)
      hxy : Or (Ne x 0) (Ne 0 0)
      ⊢ Eq ((HAdd.hAdd b 2).digits (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) 0))) (Lis …
    -/
  · simp [hxb, hxy.resolve_right (absurd rfl)]
    /-
      🎉 no goals
    -/
  /-
    case intro.succ
    x b : Nat
    h : LT.lt 1 (HAdd.hAdd b 2)
    hxb : LT.lt x (HAdd.hAdd b 2)
    n✝ : Nat
    hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
    ⊢ Eq ((HAdd.hAdd b 2).digits (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAdd.hAd …
  -/
  dsimp [digits]
  /-
    case intro.succ
    x b : Nat
    h : LT.lt 1 (HAdd.hAdd b 2)
    hxb : LT.lt x (HAdd.hAdd b 2)
    n✝ : Nat
    hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
    ⊢ Eq ((HAdd.hAdd b 2).digitsAux ⋯ (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAd …
  -/
  rw [digitsAux_def]
    /-
      case intro.succ
      x b : Nat
      h : LT.lt 1 (HAdd.hAdd b 2)
      hxb : LT.lt x (HAdd.hAdd b 2)
      n✝ : Nat
      hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
      ⊢ Eq (List.cons (HMod.hMod (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAdd.hAdd  …
    -/
  · congr
      /-
        case intro.succ.e_head
        x b : Nat
        h : LT.lt 1 (HAdd.hAdd b 2)
        hxb : LT.lt x (HAdd.hAdd b 2)
        n✝ : Nat
        hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
        ⊢ Eq (HMod.hMod (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAdd.hAdd n✝ 1))) (HA …
      -/
    · simp [Nat.add_mod, mod_eq_of_lt hxb]
      /-
        🎉 no goals
      -/
      /-
        case intro.succ.e_tail.e_a
        x b : Nat
        h : LT.lt 1 (HAdd.hAdd b 2)
        hxb : LT.lt x (HAdd.hAdd b 2)
        n✝ : Nat
        hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAdd.hAdd n✝ 1))) (HA …
      -/
    · simp [add_mul_div_left, div_eq_of_lt hxb]
      /-
        🎉 no goals
      -/
    /-
      case intro.succ.w
      x b : Nat
      h : LT.lt 1 (HAdd.hAdd b 2)
      hxb : LT.lt x (HAdd.hAdd b 2)
      n✝ : Nat
      hxy : Or (Ne x 0) (Ne (HAdd.hAdd n✝ 1) 0)
      ⊢ LT.lt 0 (HAdd.hAdd x (HMul.hMul (HAdd.hAdd b 2) (HAdd.hAdd n✝ 1)))
    -/
  · apply Nat.succ_pos
    /-
      🎉 no goals
    -/

-- If we had a function converting a list into a polynomial,
-- and appropriate lemmas about that function,
-- we could rewrite this in terms of that.

/-- `ofDigits b L` takes a list `L` of natural numbers, and interprets them
as a number in semiring, as the little-endian digits in base `b`.
-/
def ofDigits {α : Type*} [Semiring α] (b : α) : List ℕ → α
  | [] => 0
  | h :: t => h + b * ofDigits b t


theorem ofDigits_eq_foldr {α : Type*} [Semiring α] (b : α) (L : List ℕ) :
    ofDigits b L = List.foldr (fun x y => ↑x + b * y) 0 L := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    b : α
    L : List Nat
    ⊢ Eq (Nat.ofDigits b L) (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hMul b y) …
  -/
  induction' L with d L ih
    /-
      case nil
      α : Type u_1
      inst✝ : Semiring α
      b : α
      ⊢ Eq (Nat.ofDigits b List.nil) (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hM …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : Semiring α
      b : α
      d : Nat
      L : List Nat
      ih : Eq (Nat.ofDigits b L) (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hMul b …
      ⊢ Eq (Nat.ofDigits b (List.cons d L)) (List.foldr (fun x y => HAdd.hAdd (↑x) ( …
    -/
  · dsimp [ofDigits]
    /-
      case cons
      α : Type u_1
      inst✝ : Semiring α
      b : α
      d : Nat
      L : List Nat
      ih : Eq (Nat.ofDigits b L) (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hMul b …
      ⊢ Eq (HAdd.hAdd (↑d) (HMul.hMul b (Nat.ofDigits b L))) (HAdd.hAdd (↑d) (HMul.h …
    -/
    rw [ih]
    /-
      🎉 no goals
    -/


theorem ofDigits_eq_sum_map_with_index_aux (b : ℕ) (l : List ℕ) :
    ((List.range l.length).zipWith ((fun i a : ℕ => a * b ^ (i + 1))) l).sum =
      b * ((List.range l.length).zipWith (fun i a => a * b ^ i) l).sum := by
  suffices
    (List.range l.length).zipWith (fun i a : ℕ => a * b ^ (i + 1)) l =
      (List.range l.length).zipWith (fun i a => b * (a * b ^ i)) l
    by simp [this]
  /-
    b : Nat
    l : List Nat
    ⊢ Eq (List.zipWith (fun i a => HMul.hMul a (HPow.hPow b (HAdd.hAdd i 1))) (Lis …
  -/
  congr; ext; simp [pow_succ]; ring
                               /-
                                 🎉 no goals
                               -/


theorem ofDigits_eq_sum_mapIdx (b : ℕ) (L : List ℕ) :
    ofDigits b L = (L.mapIdx fun i a => a * b ^ i).sum := by
  rw [List.mapIdx_eq_enum_map, List.enum_eq_zip_range, List.map_uncurry_zip_eq_zipWith,
    ofDigits_eq_foldr]
  /-
    b : Nat
    L : List Nat
    ⊢ Eq (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hMul b y)) 0 L) (List.zipWit …
  -/
  induction' L with hd tl hl
    /-
      case nil
      b : Nat
      ⊢ Eq (List.foldr (fun x y => HAdd.hAdd (↑x) (HMul.hMul b y)) 0 List.nil) (List …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simpa [List.range_succ_eq_map, List.zipWith_map_left, ofDigits_eq_sum_map_with_index_aux] using
      Or.inl hl


@[simp]
theorem ofDigits_nil {b : ℕ} : ofDigits b [] = 0 := rfl


@[simp]
                                                                /-
                                                                  b n : Nat
                                                                  ⊢ Eq (Nat.ofDigits b (List.cons n List.nil)) n
                                                                -/
theorem ofDigits_singleton {b n : ℕ} : ofDigits b [n] = n := by simp [ofDigits]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem ofDigits_one_cons {α : Type*} [Semiring α] (h : ℕ) (L : List ℕ) :
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : Semiring α
                                                         h : Nat
                                                         L : List Nat
                                                         ⊢ Eq (Nat.ofDigits 1 (List.cons h L)) (HAdd.hAdd (↑h) (Nat.ofDigits 1 L))
                                                       -/
    ofDigits (1 : α) (h :: L) = h + ofDigits 1 L := by simp [ofDigits]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem ofDigits_cons {b hd} {tl : List ℕ} :
    ofDigits b (hd :: tl) = hd + b * ofDigits b tl := rfl


theorem ofDigits_append {b : ℕ} {l1 l2 : List ℕ} :
    ofDigits b (l1 ++ l2) = ofDigits b l1 + b ^ l1.length * ofDigits b l2 := by
  /-
    b : Nat
    l1 l2 : List Nat
    ⊢ Eq (Nat.ofDigits b (HAppend.hAppend l1 l2)) (HAdd.hAdd (Nat.ofDigits b l1) ( …
  -/
  induction' l1 with hd tl IH
    /-
      case nil
      b : Nat
      l2 : List Nat
      ⊢ Eq (Nat.ofDigits b (HAppend.hAppend List.nil l2)) (HAdd.hAdd (Nat.ofDigits b …
    -/
  · simp [ofDigits]
    /-
      🎉 no goals
    -/
    /-
      case cons
      b : Nat
      l2 : List Nat
      hd : Nat
      tl : List Nat
      IH : Eq (Nat.ofDigits b (HAppend.hAppend tl l2)) (HAdd.hAdd (Nat.ofDigits b tl …
      ⊢ Eq (Nat.ofDigits b (HAppend.hAppend (List.cons hd tl) l2)) (HAdd.hAdd (Nat.o …
    -/
  · rw [ofDigits, List.cons_append, ofDigits, IH, List.length_cons, pow_succ']
    /-
      case cons
      b : Nat
      l2 : List Nat
      hd : Nat
      tl : List Nat
      IH : Eq (Nat.ofDigits b (HAppend.hAppend tl l2)) (HAdd.hAdd (Nat.ofDigits b tl …
      ⊢ Eq (HAdd.hAdd (↑hd) (HMul.hMul b (HAdd.hAdd (Nat.ofDigits b tl) (HMul.hMul ( …
    -/
    ring
    /-
      🎉 no goals
    -/


@[norm_cast]
theorem coe_ofDigits (α : Type*) [Semiring α] (b : ℕ) (L : List ℕ) :
    ((ofDigits b L : ℕ) : α) = ofDigits (b : α) L := by
  /-
    α : Type u_1
    inst✝ : Semiring α
    b : Nat
    L : List Nat
    ⊢ Eq (↑(Nat.ofDigits b L)) (Nat.ofDigits (↑b) L)
  -/
  induction' L with d L ih
    /-
      case nil
      α : Type u_1
      inst✝ : Semiring α
      b : Nat
      ⊢ Eq (↑(Nat.ofDigits b List.nil)) (Nat.ofDigits (↑b) List.nil)
    -/
  · simp [ofDigits]
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : Semiring α
      b d : Nat
      L : List Nat
      ih : Eq (↑(Nat.ofDigits b L)) (Nat.ofDigits (↑b) L)
      ⊢ Eq (↑(Nat.ofDigits b (List.cons d L))) (Nat.ofDigits (↑b) (List.cons d L))
    -/
  · dsimp [ofDigits]; push_cast; rw [ih]
                                 /-
                                   🎉 no goals
                                 -/


@[norm_cast]
theorem coe_int_ofDigits (b : ℕ) (L : List ℕ) : ((ofDigits b L : ℕ) : ℤ) = ofDigits (b : ℤ) L := by
  /-
    b : Nat
    L : List Nat
    ⊢ Eq (↑(Nat.ofDigits b L)) (Nat.ofDigits (↑b) L)
  -/
  induction' L with d L _
    /-
      case nil
      b : Nat
      ⊢ Eq (↑(Nat.ofDigits b List.nil)) (Nat.ofDigits (↑b) List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      b d : Nat
      L : List Nat
      tail_ih✝ : Eq (↑(Nat.ofDigits b L)) (Nat.ofDigits (↑b) L)
      ⊢ Eq (↑(Nat.ofDigits b (List.cons d L))) (Nat.ofDigits (↑b) (List.cons d L))
    -/
  · dsimp [ofDigits]; push_cast; simp only
                                 /-
                                   🎉 no goals
                                 -/


theorem digits_zero_of_eq_zero {b : ℕ} (h : b ≠ 0) :
    ∀ {L : List ℕ} (_ : ofDigits b L = 0), ∀ l ∈ L, l = 0
  | _ :: _, h0, _, List.Mem.head .. => Nat.eq_zero_of_add_eq_zero_right h0
  | _ :: _, h0, _, List.Mem.tail _ hL =>
    digits_zero_of_eq_zero h (mul_right_injective₀ h (Nat.eq_zero_of_add_eq_zero_left h0)) _ hL


theorem digits_ofDigits (b : ℕ) (h : 1 < b) (L : List ℕ) (w₁ : ∀ l ∈ L, l < b)
    (w₂ : ∀ h : L ≠ [], L.getLast h ≠ 0) : digits b (ofDigits b L) = L := by
  /-
    b : Nat
    h : LT.lt 1 b
    L : List Nat
    w₁ : ∀ (l : Nat), Membership.mem L l → LT.lt l b
    w₂ : ∀ (h : Ne L List.nil), Ne (L.getLast h) 0
    ⊢ Eq (b.digits (Nat.ofDigits b L)) L
  -/
  induction' L with d L ih
    /-
      case nil
      b : Nat
      h : LT.lt 1 b
      w₁ : ∀ (l : Nat), Membership.mem List.nil l → LT.lt l b
      w₂ : ∀ (h : Ne List.nil List.nil), Ne (List.nil.getLast h) 0
      ⊢ Eq (b.digits (Nat.ofDigits b List.nil)) List.nil
    -/
  · dsimp [ofDigits]
    /-
      case nil
      b : Nat
      h : LT.lt 1 b
      w₁ : ∀ (l : Nat), Membership.mem List.nil l → LT.lt l b
      w₂ : ∀ (h : Ne List.nil List.nil), Ne (List.nil.getLast h) 0
      ⊢ Eq (b.digits 0) List.nil
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      b : Nat
      h : LT.lt 1 b
      d : Nat
      L : List Nat
      ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
      w₂ : ∀ (h : Ne (List.cons d L) List.nil), Ne ((List.cons d L).getLast h) 0
      ⊢ Eq (b.digits (Nat.ofDigits b (List.cons d L))) (List.cons d L)
    -/
  · dsimp [ofDigits]
    /-
      case cons
      b : Nat
      h : LT.lt 1 b
      d : Nat
      L : List Nat
      ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
      w₂ : ∀ (h : Ne (List.cons d L) List.nil), Ne ((List.cons d L).getLast h) 0
      ⊢ Eq (b.digits (HAdd.hAdd d (HMul.hMul b (Nat.ofDigits b L)))) (List.cons d L)
    -/
    replace w₂ := w₂ (by simp)
    /-
      case cons
      b : Nat
      h : LT.lt 1 b
      d : Nat
      L : List Nat
      ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
      w₂ : Ne ((List.cons d L).getLast ⋯) 0
      ⊢ Eq (b.digits (HAdd.hAdd d (HMul.hMul b (Nat.ofDigits b L)))) (List.cons d L)
    -/
    rw [digits_add b h]
      /-
        case cons
        b : Nat
        h : LT.lt 1 b
        d : Nat
        L : List Nat
        ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
        w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
        w₂ : Ne ((List.cons d L).getLast ⋯) 0
        ⊢ Eq (List.cons d (b.digits (Nat.ofDigits b L))) (List.cons d L)
      -/
    · rw [ih]
        /-
          case cons.w₁
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          ⊢ ∀ (l : Nat), Membership.mem L l → LT.lt l b
        -/
      · intro l m
        /-
          case cons.w₁
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          l : Nat
          m : Membership.mem L l
          ⊢ LT.lt l b
        -/
        apply w₁
        /-
          case cons.w₁.a
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          l : Nat
          m : Membership.mem L l
          ⊢ Membership.mem (List.cons d L) l
        -/
        exact List.mem_cons_of_mem _ m
        /-
          🎉 no goals
        -/
        /-
          case cons.w₂
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          ⊢ ∀ (h : Ne L List.nil), Ne (L.getLast h) 0
        -/
      · intro h
        /-
          case cons.w₂
          b : Nat
          h✝ : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          h : Ne L List.nil
          ⊢ Ne (L.getLast h) 0
        -/
        rw [List.getLast_cons h] at w₂
        /-
          case cons.w₂
          b : Nat
          h✝ : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          h : Ne L List.nil
          w₂ : Ne (L.getLast h) 0
          ⊢ Ne (L.getLast h) 0
        -/
        convert w₂
        /-
          🎉 no goals
        -/
      /-
        case cons.hxb
        b : Nat
        h : LT.lt 1 b
        d : Nat
        L : List Nat
        ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
        w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
        w₂ : Ne ((List.cons d L).getLast ⋯) 0
        ⊢ LT.lt d b
      -/
    · exact w₁ d (List.mem_cons_self _ _)
      /-
        🎉 no goals
      -/
      /-
        case cons.hxy
        b : Nat
        h : LT.lt 1 b
        d : Nat
        L : List Nat
        ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
        w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
        w₂ : Ne ((List.cons d L).getLast ⋯) 0
        ⊢ Or (Ne d 0) (Ne (Nat.ofDigits b L) 0)
      -/
    · by_cases h' : L = []
        /-
          case pos
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          h' : Eq L List.nil
          ⊢ Or (Ne d 0) (Ne (Nat.ofDigits b L) 0)
        -/
      · rcases h' with rfl
        /-
          case pos
          b : Nat
          h : LT.lt 1 b
          d : Nat
          ih : (∀ (l : Nat), Membership.mem List.nil l → LT.lt l b) → (∀ (h : Ne List.ni …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d List.nil) l → LT.lt l b
          w₂ : Ne ((List.cons d List.nil).getLast ⋯) 0
          ⊢ Or (Ne d 0) (Ne (Nat.ofDigits b List.nil) 0)
        -/
        left
        /-
          case pos.h
          b : Nat
          h : LT.lt 1 b
          d : Nat
          ih : (∀ (l : Nat), Membership.mem List.nil l → LT.lt l b) → (∀ (h : Ne List.ni …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d List.nil) l → LT.lt l b
          w₂ : Ne ((List.cons d List.nil).getLast ⋯) 0
          ⊢ Ne d 0
        -/
        simpa using w₂
        /-
          🎉 no goals
        -/
        /-
          case neg
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          h' : Not (Eq L List.nil)
          ⊢ Or (Ne d 0) (Ne (Nat.ofDigits b L) 0)
        -/
      · right
        /-
          case neg.h
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          w₂ : Ne ((List.cons d L).getLast ⋯) 0
          h' : Not (Eq L List.nil)
          ⊢ Ne (Nat.ofDigits b L) 0
        -/
        contrapose! w₂
        /-
          case neg.h
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          h' : Not (Eq L List.nil)
          w₂ : Eq (Nat.ofDigits b L) 0
          ⊢ Eq ((List.cons d L).getLast ⋯) 0
        -/
        refine digits_zero_of_eq_zero h.ne_bot w₂ _ ?_
        /-
          case neg.h
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          h' : Not (Eq L List.nil)
          w₂ : Eq (Nat.ofDigits b L) 0
          ⊢ Membership.mem L ((List.cons d L).getLast ⋯)
        -/
        rw [List.getLast_cons h']
        /-
          case neg.h
          b : Nat
          h : LT.lt 1 b
          d : Nat
          L : List Nat
          ih : (∀ (l : Nat), Membership.mem L l → LT.lt l b) → (∀ (h : Ne L List.nil), N …
          w₁ : ∀ (l : Nat), Membership.mem (List.cons d L) l → LT.lt l b
          h' : Not (Eq L List.nil)
          w₂ : Eq (Nat.ofDigits b L) 0
          ⊢ Membership.mem L (L.getLast h')
        -/
        exact List.getLast_mem h'
        /-
          🎉 no goals
        -/


theorem ofDigits_digits (b n : ℕ) : ofDigits b (digits b n) = n := by
  /-
    b n : Nat
    ⊢ Eq (Nat.ofDigits b (b.digits n)) n
  -/
  cases' b with b
    /-
      case zero
      n : Nat
      ⊢ Eq (Nat.ofDigits 0 (Nat.digits 0 n)) n
    -/
  · cases' n with n
      /-
        case zero.zero
        ⊢ Eq (Nat.ofDigits 0 (Nat.digits 0 0)) 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case zero.succ
        n : Nat
        ⊢ Eq (Nat.ofDigits 0 (Nat.digits 0 (HAdd.hAdd n 1))) (HAdd.hAdd n 1)
      -/
    · change ofDigits 0 [n + 1] = n + 1
      /-
        case zero.succ
        n : Nat
        ⊢ Eq (Nat.ofDigits 0 (List.cons (HAdd.hAdd n 1) List.nil)) (HAdd.hAdd n 1)
      -/
      dsimp [ofDigits]
      /-
        🎉 no goals
      -/
    /-
      case succ
      n b : Nat
      ⊢ Eq (Nat.ofDigits (HAdd.hAdd b 1) ((HAdd.hAdd b 1).digits n)) n
    -/
  · cases' b with b
      /-
        case succ.zero
        n : Nat
        ⊢ Eq (Nat.ofDigits (HAdd.hAdd 0 1) ((HAdd.hAdd 0 1).digits n)) n
      -/
    · induction' n with n ih
        /-
          case succ.zero.zero
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd 0 1) ((HAdd.hAdd 0 1).digits 0)) 0
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case succ.zero.succ
          n : Nat
          ih : Eq (Nat.ofDigits (HAdd.hAdd 0 1) ((HAdd.hAdd 0 1).digits n)) n
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd 0 1) ((HAdd.hAdd 0 1).digits (HAdd.hAdd n 1))) ( …
        -/
      · rw [Nat.zero_add] at ih ⊢
        /-
          case succ.zero.succ
          n : Nat
          ih : Eq (Nat.ofDigits 1 (Nat.digits 1 n)) n
          ⊢ Eq (Nat.ofDigits 1 (Nat.digits 1 (HAdd.hAdd n 1))) (HAdd.hAdd n 1)
        -/
        simp only [ih, add_comm 1, ofDigits_one_cons, Nat.cast_id, digits_one_succ]
        /-
          🎉 no goals
        -/
      /-
        case succ.succ
        n b : Nat
        ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (HAdd.hAdd b 1) 1 …
      -/
    · induction n using Nat.strongRecOn with | ind n h => ?_
      /-
        case succ.succ.ind
        b n : Nat
        h : ∀ (m : Nat), LT.lt m n → Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) (( …
        ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (HAdd.hAdd b 1) 1 …
      -/
      cases n
        /-
          case succ.succ.ind.zero
          b : Nat
          h : ∀ (m : Nat), LT.lt m 0 → Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) (( …
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (HAdd.hAdd b 1) 1 …
        -/
      · rw [digits_zero]
        /-
          case succ.succ.ind.zero
          b : Nat
          h : ∀ (m : Nat), LT.lt m 0 → Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) (( …
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) List.nil) 0
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case succ.succ.ind.succ
          b n✝ : Nat
          h : ∀ (m : Nat), LT.lt m (HAdd.hAdd n✝ 1) → Eq (Nat.ofDigits (HAdd.hAdd (HAdd. …
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (HAdd.hAdd b 1) 1 …
        -/
      · simp only [Nat.succ_eq_add_one, digits_add_two_add_one]
        /-
          case succ.succ.ind.succ
          b n✝ : Nat
          h : ∀ (m : Nat), LT.lt m (HAdd.hAdd n✝ 1) → Eq (Nat.ofDigits (HAdd.hAdd (HAdd. …
          ⊢ Eq (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) (List.cons (HMod.hMod (HAdd.h …
        -/
        dsimp [ofDigits]
        /-
          case succ.succ.ind.succ
          b n✝ : Nat
          h : ∀ (m : Nat), LT.lt m (HAdd.hAdd n✝ 1) → Eq (Nat.ofDigits (HAdd.hAdd (HAdd. …
          ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd n✝ 1) (HAdd.hAdd b 2)) (HMul.hMul (HAdd. …
        -/
        rw [h _ (Nat.div_lt_self' _ b)]
        /-
          case succ.succ.ind.succ
          b n✝ : Nat
          h : ∀ (m : Nat), LT.lt m (HAdd.hAdd n✝ 1) → Eq (Nat.ofDigits (HAdd.hAdd (HAdd. …
          ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd n✝ 1) (HAdd.hAdd b 2)) (HMul.hMul (HAdd. …
        -/
        rw [Nat.mod_add_div]
        /-
          🎉 no goals
        -/


theorem ofDigits_one (L : List ℕ) : ofDigits 1 L = L.sum := by
  induction L with
  | nil => rfl
  | cons _ _ ih => simp [ofDigits, List.sum_cons, ih]


theorem digits_eq_nil_iff_eq_zero {b n : ℕ} : digits b n = [] ↔ n = 0 := by
  /-
    b n : Nat
    ⊢ Iff (Eq (b.digits n) List.nil) (Eq n 0)
  -/
  constructor
    /-
      case mp
      b n : Nat
      ⊢ Eq (b.digits n) List.nil → Eq n 0
    -/
  · intro h
    /-
      case mp
      b n : Nat
      h : Eq (b.digits n) List.nil
      ⊢ Eq n 0
    -/
    have : ofDigits b (digits b n) = ofDigits b [] := by rw [h]
    /-
      case mp
      b n : Nat
      h : Eq (b.digits n) List.nil
      this : Eq (Nat.ofDigits b (b.digits n)) (Nat.ofDigits b List.nil)
      ⊢ Eq n 0
    -/
    convert this
    /-
      case h.e'_2
      b n : Nat
      h : Eq (b.digits n) List.nil
      this : Eq (Nat.ofDigits b (b.digits n)) (Nat.ofDigits b List.nil)
      ⊢ Eq n (Nat.ofDigits b (b.digits n))
    -/
    rw [ofDigits_digits]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      b n : Nat
      ⊢ Eq n 0 → Eq (b.digits n) List.nil
    -/
  · rintro rfl
    /-
      case mpr
      b : Nat
      ⊢ Eq (b.digits 0) List.nil
    -/
    simp
    /-
      🎉 no goals
    -/


theorem digits_ne_nil_iff_ne_zero {b n : ℕ} : digits b n ≠ [] ↔ n ≠ 0 :=
  not_congr digits_eq_nil_iff_eq_zero


theorem digits_eq_cons_digits_div {b n : ℕ} (h : 1 < b) (w : n ≠ 0) :
    digits b n = (n % b) :: digits b (n / b) := by
  /-
    b n : Nat
    h : LT.lt 1 b
    w : Ne n 0
    ⊢ Eq (b.digits n) (List.cons (HMod.hMod n b) (b.digits (HDiv.hDiv n b)))
  -/
  rcases b with (_ | _ | b)
    /-
      case zero
      n : Nat
      w : Ne n 0
      h : LT.lt 1 0
      ⊢ Eq (Nat.digits 0 n) (List.cons (HMod.hMod n 0) (Nat.digits 0 (HDiv.hDiv n 0)))
    -/
  · rw [digits_zero_succ' w, Nat.mod_zero, Nat.div_zero, Nat.digits_zero_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      n : Nat
      w : Ne n 0
      h : LT.lt 1 (HAdd.hAdd 0 1)
      ⊢ Eq ((HAdd.hAdd 0 1).digits n) (List.cons (HMod.hMod n (HAdd.hAdd 0 1)) ((HAd …
    -/
  · norm_num at h
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    n : Nat
    w : Ne n 0
    b : Nat
    h : LT.lt 1 (HAdd.hAdd (HAdd.hAdd b 1) 1)
    ⊢ Eq ((HAdd.hAdd (HAdd.hAdd b 1) 1).digits n) (List.cons (HMod.hMod n (HAdd.hA …
  -/
  rcases n with (_ | n)
    /-
      case succ.succ.zero
      b : Nat
      h : LT.lt 1 (HAdd.hAdd (HAdd.hAdd b 1) 1)
      w : Ne 0 0
      ⊢ Eq ((HAdd.hAdd (HAdd.hAdd b 1) 1).digits 0) (List.cons (HMod.hMod 0 (HAdd.hA …
    -/
  · norm_num at w
    /-
      🎉 no goals
    -/
    /-
      case succ.succ.succ
      b : Nat
      h : LT.lt 1 (HAdd.hAdd (HAdd.hAdd b 1) 1)
      n : Nat
      w : Ne (HAdd.hAdd n 1) 0
      ⊢ Eq ((HAdd.hAdd (HAdd.hAdd b 1) 1).digits (HAdd.hAdd n 1)) (List.cons (HMod.h …
    -/
  · simp only [digits_add_two_add_one, ne_eq]
    /-
      🎉 no goals
    -/


theorem digits_getLast {b : ℕ} (m : ℕ) (h : 1 < b) (p q) :
    (digits b m).getLast p = (digits b (m / b)).getLast q := by
  /-
    b m : Nat
    h : LT.lt 1 b
    p : Ne (b.digits m) List.nil
    q : Ne (b.digits (HDiv.hDiv m b)) List.nil
    ⊢ Eq ((b.digits m).getLast p) ((b.digits (HDiv.hDiv m b)).getLast q)
  -/
  by_cases hm : m = 0
    /-
      case pos
      b m : Nat
      h : LT.lt 1 b
      p : Ne (b.digits m) List.nil
      q : Ne (b.digits (HDiv.hDiv m b)) List.nil
      hm : Eq m 0
      ⊢ Eq ((b.digits m).getLast p) ((b.digits (HDiv.hDiv m b)).getLast q)
    -/
  · simp [hm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    b m : Nat
    h : LT.lt 1 b
    p : Ne (b.digits m) List.nil
    q : Ne (b.digits (HDiv.hDiv m b)) List.nil
    hm : Not (Eq m 0)
    ⊢ Eq ((b.digits m).getLast p) ((b.digits (HDiv.hDiv m b)).getLast q)
  -/
  simp only [digits_eq_cons_digits_div h hm]
  /-
    case neg
    b m : Nat
    h : LT.lt 1 b
    p : Ne (b.digits m) List.nil
    q : Ne (b.digits (HDiv.hDiv m b)) List.nil
    hm : Not (Eq m 0)
    ⊢ Eq ((List.cons (HMod.hMod m b) (b.digits (HDiv.hDiv m b))).getLast ⋯) ((b.di …
  -/
  rw [List.getLast_cons]
  /-
    🎉 no goals
  -/


theorem digits.injective (b : ℕ) : Function.Injective b.digits :=
  Function.LeftInverse.injective (ofDigits_digits b)


@[simp]
theorem digits_inj_iff {b n m : ℕ} : b.digits n = b.digits m ↔ n = m :=
  (digits.injective b).eq_iff


theorem digits_len (b n : ℕ) (hb : 1 < b) (hn : n ≠ 0) : (b.digits n).length = b.log n + 1 := by
  /-
    b n : Nat
    hb : LT.lt 1 b
    hn : Ne n 0
    ⊢ Eq (b.digits n).length (HAdd.hAdd (Nat.log b n) 1)
  -/
  induction' n using Nat.strong_induction_on with n IH
  /-
    case h
    b : Nat
    hb : LT.lt 1 b
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
    hn : Ne n 0
    ⊢ Eq (b.digits n).length (HAdd.hAdd (Nat.log b n) 1)
  -/
  rw [digits_eq_cons_digits_div hb hn, List.length]
  /-
    case h
    b : Nat
    hb : LT.lt 1 b
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
    hn : Ne n 0
    ⊢ Eq (HAdd.hAdd (b.digits (HDiv.hDiv n b)).length 1) (HAdd.hAdd (Nat.log b n) 1)
  -/
  by_cases h : n / b = 0
    /-
      case pos
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Eq (HDiv.hDiv n b) 0
      ⊢ Eq (HAdd.hAdd (b.digits (HDiv.hDiv n b)).length 1) (HAdd.hAdd (Nat.log b n) 1)
    -/
  · simp [IH, h]
    /-
      case pos
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Eq (HDiv.hDiv n b) 0
      ⊢ Or (LT.lt n b) (LE.le b 1)
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case neg
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Not (Eq (HDiv.hDiv n b) 0)
      ⊢ Eq (HAdd.hAdd (b.digits (HDiv.hDiv n b)).length 1) (HAdd.hAdd (Nat.log b n) 1)
    -/
  · have : n / b < n := div_lt_self (Nat.pos_of_ne_zero hn) hb
    /-
      case neg
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Not (Eq (HDiv.hDiv n b) 0)
      this : LT.lt (HDiv.hDiv n b) n
      ⊢ Eq (HAdd.hAdd (b.digits (HDiv.hDiv n b)).length 1) (HAdd.hAdd (Nat.log b n) 1)
    -/
    rw [IH _ this h, log_div_base, tsub_add_cancel_of_le]
    /-
      case neg
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Not (Eq (HDiv.hDiv n b) 0)
      this : LT.lt (HDiv.hDiv n b) n
      ⊢ LE.le 1 (Nat.log b n)
    -/
    refine Nat.succ_le_of_lt (log_pos hb ?_)
    /-
      case neg
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      h : Not (Eq (HDiv.hDiv n b) 0)
      this : LT.lt (HDiv.hDiv n b) n
      ⊢ LE.le b n
    -/
    contrapose! h
    /-
      case neg
      b : Nat
      hb : LT.lt 1 b
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → Ne m 0 → Eq (b.digits m).length (HAdd.hAdd (Nat. …
      hn : Ne n 0
      this : LT.lt (HDiv.hDiv n b) n
      h : LT.lt n b
      ⊢ Eq (HDiv.hDiv n b) 0
    -/
    exact div_eq_of_lt h
    /-
      🎉 no goals
    -/


theorem getLast_digit_ne_zero (b : ℕ) {m : ℕ} (hm : m ≠ 0) :
    (digits b m).getLast (digits_ne_nil_iff_ne_zero.mpr hm) ≠ 0 := by
  /-
    b m : Nat
    hm : Ne m 0
    ⊢ Ne ((b.digits m).getLast ⋯) 0
  -/
  rcases b with (_ | _ | b)
    /-
      case zero
      m : Nat
      hm : Ne m 0
      ⊢ Ne ((Nat.digits 0 m).getLast ⋯) 0
    -/
  · cases m
      /-
        case zero.zero
        hm : Ne 0 0
        ⊢ Ne ((Nat.digits 0 0).getLast ⋯) 0
      -/
    · cases hm rfl
      /-
        🎉 no goals
      -/
      /-
        case zero.succ
        n✝ : Nat
        hm : Ne (HAdd.hAdd n✝ 1) 0
        ⊢ Ne ((Nat.digits 0 (HAdd.hAdd n✝ 1)).getLast ⋯) 0
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case succ.zero
      m : Nat
      hm : Ne m 0
      ⊢ Ne (((HAdd.hAdd 0 1).digits m).getLast ⋯) 0
    -/
  · cases m
      /-
        case succ.zero.zero
        hm : Ne 0 0
        ⊢ Ne (((HAdd.hAdd 0 1).digits 0).getLast ⋯) 0
      -/
    · cases hm rfl
      /-
        🎉 no goals
      -/
    /-
      case succ.zero.succ
      n✝ : Nat
      hm : Ne (HAdd.hAdd n✝ 1) 0
      ⊢ Ne (((HAdd.hAdd 0 1).digits (HAdd.hAdd n✝ 1)).getLast ⋯) 0
    -/
    rename ℕ => m
    /-
      case succ.zero.succ
      m : Nat
      hm : Ne (HAdd.hAdd m 1) 0
      ⊢ Ne (((HAdd.hAdd 0 1).digits (HAdd.hAdd m 1)).getLast ⋯) 0
    -/
    simp only [zero_add, digits_one, List.getLast_replicate_succ m 1]
    /-
      case succ.zero.succ
      m : Nat
      hm : Ne (HAdd.hAdd m 1) 0
      ⊢ Ne 1 0
    -/
    exact Nat.one_ne_zero
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    m : Nat
    hm : Ne m 0
    b : Nat
    ⊢ Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits m).getLast ⋯) 0
  -/
  revert hm
  /-
    case succ.succ
    m b : Nat
    ⊢ ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits m).getLast ⋯) 0
  -/
  induction m using Nat.strongRecOn with | ind n IH => ?_
  /-
    case succ.succ.ind
    b n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
    ⊢ ∀ (hm : Ne n 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits n).getLast ⋯) 0
  -/
  intro hn
  /-
    case succ.succ.ind
    b n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
    hn : Ne n 0
    ⊢ Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits n).getLast ⋯) 0
  -/
  by_cases hnb : n < b + 2
    /-
      case pos
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
      hn : Ne n 0
      hnb : LT.lt n (HAdd.hAdd b 2)
      ⊢ Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits n).getLast ⋯) 0
    -/
  · simpa only [digits_of_lt (b + 2) n hn hnb]
    /-
      🎉 no goals
    -/
    /-
      case neg
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
      hn : Ne n 0
      hnb : Not (LT.lt n (HAdd.hAdd b 2))
      ⊢ Ne (((HAdd.hAdd (HAdd.hAdd b 1) 1).digits n).getLast ⋯) 0
    -/
  · rw [digits_getLast n (le_add_left 2 b)]
    /-
      case neg
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
      hn : Ne n 0
      hnb : Not (LT.lt n (HAdd.hAdd b 2))
      ⊢ Ne (((HAdd.hAdd b 2).digits (HDiv.hDiv n (HAdd.hAdd b 2))).getLast ?neg.q✝) 0
    -/
    refine IH _ (Nat.div_lt_self hn.bot_lt (one_lt_succ_succ b)) ?_
    /-
      case neg
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
      hn : Ne n 0
      hnb : Not (LT.lt n (HAdd.hAdd b 2))
      ⊢ Ne (HDiv.hDiv n b.succ.succ) 0
    -/
    rw [← pos_iff_ne_zero]
    /-
      case neg
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m n → ∀ (hm : Ne m 0), Ne (((HAdd.hAdd (HAdd.hAdd b 1) …
      hn : Ne n 0
      hnb : Not (LT.lt n (HAdd.hAdd b 2))
      ⊢ LT.lt 0 (HDiv.hDiv n b.succ.succ)
    -/
    exact Nat.div_pos (le_of_not_lt hnb) (zero_lt_succ (succ b))
    /-
      🎉 no goals
    -/


theorem mul_ofDigits (n : ℕ) {b : ℕ} {l : List ℕ} :
    n * ofDigits b l = ofDigits b (l.map (n * ·)) := by
  induction l with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.map_cons, ofDigits_cons, ofDigits_cons, ← ih]
    ring


/-- The addition of ofDigits of two lists is equal to ofDigits of digit-wise addition of them -/
theorem ofDigits_add_ofDigits_eq_ofDigits_zipWith_of_length_eq {b : ℕ} {l1 l2 : List ℕ}
    (h : l1.length = l2.length) :
    ofDigits b l1 + ofDigits b l2 = ofDigits b (l1.zipWith (· + ·) l2) := by
  induction l1 generalizing l2 with
  | nil => simp_all [eq_comm, List.length_eq_zero, ofDigits]
  | cons hd₁ tl₁ ih₁ =>
    induction l2 generalizing tl₁ with
    | nil => simp_all
    | cons hd₂ tl₂ ih₂ =>
      simp_all only [List.length_cons, succ_eq_add_one, ofDigits_cons, add_left_inj,
        eq_comm, List.zipWith_cons_cons, add_eq]
      rw [← ih₁ h.symm, mul_add]
      ac_rfl


/-- The digits in the base b+2 expansion of n are all less than b+2 -/
theorem digits_lt_base' {b m : ℕ} : ∀ {d}, d ∈ digits (b + 2) m → d < b + 2 := by
  /-
    b m : Nat
    ⊢ ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).digits m) d → LT.lt d (HAdd.hAd …
  -/
  induction m using Nat.strongRecOn with | ind n IH => ?_
  /-
    case ind
    b n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).dig …
    ⊢ ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).digits n) d → LT.lt d (HAdd.hAd …
  -/
  intro d hd
  /-
    case ind
    b n : Nat
    IH : ∀ (m : Nat), LT.lt m n → ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).dig …
    d : Nat
    hd : Membership.mem ((HAdd.hAdd b 2).digits n) d
    ⊢ LT.lt d (HAdd.hAdd b 2)
  -/
  cases' n with n
    /-
      case ind.zero
      b d : Nat
      IH : ∀ (m : Nat), LT.lt m 0 → ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).dig …
      hd : Membership.mem ((HAdd.hAdd b 2).digits 0) d
      ⊢ LT.lt d (HAdd.hAdd b 2)
    -/
  · rw [digits_zero] at hd
    /-
      case ind.zero
      b d : Nat
      IH : ∀ (m : Nat), LT.lt m 0 → ∀ {d : Nat}, Membership.mem ((HAdd.hAdd b 2).dig …
      hd : Membership.mem List.nil d
      ⊢ LT.lt d (HAdd.hAdd b 2)
    -/
    cases hd
    /-
      🎉 no goals
    -/
  -- base b+2 expansion of 0 has no digits
  /-
    case ind.succ
    b d n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
    hd : Membership.mem ((HAdd.hAdd b 2).digits (HAdd.hAdd n 1)) d
    ⊢ LT.lt d (HAdd.hAdd b 2)
  -/
  rw [digits_add_two_add_one] at hd
  /-
    case ind.succ
    b d n : Nat
    IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
    hd : Membership.mem (List.cons (HMod.hMod (HAdd.hAdd n 1) (HAdd.hAdd b 2)) ((H …
    ⊢ LT.lt d (HAdd.hAdd b 2)
  -/
  cases hd
    /-
      case ind.succ.head
      b n : Nat
      IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
      ⊢ LT.lt (HMod.hMod (HAdd.hAdd n 1) (HAdd.hAdd b 2)) (HAdd.hAdd b 2)
    -/
  · exact n.succ.mod_lt (by simp)
    /-
      🎉 no goals
    -/
  -- Porting note: Previous code (single line) contained linarith.
  -- . exact IH _ (Nat.div_lt_self (Nat.succ_pos _) (by linarith)) hd
    /-
      case ind.succ.tail
      b d n : Nat
      IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
      a✝ : List.Mem d ((HAdd.hAdd b 2).digits (HDiv.hDiv (HAdd.hAdd n 1) (HAdd.hAdd  …
      ⊢ LT.lt d (HAdd.hAdd b 2)
    -/
  · apply IH ((n + 1) / (b + 2))
      /-
        case ind.succ.tail.a
        b d n : Nat
        IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
        a✝ : List.Mem d ((HAdd.hAdd b 2).digits (HDiv.hDiv (HAdd.hAdd n 1) (HAdd.hAdd  …
        ⊢ LT.lt (HDiv.hDiv (HAdd.hAdd n 1) (HAdd.hAdd b 2)) (HAdd.hAdd n 1)
      -/
                                /-
                                  🎉 no goals
                                -/
    · apply Nat.div_lt_self <;> omega
                                /-
                                  🎉 no goals
                                -/
      /-
        case ind.succ.tail.a
        b d n : Nat
        IH : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {d : Nat}, Membership.mem ((HAdd …
        a✝ : List.Mem d ((HAdd.hAdd b 2).digits (HDiv.hDiv (HAdd.hAdd n 1) (HAdd.hAdd  …
        ⊢ Membership.mem ((HAdd.hAdd b 2).digits (HDiv.hDiv (HAdd.hAdd n 1) (HAdd.hAdd …
      -/
    · assumption
      /-
        🎉 no goals
      -/


/-- The digits in the base b expansion of n are all less than b, if b ≥ 2 -/
theorem digits_lt_base {b m d : ℕ} (hb : 1 < b) (hd : d ∈ digits b m) : d < b := by
  /-
    b m d : Nat
    hb : LT.lt 1 b
    hd : Membership.mem (b.digits m) d
    ⊢ LT.lt d b
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases b with (_ | _ | b) <;> try simp_all
  /-
    case succ.succ
    m d b : Nat
    hd : Membership.mem ((HAdd.hAdd (HAdd.hAdd b 1) 1).digits m) d
    ⊢ LT.lt d (HAdd.hAdd (HAdd.hAdd b 1) 1)
  -/
  exact digits_lt_base' hd
  /-
    🎉 no goals
  -/


/-- an n-digit number in base b + 2 is less than (b + 2)^n -/
theorem ofDigits_lt_base_pow_length' {b : ℕ} {l : List ℕ} (hl : ∀ x ∈ l, x < b + 2) :
    ofDigits (b + 2) l < (b + 2) ^ l.length := by
  /-
    b : Nat
    l : List Nat
    hl : ∀ (x : Nat), Membership.mem l x → LT.lt x (HAdd.hAdd b 2)
    ⊢ LT.lt (Nat.ofDigits (HAdd.hAdd b 2) l) (HPow.hPow (HAdd.hAdd b 2) l.length)
  -/
  induction' l with hd tl IH
    /-
      case nil
      b : Nat
      hl : ∀ (x : Nat), Membership.mem List.nil x → LT.lt x (HAdd.hAdd b 2)
      ⊢ LT.lt (Nat.ofDigits (HAdd.hAdd b 2) List.nil) (HPow.hPow (HAdd.hAdd b 2) Lis …
    -/
  · simp [ofDigits]
    /-
      🎉 no goals
    -/
    /-
      case cons
      b hd : Nat
      tl : List Nat
      IH : (∀ (x : Nat), Membership.mem tl x → LT.lt x (HAdd.hAdd b 2)) → LT.lt (Nat …
      hl : ∀ (x : Nat), Membership.mem (List.cons hd tl) x → LT.lt x (HAdd.hAdd b 2)
      ⊢ LT.lt (Nat.ofDigits (HAdd.hAdd b 2) (List.cons hd tl)) (HPow.hPow (HAdd.hAdd …
    -/
  · rw [ofDigits, List.length_cons, pow_succ]
    have : (ofDigits (b + 2) tl + 1) * (b + 2) ≤ (b + 2) ^ tl.length * (b + 2) :=
      mul_le_mul (IH fun x hx => hl _ (List.mem_cons_of_mem _ hx)) (by rfl) (by simp only [zero_le])
        (Nat.zero_le _)
    /-
      case cons
      b hd : Nat
      tl : List Nat
      IH : (∀ (x : Nat), Membership.mem tl x → LT.lt x (HAdd.hAdd b 2)) → LT.lt (Nat …
      hl : ∀ (x : Nat), Membership.mem (List.cons hd tl) x → LT.lt x (HAdd.hAdd b 2)
      this : LE.le (HMul.hMul (HAdd.hAdd (Nat.ofDigits (HAdd.hAdd b 2) tl) 1) (HAdd. …
      ⊢ LT.lt (HAdd.hAdd (↑hd) (HMul.hMul (HAdd.hAdd b 2) (Nat.ofDigits (HAdd.hAdd b …
    -/
    suffices ↑hd < b + 2 by linarith
    /-
      case cons
      b hd : Nat
      tl : List Nat
      IH : (∀ (x : Nat), Membership.mem tl x → LT.lt x (HAdd.hAdd b 2)) → LT.lt (Nat …
      hl : ∀ (x : Nat), Membership.mem (List.cons hd tl) x → LT.lt x (HAdd.hAdd b 2)
      this : LE.le (HMul.hMul (HAdd.hAdd (Nat.ofDigits (HAdd.hAdd b 2) tl) 1) (HAdd. …
      ⊢ LT.lt hd (HAdd.hAdd b 2)
    -/
    exact hl hd (List.mem_cons_self _ _)
    /-
      🎉 no goals
    -/


/-- an n-digit number in base b is less than b^n if b > 1 -/
theorem ofDigits_lt_base_pow_length {b : ℕ} {l : List ℕ} (hb : 1 < b) (hl : ∀ x ∈ l, x < b) :
    ofDigits b l < b ^ l.length := by
  /-
    b : Nat
    l : List Nat
    hb : LT.lt 1 b
    hl : ∀ (x : Nat), Membership.mem l x → LT.lt x b
    ⊢ LT.lt (Nat.ofDigits b l) (HPow.hPow b l.length)
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases b with (_ | _ | b) <;> try simp_all
  /-
    case succ.succ
    l : List Nat
    b : Nat
    hl : ∀ (x : Nat), Membership.mem l x → LT.lt x (HAdd.hAdd (HAdd.hAdd b 1) 1)
    ⊢ LT.lt (Nat.ofDigits (HAdd.hAdd (HAdd.hAdd b 1) 1) l) (HPow.hPow (HAdd.hAdd ( …
  -/
  exact ofDigits_lt_base_pow_length' hl
  /-
    🎉 no goals
  -/


/-- Any number m is less than (b+2)^(number of digits in the base b + 2 representation of m) -/
theorem lt_base_pow_length_digits' {b m : ℕ} : m < (b + 2) ^ (digits (b + 2) m).length := by
  /-
    b m : Nat
    ⊢ LT.lt m (HPow.hPow (HAdd.hAdd b 2) ((HAdd.hAdd b 2).digits m).length)
  -/
  convert @ofDigits_lt_base_pow_length' b (digits (b + 2) m) fun _ => digits_lt_base'
  /-
    case h.e'_3
    b m : Nat
    ⊢ Eq m (Nat.ofDigits (HAdd.hAdd b 2) ((HAdd.hAdd b 2).digits m))
  -/
  rw [ofDigits_digits (b + 2) m]
  /-
    🎉 no goals
  -/


/-- Any number m is less than b^(number of digits in the base b representation of m) -/
theorem lt_base_pow_length_digits {b m : ℕ} (hb : 1 < b) : m < b ^ (digits b m).length := by
  /-
    b m : Nat
    hb : LT.lt 1 b
    ⊢ LT.lt m (HPow.hPow b (b.digits m).length)
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases b with (_ | _ | b) <;> try simp_all
  /-
    case succ.succ
    m b : Nat
    ⊢ LT.lt m (HPow.hPow (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (HAdd.hAdd b 1) …
  -/
  exact lt_base_pow_length_digits'
  /-
    🎉 no goals
  -/


theorem digits_base_pow_mul {b k m : ℕ} (hb : 1 < b) (hm : 0 < m) :
    digits b (b ^ k * m) = List.replicate k 0 ++ digits b m := by
  induction k generalizing m with
  | zero => simp
  | succ k ih =>
    have hmb : 0 < m * b := lt_mul_of_lt_of_one_lt' hm hb
    let h1 := digits_def' hb hmb
    have h2 : m = m * b / b :=
      Nat.eq_div_of_mul_eq_left (not_eq_zero_of_lt hb) rfl
    simp only [mul_mod_left, ← h2] at h1
    rw [List.replicate_succ', List.append_assoc, List.singleton_append, ← h1, ← ih hmb]
    ring_nf


theorem ofDigits_digits_append_digits {b m n : ℕ} :
    ofDigits b (digits b n ++ digits b m) = n + b ^ (digits b n).length * m := by
  /-
    b m n : Nat
    ⊢ Eq (Nat.ofDigits b (HAppend.hAppend (b.digits n) (b.digits m))) (HAdd.hAdd n …
  -/
  rw [ofDigits_append, ofDigits_digits, ofDigits_digits]
  /-
    🎉 no goals
  -/


theorem digits_append_digits {b m n : ℕ} (hb : 0 < b) :
    digits b n ++ digits b m = digits b (n + b ^ (digits b n).length * m) := by
  /-
    b m n : Nat
    hb : LT.lt 0 b
    ⊢ Eq (HAppend.hAppend (b.digits n) (b.digits m)) (b.digits (HAdd.hAdd n (HMul. …
  -/
  rcases eq_or_lt_of_le (Nat.succ_le_of_lt hb) with (rfl | hb)
    /-
      case inl
      m n : Nat
      hb : LT.lt 0 (Nat.succ 0)
      ⊢ Eq (HAppend.hAppend ((Nat.succ 0).digits n) ((Nat.succ 0).digits m)) ((Nat.s …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    b m n : Nat
    hb✝ : LT.lt 0 b
    hb : LT.lt (Nat.succ 0) b
    ⊢ Eq (HAppend.hAppend (b.digits n) (b.digits m)) (b.digits (HAdd.hAdd n (HMul. …
  -/
  rw [← ofDigits_digits_append_digits]
  /-
    case inr
    b m n : Nat
    hb✝ : LT.lt 0 b
    hb : LT.lt (Nat.succ 0) b
    ⊢ Eq (HAppend.hAppend (b.digits n) (b.digits m)) (b.digits (Nat.ofDigits b (HA …
  -/
  refine (digits_ofDigits b hb _ (fun l hl => ?_) (fun h_append => ?_)).symm
    /-
      case inr.refine_1
      b m n : Nat
      hb✝ : LT.lt 0 b
      hb : LT.lt (Nat.succ 0) b
      l : Nat
      hl : Membership.mem (HAppend.hAppend (b.digits n) (b.digits m)) l
      ⊢ LT.lt l b
    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  · rcases (List.mem_append.mp hl) with (h | h) <;> exact digits_lt_base hb h
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case inr.refine_2
      b m n : Nat
      hb✝ : LT.lt 0 b
      hb : LT.lt (Nat.succ 0) b
      h_append : Ne (HAppend.hAppend (b.digits n) (b.digits m)) List.nil
      ⊢ Ne ((HAppend.hAppend (b.digits n) (b.digits m)).getLast h_append) 0
    -/
  · by_cases h : digits b m = []
      /-
        case pos
        b m n : Nat
        hb✝ : LT.lt 0 b
        hb : LT.lt (Nat.succ 0) b
        h_append : Ne (HAppend.hAppend (b.digits n) (b.digits m)) List.nil
        h : Eq (b.digits m) List.nil
        ⊢ Ne ((HAppend.hAppend (b.digits n) (b.digits m)).getLast h_append) 0
      -/
    · simp only [h, List.append_nil] at h_append ⊢
      /-
        case pos
        b m n : Nat
        hb✝ : LT.lt 0 b
        hb : LT.lt (Nat.succ 0) b
        h_append✝ : Ne (HAppend.hAppend (b.digits n) (b.digits m)) List.nil
        h : Eq (b.digits m) List.nil
        h_append : Ne (b.digits n) List.nil
        ⊢ Ne ((b.digits n).getLast ⋯) 0
      -/
      exact getLast_digit_ne_zero b <| digits_ne_nil_iff_ne_zero.mp h_append
      /-
        🎉 no goals
      -/
    · exact (List.getLast_append' _ _ h) ▸
          (getLast_digit_ne_zero _ <| digits_ne_nil_iff_ne_zero.mp h)


theorem digits_append_zeroes_append_digits {b k m n : ℕ} (hb : 1 < b) (hm : 0 < m) :
    digits b n ++ List.replicate k 0 ++ digits b m =
    digits b (n + b ^ ((digits b n).length + k) * m) := by
  /-
    b k m n : Nat
    hb : LT.lt 1 b
    hm : LT.lt 0 m
    ⊢ Eq (HAppend.hAppend (HAppend.hAppend (b.digits n) (List.replicate k 0)) (b.d …
  -/
  rw [List.append_assoc, ← digits_base_pow_mul hb hm]
  /-
    b k m n : Nat
    hb : LT.lt 1 b
    hm : LT.lt 0 m
    ⊢ Eq (HAppend.hAppend (b.digits n) (b.digits (HMul.hMul (HPow.hPow b k) m))) ( …
  -/
  simp only [digits_append_digits (zero_lt_of_lt hb), digits_inj_iff, add_right_inj]
  /-
    b k m n : Nat
    hb : LT.lt 1 b
    hm : LT.lt 0 m
    ⊢ Eq (HMul.hMul (HPow.hPow b (b.digits n).length) (HMul.hMul (HPow.hPow b k) m …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem digits_len_le_digits_len_succ (b n : ℕ) :
    (digits b n).length ≤ (digits b (n + 1)).length := by
  /-
    b n : Nat
    ⊢ LE.le (b.digits n).length (b.digits (HAdd.hAdd n 1)).length
  -/
  rcases Decidable.eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      b : Nat
      ⊢ LE.le (b.digits 0).length (b.digits (HAdd.hAdd 0 1)).length
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    b n : Nat
    hn : Ne n 0
    ⊢ LE.le (b.digits n).length (b.digits (HAdd.hAdd n 1)).length
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inr.inl
      b n : Nat
      hn : Ne n 0
      hb : LE.le b 1
      ⊢ LE.le (b.digits n).length (b.digits (HAdd.hAdd n 1)).length
    -/
                         /-
                           🎉 no goals
                         -/
  · interval_cases b <;> simp_arith [digits_zero_succ', hn]
                         /-
                           🎉 no goals
                         -/
  /-
    case inr.inr
    b n : Nat
    hn : Ne n 0
    hb : LT.lt 1 b
    ⊢ LE.le (b.digits n).length (b.digits (HAdd.hAdd n 1)).length
  -/
  simpa [digits_len, hb, hn] using log_mono_right (le_succ _)
  /-
    🎉 no goals
  -/


theorem le_digits_len_le (b n m : ℕ) (h : n ≤ m) : (digits b n).length ≤ (digits b m).length :=
  monotone_nat_of_le_succ (digits_len_le_digits_len_succ b) h


@[mono]
theorem ofDigits_monotone {p q : ℕ} (L : List ℕ) (h : p ≤ q) : ofDigits p L ≤ ofDigits q L := by
  induction L with
  | nil => rfl
  | cons _ _ hi =>
    simp only [ofDigits, cast_id, add_le_add_iff_left]
    exact Nat.mul_le_mul h hi


theorem sum_le_ofDigits {p : ℕ} (L : List ℕ) (h : 1 ≤ p) : L.sum ≤ ofDigits p L :=
  (ofDigits_one L).symm ▸ ofDigits_monotone L h


theorem digit_sum_le (p n : ℕ) : List.sum (digits p n) ≤ n := by
  /-
    p n : Nat
    ⊢ LE.le (p.digits n).sum n
  -/
  induction' n with n
    /-
      case zero
      p : Nat
      ⊢ LE.le (p.digits 0).sum 0
    -/
  · exact digits_zero _ ▸ Nat.le_refl (List.sum [])
    /-
      🎉 no goals
    -/
    /-
      case succ
      p n : Nat
      a✝ : LE.le (p.digits n).sum n
      ⊢ LE.le (p.digits (HAdd.hAdd n 1)).sum (HAdd.hAdd n 1)
    -/
  · induction' p with p
      /-
        case succ.zero
        n : Nat
        a✝ : LE.le (Nat.digits 0 n).sum n
        ⊢ LE.le (Nat.digits 0 (HAdd.hAdd n 1)).sum (HAdd.hAdd n 1)
      -/
    · rw [digits_zero_succ, List.sum_cons, List.sum_nil, add_zero]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        n p : Nat
        a✝¹ : LE.le (p.digits n).sum n → LE.le (p.digits (HAdd.hAdd n 1)).sum (HAdd.hA …
        a✝ : LE.le ((HAdd.hAdd p 1).digits n).sum n
        ⊢ LE.le ((HAdd.hAdd p 1).digits (HAdd.hAdd n 1)).sum (HAdd.hAdd n 1)
      -/
    · nth_rw 2 [← ofDigits_digits p.succ (n + 1)]
      /-
        case succ.succ
        n p : Nat
        a✝¹ : LE.le (p.digits n).sum n → LE.le (p.digits (HAdd.hAdd n 1)).sum (HAdd.hA …
        a✝ : LE.le ((HAdd.hAdd p 1).digits n).sum n
        ⊢ LE.le ((HAdd.hAdd p 1).digits (HAdd.hAdd n 1)).sum (Nat.ofDigits p.succ (p.s …
      -/
      rw [← ofDigits_one <| digits p.succ n.succ]
      /-
        case succ.succ
        n p : Nat
        a✝¹ : LE.le (p.digits n).sum n → LE.le (p.digits (HAdd.hAdd n 1)).sum (HAdd.hA …
        a✝ : LE.le ((HAdd.hAdd p 1).digits n).sum n
        ⊢ LE.le (Nat.ofDigits 1 (p.succ.digits n.succ)) (Nat.ofDigits p.succ (p.succ.d …
      -/
      exact ofDigits_monotone (digits p.succ n.succ) <| Nat.succ_pos p
      /-
        🎉 no goals
      -/


theorem pow_length_le_mul_ofDigits {b : ℕ} {l : List ℕ} (hl : l ≠ []) (hl2 : l.getLast hl ≠ 0) :
    (b + 2) ^ l.length ≤ (b + 2) * ofDigits (b + 2) l := by
  /-
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    ⊢ LE.le (HPow.hPow (HAdd.hAdd b 2) l.length) (HMul.hMul (HAdd.hAdd b 2) (Nat.o …
  -/
  rw [← List.dropLast_append_getLast hl]
  simp only [List.length_append, List.length, zero_add, List.length_dropLast, ofDigits_append,
    List.length_dropLast, ofDigits_singleton, add_comm (l.length - 1), pow_add, pow_one]
  /-
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    ⊢ LE.le (HMul.hMul (HAdd.hAdd b 2) (HPow.hPow (HAdd.hAdd b 2) (HSub.hSub l.len …
  -/
  apply Nat.mul_le_mul_left
  /-
    case h
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    ⊢ LE.le (HPow.hPow (HAdd.hAdd b 2) (HSub.hSub l.length 1)) (HAdd.hAdd (Nat.ofD …
  -/
  refine le_trans ?_ (Nat.le_add_left _ _)
  /-
    case h
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    ⊢ LE.le (HPow.hPow (HAdd.hAdd b 2) (HSub.hSub l.length 1)) (HMul.hMul (HPow.hP …
  -/
  have : 0 < l.getLast hl := by rwa [pos_iff_ne_zero]
  /-
    case h
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    this : LT.lt 0 (l.getLast hl)
    ⊢ LE.le (HPow.hPow (HAdd.hAdd b 2) (HSub.hSub l.length 1)) (HMul.hMul (HPow.hP …
  -/
  convert Nat.mul_le_mul_left ((b + 2) ^ (l.length - 1)) this using 1
  /-
    case h.e'_3
    b : Nat
    l : List Nat
    hl : Ne l List.nil
    hl2 : Ne (l.getLast hl) 0
    this : LT.lt 0 (l.getLast hl)
    ⊢ Eq (HPow.hPow (HAdd.hAdd b 2) (HSub.hSub l.length 1)) (HMul.hMul (HPow.hPow  …
  -/
  rw [Nat.mul_one]
  /-
    🎉 no goals
  -/


/-- Any non-zero natural number `m` is greater than
(b+2)^((number of digits in the base (b+2) representation of m) - 1)
-/
theorem base_pow_length_digits_le' (b m : ℕ) (hm : m ≠ 0) :
    (b + 2) ^ (digits (b + 2) m).length ≤ (b + 2) * m := by
  /-
    b m : Nat
    hm : Ne m 0
    ⊢ LE.le (HPow.hPow (HAdd.hAdd b 2) ((HAdd.hAdd b 2).digits m).length) (HMul.hM …
  -/
  have : digits (b + 2) m ≠ [] := digits_ne_nil_iff_ne_zero.mpr hm
  convert @pow_length_le_mul_ofDigits b (digits (b+2) m)
    this (getLast_digit_ne_zero _ hm)
  /-
    case h.e'_4.h.e'_6
    b m : Nat
    hm : Ne m 0
    this : Ne ((HAdd.hAdd b 2).digits m) List.nil
    ⊢ Eq m (Nat.ofDigits (HAdd.hAdd b 2) ((HAdd.hAdd b 2).digits m))
  -/
  rw [ofDigits_digits]
  /-
    🎉 no goals
  -/


/-- Any non-zero natural number `m` is greater than
b^((number of digits in the base b representation of m) - 1)
-/
theorem base_pow_length_digits_le (b m : ℕ) (hb : 1 < b) :
    m ≠ 0 → b ^ (digits b m).length ≤ b * m := by
  /-
    b m : Nat
    hb : LT.lt 1 b
    ⊢ Ne m 0 → LE.le (HPow.hPow b (b.digits m).length) (HMul.hMul b m)
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases b with (_ | _ | b) <;> try simp_all
  /-
    case succ.succ
    m b : Nat
    ⊢ Not (Eq m 0) → LE.le (HPow.hPow (HAdd.hAdd (HAdd.hAdd b 1) 1) ((HAdd.hAdd (H …
  -/
  exact base_pow_length_digits_le' b m
  /-
    🎉 no goals
  -/


/-- Interpreting as a base `p` number and dividing by `p` is the same as interpreting the tail.
-/
lemma ofDigits_div_eq_ofDigits_tail {p : ℕ} (hpos : 0 < p) (digits : List ℕ)
    (w₁ : ∀ l ∈ digits, l < p) : ofDigits p digits / p = ofDigits p digits.tail := by
  /-
    p : Nat
    hpos : LT.lt 0 p
    digits : List Nat
    w₁ : ∀ (l : Nat), Membership.mem digits l → LT.lt l p
    ⊢ Eq (HDiv.hDiv (Nat.ofDigits p digits) p) (Nat.ofDigits p digits.tail)
  -/
  induction' digits with hd tl
    /-
      case nil
      p : Nat
      hpos : LT.lt 0 p
      w₁ : ∀ (l : Nat), Membership.mem List.nil l → LT.lt l p
      ⊢ Eq (HDiv.hDiv (Nat.ofDigits p List.nil) p) (Nat.ofDigits p List.nil.tail)
    -/
  · simp [ofDigits]
    /-
      🎉 no goals
    -/
    /-
      case cons
      p : Nat
      hpos : LT.lt 0 p
      hd : Nat
      tl : List Nat
      tail_ih✝ : (∀ (l : Nat), Membership.mem tl l → LT.lt l p) → Eq (HDiv.hDiv (Nat …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
      ⊢ Eq (HDiv.hDiv (Nat.ofDigits p (List.cons hd tl)) p) (Nat.ofDigits p (List.co …
    -/
  · refine Eq.trans (add_mul_div_left hd _ hpos) ?_
    /-
      case cons
      p : Nat
      hpos : LT.lt 0 p
      hd : Nat
      tl : List Nat
      tail_ih✝ : (∀ (l : Nat), Membership.mem tl l → LT.lt l p) → Eq (HDiv.hDiv (Nat …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv hd p) (Nat.ofDigits p tl)) (Nat.ofDigits p (List.co …
    -/
    rw [Nat.div_eq_of_lt <| w₁ _ <| List.mem_cons_self _ _, zero_add]
    /-
      case cons
      p : Nat
      hpos : LT.lt 0 p
      hd : Nat
      tl : List Nat
      tail_ih✝ : (∀ (l : Nat), Membership.mem tl l → LT.lt l p) → Eq (HDiv.hDiv (Nat …
      w₁ : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
      ⊢ Eq (Nat.ofDigits p tl) (Nat.ofDigits p (List.cons hd tl).tail)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Interpreting as a base `p` number and dividing by `p^i` is the same as dropping `i`.
-/
lemma ofDigits_div_pow_eq_ofDigits_drop
    {p : ℕ} (i : ℕ) (hpos : 0 < p) (digits : List ℕ) (w₁ : ∀ l ∈ digits, l < p) :
    ofDigits p digits / p ^ i = ofDigits p (digits.drop i) := by
  /-
    p i : Nat
    hpos : LT.lt 0 p
    digits : List Nat
    w₁ : ∀ (l : Nat), Membership.mem digits l → LT.lt l p
    ⊢ Eq (HDiv.hDiv (Nat.ofDigits p digits) (HPow.hPow p i)) (Nat.ofDigits p (List …
  -/
  induction' i with i hi
    /-
      case zero
      p : Nat
      hpos : LT.lt 0 p
      digits : List Nat
      w₁ : ∀ (l : Nat), Membership.mem digits l → LT.lt l p
      ⊢ Eq (HDiv.hDiv (Nat.ofDigits p digits) (HPow.hPow p 0)) (Nat.ofDigits p (List …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [Nat.pow_succ, ← Nat.div_div_eq_div_mul, hi, ofDigits_div_eq_ofDigits_tail hpos
      (List.drop i digits) fun x hx ↦ w₁ x <| List.mem_of_mem_drop hx, ← List.drop_one,
      List.drop_drop, add_comm]


/-- Dividing `n` by `p^i` is like truncating the first `i` digits of `n` in base `p`.
-/
lemma self_div_pow_eq_ofDigits_drop {p : ℕ} (i n : ℕ) (h : 2 ≤ p) :
    n / p ^ i = ofDigits p ((p.digits n).drop i) := by
  convert ofDigits_div_pow_eq_ofDigits_drop i (zero_lt_of_lt h) (p.digits n)
    (fun l hl ↦ digits_lt_base h hl)
  /-
    case h.e'_2.h.e'_5
    p i n : Nat
    h : LE.le 2 p
    ⊢ Eq n (Nat.ofDigits p (p.digits n))
  -/
  exact (ofDigits_digits p n).symm
  /-
    🎉 no goals
  -/


theorem sub_one_mul_sum_div_pow_eq_sub_sum_digits {p : ℕ}
    (L : List ℕ) {h_nonempty} (h_ne_zero : L.getLast h_nonempty ≠ 0) (h_lt : ∀ l ∈ L, l < p) :
    (p - 1) * ∑ i ∈ range L.length, (ofDigits p L) / p ^ i.succ = (ofDigits p L) - L.sum := by
  /-
    p : Nat
    L : List Nat
    h_nonempty : Ne L List.nil
    h_ne_zero : Ne (L.getLast h_nonempty) 0
    h_lt : ∀ (l : Nat), Membership.mem L l → LT.lt l p
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range L.length).sum fun i => HDiv.hDi …
  -/
  obtain h | rfl | h : 1 < p ∨ 1 = p ∨ p < 1 := trichotomous 1 p
    /-
      case inl
      p : Nat
      L : List Nat
      h_nonempty : Ne L List.nil
      h_ne_zero : Ne (L.getLast h_nonempty) 0
      h_lt : ∀ (l : Nat), Membership.mem L l → LT.lt l p
      h : LT.lt 1 p
      ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range L.length).sum fun i => HDiv.hDi …
    -/
  · induction' L with hd tl ih
      /-
        case inl.nil
        p : Nat
        h : LT.lt 1 p
        h_nonempty : Ne List.nil List.nil
        h_ne_zero : Ne (List.nil.getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem List.nil l → LT.lt l p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range List.nil.length).sum fun i => H …
      -/
    · simp [ofDigits]
      /-
        🎉 no goals
      -/
    · simp only [List.length_cons, List.sum_cons, self_div_pow_eq_ofDigits_drop _ _ h,
          digits_ofDigits p h (hd :: tl) h_lt (fun _ => h_ne_zero)]
      /-
        case inl.cons
        p : Nat
        h : LT.lt 1 p
        hd : Nat
        tl : List Nat
        ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
        h_nonempty : Ne (List.cons hd tl) List.nil
        h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (HAdd.hAdd tl.length 1)).sum fu …
      -/
      simp only [ofDigits]
      /-
        case inl.cons
        p : Nat
        h : LT.lt 1 p
        hd : Nat
        tl : List Nat
        ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
        h_nonempty : Ne (List.cons hd tl) List.nil
        h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (HAdd.hAdd tl.length 1)).sum fu …
      -/
      rw [sum_range_succ, Nat.cast_id]
      /-
        case inl.cons
        p : Nat
        h : LT.lt 1 p
        hd : Nat
        tl : List Nat
        ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
        h_nonempty : Ne (List.cons hd tl) List.nil
        h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
      -/
      simp only [List.drop, List.drop_length]
      /-
        case inl.cons
        p : Nat
        h : LT.lt 1 p
        hd : Nat
        tl : List Nat
        ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
        h_nonempty : Ne (List.cons hd tl) List.nil
        h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
      -/
      obtain rfl | h' := em <| tl = []
        /-
          case inl.cons.inl
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          ih : ∀ {h_nonempty : Ne List.nil List.nil}, Ne (List.nil.getLast h_nonempty) 0 …
          h_nonempty : Ne (List.cons hd List.nil) List.nil
          h_ne_zero : Ne ((List.cons hd List.nil).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd List.nil) l → LT.lt l p
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range List.nil.length).sum …
        -/
      · simp [ofDigits]
        /-
          🎉 no goals
        -/
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
      · have w₁' := fun l hl ↦ h_lt l <| List.mem_cons_of_mem hd hl
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
        have w₂' := fun (h : tl ≠ []) ↦ (List.getLast_cons h) ▸ h_ne_zero
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l : N …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
        have ih := ih (w₂' h') w₁'
        simp only [self_div_pow_eq_ofDigits_drop _ _ h, digits_ofDigits p h tl w₁' w₂',
          ← Nat.one_add] at ih
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih✝ : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l :  …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ih : Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range tl.length).sum fun x => Nat. …
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
        have := sum_singleton (fun x ↦ ofDigits p <| tl.drop x) tl.length
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih✝ : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l :  …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ih : Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range tl.length).sum fun x => Nat. …
          this : Eq ((Singleton.singleton tl.length).sum fun x => Nat.ofDigits p (List.d …
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
        rw [← Ico_succ_singleton, List.drop_length, ofDigits] at this
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih✝ : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l :  …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ih : Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range tl.length).sum fun x => Nat. …
          this : Eq ((Finset.Ico tl.length (HAdd.hAdd tl.length 1)).sum fun x => Nat.ofD …
          ⊢ Eq (HMul.hMul (HSub.hSub p 1) (HAdd.hAdd ((Finset.range tl.length).sum fun x …
        -/
        have h₁ : 1 ≤ tl.length := List.length_pos.mpr h'
        rw [← sum_range_add_sum_Ico _ <| h₁, ← add_zero (∑ x ∈ Ico _ _, ofDigits p (tl.drop x)),
            ← this, sum_Ico_consecutive _  h₁ <| (le_add_right tl.length 1),
            ← sum_Ico_add _ 0 tl.length 1,
            Ico_zero_eq_range, mul_add, mul_add, ih, range_one, sum_singleton, List.drop, ofDigits,
            mul_zero, add_zero, ← Nat.add_sub_assoc <| sum_le_ofDigits _ <| Nat.le_of_lt h]
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih✝ : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l :  …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ih : Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range tl.length).sum fun x => Nat. …
          this : Eq ((Finset.Ico tl.length (HAdd.hAdd tl.length 1)).sum fun x => Nat.ofD …
          h₁ : LE.le 1 tl.length
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HSub.hSub p 1) (Nat.ofDigits p tl)) (Na …
        -/
        nth_rw 2 [← one_mul <| ofDigits p tl]
        /-
          case inl.cons.inr
          p : Nat
          h : LT.lt 1 p
          hd : Nat
          tl : List Nat
          ih✝ : ∀ {h_nonempty : Ne tl List.nil}, Ne (tl.getLast h_nonempty) 0 → (∀ (l :  …
          h_nonempty : Ne (List.cons hd tl) List.nil
          h_ne_zero : Ne ((List.cons hd tl).getLast h_nonempty) 0
          h_lt : ∀ (l : Nat), Membership.mem (List.cons hd tl) l → LT.lt l p
          h' : Not (Eq tl List.nil)
          w₁' : ∀ (l : Nat), Membership.mem tl l → LT.lt l p
          w₂' : ∀ (h : Ne tl List.nil), Ne (tl.getLast h) 0
          ih : Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range tl.length).sum fun x => Nat. …
          this : Eq ((Finset.Ico tl.length (HAdd.hAdd tl.length 1)).sum fun x => Nat.ofD …
          h₁ : LE.le 1 tl.length
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul (HSub.hSub p 1) (Nat.ofDigits p tl)) (HM …
        -/
        rw [← add_mul, Nat.sub_add_cancel (one_le_of_lt h), Nat.add_sub_add_left]
        /-
          🎉 no goals
        -/
    /-
      case inr.inl
      L : List Nat
      h_nonempty : Ne L List.nil
      h_ne_zero : Ne (L.getLast h_nonempty) 0
      h_lt : ∀ (l : Nat), Membership.mem L l → LT.lt l 1
      ⊢ Eq (HMul.hMul (HSub.hSub 1 1) ((Finset.range L.length).sum fun i => HDiv.hDi …
    -/
  · simp [ofDigits_one]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : Nat
      L : List Nat
      h_nonempty : Ne L List.nil
      h_ne_zero : Ne (L.getLast h_nonempty) 0
      h_lt : ∀ (l : Nat), Membership.mem L l → LT.lt l p
      h : LT.lt p 1
      ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range L.length).sum fun i => HDiv.hDi …
    -/
  · simp [lt_one_iff.mp h]
    /-
      case inr.inr
      p : Nat
      L : List Nat
      h_nonempty : Ne L List.nil
      h_ne_zero : Ne (L.getLast h_nonempty) 0
      h_lt : ∀ (l : Nat), Membership.mem L l → LT.lt l p
      h : LT.lt p 1
      ⊢ Eq 0 (HSub.hSub (Nat.ofDigits 0 L) L.sum)
    -/
    cases L
      /-
        case inr.inr.nil
        p : Nat
        h : LT.lt p 1
        h_nonempty : Ne List.nil List.nil
        h_ne_zero : Ne (List.nil.getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem List.nil l → LT.lt l p
        ⊢ Eq 0 (HSub.hSub (Nat.ofDigits 0 List.nil) List.nil.sum)
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.cons
        p : Nat
        h : LT.lt p 1
        head✝ : Nat
        tail✝ : List Nat
        h_nonempty : Ne (List.cons head✝ tail✝) List.nil
        h_ne_zero : Ne ((List.cons head✝ tail✝).getLast h_nonempty) 0
        h_lt : ∀ (l : Nat), Membership.mem (List.cons head✝ tail✝) l → LT.lt l p
        ⊢ Eq 0 (HSub.hSub (Nat.ofDigits 0 (List.cons head✝ tail✝)) (List.cons head✝ ta …
      -/
    · simp [ofDigits]
      /-
        🎉 no goals
      -/


theorem sub_one_mul_sum_log_div_pow_eq_sub_sum_digits {p : ℕ} (n : ℕ) :
    (p - 1) * ∑ i ∈ range (log p n).succ, n / p ^ i.succ = n - (p.digits n).sum := by
  /-
    p n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (Nat.log p n).succ).sum fun i = …
  -/
  obtain h | rfl | h : 1 < p ∨ 1 = p ∨ p < 1 := trichotomous 1 p
    /-
      case inl
      p n : Nat
      h : LT.lt 1 p
      ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (Nat.log p n).succ).sum fun i = …
    -/
  · rcases eq_or_ne n 0 with rfl | hn
      /-
        case inl.inl
        p : Nat
        h : LT.lt 1 p
        ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (Nat.log p 0).succ).sum fun i = …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · convert sub_one_mul_sum_div_pow_eq_sub_sum_digits (p.digits n) (getLast_digit_ne_zero p hn) <|
          (fun l a ↦ digits_lt_base h a)
        /-
          case h.e'_2.h.e'_6.h.h.e'_1
          p n : Nat
          h : LT.lt 1 p
          hn : Ne n 0
          ⊢ Eq (Nat.log p n).succ (p.digits n).length
        -/
      · refine (digits_len p n h hn).symm
        /-
          🎉 no goals
        -/
      /-
        case h.e'_2.h.e'_6.a.h.e'_5
        p n : Nat
        h : LT.lt 1 p
        hn : Ne n 0
        x✝ : Nat
        a✝ : Membership.mem (Finset.range (p.digits n).length) x✝
        ⊢ Eq n (Nat.ofDigits p (p.digits n))
      -/
      all_goals exact (ofDigits_digits p n).symm
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      n : Nat
      ⊢ Eq (HMul.hMul (HSub.hSub 1 1) ((Finset.range (Nat.log 1 n).succ).sum fun i = …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p n : Nat
      h : LT.lt p 1
      ⊢ Eq (HMul.hMul (HSub.hSub p 1) ((Finset.range (Nat.log p n).succ).sum fun i = …
    -/
  · simp [lt_one_iff.mp h]
    /-
      case inr.inr
      p n : Nat
      h : LT.lt p 1
      ⊢ Eq 0 (HSub.hSub n (Nat.digits 0 n).sum)
    -/
    cases n
    /-
      case inr.inr.zero
      p : Nat
      h : LT.lt p 1
      ⊢ Eq 0 (HSub.hSub 0 (Nat.digits 0 0).sum)
    -/
    all_goals simp
    /-
      🎉 no goals
    -/


theorem digits_two_eq_bits (n : ℕ) : digits 2 n = n.bits.map fun b => cond b 1 0 := by
  /-
    n : Nat
    ⊢ Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
  -/
  induction' n using Nat.binaryRecFromOne with b n h ih
    /-
      case z₀
      ⊢ Eq (Nat.digits 2 0) (List.map (fun b => cond b 1 0) (Nat.bits 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case z₁
      ⊢ Eq (Nat.digits 2 1) (List.map (fun b => cond b 1 0) (Nat.bits 1))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case f
    b : Bool
    n : Nat
    h : Ne n 0
    ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
    ⊢ Eq (Nat.digits 2 (Nat.bit b n)) (List.map (fun b => cond b 1 0) (Nat.bit b n …
  -/
  rw [bits_append_bit _ _ fun hn => absurd hn h]
  /-
    case f
    b : Bool
    n : Nat
    h : Ne n 0
    ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
    ⊢ Eq (Nat.digits 2 (Nat.bit b n)) (List.map (fun b => cond b 1 0) (List.cons b …
  -/
  cases b
    /-
      case f.false
      n : Nat
      h : Ne n 0
      ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
      ⊢ Eq (Nat.digits 2 (Nat.bit Bool.false n)) (List.map (fun b => cond b 1 0) (Li …
    -/
  · rw [digits_def' one_lt_two]
      /-
        case f.false
        n : Nat
        h : Ne n 0
        ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
        ⊢ Eq (List.cons (HMod.hMod (Nat.bit Bool.false n) 2) (Nat.digits 2 (HDiv.hDiv  …
      -/
    · simpa [Nat.bit]
      /-
        🎉 no goals
      -/
      /-
        case f.false
        n : Nat
        h : Ne n 0
        ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
        ⊢ LT.lt 0 (Nat.bit Bool.false n)
      -/
    · simpa [Nat.bit, pos_iff_ne_zero]
      /-
        🎉 no goals
      -/
    /-
      case f.true
      n : Nat
      h : Ne n 0
      ih : Eq (Nat.digits 2 n) (List.map (fun b => cond b 1 0) n.bits)
      ⊢ Eq (Nat.digits 2 (Nat.bit Bool.true n)) (List.map (fun b => cond b 1 0) (Lis …
    -/
  · simpa [Nat.bit, add_comm, digits_add 2 one_lt_two 1 n, Nat.add_mul_div_left]
    /-
      🎉 no goals
    -/


theorem dvd_ofDigits_sub_ofDigits {α : Type*} [CommRing α] {a b k : α} (h : k ∣ a - b)
    (L : List ℕ) : k ∣ ofDigits a L - ofDigits b L := by
  /-
    α : Type u_1
    inst✝ : CommRing α
    a b k : α
    h : Dvd.dvd k (HSub.hSub a b)
    L : List Nat
    ⊢ Dvd.dvd k (HSub.hSub (Nat.ofDigits a L) (Nat.ofDigits b L))
  -/
  induction' L with d L ih
    /-
      case nil
      α : Type u_1
      inst✝ : CommRing α
      a b k : α
      h : Dvd.dvd k (HSub.hSub a b)
      ⊢ Dvd.dvd k (HSub.hSub (Nat.ofDigits a List.nil) (Nat.ofDigits b List.nil))
    -/
  · change k ∣ 0 - 0
    /-
      case nil
      α : Type u_1
      inst✝ : CommRing α
      a b k : α
      h : Dvd.dvd k (HSub.hSub a b)
      ⊢ Dvd.dvd k (HSub.hSub 0 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      α : Type u_1
      inst✝ : CommRing α
      a b k : α
      h : Dvd.dvd k (HSub.hSub a b)
      d : Nat
      L : List Nat
      ih : Dvd.dvd k (HSub.hSub (Nat.ofDigits a L) (Nat.ofDigits b L))
      ⊢ Dvd.dvd k (HSub.hSub (Nat.ofDigits a (List.cons d L)) (Nat.ofDigits b (List. …
    -/
  · simp only [ofDigits, add_sub_add_left_eq_sub]
    /-
      case cons
      α : Type u_1
      inst✝ : CommRing α
      a b k : α
      h : Dvd.dvd k (HSub.hSub a b)
      d : Nat
      L : List Nat
      ih : Dvd.dvd k (HSub.hSub (Nat.ofDigits a L) (Nat.ofDigits b L))
      ⊢ Dvd.dvd k (HSub.hSub (HMul.hMul a (Nat.ofDigits a L)) (HMul.hMul b (Nat.ofDi …
    -/
    exact dvd_mul_sub_mul h ih
    /-
      🎉 no goals
    -/


theorem ofDigits_modEq' (b b' : ℕ) (k : ℕ) (h : b ≡ b' [MOD k]) (L : List ℕ) :
    ofDigits b L ≡ ofDigits b' L [MOD k] := by
  /-
    b b' k : Nat
    h : k.ModEq b b'
    L : List Nat
    ⊢ k.ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
  -/
  induction' L with d L ih
    /-
      case nil
      b b' k : Nat
      h : k.ModEq b b'
      ⊢ k.ModEq (Nat.ofDigits b List.nil) (Nat.ofDigits b' List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      b b' k : Nat
      h : k.ModEq b b'
      d : Nat
      L : List Nat
      ih : k.ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
      ⊢ k.ModEq (Nat.ofDigits b (List.cons d L)) (Nat.ofDigits b' (List.cons d L))
    -/
  · dsimp [ofDigits]
    /-
      case cons
      b b' k : Nat
      h : k.ModEq b b'
      d : Nat
      L : List Nat
      ih : k.ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
      ⊢ k.ModEq (HAdd.hAdd d (HMul.hMul b (Nat.ofDigits b L))) (HAdd.hAdd d (HMul.hM …
    -/
    dsimp [Nat.ModEq] at *
    /-
      case cons
      b b' k : Nat
      h : Eq (HMod.hMod b k) (HMod.hMod b' k)
      d : Nat
      L : List Nat
      ih : Eq (HMod.hMod (Nat.ofDigits b L) k) (HMod.hMod (Nat.ofDigits b' L) k)
      ⊢ Eq (HMod.hMod (HAdd.hAdd d (HMul.hMul b (Nat.ofDigits b L))) k) (HMod.hMod ( …
    -/
    conv_lhs => rw [Nat.add_mod, Nat.mul_mod, h, ih]
    /-
      case cons
      b b' k : Nat
      h : Eq (HMod.hMod b k) (HMod.hMod b' k)
      d : Nat
      L : List Nat
      ih : Eq (HMod.hMod (Nat.ofDigits b L) k) (HMod.hMod (Nat.ofDigits b' L) k)
      ⊢ Eq (HMod.hMod (HAdd.hAdd (HMod.hMod d k) (HMod.hMod (HMul.hMul (HMod.hMod b' …
    -/
    conv_rhs => rw [Nat.add_mod, Nat.mul_mod]
    /-
      🎉 no goals
    -/


theorem ofDigits_modEq (b k : ℕ) (L : List ℕ) : ofDigits b L ≡ ofDigits (b % k) L [MOD k] :=
  ofDigits_modEq' b (b % k) k (b.mod_modEq k).symm L


theorem ofDigits_mod (b k : ℕ) (L : List ℕ) : ofDigits b L % k = ofDigits (b % k) L % k :=
  ofDigits_modEq b k L


theorem ofDigits_mod_eq_head! (b : ℕ) (l : List ℕ) : ofDigits b l % b = l.head! % b := by
  /-
    b : Nat
    l : List Nat
    ⊢ Eq (HMod.hMod (Nat.ofDigits b l) b) (HMod.hMod l.head! b)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [Nat.ofDigits, Int.ModEq]
                  /-
                    🎉 no goals
                  -/


theorem head!_digits {b n : ℕ} (h : b ≠ 1) : (Nat.digits b n).head! = n % b := by
  /-
    b n : Nat
    h : Ne b 1
    ⊢ Eq (b.digits n).head! (HMod.hMod n b)
  -/
  by_cases hb : 1 < b
    /-
      case pos
      b n : Nat
      h : Ne b 1
      hb : LT.lt 1 b
      ⊢ Eq (b.digits n).head! (HMod.hMod n b)
    -/
  · rcases n with _ | n
      /-
        case pos.zero
        b : Nat
        h : Ne b 1
        hb : LT.lt 1 b
        ⊢ Eq (b.digits 0).head! (HMod.hMod 0 b)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case pos.succ
        b : Nat
        h : Ne b 1
        hb : LT.lt 1 b
        n : Nat
        ⊢ Eq (b.digits (HAdd.hAdd n 1)).head! (HMod.hMod (HAdd.hAdd n 1) b)
      -/
    · nth_rw 2 [← Nat.ofDigits_digits b (n + 1)]
      /-
        case pos.succ
        b : Nat
        h : Ne b 1
        hb : LT.lt 1 b
        n : Nat
        ⊢ Eq (b.digits (HAdd.hAdd n 1)).head! (HMod.hMod (Nat.ofDigits b (b.digits (HA …
      -/
      rw [Nat.ofDigits_mod_eq_head! _ _]
      exact (Nat.mod_eq_of_lt (Nat.digits_lt_base hb <| List.head!_mem_self <|
          Nat.digits_ne_nil_iff_ne_zero.mpr <| Nat.succ_ne_zero n)).symm
    /-
      case neg
      b n : Nat
      h : Ne b 1
      hb : Not (LT.lt 1 b)
      ⊢ Eq (b.digits n).head! (HMod.hMod n b)
    -/
                            /-
                              🎉 no goals
                            -/
  · rcases n with _ | _ <;> simp_all [show b = 0 by omega]
                            /-
                              🎉 no goals
                            -/


theorem ofDigits_zmodeq' (b b' : ℤ) (k : ℕ) (h : b ≡ b' [ZMOD k]) (L : List ℕ) :
    ofDigits b L ≡ ofDigits b' L [ZMOD k] := by
  /-
    b b' : Int
    k : Nat
    h : (↑k).ModEq b b'
    L : List Nat
    ⊢ (↑k).ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
  -/
  induction' L with d L ih
    /-
      case nil
      b b' : Int
      k : Nat
      h : (↑k).ModEq b b'
      ⊢ (↑k).ModEq (Nat.ofDigits b List.nil) (Nat.ofDigits b' List.nil)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      b b' : Int
      k : Nat
      h : (↑k).ModEq b b'
      d : Nat
      L : List Nat
      ih : (↑k).ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
      ⊢ (↑k).ModEq (Nat.ofDigits b (List.cons d L)) (Nat.ofDigits b' (List.cons d L))
    -/
  · dsimp [ofDigits]
    /-
      case cons
      b b' : Int
      k : Nat
      h : (↑k).ModEq b b'
      d : Nat
      L : List Nat
      ih : (↑k).ModEq (Nat.ofDigits b L) (Nat.ofDigits b' L)
      ⊢ (↑k).ModEq (HAdd.hAdd (↑d) (HMul.hMul b (Nat.ofDigits b L))) (HAdd.hAdd (↑d) …
    -/
    dsimp [Int.ModEq] at *
    /-
      case cons
      b b' : Int
      k : Nat
      h : Eq (HMod.hMod b ↑k) (HMod.hMod b' ↑k)
      d : Nat
      L : List Nat
      ih : Eq (HMod.hMod (Nat.ofDigits b L) ↑k) (HMod.hMod (Nat.ofDigits b' L) ↑k)
      ⊢ Eq (HMod.hMod (HAdd.hAdd (↑d) (HMul.hMul b (Nat.ofDigits b L))) ↑k) (HMod.hM …
    -/
    conv_lhs => rw [Int.add_emod, Int.mul_emod, h, ih]
    /-
      case cons
      b b' : Int
      k : Nat
      h : Eq (HMod.hMod b ↑k) (HMod.hMod b' ↑k)
      d : Nat
      L : List Nat
      ih : Eq (HMod.hMod (Nat.ofDigits b L) ↑k) (HMod.hMod (Nat.ofDigits b' L) ↑k)
      ⊢ Eq (HMod.hMod (HAdd.hAdd (HMod.hMod ↑d ↑k) (HMod.hMod (HMul.hMul (HMod.hMod  …
    -/
    conv_rhs => rw [Int.add_emod, Int.mul_emod]
    /-
      🎉 no goals
    -/


theorem ofDigits_zmodeq (b : ℤ) (k : ℕ) (L : List ℕ) : ofDigits b L ≡ ofDigits (b % k) L [ZMOD k] :=
  ofDigits_zmodeq' b (b % k) k (b.mod_modEq ↑k).symm L


theorem ofDigits_zmod (b : ℤ) (k : ℕ) (L : List ℕ) : ofDigits b L % k = ofDigits (b % k) L % k :=
  ofDigits_zmodeq b k L


theorem modEq_digits_sum (b b' : ℕ) (h : b' % b = 1) (n : ℕ) : n ≡ (digits b' n).sum [MOD b] := by
  /-
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ b.ModEq n (b'.digits n).sum
  -/
  rw [← ofDigits_one]
  conv =>
    congr
    · skip
    · rw [← ofDigits_digits b' n]
  /-
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ b.ModEq (Nat.ofDigits b' (b'.digits n)) (Nat.ofDigits 1 (b'.digits n))
  -/
  convert ofDigits_modEq b' b (digits b' n)
  /-
    case h.e'_3.h.e'_3
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ Eq 1 (HMod.hMod b' b)
  -/
  exact h.symm
  /-
    🎉 no goals
  -/


theorem modEq_three_digits_sum (n : ℕ) : n ≡ (digits 10 n).sum [MOD 3] :=
                            /-
                              n : Nat
                              ⊢ Eq (HMod.hMod 10 3) 1
                            -/
  modEq_digits_sum 3 10 (by norm_num) n
                            /-
                              🎉 no goals
                            -/


theorem modEq_nine_digits_sum (n : ℕ) : n ≡ (digits 10 n).sum [MOD 9] :=
                            /-
                              n : Nat
                              ⊢ Eq (HMod.hMod 10 9) 1
                            -/
  modEq_digits_sum 9 10 (by norm_num) n
                            /-
                              🎉 no goals
                            -/


theorem zmodeq_ofDigits_digits (b b' : ℕ) (c : ℤ) (h : b' ≡ c [ZMOD b]) (n : ℕ) :
    n ≡ ofDigits c (digits b' n) [ZMOD b] := by
  conv =>
    congr
    · skip
    · rw [← ofDigits_digits b' n]
  /-
    b b' : Nat
    c : Int
    h : (↑b).ModEq (↑b') c
    n : Nat
    ⊢ (↑b).ModEq (↑(Nat.ofDigits b' (b'.digits n))) (Nat.ofDigits c (b'.digits n))
  -/
  rw [coe_int_ofDigits]
  /-
    b b' : Nat
    c : Int
    h : (↑b).ModEq (↑b') c
    n : Nat
    ⊢ (↑b).ModEq (Nat.ofDigits (↑b') (b'.digits n)) (Nat.ofDigits c (b'.digits n))
  -/
  apply ofDigits_zmodeq' _ _ _ h
  /-
    🎉 no goals
  -/


theorem ofDigits_neg_one :
    ∀ L : List ℕ, ofDigits (-1 : ℤ) L = (L.map fun n : ℕ => (n : ℤ)).alternatingSum
  | [] => rfl
              /-
                n : Nat
                ⊢ Eq (Nat.ofDigits (-1) (List.cons n List.nil)) (List.map (fun n => ↑n) (List. …
              -/
  | [n] => by simp [ofDigits, List.alternatingSum]
              /-
                🎉 no goals
              -/
  | a :: b :: t => by
    /-
      a b : Nat
      t : List Nat
      ⊢ Eq (Nat.ofDigits (-1) (List.cons a (List.cons b t))) (List.map (fun n => ↑n) …
    -/
    simp only [ofDigits, List.alternatingSum, List.map_cons, ofDigits_neg_one t]
    /-
      a b : Nat
      t : List Nat
      ⊢ Eq (HAdd.hAdd (↑a) (HMul.hMul (-1) (HAdd.hAdd (↑b) (HMul.hMul (-1) (List.map …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem modEq_eleven_digits_sum (n : ℕ) :
    n ≡ ((digits 10 n).map fun n : ℕ => (n : ℤ)).alternatingSum [ZMOD 11] := by
  /-
    n : Nat
    ⊢ Int.ModEq 11 (↑n) (List.map (fun n => ↑n) (Nat.digits 10 n)).alternatingSum
  -/
  have t := zmodeq_ofDigits_digits 11 10 (-1 : ℤ) (by unfold Int.ModEq; rfl) n
  /-
    n : Nat
    t : (↑11).ModEq (↑n) (Nat.ofDigits (-1) (Nat.digits 10 n))
    ⊢ Int.ModEq 11 (↑n) (List.map (fun n => ↑n) (Nat.digits 10 n)).alternatingSum
  -/
  rwa [ofDigits_neg_one] at t
  /-
    🎉 no goals
  -/


theorem dvd_iff_dvd_digits_sum (b b' : ℕ) (h : b' % b = 1) (n : ℕ) :
    b ∣ n ↔ b ∣ (digits b' n).sum := by
  /-
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ Iff (Dvd.dvd b n) (Dvd.dvd b (b'.digits n).sum)
  -/
  rw [← ofDigits_one]
  /-
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ Iff (Dvd.dvd b n) (Dvd.dvd b (Nat.ofDigits 1 (b'.digits n)))
  -/
  conv_lhs => rw [← ofDigits_digits b' n]
  /-
    b b' : Nat
    h : Eq (HMod.hMod b' b) 1
    n : Nat
    ⊢ Iff (Dvd.dvd b (Nat.ofDigits b' (b'.digits n))) (Dvd.dvd b (Nat.ofDigits 1 ( …
  -/
  rw [Nat.dvd_iff_mod_eq_zero, Nat.dvd_iff_mod_eq_zero, ofDigits_mod, h]
  /-
    🎉 no goals
  -/


/-- **Divisibility by 3 Rule** -/
theorem three_dvd_iff (n : ℕ) : 3 ∣ n ↔ 3 ∣ (digits 10 n).sum :=
                                  /-
                                    n : Nat
                                    ⊢ Eq (HMod.hMod 10 3) 1
                                  -/
  dvd_iff_dvd_digits_sum 3 10 (by norm_num) n
                                  /-
                                    🎉 no goals
                                  -/


theorem nine_dvd_iff (n : ℕ) : 9 ∣ n ↔ 9 ∣ (digits 10 n).sum :=
                                  /-
                                    n : Nat
                                    ⊢ Eq (HMod.hMod 10 9) 1
                                  -/
  dvd_iff_dvd_digits_sum 9 10 (by norm_num) n
                                  /-
                                    🎉 no goals
                                  -/


theorem dvd_iff_dvd_ofDigits (b b' : ℕ) (c : ℤ) (h : (b : ℤ) ∣ (b' : ℤ) - c) (n : ℕ) :
    b ∣ n ↔ (b : ℤ) ∣ ofDigits c (digits b' n) := by
  /-
    b b' : Nat
    c : Int
    h : Dvd.dvd (↑b) (HSub.hSub (↑b') c)
    n : Nat
    ⊢ Iff (Dvd.dvd b n) (Dvd.dvd (↑b) (Nat.ofDigits c (b'.digits n)))
  -/
  rw [← Int.natCast_dvd_natCast]
  exact
    dvd_iff_dvd_of_dvd_sub (zmodeq_ofDigits_digits b b' c (Int.modEq_iff_dvd.2 h).symm _).symm.dvd


theorem eleven_dvd_iff :
    11 ∣ n ↔ (11 : ℤ) ∣ ((digits 10 n).map fun n : ℕ => (n : ℤ)).alternatingSum := by
  /-
    n : Nat
    ⊢ Iff (Dvd.dvd 11 n) (Dvd.dvd 11 (List.map (fun n => ↑n) (Nat.digits 10 n)).al …
  -/
  have t := dvd_iff_dvd_ofDigits 11 10 (-1 : ℤ) (by norm_num) n
  /-
    n : Nat
    t : Iff (Dvd.dvd 11 n) (Dvd.dvd (↑11) (Nat.ofDigits (-1) (Nat.digits 10 n)))
    ⊢ Iff (Dvd.dvd 11 n) (Dvd.dvd 11 (List.map (fun n => ↑n) (Nat.digits 10 n)).al …
  -/
  rw [ofDigits_neg_one] at t
  /-
    n : Nat
    t : Iff (Dvd.dvd 11 n) (Dvd.dvd (↑11) (List.map (fun n => ↑n) (Nat.digits 10 n …
    ⊢ Iff (Dvd.dvd 11 n) (Dvd.dvd 11 (List.map (fun n => ↑n) (Nat.digits 10 n)).al …
  -/
  exact t
  /-
    🎉 no goals
  -/


theorem eleven_dvd_of_palindrome (p : (digits 10 n).Palindrome) (h : Even (digits 10 n).length) :
    11 ∣ n := by
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    h : Even (Nat.digits 10 n).length
    ⊢ Dvd.dvd 11 n
  -/
  let dig := (digits 10 n).map fun n : ℕ => (n : ℤ)
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    h : Even (Nat.digits 10 n).length
    dig : List Int := List.map (fun n => ↑n) (Nat.digits 10 n)
    ⊢ Dvd.dvd 11 n
  -/
  replace h : Even dig.length := by rwa [List.length_map]
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    dig : List Int := List.map (fun n => ↑n) (Nat.digits 10 n)
    h : Even dig.length
    ⊢ Dvd.dvd 11 n
  -/
  refine eleven_dvd_iff.2 ⟨0, (?_ : dig.alternatingSum = 0)⟩
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    dig : List Int := List.map (fun n => ↑n) (Nat.digits 10 n)
    h : Even dig.length
    ⊢ Eq dig.alternatingSum 0
  -/
  have := dig.alternatingSum_reverse
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    dig : List Int := List.map (fun n => ↑n) (Nat.digits 10 n)
    h : Even dig.length
    this : Eq dig.reverse.alternatingSum (HSMul.hSMul (HPow.hPow (-1) (HAdd.hAdd d …
    ⊢ Eq dig.alternatingSum 0
  -/
  rw [(p.map _).reverse_eq, _root_.pow_succ', h.neg_one_pow, mul_one, neg_one_zsmul] at this
  /-
    n : Nat
    p : (Nat.digits 10 n).Palindrome
    dig : List Int := List.map (fun n => ↑n) (Nat.digits 10 n)
    h : Even dig.length
    this : Eq (List.map (fun n => ↑n) (Nat.digits 10 n)).alternatingSum (Neg.neg d …
    ⊢ Eq dig.alternatingSum 0
  -/
  exact eq_zero_of_neg_eq this.symm
  /-
    🎉 no goals
  -/


lemma toDigitsCore_lens_eq_aux (b f : Nat) :
    ∀ (n : Nat) (l1 l2 : List Char), l1.length = l2.length →
    (Nat.toDigitsCore b f n l1).length = (Nat.toDigitsCore b f n l2).length := by
  induction f with (simp only [Nat.toDigitsCore, List.length]; intro n l1 l2 hlen)
  | zero => assumption
  | succ f ih =>
    if hx : n / b = 0 then
      simp only [hx, if_true, List.length, congrArg (fun l ↦ l + 1) hlen]
    else
      simp only [hx, if_false]
      specialize ih (n / b) (Nat.digitChar (n % b) :: l1) (Nat.digitChar (n % b) :: l2)
      simp only [List.length, congrArg (fun l ↦ l + 1) hlen] at ih
      exact ih trivial

@[deprecated (since := "2024-02-19")] alias to_digits_core_lens_eq_aux:= toDigitsCore_lens_eq_aux


lemma toDigitsCore_lens_eq (b f : Nat) : ∀ (n : Nat) (c : Char) (tl : List Char),
    (Nat.toDigitsCore b f n (c :: tl)).length = (Nat.toDigitsCore b f n tl).length + 1 := by
  induction f with (intro n c tl; simp only [Nat.toDigitsCore, List.length])
  | succ f ih =>
    if hnb : (n / b) = 0 then
      simp only [hnb, if_true, List.length]
    else
      generalize hx : Nat.digitChar (n % b) = x
      simp only [hx, hnb, if_false] at ih
      simp only [hnb, if_false]
      specialize ih (n / b) c (x :: tl)
      rw [← ih]
      have lens_eq : (x :: (c :: tl)).length = (c :: x :: tl).length := by simp
      apply toDigitsCore_lens_eq_aux
      exact lens_eq

@[deprecated (since := "2024-02-19")] alias to_digits_core_lens_eq:= toDigitsCore_lens_eq


lemma nat_repr_len_aux (n b e : Nat) (h_b_pos : 0 < b) :  n < b ^ e.succ → n / b < b ^ e := by
  /-
    n b e : Nat
    h_b_pos : LT.lt 0 b
    ⊢ LT.lt n (HPow.hPow b e.succ) → LT.lt (HDiv.hDiv n b) (HPow.hPow b e)
  -/
  simp only [Nat.pow_succ]
  /-
    n b e : Nat
    h_b_pos : LT.lt 0 b
    ⊢ LT.lt n (HMul.hMul (HPow.hPow b e) b) → LT.lt (HDiv.hDiv n b) (HPow.hPow b e)
  -/
  exact (@Nat.div_lt_iff_lt_mul b n (b ^ e) h_b_pos).mpr
  /-
    🎉 no goals
  -/


/-- The String representation produced by toDigitsCore has the proper length relative to
the number of digits in `n < e` for some base `b`. Since this works with any base greater
than one, it can be used for binary, decimal, and hex. -/
lemma toDigitsCore_length (b : Nat) (h : 2 <= b) (f n e : Nat)
    (hlt : n < b ^ e) (h_e_pos : 0 < e) : (Nat.toDigitsCore b f n []).length <= e := by
  induction f generalizing n e hlt h_e_pos with
    simp only [Nat.toDigitsCore, List.length, Nat.zero_le]
  | succ f ih =>
    cases e with
    | zero => exact False.elim (Nat.lt_irrefl 0 h_e_pos)
    | succ e =>
      if h_pred_pos : 0 < e then
        have _ : 0 < b := Nat.lt_trans (by decide) h
        specialize ih (n / b) e (nat_repr_len_aux n b e ‹0 < b› hlt) h_pred_pos
        if hdiv_ten : n / b = 0 then
          simp only [hdiv_ten]; exact Nat.le.step h_pred_pos
        else
          simp only [hdiv_ten,
            toDigitsCore_lens_eq b f (n / b) (Nat.digitChar <| n % b), if_false]
          exact Nat.succ_le_succ ih
      else
        obtain rfl : e = 0 := Nat.eq_zero_of_not_pos h_pred_pos
        have _ : b ^ 1 = b := by simp only [Nat.pow_succ, pow_zero, Nat.one_mul]
        have _ : n < b := ‹b ^ 1 = b› ▸ hlt
        simp [(@Nat.div_eq_of_lt n b ‹n < b› : n / b = 0)]

@[deprecated (since := "2024-02-19")] alias to_digits_core_length := toDigitsCore_length


/-- The core implementation of `Nat.repr` returns a String with length less than or equal to the
number of digits in the decimal number (represented by `e`). For example, the decimal string
representation of any number less than 1000 (10 ^ 3) has a length less than or equal to 3. -/
lemma repr_length (n e : Nat) : 0 < e → n < 10 ^ e → (Nat.repr n).length <= e := by
  cases n with
    (intro e0 he; simp only [Nat.repr, Nat.toDigits, String.length, List.asString])
  | zero => assumption
  | succ n =>
    if hterm : n.succ / 10 = 0 then
      simp only [hterm, Nat.toDigitsCore]; assumption
    else
      exact toDigitsCore_length 10 (by decide) (Nat.succ n + 1) (Nat.succ n) e he e0


theorem digits_succ (b n m r l) (e : r + b * m = n) (hr : r < b)
    (h : Nat.digits b m = l ∧ 1 < b ∧ 0 < m) : (Nat.digits b n = r :: l) ∧ 1 < b ∧ 0 < n := by
  /-
    b n m r : Nat
    l : List Nat
    e : Eq (HAdd.hAdd r (HMul.hMul b m)) n
    hr : LT.lt r b
    h : And (Eq (b.digits m) l) (And (LT.lt 1 b) (LT.lt 0 m))
    ⊢ And (Eq (b.digits n) (List.cons r l)) (And (LT.lt 1 b) (LT.lt 0 n))
  -/
  rcases h with ⟨h, b2, m0⟩
  /-
    case intro.intro
    b n m r : Nat
    l : List Nat
    e : Eq (HAdd.hAdd r (HMul.hMul b m)) n
    hr : LT.lt r b
    h : Eq (b.digits m) l
    b2 : LT.lt 1 b
    m0 : LT.lt 0 m
    ⊢ And (Eq (b.digits n) (List.cons r l)) (And (LT.lt 1 b) (LT.lt 0 n))
  -/
  have b0 : 0 < b := by omega
  /-
    case intro.intro
    b n m r : Nat
    l : List Nat
    e : Eq (HAdd.hAdd r (HMul.hMul b m)) n
    hr : LT.lt r b
    h : Eq (b.digits m) l
    b2 : LT.lt 1 b
    m0 : LT.lt 0 m
    b0 : LT.lt 0 b
    ⊢ And (Eq (b.digits n) (List.cons r l)) (And (LT.lt 1 b) (LT.lt 0 n))
  -/
  have n0 : 0 < n := by linarith [mul_pos b0 m0]
  /-
    case intro.intro
    b n m r : Nat
    l : List Nat
    e : Eq (HAdd.hAdd r (HMul.hMul b m)) n
    hr : LT.lt r b
    h : Eq (b.digits m) l
    b2 : LT.lt 1 b
    m0 : LT.lt 0 m
    b0 : LT.lt 0 b
    n0 : LT.lt 0 n
    ⊢ And (Eq (b.digits n) (List.cons r l)) (And (LT.lt 1 b) (LT.lt 0 n))
  -/
  refine ⟨?_, b2, n0⟩
  /-
    case intro.intro
    b n m r : Nat
    l : List Nat
    e : Eq (HAdd.hAdd r (HMul.hMul b m)) n
    hr : LT.lt r b
    h : Eq (b.digits m) l
    b2 : LT.lt 1 b
    m0 : LT.lt 0 m
    b0 : LT.lt 0 b
    n0 : LT.lt 0 n
    ⊢ Eq (b.digits n) (List.cons r l)
  -/
  obtain ⟨rfl, rfl⟩ := (Nat.div_mod_unique b0).2 ⟨e, hr⟩
  /-
    case intro.intro.intro
    b n : Nat
    l : List Nat
    b2 : LT.lt 1 b
    b0 : LT.lt 0 b
    n0 : LT.lt 0 n
    h : Eq (b.digits (HDiv.hDiv n b)) l
    m0 : LT.lt 0 (HDiv.hDiv n b)
    hr : LT.lt (HMod.hMod n b) b
    e : Eq (HAdd.hAdd (HMod.hMod n b) (HMul.hMul b (HDiv.hDiv n b))) n
    ⊢ Eq (b.digits n) (List.cons (HMod.hMod n b) l)
  -/
  subst h; exact Nat.digits_def' b2 n0
           /-
             🎉 no goals
           -/


theorem digits_one (b n) (n0 : 0 < n) (nb : n < b) : Nat.digits b n = [n] ∧ 1 < b ∧ 0 < n := by
  have b2 : 1 < b :=
    lt_iff_add_one_le.mpr (le_trans (add_le_add_right (lt_iff_add_one_le.mp n0) 1) nb)
  /-
    b n : Nat
    n0 : LT.lt 0 n
    nb : LT.lt n b
    b2 : LT.lt 1 b
    ⊢ And (Eq (b.digits n) (List.cons n List.nil)) (And (LT.lt 1 b) (LT.lt 0 n))
  -/
  refine ⟨?_, b2, n0⟩
  /-
    b n : Nat
    n0 : LT.lt 0 n
    nb : LT.lt n b
    b2 : LT.lt 1 b
    ⊢ Eq (b.digits n) (List.cons n List.nil)
  -/
  rw [Nat.digits_def' b2 n0, Nat.mod_eq_of_lt nb, Nat.div_eq_zero_iff.2 <| .inr nb, Nat.digits_zero]
  /-
    🎉 no goals
  -/

/-
Porting note: this part of the file is tactic related.

open Tactic
-- failed to format: unknown constant 'term.pseudo.antiquot'
/-- Helper function for the `norm_digits` tactic. -/ unsafe
  def
    eval_aux
    ( eb : expr ) ( b : ℕ ) : expr → ℕ → instance_cache → tactic ( instance_cache × expr × expr )
    |
      en , n , ic
      =>
      do
        let m := n / b
          let r := n % b
          let ( ic , er ) ← ic . ofNat r
          let ( ic , pr ) ← norm_num.prove_lt_nat ic er eb
          if
            m = 0
            then
            do
              let ( _ , pn0 ) ← norm_num.prove_pos ic en
                return
                  (
                    ic
                      ,
                      q( ( [ $ ( en ) ] : List Nat ) )
                        ,
                        q( digits_one $ ( eb ) $ ( en ) $ ( pn0 ) $ ( pr ) )
                    )
            else
            do
              let em ← expr.of_nat q( ℕ ) m
                let ( _ , pe ) ← norm_num.derive q( ( $ ( er ) + $ ( eb ) * $ ( em ) : ℕ ) )
                let ( ic , el , p ) ← eval_aux em m ic
                return
                  (
                    ic
                      ,
                      q( @ List.cons ℕ $ ( er ) $ ( el ) )
                        ,
                        q(
                          digits_succ
                            $ ( eb ) $ ( en ) $ ( em ) $ ( er ) $ ( el ) $ ( pe ) $ ( pr ) $ ( p )
                          )
                    )

/-- A tactic for normalizing expressions of the form `Nat.digits a b = l` where
`a` and `b` are numerals.

```
example : Nat.digits 10 123 = [3,2,1] := by norm_num
```
-/
@[norm_num]
unsafe def eval : expr → tactic (expr × expr)
  | q(Nat.digits $(eb) $(en)) => do
    let b ← expr.to_nat eb
    let n ← expr.to_nat en
    if n = 0 then return (q(([] : List ℕ)), q(Nat.digits_zero $(eb)))
      else
        if b = 0 then do
          let ic ← mk_instance_cache q(ℕ)
          let (_, pn0) ← norm_num.prove_ne_zero' ic en
          return (q(([$(en)] : List ℕ)), q(@Nat.digits_zero_succ' $(en) $(pn0)))
        else
          if b = 1 then do
            let ic ← mk_instance_cache q(ℕ)
            let s ← simp_lemmas.add_simp simp_lemmas.mk `list.replicate
            let (rhs, p2, _) ← simplify s [] q(List.replicate $(en) 1)
            let p ← mk_eq_trans q(Nat.digits_one $(en)) p2
            return (rhs, p)
          else do
            let ic ← mk_instance_cache q(ℕ)
            let (_, l, p) ← eval_aux eb b en n ic
            let p ← mk_app `` And.left [p]
            return (l, p)
  | _ => failed
-/


