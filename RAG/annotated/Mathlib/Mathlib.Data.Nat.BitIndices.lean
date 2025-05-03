/-- The function which maps each natural number `∑ i in s, 2^i` to the list of
elements of `s` in increasing order. -/
def bitIndices (n : ℕ) : List ℕ :=
  @binaryRec (fun _ ↦ List ℕ) [] (fun b _ s ↦ b.casesOn (s.map (· + 1)) (0 :: s.map (· + 1))) n


                                                          /-
                                                            ⊢ Eq (Nat.bitIndices 0) List.nil
                                                          -/
@[simp] theorem bitIndices_zero : bitIndices 0 = [] := by simp [bitIndices]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                          /-
                                                            ⊢ Eq (Nat.bitIndices 1) (List.cons 0 List.nil)
                                                          -/
@[simp] theorem bitIndices_one : bitIndices 1 = [0] := by simp [bitIndices]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem bitIndices_bit_true (n : ℕ) :
    bitIndices (bit true n) = 0 :: ((bitIndices n).map (· + 1)) :=
  binaryRec_eq _ _ (.inl rfl)


theorem bitIndices_bit_false (n : ℕ) :
    bitIndices (bit false n) = (bitIndices n).map (· + 1) :=
  binaryRec_eq _ _ (.inl rfl)


@[simp] theorem bitIndices_two_mul_add_one (n : ℕ) :
    bitIndices (2 * n + 1) = 0 :: (bitIndices n).map (· + 1) := by
   /-
     n : Nat
     ⊢ Eq (HAdd.hAdd (HMul.hMul 2 n) 1).bitIndices (List.cons 0 (List.map (fun x => …
   -/
   rw [← bitIndices_bit_true, bit_true]
   /-
     🎉 no goals
   -/


@[simp] theorem bitIndices_two_mul (n : ℕ) :
    bitIndices (2 * n) = (bitIndices n).map (· + 1) := by
  /-
    n : Nat
    ⊢ Eq (HMul.hMul 2 n).bitIndices (List.map (fun x => HAdd.hAdd x 1) n.bitIndices)
  -/
  rw [← bitIndices_bit_false, bit_false]
  /-
    🎉 no goals
  -/


@[simp] theorem bitIndices_sorted {n : ℕ} : n.bitIndices.Sorted (· < ·) := by
  /-
    n : Nat
    ⊢ List.Sorted (fun x1 x2 => LT.lt x1 x2) n.bitIndices
  -/
  induction' n using binaryRec with b n hs
    /-
      case z
      ⊢ List.Sorted (fun x1 x2 => LT.lt x1 x2) (Nat.bitIndices 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  suffices List.Pairwise (fun a b ↦ a < b) n.bitIndices by
    cases b <;> simpa [List.Sorted, bit_false, bit_true, List.pairwise_map]
  /-
    case f
    b : Bool
    n : Nat
    hs : List.Sorted (fun x1 x2 => LT.lt x1 x2) n.bitIndices
    ⊢ List.Pairwise (fun a b => LT.lt a b) n.bitIndices
  -/
  exact List.Pairwise.imp (by simp) hs
  /-
    🎉 no goals
  -/


@[simp] theorem bitIndices_two_pow_mul (k n : ℕ) :
    bitIndices (2^k * n) = (bitIndices n).map (· + k) := by
  /-
    k n : Nat
    ⊢ Eq (HMul.hMul (HPow.hPow 2 k) n).bitIndices (List.map (fun x => HAdd.hAdd x  …
  -/
  induction' k with k ih
    /-
      case zero
      n : Nat
      ⊢ Eq (HMul.hMul (HPow.hPow 2 0) n).bitIndices (List.map (fun x => HAdd.hAdd x  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n k : Nat
    ih : Eq (HMul.hMul (HPow.hPow 2 k) n).bitIndices (List.map (fun x => HAdd.hAdd …
    ⊢ Eq (HMul.hMul (HPow.hPow 2 (HAdd.hAdd k 1)) n).bitIndices (List.map (fun x = …
  -/
  rw [add_comm, pow_add, pow_one, mul_assoc, bitIndices_two_mul, ih, List.map_map, comp_add_right]
  /-
    case succ
    n k : Nat
    ih : Eq (HMul.hMul (HPow.hPow 2 k) n).bitIndices (List.map (fun x => HAdd.hAdd …
    ⊢ Eq (List.map (fun x => HAdd.hAdd x (HAdd.hAdd k 1)) n.bitIndices) (List.map  …
  -/
  simp [add_comm (a := 1)]
  /-
    🎉 no goals
  -/


@[simp] theorem bitIndices_two_pow (k : ℕ) : bitIndices (2^k) = [k] := by
  /-
    k : Nat
    ⊢ Eq (HPow.hPow 2 k).bitIndices (List.cons k List.nil)
  -/
  rw [← mul_one (a := 2^k), bitIndices_two_pow_mul]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp] theorem twoPowSum_bitIndices (n : ℕ) : (n.bitIndices.map (fun i ↦ 2 ^ i)).sum = n := by
  /-
    n : Nat
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) n.bitIndices).sum n
  -/
  induction' n using binaryRec with b n hs
    /-
      case z
      ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (Nat.bitIndices 0)).sum 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  have hrw : (fun i ↦ 2^i) ∘ (fun x ↦ x+1) = fun i ↦ 2 * 2 ^ i := by
    ext i; simp [pow_add, mul_comm]
  /-
    case f
    b : Bool
    n : Nat
    hs : Eq (List.map (fun i => HPow.hPow 2 i) n.bitIndices).sum n
    hrw : Eq (Function.comp (fun i => HPow.hPow 2 i) fun x => HAdd.hAdd x 1) fun i …
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (Nat.bit b n).bitIndices).sum (Nat.bit …
  -/
  cases b
    /-
      case f.false
      n : Nat
      hs : Eq (List.map (fun i => HPow.hPow 2 i) n.bitIndices).sum n
      hrw : Eq (Function.comp (fun i => HPow.hPow 2 i) fun x => HAdd.hAdd x 1) fun i …
      ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (Nat.bit Bool.false n).bitIndices).sum …
    -/
  · simpa [hrw, List.sum_map_mul_left]
    /-
      🎉 no goals
    -/
  /-
    case f.true
    n : Nat
    hs : Eq (List.map (fun i => HPow.hPow 2 i) n.bitIndices).sum n
    hrw : Eq (Function.comp (fun i => HPow.hPow 2 i) fun x => HAdd.hAdd x 1) fun i …
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (Nat.bit Bool.true n).bitIndices).sum  …
  -/
  simp [hrw, List.sum_map_mul_left, hs, add_comm (a := 1)]
  /-
    🎉 no goals
  -/


/-- Together with `Nat.twoPowSum_bitIndices`, this implies a bijection between `ℕ` and `Finset ℕ`.
See `Finset.equivBitIndices` for this bijection. -/
theorem bitIndices_twoPowsum {L : List ℕ} (hL : List.Sorted (· < ·) L) :
    (L.map (fun i ↦ 2^i)).sum.bitIndices = L := by
  /-
    L : List Nat
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) L
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) L).sum.bitIndices L
  -/
  cases' L with a L
    /-
      case nil
      hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) List.nil
      ⊢ Eq (List.map (fun i => HPow.hPow 2 i) List.nil).sum.bitIndices List.nil
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case cons
    a : Nat
    L : List Nat
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a L)
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (List.cons a L)).sum.bitIndices (List. …
  -/
  obtain ⟨haL, hL⟩ := sorted_cons.1 hL
  /-
    case cons.intro
    a : Nat
    L : List Nat
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a L)
    haL : ∀ (b : Nat), Membership.mem L b → LT.lt a b
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) L
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (List.cons a L)).sum.bitIndices (List. …
  -/
  simp_rw [Nat.lt_iff_add_one_le] at haL
  have h' : ∃ (L₀ : List ℕ), L₀.Sorted (· < ·) ∧ L = L₀.map (· + a + 1) := by
    refine ⟨L.map (· - (a+1)), ?_, ?_⟩
    · rwa [Sorted, pairwise_map, Pairwise.and_mem,
        Pairwise.iff (S := fun x y ↦ x ∈ L ∧ y ∈ L ∧ x < y), ← Pairwise.and_mem]
      simp only [and_congr_right_iff]
      exact fun x y hx _ ↦ by rw [tsub_lt_tsub_iff_right (haL _ hx)]
    have h' : ∀ x ∈ L, ((fun x ↦ x + a + 1) ∘ (fun x ↦ x - (a + 1))) x = x := fun x hx ↦ by
      simp only [add_assoc, Function.comp_apply]; rw [tsub_add_cancel_of_le (haL _ hx)]
    simp [List.map_congr_left h']
  /-
    case cons.intro
    a : Nat
    L : List Nat
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a L)
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) L
    haL : ∀ (b : Nat), Membership.mem L b → LE.le (HAdd.hAdd a 1) b
    h' : Exists fun L₀ => And (List.Sorted (fun x1 x2 => LT.lt x1 x2) L₀) (Eq L (L …
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (List.cons a L)).sum.bitIndices (List. …
  -/
  obtain ⟨L₀, hL₀, rfl⟩ := h'
  /-
    case cons.intro.intro.intro
    a : Nat
    L₀ : List Nat
    hL₀ : List.Sorted (fun x1 x2 => LT.lt x1 x2) L₀
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a (List.map (fun x =>  …
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.map (fun x => HAdd.hAdd (HAd …
    haL : ∀ (b : Nat), Membership.mem (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (List.cons a (List.map (fun x => HAdd. …
  -/
  have _ : L₀.length < (a :: (L₀.map (· + a + 1))).length := by simp
  have hrw : (2^·) ∘ (· + a + 1) = fun i ↦ 2^a * (2 * 2^i) := by
    ext x; simp only [Function.comp_apply, pow_add, pow_one]; ac_rfl
  /-
    case cons.intro.intro.intro
    a : Nat
    L₀ : List Nat
    hL₀ : List.Sorted (fun x1 x2 => LT.lt x1 x2) L₀
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a (List.map (fun x =>  …
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.map (fun x => HAdd.hAdd (HAd …
    haL : ∀ (b : Nat), Membership.mem (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    x✝ : LT.lt L₀.length (List.cons a (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    hrw : Eq (Function.comp (fun x => HPow.hPow 2 x) fun x => HAdd.hAdd (HAdd.hAdd …
    ⊢ Eq (List.map (fun i => HPow.hPow 2 i) (List.cons a (List.map (fun x => HAdd. …
  -/
  simp only [List.map_cons, List.map_map, List.sum_map_mul_left, List.sum_cons, hrw]
  /-
    case cons.intro.intro.intro
    a : Nat
    L₀ : List Nat
    hL₀ : List.Sorted (fun x1 x2 => LT.lt x1 x2) L₀
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a (List.map (fun x =>  …
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.map (fun x => HAdd.hAdd (HAd …
    haL : ∀ (b : Nat), Membership.mem (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    x✝ : LT.lt L₀.length (List.cons a (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    hrw : Eq (Function.comp (fun x => HPow.hPow 2 x) fun x => HAdd.hAdd (HAdd.hAdd …
    ⊢ Eq (HAdd.hAdd (HPow.hPow 2 a) (HMul.hMul (HPow.hPow 2 a) (HMul.hMul 2 (List. …
  -/
  nth_rw 1 [← mul_one (a := 2^a)]
  rw [← mul_add, bitIndices_two_pow_mul, add_comm, bitIndices_two_mul_add_one,
    bitIndices_twoPowsum hL₀]
  /-
    case cons.intro.intro.intro
    a : Nat
    L₀ : List Nat
    hL₀ : List.Sorted (fun x1 x2 => LT.lt x1 x2) L₀
    hL✝ : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.cons a (List.map (fun x =>  …
    hL : List.Sorted (fun x1 x2 => LT.lt x1 x2) (List.map (fun x => HAdd.hAdd (HAd …
    haL : ∀ (b : Nat), Membership.mem (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    x✝ : LT.lt L₀.length (List.cons a (List.map (fun x => HAdd.hAdd (HAdd.hAdd x a …
    hrw : Eq (Function.comp (fun x => HPow.hPow 2 x) fun x => HAdd.hAdd (HAdd.hAdd …
    ⊢ Eq (List.map (fun x => HAdd.hAdd x a) (List.cons 0 (List.map (fun x => HAdd. …
  -/
  simp [add_comm (a := 1), add_assoc]
  /-
    🎉 no goals
  -/
termination_by L.length


theorem two_pow_le_of_mem_bitIndices (ha : a ∈ n.bitIndices) : 2^a ≤ n := by
  /-
    a n : Nat
    ha : Membership.mem n.bitIndices a
    ⊢ LE.le (HPow.hPow 2 a) n
  -/
  rw [← twoPowSum_bitIndices n]
  /-
    a n : Nat
    ha : Membership.mem n.bitIndices a
    ⊢ LE.le (HPow.hPow 2 a) (List.map (fun i => HPow.hPow 2 i) n.bitIndices).sum
  -/
  exact List.single_le_sum (by simp) _ <| mem_map_of_mem _ ha
  /-
    🎉 no goals
  -/


theorem not_mem_bitIndices_self (n : ℕ) : n ∉ n.bitIndices :=
  fun h ↦ (n.lt_two_pow_self).not_le <| two_pow_le_of_mem_bitIndices h


