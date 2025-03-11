theorem geom_sum_succ {x : α} {n : ℕ} :
    ∑ i ∈ range (n + 1), x ^ i = (x * ∑ i ∈ range n, x ^ i) + 1 := by
  /-
    α : Type u
    inst✝ : Semiring α
    x : α
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun i => HPow.hPow x i) (HAdd.hAdd (H …
  -/
  simp only [mul_sum, ← pow_succ', sum_range_succ', pow_zero]
  /-
    🎉 no goals
  -/


theorem geom_sum_succ' {x : α} {n : ℕ} :
    ∑ i ∈ range (n + 1), x ^ i = x ^ n + ∑ i ∈ range n, x ^ i :=
  (sum_range_succ _ _).trans (add_comm _ _)


theorem geom_sum_zero (x : α) : ∑ i ∈ range 0, x ^ i = 0 :=
  rfl


                                                              /-
                                                                α : Type u
                                                                inst✝ : Semiring α
                                                                x : α
                                                                ⊢ Eq ((Finset.range 1).sum fun i => HPow.hPow x i) 1
                                                              -/
theorem geom_sum_one (x : α) : ∑ i ∈ range 1, x ^ i = 1 := by simp [geom_sum_succ']
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                  /-
                                                                    α : Type u
                                                                    inst✝ : Semiring α
                                                                    x : α
                                                                    ⊢ Eq ((Finset.range 2).sum fun i => HPow.hPow x i) (HAdd.hAdd x 1)
                                                                  -/
theorem geom_sum_two {x : α} : ∑ i ∈ range 2, x ^ i = x + 1 := by simp [geom_sum_succ']
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem zero_geom_sum : ∀ {n}, ∑ i ∈ range n, (0 : α) ^ i = if n = 0 then 0 else 1
            /-
              α : Type u
              inst✝ : Semiring α
              ⊢ Eq ((Finset.range 0).sum fun i => HPow.hPow 0 i) (ite (Eq 0 0) 0 1)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              α : Type u
              inst✝ : Semiring α
              ⊢ Eq ((Finset.range 1).sum fun i => HPow.hPow 0 i) (ite (Eq 1 0) 0 1)
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
  | n + 2 => by
    /-
      α : Type u
      inst✝ : Semiring α
      n : Nat
      ⊢ Eq ((Finset.range (HAdd.hAdd n 2)).sum fun i => HPow.hPow 0 i) (ite (Eq (HAd …
    -/
    rw [geom_sum_succ']
    /-
      α : Type u
      inst✝ : Semiring α
      n : Nat
      ⊢ Eq (HAdd.hAdd (HPow.hPow 0 (HAdd.hAdd n 1)) ((Finset.range (HAdd.hAdd n 1)). …
    -/
    simp [zero_geom_sum]
    /-
      🎉 no goals
    -/


                                                                    /-
                                                                      α : Type u
                                                                      inst✝ : Semiring α
                                                                      n : Nat
                                                                      ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow 1 i) ↑n
                                                                    -/
theorem one_geom_sum (n : ℕ) : ∑ i ∈ range n, (1 : α) ^ i = n := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem op_geom_sum (x : α) (n : ℕ) : op (∑ i ∈ range n, x ^ i) = ∑ i ∈ range n, op x ^ i := by
  /-
    α : Type u
    inst✝ : Semiring α
    x : α
    n : Nat
    ⊢ Eq (MulOpposite.op ((Finset.range n).sum fun i => HPow.hPow x i)) ((Finset.r …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem op_geom_sum₂ (x y : α) (n : ℕ) : ∑ i ∈ range n, op y ^ (n - 1 - i) * op x ^ i =
    ∑ i ∈ range n, op y ^ i * op x ^ (n - 1 - i) := by
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow (MulOpposite.op y) (H …
  -/
  rw [← sum_range_reflect]
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun j => HMul.hMul (HPow.hPow (MulOpposite.op y) (H …
  -/
  refine sum_congr rfl fun j j_in => ?_
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    n j : Nat
    j_in : Membership.mem (Finset.range n) j
    ⊢ Eq (HMul.hMul (HPow.hPow (MulOpposite.op y) (HSub.hSub (HSub.hSub n 1) (HSub …
  -/
  rw [mem_range, Nat.lt_iff_add_one_le] at j_in
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    n j : Nat
    j_in : LE.le (HAdd.hAdd j 1) n
    ⊢ Eq (HMul.hMul (HPow.hPow (MulOpposite.op y) (HSub.hSub (HSub.hSub n 1) (HSub …
  -/
  congr
  /-
    case e_a.e_a
    α : Type u
    inst✝ : Semiring α
    x y : α
    n j : Nat
    j_in : LE.le (HAdd.hAdd j 1) n
    ⊢ Eq (HSub.hSub (HSub.hSub n 1) (HSub.hSub (HSub.hSub n 1) j)) j
  -/
  apply tsub_tsub_cancel_of_le
  /-
    case e_a.e_a.h
    α : Type u
    inst✝ : Semiring α
    x y : α
    n j : Nat
    j_in : LE.le (HAdd.hAdd j 1) n
    ⊢ LE.le j (HSub.hSub n 1)
  -/
  exact le_tsub_of_add_le_right j_in
  /-
    🎉 no goals
  -/


theorem geom_sum₂_with_one (x : α) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * 1 ^ (n - 1 - i) = ∑ i ∈ range n, x ^ i :=
                              /-
                                α : Type u
                                inst✝ : Semiring α
                                x : α
                                n i : Nat
                                x✝ : Membership.mem (Finset.range n) i
                                ⊢ Eq (HMul.hMul (HPow.hPow x i) (HPow.hPow 1 (HSub.hSub (HSub.hSub n 1) i))) ( …
                              -/
  sum_congr rfl fun i _ => by rw [one_pow, mul_one]
                              /-
                                🎉 no goals
                              -/


/-- $x^n-y^n = (x-y) \sum x^ky^{n-1-k}$ reformulated without `-` signs. -/
protected theorem Commute.geom_sum₂_mul_add {x y : α} (h : Commute x y) (n : ℕ) :
    (∑ i ∈ range n, (x + y) ^ i * y ^ (n - 1 - i)) * x + y ^ n = (x + y) ^ n := by
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow …
  -/
  let f :  ℕ → ℕ → α := fun m i : ℕ => (x + y) ^ i * y ^ (m - 1 - i)
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n : Nat
    f : Nat → Nat → α := fun m i => HMul.hMul (HPow.hPow (HAdd.hAdd x y) i) (HPow. …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow …
  -/
  change (∑ i ∈ range n, (f n) i) * x + y ^ n = (x + y) ^ n
  induction n with
  | zero => rw [range_zero, sum_empty, zero_mul, zero_add, pow_zero, pow_zero]
  | succ n ih =>
    have f_last : f (n + 1) n = (x + y) ^ n := by
      dsimp only [f]
      rw [← tsub_add_eq_tsub_tsub, Nat.add_comm, tsub_self, pow_zero, mul_one]
    have f_succ : ∀ i, i ∈ range n → f (n + 1) i = y * f n i := fun i hi => by
      dsimp only [f]
      have : Commute y ((x + y) ^ i) := (h.symm.add_right (Commute.refl y)).pow_right i
      rw [← mul_assoc, this.eq, mul_assoc, ← pow_succ' y (n - 1 - i), add_tsub_cancel_right,
        ← tsub_add_eq_tsub_tsub, add_comm 1 i]
      have : i + 1 + (n - (i + 1)) = n := add_tsub_cancel_of_le (mem_range.mp hi)
      rw [add_comm (i + 1)] at this
      rw [← this, add_tsub_cancel_right, add_comm i 1, ← add_assoc, add_tsub_cancel_right]
    rw [pow_succ' (x + y), add_mul, sum_range_succ_comm, add_mul, f_last, add_assoc,
      (((Commute.refl x).add_right h).pow_right n).eq, sum_congr rfl f_succ, ← mul_sum,
      pow_succ' y, mul_assoc, ← mul_add y, ih]


@[simp]
theorem neg_one_geom_sum [Ring α] {n : ℕ} :
    ∑ i ∈ range n, (-1 : α) ^ i = if Even n then 0 else 1 := by
  induction n with
  | zero => simp
  | succ k hk =>
    simp only [geom_sum_succ', Nat.even_add_one, hk]
    split_ifs with h
    · rw [h.neg_one_pow, add_zero]
    · rw [(Nat.not_even_iff_odd.1 h).neg_one_pow, neg_add_cancel]


theorem geom_sum₂_self {α : Type*} [CommRing α] (x : α) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * x ^ (n - 1 - i) = n * x ^ (n - 1) :=
  calc
    ∑ i ∈ Finset.range n, x ^ i * x ^ (n - 1 - i) =
        ∑ i ∈ Finset.range n, x ^ (i + (n - 1 - i)) := by
      /-
        α : Type u_1
        inst✝ : CommRing α
        x : α
        n : Nat
        ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow x (HS …
      -/
      simp_rw [← pow_add]
      /-
        🎉 no goals
      -/
    _ = ∑ _i ∈ Finset.range n, x ^ (n - 1) :=
      Finset.sum_congr rfl fun _ hi =>
        congr_arg _ <| add_tsub_cancel_of_le <| Nat.le_sub_one_of_lt <| Finset.mem_range.1 hi
    _ = #(range n) • x ^ (n - 1) := sum_const _
                              /-
                                α : Type u_1
                                inst✝ : CommRing α
                                x : α
                                n : Nat
                                ⊢ Eq (HSMul.hSMul (Finset.range n).card (HPow.hPow x (HSub.hSub n 1))) (HMul.h …
                              -/
    _ = n * x ^ (n - 1) := by rw [Finset.card_range, nsmul_eq_mul]
                              /-
                                🎉 no goals
                              -/


/-- $x^n-y^n = (x-y) \sum x^ky^{n-1-k}$ reformulated without `-` signs. -/
theorem geom_sum₂_mul_add [CommSemiring α] (x y : α) (n : ℕ) :
    (∑ i ∈ range n, (x + y) ^ i * y ^ (n - 1 - i)) * x + y ^ n = (x + y) ^ n :=
  (Commute.all x y).geom_sum₂_mul_add n


theorem geom_sum_mul_add [Semiring α] (x : α) (n : ℕ) :
    (∑ i ∈ range n, (x + 1) ^ i) * x + 1 = (x + 1) ^ n := by
  /-
    α : Type u
    inst✝ : Semiring α
    x : α
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow (HAdd.hAdd …
  -/
  have := (Commute.one_right x).geom_sum₂_mul_add n
  /-
    α : Type u
    inst✝ : Semiring α
    x : α
    n : Nat
    this : Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow (HAdd.hAdd …
  -/
  rw [one_pow, geom_sum₂_with_one] at this
  /-
    α : Type u
    inst✝ : Semiring α
    x : α
    n : Nat
    this : Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow (HAdd …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow (HAdd.hAdd …
  -/
  exact this
  /-
    🎉 no goals
  -/


protected theorem Commute.geom_sum₂_mul [Ring α] {x y : α} (h : Commute x y) (n : ℕ) :
    (∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) * (x - y) = x ^ n - y ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  have := (h.sub_left (Commute.refl y)).geom_sum₂_mul_add n
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    this : Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  rw [sub_add_cancel] at this
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    this : Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  rw [← this, add_sub_cancel_right]
  /-
    🎉 no goals
  -/


theorem Commute.mul_neg_geom_sum₂ [Ring α] {x y : α} (h : Commute x y) (n : ℕ) :
    ((y - x) * ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) = y ^ n - x ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub y x) ((Finset.range n).sum fun i => HMul.hMul (HPow …
  -/
  apply op_injective
  /-
    case a
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (MulOpposite.op (HMul.hMul (HSub.hSub y x) ((Finset.range n).sum fun i => …
  -/
  simp only [op_mul, op_sub, op_geom_sum₂, op_pow]
  /-
    case a
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (HMul.hMul (MulOpposite.op ((Finset.range n).sum fun i => HMul.hMul (HPow …
  -/
  simp [(Commute.op h.symm).geom_sum₂_mul n]
  /-
    🎉 no goals
  -/


theorem Commute.mul_geom_sum₂ [Ring α] {x y : α} (h : Commute x y) (n : ℕ) :
    ((x - y) * ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) = x ^ n - y ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq (HMul.hMul (HSub.hSub x y) ((Finset.range n).sum fun i => HMul.hMul (HPow …
  -/
  rw [← neg_sub (y ^ n), ← h.mul_neg_geom_sum₂, ← neg_mul, neg_sub]
  /-
    🎉 no goals
  -/


theorem geom_sum₂_mul [CommRing α] (x y : α) (n : ℕ) :
    (∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) * (x - y) = x ^ n - y ^ n :=
  (Commute.all x y).geom_sum₂_mul n


theorem geom_sum₂_mul_of_ge [CommSemiring α] [PartialOrder α] [AddLeftReflectLE α] [AddLeftMono α]
    [ExistsAddOfLE α] [Sub α] [OrderedSub α] {x y : α} (hxy : y ≤ x) (n : ℕ) :
    (∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) * (x - y) = x ^ n - y ^ n := by
  /-
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le y x
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  apply eq_tsub_of_add_eq
  /-
    case h
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le y x
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow …
  -/
  simpa only [tsub_add_cancel_of_le hxy] using geom_sum₂_mul_add (x - y) y n
  /-
    🎉 no goals
  -/


theorem geom_sum₂_mul_of_le [CommSemiring α] [PartialOrder α] [AddLeftReflectLE α] [AddLeftMono α]
    [ExistsAddOfLE α] [Sub α] [OrderedSub α] {x y : α} (hxy : x ≤ y) (n : ℕ) :
    (∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) * (y - x) = y ^ n - x ^ n := by
  /-
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  rw [← Finset.sum_range_reflect]
  /-
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun j => HMul.hMul (HPow.hPow x (HSub.hS …
  -/
  convert geom_sum₂_mul_of_ge hxy n using 3
  /-
    case h.e'_2.h.e'_5.a
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n x✝ : Nat
    a✝ : Membership.mem (Finset.range n) x✝
    ⊢ Eq (HMul.hMul (HPow.hPow x (HSub.hSub (HSub.hSub n 1) x✝)) (HPow.hPow y (HSu …
  -/
  simp_all only [Finset.mem_range]
  /-
    case h.e'_2.h.e'_5.a
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n x✝ : Nat
    a✝ : LT.lt x✝ n
    ⊢ Eq (HMul.hMul (HPow.hPow x (HSub.hSub (HSub.hSub n 1) x✝)) (HPow.hPow y (HSu …
  -/
  rw [mul_comm]
  /-
    case h.e'_2.h.e'_5.a
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n x✝ : Nat
    a✝ : LT.lt x✝ n
    ⊢ Eq (HMul.hMul (HPow.hPow y (HSub.hSub (HSub.hSub n 1) (HSub.hSub (HSub.hSub  …
  -/
  congr
  /-
    case h.e'_2.h.e'_5.a.e_a.e_a
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x y : α
    hxy : LE.le x y
    n x✝ : Nat
    a✝ : LT.lt x✝ n
    ⊢ Eq (HSub.hSub (HSub.hSub n 1) (HSub.hSub (HSub.hSub n 1) x✝)) x✝
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Commute.sub_dvd_pow_sub_pow [Ring α] {x y : α} (h : Commute x y) (n : ℕ) :
    x - y ∣ x ^ n - y ^ n :=
  Dvd.intro _ <| h.mul_geom_sum₂ _


theorem sub_dvd_pow_sub_pow [CommRing α] (x y : α) (n : ℕ) : x - y ∣ x ^ n - y ^ n :=
  (Commute.all x y).sub_dvd_pow_sub_pow n


theorem nat_sub_dvd_pow_sub_pow (x y n : ℕ) : x - y ∣ x ^ n - y ^ n := by
  /-
    x y n : Nat
    ⊢ Dvd.dvd (HSub.hSub x y) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
  -/
  rcases le_or_lt y x with h | h
    /-
      case inl
      x y n : Nat
      h : LE.le y x
      ⊢ Dvd.dvd (HSub.hSub x y) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
  · have : y ^ n ≤ x ^ n := Nat.pow_le_pow_left h _
    /-
      case inl
      x y n : Nat
      h : LE.le y x
      this : LE.le (HPow.hPow y n) (HPow.hPow x n)
      ⊢ Dvd.dvd (HSub.hSub x y) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
    exact mod_cast sub_dvd_pow_sub_pow (x : ℤ) (↑y) n
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y n : Nat
      h : LT.lt x y
      ⊢ Dvd.dvd (HSub.hSub x y) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
  · have : x ^ n ≤ y ^ n := Nat.pow_le_pow_left h.le _
    /-
      case inr
      x y n : Nat
      h : LT.lt x y
      this : LE.le (HPow.hPow x n) (HPow.hPow y n)
      ⊢ Dvd.dvd (HSub.hSub x y) (HSub.hSub (HPow.hPow x n) (HPow.hPow y n))
    -/
    exact (Nat.sub_eq_zero_of_le this).symm ▸ dvd_zero (x - y)
    /-
      🎉 no goals
    -/


theorem one_sub_dvd_one_sub_pow [Ring α] (x : α) (n : ℕ) :
    1 - x ∣ 1 - x ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Dvd.dvd (HSub.hSub 1 x) (HSub.hSub 1 (HPow.hPow x n))
  -/
  conv_rhs => rw [← one_pow n]
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Dvd.dvd (HSub.hSub 1 x) (HSub.hSub (HPow.hPow 1 n) (HPow.hPow x n))
  -/
  exact (Commute.one_left x).sub_dvd_pow_sub_pow n
  /-
    🎉 no goals
  -/


theorem sub_one_dvd_pow_sub_one [Ring α] (x : α) (n : ℕ) :
    x - 1 ∣ x ^ n - 1 := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Dvd.dvd (HSub.hSub x 1) (HSub.hSub (HPow.hPow x n) 1)
  -/
  conv_rhs => rw [← one_pow n]
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Dvd.dvd (HSub.hSub x 1) (HSub.hSub (HPow.hPow x n) (HPow.hPow 1 n))
  -/
  exact (Commute.one_right x).sub_dvd_pow_sub_pow n
  /-
    🎉 no goals
  -/


lemma pow_one_sub_dvd_pow_mul_sub_one [Ring α] (x : α) (m n : ℕ) :
    ((x ^ m) - 1 : α) ∣ (x ^ (m * n) - 1) := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    m n : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow x m) 1) (HSub.hSub (HPow.hPow x (HMul.hMul m n …
  -/
  rw [npow_mul]
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    m n : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow x m) 1) (HSub.hSub (HPow.hPow (HPow.hPow x m)  …
  -/
  exact sub_one_dvd_pow_sub_one (x := x ^ m) (n := n)
  /-
    🎉 no goals
  -/


lemma nat_pow_one_sub_dvd_pow_mul_sub_one (x m n : ℕ) : x ^ m - 1 ∣ x ^ (m * n) - 1 := by
  /-
    x m n : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow x m) 1) (HSub.hSub (HPow.hPow x (HMul.hMul m n …
  -/
  nth_rw 2 [← Nat.one_pow n]
  /-
    x m n : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow x m) 1) (HSub.hSub (HPow.hPow x (HMul.hMul m n …
  -/
  rw [Nat.pow_mul x m n]
  /-
    x m n : Nat
    ⊢ Dvd.dvd (HSub.hSub (HPow.hPow x m) 1) (HSub.hSub (HPow.hPow (HPow.hPow x m)  …
  -/
  apply nat_sub_dvd_pow_sub_pow (x ^ m) 1
  /-
    🎉 no goals
  -/


theorem Odd.add_dvd_pow_add_pow [CommRing α] (x y : α) {n : ℕ} (h : Odd n) :
    x + y ∣ x ^ n + y ^ n := by
  /-
    α : Type u
    inst✝ : CommRing α
    x y : α
    n : Nat
    h : Odd n
    ⊢ Dvd.dvd (HAdd.hAdd x y) (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))
  -/
  have h₁ := geom_sum₂_mul x (-y) n
  /-
    α : Type u
    inst✝ : CommRing α
    x y : α
    n : Nat
    h : Odd n
    h₁ : Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (H …
    ⊢ Dvd.dvd (HAdd.hAdd x y) (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))
  -/
  rw [Odd.neg_pow h y, sub_neg_eq_add, sub_neg_eq_add] at h₁
  /-
    α : Type u
    inst✝ : CommRing α
    x y : α
    n : Nat
    h : Odd n
    h₁ : Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (H …
    ⊢ Dvd.dvd (HAdd.hAdd x y) (HAdd.hAdd (HPow.hPow x n) (HPow.hPow y n))
  -/
  exact Dvd.intro_left _ h₁
  /-
    🎉 no goals
  -/


theorem Odd.nat_add_dvd_pow_add_pow (x y : ℕ) {n : ℕ} (h : Odd n) : x + y ∣ x ^ n + y ^ n :=
  mod_cast Odd.add_dvd_pow_add_pow (x : ℤ) (↑y) h


theorem geom_sum_mul [Ring α] (x : α) (n : ℕ) : (∑ i ∈ range n, x ^ i) * (x - 1) = x ^ n - 1 := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub x 1)) …
  -/
  have := (Commute.one_right x).geom_sum₂_mul n
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    this : Eq (HMul.hMul ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i)  …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub x 1)) …
  -/
  rw [one_pow, geom_sum₂_with_one] at this
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    this : Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub  …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub x 1)) …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem geom_sum_mul_of_one_le [CommSemiring α] [PartialOrder α] [AddLeftReflectLE α]
    [AddLeftMono α] [ExistsAddOfLE α] [Sub α] [OrderedSub α] {x : α} (hx : 1 ≤ x) (n : ℕ) :
    (∑ i ∈ range n, x ^ i) * (x - 1) = x ^ n - 1 := by
  /-
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x : α
    hx : LE.le 1 x
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub x 1)) …
  -/
  simpa using geom_sum₂_mul_of_ge hx n
  /-
    🎉 no goals
  -/


theorem geom_sum_mul_of_le_one [CommSemiring α] [PartialOrder α] [AddLeftReflectLE α]
    [AddLeftMono α] [ExistsAddOfLE α] [Sub α] [OrderedSub α] {x : α} (hx : x ≤ 1) (n : ℕ) :
    (∑ i ∈ range n, x ^ i) * (1 - x) = 1 - x ^ n := by
  /-
    α : Type u
    inst✝⁶ : CommSemiring α
    inst✝⁵ : PartialOrder α
    inst✝⁴ : AddLeftReflectLE α
    inst✝³ : AddLeftMono α
    inst✝² : ExistsAddOfLE α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    x : α
    hx : LE.le x 1
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub 1 x)) …
  -/
  simpa using geom_sum₂_mul_of_le hx n
  /-
    🎉 no goals
  -/


theorem mul_geom_sum [Ring α] (x : α) (n : ℕ) : ((x - 1) * ∑ i ∈ range n, x ^ i) = x ^ n - 1 :=
                     /-
                       α : Type u
                       inst✝ : Ring α
                       x : α
                       n : Nat
                       ⊢ Eq (MulOpposite.op (HMul.hMul (HSub.hSub x 1) ((Finset.range n).sum fun i => …
                     -/
  op_injective <| by simpa using geom_sum_mul (op x) n
                     /-
                       🎉 no goals
                     -/


theorem geom_sum_mul_neg [Ring α] (x : α) (n : ℕ) :
    (∑ i ∈ range n, x ^ i) * (1 - x) = 1 - x ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub 1 x)) …
  -/
  have := congr_arg Neg.neg (geom_sum_mul x n)
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    this : Eq (Neg.neg (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (H …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub 1 x)) …
  -/
  rw [neg_sub, ← mul_neg, neg_sub] at this
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    n : Nat
    this : Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub  …
    ⊢ Eq (HMul.hMul ((Finset.range n).sum fun i => HPow.hPow x i) (HSub.hSub 1 x)) …
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem mul_neg_geom_sum [Ring α] (x : α) (n : ℕ) : ((1 - x) * ∑ i ∈ range n, x ^ i) = 1 - x ^ n :=
                     /-
                       α : Type u
                       inst✝ : Ring α
                       x : α
                       n : Nat
                       ⊢ Eq (MulOpposite.op (HMul.hMul (HSub.hSub 1 x) ((Finset.range n).sum fun i => …
                     -/
  op_injective <| by simpa using geom_sum_mul_neg (op x) n
                     /-
                       🎉 no goals
                     -/


protected theorem Commute.geom_sum₂_comm {α : Type u} [Semiring α] {x y : α} (n : ℕ)
    (h : Commute x y) :
    ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = ∑ i ∈ range n, y ^ i * x ^ (n - 1 - i) := by
  /-
    α : Type u
    inst✝ : Semiring α
    x y : α
    n : Nat
    h : Commute x y
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  cases n; · simp
             /-
               🎉 no goals
             -/
  /-
    case succ
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n✝ : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n✝ 1)).sum fun i => HMul.hMul (HPow.hPow x i) ( …
  -/
  simp only [Nat.succ_eq_add_one, Nat.add_sub_cancel]
  /-
    case succ
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n✝ : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n✝ 1)).sum fun x_1 => HMul.hMul (HPow.hPow x x_ …
  -/
  rw [← Finset.sum_flip]
  /-
    case succ
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n✝ : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n✝ 1)).sum fun r => HMul.hMul (HPow.hPow x (HSu …
  -/
  refine Finset.sum_congr rfl fun i hi => ?_
  /-
    case succ
    α : Type u
    inst✝ : Semiring α
    x y : α
    h : Commute x y
    n✝ i : Nat
    hi : Membership.mem (Finset.range (HAdd.hAdd n✝ 1)) i
    ⊢ Eq (HMul.hMul (HPow.hPow x (HSub.hSub n✝ i)) (HPow.hPow y (HSub.hSub n✝ (HSu …
  -/
  simpa [Nat.sub_sub_self (Nat.succ_le_succ_iff.mp (Finset.mem_range.mp hi))] using h.pow_pow _ _
  /-
    🎉 no goals
  -/


theorem geom_sum₂_comm {α : Type u} [CommSemiring α] (x y : α) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = ∑ i ∈ range n, y ^ i * x ^ (n - 1 - i) :=
  (Commute.all x y).geom_sum₂_comm n


protected theorem Commute.geom_sum₂ [DivisionRing α] {x y : α} (h' : Commute x y) (h : x ≠ y)
    (n : ℕ) : ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = (x ^ n - y ^ n) / (x - y) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x y : α
    h' : Commute x y
    h : Ne x y
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  have : x - y ≠ 0 := by simp_all [sub_eq_iff_eq_add]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x y : α
    h' : Commute x y
    h : Ne x y
    n : Nat
    this : Ne (HSub.hSub x y) 0
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  rw [← h'.geom_sum₂_mul, mul_div_cancel_right₀ _ this]
  /-
    🎉 no goals
  -/


theorem geom₂_sum [Field α] {x y : α} (h : x ≠ y) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = (x ^ n - y ^ n) / (x - y) :=
  (Commute.all x y).geom_sum₂ h n


theorem geom₂_sum_of_gt {α : Type*} [CanonicallyLinearOrderedSemifield α] [Sub α] [OrderedSub α]
    {x y : α} (h : y < x) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = (x ^ n - y ^ n) / (x - y) :=
  eq_div_of_mul_eq (tsub_pos_of_lt h).ne' (geom_sum₂_mul_of_ge h.le n)


theorem geom₂_sum_of_lt {α : Type*} [CanonicallyLinearOrderedSemifield α] [Sub α] [OrderedSub α]
    {x y : α} (h : x < y) (n : ℕ) :
    ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) = (y ^ n - x ^ n) / (y - x) :=
  eq_div_of_mul_eq (tsub_pos_of_lt h).ne' (geom_sum₂_mul_of_le h.le n)


theorem geom_sum_eq [DivisionRing α] {x : α} (h : x ≠ 1) (n : ℕ) :
    ∑ i ∈ range n, x ^ i = (x ^ n - 1) / (x - 1) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    h : Ne x 1
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HSub.hSub (HPow …
  -/
  have : x - 1 ≠ 0 := by simp_all [sub_eq_iff_eq_add]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    h : Ne x 1
    n : Nat
    this : Ne (HSub.hSub x 1) 0
    ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HSub.hSub (HPow …
  -/
  rw [← geom_sum_mul, mul_div_cancel_right₀ _ this]
  /-
    🎉 no goals
  -/


lemma geom_sum_of_one_lt {x : α} [CanonicallyLinearOrderedSemifield α] [Sub α] [OrderedSub α]
    (h : 1 < x) (n : ℕ) :
    ∑ i ∈ Finset.range n, x ^ i = (x ^ n - 1) / (x - 1) :=
  eq_div_of_mul_eq (tsub_pos_of_lt h).ne' (geom_sum_mul_of_one_le h.le n)


lemma geom_sum_of_lt_one {x : α} [CanonicallyLinearOrderedSemifield α] [Sub α] [OrderedSub α]
    (h : x < 1) (n : ℕ) :
    ∑ i ∈ Finset.range n, x ^ i = (1 - x ^ n) / (1 - x) :=
  eq_div_of_mul_eq (tsub_pos_of_lt h).ne' (geom_sum_mul_of_le_one h.le n)


theorem geom_sum_lt {x : α} [CanonicallyLinearOrderedSemifield α] [Sub α] [OrderedSub α]
    (h0 : x ≠ 0) (h1 : x < 1) (n : ℕ) : ∑ i ∈ range n, x ^ i < (1 - x)⁻¹ := by
  /-
    α : Type u
    x : α
    inst✝² : CanonicallyLinearOrderedSemifield α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    h0 : Ne x 0
    h1 : LT.lt x 1
    n : Nat
    ⊢ LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) (Inv.inv (HSub.hSub 1 x))
  -/
  rw [← zero_lt_iff] at h0
  /-
    α : Type u
    x : α
    inst✝² : CanonicallyLinearOrderedSemifield α
    inst✝¹ : Sub α
    inst✝ : OrderedSub α
    h0 : LT.lt 0 x
    h1 : LT.lt x 1
    n : Nat
    ⊢ LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) (Inv.inv (HSub.hSub 1 x))
  -/
  rw [geom_sum_of_lt_one h1, div_lt_iff₀, inv_mul_cancel₀, tsub_lt_self_iff]
    /-
      α : Type u
      x : α
      inst✝² : CanonicallyLinearOrderedSemifield α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      h0 : LT.lt 0 x
      h1 : LT.lt x 1
      n : Nat
      ⊢ And (LT.lt 0 1) (LT.lt 0 (HPow.hPow x n))
    -/
  · exact ⟨h0.trans h1, pow_pos h0 n⟩
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      x : α
      inst✝² : CanonicallyLinearOrderedSemifield α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      h0 : LT.lt 0 x
      h1 : LT.lt x 1
      n : Nat
      ⊢ Ne (HSub.hSub 1 x) 0
    -/
  · rwa [ne_eq, tsub_eq_zero_iff_le, not_le]
    /-
      🎉 no goals
    -/
    /-
      α : Type u
      x : α
      inst✝² : CanonicallyLinearOrderedSemifield α
      inst✝¹ : Sub α
      inst✝ : OrderedSub α
      h0 : LT.lt 0 x
      h1 : LT.lt x 1
      n : Nat
      ⊢ LT.lt 0 (HSub.hSub 1 x)
    -/
  · rwa [tsub_pos_iff_lt]
    /-
      🎉 no goals
    -/


protected theorem Commute.mul_geom_sum₂_Ico [Ring α] {x y : α} (h : Commute x y) {m n : ℕ}
    (hmn : m ≤ n) :
    ((x - y) * ∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) = x ^ n - x ^ m * y ^ (n - m) := by
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (HMul.hMul (HSub.hSub x y) ((Finset.Ico m n).sum fun i => HMul.hMul (HPow …
  -/
  rw [sum_Ico_eq_sub _ hmn]
  have :
    ∑ k ∈ range m, x ^ k * y ^ (n - 1 - k) =
      ∑ k ∈ range m, x ^ k * (y ^ (n - m) * y ^ (m - 1 - k)) := by
    refine sum_congr rfl fun j j_in => ?_
    rw [← pow_add]
    congr
    rw [mem_range] at j_in
    omega
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    this : Eq ((Finset.range m).sum fun k => HMul.hMul (HPow.hPow x k) (HPow.hPow  …
    ⊢ Eq (HMul.hMul (HSub.hSub x y) (HSub.hSub ((Finset.range n).sum fun k => HMul …
  -/
  rw [this]
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    this : Eq ((Finset.range m).sum fun k => HMul.hMul (HPow.hPow x k) (HPow.hPow  …
    ⊢ Eq (HMul.hMul (HSub.hSub x y) (HSub.hSub ((Finset.range n).sum fun k => HMul …
  -/
  simp_rw [pow_mul_comm y (n - m) _]
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    this : Eq ((Finset.range m).sum fun k => HMul.hMul (HPow.hPow x k) (HPow.hPow  …
    ⊢ Eq (HMul.hMul (HSub.hSub x y) (HSub.hSub ((Finset.range n).sum fun k => HMul …
  -/
  simp_rw [← mul_assoc]
  rw [← sum_mul, mul_sub, h.mul_geom_sum₂, ← mul_assoc, h.mul_geom_sum₂, sub_mul, ← pow_add,
    add_tsub_cancel_of_le hmn, sub_sub_sub_cancel_right (x ^ n) (x ^ m * y ^ (n - m)) (y ^ n)]


protected theorem Commute.geom_sum₂_succ_eq {α : Type u} [Ring α] {x y : α} (h : Commute x y)
    {n : ℕ} :
    ∑ i ∈ range (n + 1), x ^ i * y ^ (n - i) =
      x ^ n + y * ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) := by
  simp_rw [mul_sum, sum_range_succ_comm, tsub_self, pow_zero, mul_one, add_right_inj, ← mul_assoc,
    (h.symm.pow_right _).eq, mul_assoc, ← pow_succ']
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  refine sum_congr rfl fun i hi => ?_
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n i : Nat
    hi : Membership.mem (Finset.range n) i
    ⊢ Eq (HMul.hMul (HPow.hPow x i) (HPow.hPow y (HSub.hSub n i))) (HMul.hMul (HPo …
  -/
  suffices n - 1 - i + 1 = n - i by rw [this]
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n i : Nat
    hi : Membership.mem (Finset.range n) i
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub n 1) i) 1) (HSub.hSub n i)
  -/
  rw [Finset.mem_range] at hi
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    n i : Nat
    hi : LT.lt i n
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub n 1) i) 1) (HSub.hSub n i)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem geom_sum₂_succ_eq {α : Type u} [CommRing α] (x y : α) {n : ℕ} :
    ∑ i ∈ range (n + 1), x ^ i * y ^ (n - i) =
      x ^ n + y * ∑ i ∈ range n, x ^ i * y ^ (n - 1 - i) :=
  (Commute.all x y).geom_sum₂_succ_eq


theorem mul_geom_sum₂_Ico [CommRing α] (x y : α) {m n : ℕ} (hmn : m ≤ n) :
    ((x - y) * ∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) = x ^ n - x ^ m * y ^ (n - m) :=
  (Commute.all x y).mul_geom_sum₂_Ico hmn


protected theorem Commute.geom_sum₂_Ico_mul [Ring α] {x y : α} (h : Commute x y) {m n : ℕ}
    (hmn : m ≤ n) :
    (∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) * (x - y) = x ^ n - y ^ (n - m) * x ^ m := by
  /-
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (HMul.hMul ((Finset.Ico m n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow …
  -/
  apply op_injective
  /-
    case a
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (MulOpposite.op (HMul.hMul ((Finset.Ico m n).sum fun i => HMul.hMul (HPow …
  -/
  simp only [op_sub, op_mul, op_pow, op_sum]
  have : (∑ k ∈ Ico m n, MulOpposite.op y ^ (n - 1 - k) * MulOpposite.op x ^ k) =
      ∑ k ∈ Ico m n, MulOpposite.op x ^ k * MulOpposite.op y ^ (n - 1 - k) := by
    refine sum_congr rfl fun k _ => ?_
    have hp := Commute.pow_pow (Commute.op h.symm) (n - 1 - k) k
    simpa [Commute, SemiconjBy] using hp
  /-
    case a
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    this : Eq ((Finset.Ico m n).sum fun k => HMul.hMul (HPow.hPow (MulOpposite.op  …
    ⊢ Eq (HMul.hMul (HSub.hSub (MulOpposite.op x) (MulOpposite.op y)) ((Finset.Ico …
  -/
  simp only [this]
  /-
    case a
    α : Type u
    inst✝ : Ring α
    x y : α
    h : Commute x y
    m n : Nat
    hmn : LE.le m n
    this : Eq ((Finset.Ico m n).sum fun k => HMul.hMul (HPow.hPow (MulOpposite.op  …
    ⊢ Eq (HMul.hMul (HSub.hSub (MulOpposite.op x) (MulOpposite.op y)) ((Finset.Ico …
  -/
  convert (Commute.op h).mul_geom_sum₂_Ico hmn
  /-
    🎉 no goals
  -/


theorem geom_sum_Ico_mul [Ring α] (x : α) {m n : ℕ} (hmn : m ≤ n) :
    (∑ i ∈ Finset.Ico m n, x ^ i) * (x - 1) = x ^ n - x ^ m := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (HMul.hMul ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HSub.hSub x 1)) …
  -/
  rw [sum_Ico_eq_sub _ hmn, sub_mul, geom_sum_mul, geom_sum_mul, sub_sub_sub_cancel_right]
  /-
    🎉 no goals
  -/


theorem geom_sum_Ico_mul_neg [Ring α] (x : α) {m n : ℕ} (hmn : m ≤ n) :
    (∑ i ∈ Finset.Ico m n, x ^ i) * (1 - x) = x ^ m - x ^ n := by
  /-
    α : Type u
    inst✝ : Ring α
    x : α
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (HMul.hMul ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HSub.hSub 1 x)) …
  -/
  rw [sum_Ico_eq_sub _ hmn, sub_mul, geom_sum_mul_neg, geom_sum_mul_neg, sub_sub_sub_cancel_left]
  /-
    🎉 no goals
  -/


protected theorem Commute.geom_sum₂_Ico [DivisionRing α] {x y : α} (h : Commute x y) (hxy : x ≠ y)
    {m n : ℕ} (hmn : m ≤ n) :
    (∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) = (x ^ n - y ^ (n - m) * x ^ m) / (x - y) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x y : α
    h : Commute x y
    hxy : Ne x y
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((Finset.Ico m n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  have : x - y ≠ 0 := by simp_all [sub_eq_iff_eq_add]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x y : α
    h : Commute x y
    hxy : Ne x y
    m n : Nat
    hmn : LE.le m n
    this : Ne (HSub.hSub x y) 0
    ⊢ Eq ((Finset.Ico m n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y (HS …
  -/
  rw [← h.geom_sum₂_Ico_mul hmn, mul_div_cancel_right₀ _ this]
  /-
    🎉 no goals
  -/


theorem geom_sum₂_Ico [Field α] {x y : α} (hxy : x ≠ y) {m n : ℕ} (hmn : m ≤ n) :
    (∑ i ∈ Finset.Ico m n, x ^ i * y ^ (n - 1 - i)) = (x ^ n - y ^ (n - m) * x ^ m) / (x - y) :=
  (Commute.all x y).geom_sum₂_Ico hxy hmn


theorem geom_sum_Ico [DivisionRing α] {x : α} (hx : x ≠ 1) {m n : ℕ} (hmn : m ≤ n) :
    ∑ i ∈ Finset.Ico m n, x ^ i = (x ^ n - x ^ m) / (x - 1) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx : Ne x 1
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HSub.hSub (HPow …
  -/
  simp only [sum_Ico_eq_sub _ hmn, geom_sum_eq hx, div_sub_div_same, sub_sub_sub_cancel_right]
  /-
    🎉 no goals
  -/


theorem geom_sum_Ico' [DivisionRing α] {x : α} (hx : x ≠ 1) {m n : ℕ} (hmn : m ≤ n) :
    ∑ i ∈ Finset.Ico m n, x ^ i = (x ^ m - x ^ n) / (1 - x) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx : Ne x 1
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HSub.hSub (HPow …
  -/
  simp only [geom_sum_Ico hx hmn]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx : Ne x 1
    m n : Nat
    hmn : LE.le m n
    ⊢ Eq (HDiv.hDiv (HSub.hSub (HPow.hPow x n) (HPow.hPow x m)) (HSub.hSub x 1)) ( …
  -/
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  convert neg_div_neg_eq (x ^ m - x ^ n) (1 - x) using 2 <;> abel
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem geom_sum_Ico_le_of_lt_one [LinearOrderedField α] {x : α} (hx : 0 ≤ x) (h'x : x < 1)
    {m n : ℕ} : ∑ i ∈ Ico m n, x ^ i ≤ x ^ m / (1 - x) := by
  /-
    α : Type u
    inst✝ : LinearOrderedField α
    x : α
    hx : LE.le 0 x
    h'x : LT.lt x 1
    m n : Nat
    ⊢ LE.le ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HPow.hPow x  …
  -/
  rcases le_or_lt m n with (hmn | hmn)
    /-
      case inl
      α : Type u
      inst✝ : LinearOrderedField α
      x : α
      hx : LE.le 0 x
      h'x : LT.lt x 1
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HPow.hPow x  …
    -/
  · rw [geom_sum_Ico' h'x.ne hmn]
    /-
      case inl
      α : Type u
      inst✝ : LinearOrderedField α
      x : α
      hx : LE.le 0 x
      h'x : LT.lt x 1
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le (HDiv.hDiv (HSub.hSub (HPow.hPow x m) (HPow.hPow x n)) (HSub.hSub 1 x) …
    -/
    apply div_le_div₀ (pow_nonneg hx _) _ (sub_pos.2 h'x) le_rfl
    /-
      α : Type u
      inst✝ : LinearOrderedField α
      x : α
      hx : LE.le 0 x
      h'x : LT.lt x 1
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le (HSub.hSub (HPow.hPow x m) (HPow.hPow x n)) (HPow.hPow x m)
    -/
    simpa using pow_nonneg hx _
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : LinearOrderedField α
      x : α
      hx : LE.le 0 x
      h'x : LT.lt x 1
      m n : Nat
      hmn : LT.lt n m
      ⊢ LE.le ((Finset.Ico m n).sum fun i => HPow.hPow x i) (HDiv.hDiv (HPow.hPow x  …
    -/
  · rw [Ico_eq_empty, sum_empty]
      /-
        case inr
        α : Type u
        inst✝ : LinearOrderedField α
        x : α
        hx : LE.le 0 x
        h'x : LT.lt x 1
        m n : Nat
        hmn : LT.lt n m
        ⊢ LE.le 0 (HDiv.hDiv (HPow.hPow x m) (HSub.hSub 1 x))
      -/
    · apply div_nonneg (pow_nonneg hx _)
      /-
        case inr
        α : Type u
        inst✝ : LinearOrderedField α
        x : α
        hx : LE.le 0 x
        h'x : LT.lt x 1
        m n : Nat
        hmn : LT.lt n m
        ⊢ LE.le 0 (HSub.hSub 1 x)
      -/
      simpa using h'x.le
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        inst✝ : LinearOrderedField α
        x : α
        hx : LE.le 0 x
        h'x : LT.lt x 1
        m n : Nat
        hmn : LT.lt n m
        ⊢ Not (LT.lt m n)
      -/
    · simpa using hmn.le
      /-
        🎉 no goals
      -/


theorem geom_sum_inv [DivisionRing α] {x : α} (hx1 : x ≠ 1) (hx0 : x ≠ 0) (n : ℕ) :
    ∑ i ∈ range n, x⁻¹ ^ i = (x - 1)⁻¹ * (x - x⁻¹ ^ n * x) := by
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx1 : Ne x 1
    hx0 : Ne x 0
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow (Inv.inv x) i) (HMul.hMul (Inv.i …
  -/
  have h₁ : x⁻¹ ≠ 1 := by rwa [inv_eq_one_div, Ne, div_eq_iff_mul_eq hx0, one_mul]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx1 : Ne x 1
    hx0 : Ne x 0
    n : Nat
    h₁ : Ne (Inv.inv x) 1
    ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow (Inv.inv x) i) (HMul.hMul (Inv.i …
  -/
  have h₂ : x⁻¹ - 1 ≠ 0 := mt sub_eq_zero.1 h₁
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx1 : Ne x 1
    hx0 : Ne x 0
    n : Nat
    h₁ : Ne (Inv.inv x) 1
    h₂ : Ne (HSub.hSub (Inv.inv x) 1) 0
    ⊢ Eq ((Finset.range n).sum fun i => HPow.hPow (Inv.inv x) i) (HMul.hMul (Inv.i …
  -/
  have h₃ : x - 1 ≠ 0 := mt sub_eq_zero.1 hx1
  have h₄ : x * (x ^ n)⁻¹ = (x ^ n)⁻¹ * x :=
    Nat.recOn n (by simp) fun n h => by
      rw [pow_succ', mul_inv_rev, ← mul_assoc, h, mul_assoc, mul_inv_cancel₀ hx0, mul_assoc,
        inv_mul_cancel₀ hx0]
  rw [geom_sum_eq h₁, div_eq_iff_mul_eq h₂, ← mul_right_inj' h₃, ← mul_assoc, ← mul_assoc,
    mul_inv_cancel₀ h₃]
  simp [mul_add, add_mul, mul_inv_cancel₀ hx0, mul_assoc, h₄, sub_eq_add_neg, add_comm,
    add_left_comm]
  /-
    α : Type u
    inst✝ : DivisionRing α
    x : α
    hx1 : Ne x 1
    hx0 : Ne x 0
    n : Nat
    h₁ : Ne (Inv.inv x) 1
    h₂ : Ne (HSub.hSub (Inv.inv x) 1) 0
    h₃ : Ne (HSub.hSub x 1) 0
    h₄ : Eq (HMul.hMul x (Inv.inv (HPow.hPow x n))) (HMul.hMul (Inv.inv (HPow.hPow …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Inv.inv (HPow.hPow x n)) x) (Neg.neg x) …
  -/
  rw [add_comm _ (-x), add_assoc, add_assoc _ _ 1]
  /-
    🎉 no goals
  -/


theorem RingHom.map_geom_sum [Semiring α] [Semiring β] (x : α) (n : ℕ) (f : α →+* β) :
                                                            /-
                                                              α : Type u
                                                              β : Type u_1
                                                              inst✝¹ : Semiring α
                                                              inst✝ : Semiring β
                                                              x : α
                                                              n : Nat
                                                              f : RingHom α β
                                                              ⊢ Eq (f ((Finset.range n).sum fun i => HPow.hPow x i)) ((Finset.range n).sum f …
                                                            -/
    f (∑ i ∈ range n, x ^ i) = ∑ i ∈ range n, f x ^ i := by simp [map_sum f]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem RingHom.map_geom_sum₂ [Semiring α] [Semiring β] (x y : α) (n : ℕ) (f : α →+* β) :
    f (∑ i ∈ range n, x ^ i * y ^ (n - 1 - i)) = ∑ i ∈ range n, f x ^ i * f y ^ (n - 1 - i) := by
  /-
    α : Type u
    β : Type u_1
    inst✝¹ : Semiring α
    inst✝ : Semiring β
    x y : α
    n : Nat
    f : RingHom α β
    ⊢ Eq (f ((Finset.range n).sum fun i => HMul.hMul (HPow.hPow x i) (HPow.hPow y  …
  -/
  simp [map_sum f]
  /-
    🎉 no goals
  -/


theorem Nat.pred_mul_geom_sum_le (a b n : ℕ) :
    ((b - 1) * ∑ i ∈ range n.succ, a / b ^ i) ≤ a * b - a / b ^ n :=
  calc
    ((b - 1) * ∑ i ∈ range n.succ, a / b ^ i) =
    (∑ i ∈ range n, a / b ^ (i + 1) * b) + a * b - ((∑ i ∈ range n, a / b ^ i) + a / b ^ n) := by
      rw [tsub_mul, mul_comm, sum_mul, one_mul, sum_range_succ', sum_range_succ, pow_zero,
        Nat.div_one]
    _ ≤ (∑ i ∈ range n, a / b ^ i) + a * b - ((∑ i ∈ range n, a / b ^ i) + a / b ^ n) := by
      /-
        a b n : Nat
        ⊢ LE.le (HSub.hSub (HAdd.hAdd ((Finset.range n).sum fun i => HMul.hMul (HDiv.h …
      -/
      gcongr with i hi
      /-
        case h.bc.h
        a b n i : Nat
        hi : Membership.mem (Finset.range n) i
        ⊢ LE.le (HMul.hMul (HDiv.hDiv a (HPow.hPow b (HAdd.hAdd i 1))) b) (HDiv.hDiv a …
      -/
      rw [pow_succ, ← Nat.div_div_eq_div_mul]
      /-
        case h.bc.h
        a b n i : Nat
        hi : Membership.mem (Finset.range n) i
        ⊢ LE.le (HMul.hMul (HDiv.hDiv (HDiv.hDiv a (HPow.hPow b i)) b) b) (HDiv.hDiv a …
      -/
      exact Nat.div_mul_le_self _ _
      /-
        🎉 no goals
      -/
    _ = a * b - a / b ^ n := add_tsub_add_eq_tsub_left _ _ _


theorem Nat.geom_sum_le {b : ℕ} (hb : 2 ≤ b) (a n : ℕ) :
    ∑ i ∈ range n, a / b ^ i ≤ a * b / (b - 1) := by
  /-
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le ((Finset.range n).sum fun i => HDiv.hDiv a (HPow.hPow b i)) (HDiv.hDiv …
  -/
  refine (Nat.le_div_iff_mul_le <| tsub_pos_of_lt hb).2 ?_
  /-
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le (HMul.hMul ((Finset.range n).sum fun i => HDiv.hDiv a (HPow.hPow b i)) …
  -/
  rcases n with - | n
    /-
      case zero
      b : Nat
      hb : LE.le 2 b
      a : Nat
      ⊢ LE.le (HMul.hMul ((Finset.range 0).sum fun i => HDiv.hDiv a (HPow.hPow b i)) …
    -/
  · rw [sum_range_zero, zero_mul]
    /-
      case zero
      b : Nat
      hb : LE.le 2 b
      a : Nat
      ⊢ LE.le 0 (HMul.hMul a b)
    -/
    exact Nat.zero_le _
    /-
      🎉 no goals
    -/
  /-
    case succ
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le (HMul.hMul ((Finset.range (HAdd.hAdd n 1)).sum fun i => HDiv.hDiv a (H …
  -/
  rw [mul_comm]
  /-
    case succ
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le (HMul.hMul (HSub.hSub b 1) ((Finset.range (HAdd.hAdd n 1)).sum fun i = …
  -/
  exact (Nat.pred_mul_geom_sum_le a b n).trans tsub_le_self
  /-
    🎉 no goals
  -/


theorem Nat.geom_sum_Ico_le {b : ℕ} (hb : 2 ≤ b) (a n : ℕ) :
    ∑ i ∈ Ico 1 n, a / b ^ i ≤ a / (b - 1) := by
  /-
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le ((Finset.Ico 1 n).sum fun i => HDiv.hDiv a (HPow.hPow b i)) (HDiv.hDiv …
  -/
  rcases n with - | n
    /-
      case zero
      b : Nat
      hb : LE.le 2 b
      a : Nat
      ⊢ LE.le ((Finset.Ico 1 0).sum fun i => HDiv.hDiv a (HPow.hPow b i)) (HDiv.hDiv …
    -/
  · rw [Ico_eq_empty_of_le (zero_le_one' ℕ), sum_empty]
    /-
      case zero
      b : Nat
      hb : LE.le 2 b
      a : Nat
      ⊢ LE.le 0 (HDiv.hDiv a (HSub.hSub b 1))
    -/
    exact Nat.zero_le _
    /-
      🎉 no goals
    -/
  /-
    case succ
    b : Nat
    hb : LE.le 2 b
    a n : Nat
    ⊢ LE.le ((Finset.Ico 1 (HAdd.hAdd n 1)).sum fun i => HDiv.hDiv a (HPow.hPow b  …
  -/
  rw [← add_le_add_iff_left a]
  calc
    (a + ∑ i ∈ Ico 1 n.succ, a / b ^ i) = a / b ^ 0 + ∑ i ∈ Ico 1 n.succ, a / b ^ i := by
      rw [pow_zero, Nat.div_one]
    _ = ∑ i ∈ range n.succ, a / b ^ i := by
      rw [range_eq_Ico, ← Nat.Ico_insert_succ_left (Nat.succ_pos _), sum_insert]
      exact fun h => zero_lt_one.not_le (mem_Ico.1 h).1
    _ ≤ a * b / (b - 1) := Nat.geom_sum_le hb a _
    _ = (a * 1 + a * (b - 1)) / (b - 1) := by
      rw [← mul_add, add_tsub_cancel_of_le (one_le_two.trans hb)]
    _ = a + a / (b - 1) := by rw [mul_one, Nat.add_mul_div_right _ _ (tsub_pos_of_lt hb), add_comm]


theorem geom_sum_pos [StrictOrderedSemiring α] (hx : 0 ≤ x) (hn : n ≠ 0) :
    0 < ∑ i ∈ range n, x ^ i :=
                                                                      /-
                                                                        α : Type u
                                                                        n : Nat
                                                                        x : α
                                                                        inst✝ : StrictOrderedSemiring α
                                                                        hx : LE.le 0 x
                                                                        hn : Ne n 0
                                                                        ⊢ LT.lt 0 (HPow.hPow x 0)
                                                                      -/
  sum_pos' (fun _ _ => pow_nonneg hx _) ⟨0, mem_range.2 hn.bot_lt, by simp⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem geom_sum_pos_and_lt_one [StrictOrderedRing α] (hx : x < 0) (hx' : 0 < x + 1) (hn : 1 < n) :
    (0 < ∑ i ∈ range n, x ^ i) ∧ ∑ i ∈ range n, x ^ i < 1 := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt x 0
    hx' : LT.lt 0 (HAdd.hAdd x 1)
    hn : LT.lt 1 n
    ⊢ And (LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)) (LT.lt ((Finset. …
  -/
  refine Nat.le_induction ?_ ?_ n (show 2 ≤ n from hn)
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt x 0
      hx' : LT.lt 0 (HAdd.hAdd x 1)
      hn : LT.lt 1 n
      ⊢ And (LT.lt 0 ((Finset.range 2).sum fun i => HPow.hPow x i)) (LT.lt ((Finset. …
    -/
  · rw [geom_sum_two]
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt x 0
      hx' : LT.lt 0 (HAdd.hAdd x 1)
      hn : LT.lt 1 n
      ⊢ And (LT.lt 0 (HAdd.hAdd x 1)) (LT.lt (HAdd.hAdd x 1) 1)
    -/
    exact ⟨hx', (add_lt_iff_neg_right _).2 hx⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt x 0
    hx' : LT.lt 0 (HAdd.hAdd x 1)
    hn : LT.lt 1 n
    ⊢ ∀ (n : Nat), LE.le 2 n → And (LT.lt 0 ((Finset.range n).sum fun i => HPow.hP …
  -/
  clear hn
  /-
    case refine_2
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt x 0
    hx' : LT.lt 0 (HAdd.hAdd x 1)
    ⊢ ∀ (n : Nat), LE.le 2 n → And (LT.lt 0 ((Finset.range n).sum fun i => HPow.hP …
  -/
  intro n _ ihn
  /-
    case refine_2
    α : Type u
    n✝ : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt x 0
    hx' : LT.lt 0 (HAdd.hAdd x 1)
    n : Nat
    hmn✝ : LE.le 2 n
    ihn : And (LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)) (LT.lt ((Fin …
    ⊢ And (LT.lt 0 ((Finset.range (HAdd.hAdd n 1)).sum fun i => HPow.hPow x i)) (L …
  -/
  rw [geom_sum_succ, add_lt_iff_neg_right, ← neg_lt_iff_pos_add', neg_mul_eq_neg_mul]
  exact
    ⟨mul_lt_one_of_nonneg_of_lt_one_left (neg_nonneg.2 hx.le) (neg_lt_iff_pos_add'.2 hx') ihn.2.le,
      mul_neg_of_neg_of_pos hx ihn.1⟩


theorem geom_sum_alternating_of_le_neg_one [StrictOrderedRing α] (hx : x + 1 ≤ 0) (n : ℕ) :
    if Even n then (∑ i ∈ range n, x ^ i) ≤ 0 else 1 ≤ ∑ i ∈ range n, x ^ i := by
  /-
    α : Type u
    x : α
    inst✝ : StrictOrderedRing α
    hx : LE.le (HAdd.hAdd x 1) 0
    n : Nat
    ⊢ ite (Even n) (LE.le ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LE.le  …
  -/
  have hx0 : x ≤ 0 := (le_add_of_nonneg_right zero_le_one).trans hx
  induction n with
  | zero => simp only [range_zero, sum_empty, le_refl, ite_true, even_zero]
  | succ n ih =>
    simp only [Nat.even_add_one, geom_sum_succ]
    split_ifs at ih with h
    · rw [if_neg (not_not_intro h), le_add_iff_nonneg_left]
      exact mul_nonneg_of_nonpos_of_nonpos hx0 ih
    · rw [if_pos h]
      refine (add_le_add_right ?_ _).trans hx
      simpa only [mul_one] using mul_le_mul_of_nonpos_left ih hx0


theorem geom_sum_alternating_of_lt_neg_one [StrictOrderedRing α] (hx : x + 1 < 0) (hn : 1 < n) :
    if Even n then (∑ i ∈ range n, x ^ i) < 0 else 1 < ∑ i ∈ range n, x ^ i := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hn : LT.lt 1 n
    ⊢ ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT.lt  …
  -/
  have hx0 : x < 0 := (le_add_of_nonneg_right zero_le_one).trans_lt hx
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hn : LT.lt 1 n
    hx0 : LT.lt x 0
    ⊢ ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT.lt  …
  -/
  refine Nat.le_induction ?_ ?_ n (show 2 ≤ n from hn)
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hn : LT.lt 1 n
      hx0 : LT.lt x 0
      ⊢ ite (Even 2) (LT.lt ((Finset.range 2).sum fun i => HPow.hPow x i) 0) (LT.lt  …
    -/
  · simp only [geom_sum_two, lt_add_iff_pos_left, ite_true, gt_iff_lt, hx, even_two]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hn : LT.lt 1 n
    hx0 : LT.lt x 0
    ⊢ ∀ (n : Nat), LE.le 2 n → ite (Even n) (LT.lt ((Finset.range n).sum fun i =>  …
  -/
  clear hn
  /-
    case refine_2
    α : Type u
    n : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hx0 : LT.lt x 0
    ⊢ ∀ (n : Nat), LE.le 2 n → ite (Even n) (LT.lt ((Finset.range n).sum fun i =>  …
  -/
  intro n _ ihn
  /-
    case refine_2
    α : Type u
    n✝ : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hx0 : LT.lt x 0
    n : Nat
    hmn✝ : LE.le 2 n
    ihn : ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT …
    ⊢ ite (Even (HAdd.hAdd n 1)) (LT.lt ((Finset.range (HAdd.hAdd n 1)).sum fun i  …
  -/
  simp only [Nat.even_add_one, geom_sum_succ]
  /-
    case refine_2
    α : Type u
    n✝ : Nat
    x : α
    inst✝ : StrictOrderedRing α
    hx : LT.lt (HAdd.hAdd x 1) 0
    hx0 : LT.lt x 0
    n : Nat
    hmn✝ : LE.le 2 n
    ihn : ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT …
    ⊢ ite (Not (Even n)) (LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun  …
  -/
  by_cases hn' : Even n
    /-
      case pos
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT …
      hn' : Even n
      ⊢ ite (Not (Even n)) (LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun  …
    -/
  · rw [if_pos hn'] at ihn
    /-
      case pos
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
      hn' : Even n
      ⊢ ite (Not (Even n)) (LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun  …
    -/
    rw [if_neg, lt_add_iff_pos_left]
      /-
        case pos
        α : Type u
        n✝ : Nat
        x : α
        inst✝ : StrictOrderedRing α
        hx : LT.lt (HAdd.hAdd x 1) 0
        hx0 : LT.lt x 0
        n : Nat
        hmn✝ : LE.le 2 n
        ihn : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
        hn' : Even n
        ⊢ LT.lt 0 (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow x i))
      -/
    · exact mul_pos_of_neg_of_neg hx0 ihn
      /-
        🎉 no goals
      -/
      /-
        case pos.hnc
        α : Type u
        n✝ : Nat
        x : α
        inst✝ : StrictOrderedRing α
        hx : LT.lt (HAdd.hAdd x 1) 0
        hx0 : LT.lt x 0
        n : Nat
        hmn✝ : LE.le 2 n
        ihn : LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0
        hn' : Even n
        ⊢ Not (Not (Even n))
      -/
    · exact not_not_intro hn'
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : ite (Even n) (LT.lt ((Finset.range n).sum fun i => HPow.hPow x i) 0) (LT …
      hn' : Not (Even n)
      ⊢ ite (Not (Even n)) (LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun  …
    -/
  · rw [if_neg hn'] at ihn
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn' : Not (Even n)
      ⊢ ite (Not (Even n)) (LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun  …
    -/
    rw [if_pos]
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn' : Not (Even n)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow x i)) …
    -/
    swap
      /-
        case neg.hc
        α : Type u
        n✝ : Nat
        x : α
        inst✝ : StrictOrderedRing α
        hx : LT.lt (HAdd.hAdd x 1) 0
        hx0 : LT.lt x 0
        n : Nat
        hmn✝ : LE.le 2 n
        ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
        hn' : Not (Even n)
        ⊢ Not (Even n)
      -/
    · exact hn'
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn' : Not (Even n)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow x i)) …
    -/
    have := add_lt_add_right (mul_lt_mul_of_neg_left ihn hx0) 1
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn' : Not (Even n)
      this : LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow  …
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow x i)) …
    -/
    rw [mul_one] at this
    /-
      case neg
      α : Type u
      n✝ : Nat
      x : α
      inst✝ : StrictOrderedRing α
      hx : LT.lt (HAdd.hAdd x 1) 0
      hx0 : LT.lt x 0
      n : Nat
      hmn✝ : LE.le 2 n
      ihn : LT.lt 1 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn' : Not (Even n)
      this : LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow  …
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul x ((Finset.range n).sum fun i => HPow.hPow x i)) …
    -/
    exact this.trans hx
    /-
      🎉 no goals
    -/


theorem geom_sum_pos' [LinearOrderedRing α] (hx : 0 < x + 1) (hn : n ≠ 0) :
    0 < ∑ i ∈ range n, x ^ i := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hx : LT.lt 0 (HAdd.hAdd x 1)
    hn : Ne n 0
    ⊢ LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
  -/
  obtain _ | _ | n := n
    /-
      case zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : LT.lt 0 (HAdd.hAdd x 1)
      hn : Ne 0 0
      ⊢ LT.lt 0 ((Finset.range 0).sum fun i => HPow.hPow x i)
    -/
  · cases hn rfl
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : LT.lt 0 (HAdd.hAdd x 1)
      hn : Ne (HAdd.hAdd 0 1) 0
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd 0 1)).sum fun i => HPow.hPow x i)
    -/
  · simp only [zero_add, range_one, sum_singleton, pow_zero, zero_lt_one]
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    α : Type u
    x : α
    inst✝ : LinearOrderedRing α
    hx : LT.lt 0 (HAdd.hAdd x 1)
    n : Nat
    hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
    ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow …
  -/
  obtain hx' | hx' := lt_or_le x 0
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : LT.lt 0 (HAdd.hAdd x 1)
      n : Nat
      hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
      hx' : LT.lt x 0
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow …
    -/
  · exact (geom_sum_pos_and_lt_one hx' hx n.one_lt_succ_succ).1
    /-
      🎉 no goals
    -/
    /-
      case succ.succ.inr
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : LT.lt 0 (HAdd.hAdd x 1)
      n : Nat
      hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
      hx' : LE.le 0 x
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow …
    -/
  · exact geom_sum_pos hx' (by simp only [Nat.succ_ne_zero, Ne, not_false_iff])
    /-
      🎉 no goals
    -/


theorem Odd.geom_sum_pos [LinearOrderedRing α] (h : Odd n) : 0 < ∑ i ∈ range n, x ^ i := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    h : Odd n
    ⊢ LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
  -/
  rcases n with (_ | _ | k)
    /-
      case zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      h : Odd 0
      ⊢ LT.lt 0 ((Finset.range 0).sum fun i => HPow.hPow x i)
    -/
  · exact (Nat.not_odd_zero h).elim
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      h : Odd (HAdd.hAdd 0 1)
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd 0 1)).sum fun i => HPow.hPow x i)
    -/
  · simp only [zero_add, range_one, sum_singleton, pow_zero, zero_lt_one]
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    α : Type u
    x : α
    inst✝ : LinearOrderedRing α
    k : Nat
    h : Odd (HAdd.hAdd (HAdd.hAdd k 1) 1)
    ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
  -/
  rw [← Nat.not_even_iff_odd] at h
  /-
    case succ.succ
    α : Type u
    x : α
    inst✝ : LinearOrderedRing α
    k : Nat
    h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
    ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
  -/
  rcases lt_trichotomy (x + 1) 0 with (hx | hx | hx)
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      k : Nat
      h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
      hx : LT.lt (HAdd.hAdd x 1) 0
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
    -/
  · have := geom_sum_alternating_of_lt_neg_one hx k.one_lt_succ_succ
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      k : Nat
      h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
      hx : LT.lt (HAdd.hAdd x 1) 0
      this : ite (Even k.succ.succ) (LT.lt ((Finset.range k.succ.succ).sum fun i =>  …
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
    -/
    simp only [h, if_false] at this
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      k : Nat
      h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
      hx : LT.lt (HAdd.hAdd x 1) 0
      this : LT.lt 1 ((Finset.range k.succ.succ).sum fun i => HPow.hPow x i)
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
    -/
    exact zero_lt_one.trans this
    /-
      🎉 no goals
    -/
    /-
      case succ.succ.inr.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      k : Nat
      h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
      hx : Eq (HAdd.hAdd x 1) 0
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
    -/
  · simp only [eq_neg_of_add_eq_zero_left hx, h, neg_one_geom_sum, if_false, zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case succ.succ.inr.inr
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      k : Nat
      h : Not (Even (HAdd.hAdd (HAdd.hAdd k 1) 1))
      hx : LT.lt 0 (HAdd.hAdd x 1)
      ⊢ LT.lt 0 ((Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)).sum fun i => HPow.hPow …
    -/
  · exact geom_sum_pos' hx k.succ.succ_ne_zero
    /-
      🎉 no goals
    -/


theorem geom_sum_pos_iff [LinearOrderedRing α] (hn : n ≠ 0) :
    (0 < ∑ i ∈ range n, x ^ i) ↔ Odd n ∨ 0 < x + 1 := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hn : Ne n 0
    ⊢ Iff (LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)) (Or (Odd n) (LT. …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ Or (Odd n) (LT.lt 0 (HAdd.hAdd x 1))
    -/
  · rw [or_iff_not_imp_left, ← not_le, Nat.not_odd_iff_even]
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      ⊢ Even n → Not (LE.le (HAdd.hAdd x 1) 0)
    -/
    refine fun hn hx => h.not_le ?_
    /-
      case refine_1
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn✝ : Ne n 0
      h : LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      hn : Even n
      hx : LE.le (HAdd.hAdd x 1) 0
      ⊢ LE.le ((Finset.range n).sum fun i => HPow.hPow x i) 0
    -/
    simpa [if_pos hn] using geom_sum_alternating_of_le_neg_one hx n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      ⊢ Or (Odd n) (LT.lt 0 (HAdd.hAdd x 1)) → LT.lt 0 ((Finset.range n).sum fun i = …
    -/
  · rintro (hn | hx')
      /-
        case refine_2.inl
        α : Type u
        n : Nat
        x : α
        inst✝ : LinearOrderedRing α
        hn✝ : Ne n 0
        hn : Odd n
        ⊢ LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      -/
    · exact hn.geom_sum_pos
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        α : Type u
        n : Nat
        x : α
        inst✝ : LinearOrderedRing α
        hn : Ne n 0
        hx' : LT.lt 0 (HAdd.hAdd x 1)
        ⊢ LT.lt 0 ((Finset.range n).sum fun i => HPow.hPow x i)
      -/
    · exact geom_sum_pos' hx' hn
      /-
        🎉 no goals
      -/


theorem geom_sum_ne_zero [LinearOrderedRing α] (hx : x ≠ -1) (hn : n ≠ 0) :
    ∑ i ∈ range n, x ^ i ≠ 0 := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hx : Ne x (-1)
    hn : Ne n 0
    ⊢ Ne ((Finset.range n).sum fun i => HPow.hPow x i) 0
  -/
  obtain _ | _ | n := n
    /-
      case zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : Ne x (-1)
      hn : Ne 0 0
      ⊢ Ne ((Finset.range 0).sum fun i => HPow.hPow x i) 0
    -/
  · cases hn rfl
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : Ne x (-1)
      hn : Ne (HAdd.hAdd 0 1) 0
      ⊢ Ne ((Finset.range (HAdd.hAdd 0 1)).sum fun i => HPow.hPow x i) 0
    -/
  · simp only [zero_add, range_one, sum_singleton, pow_zero, ne_eq, one_ne_zero, not_false_eq_true]
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    α : Type u
    x : α
    inst✝ : LinearOrderedRing α
    hx : Ne x (-1)
    n : Nat
    hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
    ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
  -/
  rw [Ne, eq_neg_iff_add_eq_zero, ← Ne] at hx
  /-
    case succ.succ
    α : Type u
    x : α
    inst✝ : LinearOrderedRing α
    hx : Ne (HAdd.hAdd x 1) 0
    n : Nat
    hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
    ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
  -/
  obtain h | h := hx.lt_or_lt
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : Ne (HAdd.hAdd x 1) 0
      n : Nat
      hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
      h : LT.lt (HAdd.hAdd x 1) 0
      ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
    -/
  · have := geom_sum_alternating_of_lt_neg_one h n.one_lt_succ_succ
    /-
      case succ.succ.inl
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : Ne (HAdd.hAdd x 1) 0
      n : Nat
      hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
      h : LT.lt (HAdd.hAdd x 1) 0
      this : ite (Even n.succ.succ) (LT.lt ((Finset.range n.succ.succ).sum fun i =>  …
      ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
    -/
    split_ifs at this
      /-
        case pos
        α : Type u
        x : α
        inst✝ : LinearOrderedRing α
        hx : Ne (HAdd.hAdd x 1) 0
        n : Nat
        hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
        h : LT.lt (HAdd.hAdd x 1) 0
        h✝ : Even n.succ.succ
        this : LT.lt ((Finset.range n.succ.succ).sum fun i => HPow.hPow x i) 0
        ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
      -/
    · exact this.ne
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        x : α
        inst✝ : LinearOrderedRing α
        hx : Ne (HAdd.hAdd x 1) 0
        n : Nat
        hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
        h : LT.lt (HAdd.hAdd x 1) 0
        h✝ : Not (Even n.succ.succ)
        this : LT.lt 1 ((Finset.range n.succ.succ).sum fun i => HPow.hPow x i)
        ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
      -/
    · exact (zero_lt_one.trans this).ne'
      /-
        🎉 no goals
      -/
    /-
      case succ.succ.inr
      α : Type u
      x : α
      inst✝ : LinearOrderedRing α
      hx : Ne (HAdd.hAdd x 1) 0
      n : Nat
      hn : Ne (HAdd.hAdd (HAdd.hAdd n 1) 1) 0
      h : LT.lt 0 (HAdd.hAdd x 1)
      ⊢ Ne ((Finset.range (HAdd.hAdd (HAdd.hAdd n 1) 1)).sum fun i => HPow.hPow x i) 0
    -/
  · exact (geom_sum_pos' h n.succ.succ_ne_zero).ne'
    /-
      🎉 no goals
    -/


theorem geom_sum_eq_zero_iff_neg_one [LinearOrderedRing α] (hn : n ≠ 0) :
    ∑ i ∈ range n, x ^ i = 0 ↔ x = -1 ∧ Even n := by
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hn : Ne n 0
    ⊢ Iff (Eq ((Finset.range n).sum fun i => HPow.hPow x i) 0) (And (Eq x (-1)) (E …
  -/
  refine ⟨fun h => ?_, @fun ⟨h, hn⟩ => by simp only [h, hn, neg_one_geom_sum, if_true]⟩
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hn : Ne n 0
    h : Eq ((Finset.range n).sum fun i => HPow.hPow x i) 0
    ⊢ And (Eq x (-1)) (Even n)
  -/
  contrapose! h
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hn : Ne n 0
    h : Eq x (-1) → Not (Even n)
    ⊢ Ne ((Finset.range n).sum fun i => HPow.hPow x i) 0
  -/
  have hx := eq_or_ne x (-1)
  /-
    α : Type u
    n : Nat
    x : α
    inst✝ : LinearOrderedRing α
    hn : Ne n 0
    h : Eq x (-1) → Not (Even n)
    hx : Or (Eq x (-1)) (Ne x (-1))
    ⊢ Ne ((Finset.range n).sum fun i => HPow.hPow x i) 0
  -/
  rcases hx with hx | hx
    /-
      case inl
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      h : Eq x (-1) → Not (Even n)
      hx : Eq x (-1)
      ⊢ Ne ((Finset.range n).sum fun i => HPow.hPow x i) 0
    -/
  · rw [hx, neg_one_geom_sum]
    /-
      case inl
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      h : Eq x (-1) → Not (Even n)
      hx : Eq x (-1)
      ⊢ Ne (ite (Even n) 0 1) 0
    -/
    simp only [h hx, ite_false, ne_eq, one_ne_zero, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      n : Nat
      x : α
      inst✝ : LinearOrderedRing α
      hn : Ne n 0
      h : Eq x (-1) → Not (Even n)
      hx : Ne x (-1)
      ⊢ Ne ((Finset.range n).sum fun i => HPow.hPow x i) 0
    -/
  · exact geom_sum_ne_zero hx hn
    /-
      🎉 no goals
    -/


theorem geom_sum_neg_iff [LinearOrderedRing α] (hn : n ≠ 0) :
    ∑ i ∈ range n, x ^ i < 0 ↔ Even n ∧ x + 1 < 0 := by
  rw [← not_iff_not, not_lt, le_iff_lt_or_eq, eq_comm,
    or_congr (geom_sum_pos_iff hn) (geom_sum_eq_zero_iff_neg_one hn), ← Nat.not_even_iff_odd, ←
    add_eq_zero_iff_eq_neg, not_and, not_lt, le_iff_lt_or_eq, eq_comm, ← imp_iff_not_or, or_comm,
    and_comm, Decidable.and_or_imp, or_comm]


/-- Value of a geometric sum over the naturals. Note: see `geom_sum_mul_add` for a formulation
that avoids division and subtraction. -/
lemma Nat.geomSum_eq (hm : 2 ≤ m) (n : ℕ) :
    ∑ k ∈ range n, m ^ k = (m ^ n - 1) / (m - 1) := by
  /-
    m : Nat
    hm : LE.le 2 m
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun k => HPow.hPow m k) (HDiv.hDiv (HSub.hSub (HPow …
  -/
  refine (Nat.div_eq_of_eq_mul_left (tsub_pos_iff_lt.2 hm) <| tsub_eq_of_eq_add ?_).symm
  /-
    m : Nat
    hm : LE.le 2 m
    n : Nat
    ⊢ Eq (HPow.hPow m n) (HAdd.hAdd (HMul.hMul ((Finset.range n).sum fun k => HPow …
  -/
  simpa only [tsub_add_cancel_of_le (one_le_two.trans hm), eq_comm] using geom_sum_mul_add (m - 1) n
  /-
    🎉 no goals
  -/


/-- If all the elements of a finset of naturals are less than `n`, then the sum of their powers of
`m ≥ 2` is less than `m ^ n`. -/
lemma Nat.geomSum_lt (hm : 2 ≤ m) (hs : ∀ k ∈ s, k < n) : ∑ k ∈ s, m ^ k < m ^ n :=
  calc
    ∑ k ∈ s, m ^ k ≤ ∑ k ∈ range n, m ^ k := sum_le_sum_of_subset fun _ hk ↦
      mem_range.2 <| hs _ hk
    _ = (m ^ n - 1) / (m - 1) := Nat.geomSum_eq hm _
    _ ≤ m ^ n - 1 := Nat.div_le_self _ _
                                  /-
                                    m n : Nat
                                    s : Finset Nat
                                    hm : LE.le 2 m
                                    hs : ∀ (k : Nat), Membership.mem s k → LT.lt k n
                                    ⊢ LT.lt 0 (HPow.hPow m n)
                                  -/
    _ < m ^ n := tsub_lt_self (by positivity) zero_lt_one
                                  /-
                                    🎉 no goals
                                  -/

