@[to_additive]
theorem inv_mem_Icc_iff : a⁻¹ ∈ Set.Icc c d ↔ a ∈ Set.Icc d⁻¹ c⁻¹ :=
  and_comm.trans <| and_congr inv_le' le_inv'


@[to_additive]
theorem inv_mem_Ico_iff : a⁻¹ ∈ Set.Ico c d ↔ a ∈ Set.Ioc d⁻¹ c⁻¹ :=
  and_comm.trans <| and_congr inv_lt' le_inv'


@[to_additive]
theorem inv_mem_Ioc_iff : a⁻¹ ∈ Set.Ioc c d ↔ a ∈ Set.Ico d⁻¹ c⁻¹ :=
  and_comm.trans <| and_congr inv_le' lt_inv'


@[to_additive]
theorem inv_mem_Ioo_iff : a⁻¹ ∈ Set.Ioo c d ↔ a ∈ Set.Ioo d⁻¹ c⁻¹ :=
  and_comm.trans <| and_congr inv_lt' lt_inv'


theorem add_mem_Icc_iff_left : a + b ∈ Set.Icc c d ↔ a ∈ Set.Icc (c - b) (d - b) :=
  (and_congr (sub_le_iff_le_add (α := α)) (le_sub_iff_add_le (α := α))).symm


theorem add_mem_Ico_iff_left : a + b ∈ Set.Ico c d ↔ a ∈ Set.Ico (c - b) (d - b) :=
  (and_congr (sub_le_iff_le_add (α := α)) (lt_sub_iff_add_lt (α := α))).symm


theorem add_mem_Ioc_iff_left : a + b ∈ Set.Ioc c d ↔ a ∈ Set.Ioc (c - b) (d - b) :=
  (and_congr (sub_lt_iff_lt_add (α := α)) (le_sub_iff_add_le (α := α))).symm


theorem add_mem_Ioo_iff_left : a + b ∈ Set.Ioo c d ↔ a ∈ Set.Ioo (c - b) (d - b) :=
  (and_congr (sub_lt_iff_lt_add (α := α)) (lt_sub_iff_add_lt (α := α))).symm


theorem add_mem_Icc_iff_right : a + b ∈ Set.Icc c d ↔ b ∈ Set.Icc (c - a) (d - a) :=
  (and_congr sub_le_iff_le_add' le_sub_iff_add_le').symm


theorem add_mem_Ico_iff_right : a + b ∈ Set.Ico c d ↔ b ∈ Set.Ico (c - a) (d - a) :=
  (and_congr sub_le_iff_le_add' lt_sub_iff_add_lt').symm


theorem add_mem_Ioc_iff_right : a + b ∈ Set.Ioc c d ↔ b ∈ Set.Ioc (c - a) (d - a) :=
  (and_congr sub_lt_iff_lt_add' le_sub_iff_add_le').symm


theorem add_mem_Ioo_iff_right : a + b ∈ Set.Ioo c d ↔ b ∈ Set.Ioo (c - a) (d - a) :=
  (and_congr sub_lt_iff_lt_add' lt_sub_iff_add_lt').symm


theorem sub_mem_Icc_iff_left : a - b ∈ Set.Icc c d ↔ a ∈ Set.Icc (c + b) (d + b) :=
  and_congr le_sub_iff_add_le sub_le_iff_le_add


theorem sub_mem_Ico_iff_left : a - b ∈ Set.Ico c d ↔ a ∈ Set.Ico (c + b) (d + b) :=
  and_congr le_sub_iff_add_le sub_lt_iff_lt_add


theorem sub_mem_Ioc_iff_left : a - b ∈ Set.Ioc c d ↔ a ∈ Set.Ioc (c + b) (d + b) :=
  and_congr lt_sub_iff_add_lt sub_le_iff_le_add


theorem sub_mem_Ioo_iff_left : a - b ∈ Set.Ioo c d ↔ a ∈ Set.Ioo (c + b) (d + b) :=
  and_congr lt_sub_iff_add_lt sub_lt_iff_lt_add


theorem sub_mem_Icc_iff_right : a - b ∈ Set.Icc c d ↔ b ∈ Set.Icc (a - d) (a - c) :=
  and_comm.trans <| and_congr sub_le_comm le_sub_comm


theorem sub_mem_Ico_iff_right : a - b ∈ Set.Ico c d ↔ b ∈ Set.Ioc (a - d) (a - c) :=
  and_comm.trans <| and_congr sub_lt_comm le_sub_comm


theorem sub_mem_Ioc_iff_right : a - b ∈ Set.Ioc c d ↔ b ∈ Set.Ico (a - d) (a - c) :=
  and_comm.trans <| and_congr sub_le_comm lt_sub_comm


theorem sub_mem_Ioo_iff_right : a - b ∈ Set.Ioo c d ↔ b ∈ Set.Ioo (a - d) (a - c) :=
  and_comm.trans <| and_congr sub_lt_comm lt_sub_comm

-- I think that symmetric intervals deserve attention and API: they arise all the time,
-- for instance when considering metric balls in `ℝ`.

theorem mem_Icc_iff_abs_le {R : Type*} [LinearOrderedAddCommGroup R] {x y z : R} :
    |x - y| ≤ z ↔ y ∈ Icc (x - z) (x + z) :=
  abs_le.trans <| and_comm.trans <| and_congr sub_le_comm neg_le_sub_iff_le_add


theorem sub_mem_Icc_zero_iff_right : b - a ∈ Icc 0 b ↔ a ∈ Icc 0 b := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Iff (Membership.mem (Set.Icc 0 b) (HSub.hSub b a)) (Membership.mem (Set.Icc  …
  -/
  simp only [sub_mem_Icc_iff_right, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


theorem sub_mem_Ico_zero_iff_right : b - a ∈ Ico 0 b ↔ a ∈ Ioc 0 b := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Iff (Membership.mem (Set.Ico 0 b) (HSub.hSub b a)) (Membership.mem (Set.Ioc  …
  -/
  simp only [sub_mem_Ico_iff_right, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


theorem sub_mem_Ioc_zero_iff_right : b - a ∈ Ioc 0 b ↔ a ∈ Ico 0 b := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Iff (Membership.mem (Set.Ioc 0 b) (HSub.hSub b a)) (Membership.mem (Set.Ico  …
  -/
  simp only [sub_mem_Ioc_iff_right, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


theorem sub_mem_Ioo_zero_iff_right : b - a ∈ Ioo 0 b ↔ a ∈ Ioo 0 b := by
  /-
    α : Type u_1
    inst✝ : OrderedAddCommGroup α
    a b : α
    ⊢ Iff (Membership.mem (Set.Ioo 0 b) (HSub.hSub b a)) (Membership.mem (Set.Ioo  …
  -/
  simp only [sub_mem_Ioo_iff_right, sub_self, sub_zero]
  /-
    🎉 no goals
  -/


/-- If we remove a smaller interval from a larger, the result is nonempty -/
theorem nonempty_Ico_sdiff {x dx y dy : α} (h : dy < dx) (hx : 0 < dx) :
    Nonempty ↑(Ico x (x + dx) \ Ico y (y + dy)) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    x dx y dy : α
    h : LT.lt dy dx
    hx : LT.lt 0 dx
    ⊢ Nonempty ↑(SDiff.sdiff (Set.Ico x (HAdd.hAdd x dx)) (Set.Ico y (HAdd.hAdd y  …
  -/
  cases' lt_or_le x y with h' h'
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      x dx y dy : α
      h : LT.lt dy dx
      hx : LT.lt 0 dx
      h' : LT.lt x y
      ⊢ Nonempty ↑(SDiff.sdiff (Set.Ico x (HAdd.hAdd x dx)) (Set.Ico y (HAdd.hAdd y  …
    -/
  · use x
    /-
      case property
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      x dx y dy : α
      h : LT.lt dy dx
      hx : LT.lt 0 dx
      h' : LT.lt x y
      ⊢ Membership.mem (SDiff.sdiff (Set.Ico x (HAdd.hAdd x dx)) (Set.Ico y (HAdd.hA …
    -/
    simp [*, not_le.2 h']
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      x dx y dy : α
      h : LT.lt dy dx
      hx : LT.lt 0 dx
      h' : LE.le y x
      ⊢ Nonempty ↑(SDiff.sdiff (Set.Ico x (HAdd.hAdd x dx)) (Set.Ico y (HAdd.hAdd y  …
    -/
  · use max x (x + dy)
    /-
      case property
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      x dx y dy : α
      h : LT.lt dy dx
      hx : LT.lt 0 dx
      h' : LE.le y x
      ⊢ Membership.mem (SDiff.sdiff (Set.Ico x (HAdd.hAdd x dx)) (Set.Ico y (HAdd.hA …
    -/
    simp [*, le_refl]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem pairwise_disjoint_Ioc_mul_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ioc (a * b ^ n) (a * b ^ (n + 1))) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioc (HMul.hMul a (HPow.hPow b …
  -/
  simp (config := { unfoldPartialApp := true }) only [Function.onFun]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise fun x y => Disjoint (Set.Ioc (HMul.hMul a (HPow.hPow b x)) (HMul.hM …
  -/
  simp_rw [Set.disjoint_iff]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise fun x y => HasSubset.Subset (Inter.inter (Set.Ioc (HMul.hMul a (HPo …
  -/
  intro m n hmn x hx
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ioc (HMul.hMul a (HPow.hPow b m)) (HMul. …
    ⊢ Membership.mem EmptyCollection.emptyCollection x
  -/
  apply hmn
  have hb : 1 < b := by
    have : a * b ^ m < a * b ^ (m + 1) := hx.1.1.trans_le hx.1.2
    rwa [mul_lt_mul_iff_left, ← mul_one (b ^ m), zpow_add_one, mul_lt_mul_iff_left] at this
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ioc (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    ⊢ Eq m n
  -/
  have i1 := hx.1.1.trans_le hx.2.2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ioc (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LT.lt (HMul.hMul a (HPow.hPow b m)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    ⊢ Eq m n
  -/
  have i2 := hx.2.1.trans_le hx.1.2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ioc (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LT.lt (HMul.hMul a (HPow.hPow b m)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    i2 : LT.lt (HMul.hMul a (HPow.hPow b n)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    ⊢ Eq m n
  -/
  rw [mul_lt_mul_iff_left, zpow_lt_zpow_iff_right hb, Int.lt_add_one_iff] at i1 i2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ioc (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LE.le m n
    i2 : LE.le n m
    ⊢ Eq m n
  -/
  exact le_antisymm i1 i2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_disjoint_Ico_mul_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ico (a * b ^ n) (a * b ^ (n + 1))) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ico (HMul.hMul a (HPow.hPow b …
  -/
  simp (config := { unfoldPartialApp := true }) only [Function.onFun]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise fun x y => Disjoint (Set.Ico (HMul.hMul a (HPow.hPow b x)) (HMul.hM …
  -/
  simp_rw [Set.disjoint_iff]
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    ⊢ Pairwise fun x y => HasSubset.Subset (Inter.inter (Set.Ico (HMul.hMul a (HPo …
  -/
  intro m n hmn x hx
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ico (HMul.hMul a (HPow.hPow b m)) (HMul. …
    ⊢ Membership.mem EmptyCollection.emptyCollection x
  -/
  apply hmn
  have hb : 1 < b := by
    have : a * b ^ m < a * b ^ (m + 1) := hx.1.1.trans_lt hx.1.2
    rwa [mul_lt_mul_iff_left, ← mul_one (b ^ m), zpow_add_one, mul_lt_mul_iff_left] at this
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ico (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    ⊢ Eq m n
  -/
  have i1 := hx.1.1.trans_lt hx.2.2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ico (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LT.lt (HMul.hMul a (HPow.hPow b m)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    ⊢ Eq m n
  -/
  have i2 := hx.2.1.trans_lt hx.1.2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ico (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LT.lt (HMul.hMul a (HPow.hPow b m)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    i2 : LT.lt (HMul.hMul a (HPow.hPow b n)) (HMul.hMul a (HPow.hPow b (HAdd.hAdd  …
    ⊢ Eq m n
  -/
  rw [mul_lt_mul_iff_left, zpow_lt_zpow_iff_right hb, Int.lt_add_one_iff] at i1 i2
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    a b : α
    m n : Int
    hmn : Ne m n
    x : α
    hx : Membership.mem (Inter.inter (Set.Ico (HMul.hMul a (HPow.hPow b m)) (HMul. …
    hb : LT.lt 1 b
    i1 : LE.le m n
    i2 : LE.le n m
    ⊢ Eq m n
  -/
  exact le_antisymm i1 i2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_disjoint_Ioo_mul_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ioo (a * b ^ n) (a * b ^ (n + 1))) := fun _ _ hmn =>
  (pairwise_disjoint_Ioc_mul_zpow a b hmn).mono Ioo_subset_Ioc_self Ioo_subset_Ioc_self


@[to_additive]
theorem pairwise_disjoint_Ioc_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ioc (b ^ n) (b ^ (n + 1))) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    b : α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioc (HPow.hPow b n) (HPow.hPo …
  -/
  simpa only [one_mul] using pairwise_disjoint_Ioc_mul_zpow 1 b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_disjoint_Ico_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ico (b ^ n) (b ^ (n + 1))) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    b : α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ico (HPow.hPow b n) (HPow.hPo …
  -/
  simpa only [one_mul] using pairwise_disjoint_Ico_mul_zpow 1 b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pairwise_disjoint_Ioo_zpow :
    Pairwise (Disjoint on fun n : ℤ => Ioo (b ^ n) (b ^ (n + 1))) := by
  /-
    α : Type u_1
    inst✝ : OrderedCommGroup α
    b : α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioo (HPow.hPow b n) (HPow.hPo …
  -/
  simpa only [one_mul] using pairwise_disjoint_Ioo_mul_zpow 1 b
  /-
    🎉 no goals
  -/


theorem pairwise_disjoint_Ioc_add_intCast :
    Pairwise (Disjoint on fun n : ℤ => Ioc (a + n) (a + n + 1)) := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    pairwise_disjoint_Ioc_add_zsmul a (1 : α)


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ioc_add_int_cast := pairwise_disjoint_Ioc_add_intCast


theorem pairwise_disjoint_Ico_add_intCast :
    Pairwise (Disjoint on fun n : ℤ => Ico (a + n) (a + n + 1)) := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    pairwise_disjoint_Ico_add_zsmul a (1 : α)


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ico_add_int_cast := pairwise_disjoint_Ico_add_intCast


theorem pairwise_disjoint_Ioo_add_intCast :
    Pairwise (Disjoint on fun n : ℤ => Ioo (a + n) (a + n + 1)) := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    pairwise_disjoint_Ioo_add_zsmul a (1 : α)


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ioo_add_int_cast := pairwise_disjoint_Ioo_add_intCast


theorem pairwise_disjoint_Ico_intCast :
    Pairwise (Disjoint on fun n : ℤ => Ico (n : α) (n + 1)) := by
  /-
    α : Type u_1
    inst✝ : OrderedRing α
    ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ico (↑n) (HAdd.hAdd (↑n) 1))
  -/
  simpa only [zero_add] using pairwise_disjoint_Ico_add_intCast (0 : α)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ico_int_cast := pairwise_disjoint_Ico_intCast


theorem pairwise_disjoint_Ioo_intCast : Pairwise (Disjoint on fun n : ℤ => Ioo (n : α) (n + 1)) :=
     /-
       α : Type u_1
       inst✝ : OrderedRing α
       ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioo (↑n) (HAdd.hAdd (↑n) 1))
     -/
  by simpa only [zero_add] using pairwise_disjoint_Ioo_add_intCast (0 : α)
     /-
       🎉 no goals
     -/


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ioo_int_cast := pairwise_disjoint_Ioo_intCast


theorem pairwise_disjoint_Ioc_intCast : Pairwise (Disjoint on fun n : ℤ => Ioc (n : α) (n + 1)) :=
     /-
       α : Type u_1
       inst✝ : OrderedRing α
       ⊢ Pairwise (Function.onFun Disjoint fun n => Set.Ioc (↑n) (HAdd.hAdd (↑n) 1))
     -/
  by simpa only [zero_add] using pairwise_disjoint_Ioc_add_intCast (0 : α)
     /-
       🎉 no goals
     -/


@[deprecated (since := "2024-04-17")]
alias pairwise_disjoint_Ioc_int_cast := pairwise_disjoint_Ioc_intCast


