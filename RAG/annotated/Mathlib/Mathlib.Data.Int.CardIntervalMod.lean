lemma Ico_filter_modEq_eq (v : ℤ) :
    {x ∈ Ico a b | x ≡ v [ZMOD r]} =
    {x ∈ Ico (a - v) (b - v) | r ∣ x}.map ⟨(· + v), add_left_injective v⟩ := by
  /-
    a b r v : Int
    ⊢ Eq (Finset.filter (fun x => r.ModEq x v) (Finset.Ico a b)) (Finset.map { toF …
  -/
  ext x
  simp_rw [mem_map, mem_filter, mem_Ico, Function.Embedding.coeFn_mk, ← eq_sub_iff_add_eq,
    exists_eq_right, modEq_comm, modEq_iff_dvd, sub_lt_sub_iff_right, sub_le_sub_iff_right]


lemma Ioc_filter_modEq_eq (v : ℤ) :
    {x ∈ Ioc a b | x ≡ v [ZMOD r]} =
    {x ∈ Ioc (a - v) (b - v) | r ∣ x}.map ⟨(· + v), add_left_injective v⟩ := by
  /-
    a b r v : Int
    ⊢ Eq (Finset.filter (fun x => r.ModEq x v) (Finset.Ioc a b)) (Finset.map { toF …
  -/
  ext x
  simp_rw [mem_map, mem_filter, mem_Ioc, Function.Embedding.coeFn_mk, ← eq_sub_iff_add_eq,
    exists_eq_right, modEq_comm, modEq_iff_dvd, sub_lt_sub_iff_right, sub_le_sub_iff_right]


lemma Ico_filter_dvd_eq :
    {x ∈ Ico a b | r ∣ x} =
      (Ico ⌈a / (r : ℚ)⌉ ⌈b / (r : ℚ)⌉).map ⟨(· * r), mul_left_injective₀ hr.ne'⟩ := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    ⊢ Eq (Finset.filter (fun x => Dvd.dvd r x) (Finset.Ico a b)) (Finset.map { toF …
  -/
  ext x
  simp only [mem_map, mem_filter, mem_Ico, ceil_le, lt_ceil, div_le_iff₀, lt_div_iff₀,
    dvd_iff_exists_eq_mul_left, cast_pos.2 hr, ← cast_mul, cast_lt, cast_le]
  /-
    case h
    a b r : Int
    hr : LT.lt 0 r
    x : Int
    ⊢ Iff (And (And (LE.le a x) (LT.lt x b)) (Exists fun c => Eq x (HMul.hMul c r) …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma Ioc_filter_dvd_eq :
    {x ∈ Ioc a b | r ∣ x} =
      (Ioc ⌊a / (r : ℚ)⌋ ⌊b / (r : ℚ)⌋).map ⟨(· * r), mul_left_injective₀ hr.ne'⟩ := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    ⊢ Eq (Finset.filter (fun x => Dvd.dvd r x) (Finset.Ioc a b)) (Finset.map { toF …
  -/
  ext x
  simp only [mem_map, mem_filter, mem_Ioc, floor_lt, le_floor, div_lt_iff₀, le_div_iff₀,
    dvd_iff_exists_eq_mul_left, cast_pos.2 hr, ← cast_mul, cast_lt, cast_le]
  /-
    case h
    a b r : Int
    hr : LT.lt 0 r
    x : Int
    ⊢ Iff (And (And (LT.lt a x) (LE.le x b)) (Exists fun c => Eq x (HMul.hMul c r) …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- There are `⌈b / r⌉ - ⌈a / r⌉` multiples of `r` in `[a, b)`, if `a ≤ b`. -/
theorem Ico_filter_dvd_card : #{x ∈ Ico a b | r ∣ x} = max (⌈b / (r : ℚ)⌉ - ⌈a / (r : ℚ)⌉) 0 := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    ⊢ Eq (↑(Finset.filter (fun x => Dvd.dvd r x) (Finset.Ico a b)).card) (Max.max  …
  -/
  rw [Ico_filter_dvd_eq _ _ hr, card_map, card_Ico, toNat_eq_max]
  /-
    🎉 no goals
  -/


/-- There are `⌊b / r⌋ - ⌊a / r⌋` multiples of `r` in `(a, b]`, if `a ≤ b`. -/
theorem Ioc_filter_dvd_card : #{x ∈ Ioc a b | r ∣ x} = max (⌊b / (r : ℚ)⌋ - ⌊a / (r : ℚ)⌋) 0 := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    ⊢ Eq (↑(Finset.filter (fun x => Dvd.dvd r x) (Finset.Ioc a b)).card) (Max.max  …
  -/
  rw [Ioc_filter_dvd_eq _ _ hr, card_map, card_Ioc, toNat_eq_max]
  /-
    🎉 no goals
  -/


/-- There are `⌈(b - v) / r⌉ - ⌈(a - v) / r⌉` numbers congruent to `v` mod `r` in `[a, b)`,
if `a ≤ b`. -/
theorem Ico_filter_modEq_card (v : ℤ) :
    #{x ∈ Ico a b | x ≡ v [ZMOD r]} = max (⌈(b - v) / (r : ℚ)⌉ - ⌈(a - v) / (r : ℚ)⌉) 0 := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    v : Int
    ⊢ Eq (↑(Finset.filter (fun x => r.ModEq x v) (Finset.Ico a b)).card) (Max.max  …
  -/
  simp [Ico_filter_modEq_eq, Ico_filter_dvd_eq, toNat_eq_max, hr]
  /-
    🎉 no goals
  -/


/-- There are `⌊(b - v) / r⌋ - ⌊(a - v) / r⌋` numbers congruent to `v` mod `r` in `(a, b]`,
if `a ≤ b`. -/
theorem Ioc_filter_modEq_card (v : ℤ) :
    #{x ∈ Ioc a b | x ≡ v [ZMOD r]} = max (⌊(b - v) / (r : ℚ)⌋ - ⌊(a - v) / (r : ℚ)⌋) 0 := by
  /-
    a b r : Int
    hr : LT.lt 0 r
    v : Int
    ⊢ Eq (↑(Finset.filter (fun x => r.ModEq x v) (Finset.Ioc a b)).card) (Max.max  …
  -/
  simp [Ioc_filter_modEq_eq, Ioc_filter_dvd_eq, toNat_eq_max, hr]
  /-
    🎉 no goals
  -/


lemma Ico_filter_modEq_cast {v : ℕ} :
    {x ∈ Ico a b | x ≡ v [MOD r]}.map castEmbedding =
      {x ∈ Ico (a : ℤ) (b : ℤ) | x ≡ v [ZMOD r]} := by
  /-
    a b r v : Nat
    ⊢ Eq (Finset.map Nat.castEmbedding (Finset.filter (fun x => r.ModEq x v) (Fins …
  -/
  ext x
  /-
    case h
    a b r v : Nat
    x : Int
    ⊢ Iff (Membership.mem (Finset.map Nat.castEmbedding (Finset.filter (fun x => r …
  -/
  simp only [mem_map, mem_filter, mem_Ico, castEmbedding_apply]
  /-
    case h
    a b r v : Nat
    x : Int
    ⊢ Iff (Exists fun a_1 => And (And (And (LE.le a a_1) (LT.lt a_1 b)) (r.ModEq a …
  -/
  constructor
    /-
      case h.mp
      a b r v : Nat
      x : Int
      ⊢ (Exists fun a_1 => And (And (And (LE.le a a_1) (LT.lt a_1 b)) (r.ModEq a_1 v …
    -/
  · simp_rw [forall_exists_index, ← natCast_modEq_iff]; intro y ⟨h, c⟩; subst c; exact_mod_cast h
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    /-
      case h.mpr
      a b r v : Nat
      x : Int
      ⊢ And (And (LE.le (↑a) x) (LT.lt x ↑b)) ((↑r).ModEq x ↑v) → Exists fun a_2 =>  …
    -/
  · intro h; lift x to ℕ using (by omega); exact ⟨x, by simp_all [natCast_modEq_iff]⟩
                                           /-
                                             🎉 no goals
                                           -/


lemma Ioc_filter_modEq_cast {v : ℕ} :
    {x ∈ Ioc a b | x ≡ v [MOD r]}.map castEmbedding =
      {x ∈ Ioc (a : ℤ) (b : ℤ) | x ≡ v [ZMOD r]} := by
  /-
    a b r v : Nat
    ⊢ Eq (Finset.map Nat.castEmbedding (Finset.filter (fun x => r.ModEq x v) (Fins …
  -/
  ext x
  /-
    case h
    a b r v : Nat
    x : Int
    ⊢ Iff (Membership.mem (Finset.map Nat.castEmbedding (Finset.filter (fun x => r …
  -/
  simp only [mem_map, mem_filter, mem_Ioc, castEmbedding_apply]
  /-
    case h
    a b r v : Nat
    x : Int
    ⊢ Iff (Exists fun a_1 => And (And (And (LT.lt a a_1) (LE.le a_1 b)) (r.ModEq a …
  -/
  constructor
    /-
      case h.mp
      a b r v : Nat
      x : Int
      ⊢ (Exists fun a_1 => And (And (And (LT.lt a a_1) (LE.le a_1 b)) (r.ModEq a_1 v …
    -/
  · simp_rw [forall_exists_index, ← natCast_modEq_iff]; intro y ⟨h, c⟩; subst c; exact_mod_cast h
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    /-
      case h.mpr
      a b r v : Nat
      x : Int
      ⊢ And (And (LT.lt (↑a) x) (LE.le x ↑b)) ((↑r).ModEq x ↑v) → Exists fun a_2 =>  …
    -/
  · intro h; lift x to ℕ using (by omega); exact ⟨x, by simp_all [natCast_modEq_iff]⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- There are `⌈(b - v) / r⌉ - ⌈(a - v) / r⌉` numbers congruent to `v` mod `r` in `[a, b)`,
if `a ≤ b`. `Nat` version of `Int.Ico_filter_modEq_card`. -/
theorem Ico_filter_modEq_card (v : ℕ) :
    #{x ∈ Ico a b | x ≡ v [MOD r]} = max (⌈(b - v) / (r : ℚ)⌉ - ⌈(a - v) / (r : ℚ)⌉) 0 := by
  simp_rw [← Ico_filter_modEq_cast _ _ ▸ card_map _,
    Int.Ico_filter_modEq_card _ _ (cast_lt.mpr hr), Int.cast_natCast]


/-- There are `⌊(b - v) / r⌋ - ⌊(a - v) / r⌋` numbers congruent to `v` mod `r` in `(a, b]`,
if `a ≤ b`. `Nat` version of `Int.Ioc_filter_modEq_card`. -/
theorem Ioc_filter_modEq_card (v : ℕ) :
    #{x ∈ Ioc a b | x ≡ v [MOD r]} = max (⌊(b - v) / (r : ℚ)⌋ - ⌊(a - v) / (r : ℚ)⌋) 0 := by
  simp_rw [← Ioc_filter_modEq_cast _ _ ▸ card_map _,
    Int.Ioc_filter_modEq_card _ _ (cast_lt.mpr hr), Int.cast_natCast]


/-- There are `⌈(b - v % r) / r⌉` numbers in `[0, b)` congruent to `v` mod `r`. -/
theorem count_modEq_card_eq_ceil (v : ℕ) :
    b.count (· ≡ v [MOD r]) = ⌈(b - (v % r : ℕ)) / (r : ℚ)⌉ := by
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    ⊢ Eq (↑(Nat.count (fun x => r.ModEq x v) b)) (Int.ceil (HDiv.hDiv (HSub.hSub ↑ …
  -/
  have hr' : 0 < (r : ℚ) := by positivity
  rw [count_eq_card_filter_range, ← Ico_zero_eq_range, Ico_filter_modEq_card _ _ hr,
    max_eq_left (sub_nonneg.mpr <| by gcongr <;> positivity)]
  conv_lhs =>
    rw [← div_add_mod v r, cast_add, cast_mul, add_comm]
    tactic => simp_rw [← sub_sub, sub_div (_ - _), mul_div_cancel_left₀ _ hr'.ne', ceil_sub_nat]
    rw [sub_sub_sub_cancel_right, cast_zero, zero_sub]
  rw [sub_eq_self, ceil_eq_zero_iff, Set.mem_Ioc, div_le_iff₀ hr', lt_div_iff₀ hr', neg_one_mul,
    zero_mul, neg_lt_neg_iff, cast_lt]
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    hr' : LT.lt 0 ↑r
    ⊢ And (LT.lt (HMod.hMod v r) r) (LE.le (Neg.neg ↑(HMod.hMod v r)) 0)
  -/
  exact ⟨mod_lt _ hr, by simp⟩
  /-
    🎉 no goals
  -/


/-- There are `b / r + [v % r < b % r]` numbers in `[0, b)` congruent to `v` mod `r`,
where `[·]` is the Iverson bracket. -/
theorem count_modEq_card (v : ℕ) :
    b.count (· ≡ v [MOD r]) = b / r + if v % r < b % r then 1 else 0 := by
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    ⊢ Eq (Nat.count (fun x => r.ModEq x v) b) (HAdd.hAdd (HDiv.hDiv b r) (ite (LT. …
  -/
  have hr' : 0 < (r : ℚ) := by positivity
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    hr' : LT.lt 0 ↑r
    ⊢ Eq (Nat.count (fun x => r.ModEq x v) b) (HAdd.hAdd (HDiv.hDiv b r) (ite (LT. …
  -/
  rw [← ofNat_inj, count_modEq_card_eq_ceil _ hr, cast_add]
  conv_lhs => rw [← div_add_mod b r, cast_add, cast_mul, ← add_sub, _root_.add_div,
    mul_div_cancel_left₀ _ hr'.ne', add_comm, Int.ceil_add_nat, add_comm]
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    hr' : LT.lt 0 ↑r
    ⊢ Eq (HAdd.hAdd (↑(HDiv.hDiv b r)) (Int.ceil (HDiv.hDiv (HSub.hSub ↑(HMod.hMod …
  -/
  rw [add_right_inj]
  /-
    b r : Nat
    hr : LT.lt 0 r
    v : Nat
    hr' : LT.lt 0 ↑r
    ⊢ Eq (Int.ceil (HDiv.hDiv (HSub.hSub ↑(HMod.hMod b r) ↑(HMod.hMod v r)) ↑r)) ↑ …
  -/
  split_ifs with h
  · rw [← cast_sub h.le, Int.ceil_eq_iff, div_le_iff₀ hr', lt_div_iff₀ hr', cast_one, Int.cast_one,
      sub_self, zero_mul, cast_pos, tsub_pos_iff_lt, one_mul, cast_le, tsub_le_iff_right]
    /-
      case pos
      b r : Nat
      hr : LT.lt 0 r
      v : Nat
      hr' : LT.lt 0 ↑r
      h : LT.lt (HMod.hMod v r) (HMod.hMod b r)
      ⊢ And (LT.lt (HMod.hMod v r) (HMod.hMod b r)) (LE.le (HMod.hMod b r) (HAdd.hAd …
    -/
    exact ⟨h, ((mod_lt _ hr).trans_le (by simp)).le⟩
    /-
      🎉 no goals
    -/
  · rw [cast_zero, ceil_eq_zero_iff, Set.mem_Ioc, div_le_iff₀ hr', lt_div_iff₀ hr', zero_mul,
      tsub_nonpos, ← neg_eq_neg_one_mul, neg_lt_sub_iff_lt_add, ← cast_add, cast_lt, cast_le]
    /-
      case neg
      b r : Nat
      hr : LT.lt 0 r
      v : Nat
      hr' : LT.lt 0 ↑r
      h : Not (LT.lt (HMod.hMod v r) (HMod.hMod b r))
      ⊢ And (LT.lt (HMod.hMod v r) (HAdd.hAdd r (HMod.hMod b r))) (LE.le (HMod.hMod  …
    -/
    exact ⟨(mod_lt _ hr).trans_le (by simp), not_lt.mp h⟩
    /-
      🎉 no goals
    -/


