@[to_additive]
theorem isPeriodicPt_mul_iff_pow_eq_one (x : G) : IsPeriodicPt (x * ·) n 1 ↔ x ^ n = 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    n : Nat
    x : G
    ⊢ Iff (Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1) (Eq (HPow.hPow  …
  -/
  rw [IsPeriodicPt, IsFixedPt, mul_left_iterate]; beta_reduce; rw [mul_one]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- `IsOfFinOrder` is a predicate on an element `x` of a monoid to be of finite order, i.e. there
exists `n ≥ 1` such that `x ^ n = 1`. -/
@[to_additive "`IsOfFinAddOrder` is a predicate on an element `a` of an
additive monoid to be of finite order, i.e. there exists `n ≥ 1` such that `n • a = 0`."]
def IsOfFinOrder (x : G) : Prop :=
  (1 : G) ∈ periodicPts (x * ·)


theorem isOfFinAddOrder_ofMul_iff : IsOfFinAddOrder (Additive.ofMul x) ↔ IsOfFinOrder x :=
  Iff.rfl


theorem isOfFinOrder_ofAdd_iff {α : Type*} [AddMonoid α] {x : α} :
    IsOfFinOrder (Multiplicative.ofAdd x) ↔ IsOfFinAddOrder x := Iff.rfl


@[to_additive]
theorem isOfFinOrder_iff_pow_eq_one : IsOfFinOrder x ↔ ∃ n, 0 < n ∧ x ^ n = 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Iff (IsOfFinOrder x) (Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow x n) 1))
  -/
  simp [IsOfFinOrder, mem_periodicPts, isPeriodicPt_mul_iff_pow_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨IsOfFinOrder.exists_pow_eq_one, _⟩ := isOfFinOrder_iff_pow_eq_one


@[to_additive]
lemma isOfFinOrder_iff_zpow_eq_one {G} [Group G] {x : G} :
    IsOfFinOrder x ↔ ∃ (n : ℤ), n ≠ 0 ∧ x ^ n = 1 := by
  /-
    G : Type u_6
    inst✝ : Group G
    x : G
    ⊢ Iff (IsOfFinOrder x) (Exists fun n => And (Ne n 0) (Eq (HPow.hPow x n) 1))
  -/
  rw [isOfFinOrder_iff_pow_eq_one]
  refine ⟨fun ⟨n, hn, hn'⟩ ↦ ⟨n, Int.natCast_ne_zero_iff_pos.mpr hn, zpow_natCast x n ▸ hn'⟩,
    fun ⟨n, hn, hn'⟩ ↦ ⟨n.natAbs, Int.natAbs_pos.mpr hn, ?_⟩⟩
  /-
    G : Type u_6
    inst✝ : Group G
    x : G
    x✝ : Exists fun n => And (Ne n 0) (Eq (HPow.hPow x n) 1)
    n : Int
    hn : Ne n 0
    hn' : Eq (HPow.hPow x n) 1
    ⊢ Eq (HPow.hPow x n.natAbs) 1
  -/
  cases' (Int.natAbs_eq_iff (a := n)).mp rfl with h h
    /-
      case inl
      G : Type u_6
      inst✝ : Group G
      x : G
      x✝ : Exists fun n => And (Ne n 0) (Eq (HPow.hPow x n) 1)
      n : Int
      hn : Ne n 0
      hn' : Eq (HPow.hPow x n) 1
      h : Eq n ↑n.natAbs
      ⊢ Eq (HPow.hPow x n.natAbs) 1
    -/
  · rwa [h, zpow_natCast] at hn'
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_6
      inst✝ : Group G
      x : G
      x✝ : Exists fun n => And (Ne n 0) (Eq (HPow.hPow x n) 1)
      n : Int
      hn : Ne n 0
      hn' : Eq (HPow.hPow x n) 1
      h : Eq n (Neg.neg ↑n.natAbs)
      ⊢ Eq (HPow.hPow x n.natAbs) 1
    -/
  · rwa [h, zpow_neg, inv_eq_one, zpow_natCast] at hn'
    /-
      🎉 no goals
    -/


/-- See also `injective_pow_iff_not_isOfFinOrder`. -/
@[to_additive "See also `injective_nsmul_iff_not_isOfFinAddOrder`."]
theorem not_isOfFinOrder_of_injective_pow {x : G} (h : Injective fun n : ℕ => x ^ n) :
    ¬IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Function.Injective fun n => HPow.hPow x n
    ⊢ Not (IsOfFinOrder x)
  -/
  simp_rw [isOfFinOrder_iff_pow_eq_one, not_exists, not_and]
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Function.Injective fun n => HPow.hPow x n
    ⊢ ∀ (x_1 : Nat), LT.lt 0 x_1 → Not (Eq (HPow.hPow x x_1) 1)
  -/
  intro n hn_pos hnx
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Function.Injective fun n => HPow.hPow x n
    n : Nat
    hn_pos : LT.lt 0 n
    hnx : Eq (HPow.hPow x n) 1
    ⊢ False
  -/
  rw [← pow_zero x] at hnx
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Function.Injective fun n => HPow.hPow x n
    n : Nat
    hn_pos : LT.lt 0 n
    hnx : Eq (HPow.hPow x n) (HPow.hPow x 0)
    ⊢ False
  -/
  rw [h hnx] at hn_pos
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Function.Injective fun n => HPow.hPow x n
    n : Nat
    hn_pos : LT.lt 0 0
    hnx : Eq (HPow.hPow x n) (HPow.hPow x 0)
    ⊢ False
  -/
  exact irrefl 0 hn_pos
  /-
    🎉 no goals
  -/


/-- 1 is of finite order in any monoid. -/
@[to_additive (attr := simp) "0 is of finite order in any additive monoid."]
theorem IsOfFinOrder.one : IsOfFinOrder (1 : G) :=
  isOfFinOrder_iff_pow_eq_one.mpr ⟨1, Nat.one_pos, one_pow 1⟩


@[to_additive]
alias isOfFinOrder_one := IsOfFinOrder.one

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive]
lemma IsOfFinOrder.pow {n : ℕ} : IsOfFinOrder a → IsOfFinOrder (a ^ n) := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    ⊢ IsOfFinOrder a → IsOfFinOrder (HPow.hPow a n)
  -/
  simp_rw [isOfFinOrder_iff_pow_eq_one]
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    ⊢ (Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow a n) 1)) → Exists fun n_1 => …
  -/
  rintro ⟨m, hm, ha⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n m : Nat
    hm : LT.lt 0 m
    ha : Eq (HPow.hPow a m) 1
    ⊢ Exists fun n_1 => And (LT.lt 0 n_1) (Eq (HPow.hPow (HPow.hPow a n) n_1) 1)
  -/
  exact ⟨m, hm, by simp [pow_right_comm _ n, ha]⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsOfFinOrder.of_pow {n : ℕ} (h : IsOfFinOrder (a ^ n)) (hn : n ≠ 0) : IsOfFinOrder a := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    h : IsOfFinOrder (HPow.hPow a n)
    hn : Ne n 0
    ⊢ IsOfFinOrder a
  -/
  rw [isOfFinOrder_iff_pow_eq_one] at *
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    h : Exists fun n_1 => And (LT.lt 0 n_1) (Eq (HPow.hPow (HPow.hPow a n) n_1) 1)
    hn : Ne n 0
    ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow a n) 1)
  -/
  rcases h with ⟨m, hm, ha⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    hn : Ne n 0
    m : Nat
    hm : LT.lt 0 m
    ha : Eq (HPow.hPow (HPow.hPow a n) m) 1
    ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow a n) 1)
  -/
  exact ⟨n * m, by positivity, by rwa [pow_mul]⟩
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma isOfFinOrder_pow {n : ℕ} : IsOfFinOrder (a ^ n) ↔ IsOfFinOrder a ∨ n = 0 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    n : Nat
    ⊢ Iff (IsOfFinOrder (HPow.hPow a n)) (Or (IsOfFinOrder a) (Eq n 0))
  -/
  rcases Decidable.eq_or_ne n 0 with rfl | hn
    /-
      case inl
      G : Type u_1
      inst✝ : Monoid G
      a : G
      ⊢ Iff (IsOfFinOrder (HPow.hPow a 0)) (Or (IsOfFinOrder a) (Eq 0 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝ : Monoid G
      a : G
      n : Nat
      hn : Ne n 0
      ⊢ Iff (IsOfFinOrder (HPow.hPow a n)) (Or (IsOfFinOrder a) (Eq n 0))
    -/
  · exact ⟨fun h ↦ .inl <| h.of_pow hn, fun h ↦ (h.resolve_right hn).pow⟩
    /-
      🎉 no goals
    -/


/-- Elements of finite order are of finite order in submonoids. -/
@[to_additive "Elements of finite order are of finite order in submonoids."]
theorem Submonoid.isOfFinOrder_coe {H : Submonoid G} {x : H} :
    IsOfFinOrder (x : G) ↔ IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    H : Submonoid G
    x : Subtype fun x => Membership.mem H x
    ⊢ Iff (IsOfFinOrder ↑x) (IsOfFinOrder x)
  -/
  rw [isOfFinOrder_iff_pow_eq_one, isOfFinOrder_iff_pow_eq_one]
  /-
    G : Type u_1
    inst✝ : Monoid G
    H : Submonoid G
    x : Subtype fun x => Membership.mem H x
    ⊢ Iff (Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow (↑x) n) 1)) (Exists fun  …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- The image of an element of finite order has finite order. -/
@[to_additive "The image of an element of finite additive order has finite additive order."]
theorem MonoidHom.isOfFinOrder [Monoid H] (f : G →* H) {x : G} (h : IsOfFinOrder x) :
    IsOfFinOrder <| f x :=
  isOfFinOrder_iff_pow_eq_one.mpr <| by
    /-
      G : Type u_1
      H : Type u_2
      inst✝¹ : Monoid G
      inst✝ : Monoid H
      f : MonoidHom G H
      x : G
      h : IsOfFinOrder x
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow (f x) n) 1)
    -/
    obtain ⟨n, npos, hn⟩ := h.exists_pow_eq_one
    /-
      case intro.intro
      G : Type u_1
      H : Type u_2
      inst✝¹ : Monoid G
      inst✝ : Monoid H
      f : MonoidHom G H
      x : G
      h : IsOfFinOrder x
      n : Nat
      npos : LT.lt 0 n
      hn : Eq (HPow.hPow x n) 1
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow (f x) n) 1)
    -/
    exact ⟨n, npos, by rw [← f.map_pow, hn, f.map_one]⟩
    /-
      🎉 no goals
    -/


/-- If a direct product has finite order then so does each component. -/
@[to_additive "If a direct product has finite additive order then so does each component."]
theorem IsOfFinOrder.apply {η : Type*} {Gs : η → Type*} [∀ i, Monoid (Gs i)] {x : ∀ i, Gs i}
    (h : IsOfFinOrder x) : ∀ i, IsOfFinOrder (x i) := by
  /-
    η : Type u_6
    Gs : η → Type u_7
    inst✝ : (i : η) → Monoid (Gs i)
    x : (i : η) → Gs i
    h : IsOfFinOrder x
    ⊢ ∀ (i : η), IsOfFinOrder (x i)
  -/
  obtain ⟨n, npos, hn⟩ := h.exists_pow_eq_one
  /-
    case intro.intro
    η : Type u_6
    Gs : η → Type u_7
    inst✝ : (i : η) → Monoid (Gs i)
    x : (i : η) → Gs i
    h : IsOfFinOrder x
    n : Nat
    npos : LT.lt 0 n
    hn : Eq (HPow.hPow x n) 1
    ⊢ ∀ (i : η), IsOfFinOrder (x i)
  -/
  exact fun _ => isOfFinOrder_iff_pow_eq_one.mpr ⟨n, npos, (congr_fun hn.symm _).symm⟩
  /-
    🎉 no goals
  -/


/-- The submonoid generated by an element is a group if that element has finite order. -/
@[to_additive "The additive submonoid generated by an element is
an additive group if that element has finite order."]
noncomputable abbrev IsOfFinOrder.groupPowers (hx : IsOfFinOrder x) :
    Group (Submonoid.powers x) := by
  /-
    G : Type u_1
    H : Type u_2
    A : Type u_3
    α : Type u_4
    β : Type u_5
    inst✝ : Monoid G
    a b x y : G
    n m : Nat
    hx : IsOfFinOrder x
    ⊢ Group (Subtype fun x_1 => Membership.mem (Submonoid.powers x) x_1)
  -/
  obtain ⟨hpos, hx⟩ := hx.exists_pow_eq_one.choose_spec
  /-
    case intro
    G : Type u_1
    H : Type u_2
    A : Type u_3
    α : Type u_4
    β : Type u_5
    inst✝ : Monoid G
    a b x y : G
    n m : Nat
    hx✝ : IsOfFinOrder x
    hpos : LT.lt 0 ⋯.choose
    hx : Eq (HPow.hPow x ⋯.choose) 1
    ⊢ Group (Subtype fun x_1 => Membership.mem (Submonoid.powers x) x_1)
  -/
  exact Submonoid.groupPowers hpos hx
  /-
    🎉 no goals
  -/


/-- `orderOf x` is the order of the element `x`, i.e. the `n ≥ 1`, s.t. `x ^ n = 1` if it exists.
Otherwise, i.e. if `x` is of infinite order, then `orderOf x` is `0` by convention. -/
@[to_additive
  "`addOrderOf a` is the order of the element `a`, i.e. the `n ≥ 1`, s.t. `n • a = 0` if it
  exists. Otherwise, i.e. if `a` is of infinite order, then `addOrderOf a` is `0` by convention."]
noncomputable def orderOf (x : G) : ℕ :=
  minimalPeriod (x * ·) 1


@[simp]
theorem addOrderOf_ofMul_eq_orderOf (x : G) : addOrderOf (Additive.ofMul x) = orderOf x :=
  rfl


@[simp]
lemma orderOf_ofAdd_eq_addOrderOf {α : Type*} [AddMonoid α] (a : α) :
    orderOf (Multiplicative.ofAdd a) = addOrderOf a := rfl


@[to_additive]
protected lemma IsOfFinOrder.orderOf_pos (h : IsOfFinOrder x) : 0 < orderOf x :=
  minimalPeriod_pos_of_mem_periodicPts h


@[to_additive addOrderOf_nsmul_eq_zero]
theorem pow_orderOf_eq_one (x : G) : x ^ orderOf x = 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Eq (HPow.hPow x (orderOf x)) 1
  -/
  convert Eq.trans _ (isPeriodicPt_minimalPeriod (x * ·) 1)
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed in the middle of the rewrite
  /-
    case convert_2
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Eq (HPow.hPow x (orderOf x)) (Nat.iterate (fun x_1 => HMul.hMul x x_1) (Func …
  -/
  rw [orderOf, mul_left_iterate]; beta_reduce; rw [mul_one]
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
theorem orderOf_eq_zero (h : ¬IsOfFinOrder x) : orderOf x = 0 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    h : Not (IsOfFinOrder x)
    ⊢ Eq (orderOf x) 0
  -/
  rwa [orderOf, minimalPeriod, dif_neg]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_eq_zero_iff : orderOf x = 0 ↔ ¬IsOfFinOrder x :=
  ⟨fun h H ↦ H.orderOf_pos.ne' h, orderOf_eq_zero⟩


@[to_additive]
theorem orderOf_eq_zero_iff' : orderOf x = 0 ↔ ∀ n : ℕ, 0 < n → x ^ n ≠ 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Iff (Eq (orderOf x) 0) (∀ (n : Nat), LT.lt 0 n → Ne (HPow.hPow x n) 1)
  -/
  simp_rw [orderOf_eq_zero_iff, isOfFinOrder_iff_pow_eq_one, not_exists, not_and]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_eq_iff {n} (h : 0 < n) :
    orderOf x = n ↔ x ^ n = 1 ∧ ∀ m, m < n → 0 < m → x ^ m ≠ 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : LT.lt 0 n
    ⊢ Iff (Eq (orderOf x) n) (And (Eq (HPow.hPow x n) 1) (∀ (m : Nat), LT.lt m n → …
  -/
  simp_rw [Ne, ← isPeriodicPt_mul_iff_pow_eq_one, orderOf, minimalPeriod]
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : LT.lt 0 n
    ⊢ Iff (Eq (dite (Membership.mem (Function.periodicPts fun x_1 => HMul.hMul x x …
  -/
  split_ifs with h1
  · classical
    rw [find_eq_iff]
    simp only [h, true_and]
    push_neg
    rfl
    /-
      case neg
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : LT.lt 0 n
      h1 : Not (Membership.mem (Function.periodicPts fun x_1 => HMul.hMul x x_1) 1)
      ⊢ Iff (Eq 0 n) (And (Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1) ( …
    -/
  · rw [iff_false_left h.ne]
    /-
      case neg
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : LT.lt 0 n
      h1 : Not (Membership.mem (Function.periodicPts fun x_1 => HMul.hMul x x_1) 1)
      ⊢ Not (And (Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1) (∀ (m : Na …
    -/
    rintro ⟨h', -⟩
    /-
      case neg.intro
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : LT.lt 0 n
      h1 : Not (Membership.mem (Function.periodicPts fun x_1 => HMul.hMul x x_1) 1)
      h' : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1
      ⊢ False
    -/
    exact h1 ⟨n, h, h'⟩
    /-
      🎉 no goals
    -/


/-- A group element has finite order iff its order is positive. -/
@[to_additive
      "A group element has finite additive order iff its order is positive."]
theorem orderOf_pos_iff : 0 < orderOf x ↔ IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Iff (LT.lt 0 (orderOf x)) (IsOfFinOrder x)
  -/
  rw [iff_not_comm.mp orderOf_eq_zero_iff, pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsOfFinOrder.mono [Monoid β] {y : β} (hx : IsOfFinOrder x) (h : orderOf y ∣ orderOf x) :
                         /-
                           G : Type u_1
                           β : Type u_5
                           inst✝¹ : Monoid G
                           x : G
                           inst✝ : Monoid β
                           y : β
                           hx : IsOfFinOrder x
                           h : Dvd.dvd (orderOf y) (orderOf x)
                           ⊢ IsOfFinOrder y
                         -/
    IsOfFinOrder y := by rw [← orderOf_pos_iff] at hx ⊢; exact Nat.pos_of_dvd_of_pos h hx
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
theorem pow_ne_one_of_lt_orderOf (n0 : n ≠ 0) (h : n < orderOf x) : x ^ n ≠ 1 := fun j =>
  not_isPeriodicPt_of_pos_of_lt_minimalPeriod n0 h ((isPeriodicPt_mul_iff_pow_eq_one x).mpr j)

@[deprecated (since := "2024-07-20")] alias pow_ne_one_of_lt_orderOf' := pow_ne_one_of_lt_orderOf

@[deprecated (since := "2024-07-20")] alias
  nsmul_ne_zero_of_lt_addOrderOf' := nsmul_ne_zero_of_lt_addOrderOf


@[to_additive]
theorem orderOf_le_of_pow_eq_one (hn : 0 < n) (h : x ^ n = 1) : orderOf x ≤ n :=
                                       /-
                                         G : Type u_1
                                         inst✝ : Monoid G
                                         x : G
                                         n : Nat
                                         hn : LT.lt 0 n
                                         h : Eq (HPow.hPow x n) 1
                                         ⊢ Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1
                                       -/
  IsPeriodicPt.minimalPeriod_le hn (by rwa [isPeriodicPt_mul_iff_pow_eq_one])
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive (attr := simp)]
theorem orderOf_one : orderOf (1 : G) = 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    ⊢ Eq (orderOf 1) 1
  -/
  rw [orderOf, ← minimalPeriod_id (x := (1 : G)), ← one_mul_eq_id]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) AddMonoid.addOrderOf_eq_one_iff]
theorem orderOf_eq_one_iff : orderOf x = 1 ↔ x = 1 := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    ⊢ Iff (Eq (orderOf x) 1) (Eq x 1)
  -/
  rw [orderOf, minimalPeriod_eq_one_iff_isFixedPt, IsFixedPt, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) mod_addOrderOf_nsmul]
lemma pow_mod_orderOf (x : G) (n : ℕ) : x ^ (n % orderOf x) = x ^ n :=
  calc
    x ^ (n % orderOf x) = x ^ (n % orderOf x + orderOf x * (n / orderOf x)) := by
        /-
          G : Type u_1
          inst✝ : Monoid G
          x : G
          n : Nat
          ⊢ Eq (HPow.hPow x (HMod.hMod n (orderOf x))) (HPow.hPow x (HAdd.hAdd (HMod.hMo …
        -/
        simp [pow_add, pow_mul, pow_orderOf_eq_one]
        /-
          🎉 no goals
        -/
                    /-
                      G : Type u_1
                      inst✝ : Monoid G
                      x : G
                      n : Nat
                      ⊢ Eq (HPow.hPow x (HAdd.hAdd (HMod.hMod n (orderOf x)) (HMul.hMul (orderOf x)  …
                    -/
    _ = x ^ n := by rw [Nat.mod_add_div]
                    /-
                      🎉 no goals
                    -/


@[to_additive]
theorem orderOf_dvd_of_pow_eq_one (h : x ^ n = 1) : orderOf x ∣ n :=
  IsPeriodicPt.minimalPeriod_dvd ((isPeriodicPt_mul_iff_pow_eq_one _).mpr h)


@[to_additive]
theorem orderOf_dvd_iff_pow_eq_one {n : ℕ} : orderOf x ∣ n ↔ x ^ n = 1 :=
               /-
                 G : Type u_1
                 inst✝ : Monoid G
                 x : G
                 n : Nat
                 h : Dvd.dvd (orderOf x) n
                 ⊢ Eq (HPow.hPow x n) 1
               -/
  ⟨fun h => by rw [← pow_mod_orderOf, Nat.mod_eq_zero_of_dvd h, _root_.pow_zero],
               /-
                 🎉 no goals
               -/
    orderOf_dvd_of_pow_eq_one⟩


@[to_additive addOrderOf_smul_dvd]
theorem orderOf_pow_dvd (n : ℕ) : orderOf (x ^ n) ∣ orderOf x := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    ⊢ Dvd.dvd (orderOf (HPow.hPow x n)) (orderOf x)
  -/
  rw [orderOf_dvd_iff_pow_eq_one, pow_right_comm, pow_orderOf_eq_one, one_pow]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma pow_injOn_Iio_orderOf : (Set.Iio <| orderOf x).InjOn (x ^ ·) := by
  simpa only [mul_left_iterate, mul_one]
    using iterate_injOn_Iio_minimalPeriod (f := (x * ·)) (x := 1)


@[to_additive]
protected lemma IsOfFinOrder.mem_powers_iff_mem_range_orderOf [DecidableEq G]
    (hx : IsOfFinOrder x) :
    y ∈ Submonoid.powers x ↔ y ∈ (Finset.range (orderOf x)).image (x ^ ·) :=
  Finset.mem_range_iff_mem_finset_range_of_mod_eq' hx.orderOf_pos <| pow_mod_orderOf _


@[to_additive]
protected lemma IsOfFinOrder.powers_eq_image_range_orderOf [DecidableEq G] (hx : IsOfFinOrder x) :
    (Submonoid.powers x : Set G) = (Finset.range (orderOf x)).image (x ^ ·) :=
  Set.ext fun _ ↦ hx.mem_powers_iff_mem_range_orderOf

@[deprecated (since := "2024-02-21")]
alias IsOfFinAddOrder.powers_eq_image_range_orderOf :=
  IsOfFinAddOrder.multiples_eq_image_range_addOrderOf


@[to_additive]
theorem pow_eq_one_iff_modEq : x ^ n = 1 ↔ n ≡ 0 [MOD orderOf x] := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    ⊢ Iff (Eq (HPow.hPow x n) 1) ((orderOf x).ModEq n 0)
  -/
  rw [modEq_zero_iff_dvd, orderOf_dvd_iff_pow_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_map_dvd {H : Type*} [Monoid H] (ψ : G →* H) (x : G) :
    orderOf (ψ x) ∣ orderOf x := by
  /-
    G : Type u_1
    inst✝¹ : Monoid G
    H : Type u_6
    inst✝ : Monoid H
    ψ : MonoidHom G H
    x : G
    ⊢ Dvd.dvd (orderOf (ψ x)) (orderOf x)
  -/
  apply orderOf_dvd_of_pow_eq_one
  /-
    case h
    G : Type u_1
    inst✝¹ : Monoid G
    H : Type u_6
    inst✝ : Monoid H
    ψ : MonoidHom G H
    x : G
    ⊢ Eq (HPow.hPow (ψ x) (orderOf x)) 1
  -/
  rw [← map_pow, pow_orderOf_eq_one]
  /-
    case h
    G : Type u_1
    inst✝¹ : Monoid G
    H : Type u_6
    inst✝ : Monoid H
    ψ : MonoidHom G H
    x : G
    ⊢ Eq (ψ 1) 1
  -/
  apply map_one
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_pow_eq_self_of_coprime (h : n.Coprime (orderOf x)) : ∃ m : ℕ, (x ^ n) ^ m = x := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : n.Coprime (orderOf x)
    ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
  -/
  by_cases h0 : orderOf x = 0
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : n.Coprime (orderOf x)
      h0 : Eq (orderOf x) 0
      ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
    -/
  · rw [h0, coprime_zero_right] at h
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : Eq n 1
      h0 : Eq (orderOf x) 0
      ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
    -/
    exact ⟨1, by rw [h, pow_one, pow_one]⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : n.Coprime (orderOf x)
    h0 : Not (Eq (orderOf x) 0)
    ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
  -/
  by_cases h1 : orderOf x = 1
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      h : n.Coprime (orderOf x)
      h0 : Not (Eq (orderOf x) 0)
      h1 : Eq (orderOf x) 1
      ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
    -/
  · exact ⟨0, by rw [orderOf_eq_one_iff.mp h1, one_pow, one_pow]⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : n.Coprime (orderOf x)
    h0 : Not (Eq (orderOf x) 0)
    h1 : Not (Eq (orderOf x) 1)
    ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
  -/
  obtain ⟨m, h⟩ := exists_mul_emod_eq_one_of_coprime h (one_lt_iff_ne_zero_and_ne_one.mpr ⟨h0, h1⟩)
  /-
    case neg.intro
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h✝ : n.Coprime (orderOf x)
    h0 : Not (Eq (orderOf x) 0)
    h1 : Not (Eq (orderOf x) 1)
    m : Nat
    h : Eq (HMod.hMod (HMul.hMul n m) (orderOf x)) 1
    ⊢ Exists fun m => Eq (HPow.hPow (HPow.hPow x n) m) x
  -/
  exact ⟨m, by rw [← pow_mul, ← pow_mod_orderOf, h, pow_one]⟩
  /-
    🎉 no goals
  -/


/-- If `x^n = 1`, but `x^(n/p) ≠ 1` for all prime factors `p` of `n`,
then `x` has order `n` in `G`. -/
@[to_additive addOrderOf_eq_of_nsmul_and_div_prime_nsmul "If `n * x = 0`, but `n/p * x ≠ 0` for
all prime factors `p` of `n`, then `x` has order `n` in `G`."]
theorem orderOf_eq_of_pow_and_pow_div_prime (hn : 0 < n) (hx : x ^ n = 1)
    (hd : ∀ p : ℕ, p.Prime → p ∣ n → x ^ (n / p) ≠ 1) : orderOf x = n := by
  -- Let `a` be `n/(orderOf x)`, and show `a = 1`
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 1
    hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
    ⊢ Eq (orderOf x) n
  -/
  cases' exists_eq_mul_right_of_dvd (orderOf_dvd_of_pow_eq_one hx) with a ha
  /-
    case intro
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 1
    hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
    a : Nat
    ha : Eq n (HMul.hMul (orderOf x) a)
    ⊢ Eq (orderOf x) n
  -/
  suffices a = 1 by simp [this, ha]
  -- Assume `a` is not one...
  /-
    case intro
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 1
    hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
    a : Nat
    ha : Eq n (HMul.hMul (orderOf x) a)
    ⊢ Eq a 1
  -/
  by_contra h
  have a_min_fac_dvd_p_sub_one : a.minFac ∣ n := by
    obtain ⟨b, hb⟩ : ∃ b : ℕ, a = b * a.minFac := exists_eq_mul_left_of_dvd a.minFac_dvd
    rw [hb, ← mul_assoc] at ha
    exact Dvd.intro_left (orderOf x * b) ha.symm
  -- Use the minimum prime factor of `a` as `p`.
  /-
    case intro
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 1
    hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
    a : Nat
    ha : Eq n (HMul.hMul (orderOf x) a)
    h : Not (Eq a 1)
    a_min_fac_dvd_p_sub_one : Dvd.dvd a.minFac n
    ⊢ False
  -/
  refine hd a.minFac (Nat.minFac_prime h) a_min_fac_dvd_p_sub_one ?_
  rw [← orderOf_dvd_iff_pow_eq_one, Nat.dvd_div_iff_mul_dvd a_min_fac_dvd_p_sub_one, ha, mul_comm,
    Nat.mul_dvd_mul_iff_left (IsOfFinOrder.orderOf_pos _)]
    /-
      case intro
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
      a : Nat
      ha : Eq n (HMul.hMul (orderOf x) a)
      h : Not (Eq a 1)
      a_min_fac_dvd_p_sub_one : Dvd.dvd a.minFac n
      ⊢ Dvd.dvd a.minFac a
    -/
  · exact Nat.minFac_dvd a
    /-
      🎉 no goals
    -/
    /-
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
      a : Nat
      ha : Eq n (HMul.hMul (orderOf x) a)
      h : Not (Eq a 1)
      a_min_fac_dvd_p_sub_one : Dvd.dvd a.minFac n
      ⊢ IsOfFinOrder x
    -/
  · rw [isOfFinOrder_iff_pow_eq_one]
    /-
      G : Type u_1
      inst✝ : Monoid G
      x : G
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      hd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p n → Ne (HPow.hPow x (HDiv.hDiv n p)) 1
      a : Nat
      ha : Eq n (HMul.hMul (orderOf x) a)
      h : Not (Eq a 1)
      a_min_fac_dvd_p_sub_one : Dvd.dvd a.minFac n
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow x n) 1)
    -/
    exact Exists.intro n (id ⟨hn, hx⟩)
    /-
      🎉 no goals
    -/


@[to_additive]
theorem orderOf_eq_orderOf_iff {H : Type*} [Monoid H] {y : H} :
    orderOf x = orderOf y ↔ ∀ n : ℕ, x ^ n = 1 ↔ y ^ n = 1 := by
  /-
    G : Type u_1
    inst✝¹ : Monoid G
    x : G
    H : Type u_6
    inst✝ : Monoid H
    y : H
    ⊢ Iff (Eq (orderOf x) (orderOf y)) (∀ (n : Nat), Iff (Eq (HPow.hPow x n) 1) (E …
  -/
  simp_rw [← isPeriodicPt_mul_iff_pow_eq_one, ← minimalPeriod_eq_minimalPeriod_iff, orderOf]
  /-
    🎉 no goals
  -/


/-- An injective homomorphism of monoids preserves orders of elements. -/
@[to_additive "An injective homomorphism of additive monoids preserves orders of elements."]
theorem orderOf_injective {H : Type*} [Monoid H] (f : G →* H) (hf : Function.Injective f) (x : G) :
    orderOf (f x) = orderOf x := by
  /-
    G : Type u_1
    inst✝¹ : Monoid G
    H : Type u_6
    inst✝ : Monoid H
    f : MonoidHom G H
    hf : Function.Injective ⇑f
    x : G
    ⊢ Eq (orderOf (f x)) (orderOf x)
  -/
  simp_rw [orderOf_eq_orderOf_iff, ← f.map_pow, ← f.map_one, hf.eq_iff, forall_const]
  /-
    🎉 no goals
  -/


/-- A multiplicative equivalence preserves orders of elements. -/
@[to_additive (attr := simp) "An additive equivalence preserves orders of elements."]
lemma MulEquiv.orderOf_eq {H : Type*} [Monoid H] (e : G ≃* H) (x : G) :
    orderOf (e x) = orderOf x :=
  orderOf_injective e.toMonoidHom e.injective x


@[to_additive]
theorem Function.Injective.isOfFinOrder_iff [Monoid H] {f : G →* H} (hf : Injective f) :
    IsOfFinOrder (f x) ↔ IsOfFinOrder x := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Monoid G
    x : G
    inst✝ : Monoid H
    f : MonoidHom G H
    hf : Function.Injective ⇑f
    ⊢ Iff (IsOfFinOrder (f x)) (IsOfFinOrder x)
  -/
  rw [← orderOf_pos_iff, orderOf_injective f hf x, ← orderOf_pos_iff]
  /-
    🎉 no goals
  -/


@[to_additive (attr := norm_cast, simp)]
theorem orderOf_submonoid {H : Submonoid G} (y : H) : orderOf (y : G) = orderOf y :=
  orderOf_injective H.subtype Subtype.coe_injective y


@[to_additive]
theorem orderOf_units {y : Gˣ} : orderOf (y : G) = orderOf y :=
  orderOf_injective (Units.coeHom G) Units.ext y


/-- If the order of `x` is finite, then `x` is a unit with inverse `x ^ (orderOf x - 1)`. -/
@[simps]
noncomputable
def IsOfFinOrder.unit {M} [Monoid M] {x : M} (hx : IsOfFinOrder x) : Mˣ :=
⟨x, x ^ (orderOf x - 1),
     /-
       G : Type u_1
       H : Type u_2
       A : Type u_3
       α : Type u_4
       β : Type u_5
       inst✝¹ : Monoid G
       a b x✝ y : G
       n m : Nat
       M : Type ?u.49933
       inst✝ : Monoid M
       x : M
       hx : IsOfFinOrder x
       ⊢ Eq (HMul.hMul x (HPow.hPow x (HSub.hSub (orderOf x) 1))) 1
     -/
  by rw [← _root_.pow_succ', tsub_add_cancel_of_le (by exact hx.orderOf_pos), pow_orderOf_eq_one],
     /-
       🎉 no goals
     -/
     /-
       G : Type u_1
       H : Type u_2
       A : Type u_3
       α : Type u_4
       β : Type u_5
       inst✝¹ : Monoid G
       a b x✝ y : G
       n m : Nat
       M : Type ?u.49933
       inst✝ : Monoid M
       x : M
       hx : IsOfFinOrder x
       ⊢ Eq (HMul.hMul (HPow.hPow x (HSub.hSub (orderOf x) 1)) x) 1
     -/
  by rw [← _root_.pow_succ, tsub_add_cancel_of_le (by exact hx.orderOf_pos), pow_orderOf_eq_one]⟩
     /-
       🎉 no goals
     -/


lemma IsOfFinOrder.isUnit {M} [Monoid M] {x : M} (hx : IsOfFinOrder x) : IsUnit x := ⟨hx.unit, rfl⟩


@[to_additive]
theorem orderOf_pow' (h : n ≠ 0) : orderOf (x ^ n) = orderOf x / gcd (orderOf x) n := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : Ne n 0
    ⊢ Eq (orderOf (HPow.hPow x n)) (HDiv.hDiv (orderOf x) ((orderOf x).gcd n))
  -/
  unfold orderOf
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : Ne n 0
    ⊢ Eq (Function.minimalPeriod (fun x_1 => HMul.hMul (HPow.hPow x n) x_1) 1) (HD …
  -/
  rw [← minimalPeriod_iterate_eq_div_gcd h, mul_left_iterate]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orderOf_pow_of_dvd {x : G} {n : ℕ} (hn : n ≠ 0) (dvd : n ∣ orderOf x) :
                                          /-
                                            G : Type u_1
                                            inst✝ : Monoid G
                                            x : G
                                            n : Nat
                                            hn : Ne n 0
                                            dvd : Dvd.dvd n (orderOf x)
                                            ⊢ Eq (orderOf (HPow.hPow x n)) (HDiv.hDiv (orderOf x) n)
                                          -/
    orderOf (x ^ n) = orderOf x / n := by rw [orderOf_pow' _ hn, Nat.gcd_eq_right dvd]
                                          /-
                                            🎉 no goals
                                          -/


@[to_additive]
lemma orderOf_pow_orderOf_div {x : G} {n : ℕ} (hx : orderOf x ≠ 0) (hn : n ∣ orderOf x) :
    orderOf (x ^ (orderOf x / n)) = n := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hx : Ne (orderOf x) 0
    hn : Dvd.dvd n (orderOf x)
    ⊢ Eq (orderOf (HPow.hPow x (HDiv.hDiv (orderOf x) n))) n
  -/
  rw [orderOf_pow_of_dvd _ (Nat.div_dvd_of_dvd hn), Nat.div_div_self hn hx]
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    hx : Ne (orderOf x) 0
    hn : Dvd.dvd n (orderOf x)
    ⊢ Ne (HDiv.hDiv (orderOf x) n) 0
  -/
  rw [← Nat.div_mul_cancel hn] at hx; exact left_ne_zero_of_mul hx
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
protected lemma IsOfFinOrder.orderOf_pow (h : IsOfFinOrder x) :
    orderOf (x ^ n) = orderOf x / gcd (orderOf x) n := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : IsOfFinOrder x
    ⊢ Eq (orderOf (HPow.hPow x n)) (HDiv.hDiv (orderOf x) ((orderOf x).gcd n))
  -/
  unfold orderOf
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n : Nat
    h : IsOfFinOrder x
    ⊢ Eq (Function.minimalPeriod (fun x_1 => HMul.hMul (HPow.hPow x n) x_1) 1) (HD …
  -/
  rw [← minimalPeriod_iterate_eq_div_gcd' h, mul_left_iterate]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Nat.Coprime.orderOf_pow (h : (orderOf y).Coprime m) : orderOf (y ^ m) = orderOf y := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    y : G
    m : Nat
    h : (orderOf y).Coprime m
    ⊢ Eq (orderOf (HPow.hPow y m)) (orderOf y)
  -/
  by_cases hg : IsOfFinOrder y
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      y : G
      m : Nat
      h : (orderOf y).Coprime m
      hg : IsOfFinOrder y
      ⊢ Eq (orderOf (HPow.hPow y m)) (orderOf y)
    -/
  · rw [hg.orderOf_pow y m , h.gcd_eq_one, Nat.div_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      inst✝ : Monoid G
      y : G
      m : Nat
      h : (orderOf y).Coprime m
      hg : Not (IsOfFinOrder y)
      ⊢ Eq (orderOf (HPow.hPow y m)) (orderOf y)
    -/
  · rw [m.coprime_zero_left.1 (orderOf_eq_zero hg ▸ h), pow_one]
    /-
      🎉 no goals
    -/


@[to_additive]
lemma IsOfFinOrder.natCard_powers_le_orderOf (ha : IsOfFinOrder a) :
    Nat.card (powers a : Set G) ≤ orderOf a := by
  classical
  simpa [ha.powers_eq_image_range_orderOf, Finset.card_range, Nat.Iio_eq_range]
    using Finset.card_image_le (s := Finset.range (orderOf a))


@[to_additive]
lemma IsOfFinOrder.finite_powers (ha : IsOfFinOrder a) : (powers a : Set G).Finite := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    a : G
    ha : IsOfFinOrder a
    ⊢ (↑(Submonoid.powers a)).Finite
  -/
  classical rw [ha.powers_eq_image_range_orderOf]; exact Finset.finite_toSet _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_mul_dvd_lcm (h : Commute x y) :
    orderOf (x * y) ∣ Nat.lcm (orderOf x) (orderOf y) := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    ⊢ Dvd.dvd (orderOf (HMul.hMul x y)) ((orderOf x).lcm (orderOf y))
  -/
  rw [orderOf, ← comp_mul_left]
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    ⊢ Dvd.dvd (Function.minimalPeriod (Function.comp (fun x_1 => HMul.hMul x x_1)  …
  -/
  exact Function.Commute.minimalPeriod_of_comp_dvd_lcm h.function_commute_mul_left
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_dvd_lcm_mul (h : Commute x y):
    orderOf y ∣ Nat.lcm (orderOf x) (orderOf (x * y)) := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    ⊢ Dvd.dvd (orderOf y) ((orderOf x).lcm (orderOf (HMul.hMul x y)))
  -/
  by_cases h0 : orderOf x = 0
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      h0 : Eq (orderOf x) 0
      ⊢ Dvd.dvd (orderOf y) ((orderOf x).lcm (orderOf (HMul.hMul x y)))
    -/
  · rw [h0, lcm_zero_left]
    /-
      case pos
      G : Type u_1
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      h0 : Eq (orderOf x) 0
      ⊢ Dvd.dvd (orderOf y) 0
    -/
    apply dvd_zero
    /-
      🎉 no goals
    -/
  conv_lhs =>
    rw [← one_mul y, ← pow_orderOf_eq_one x, ← succ_pred_eq_of_pos (Nat.pos_of_ne_zero h0),
      _root_.pow_succ, mul_assoc]
  exact
    (((Commute.refl x).mul_right h).pow_left _).orderOf_mul_dvd_lcm.trans
      (lcm_dvd_iff.2 ⟨(orderOf_pow_dvd _).trans (dvd_lcm_left _ _), dvd_lcm_right _ _⟩)


@[to_additive addOrderOf_add_dvd_mul_addOrderOf]
theorem orderOf_mul_dvd_mul_orderOf (h : Commute x y):
    orderOf (x * y) ∣ orderOf x * orderOf y :=
  dvd_trans h.orderOf_mul_dvd_lcm (lcm_dvd_mul _ _)


@[to_additive addOrderOf_add_eq_mul_addOrderOf_of_coprime]
theorem orderOf_mul_eq_mul_orderOf_of_coprime (h : Commute x y)
    (hco : (orderOf x).Coprime (orderOf y)) : orderOf (x * y) = orderOf x * orderOf y := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hco : (orderOf x).Coprime (orderOf y)
    ⊢ Eq (orderOf (HMul.hMul x y)) (HMul.hMul (orderOf x) (orderOf y))
  -/
  rw [orderOf, ← comp_mul_left]
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hco : (orderOf x).Coprime (orderOf y)
    ⊢ Eq (Function.minimalPeriod (Function.comp (fun x_1 => HMul.hMul x x_1) fun x …
  -/
  exact h.function_commute_mul_left.minimalPeriod_of_comp_eq_mul_of_coprime hco
  /-
    🎉 no goals
  -/


/-- Commuting elements of finite order are closed under multiplication. -/
@[to_additive "Commuting elements of finite additive order are closed under addition."]
theorem isOfFinOrder_mul (h : Commute x y) (hx : IsOfFinOrder x) (hy : IsOfFinOrder y) :
    IsOfFinOrder (x * y) :=
  orderOf_pos_iff.mp <|
    pos_of_dvd_of_pos h.orderOf_mul_dvd_mul_orderOf <| mul_pos hx.orderOf_pos hy.orderOf_pos


/-- If each prime factor of `orderOf x` has higher multiplicity in `orderOf y`, and `x` commutes
  with `y`, then `x * y` has the same order as `y`. -/
@[to_additive addOrderOf_add_eq_right_of_forall_prime_mul_dvd
  "If each prime factor of
  `addOrderOf x` has higher multiplicity in `addOrderOf y`, and `x` commutes with `y`,
  then `x + y` has the same order as `y`."]
theorem orderOf_mul_eq_right_of_forall_prime_mul_dvd (h : Commute x y) (hy : IsOfFinOrder y)
    (hdvd : ∀ p : ℕ, p.Prime → p ∣ orderOf x → p * orderOf x ∣ orderOf y) :
    orderOf (x * y) = orderOf y := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    ⊢ Eq (orderOf (HMul.hMul x y)) (orderOf y)
  -/
  have hoy := hy.orderOf_pos
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    ⊢ Eq (orderOf (HMul.hMul x y)) (orderOf y)
  -/
  have hxy := dvd_of_forall_prime_mul_dvd hdvd
  /-
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    ⊢ Eq (orderOf (HMul.hMul x y)) (orderOf y)
  -/
  apply orderOf_eq_of_pow_and_pow_div_prime hoy <;> simp only [Ne, ← orderOf_dvd_iff_pow_eq_one]
    /-
      case hx
      G : Type u_1
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      hy : IsOfFinOrder y
      hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
      hoy : LT.lt 0 (orderOf y)
      hxy : Dvd.dvd (orderOf x) (orderOf y)
      ⊢ Dvd.dvd (orderOf (HMul.hMul x y)) (orderOf y)
    -/
  · exact h.orderOf_mul_dvd_lcm.trans (lcm_dvd hxy dvd_rfl)
    /-
      🎉 no goals
    -/
  /-
    case hd
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    ⊢ ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf y) → Not (Dvd.dvd (orderOf (HM …
  -/
  refine fun p hp hpy hd => hp.ne_one ?_
  /-
    case hd
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p (orderOf y)
    hd : Dvd.dvd (orderOf (HMul.hMul x y)) (HDiv.hDiv (orderOf y) p)
    ⊢ Eq p 1
  -/
  rw [← Nat.dvd_one, ← mul_dvd_mul_iff_right hoy.ne', one_mul, ← dvd_div_iff_mul_dvd hpy]
  /-
    case hd
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p (orderOf y)
    hd : Dvd.dvd (orderOf (HMul.hMul x y)) (HDiv.hDiv (orderOf y) p)
    ⊢ Dvd.dvd (orderOf y) (HDiv.hDiv (orderOf y) p)
  -/
  refine (orderOf_dvd_lcm_mul h).trans (lcm_dvd ((dvd_div_iff_mul_dvd hpy).2 ?_) hd)
  /-
    case hd
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p (orderOf y)
    hd : Dvd.dvd (orderOf (HMul.hMul x y)) (HDiv.hDiv (orderOf y) p)
    ⊢ Dvd.dvd (HMul.hMul p (orderOf x)) (orderOf y)
  -/
  by_cases h : p ∣ orderOf x
  /-
    case pos
    G : Type u_1
    inst✝ : Monoid G
    x y : G
    h✝ : Commute x y
    hy : IsOfFinOrder y
    hdvd : ∀ (p : Nat), Nat.Prime p → Dvd.dvd p (orderOf x) → Dvd.dvd (HMul.hMul p …
    hoy : LT.lt 0 (orderOf y)
    hxy : Dvd.dvd (orderOf x) (orderOf y)
    p : Nat
    hp : Nat.Prime p
    hpy : Dvd.dvd p (orderOf y)
    hd : Dvd.dvd (orderOf (HMul.hMul x y)) (HDiv.hDiv (orderOf y) p)
    h : Dvd.dvd p (orderOf x)
    ⊢ Dvd.dvd (HMul.hMul p (orderOf x)) (orderOf y)
  -/
  exacts [hdvd p hp h, (hp.coprime_iff_not_dvd.2 h).mul_dvd_of_dvd_of_dvd hpy hxy]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_eq_prime (hg : x ^ p = 1) (hg1 : x ≠ 1) : orderOf x = p :=
  minimalPeriod_eq_prime ((isPeriodicPt_mul_iff_pow_eq_one _).mpr hg)
        /-
          G : Type u_1
          inst✝ : Monoid G
          x : G
          p : Nat
          hp : Fact (Nat.Prime p)
          hg : Eq (HPow.hPow x p) 1
          hg1 : Ne x 1
          ⊢ Not (Function.IsFixedPt (fun x_1 => HMul.hMul x x_1) 1)
        -/
    (by rwa [IsFixedPt, mul_one])
        /-
          🎉 no goals
        -/


@[to_additive addOrderOf_eq_prime_pow]
theorem orderOf_eq_prime_pow (hnot : ¬x ^ p ^ n = 1) (hfin : x ^ p ^ (n + 1) = 1) :
    orderOf x = p ^ (n + 1) := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    x : G
    n p : Nat
    hp : Fact (Nat.Prime p)
    hnot : Not (Eq (HPow.hPow x (HPow.hPow p n)) 1)
    hfin : Eq (HPow.hPow x (HPow.hPow p (HAdd.hAdd n 1))) 1
    ⊢ Eq (orderOf x) (HPow.hPow p (HAdd.hAdd n 1))
  -/
                                       /-
                                         🎉 no goals
                                       -/
  apply minimalPeriod_eq_prime_pow <;> rwa [isPeriodicPt_mul_iff_pow_eq_one]
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive exists_addOrderOf_eq_prime_pow_iff]
theorem exists_orderOf_eq_prime_pow_iff :
    (∃ k : ℕ, orderOf x = p ^ k) ↔ ∃ m : ℕ, x ^ (p : ℕ) ^ m = 1 :=
                         /-
                           G : Type u_1
                           inst✝ : Monoid G
                           x : G
                           p : Nat
                           hp : Fact (Nat.Prime p)
                           x✝ : Exists fun k => Eq (orderOf x) (HPow.hPow p k)
                           k : Nat
                           hk : Eq (orderOf x) (HPow.hPow p k)
                           ⊢ Eq (HPow.hPow x (HPow.hPow p k)) 1
                         -/
  ⟨fun ⟨k, hk⟩ => ⟨k, by rw [← hk, pow_orderOf_eq_one]⟩, fun ⟨_, hm⟩ => by
                         /-
                           🎉 no goals
                         -/
    /-
      G : Type u_1
      inst✝ : Monoid G
      x : G
      p : Nat
      hp : Fact (Nat.Prime p)
      x✝ : Exists fun m => Eq (HPow.hPow x (HPow.hPow p m)) 1
      w✝ : Nat
      hm : Eq (HPow.hPow x (HPow.hPow p w✝)) 1
      ⊢ Exists fun k => Eq (orderOf x) (HPow.hPow p k)
    -/
    obtain ⟨k, _, hk⟩ := (Nat.dvd_prime_pow hp.elim).mp (orderOf_dvd_of_pow_eq_one hm)
    /-
      case intro.intro
      G : Type u_1
      inst✝ : Monoid G
      x : G
      p : Nat
      hp : Fact (Nat.Prime p)
      x✝ : Exists fun m => Eq (HPow.hPow x (HPow.hPow p m)) 1
      w✝ : Nat
      hm : Eq (HPow.hPow x (HPow.hPow p w✝)) 1
      k : Nat
      left✝ : LE.le k w✝
      hk : Eq (orderOf x) (HPow.hPow p k)
      ⊢ Exists fun k => Eq (orderOf x) (HPow.hPow p k)
    -/
    exact ⟨k, hk⟩⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem pow_eq_pow_iff_modEq : x ^ n = x ^ m ↔ n ≡ m [MOD orderOf x] := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    m n : Nat
    ⊢ Iff (Eq (HPow.hPow x n) (HPow.hPow x m)) ((orderOf x).ModEq n m)
  -/
  wlog hmn : m ≤ n generalizing m n
    /-
      case inr
      G : Type u_1
      inst✝ : LeftCancelMonoid G
      x : G
      m n : Nat
      this : ∀ {m n : Nat}, LE.le m n → Iff (Eq (HPow.hPow x n) (HPow.hPow x m)) ((o …
      hmn : Not (LE.le m n)
      ⊢ Iff (Eq (HPow.hPow x n) (HPow.hPow x m)) ((orderOf x).ModEq n m)
    -/
  · rw [eq_comm, ModEq.comm, this (le_of_not_le hmn)]
    /-
      🎉 no goals
    -/
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    m✝ n✝ m n : Nat
    hmn : LE.le m n
    ⊢ Iff (Eq (HPow.hPow x n) (HPow.hPow x m)) ((orderOf x).ModEq n m)
  -/
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  /-
    case intro
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    m✝ n m k : Nat
    hmn : LE.le m (HAdd.hAdd m k)
    ⊢ Iff (Eq (HPow.hPow x (HAdd.hAdd m k)) (HPow.hPow x m)) ((orderOf x).ModEq (H …
  -/
  rw [← mul_one (x ^ m), pow_add, mul_left_cancel_iff, pow_eq_one_iff_modEq]
  /-
    case intro
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    m✝ n m k : Nat
    hmn : LE.le m (HAdd.hAdd m k)
    ⊢ Iff ((orderOf x).ModEq k 0) ((orderOf x).ModEq (HAdd.hAdd m k) m)
  -/
  exact ⟨fun h => Nat.ModEq.add_left _ h, fun h => Nat.ModEq.add_left_cancel' _ h⟩
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma injective_pow_iff_not_isOfFinOrder : Injective (fun n : ℕ ↦ x ^ n) ↔ ¬IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    ⊢ Iff (Function.Injective fun n => HPow.hPow x n) (Not (IsOfFinOrder x))
  -/
  refine ⟨fun h => not_isOfFinOrder_of_injective_pow h, fun h n m hnm => ?_⟩
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    h : Not (IsOfFinOrder x)
    n m : Nat
    hnm : Eq ((fun n => HPow.hPow x n) n) ((fun n => HPow.hPow x n) m)
    ⊢ Eq n m
  -/
  rwa [pow_eq_pow_iff_modEq, orderOf_eq_zero_iff.mpr h, modEq_zero_iff] at hnm
  /-
    🎉 no goals
  -/


@[to_additive]
lemma pow_inj_mod {n m : ℕ} : x ^ n = x ^ m ↔ n % orderOf x = m % orderOf x := pow_eq_pow_iff_modEq


@[to_additive]
theorem pow_inj_iff_of_orderOf_eq_zero (h : orderOf x = 0) {n m : ℕ} : x ^ n = x ^ m ↔ n = m := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    h : Eq (orderOf x) 0
    n m : Nat
    ⊢ Iff (Eq (HPow.hPow x n) (HPow.hPow x m)) (Eq n m)
  -/
  rw [pow_eq_pow_iff_modEq, h, modEq_zero_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem infinite_not_isOfFinOrder {x : G} (h : ¬IsOfFinOrder x) :
    { y : G | ¬IsOfFinOrder y }.Infinite := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    h : Not (IsOfFinOrder x)
    ⊢ (setOf fun y => Not (IsOfFinOrder y)).Infinite
  -/
  let s := { n | 0 < n }.image fun n : ℕ => x ^ n
  have hs : s ⊆ { y : G | ¬IsOfFinOrder y } := by
    rintro - ⟨n, hn : 0 < n, rfl⟩ (contra : IsOfFinOrder (x ^ n))
    apply h
    rw [isOfFinOrder_iff_pow_eq_one] at contra ⊢
    obtain ⟨m, hm, hm'⟩ := contra
    exact ⟨n * m, mul_pos hn hm, by rwa [pow_mul]⟩
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    h : Not (IsOfFinOrder x)
    s : Set G := Set.image (fun n => HPow.hPow x n) (setOf fun n => LT.lt 0 n)
    hs : HasSubset.Subset s (setOf fun y => Not (IsOfFinOrder y))
    ⊢ (setOf fun y => Not (IsOfFinOrder y)).Infinite
  -/
  suffices s.Infinite by exact this.mono hs
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    h : Not (IsOfFinOrder x)
    s : Set G := Set.image (fun n => HPow.hPow x n) (setOf fun n => LT.lt 0 n)
    hs : HasSubset.Subset s (setOf fun y => Not (IsOfFinOrder y))
    ⊢ s.Infinite
  -/
  contrapose! h
  have : ¬Injective fun n : ℕ => x ^ n := by
    have := Set.not_injOn_infinite_finite_image (Set.Ioi_infinite 0) (Set.not_infinite.mp h)
    contrapose! this
    exact Set.injOn_of_injective this
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    s : Set G := Set.image (fun n => HPow.hPow x n) (setOf fun n => LT.lt 0 n)
    hs : HasSubset.Subset s (setOf fun y => Not (IsOfFinOrder y))
    h : Not s.Infinite
    this : Not (Function.Injective fun n => HPow.hPow x n)
    ⊢ IsOfFinOrder x
  -/
  rwa [injective_pow_iff_not_isOfFinOrder, Classical.not_not] at this
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma finite_powers : (powers a : Set G).Finite ↔ IsOfFinOrder a := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    a : G
    ⊢ Iff (↑(Submonoid.powers a)).Finite (IsOfFinOrder a)
  -/
  refine ⟨fun h ↦ ?_, IsOfFinOrder.finite_powers⟩
  obtain ⟨m, n, hmn, ha⟩ := h.exists_lt_map_eq_of_forall_mem (f := fun n : ℕ ↦ a ^ n)
    (fun n ↦ by simp [mem_powers_iff])
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    a : G
    h : (↑(Submonoid.powers a)).Finite
    m n : Nat
    hmn : LT.lt m n
    ha : Eq (HPow.hPow a m) (HPow.hPow a n)
    ⊢ IsOfFinOrder a
  -/
  refine isOfFinOrder_iff_pow_eq_one.2 ⟨n - m, tsub_pos_iff_lt.2 hmn, ?_⟩
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    a : G
    h : (↑(Submonoid.powers a)).Finite
    m n : Nat
    hmn : LT.lt m n
    ha : Eq (HPow.hPow a m) (HPow.hPow a n)
    ⊢ Eq (HPow.hPow a (HSub.hSub n m)) 1
  -/
  rw [← mul_left_cancel_iff (a := a ^ m), ← pow_add, add_tsub_cancel_of_le hmn.le, ha, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma infinite_powers : (powers a : Set G).Infinite ↔ ¬ IsOfFinOrder a := finite_powers.not


/-- The equivalence between `Fin (orderOf x)` and `Submonoid.powers x`, sending `i` to `x ^ i`."-/
@[to_additive "The equivalence between `Fin (addOrderOf a)` and
`AddSubmonoid.multiples a`, sending `i` to `i • a`."]
noncomputable def finEquivPowers (x : G) (hx : IsOfFinOrder x) : Fin (orderOf x) ≃ powers x :=
  Equiv.ofBijective (fun n ↦ ⟨x ^ (n : ℕ), ⟨n, rfl⟩⟩) ⟨fun ⟨_, h₁⟩ ⟨_, h₂⟩ ij ↦
    Fin.ext (pow_injOn_Iio_orderOf h₁ h₂ (Subtype.mk_eq_mk.1 ij)), fun ⟨_, i, rfl⟩ ↦
      ⟨⟨i % orderOf x, mod_lt _ hx.orderOf_pos⟩, Subtype.eq <| pow_mod_orderOf _ _⟩⟩


@[to_additive (attr := simp)]
lemma finEquivPowers_apply (x : G) (hx) {n : Fin (orderOf x)} :
    finEquivPowers x hx n = ⟨x ^ (n : ℕ), n, rfl⟩ := rfl


@[to_additive (attr := simp)]
lemma finEquivPowers_symm_apply (x : G) (hx) (n : ℕ) {hn : ∃ m : ℕ, x ^ m = x ^ n} :
    (finEquivPowers x hx).symm ⟨x ^ n, hn⟩ = ⟨n % orderOf x, Nat.mod_lt _ hx.orderOf_pos⟩ := by
  /-
    G : Type u_1
    inst✝ : LeftCancelMonoid G
    x : G
    hx : IsOfFinOrder x
    n : Nat
    hn : Exists fun m => Eq (HPow.hPow x m) (HPow.hPow x n)
    ⊢ Eq ((finEquivPowers x hx).symm ⟨HPow.hPow x n, hn⟩) ⟨HMod.hMod n (orderOf x) …
  -/
  rw [Equiv.symm_apply_eq, finEquivPowers_apply, Subtype.mk_eq_mk, ← pow_mod_orderOf, Fin.val_mk]
  /-
    🎉 no goals
  -/


/-- See also `orderOf_eq_card_powers`. -/
@[to_additive "See also `addOrder_eq_card_multiples`."]
lemma Nat.card_submonoidPowers : Nat.card (powers a) = orderOf a := by
  classical
  by_cases ha : IsOfFinOrder a
  · exact (Nat.card_congr (finEquivPowers _ ha).symm).trans <| by simp
  · have := (infinite_powers.2 ha).to_subtype
    rw [orderOf_eq_zero ha, Nat.card_eq_zero_of_infinite]


/-- Inverses of elements of finite order have finite order. -/
@[to_additive (attr := simp) "Inverses of elements of finite additive order
have finite additive order."]
theorem isOfFinOrder_inv_iff {x : G} : IsOfFinOrder x⁻¹ ↔ IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    ⊢ Iff (IsOfFinOrder (Inv.inv x)) (IsOfFinOrder x)
  -/
  simp [isOfFinOrder_iff_pow_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive] alias ⟨IsOfFinOrder.of_inv, IsOfFinOrder.inv⟩ := isOfFinOrder_inv_iff


@[to_additive]
theorem orderOf_dvd_iff_zpow_eq_one : (orderOf x : ℤ) ∣ i ↔ x ^ i = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    i : Int
    ⊢ Iff (Dvd.dvd (↑(orderOf x)) i) (Eq (HPow.hPow x i) 1)
  -/
  rcases Int.eq_nat_or_neg i with ⟨i, rfl | rfl⟩
    /-
      case intro.inl
      G : Type u_1
      inst✝ : Group G
      x : G
      i : Nat
      ⊢ Iff (Dvd.dvd ↑(orderOf x) ↑i) (Eq (HPow.hPow x ↑i) 1)
    -/
  · rw [Int.natCast_dvd_natCast, orderOf_dvd_iff_pow_eq_one, zpow_natCast]
    /-
      🎉 no goals
    -/
  · rw [dvd_neg, Int.natCast_dvd_natCast, zpow_neg, inv_eq_one, zpow_natCast,
      orderOf_dvd_iff_pow_eq_one]


@[to_additive (attr := simp)]
                                                            /-
                                                              G : Type u_1
                                                              inst✝ : Group G
                                                              x : G
                                                              ⊢ Eq (orderOf (Inv.inv x)) (orderOf x)
                                                            -/
theorem orderOf_inv (x : G) : orderOf x⁻¹ = orderOf x := by simp [orderOf_eq_orderOf_iff]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[to_additive]
theorem orderOf_dvd_sub_iff_zpow_eq_zpow {a b : ℤ} : (orderOf x : ℤ) ∣ a - b ↔ x ^ a = x ^ b := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    a b : Int
    ⊢ Iff (Dvd.dvd (↑(orderOf x)) (HSub.hSub a b)) (Eq (HPow.hPow x a) (HPow.hPow  …
  -/
  rw [orderOf_dvd_iff_zpow_eq_one, zpow_sub, mul_inv_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive (attr := norm_cast)]
lemma orderOf_coe (a : H) : orderOf (a : G) = orderOf a :=
  orderOf_injective H.subtype Subtype.coe_injective _


@[to_additive (attr := simp)]
lemma orderOf_mk (a : G) (ha) : orderOf (⟨a, ha⟩ : H) = orderOf a := (orderOf_coe _).symm


@[to_additive mod_addOrderOf_zsmul]
lemma zpow_mod_orderOf (x : G) (z : ℤ) : x ^ (z % (orderOf x : ℤ)) = x ^ z :=
  calc
    x ^ (z % (orderOf x : ℤ)) = x ^ (z % orderOf x + orderOf x * (z / orderOf x) : ℤ) := by
        /-
          G : Type u_1
          inst✝ : Group G
          x : G
          z : Int
          ⊢ Eq (HPow.hPow x (HMod.hMod z ↑(orderOf x))) (HPow.hPow x (HAdd.hAdd (HMod.hM …
        -/
        simp [zpow_add, zpow_mul, pow_orderOf_eq_one]
        /-
          🎉 no goals
        -/
                    /-
                      G : Type u_1
                      inst✝ : Group G
                      x : G
                      z : Int
                      ⊢ Eq (HPow.hPow x (HAdd.hAdd (HMod.hMod z ↑(orderOf x)) (HMul.hMul (↑(orderOf  …
                    -/
    _ = x ^ z := by rw [Int.emod_add_ediv]
                    /-
                      🎉 no goals
                    -/


@[to_additive (attr := simp) zsmul_smul_addOrderOf]
theorem zpow_pow_orderOf : (x ^ i) ^ orderOf x = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    i : Int
    ⊢ Eq (HPow.hPow (HPow.hPow x i) (orderOf x)) 1
  -/
  by_cases h : IsOfFinOrder x
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      x : G
      i : Int
      h : IsOfFinOrder x
      ⊢ Eq (HPow.hPow (HPow.hPow x i) (orderOf x)) 1
    -/
  · rw [← zpow_natCast, ← zpow_mul, mul_comm, zpow_mul, zpow_natCast, pow_orderOf_eq_one, one_zpow]
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      x : G
      i : Int
      h : Not (IsOfFinOrder x)
      ⊢ Eq (HPow.hPow (HPow.hPow x i) (orderOf x)) 1
    -/
  · rw [orderOf_eq_zero h, _root_.pow_zero]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem IsOfFinOrder.zpow (h : IsOfFinOrder x) {i : ℤ} : IsOfFinOrder (x ^ i) :=
  isOfFinOrder_iff_pow_eq_one.mpr ⟨orderOf x, h.orderOf_pos, zpow_pow_orderOf⟩


@[to_additive]
theorem IsOfFinOrder.of_mem_zpowers (h : IsOfFinOrder x) (h' : y ∈ Subgroup.zpowers x) :
    IsOfFinOrder y := by
  /-
    G : Type u_1
    inst✝ : Group G
    x y : G
    h : IsOfFinOrder x
    h' : Membership.mem (Subgroup.zpowers x) y
    ⊢ IsOfFinOrder y
  -/
  obtain ⟨k, rfl⟩ := Subgroup.mem_zpowers_iff.mp h'
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    x : G
    h : IsOfFinOrder x
    k : Int
    h' : Membership.mem (Subgroup.zpowers x) (HPow.hPow x k)
    ⊢ IsOfFinOrder (HPow.hPow x k)
  -/
  exact h.zpow
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_dvd_of_mem_zpowers (h : y ∈ Subgroup.zpowers x) : orderOf y ∣ orderOf x := by
  /-
    G : Type u_1
    inst✝ : Group G
    x y : G
    h : Membership.mem (Subgroup.zpowers x) y
    ⊢ Dvd.dvd (orderOf y) (orderOf x)
  -/
  obtain ⟨k, rfl⟩ := Subgroup.mem_zpowers_iff.mp h
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    x : G
    k : Int
    h : Membership.mem (Subgroup.zpowers x) (HPow.hPow x k)
    ⊢ Dvd.dvd (orderOf (HPow.hPow x k)) (orderOf x)
  -/
  rw [orderOf_dvd_iff_pow_eq_one]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    x : G
    k : Int
    h : Membership.mem (Subgroup.zpowers x) (HPow.hPow x k)
    ⊢ Eq (HPow.hPow (HPow.hPow x k) (orderOf x)) 1
  -/
  exact zpow_pow_orderOf
  /-
    🎉 no goals
  -/


theorem smul_eq_self_of_mem_zpowers {α : Type*} [MulAction G α] (hx : x ∈ Subgroup.zpowers y)
    {a : α} (hs : y • a = a) : x • a = a := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    x y : G
    α : Type u_6
    inst✝ : MulAction G α
    hx : Membership.mem (Subgroup.zpowers y) x
    a : α
    hs : Eq (HSMul.hSMul y a) a
    ⊢ Eq (HSMul.hSMul x a) a
  -/
  obtain ⟨k, rfl⟩ := Subgroup.mem_zpowers_iff.mp hx
  rw [← MulAction.toPerm_apply, ← MulAction.toPermHom_apply, MonoidHom.map_zpow _ y k,
    MulAction.toPermHom_apply]
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    y : G
    α : Type u_6
    inst✝ : MulAction G α
    a : α
    hs : Eq (HSMul.hSMul y a) a
    k : Int
    hx : Membership.mem (Subgroup.zpowers y) (HPow.hPow y k)
    ⊢ Eq ((HPow.hPow (MulAction.toPerm y) k) a) a
  -/
  exact Function.IsFixedPt.perm_zpow (by exact hs) k -- Porting note: help elab'n with `by exact`
  /-
    🎉 no goals
  -/


theorem vadd_eq_self_of_mem_zmultiples {α G : Type*} [AddGroup G] [AddAction G α] {x y : G}
    (hx : x ∈ AddSubgroup.zmultiples y) {a : α} (hs : y +ᵥ a = a) : x +ᵥ a = a :=
  @smul_eq_self_of_mem_zpowers (Multiplicative G) _ _ _ α _ hx a hs


@[to_additive]
lemma IsOfFinOrder.mem_powers_iff_mem_zpowers (hx : IsOfFinOrder x) :
    y ∈ powers x ↔ y ∈ zpowers x :=
                        /-
                          G : Type u_1
                          inst✝ : Group G
                          x y : G
                          hx : IsOfFinOrder x
                          x✝ : Membership.mem (Submonoid.powers x) y
                          n : Nat
                          hn : Eq ((fun x_1 => HPow.hPow x x_1) n) y
                          ⊢ Eq ((fun x_1 => HPow.hPow x x_1) ↑n) y
                        -/
  ⟨fun ⟨n, hn⟩ ↦ ⟨n, by simp_all⟩, fun ⟨i, hi⟩ ↦ ⟨(i % orderOf x).natAbs, by
                        /-
                          🎉 no goals
                        -/
    /-
      G : Type u_1
      inst✝ : Group G
      x y : G
      hx : IsOfFinOrder x
      x✝ : Membership.mem (Subgroup.zpowers x) y
      i : Int
      hi : Eq ((fun x_1 => HPow.hPow x x_1) i) y
      ⊢ Eq ((fun x_1 => HPow.hPow x x_1) (HMod.hMod i ↑(orderOf x)).natAbs) y
    -/
    dsimp only
    rwa [← zpow_natCast, Int.natAbs_of_nonneg <| Int.emod_nonneg _ <|
      Int.natCast_ne_zero_iff_pos.2 <| hx.orderOf_pos, zpow_mod_orderOf]⟩⟩


@[to_additive]
lemma IsOfFinOrder.powers_eq_zpowers (hx : IsOfFinOrder x) : (powers x : Set G) = zpowers x :=
  Set.ext fun _ ↦ hx.mem_powers_iff_mem_zpowers


@[to_additive]
lemma IsOfFinOrder.mem_zpowers_iff_mem_range_orderOf [DecidableEq G] (hx : IsOfFinOrder x) :
    y ∈ zpowers x ↔ y ∈ (Finset.range (orderOf x)).image (x ^ ·) :=
  hx.mem_powers_iff_mem_zpowers.symm.trans hx.mem_powers_iff_mem_range_orderOf


/-- The equivalence between `Fin (orderOf x)` and `Subgroup.zpowers x`, sending `i` to `x ^ i`. -/
@[to_additive "The equivalence between `Fin (addOrderOf a)` and
`Subgroup.zmultiples a`, sending `i` to `i • a`."]
noncomputable def finEquivZPowers (x : G) (hx : IsOfFinOrder x) :
    Fin (orderOf x) ≃ (zpowers x : Set G) :=
  (finEquivPowers x hx).trans <| Equiv.Set.ofEq hx.powers_eq_zpowers

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[to_additive (attr := simp, nolint simpNF)]
lemma finEquivZPowers_apply (hx) {n : Fin (orderOf x)} :
    finEquivZPowers x hx n = ⟨x ^ (n : ℕ), n, zpow_natCast x n⟩ := rfl

 -- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[to_additive (attr := simp, nolint simpNF)]
lemma finEquivZPowers_symm_apply (x : G) (hx) (n : ℕ) :
                                               /-
                                                 G : Type u_1
                                                 H : Type u_2
                                                 A : Type u_3
                                                 α : Type u_4
                                                 β : Type u_5
                                                 inst✝ : Group G
                                                 x✝ y : G
                                                 i : Int
                                                 x : G
                                                 hx : IsOfFinOrder x
                                                 n : Nat
                                                 ⊢ Eq ((fun x_1 => HPow.hPow x x_1) ↑n) (HPow.hPow x n)
                                               -/
    (finEquivZPowers x hx).symm ⟨x ^ n, ⟨n, by simp⟩⟩ =
                                               /-
                                                 🎉 no goals
                                               -/
    ⟨n % orderOf x, Nat.mod_lt _ hx.orderOf_pos⟩ := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    hx : IsOfFinOrder x
    n : Nat
    ⊢ Eq ((finEquivZPowers x hx).symm ⟨HPow.hPow x n, ⋯⟩) ⟨HMod.hMod n (orderOf x) …
  -/
  rw [finEquivZPowers, Equiv.symm_trans_apply]; exact finEquivPowers_symm_apply x _ n
                                                /-
                                                  🎉 no goals
                                                -/


/-- Elements of finite order are closed under multiplication. -/
@[to_additive "Elements of finite additive order are closed under addition."]
theorem IsOfFinOrder.mul (hx : IsOfFinOrder x) (hy : IsOfFinOrder y) : IsOfFinOrder (x * y) :=
  (Commute.all x y).isOfFinOrder_mul hx hy


@[to_additive]
theorem sum_card_orderOf_eq_card_pow_eq_one [Fintype G] [DecidableEq G] (hn : n ≠ 0) :
    (∑ m ∈ (Finset.range n.succ).filter (· ∣ n),
        (Finset.univ.filter fun x : G => orderOf x = m).card) =
      (Finset.univ.filter fun x : G => x ^ n = 1).card :=
  calc
    (∑ m ∈ (Finset.range n.succ).filter (· ∣ n),
          (Finset.univ.filter fun x : G => orderOf x = m).card) = _ :=
      (Finset.card_biUnion
          (by
            /-
              G : Type u_1
              inst✝² : Monoid G
              n : Nat
              inst✝¹ : Fintype G
              inst✝ : DecidableEq G
              hn : Ne n 0
              ⊢ ∀ (x : Nat), Membership.mem (Finset.filter (fun x => Dvd.dvd x n) (Finset.ra …
            -/
            intros
            /-
              G : Type u_1
              inst✝² : Monoid G
              n : Nat
              inst✝¹ : Fintype G
              inst✝ : DecidableEq G
              hn : Ne n 0
              x✝ : Nat
              a✝² : Membership.mem (Finset.filter (fun x => Dvd.dvd x n) (Finset.range n.suc …
              y✝ : Nat
              a✝¹ : Membership.mem (Finset.filter (fun x => Dvd.dvd x n) (Finset.range n.suc …
              a✝ : Ne x✝ y✝
              ⊢ Disjoint (Finset.filter (fun x => Eq (orderOf x) x✝) Finset.univ) (Finset.fi …
            -/
            apply Finset.disjoint_filter.2
            /-
              G : Type u_1
              inst✝² : Monoid G
              n : Nat
              inst✝¹ : Fintype G
              inst✝ : DecidableEq G
              hn : Ne n 0
              x✝ : Nat
              a✝² : Membership.mem (Finset.filter (fun x => Dvd.dvd x n) (Finset.range n.suc …
              y✝ : Nat
              a✝¹ : Membership.mem (Finset.filter (fun x => Dvd.dvd x n) (Finset.range n.suc …
              a✝ : Ne x✝ y✝
              ⊢ ∀ (x : G), Membership.mem Finset.univ x → Eq (orderOf x) x✝ → Not (Eq (order …
            -/
            rintro _ _ rfl; assumption)).symm
                            /-
                              🎉 no goals
                            -/
    _ = _ :=
      congr_arg Finset.card
        (Finset.ext
          (by
            /-
              G : Type u_1
              inst✝² : Monoid G
              n : Nat
              inst✝¹ : Fintype G
              inst✝ : DecidableEq G
              hn : Ne n 0
              ⊢ ∀ (a : G), Iff (Membership.mem ((Finset.filter (fun x => Dvd.dvd x n) (Finse …
            -/
            intro x
            /-
              G : Type u_1
              inst✝² : Monoid G
              n : Nat
              inst✝¹ : Fintype G
              inst✝ : DecidableEq G
              hn : Ne n 0
              x : G
              ⊢ Iff (Membership.mem ((Finset.filter (fun x => Dvd.dvd x n) (Finset.range n.s …
            -/
            suffices orderOf x ≤ n ∧ orderOf x ∣ n ↔ x ^ n = 1 by simpa [Nat.lt_succ_iff]
            exact
              ⟨fun h => by
                let ⟨m, hm⟩ := h.2
                rw [hm, pow_mul, pow_orderOf_eq_one, one_pow], fun h =>
                ⟨orderOf_le_of_pow_eq_one hn.bot_lt h, orderOf_dvd_of_pow_eq_one h⟩⟩))


@[to_additive]
theorem orderOf_le_card_univ [Fintype G] : orderOf x ≤ Fintype.card G :=
  Finset.le_card_of_inj_on_range (x ^ ·) (fun _ _ ↦ Finset.mem_univ _) pow_injOn_Iio_orderOf


@[to_additive]
lemma isOfFinOrder_of_finite (x : G) : IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝¹ : LeftCancelMonoid G
    inst✝ : Finite G
    x : G
    ⊢ IsOfFinOrder x
  -/
  by_contra h; exact infinite_not_isOfFinOrder h <| Set.toFinite _
               /-
                 🎉 no goals
               -/


/-- This is the same as `IsOfFinOrder.orderOf_pos` but with one fewer explicit assumption since this
is automatic in case of a finite cancellative monoid. -/
@[to_additive "This is the same as `IsOfFinAddOrder.addOrderOf_pos` but with one fewer explicit
assumption since this is automatic in case of a finite cancellative additive monoid."]
lemma orderOf_pos (x : G) : 0 < orderOf x := (isOfFinOrder_of_finite x).orderOf_pos


/-- This is the same as `orderOf_pow'` and `orderOf_pow''` but with one assumption less which is
automatic in the case of a finite cancellative monoid. -/
@[to_additive "This is the same as `addOrderOf_nsmul'` and
`addOrderOf_nsmul` but with one assumption less which is automatic in the case of a
finite cancellative additive monoid."]
theorem orderOf_pow (x : G) : orderOf (x ^ n) = orderOf x / gcd (orderOf x) n :=
  (isOfFinOrder_of_finite _).orderOf_pow ..


@[to_additive]
theorem mem_powers_iff_mem_range_orderOf [DecidableEq G] :
    y ∈ powers x ↔ y ∈ (Finset.range (orderOf x)).image (x ^ ·) :=
  Finset.mem_range_iff_mem_finset_range_of_mod_eq' (orderOf_pos x) <| pow_mod_orderOf _


/-- The equivalence between `Submonoid.powers` of two elements `x, y` of the same order, mapping
  `x ^ i` to `y ^ i`. -/
@[to_additive
  "The equivalence between `Submonoid.multiples` of two elements `a, b` of the same additive order,
  mapping `i • a` to `i • b`."]
noncomputable def powersEquivPowers (h : orderOf x = orderOf y) : powers x ≃ powers y :=
  (finEquivPowers x <| isOfFinOrder_of_finite _).symm.trans <|
    (finCongr h).trans <| finEquivPowers y <| isOfFinOrder_of_finite _


@[to_additive (attr := simp)]
theorem powersEquivPowers_apply (h : orderOf x = orderOf y) (n : ℕ) :
    powersEquivPowers h ⟨x ^ n, n, rfl⟩ = ⟨y ^ n, n, rfl⟩ := by
  rw [powersEquivPowers, Equiv.trans_apply, Equiv.trans_apply, finEquivPowers_symm_apply, ←
    Equiv.eq_symm_apply, finEquivPowers_symm_apply]
  /-
    G : Type u_1
    inst✝¹ : LeftCancelMonoid G
    inst✝ : Finite G
    x y : G
    h : Eq (orderOf x) (orderOf y)
    n : Nat
    ⊢ Eq ((finCongr h) ⟨HMod.hMod n (orderOf x), ⋯⟩) ⟨HMod.hMod n (orderOf y), ⋯⟩
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orderOf_eq_card_powers : orderOf x = Fintype.card (powers x : Submonoid G) :=
  (Fintype.card_fin (orderOf x)).symm.trans <|
    Fintype.card_eq.2 ⟨finEquivPowers x <| isOfFinOrder_of_finite _⟩


@[to_additive]
theorem zpow_eq_one_iff_modEq {n : ℤ} : x ^ n = 1 ↔ n ≡ 0 [ZMOD orderOf x] := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    n : Int
    ⊢ Iff (Eq (HPow.hPow x n) 1) ((↑(orderOf x)).ModEq n 0)
  -/
  rw [Int.modEq_zero_iff_dvd, orderOf_dvd_iff_zpow_eq_one]
  /-
    🎉 no goals
  -/



@[to_additive]
theorem zpow_eq_zpow_iff_modEq {m n : ℤ} : x ^ m = x ^ n ↔ m ≡ n [ZMOD orderOf x] := by
  rw [← mul_inv_eq_one, ← zpow_sub, zpow_eq_one_iff_modEq, Int.modEq_iff_dvd, Int.modEq_iff_dvd,
    zero_sub, neg_sub]


@[to_additive (attr := simp)]
theorem injective_zpow_iff_not_isOfFinOrder : (Injective fun n : ℤ => x ^ n) ↔ ¬IsOfFinOrder x := by
  /-
    G : Type u_1
    inst✝ : Group G
    x : G
    ⊢ Iff (Function.Injective fun n => HPow.hPow x n) (Not (IsOfFinOrder x))
  -/
  refine ⟨?_, fun h n m hnm => ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      x : G
      ⊢ (Function.Injective fun n => HPow.hPow x n) → Not (IsOfFinOrder x)
    -/
  · simp_rw [isOfFinOrder_iff_pow_eq_one]
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      x : G
      ⊢ (Function.Injective fun n => HPow.hPow x n) → Not (Exists fun n => And (LT.l …
    -/
    rintro h ⟨n, hn, hx⟩
    /-
      case refine_1.intro.intro
      G : Type u_1
      inst✝ : Group G
      x : G
      h : Function.Injective fun n => HPow.hPow x n
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow x n) 1
      ⊢ False
    -/
    exact Nat.cast_ne_zero.2 hn.ne' (h <| by simpa using hx)
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    G : Type u_1
    inst✝ : Group G
    x : G
    h : Not (IsOfFinOrder x)
    n m : Int
    hnm : Eq ((fun n => HPow.hPow x n) n) ((fun n => HPow.hPow x n) m)
    ⊢ Eq n m
  -/
  rwa [zpow_eq_zpow_iff_modEq, orderOf_eq_zero_iff.2 h, Nat.cast_zero, Int.modEq_zero_iff] at hnm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_zpow_eq_one (x : G) : ∃ (i : ℤ) (_ : i ≠ 0), x ^ (i : ℤ) = 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    x : G
    ⊢ Exists fun i => Exists fun x_1 => Eq (HPow.hPow x i) 1
  -/
  obtain ⟨w, hw1, hw2⟩ := isOfFinOrder_of_finite x
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    x : G
    w : Nat
    hw1 : GT.gt w 0
    hw2 : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) w 1
    ⊢ Exists fun i => Exists fun x_1 => Eq (HPow.hPow x i) 1
  -/
  refine ⟨w, Int.natCast_ne_zero.mpr (_root_.ne_of_gt hw1), ?_⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    x : G
    w : Nat
    hw1 : GT.gt w 0
    hw2 : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) w 1
    ⊢ Eq (HPow.hPow x ↑w) 1
  -/
  rw [zpow_natCast]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Finite G
    x : G
    w : Nat
    hw1 : GT.gt w 0
    hw2 : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) w 1
    ⊢ Eq (HPow.hPow x w) 1
  -/
  exact (isPeriodicPt_mul_iff_pow_eq_one _).mp hw2
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_powers_iff_mem_zpowers : y ∈ powers x ↔ y ∈ zpowers x :=
  (isOfFinOrder_of_finite _).mem_powers_iff_mem_zpowers


@[to_additive]
lemma powers_eq_zpowers (x : G) : (powers x : Set G) = zpowers x :=
  (isOfFinOrder_of_finite _).powers_eq_zpowers


@[to_additive]
lemma mem_zpowers_iff_mem_range_orderOf [DecidableEq G] :
    y ∈ zpowers x ↔ y ∈ (Finset.range (orderOf x)).image (x ^ ·) :=
  (isOfFinOrder_of_finite _).mem_zpowers_iff_mem_range_orderOf


/-- The equivalence between `Subgroup.zpowers` of two elements `x, y` of the same order, mapping
  `x ^ i` to `y ^ i`. -/
@[to_additive
  "The equivalence between `Subgroup.zmultiples` of two elements `a, b` of the same additive order,
  mapping `i • a` to `i • b`."]
noncomputable def zpowersEquivZPowers (h : orderOf x = orderOf y) :
    (Subgroup.zpowers x : Set G) ≃ (Subgroup.zpowers y : Set G) :=
  (finEquivZPowers x <| isOfFinOrder_of_finite _).symm.trans <| (finCongr h).trans <|
    finEquivZPowers y <| isOfFinOrder_of_finite _

-- Porting note: the simpNF linter complains that simp can change the LHS to something
-- that looks the same as the current LHS even with `pp.explicit`

@[to_additive (attr := simp, nolint simpNF) zmultiples_equiv_zmultiples_apply]
theorem zpowersEquivZPowers_apply (h : orderOf x = orderOf y) (n : ℕ) :
    zpowersEquivZPowers h ⟨x ^ n, n, zpow_natCast x n⟩ = ⟨y ^ n, n, zpow_natCast y n⟩ := by
  rw [zpowersEquivZPowers, Equiv.trans_apply, Equiv.trans_apply, finEquivZPowers_symm_apply, ←
    Equiv.eq_symm_apply, finEquivZPowers_symm_apply]
  /-
    G : Type u_1
    inst✝¹ : Group G
    x y : G
    inst✝ : Finite G
    h : Eq (orderOf x) (orderOf y)
    n : Nat
    ⊢ Eq ((finCongr h) ⟨HMod.hMod n (orderOf x), ⋯⟩) ⟨HMod.hMod n (orderOf y), ⋯⟩
  -/
  simp [h]
  /-
    🎉 no goals
  -/


/-- See also `Nat.card_addSubgroupZPowers`. -/
@[to_additive "See also `Nat.card_subgroup`."]
theorem Fintype.card_zpowers : Fintype.card (zpowers x) = orderOf x :=
  letI : Fintype (zpowers x) := (Subgroup.zpowers x).instFintypeSubtypeMemOfDecidablePred
  (Fintype.card_eq.2 ⟨finEquivZPowers x <| isOfFinOrder_of_finite _⟩).symm.trans <|
    Fintype.card_fin (orderOf x)


@[to_additive]
theorem card_zpowers_le (a : G) {k : ℕ} (k_pos : k ≠ 0)
    (ha : a ^ k = 1) : Fintype.card (Subgroup.zpowers a) ≤ k := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fintype G
    a : G
    k : Nat
    k_pos : Ne k 0
    ha : Eq (HPow.hPow a k) 1
    ⊢ LE.le (Fintype.card (Subtype fun x => Membership.mem (Subgroup.zpowers a) x) …
  -/
  rw [Fintype.card_zpowers]
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fintype G
    a : G
    k : Nat
    k_pos : Ne k 0
    ha : Eq (HPow.hPow a k) 1
    ⊢ LE.le (orderOf a) k
  -/
  apply orderOf_le_of_pow_eq_one k_pos.bot_lt ha
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_dvd_card : orderOf x ∣ Fintype.card G := by
  classical
    have ft_prod : Fintype ((G ⧸ zpowers x) × zpowers x) :=
      Fintype.ofEquiv G groupEquivQuotientProdSubgroup
    have ft_s : Fintype (zpowers x) := @Fintype.prodRight _ _ _ ft_prod _
    have ft_cosets : Fintype (G ⧸ zpowers x) :=
      @Fintype.prodLeft _ _ _ ft_prod ⟨⟨1, (zpowers x).one_mem⟩⟩
    have eq₁ : Fintype.card G = @Fintype.card _ ft_cosets * @Fintype.card _ ft_s :=
      calc
        Fintype.card G = @Fintype.card _ ft_prod :=
          @Fintype.card_congr _ _ _ ft_prod groupEquivQuotientProdSubgroup
        _ = @Fintype.card _ (@instFintypeProd _ _ ft_cosets ft_s) :=
          congr_arg (@Fintype.card _) <| Subsingleton.elim _ _
        _ = @Fintype.card _ ft_cosets * @Fintype.card _ ft_s :=
          @Fintype.card_prod _ _ ft_cosets ft_s

    have eq₂ : orderOf x = @Fintype.card _ ft_s :=
      calc
        orderOf x = _ := Fintype.card_zpowers.symm
        _ = _ := congr_arg (@Fintype.card _) <| Subsingleton.elim _ _

    exact Dvd.intro (@Fintype.card (G ⧸ Subgroup.zpowers x) ft_cosets) (by rw [eq₁, eq₂, mul_comm])


@[to_additive]
theorem orderOf_dvd_natCard {G : Type*} [Group G] (x : G) : orderOf x ∣ Nat.card G := by
  /-
    G : Type u_6
    inst✝ : Group G
    x : G
    ⊢ Dvd.dvd (orderOf x) (Nat.card G)
  -/
  cases' fintypeOrInfinite G with h h
    /-
      case inl
      G : Type u_6
      inst✝ : Group G
      x : G
      h : Fintype G
      ⊢ Dvd.dvd (orderOf x) (Nat.card G)
    -/
  · simp only [Nat.card_eq_fintype_card, orderOf_dvd_card]
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_6
      inst✝ : Group G
      x : G
      h : Infinite G
      ⊢ Dvd.dvd (orderOf x) (Nat.card G)
    -/
  · simp only [card_eq_zero_of_infinite, dvd_zero]
    /-
      🎉 no goals
    -/


@[to_additive]
nonrec lemma Subgroup.orderOf_dvd_natCard {G : Type*} [Group G] (s : Subgroup G) {x} (hx : x ∈ s) :
                               /-
                                 G : Type u_6
                                 inst✝ : Group G
                                 s : Subgroup G
                                 x : G
                                 hx : Membership.mem s x
                                 ⊢ Dvd.dvd (orderOf x) (Nat.card (Subtype fun x => Membership.mem s x))
                               -/
  orderOf x ∣ Nat.card s := by simpa using orderOf_dvd_natCard (⟨x, hx⟩ : s)
                               /-
                                 🎉 no goals
                               -/


@[to_additive]
lemma Subgroup.orderOf_le_card {G : Type*} [Group G] (s : Subgroup G) (hs : (s : Set G).Finite)
    {x} (hx : x ∈ s) : orderOf x ≤ Nat.card s :=
  le_of_dvd (Nat.card_pos_iff.2 <| ⟨s.coe_nonempty.to_subtype, hs.to_subtype⟩) <|
    s.orderOf_dvd_natCard hx


@[to_additive]
lemma Submonoid.orderOf_le_card {G : Type*} [Group G] (s : Submonoid G) (hs : (s : Set G).Finite)
    {x} (hx : x ∈ s) : orderOf x ≤ Nat.card s := by
  /-
    G : Type u_6
    inst✝ : Group G
    s : Submonoid G
    hs : (↑s).Finite
    x : G
    hx : Membership.mem s x
    ⊢ LE.le (orderOf x) (Nat.card (Subtype fun x => Membership.mem s x))
  -/
  rw [← Nat.card_submonoidPowers]; exact Nat.card_mono hs <| powers_le.2 hx
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive (attr := simp) card_nsmul_eq_zero']
theorem pow_card_eq_one' {G : Type*} [Group G] {x : G} : x ^ Nat.card G = 1 :=
  orderOf_dvd_iff_pow_eq_one.mp <| orderOf_dvd_natCard _


@[to_additive (attr := simp) card_nsmul_eq_zero]
theorem pow_card_eq_one : x ^ Fintype.card G = 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fintype G
    x : G
    ⊢ Eq (HPow.hPow x (Fintype.card G)) 1
  -/
  rw [← Nat.card_eq_fintype_card, pow_card_eq_one']
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Subgroup.pow_index_mem {G : Type*} [Group G] (H : Subgroup G) [Normal H] (g : G) :
                          /-
                            G : Type u_6
                            inst✝¹ : Group G
                            H : Subgroup G
                            inst✝ : H.Normal
                            g : G
                            ⊢ Membership.mem H (HPow.hPow g H.index)
                          -/
    g ^ index H ∈ H := by rw [← eq_one_iff, QuotientGroup.mk_pow H, index, pow_card_eq_one']
                          /-
                            🎉 no goals
                          -/



@[to_additive (attr := simp) mod_card_nsmul]
lemma pow_mod_card (a : G) (n : ℕ) : a ^ (n % card G) = a ^ n := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : Fintype G
    a : G
    n : Nat
    ⊢ Eq (HPow.hPow a (HMod.hMod n (Fintype.card G))) (HPow.hPow a n)
  -/
  rw [eq_comm, ← pow_mod_orderOf, ← Nat.mod_mod_of_dvd n orderOf_dvd_card, pow_mod_orderOf]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) mod_card_zsmul]
theorem zpow_mod_card (a : G) (n : ℤ) : a ^ (n % Fintype.card G : ℤ) = a ^ n := by
  rw [eq_comm, ← zpow_mod_orderOf, ← Int.emod_emod_of_dvd n
    (Int.natCast_dvd_natCast.2 orderOf_dvd_card), zpow_mod_orderOf]


@[to_additive (attr := simp) mod_natCard_nsmul]
lemma pow_mod_natCard {G} [Group G] (a : G) (n : ℕ) : a ^ (n % Nat.card G) = a ^ n := by
  /-
    G : Type u_6
    inst✝ : Group G
    a : G
    n : Nat
    ⊢ Eq (HPow.hPow a (HMod.hMod n (Nat.card G))) (HPow.hPow a n)
  -/
  rw [eq_comm, ← pow_mod_orderOf, ← Nat.mod_mod_of_dvd n <| orderOf_dvd_natCard _, pow_mod_orderOf]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) mod_natCard_zsmul]
lemma zpow_mod_natCard {G} [Group G] (a : G) (n : ℤ) : a ^ (n % Nat.card G : ℤ) = a ^ n := by
  rw [eq_comm, ← zpow_mod_orderOf, ← Int.emod_emod_of_dvd n <|
    Int.natCast_dvd_natCast.2 <| orderOf_dvd_natCard _, zpow_mod_orderOf]


/-- If `gcd(|G|,n)=1` then the `n`th power map is a bijection -/
@[to_additive (attr := simps) "If `gcd(|G|,n)=1` then the smul by `n` is a bijection"]
noncomputable def powCoprime {G : Type*} [Group G] (h : (Nat.card G).Coprime n) : G ≃ G where
  toFun g := g ^ n
  invFun g := g ^ (Nat.card G).gcdB n
  left_inv g := by
    /-
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝² : Group G✝
      x✝ y : G✝
      inst✝¹ : Fintype G✝
      x : G✝
      n : Nat
      G : Type u_6
      inst✝ : Group G
      h : (Nat.card G).Coprime n
      g : G
      ⊢ Eq ((fun g => HPow.hPow g ((Nat.card G).gcdB n)) ((fun g => HPow.hPow g n) g …
    -/
    have key := congr_arg (g ^ ·) ((Nat.card G).gcd_eq_gcd_ab n)
    /-
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝² : Group G✝
      x✝ y : G✝
      inst✝¹ : Fintype G✝
      x : G✝
      n : Nat
      G : Type u_6
      inst✝ : Group G
      h : (Nat.card G).Coprime n
      g : G
      key : Eq ((fun x => HPow.hPow g x) ↑((Nat.card G).gcd n)) ((fun x => HPow.hPow …
      ⊢ Eq ((fun g => HPow.hPow g ((Nat.card G).gcdB n)) ((fun g => HPow.hPow g n) g …
    -/
    dsimp only at key
    rwa [zpow_add, zpow_mul, zpow_mul, zpow_natCast, zpow_natCast, zpow_natCast, h.gcd_eq_one,
      pow_one, pow_card_eq_one', one_zpow, one_mul, eq_comm] at key
  right_inv g := by
    /-
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝² : Group G✝
      x✝ y : G✝
      inst✝¹ : Fintype G✝
      x : G✝
      n : Nat
      G : Type u_6
      inst✝ : Group G
      h : (Nat.card G).Coprime n
      g : G
      ⊢ Eq ((fun g => HPow.hPow g n) ((fun g => HPow.hPow g ((Nat.card G).gcdB n)) g …
    -/
    have key := congr_arg (g ^ ·) ((Nat.card G).gcd_eq_gcd_ab n)
    /-
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      inst✝² : Group G✝
      x✝ y : G✝
      inst✝¹ : Fintype G✝
      x : G✝
      n : Nat
      G : Type u_6
      inst✝ : Group G
      h : (Nat.card G).Coprime n
      g : G
      key : Eq ((fun x => HPow.hPow g x) ↑((Nat.card G).gcd n)) ((fun x => HPow.hPow …
      ⊢ Eq ((fun g => HPow.hPow g n) ((fun g => HPow.hPow g ((Nat.card G).gcdB n)) g …
    -/
    dsimp only at key
    rwa [zpow_add, zpow_mul, zpow_mul', zpow_natCast, zpow_natCast, zpow_natCast, h.gcd_eq_one,
      pow_one, pow_card_eq_one', one_zpow, one_mul, eq_comm] at key


@[to_additive]
theorem powCoprime_one {G : Type*} [Group G] (h : (Nat.card G).Coprime n) : powCoprime h 1 = 1 :=
  one_pow n


@[to_additive]
theorem powCoprime_inv {G : Type*} [Group G] (h : (Nat.card G).Coprime n) {g : G} :
    powCoprime h g⁻¹ = (powCoprime h g)⁻¹ :=
  inv_pow g n


@[to_additive Nat.Coprime.nsmul_right_bijective]
lemma Nat.Coprime.pow_left_bijective {G} [Group G] (hn : (Nat.card G).Coprime n) :
    Bijective (· ^ n : G → G) :=
  (powCoprime hn).bijective

/- TODO: Generalise to `Submonoid.powers`. -/

@[to_additive]
theorem image_range_orderOf [DecidableEq G] :
    letI : Fintype (zpowers x) := (Subgroup.zpowers x).instFintypeSubtypeMemOfDecidablePred
    Finset.image (fun i => x ^ i) (Finset.range (orderOf x)) = (zpowers x : Set G).toFinset := by
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fintype G
    x : G
    inst✝ : DecidableEq G
    ⊢ Eq (Finset.image (fun i => HPow.hPow x i) (Finset.range (orderOf x))) (↑(Sub …
  -/
  letI : Fintype (zpowers x) := (Subgroup.zpowers x).instFintypeSubtypeMemOfDecidablePred
  /-
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fintype G
    x : G
    inst✝ : DecidableEq G
    this : Fintype (Subtype fun x_1 => Membership.mem (Subgroup.zpowers x) x_1) := …
    ⊢ Eq (Finset.image (fun i => HPow.hPow x i) (Finset.range (orderOf x))) (↑(Sub …
  -/
  ext x
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    inst✝¹ : Fintype G
    x✝ : G
    inst✝ : DecidableEq G
    this : Fintype (Subtype fun x => Membership.mem (Subgroup.zpowers x✝) x) := (S …
    x : G
    ⊢ Iff (Membership.mem (Finset.image (fun i => HPow.hPow x✝ i) (Finset.range (o …
  -/
  rw [Set.mem_toFinset, SetLike.mem_coe, mem_zpowers_iff_mem_range_orderOf]
  /-
    🎉 no goals
  -/

/- TODO: Generalise to `Finite` + `CancelMonoid`. -/

@[to_additive gcd_nsmul_card_eq_zero_iff]
theorem pow_gcd_card_eq_one_iff : x ^ n = 1 ↔ x ^ gcd n (Fintype.card G) = 1 :=
  ⟨fun h => pow_gcd_eq_one _ h <| pow_card_eq_one, fun h => by
    /-
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Fintype G
      x : G
      n : Nat
      h : Eq (HPow.hPow x (n.gcd (Fintype.card G))) 1
      ⊢ Eq (HPow.hPow x n) 1
    -/
    let ⟨m, hm⟩ := gcd_dvd_left n (Fintype.card G)
    /-
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : Fintype G
      x : G
      n : Nat
      h : Eq (HPow.hPow x (n.gcd (Fintype.card G))) 1
      m : Nat
      hm : Eq n (HMul.hMul (n.gcd (Fintype.card G)) m)
      ⊢ Eq (HPow.hPow x n) 1
    -/
    rw [hm, pow_mul, h, one_pow]⟩
    /-
      🎉 no goals
    -/


lemma smul_eq_of_le_smul
    {G : Type*} [Group G] [Finite G] {α : Type*} [PartialOrder α] {g : G} {a : α}
    [MulAction G α] [CovariantClass G α HSMul.hSMul LE.le] (h : a ≤ g • a) : g • a = a := by
  /-
    G : Type u_6
    inst✝⁴ : Group G
    inst✝³ : Finite G
    α : Type u_7
    inst✝² : PartialOrder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le a (HSMul.hSMul g a)
    ⊢ Eq (HSMul.hSMul g a) a
  -/
  have key := smul_mono_right g (le_pow_smul h (Nat.card G - 1))
  rw [smul_smul, ← _root_.pow_succ',
    Nat.sub_one_add_one_eq_of_pos Nat.card_pos, pow_card_eq_one', one_smul] at key
  /-
    G : Type u_6
    inst✝⁴ : Group G
    inst✝³ : Finite G
    α : Type u_7
    inst✝² : PartialOrder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le a (HSMul.hSMul g a)
    key : LE.le (HSMul.hSMul g a) a
    ⊢ Eq (HSMul.hSMul g a) a
  -/
  exact le_antisymm key h
  /-
    🎉 no goals
  -/


lemma smul_eq_of_smul_le
    {G : Type*} [Group G] [Finite G] {α : Type*} [PartialOrder α] {g : G} {a : α}
    [MulAction G α] [CovariantClass G α HSMul.hSMul LE.le] (h : g • a ≤ a) : g • a = a := by
  /-
    G : Type u_6
    inst✝⁴ : Group G
    inst✝³ : Finite G
    α : Type u_7
    inst✝² : PartialOrder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le (HSMul.hSMul g a) a
    ⊢ Eq (HSMul.hSMul g a) a
  -/
  have key := smul_mono_right g (pow_smul_le h (Nat.card G - 1))
  rw [smul_smul, ← _root_.pow_succ',
    Nat.sub_one_add_one_eq_of_pos Nat.card_pos, pow_card_eq_one', one_smul] at key
  /-
    G : Type u_6
    inst✝⁴ : Group G
    inst✝³ : Finite G
    α : Type u_7
    inst✝² : PartialOrder α
    g : G
    a : α
    inst✝¹ : MulAction G α
    inst✝ : CovariantClass G α HSMul.hSMul LE.le
    h : LE.le (HSMul.hSMul g a) a
    key : LE.le a (HSMul.hSMul g a)
    ⊢ Eq (HSMul.hSMul g a) a
  -/
  exact le_antisymm h key
  /-
    🎉 no goals
  -/


/-- A nonempty idempotent subset of a finite cancellative monoid is a submonoid -/
@[to_additive "A nonempty idempotent subset of a finite cancellative add monoid is a submonoid"]
def submonoidOfIdempotent {M : Type*} [LeftCancelMonoid M] [Finite M] (S : Set M)
    (hS1 : S.Nonempty) (hS2 : S * S = S) : Submonoid M :=
  have pow_mem (a : M) (ha : a ∈ S) (n : ℕ) : a ^ (n + 1) ∈ S := by
    induction n with
    | zero => rwa [zero_add, pow_one]
    | succ n ih =>
      rw [← hS2, pow_succ]
      exact Set.mul_mem_mul ih ha
  { carrier := S
    one_mem' := by
      /-
        G : Type u_1
        H : Type u_2
        A : Type u_3
        α : Type u_4
        β : Type u_5
        M : Type u_6
        inst✝¹ : LeftCancelMonoid M
        inst✝ : Finite M
        S : Set M
        hS1 : S.Nonempty
        hS2 : Eq (HMul.hMul S S) S
        pow_mem : ∀ (a : M), Membership.mem S a → ∀ (n : Nat), Membership.mem S (HPow. …
        ⊢ Membership.mem { carrier := S, mul_mem' := ⋯ }.carrier 1
      -/
      obtain ⟨a, ha⟩ := hS1
      /-
        case intro
        G : Type u_1
        H : Type u_2
        A : Type u_3
        α : Type u_4
        β : Type u_5
        M : Type u_6
        inst✝¹ : LeftCancelMonoid M
        inst✝ : Finite M
        S : Set M
        hS2 : Eq (HMul.hMul S S) S
        pow_mem : ∀ (a : M), Membership.mem S a → ∀ (n : Nat), Membership.mem S (HPow. …
        a : M
        ha : Membership.mem S a
        ⊢ Membership.mem { carrier := S, mul_mem' := ⋯ }.carrier 1
      -/
      rw [← pow_orderOf_eq_one a, ← tsub_add_cancel_of_le (succ_le_of_lt (orderOf_pos a))]
      /-
        case intro
        G : Type u_1
        H : Type u_2
        A : Type u_3
        α : Type u_4
        β : Type u_5
        M : Type u_6
        inst✝¹ : LeftCancelMonoid M
        inst✝ : Finite M
        S : Set M
        hS2 : Eq (HMul.hMul S S) S
        pow_mem : ∀ (a : M), Membership.mem S a → ∀ (n : Nat), Membership.mem S (HPow. …
        a : M
        ha : Membership.mem S a
        ⊢ Membership.mem { carrier := S, mul_mem' := ⋯ }.carrier (HPow.hPow a (HAdd.hA …
      -/
      exact pow_mem a ha (orderOf a - 1)
      /-
        🎉 no goals
      -/
    mul_mem' := fun ha hb => (congr_arg₂ (· ∈ ·) rfl hS2).mp (Set.mul_mem_mul ha hb) }


/-- A nonempty idempotent subset of a finite group is a subgroup -/
@[to_additive "A nonempty idempotent subset of a finite add group is a subgroup"]
def subgroupOfIdempotent {G : Type*} [Group G] [Finite G] (S : Set G) (hS1 : S.Nonempty)
    (hS2 : S * S = S) : Subgroup G :=
  { submonoidOfIdempotent S hS1 hS2 with
    carrier := S
    inv_mem' := fun {a} ha => show a⁻¹ ∈ submonoidOfIdempotent S hS1 hS2 by
      /-
        G✝ : Type u_1
        H : Type u_2
        A : Type u_3
        α : Type u_4
        β : Type u_5
        G : Type u_6
        inst✝¹ : Group G
        inst✝ : Finite G
        S : Set G
        hS1 : S.Nonempty
        hS2 : Eq (HMul.hMul S S) S
        a : G
        ha : Membership.mem { carrier := S, mul_mem' := ⋯, one_mem' := ⋯ }.carrier a
        ⊢ Membership.mem (submonoidOfIdempotent S hS1 hS2) (Inv.inv a)
      -/
      rw [← one_mul a⁻¹, ← pow_one a, ← pow_orderOf_eq_one a, ← pow_sub a (orderOf_pos a)]
      /-
        G✝ : Type u_1
        H : Type u_2
        A : Type u_3
        α : Type u_4
        β : Type u_5
        G : Type u_6
        inst✝¹ : Group G
        inst✝ : Finite G
        S : Set G
        hS1 : S.Nonempty
        hS2 : Eq (HMul.hMul S S) S
        a : G
        ha : Membership.mem { carrier := S, mul_mem' := ⋯, one_mem' := ⋯ }.carrier a
        ⊢ Membership.mem (submonoidOfIdempotent S hS1 hS2) (HPow.hPow a (HSub.hSub (or …
      -/
      exact pow_mem ha (orderOf a - 1) }
      /-
        🎉 no goals
      -/


/-- If `S` is a nonempty subset of a finite group `G`, then `S ^ |G|` is a subgroup -/
@[to_additive (attr := simps!) smulCardAddSubgroup
  "If `S` is a nonempty subset of a finite add group `G`, then `|G| • S` is a subgroup"]
def powCardSubgroup {G : Type*} [Group G] [Fintype G] (S : Set G) (hS : S.Nonempty) : Subgroup G :=
  have one_mem : (1 : G) ∈ S ^ Fintype.card G := by
    /-
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      G : Type u_6
      inst✝¹ : Group G
      inst✝ : Fintype G
      S : Set G
      hS : S.Nonempty
      ⊢ Membership.mem (HPow.hPow S (Fintype.card G)) 1
    -/
    obtain ⟨a, ha⟩ := hS
    /-
      case intro
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      G : Type u_6
      inst✝¹ : Group G
      inst✝ : Fintype G
      S : Set G
      a : G
      ha : Membership.mem S a
      ⊢ Membership.mem (HPow.hPow S (Fintype.card G)) 1
    -/
    rw [← pow_card_eq_one]
    /-
      case intro
      G✝ : Type u_1
      H : Type u_2
      A : Type u_3
      α : Type u_4
      β : Type u_5
      G : Type u_6
      inst✝¹ : Group G
      inst✝ : Fintype G
      S : Set G
      a : G
      ha : Membership.mem S a
      ⊢ Membership.mem (HPow.hPow S (Fintype.card G)) (HPow.hPow ?m.744790 (Fintype. …
    -/
    exact Set.pow_mem_pow ha
    /-
      🎉 no goals
    -/
  subgroupOfIdempotent (S ^ Fintype.card G) ⟨1, one_mem⟩ <| by
    classical
    apply (Set.eq_of_subset_of_card_le (Set.subset_mul_left _ one_mem) (ge_of_eq _)).symm
    simp_rw [← pow_add,
        Group.card_pow_eq_card_pow_card_univ S (Fintype.card G + Fintype.card G) le_add_self]


protected lemma IsOfFinOrder.eq_one (ha₀ : 0 ≤ a) (ha : IsOfFinOrder a) : a = 1 := by
  /-
    G : Type u_1
    inst✝ : LinearOrderedSemiring G
    a : G
    ha₀ : LE.le 0 a
    ha : IsOfFinOrder a
    ⊢ Eq a 1
  -/
  obtain ⟨n, hn, ha⟩ := ha.exists_pow_eq_one
  /-
    case intro.intro
    G : Type u_1
    inst✝ : LinearOrderedSemiring G
    a : G
    ha₀ : LE.le 0 a
    ha✝ : IsOfFinOrder a
    n : Nat
    hn : LT.lt 0 n
    ha : Eq (HPow.hPow a n) 1
    ⊢ Eq a 1
  -/
  exact (pow_eq_one_iff_of_nonneg ha₀ hn.ne').1 ha
  /-
    🎉 no goals
  -/


protected lemma IsOfFinOrder.eq_neg_one (ha₀ : a ≤ 0) (ha : IsOfFinOrder a) : a = -1 :=
  (sq_eq_one_iff.1 <| ha.pow.eq_one <| sq_nonneg a).resolve_left <| by
    /-
      G : Type u_1
      inst✝ : LinearOrderedRing G
      a : G
      ha₀ : LE.le a 0
      ha : IsOfFinOrder a
      ⊢ Not (Eq a 1)
    -/
    rintro rfl; exact one_pos.not_le ha₀
                /-
                  🎉 no goals
                -/


theorem orderOf_abs_ne_one (h : |x| ≠ 1) : orderOf x = 0 := by
  /-
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    h : Ne (abs x) 1
    ⊢ Eq (orderOf x) 0
  -/
  rw [orderOf_eq_zero_iff']
  /-
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    h : Ne (abs x) 1
    ⊢ ∀ (n : Nat), LT.lt 0 n → Ne (HPow.hPow x n) 1
  -/
  intro n hn hx
  /-
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    h : Ne (abs x) 1
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow x n) 1
    ⊢ False
  -/
  replace hx : |x| ^ n = 1 := by simpa only [abs_one, abs_pow] using congr_arg abs hx
  /-
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    h : Ne (abs x) 1
    n : Nat
    hn : LT.lt 0 n
    hx : Eq (HPow.hPow (abs x) n) 1
    ⊢ False
  -/
  cases' h.lt_or_lt with h h
    /-
      case inl
      G : Type u_1
      inst✝ : LinearOrderedRing G
      x : G
      h✝ : Ne (abs x) 1
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow (abs x) n) 1
      h : LT.lt (abs x) 1
      ⊢ False
    -/
  · exact ((pow_lt_one₀ (abs_nonneg x) h hn.ne').ne hx).elim
    /-
      🎉 no goals
    -/
    /-
      case inr
      G : Type u_1
      inst✝ : LinearOrderedRing G
      x : G
      h✝ : Ne (abs x) 1
      n : Nat
      hn : LT.lt 0 n
      hx : Eq (HPow.hPow (abs x) n) 1
      h : LT.lt 1 (abs x)
      ⊢ False
    -/
  · exact ((one_lt_pow₀ h hn.ne').ne' hx).elim
    /-
      🎉 no goals
    -/


theorem LinearOrderedRing.orderOf_le_two : orderOf x ≤ 2 := by
  /-
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    ⊢ LE.le (orderOf x) 2
  -/
  cases' ne_or_eq |x| 1 with h h
    /-
      case inl
      G : Type u_1
      inst✝ : LinearOrderedRing G
      x : G
      h : Ne (abs x) 1
      ⊢ LE.le (orderOf x) 2
    -/
  · simp [orderOf_abs_ne_one h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝ : LinearOrderedRing G
    x : G
    h : Eq (abs x) 1
    ⊢ LE.le (orderOf x) 2
  -/
  rcases eq_or_eq_neg_of_abs_eq h with (rfl | rfl)
    /-
      case inr.inl
      G : Type u_1
      inst✝ : LinearOrderedRing G
      h : Eq (abs 1) 1
      ⊢ LE.le (orderOf 1) 2
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    G : Type u_1
    inst✝ : LinearOrderedRing G
    h : Eq (abs (-1)) 1
    ⊢ LE.le (orderOf (-1)) 2
  -/
                                     /-
                                       🎉 no goals
                                     -/
  apply orderOf_le_of_pow_eq_one <;> norm_num
                                     /-
                                       🎉 no goals
                                     -/


@[to_additive]
protected theorem Prod.orderOf (x : α × β) : orderOf x = (orderOf x.1).lcm (orderOf x.2) :=
  minimalPeriod_prod_map _ _ _

@[deprecated (since := "2024-02-21")] alias Prod.add_orderOf := Prod.addOrderOf


@[to_additive]
theorem orderOf_fst_dvd_orderOf : orderOf x.1 ∣ orderOf x :=
  minimalPeriod_fst_dvd

@[deprecated (since := "2024-02-21")]
alias add_orderOf_fst_dvd_add_orderOf := addOrderOf_fst_dvd_addOrderOf


@[to_additive]
theorem orderOf_snd_dvd_orderOf : orderOf x.2 ∣ orderOf x :=
  minimalPeriod_snd_dvd

@[deprecated (since := "2024-02-21")] alias
add_orderOf_snd_dvd_add_orderOf := addOrderOf_snd_dvd_addOrderOf


@[to_additive]
theorem IsOfFinOrder.fst {x : α × β} (hx : IsOfFinOrder x) : IsOfFinOrder x.1 :=
  hx.mono orderOf_fst_dvd_orderOf


@[to_additive]
theorem IsOfFinOrder.snd {x : α × β} (hx : IsOfFinOrder x) : IsOfFinOrder x.2 :=
  hx.mono orderOf_snd_dvd_orderOf


@[to_additive IsOfFinAddOrder.prod_mk]
theorem IsOfFinOrder.prod_mk : IsOfFinOrder a → IsOfFinOrder b → IsOfFinOrder (a, b) := by
  /-
    α : Type u_4
    β : Type u_5
    inst✝¹ : Monoid α
    inst✝ : Monoid β
    a : α
    b : β
    ⊢ IsOfFinOrder a → IsOfFinOrder b → IsOfFinOrder { fst := a, snd := b }
  -/
  simpa only [← orderOf_pos_iff, Prod.orderOf] using Nat.lcm_pos
  /-
    🎉 no goals
  -/


@[to_additive]
lemma Prod.orderOf_mk : orderOf (a, b) = Nat.lcm (orderOf a) (orderOf b) :=
  (a, b).orderOf


@[simp]
lemma Nat.cast_card_eq_zero (R) [AddGroupWithOne R] [Fintype R] : (Fintype.card R : R) = 0 := by
  /-
    R : Type u_6
    inst✝¹ : AddGroupWithOne R
    inst✝ : Fintype R
    ⊢ Eq (↑(Fintype.card R)) 0
  -/
  rw [← nsmul_one, card_nsmul_eq_zero]
  /-
    🎉 no goals
  -/


lemma CharP.addOrderOf_one : CharP R (addOrderOf (1 : R)) where
                            /-
                              R : Type u_6
                              inst✝ : NonAssocRing R
                              n : Nat
                              ⊢ Iff (Eq (↑n) 0) (Dvd.dvd (addOrderOf 1) n)
                            -/
  cast_eq_zero_iff' n := by rw [← Nat.smul_one_eq_cast, addOrderOf_dvd_iff_nsmul_eq_zero]
                            /-
                              🎉 no goals
                            -/


variable {R} in
lemma charP_of_ne_zero (hn : card R = p) (hR : ∀ i < p, (i : R) = 0 → i = 0) : CharP R p where
  cast_eq_zero_iff' n := by
    /-
      R : Type u_6
      inst✝¹ : NonAssocRing R
      p : Nat
      inst✝ : Fintype R
      hn : Eq (Fintype.card R) p
      hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
      n : Nat
      ⊢ Iff (Eq (↑n) 0) (Dvd.dvd p n)
    -/
    have H : (p : R) = 0 := by rw [← hn, Nat.cast_card_eq_zero]
    /-
      R : Type u_6
      inst✝¹ : NonAssocRing R
      p : Nat
      inst✝ : Fintype R
      hn : Eq (Fintype.card R) p
      hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
      n : Nat
      H : Eq (↑p) 0
      ⊢ Iff (Eq (↑n) 0) (Dvd.dvd p n)
    -/
    constructor
      /-
        case mp
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        ⊢ Eq (↑n) 0 → Dvd.dvd p n
      -/
    · intro h
      /-
        case mp
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        h : Eq (↑n) 0
        ⊢ Dvd.dvd p n
      -/
      rw [← Nat.mod_add_div n p, Nat.cast_add, Nat.cast_mul, H, zero_mul, add_zero] at h
      /-
        case mp
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        h : Eq (↑(HMod.hMod n p)) 0
        ⊢ Dvd.dvd p n
      -/
      rw [Nat.dvd_iff_mod_eq_zero]
      /-
        case mp
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        h : Eq (↑(HMod.hMod n p)) 0
        ⊢ Eq (HMod.hMod n p) 0
      -/
      apply hR _ (Nat.mod_lt _ _) h
      /-
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        h : Eq (↑(HMod.hMod n p)) 0
        ⊢ GT.gt p 0
      -/
      rw [← hn, gt_iff_lt, Fintype.card_pos_iff]
      /-
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        h : Eq (↑(HMod.hMod n p)) 0
        ⊢ Nonempty R
      -/
      exact ⟨0⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        n : Nat
        H : Eq (↑p) 0
        ⊢ Dvd.dvd p n → Eq (↑n) 0
      -/
    · rintro ⟨n, rfl⟩
      /-
        case mpr.intro
        R : Type u_6
        inst✝¹ : NonAssocRing R
        p : Nat
        inst✝ : Fintype R
        hn : Eq (Fintype.card R) p
        hR : ∀ (i : Nat), LT.lt i p → Eq (↑i) 0 → Eq i 0
        H : Eq (↑p) 0
        n : Nat
        ⊢ Eq (↑(HMul.hMul p n)) 0
      -/
      rw [Nat.cast_mul, H, zero_mul]
      /-
        🎉 no goals
      -/


lemma charP_of_prime_pow_injective (R) [Ring R] [Fintype R] (p n : ℕ) [hp : Fact p.Prime]
    (hn : card R = p ^ n) (hR : ∀ i ≤ n, (p : R) ^ i = 0 → i = n) : CharP R (p ^ n) := by
  /-
    R : Type u_6
    inst✝¹ : Ring R
    inst✝ : Fintype R
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Eq (Fintype.card R) (HPow.hPow p n)
    hR : ∀ (i : Nat), LE.le i n → Eq (HPow.hPow (↑p) i) 0 → Eq i n
    ⊢ CharP R (HPow.hPow p n)
  -/
  obtain ⟨c, hc⟩ := CharP.exists R
  /-
    case intro
    R : Type u_6
    inst✝¹ : Ring R
    inst✝ : Fintype R
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Eq (Fintype.card R) (HPow.hPow p n)
    hR : ∀ (i : Nat), LE.le i n → Eq (HPow.hPow (↑p) i) 0 → Eq i n
    c : Nat
    hc : CharP R c
    ⊢ CharP R (HPow.hPow p n)
  -/
  have hcpn : c ∣ p ^ n := by rw [← CharP.cast_eq_zero_iff R c, ← hn, Nat.cast_card_eq_zero]
  /-
    case intro
    R : Type u_6
    inst✝¹ : Ring R
    inst✝ : Fintype R
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Eq (Fintype.card R) (HPow.hPow p n)
    hR : ∀ (i : Nat), LE.le i n → Eq (HPow.hPow (↑p) i) 0 → Eq i n
    c : Nat
    hc : CharP R c
    hcpn : Dvd.dvd c (HPow.hPow p n)
    ⊢ CharP R (HPow.hPow p n)
  -/
  obtain ⟨i, hi, rfl⟩ : ∃ i ≤ n, c = p ^ i := by rwa [Nat.dvd_prime_pow hp.1] at hcpn
  /-
    case intro.intro.intro
    R : Type u_6
    inst✝¹ : Ring R
    inst✝ : Fintype R
    p n : Nat
    hp : Fact (Nat.Prime p)
    hn : Eq (Fintype.card R) (HPow.hPow p n)
    hR : ∀ (i : Nat), LE.le i n → Eq (HPow.hPow (↑p) i) 0 → Eq i n
    i : Nat
    hi : LE.le i n
    hc : CharP R (HPow.hPow p i)
    hcpn : Dvd.dvd (HPow.hPow p i) (HPow.hPow p n)
    ⊢ CharP R (HPow.hPow p n)
  -/
  obtain rfl : i = n := hR i hi <| by rw [← Nat.cast_pow, CharP.cast_eq_zero]
  /-
    case intro.intro.intro
    R : Type u_6
    inst✝¹ : Ring R
    inst✝ : Fintype R
    p : Nat
    hp : Fact (Nat.Prime p)
    i : Nat
    hc : CharP R (HPow.hPow p i)
    hn : Eq (Fintype.card R) (HPow.hPow p i)
    hR : ∀ (i_1 : Nat), LE.le i_1 i → Eq (HPow.hPow (↑p) i_1) 0 → Eq i_1 i
    hi : LE.le i i
    hcpn : Dvd.dvd (HPow.hPow p i) (HPow.hPow p i)
    ⊢ CharP R (HPow.hPow p i)
  -/
  assumption
  /-
    🎉 no goals
  -/


@[to_additive]
lemma orderOf_eq [Group G] (a : G) {x y : G} (h : SemiconjBy a x y) : orderOf x = orderOf y := by
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    h : SemiconjBy a x y
    ⊢ Eq (orderOf x) (orderOf y)
  -/
  rw [orderOf_eq_orderOf_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    h : SemiconjBy a x y
    ⊢ ∀ (n : Nat), Iff (Eq (HPow.hPow x n) 1) (Eq (HPow.hPow y n) 1)
  -/
  intro n
  /-
    G : Type u_1
    inst✝ : Group G
    a x y : G
    h : SemiconjBy a x y
    n : Nat
    ⊢ Iff (Eq (HPow.hPow x n) 1) (Eq (HPow.hPow y n) 1)
  -/
  exact (h.pow_right n).eq_one_iff
  /-
    🎉 no goals
  -/


lemma orderOf_piMulSingle {ι : Type*} [DecidableEq ι] {M : ι → Type*} [(i : ι) → Monoid (M i)]
    (i : ι) (g : M i) :
    orderOf (Pi.mulSingle i g) = orderOf g :=
  orderOf_injective (MonoidHom.mulSingle M i) (Pi.mulSingle_injective M i) g


