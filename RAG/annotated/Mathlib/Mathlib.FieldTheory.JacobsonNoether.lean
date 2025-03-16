local notation3 "k" => Subring.center D


/-- If `D` is a purely inseparable extension of `k` with characteristic `p`,
  then for every element `a` of `D`, there exists a natural number `n`
  such that `a ^ (p ^ n)` is contained in `k`. -/
lemma exists_pow_mem_center_of_inseparable (p : ℕ) [hchar : ExpChar D p] (a : D)
    (hinsep : ∀ x : D, IsSeparable k x → x ∈ k) : ∃ n, a ^ (p ^ n) ∈ k := by
  /-
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    ⊢ Exists fun n => Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p  …
  -/
  have := (@isPurelyInseparable_iff_pow_mem k D _ _ _ _ p (ExpChar.expChar_center_iff.2 hchar)).1
  have pure : IsPurelyInseparable k D := ⟨Algebra.IsAlgebraic.isIntegral, fun x hx ↦ by
    rw [RingHom.mem_range, Subtype.exists]
    exact ⟨x, ⟨hinsep x hx, rfl⟩⟩⟩
  /-
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    this : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D) …
    pure : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D) …
    ⊢ Exists fun n => Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p  …
  -/
  obtain ⟨n, ⟨m, hm⟩⟩ := this pure a
  /-
    case intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    this : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D) …
    pure : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D) …
    n : Nat
    m : Subtype fun x => Membership.mem (Subring.center D) x
    hm : Eq ((algebraMap (Subtype fun x => Membership.mem (Subring.center D) x) D) …
    ⊢ Exists fun n => Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p  …
  -/
  have := Subalgebra.range_subset (R := k) ⟨(k).toSubsemiring, fun r ↦ r.2⟩
  /-
    case intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    this✝ : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D …
    pure : IsPurelyInseparable (Subtype fun x => Membership.mem (Subring.center D) …
    n : Nat
    m : Subtype fun x => Membership.mem (Subring.center D) x
    hm : Eq ((algebraMap (Subtype fun x => Membership.mem (Subring.center D) x) D) …
    this : HasSubset.Subset (Set.range ⇑(algebraMap (Subtype fun x => Membership.m …
    ⊢ Exists fun n => Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p  …
  -/
  exact ⟨n, Set.mem_of_subset_of_mem this <| Set.mem_range.2 ⟨m, hm⟩⟩
  /-
    🎉 no goals
  -/


/-- If `D` is a purely inseparable extension of `k` with characteristic `p`,
  then for every element `a` of `D \ k`, there exists a natural number `n`
  **greater than 0** such that `a ^ (p ^ n)` is contained in `k`. -/
lemma exists_pow_mem_center_of_inseparable' (p : ℕ) [ExpChar D p] {a : D}
    (ha : a ∉ k) (hinsep : ∀ x : D, IsSeparable k x → x ∈ k) : ∃ n, 1 ≤ n ∧ a ^ (p ^ n) ∈ k := by
  /-
    D : Type u_1
    inst✝² : DivisionRing D
    inst✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center  …
    p : Nat
    inst✝ : ExpChar D p
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    ⊢ Exists fun n => And (LE.le 1 n) (Membership.mem (Subring.center D) (HPow.hPo …
  -/
  obtain ⟨n, hn⟩ := exists_pow_mem_center_of_inseparable p a hinsep
  /-
    case intro
    D : Type u_1
    inst✝² : DivisionRing D
    inst✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center  …
    p : Nat
    inst✝ : ExpChar D p
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    n : Nat
    hn : Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p n))
    ⊢ Exists fun n => And (LE.le 1 n) (Membership.mem (Subring.center D) (HPow.hPo …
  -/
  by_cases nzero : n = 0
    /-
      case pos
      D : Type u_1
      inst✝² : DivisionRing D
      inst✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center  …
      p : Nat
      inst✝ : ExpChar D p
      a : D
      ha : Not (Membership.mem (Subring.center D) a)
      hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
      n : Nat
      hn : Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p n))
      nzero : Eq n 0
      ⊢ Exists fun n => And (LE.le 1 n) (Membership.mem (Subring.center D) (HPow.hPo …
    -/
  · rw [nzero, pow_zero, pow_one] at hn
    /-
      case pos
      D : Type u_1
      inst✝² : DivisionRing D
      inst✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center  …
      p : Nat
      inst✝ : ExpChar D p
      a : D
      ha : Not (Membership.mem (Subring.center D) a)
      hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
      n : Nat
      hn : Membership.mem (Subring.center D) a
      nzero : Eq n 0
      ⊢ Exists fun n => And (LE.le 1 n) (Membership.mem (Subring.center D) (HPow.hPo …
    -/
    exact (ha hn).elim
    /-
      🎉 no goals
    -/
    /-
      case neg
      D : Type u_1
      inst✝² : DivisionRing D
      inst✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center  …
      p : Nat
      inst✝ : ExpChar D p
      a : D
      ha : Not (Membership.mem (Subring.center D) a)
      hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
      n : Nat
      hn : Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPow p n))
      nzero : Not (Eq n 0)
      ⊢ Exists fun n => And (LE.le 1 n) (Membership.mem (Subring.center D) (HPow.hPo …
    -/
  · exact ⟨n, ⟨Nat.one_le_iff_ne_zero.mpr nzero, hn⟩⟩
    /-
      🎉 no goals
    -/


/-- If `D` is a purely inseparable extension of `k` of characteristic `p`,
  then for every element `a` of `D \ k`, there exists a natural number `m`
  greater than 0 such that `(a * x - x * a) ^ n = 0` (as linear maps) for
  every `n` greater than `(p ^ m)`. -/
lemma exist_pow_eq_zero_of_le (p : ℕ) [hchar : ExpChar D p]
    {a : D} (ha : a ∉ k) (hinsep : ∀ x : D, IsSeparable k x → x ∈ k):
  ∃ m, 1 ≤ m ∧ ∀ n, p ^ m ≤ n → (ad k D a)^[n] = 0 := by
  /-
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    ⊢ Exists fun m => And (LE.le 1 m) (∀ (n : Nat), LE.le (HPow.hPow p m) n → Eq ( …
  -/
  obtain ⟨m, hm⟩ := exists_pow_mem_center_of_inseparable' p ha hinsep
  /-
    case intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    p : Nat
    hchar : ExpChar D p
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    hinsep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cent …
    m : Nat
    hm : And (LE.le 1 m) (Membership.mem (Subring.center D) (HPow.hPow a (HPow.hPo …
    ⊢ Exists fun m => And (LE.le 1 m) (∀ (n : Nat), LE.le (HPow.hPow p m) n → Eq ( …
  -/
  refine ⟨m, ⟨hm.1, fun n hn ↦ ?_⟩⟩
  have inter : (ad k D a)^[p ^ m] = 0 := by
    ext x
    rw [ad_eq_lmul_left_sub_lmul_right, ← pow_apply, Pi.sub_apply,
      sub_pow_expChar_pow_of_commute p m (commute_mulLeft_right a a), sub_apply,
      pow_mulLeft, mulLeft_apply, pow_mulRight, mulRight_apply, Pi.zero_apply,
      Subring.mem_center_iff.1 hm.2 x]
    exact sub_eq_zero_of_eq rfl
  rw [(Nat.sub_eq_iff_eq_add hn).1 rfl, Function.iterate_add, inter, Pi.comp_zero,
    iterate_map_zero, Function.const_zero]


variable (D) in
/-- Jacobson-Noether theorem: For a non-commutative division algebra
  `D` that is algebraic over its center `k`, there exists an element
  `x` of `D \ k` that is separable over `k`. -/
theorem exists_separable_and_not_isCentral (H : k ≠ (⊤ : Subring D)) :
    ∃ x : D, x ∉ k ∧ IsSeparable k x := by
  /-
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    ⊢ Exists fun x => And (Not (Membership.mem (Subring.center D) x)) (IsSeparable …
  -/
  obtain ⟨p, hp⟩ := ExpChar.exists D
  /-
    case intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    ⊢ Exists fun x => And (Not (Membership.mem (Subring.center D) x)) (IsSeparable …
  -/
  by_contra! insep
  replace insep : ∀ x : D, IsSeparable k x → x ∈ k :=
    fun x h ↦ Classical.byContradiction fun hx ↦ insep x hx h
  -- The element `a` below is in `D` but not in `k`.
  /-
    case intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    ⊢ False
  -/
  obtain ⟨a, ha⟩ := not_forall.mp <| mt (Subring.eq_top_iff' k).mpr H
  /-
    case intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ⊢ False
  -/
  have ha₀ : a ≠ 0 := fun nh ↦ nh ▸ ha <| Subring.zero_mem k
  -- We construct another element `b` that does not commute with `a`.
  obtain ⟨b, hb1⟩ : ∃ b : D , ad k D a b ≠ 0 := by
    rw [Subring.mem_center_iff, not_forall] at ha
    use ha.choose
    show a * ha.choose - ha.choose * a ≠ 0
    simpa only [ne_eq, sub_eq_zero] using Ne.symm ha.choose_spec
  -- We find a maximum natural number `n` such that `(a * x - x * a) ^ n b ≠ 0`.
  obtain ⟨n, hn, hb⟩ : ∃ n, 0 < n ∧ (ad k D a)^[n] b ≠ 0 ∧ (ad k D a)^[n + 1] b = 0 := by
    obtain ⟨m, -, hm2⟩ := exist_pow_eq_zero_of_le p ha insep
    have h_exist : ∃ n, 0 < n ∧ (ad k D a)^[n + 1] b = 0 := ⟨p ^ m,
      ⟨expChar_pow_pos D p m, by rw [hm2 (p ^ m + 1) (Nat.le_add_right _ _)]; rfl⟩⟩
    classical
    refine ⟨Nat.find h_exist, ⟨(Nat.find_spec h_exist).1, ?_, (Nat.find_spec h_exist).2⟩⟩
    set t := (Nat.find h_exist - 1 : ℕ) with ht
    by_cases h_pos : 0 < t
    · convert (ne_eq _ _) ▸ not_and.mp (Nat.find_min h_exist (m := t) (by omega)) h_pos
      omega
    · suffices h_find: Nat.find h_exist = 1 by
        rwa [h_find]
      rw [not_lt, Nat.le_zero, ht, Nat.sub_eq_zero_iff_le] at h_pos
      linarith [(Nat.find_spec h_exist).1]
  -- We define `c` to be the value that we proved above to be non-zero.
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    hb : And (Ne (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem ( …
    ⊢ False
  -/
  set c := (ad k D a)^[n] b with hc_def
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    ⊢ False
  -/
  let _ : Invertible c := ⟨c⁻¹, inv_mul_cancel₀ hb.1, mul_inv_cancel₀ hb.1⟩
  -- We prove that `c` commutes with `a`.
  have hc : a * c = c * a := by
    apply eq_of_sub_eq_zero
    rw [← mulLeft_apply (R := k), ← mulRight_apply (R := k)]
    suffices ad k D a c = 0 from by
      rw [← this]; rfl
    rw [← Function.iterate_succ_apply' (ad k D a) n b, hb.2]
  -- We now make some computation to obtain the final equation.
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    ⊢ False
  -/
  set d := c⁻¹ * a * (ad k D a)^[n - 1] b with hd_def
  have hc': c⁻¹ * a = a * c⁻¹ := by
    apply_fun (c⁻¹ * · * c⁻¹) at hc
    rw [mul_assoc, mul_assoc, mul_inv_cancel₀ hb.1, mul_one, ← mul_assoc,
      inv_mul_cancel₀ hb.1, one_mul] at hc
    exact hc
  have c_eq : a * (ad k D a)^[n - 1] b - (ad k D a)^[n - 1] b * a = c := by
    rw [hc_def, ← Nat.sub_add_cancel hn, Function.iterate_succ_apply' (ad k D a) _ b]; rfl
  have eq1 : c⁻¹ * a * (ad k D a)^[n - 1] b - c⁻¹ * (ad k D a)^[n - 1] b * a = 1 := by
    simp_rw [mul_assoc, (mul_sub_left_distrib c⁻¹ _ _).symm, c_eq, inv_mul_cancel_of_invertible]
  -- We show that `a` commutes with `d`.
  have deq : a * d - d * a = a := by
    nth_rw 3 [← mul_one a]
    rw [hd_def, ← eq1, mul_sub, mul_assoc _ _ a, sub_right_inj, hc',
      ← mul_assoc, ← mul_assoc, ← mul_assoc]
  -- This then yields a contradiction.
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    d : D := HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra.ad (S …
    hd_def : Eq d (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra …
    hc' : Eq (HMul.hMul (Inv.inv c) a) (HMul.hMul a (Inv.inv c))
    c_eq : Eq (HSub.hSub (HMul.hMul a (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun  …
    eq1 : Eq (HSub.hSub (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieA …
    deq : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul d a)) a
    ⊢ False
  -/
  apply_fun (a⁻¹ * · ) at deq
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    d : D := HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra.ad (S …
    hd_def : Eq d (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra …
    hc' : Eq (HMul.hMul (Inv.inv c) a) (HMul.hMul a (Inv.inv c))
    c_eq : Eq (HSub.hSub (HMul.hMul a (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun  …
    eq1 : Eq (HSub.hSub (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieA …
    deq : Eq (HMul.hMul (Inv.inv a) (HSub.hSub (HMul.hMul a d) (HMul.hMul d a))) ( …
    ⊢ False
  -/
  rw [mul_sub, ← mul_assoc, inv_mul_cancel₀ ha₀, one_mul, ← mul_assoc, sub_eq_iff_eq_add] at deq
  /-
    case intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    d : D := HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra.ad (S …
    hd_def : Eq d (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra …
    hc' : Eq (HMul.hMul (Inv.inv c) a) (HMul.hMul a (Inv.inv c))
    c_eq : Eq (HSub.hSub (HMul.hMul a (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun  …
    eq1 : Eq (HSub.hSub (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieA …
    deq : Eq d (HAdd.hAdd 1 (HMul.hMul (HMul.hMul (Inv.inv a) d) a))
    ⊢ False
  -/
  obtain ⟨r, hr⟩ := exists_pow_mem_center_of_inseparable p d insep
  /-
    case intro.intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    d : D := HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra.ad (S …
    hd_def : Eq d (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra …
    hc' : Eq (HMul.hMul (Inv.inv c) a) (HMul.hMul a (Inv.inv c))
    c_eq : Eq (HSub.hSub (HMul.hMul a (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun  …
    eq1 : Eq (HSub.hSub (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieA …
    deq : Eq d (HAdd.hAdd 1 (HMul.hMul (HMul.hMul (Inv.inv a) d) a))
    r : Nat
    hr : Membership.mem (Subring.center D) (HPow.hPow d (HPow.hPow p r))
    ⊢ False
  -/
  apply_fun (· ^ (p ^ r)) at deq
  rw [add_pow_expChar_pow_of_commute p r (Commute.one_left _) , one_pow,
    GroupWithZero.conj_pow₀ ha₀, ← hr.comm, mul_assoc, inv_mul_cancel₀ ha₀, mul_one,
    self_eq_add_left] at deq
  /-
    case intro.intro.intro.intro.intro.intro
    D : Type u_1
    inst✝¹ : DivisionRing D
    inst✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D …
    H : Ne (Subring.center D) Top.top
    p : Nat
    hp : ExpChar D p
    insep : ∀ (x : D), IsSeparable (Subtype fun x => Membership.mem (Subring.cente …
    a : D
    ha : Not (Membership.mem (Subring.center D) a)
    ha₀ : Ne a 0
    b : D
    hb1 : Ne (((LieAlgebra.ad (Subtype fun x => Membership.mem (Subring.center D)  …
    n : Nat
    hn : LT.lt 0 n
    c : D := Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem (Subri …
    hb : And (Ne c 0) (Eq (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Members …
    hc_def : Eq c (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun x => Membership.mem  …
    x✝ : Invertible c := { invOf := Inv.inv c, invOf_mul_self := ⋯, mul_invOf_self …
    hc : Eq (HMul.hMul a c) (HMul.hMul c a)
    d : D := HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra.ad (S …
    hd_def : Eq d (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieAlgebra …
    hc' : Eq (HMul.hMul (Inv.inv c) a) (HMul.hMul a (Inv.inv c))
    c_eq : Eq (HSub.hSub (HMul.hMul a (Nat.iterate (⇑((LieAlgebra.ad (Subtype fun  …
    eq1 : Eq (HSub.hSub (HMul.hMul (HMul.hMul (Inv.inv c) a) (Nat.iterate (⇑((LieA …
    r : Nat
    hr : Membership.mem (Subring.center D) (HPow.hPow d (HPow.hPow p r))
    deq : Eq 1 0
    ⊢ False
  -/
  exact one_ne_zero deq
  /-
    🎉 no goals
  -/


open Subring Algebra in
/-- Jacobson-Noether theorem: For a non-commutative division algebra `D`
  that is algebraic over a field `L`, if the center of
  `D` coincides with `L`, then there exist an element `x` of `D \ L`
  that is separable over `L`. -/
theorem exists_separable_and_not_isCentral' {L D : Type*} [Field L] [DivisionRing D]
    [Algebra L D] [Algebra.IsAlgebraic L D] [Algebra.IsCentral L D]
  (hneq : (⊥ : Subalgebra L D) ≠ ⊤) :
    ∃ x : D, x ∉ (⊥ : Subalgebra L D) ∧ IsSeparable L x := by
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  have hcenter : Subalgebra.center L D = ⊥ := le_bot_iff.mp IsCentral.out
  have ntrivial : Subring.center D ≠ ⊤ :=
    congr(Subalgebra.toSubring $hcenter).trans_ne (Subalgebra.toSubring_injective.ne hneq)
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  set φ := Subalgebra.equivOfEq (⊥ : Subalgebra L D) (.center L D) hcenter.symm
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  set equiv : L ≃+* (center D) := ((botEquiv L D).symm.trans φ).toRingEquiv
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  let _ : Algebra L (center D) := equiv.toRingHom.toAlgebra
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    x✝ : Algebra L (Subtype fun x => Membership.mem (Subring.center D) x) := equiv …
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  let _ : Algebra (center D) L := equiv.symm.toRingHom.toAlgebra
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    x✝¹ : Algebra L (Subtype fun x => Membership.mem (Subring.center D) x) := equi …
    x✝ : Algebra (Subtype fun x => Membership.mem (Subring.center D) x) L := equiv …
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  have _ : IsScalarTower L (center D) D := .of_algebraMap_eq fun _ ↦ rfl
  have _ : IsScalarTower (center D) L D := .of_algebraMap_eq fun x ↦ by
    rw [IsScalarTower.algebraMap_apply L (center D)]
    congr
    exact (equiv.apply_symm_apply x).symm
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    x✝³ : Algebra L (Subtype fun x => Membership.mem (Subring.center D) x) := equi …
    x✝² : Algebra (Subtype fun x => Membership.mem (Subring.center D) x) L := equi …
    x✝¹ : IsScalarTower L (Subtype fun x => Membership.mem (Subring.center D) x) D
    x✝ : IsScalarTower (Subtype fun x => Membership.mem (Subring.center D) x) L D
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  have _ : Algebra.IsAlgebraic (center D) D := .tower_top (K := L) _
  /-
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    x✝⁴ : Algebra L (Subtype fun x => Membership.mem (Subring.center D) x) := equi …
    x✝³ : Algebra (Subtype fun x => Membership.mem (Subring.center D) x) L := equi …
    x✝² : IsScalarTower L (Subtype fun x => Membership.mem (Subring.center D) x) D
    x✝¹ : IsScalarTower (Subtype fun x => Membership.mem (Subring.center D) x) L D
    x✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D) x …
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  obtain ⟨x, hxd, hx⟩ := exists_separable_and_not_isCentral D ntrivial
  /-
    case intro.intro
    L : Type u_2
    D : Type u_3
    inst✝⁴ : Field L
    inst✝³ : DivisionRing D
    inst✝² : Algebra L D
    inst✝¹ : Algebra.IsAlgebraic L D
    inst✝ : Algebra.IsCentral L D
    hneq : Ne Bot.bot Top.top
    hcenter : Eq (Subalgebra.center L D) Bot.bot
    ntrivial : Ne (Subring.center D) Top.top
    φ : AlgEquiv L (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => M …
    equiv : RingEquiv L (Subtype fun x => Membership.mem (Subring.center D) x) :=  …
    x✝⁴ : Algebra L (Subtype fun x => Membership.mem (Subring.center D) x) := equi …
    x✝³ : Algebra (Subtype fun x => Membership.mem (Subring.center D) x) L := equi …
    x✝² : IsScalarTower L (Subtype fun x => Membership.mem (Subring.center D) x) D
    x✝¹ : IsScalarTower (Subtype fun x => Membership.mem (Subring.center D) x) L D
    x✝ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Subring.center D) x …
    x : D
    hxd : Not (Membership.mem (Subring.center D) x)
    hx : IsSeparable (Subtype fun x => Membership.mem (Subring.center D) x) x
    ⊢ Exists fun x => And (Not (Membership.mem Bot.bot x)) (IsSeparable L x)
  -/
  exact ⟨x, ⟨by rwa [← Subalgebra.center_toSubring L, hcenter] at hxd, IsSeparable.tower_top _ hx⟩⟩
  /-
    🎉 no goals
  -/


