open Nat.Primes in
@[to_additive tsum_eq_tsum_primes_of_support_subset_prime_powers]
lemma tprod_eq_tprod_primes_of_mulSupport_subset_prime_powers {f : ℕ → α}
    (hfm : Multipliable f) (hf : Function.mulSupport f ⊆ {n | IsPrimePow n}) :
    ∏' n : ℕ, f n = ∏' (p : Nat.Primes) (k : ℕ), f (p ^ (k + 1)) := by
  have hfm' : Multipliable fun pk : Nat.Primes × ℕ ↦ f (pk.fst ^ (pk.snd + 1)) :=
    prodNatEquiv.symm.multipliable_iff.mp <| by
      simpa only [← coe_prodNatEquiv_apply, Prod.eta, Function.comp_def, Equiv.apply_symm_apply]
        using hfm.subtype _
  simp only [← tprod_subtype_eq_of_mulSupport_subset hf, Set.coe_setOf, ← prodNatEquiv.tprod_eq,
    ← tprod_prod hfm']
  /-
    α : Type u_1
    inst✝⁴ : CommGroup α
    inst✝³ : UniformSpace α
    inst✝² : UniformGroup α
    inst✝¹ : CompleteSpace α
    inst✝ : T0Space α
    f : Nat → α
    hfm : Multipliable f
    hf : HasSubset.Subset (Function.mulSupport f) (setOf fun n => IsPrimePow n)
    hfm' : Multipliable fun pk => f (HPow.hPow (↑pk.1) (HAdd.hAdd pk.2 1))
    ⊢ Eq (tprod fun c => f ↑(Nat.Primes.prodNatEquiv c)) (tprod fun p => f (HPow.h …
  -/
  refine tprod_congr fun (p, k) ↦ congrArg f <| coe_prodNatEquiv_apply ..
  /-
    🎉 no goals
  -/


@[to_additive tsum_eq_tsum_primes_add_tsum_primes_of_support_subset_prime_powers]
lemma tprod_eq_tprod_primes_mul_tprod_primes_of_mulSupport_subset_prime_powers {f : ℕ → α}
    (hfm : Multipliable f) (hf : Function.mulSupport f ⊆ {n | IsPrimePow n}) :
    ∏' n : ℕ, f n = (∏' p : Nat.Primes, f p) *  ∏' (p : Nat.Primes) (k : ℕ), f (p ^ (k + 2)) := by
  /-
    α : Type u_1
    inst✝⁴ : CommGroup α
    inst✝³ : UniformSpace α
    inst✝² : UniformGroup α
    inst✝¹ : CompleteSpace α
    inst✝ : T0Space α
    f : Nat → α
    hfm : Multipliable f
    hf : HasSubset.Subset (Function.mulSupport f) (setOf fun n => IsPrimePow n)
    ⊢ Eq (tprod fun n => f n) (HMul.hMul (tprod fun p => f ↑p) (tprod fun p => tpr …
  -/
  rw [tprod_eq_tprod_primes_of_mulSupport_subset_prime_powers hfm hf]
  have hfs' (p : Nat.Primes) : Multipliable fun k : ℕ ↦ f (p ^ (k + 1)) :=
    hfm.comp_injective <| (strictMono_nat_of_lt_succ
      fun k ↦ pow_lt_pow_right₀ p.prop.one_lt <| lt_add_one (k + 1)).injective
  conv_lhs =>
    enter [1, p]; rw [tprod_eq_zero_mul (hfs' p), zero_add, pow_one]
    enter [2, 1, k]; rw [add_assoc, one_add_one_eq_two]
  exact tprod_mul (Multipliable.subtype hfm _) <|
    Multipliable.prod (f := fun (pk : Nat.Primes × ℕ) ↦ f (pk.1 ^ (pk.2 + 2))) <|
    hfm.comp_injective <| Subtype.val_injective |>.comp
    Nat.Primes.prodNatEquiv.injective |>.comp <|
    Function.Injective.prodMap (fun ⦃_ _⦄ a ↦ a) <| add_left_injective 1


/-- The von Mangoldt function restricted to the residue class `a` mod `q`. -/
noncomputable abbrev residueClass : ℕ → ℝ :=
  {n : ℕ | (n : ZMod q) = a}.indicator (vonMangoldt ·)


lemma residueClass_nonneg (n : ℕ) : 0 ≤ residueClass a n :=
  Set.indicator_apply_nonneg fun _ ↦ vonMangoldt_nonneg


lemma residueClass_le (n : ℕ) : residueClass a n ≤ vonMangoldt n :=
  Set.indicator_apply_le' (fun _ ↦ le_rfl) (fun _ ↦ vonMangoldt_nonneg)


@[simp]
lemma residueClass_apply_zero : residueClass a 0 = 0 := by
  simp only [Set.indicator_apply_eq_zero, Set.mem_setOf_eq, Nat.cast_zero, map_zero, ofReal_zero,
    implies_true]


lemma abscissaOfAbsConv_residueClass_le_one :
    abscissaOfAbsConv ↗(residueClass a) ≤ 1 := by
  /-
    q : Nat
    a : ZMod q
    ⊢ LE.le (LSeries.abscissaOfAbsConv fun n => ↑(ArithmeticFunction.vonMangoldt.r …
  -/
  refine abscissaOfAbsConv_le_of_forall_lt_LSeriesSummable fun y hy ↦ ?_
  /-
    q : Nat
    a : ZMod q
    y : Real
    hy : LT.lt 1 y
    ⊢ LSeriesSummable (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n) …
  -/
  unfold LSeriesSummable
  /-
    q : Nat
    a : ZMod q
    y : Real
    hy : LT.lt 1 y
    ⊢ Summable (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueCla …
  -/
  have := LSeriesSummable_vonMangoldt <| show 1 < (y : ℂ).re by simp only [ofReal_re, hy]
  /-
    q : Nat
    a : ZMod q
    y : Real
    hy : LT.lt 1 y
    this : LSeriesSummable (fun n => ↑(ArithmeticFunction.vonMangoldt n)) ↑y
    ⊢ Summable (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueCla …
  -/
  convert this.indicator {n : ℕ | (n : ZMod q) = a}
  /-
    case h.e'_5
    q : Nat
    a : ZMod q
    y : Real
    hy : LT.lt 1 y
    this : LSeriesSummable (fun n => ↑(ArithmeticFunction.vonMangoldt n)) ↑y
    ⊢ Eq (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n …
  -/
  ext1 n
  /-
    case h.e'_5.h
    q : Nat
    a : ZMod q
    y : Real
    hy : LT.lt 1 y
    this : LSeriesSummable (fun n => ↑(ArithmeticFunction.vonMangoldt n)) ↑y
    n : Nat
    ⊢ Eq (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n …
  -/
  by_cases hn : (n : ZMod q) = a
  · simp +contextual only [term, Set.indicator, Set.mem_setOf_eq, hn, ↓reduceIte, apply_ite,
      ite_self]
  · simp +contextual only [term, Set.mem_setOf_eq, hn, not_false_eq_true, Set.indicator_of_not_mem,
      ofReal_zero, zero_div, ite_self]


/-- The set we are interested in (prime numbers in the residue class `a`) is the same as the support
of `ArithmeticFunction.vonMangoldt.residueClass` restricted to primes (and divided by `n`;
this is how this result is used later). -/
lemma support_residueClass_prime_div :
    Function.support (fun n : ℕ ↦ (if n.Prime then residueClass a n else 0) / n) =
      {p : ℕ | p.Prime ∧ (p : ZMod q) = a} := by
  simp only [Function.support, ne_eq, div_eq_zero_iff, ite_eq_right_iff,
    Set.indicator_apply_eq_zero, Set.mem_setOf_eq, Nat.cast_eq_zero, not_or, Classical.not_imp]
  /-
    q : Nat
    a : ZMod q
    ⊢ Eq (setOf fun x => And (And (Nat.Prime x) (And (Eq (↑x) a) (Not (Eq (Arithme …
  -/
  ext1 p
  /-
    case h
    q : Nat
    a : ZMod q
    p : Nat
    ⊢ Iff (Membership.mem (setOf fun x => And (And (Nat.Prime x) (And (Eq (↑x) a)  …
  -/
  simp only [Set.mem_setOf_eq]
  exact ⟨fun H ↦ ⟨H.1.1, H.1.2.1⟩,
    fun H ↦ ⟨⟨H.1, H.2, vonMangoldt_ne_zero_iff.mpr H.1.isPrimePow⟩, H.1.ne_zero⟩⟩


private noncomputable def F₀ (n : ℕ) : ℝ := (if n.Prime then 0 else vonMangoldt n) / n


private noncomputable def F' (pk : Nat.Primes × ℕ) : ℝ := F₀ (pk.1 ^ (pk.2 + 1))


private noncomputable def F'' : Nat.Primes × ℕ → ℝ := F' ∘ (Prod.map _root_.id (· + 1))


private lemma F''_le (p : Nat.Primes) (k : ℕ) : F'' (p, k) ≤ 2 * (p : ℝ)⁻¹ ^ (k + 3 / 2 : ℝ) :=
  calc _
    _ = Real.log p * (p : ℝ)⁻¹ ^ (k + 2) := by
      simp only [F'', Function.comp_apply, F', F₀, Prod.map_apply, id_eq, le_add_iff_nonneg_left,
        zero_le, Nat.Prime.not_prime_pow, ↓reduceIte, vonMangoldt_apply_prime p.prop,
        vonMangoldt_apply_pow (Nat.zero_ne_add_one _).symm, Nat.cast_pow, div_eq_mul_inv,
        inv_pow (p : ℝ) (k + 2)]
    _ ≤ (p: ℝ) ^ (1 / 2 : ℝ) / (1 / 2) * (p : ℝ)⁻¹ ^ (k + 2) :=
        mul_le_mul_of_nonneg_right (Real.log_le_rpow_div p.val.cast_nonneg one_half_pos)
          (pow_nonneg (inv_nonneg_of_nonneg (Nat.cast_nonneg ↑p)) (k + 2))
    _ = 2 * (p : ℝ)⁻¹ ^ (-1 / 2 : ℝ) * (p : ℝ)⁻¹ ^ (k + 2) := by
      simp only [← div_mul, div_one, mul_comm, neg_div, Real.inv_rpow p.val.cast_nonneg,
        ← Real.rpow_neg p.val.cast_nonneg, neg_neg]
    _ = _ := by
      rw [mul_assoc, ← Real.rpow_natCast,
        ← Real.rpow_add <| by have := p.prop.pos; positivity, Nat.cast_add, Nat.cast_two,
        add_comm, add_assoc]
      /-
        p : Nat.Primes
        k : Nat
        ⊢ Eq (HMul.hMul 2 (HPow.hPow (Inv.inv ↑↑p) (HAdd.hAdd (↑k) (HAdd.hAdd 2 (-1 /  …
      -/
      norm_num
      /-
        🎉 no goals
      -/


private lemma summable_F'' : Summable F'' := by
  /-
    ⊢ Summable ArithmeticFunction.vonMangoldt.F''
  -/
  have hp₀ (p : Nat.Primes) : 0 < (p : ℝ)⁻¹ := inv_pos_of_pos (Nat.cast_pos.mpr p.prop.pos)
  have hp₁ (p : Nat.Primes) : (p : ℝ)⁻¹ < 1 :=
    (inv_lt_one₀ <| mod_cast p.prop.pos).mpr <| Nat.one_lt_cast.mpr <| p.prop.one_lt
  suffices Summable fun (pk : Nat.Primes × ℕ) ↦ (pk.1 : ℝ)⁻¹ ^ (pk.2 + 3 / 2 : ℝ) by
    refine (Summable.mul_left 2 this).of_nonneg_of_le (fun pk ↦ ?_) (fun pk ↦ F''_le pk.1 pk.2)
    simp only [F'', Function.comp_apply, F', F₀, Prod.map_fst, id_eq, Prod.map_snd, Nat.cast_pow]
    have := vonMangoldt_nonneg (n := (pk.1 : ℕ) ^ (pk.2 + 2))
    positivity
  /-
    hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
    hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
    ⊢ Summable fun pk => HPow.hPow (Inv.inv ↑↑pk.1) (HAdd.hAdd (↑pk.2) (3 / 2))
  -/
  conv => enter [1, pk]; rw [Real.rpow_add <| hp₀ pk.1, Real.rpow_natCast]
  /-
    hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
    hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
    ⊢ Summable fun pk => HMul.hMul (HPow.hPow (Inv.inv ↑↑pk.1) pk.2) (HPow.hPow (I …
  -/
  refine (summable_prod_of_nonneg (fun _ ↦ by positivity)).mpr ⟨(fun p ↦ ?_), ?_⟩
    /-
      case refine_1
      hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
      hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
      p : Nat.Primes
      ⊢ Summable fun y => HMul.hMul (HPow.hPow (Inv.inv ↑↑{ fst := p, snd := y }.1)  …
    -/
  · dsimp only -- otherwise the `exact` below times out
    /-
      case refine_1
      hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
      hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
      p : Nat.Primes
      ⊢ Summable fun y => HMul.hMul (HPow.hPow (Inv.inv ↑↑p) y) (HPow.hPow (Inv.inv  …
    -/
    exact Summable.mul_right _ <| summable_geometric_of_lt_one (hp₀ p).le (hp₁ p)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
      hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
      ⊢ Summable fun x => tsum fun y => HMul.hMul (HPow.hPow (Inv.inv ↑↑{ fst := x,  …
    -/
  · dsimp only
    /-
      case refine_2
      hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
      hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
      ⊢ Summable fun x => tsum fun y => HMul.hMul (HPow.hPow (Inv.inv ↑↑x) y) (HPow. …
    -/
    conv => enter [1, p]; rw [tsum_mul_right, tsum_geometric_of_lt_one (hp₀ p).le (hp₁ p)]
    refine (summable_rpow.mpr (by norm_num : -(3 / 2 : ℝ) < -1)).mul_left 2
      |>.of_nonneg_of_le (fun p ↦ ?_) (fun p ↦ ?_)
      /-
        case refine_2.refine_1
        hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
        hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
        p : Nat.Primes
        ⊢ LE.le 0 (HMul.hMul (Inv.inv (HSub.hSub 1 (Inv.inv ↑↑p))) (HPow.hPow (Inv.inv …
      -/
    · have := sub_pos.mpr (hp₁ p)
      /-
        case refine_2.refine_1
        hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
        hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
        p : Nat.Primes
        this : LT.lt 0 (HSub.hSub 1 (Inv.inv ↑↑p))
        ⊢ LE.le 0 (HMul.hMul (Inv.inv (HSub.hSub 1 (Inv.inv ↑↑p))) (HPow.hPow (Inv.inv …
      -/
      positivity
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
        hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
        p : Nat.Primes
        ⊢ LE.le (HMul.hMul (Inv.inv (HSub.hSub 1 (Inv.inv ↑↑p))) (HPow.hPow (Inv.inv ↑ …
      -/
    · rw [Real.inv_rpow p.val.cast_nonneg, Real.rpow_neg p.val.cast_nonneg]
      /-
        case refine_2.refine_2
        hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
        hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
        p : Nat.Primes
        ⊢ LE.le (HMul.hMul (Inv.inv (HSub.hSub 1 (Inv.inv ↑↑p))) (Inv.inv (HPow.hPow ( …
      -/
      gcongr
      rw [inv_le_comm₀ (sub_pos.mpr (hp₁ p)) zero_lt_two, le_sub_comm,
        show (1 : ℝ) - 2⁻¹ = 2⁻¹ by norm_num, inv_le_inv₀ (mod_cast p.prop.pos) zero_lt_two]
      /-
        case refine_2.refine_2.h
        hp₀ : ∀ (p : Nat.Primes), LT.lt 0 (Inv.inv ↑↑p)
        hp₁ : ∀ (p : Nat.Primes), LT.lt (Inv.inv ↑↑p) 1
        p : Nat.Primes
        ⊢ LE.le 2 ↑↑p
      -/
      exact Nat.ofNat_le_cast.mpr p.prop.two_le
      /-
        🎉 no goals
      -/


/-- The function `n ↦ Λ n / n`, restricted to non-primes in a residue class, is summable.
This is used to convert results on `ArithmeticFunction.vonMangoldt.residueClass` to results
on primes in an arithmetic progression. -/
lemma summable_residueClass_non_primes_div :
    Summable fun n : ℕ ↦ (if n.Prime then 0 else residueClass a n) / n := by
  have h₀ (n : ℕ) : 0 ≤ (if n.Prime then 0 else residueClass a n) / n := by
    have := residueClass_nonneg a n
    positivity
  have hleF₀ (n : ℕ) : (if n.Prime then 0 else residueClass a n) / n ≤ F₀ n := by
    refine div_le_div_of_nonneg_right ?_ n.cast_nonneg
    split_ifs; exacts [le_rfl, residueClass_le a n]
  /-
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    ⊢ Summable fun n => HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction.vonMang …
  -/
  refine Summable.of_nonneg_of_le h₀ hleF₀ ?_
  have hF₀ (p : Nat.Primes) : F₀ p.val = 0 := by
    simp only [p.prop, ↓reduceIte, zero_div, F₀]
  refine (summable_subtype_iff_indicator (s := {n | IsPrimePow n}).mp ?_).congr
      fun n ↦ Set.indicator_apply_eq_self.mpr fun (hn : ¬ IsPrimePow n) ↦ ?_
  /-
    case refine_1
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
    ⊢ Summable (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val)
  -/
  swap
  · simp +contextual only [div_eq_zero_iff, ite_eq_left_iff, vonMangoldt_eq_zero_iff, hn,
      not_false_eq_true, implies_true, Nat.cast_eq_zero, true_or, F₀]
  have hFF' :
      F₀ ∘ Subtype.val (p := fun n ↦ n ∈ {n | IsPrimePow n}) = F' ∘ ⇑prodNatEquiv.symm := by
    refine (Equiv.eq_comp_symm prodNatEquiv (F₀ ∘ Subtype.val) F').mpr ?_
    ext1 n
    simp only [Function.comp_apply, F']
    congr
  /-
    case refine_1
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
    hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
    ⊢ Summable (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val)
  -/
  rw [hFF']
  /-
    case refine_1
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
    hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
    ⊢ Summable (Function.comp ArithmeticFunction.vonMangoldt.F' ⇑Nat.Primes.prodNa …
  -/
  refine (Nat.Primes.prodNatEquiv.symm.summable_iff (f := F')).mpr ?_
  /-
    case refine_1
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
    hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
    ⊢ Summable ArithmeticFunction.vonMangoldt.F'
  -/
  have hF'₀ (p : Nat.Primes) : F' (p, 0) = 0 := by simp only [zero_add, pow_one, hF₀, F']
  have hF'₁ : F'' = F' ∘ (Prod.map _root_.id (· + 1)) := by
    ext1
    simp only [Function.comp_apply, Prod.map_fst, id_eq, Prod.map_snd, F'', F']
  /-
    case refine_1
    q : Nat
    a : ZMod q
    h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
    hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
    hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
    hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
    hF'₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F' { fst := p, s …
    hF'₁ : Eq ArithmeticFunction.vonMangoldt.F'' (Function.comp ArithmeticFunction …
    ⊢ Summable ArithmeticFunction.vonMangoldt.F'
  -/
  refine (Function.Injective.summable_iff ?_ fun u hu ↦ ?_).mp <| hF'₁ ▸ summable_F''
    /-
      case refine_1.refine_1
      q : Nat
      a : ZMod q
      h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
      hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
      hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
      hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
      hF'₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F' { fst := p, s …
      hF'₁ : Eq ArithmeticFunction.vonMangoldt.F'' (Function.comp ArithmeticFunction …
      ⊢ Function.Injective (Prod.map _root_.id fun x => HAdd.hAdd x 1)
    -/
  · exact Function.Injective.prodMap (fun ⦃a₁ a₂⦄ a ↦ a) <| add_left_injective 1
    /-
      🎉 no goals
    -/
  · simp only [Set.range_prod_map, Set.range_id, Set.mem_prod, Set.mem_univ, Set.mem_range,
      Nat.exists_add_one_eq, true_and, not_lt, nonpos_iff_eq_zero] at hu
    /-
      case refine_1.refine_2
      q : Nat
      a : ZMod q
      h₀ : ∀ (n : Nat), LE.le 0 (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction. …
      hleF₀ : ∀ (n : Nat), LE.le (HDiv.hDiv (ite (Nat.Prime n) 0 (ArithmeticFunction …
      hF₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F₀ ↑p) 0
      hFF' : Eq (Function.comp ArithmeticFunction.vonMangoldt.F₀ Subtype.val) (Funct …
      hF'₀ : ∀ (p : Nat.Primes), Eq (ArithmeticFunction.vonMangoldt.F' { fst := p, s …
      hF'₁ : Eq ArithmeticFunction.vonMangoldt.F'' (Function.comp ArithmeticFunction …
      u : Prod Nat.Primes Nat
      hu : Eq u.2 0
      ⊢ Eq (ArithmeticFunction.vonMangoldt.F' u) 0
    -/
    rw [← hF'₀ u.1, ← hu]
    /-
      🎉 no goals
    -/


/-- We can express `ArithmeticFunction.vonMangoldt.residueClass` as a linear combination
of twists of the von Mangoldt function by Dirichlet characters. -/
lemma residueClass_apply (ha : IsUnit a) (n : ℕ) :
    residueClass a n =
      (q.totient : ℂ)⁻¹ * ∑ χ : DirichletCharacter ℂ q, χ a⁻¹ * χ n * vonMangoldt n := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    n : Nat
    ⊢ Eq (↑(ArithmeticFunction.vonMangoldt.residueClass a n)) (HMul.hMul (Inv.inv  …
  -/
  rw [eq_inv_mul_iff_mul_eq₀ <| mod_cast (Nat.totient_pos.mpr q.pos_of_neZero).ne']
  simp +contextual only [residueClass, Set.indicator_apply, Set.mem_setOf_eq, apply_ite,
    ofReal_zero, mul_zero, ← Finset.sum_mul, sum_char_inv_mul_char_eq ℂ ha n, eq_comm (a := a),
    ite_mul, zero_mul, ↓reduceIte, ite_self]


/-- We can express `ArithmeticFunction.vonMangoldt.residueClass` as a linear combination
of twists of the von Mangoldt function by Dirichlet characters. -/
lemma residueClass_eq (ha : IsUnit a) :
    ↗(residueClass a) = (q.totient : ℂ)⁻¹ •
      ∑ χ : DirichletCharacter ℂ q, χ a⁻¹ • (fun n : ℕ ↦ χ n * vonMangoldt n) := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    ⊢ Eq (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n)) (HSMul.hSMu …
  -/
  ext1 n
  simpa only [Pi.smul_apply, Finset.sum_apply, smul_eq_mul, ← mul_assoc]
    using residueClass_apply ha n


/-- The L-series of the von Mangoldt function restricted to the residue class `a` mod `q`
with `a` invertible in `ZMod q` is a linear combination of logarithmic derivatives of
L-functions of the Dirichlet characters mod `q` (on `re s > 1`). -/
lemma LSeries_residueClass_eq (ha : IsUnit a) {s : ℂ} (hs : 1 < s.re) :
    LSeries ↗(residueClass a) s =
      -(q.totient : ℂ)⁻¹ * ∑ χ : DirichletCharacter ℂ q, χ a⁻¹ *
        (deriv (LFunction χ) s / LFunction χ s) := by
  simp only [deriv_LFunction_eq_deriv_LSeries _ hs, LFunction_eq_LSeries _ hs, neg_mul, ← mul_neg,
    ← Finset.sum_neg_distrib, ← neg_div, ← LSeries_twist_vonMangoldt_eq _ hs]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (LSeries (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n)) s) …
  -/
  rw [eq_inv_mul_iff_mul_eq₀ <| mod_cast (Nat.totient_pos.mpr q.pos_of_neZero).ne']
  simp_rw [← LSeries_smul,
    ← LSeries_sum <| fun χ _ ↦ (LSeriesSummable_twist_vonMangoldt χ hs).smul _]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (LSeries (HSMul.hSMul ↑q.totient fun n => ↑(ArithmeticFunction.vonMangold …
  -/
  refine LSeries_congr s fun {n} _ ↦ ?_
  simp only [Pi.smul_apply, residueClass_apply ha, smul_eq_mul, ← mul_assoc,
    mul_inv_cancel_of_invertible, one_mul, Finset.sum_apply, Pi.mul_apply]


open Classical in
/-- The auxiliary function used, e.g., with the Wiener-Ikehara Theorem to prove
Dirichlet's Theorem. On `re s > 1`, it agrees with the L-series of the von Mangoldt
function restricted to the residue class `a : ZMod q` minus the principal part
`(q.totient)⁻¹/(s-1)` of the pole at `s = 1`;
see `ArithmeticFunction.vonMangoldt.eqOn_LFunctionResidueClassAux`. -/
noncomputable
abbrev LFunctionResidueClassAux (s : ℂ) : ℂ :=
  (q.totient : ℂ)⁻¹ * (-deriv (LFunctionTrivChar₁ q) s / LFunctionTrivChar₁ q s -
    ∑ χ ∈ ({1}ᶜ : Finset (DirichletCharacter ℂ q)), χ a⁻¹ * deriv (LFunction χ) s / LFunction χ s)


/-- The auxiliary function is continuous away from the zeros of the L-functions of the Dirichlet
characters mod `q` (including at `s = 1`). -/
lemma continuousOn_LFunctionResidueClassAux' :
    ContinuousOn (LFunctionResidueClassAux a)
      {s | s = 1 ∨ ∀ χ : DirichletCharacter ℂ q, LFunction χ s ≠ 0} := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ⊢ ContinuousOn (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a) (se …
  -/
  rw [show LFunctionResidueClassAux a = fun s ↦ _ from rfl]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ⊢ ContinuousOn (fun s => ArithmeticFunction.vonMangoldt.LFunctionResidueClassA …
  -/
  simp only [LFunctionResidueClassAux, sub_eq_add_neg]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ⊢ ContinuousOn (fun s => HMul.hMul (Inv.inv ↑q.totient) (HAdd.hAdd (HDiv.hDiv  …
  -/
  refine continuousOn_const.mul <| ContinuousOn.add ?_ ?_
    /-
      case refine_1
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ⊢ ContinuousOn (fun s => HDiv.hDiv (Neg.neg (deriv (DirichletCharacter.LFuncti …
    -/
  · refine (continuousOn_neg_logDeriv_LFunctionTrivChar₁ q).mono fun s hs ↦ ?_
    /-
      case refine_1
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      s : Complex
      hs : Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Com …
      ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (Ne (DirichletCharacter.LFunction …
    -/
    have := LFunction_ne_zero_of_one_le_re (1 : DirichletCharacter ℂ q) (s := s)
    /-
      case refine_1
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      s : Complex
      hs : Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Com …
      this : Or (Ne 1 1) (Ne s 1) → LE.le 1 s.re → Ne (DirichletCharacter.LFunction  …
      ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (Ne (DirichletCharacter.LFunction …
    -/
    simp only [ne_eq, Set.mem_setOf_eq] at hs
    /-
      case refine_1
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      s : Complex
      hs : Or (Eq s 1) (∀ (χ : DirichletCharacter Complex q), Not (Eq (DirichletChar …
      this : Or (Ne 1 1) (Ne s 1) → LE.le 1 s.re → Ne (DirichletCharacter.LFunction  …
      ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (Ne (DirichletCharacter.LFunction …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ⊢ ContinuousOn (fun s => Neg.neg ((HasCompl.compl (Singleton.singleton 1)).sum …
    -/
  · simp only [← Finset.sum_neg_distrib, mul_div_assoc, ← mul_neg, ← neg_div]
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ⊢ ContinuousOn (fun s => (HasCompl.compl (Singleton.singleton 1)).sum fun x => …
    -/
    refine continuousOn_finset_sum _ fun χ hχ ↦ continuousOn_const.mul ?_
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      χ : DirichletCharacter Complex q
      hχ : Membership.mem (HasCompl.compl (Singleton.singleton 1)) χ
      ⊢ ContinuousOn (fun s => HDiv.hDiv (Neg.neg (deriv (DirichletCharacter.LFuncti …
    -/
    replace hχ : χ ≠ 1 := by simpa only [ne_eq, Finset.mem_compl, Finset.mem_singleton] using hχ
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      χ : DirichletCharacter Complex q
      hχ : Ne χ 1
      ⊢ ContinuousOn (fun s => HDiv.hDiv (Neg.neg (deriv (DirichletCharacter.LFuncti …
    -/
    refine (continuousOn_neg_logDeriv_LFunction_of_nontriv hχ).mono fun s hs ↦ ?_
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      χ : DirichletCharacter Complex q
      hχ : Ne χ 1
      s : Complex
      hs : Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Com …
      ⊢ Membership.mem (setOf fun s => Ne (DirichletCharacter.LFunction χ s) 0) s
    -/
    simp only [ne_eq, Set.mem_setOf_eq] at hs
    /-
      case refine_2
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      χ : DirichletCharacter Complex q
      hχ : Ne χ 1
      s : Complex
      hs : Or (Eq s 1) (∀ (χ : DirichletCharacter Complex q), Not (Eq (DirichletChar …
      ⊢ Membership.mem (setOf fun s => Ne (DirichletCharacter.LFunction χ s) 0) s
    -/
    rcases hs with rfl | hs
    · simp only [ne_eq, Set.mem_setOf_eq, one_re, le_refl,
        LFunction_ne_zero_of_one_le_re χ (.inl hχ), not_false_eq_true]
      /-
        case refine_2.inr
        q : Nat
        a : ZMod q
        inst✝ : NeZero q
        χ : DirichletCharacter Complex q
        hχ : Ne χ 1
        s : Complex
        hs : ∀ (χ : DirichletCharacter Complex q), Not (Eq (DirichletCharacter.LFuncti …
        ⊢ Membership.mem (setOf fun s => Ne (DirichletCharacter.LFunction χ s) 0) s
      -/
    · exact hs χ
      /-
        🎉 no goals
      -/


/-- The L-series of the von Mangoldt function restricted to the prime residue class `a` mod `q`
is continuous on `re s ≥ 1` except for a simple pole at `s = 1` with residue `(q.totient)⁻¹`.
The statement as given here in terms of `ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux`
is equivalent. -/
lemma continuousOn_LFunctionResidueClassAux :
    ContinuousOn (LFunctionResidueClassAux a) {s | 1 ≤ s.re} := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ⊢ ContinuousOn (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a) (se …
  -/
  refine (continuousOn_LFunctionResidueClassAux' a).mono fun s hs ↦ ?_
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    s : Complex
    hs : Membership.mem (setOf fun s => LE.le 1 s.re) s
    ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Comple …
  -/
  rcases eq_or_ne s 1 with rfl | hs₁
    /-
      case inl
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      hs : Membership.mem (setOf fun s => LE.le 1 s.re) 1
      ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Comple …
    -/
  · simp only [ne_eq, Set.mem_setOf_eq, true_or]
    /-
      🎉 no goals
    -/
    /-
      case inr
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      s : Complex
      hs : Membership.mem (setOf fun s => LE.le 1 s.re) s
      hs₁ : Ne s 1
      ⊢ Membership.mem (setOf fun s => Or (Eq s 1) (∀ (χ : DirichletCharacter Comple …
    -/
  · simp only [ne_eq, Set.mem_setOf_eq, hs₁, false_or]
    /-
      case inr
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      s : Complex
      hs : Membership.mem (setOf fun s => LE.le 1 s.re) s
      hs₁ : Ne s 1
      ⊢ ∀ (χ : DirichletCharacter Complex q), Not (Eq (DirichletCharacter.LFunction  …
    -/
    exact fun χ ↦ LFunction_ne_zero_of_one_le_re χ (.inr hs₁) <| Set.mem_setOf.mp hs
    /-
      🎉 no goals
    -/


/-- The auxiliary function agrees on `re s > 1` with the L-series of the von Mangoldt function
restricted to the residue class `a : ZMod q` minus the principal part `(q.totient)⁻¹/(s-1)`
of its pole at `s = 1`. -/
lemma eqOn_LFunctionResidueClassAux (ha : IsUnit a) :
    Set.EqOn (LFunctionResidueClassAux a)
      (fun s ↦ L ↗(residueClass a) s - (q.totient : ℂ)⁻¹ / (s - 1))
      {s | 1 < s.re} := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    ⊢ Set.EqOn (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a) (fun s  …
  -/
  intro s hs
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : Membership.mem (setOf fun s => LT.lt 1 s.re) s
    ⊢ Eq (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a s) ((fun s =>  …
  -/
  replace hs := Set.mem_setOf.mp hs
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a s) ((fun s =>  …
  -/
  simp only [LSeries_residueClass_eq ha hs, LFunctionResidueClassAux]
  rw [neg_div, ← neg_add', mul_neg, ← neg_mul, div_eq_mul_one_div (q.totient : ℂ)⁻¹,
    sub_eq_add_neg, ← neg_mul, ← mul_add]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (HMul.hMul (Neg.neg (Inv.inv ↑q.totient)) (HAdd.hAdd (HDiv.hDiv (deriv (D …
  -/
  congrm (_ * ?_)
  -- this should be easier, but `IsUnit.inv ha` does not work here
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    s : Complex
    hs : LT.lt 1 s.re
    ⊢ Eq (HAdd.hAdd (HDiv.hDiv (deriv (DirichletCharacter.LFunctionTrivChar₁ q) s) …
  -/
  have ha' : IsUnit a⁻¹ := isUnit_of_dvd_one ⟨a, (ZMod.inv_mul_of_unit a ha).symm⟩
  classical -- for `Fintype.sum_eq_add_sum_compl`
  rw [Fintype.sum_eq_add_sum_compl 1, MulChar.one_apply ha', one_mul, add_right_comm]
  simp only [mul_div_assoc]
  congrm (?_ + _)
  have hs₁ : s ≠ 1 := fun h ↦ ((h ▸ hs).trans_eq one_re).false
  rw [deriv_LFunctionTrivChar₁_apply_of_ne_one _ hs₁, LFunctionTrivChar₁,
    Function.update_of_ne hs₁, LFunctionTrivChar, add_div,
    mul_div_mul_left _ _ (sub_ne_zero_of_ne hs₁)]
  conv_lhs => enter [2, 1]; rw [← mul_one (LFunction ..)]
  rw [mul_comm _ 1, mul_div_mul_right _ _ <| LFunction_ne_zero_of_one_le_re 1 (.inr hs₁) hs.le]


/-- The auxiliary function takes real values for real arguments `x > 1`. -/
lemma LFunctionResidueClassAux_real (ha : IsUnit a) {x : ℝ} (hx : 1 < x) :
    LFunctionResidueClassAux a x = (LFunctionResidueClassAux a x).re := by
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    x : Real
    hx : LT.lt 1 x
    ⊢ Eq (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a ↑x) ↑(Arithmet …
  -/
  rw [eqOn_LFunctionResidueClassAux ha hx]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    x : Real
    hx : LT.lt 1 x
    ⊢ Eq ((fun s => HSub.hSub (LSeries (fun n => ↑(ArithmeticFunction.vonMangoldt. …
  -/
  simp only [sub_re, ofReal_sub]
  /-
    q : Nat
    a : ZMod q
    inst✝ : NeZero q
    ha : IsUnit a
    x : Real
    hx : LT.lt 1 x
    ⊢ Eq (HSub.hSub (LSeries (fun n => ↑(ArithmeticFunction.vonMangoldt.residueCla …
  -/
  congr 1
  · rw [LSeries, re_tsum <| LSeriesSummable_of_abscissaOfAbsConv_lt_re <|
      (abscissaOfAbsConv_residueClass_le_one a).trans_lt <| by norm_cast]
    /-
      case e_a
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ha : IsUnit a
      x : Real
      hx : LT.lt 1 x
      ⊢ Eq (tsum fun n => LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.re …
    -/
    push_cast
    /-
      case e_a
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ha : IsUnit a
      x : Real
      hx : LT.lt 1 x
      ⊢ Eq (tsum fun n => LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.re …
    -/
    refine tsum_congr fun n ↦ ?_
    /-
      case e_a
      q : Nat
      a : ZMod q
      inst✝ : NeZero q
      ha : IsUnit a
      x : Real
      hx : LT.lt 1 x
      n : Nat
      ⊢ Eq (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n …
    -/
    rcases eq_or_ne n 0 with rfl | hn
      /-
        case e_a.inl
        q : Nat
        a : ZMod q
        inst✝ : NeZero q
        ha : IsUnit a
        x : Real
        hx : LT.lt 1 x
        ⊢ Eq (LSeries.term (fun n => ↑(ArithmeticFunction.vonMangoldt.residueClass a n …
      -/
    · simp only [term_zero, zero_re, ofReal_zero]
      /-
        🎉 no goals
      -/
    · simp only [term_of_ne_zero hn, ← ofReal_natCast n, ← ofReal_cpow n.cast_nonneg, ← ofReal_div,
        ofReal_re]
  · rw [show (q.totient : ℂ) = (q.totient : ℝ) from rfl, ← ofReal_one, ← ofReal_sub, ← ofReal_inv,
      ← ofReal_div, ofReal_re]


/-- As `x` approaches `1` from the right along the real axis, the L-series of
`ArithmeticFunction.vonMangoldt.residueClass` is bounded below by `(q.totient)⁻¹/(x-1) - C`. -/
lemma LSeries_residueClass_lower_bound (ha : IsUnit a) :
    ∃ C : ℝ, ∀ {x : ℝ} (_ : x ∈ Set.Ioc 1 2),
      (q.totient : ℝ)⁻¹ / (x - 1) - C ≤ ∑' n, residueClass a n / (n : ℝ) ^ x := by
  have H {x : ℝ} (hx : 1 < x) :
      ∑' n, residueClass a n / (n : ℝ) ^ x =
        (LFunctionResidueClassAux a x).re + (q.totient : ℝ)⁻¹ / (x - 1) := by
    refine ofReal_injective ?_
    simp only [ofReal_tsum, ofReal_div, ofReal_cpow (Nat.cast_nonneg _), ofReal_natCast,
      ofReal_add, ofReal_inv, ofReal_sub, ofReal_one]
    simp_rw [← LFunctionResidueClassAux_real ha hx,
      eqOn_LFunctionResidueClassAux ha <| Set.mem_setOf.mpr (ofReal_re x ▸ hx), sub_add_cancel,
      LSeries, term]
    refine tsum_congr fun n ↦ ?_
    split_ifs with hn
    · simp only [hn, residueClass_apply_zero, ofReal_zero, zero_div]
    · rfl
  have : ContinuousOn (fun x : ℝ ↦ (LFunctionResidueClassAux a x).re) (Set.Icc 1 2) :=
    continuous_re.continuousOn.comp (t := Set.univ) (continuousOn_LFunctionResidueClassAux a)
      (fun ⦃x⦄ a ↦ trivial) |>.comp continuous_ofReal.continuousOn fun x hx ↦ by
        simpa only [Set.mem_setOf_eq, ofReal_re] using hx.1
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : ∀ {x : Real}, LT.lt 1 x → Eq (tsum fun n => HDiv.hDiv (ArithmeticFunction. …
    this : ContinuousOn (fun x => (ArithmeticFunction.vonMangoldt.LFunctionResidue …
    ⊢ Exists fun C => ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.h …
  -/
  obtain ⟨C, hC⟩ := bddBelow_def.mp <| IsCompact.bddBelow_image isCompact_Icc this
  replace hC {x : ℝ} (hx : x ∈ Set.Icc 1 2) : C ≤ (LFunctionResidueClassAux a x).re :=
    hC (LFunctionResidueClassAux a x).re <|
      Set.mem_image_of_mem (fun x : ℝ ↦ (LFunctionResidueClassAux a x).re) hx
  /-
    case intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : ∀ {x : Real}, LT.lt 1 x → Eq (tsum fun n => HDiv.hDiv (ArithmeticFunction. …
    this : ContinuousOn (fun x => (ArithmeticFunction.vonMangoldt.LFunctionResidue …
    C : Real
    hC : ∀ {x : Real}, Membership.mem (Set.Icc 1 2) x → LE.le C (ArithmeticFunctio …
    ⊢ Exists fun C => ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.h …
  -/
  refine ⟨-C, fun {x} hx ↦ ?_⟩
  /-
    case intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : ∀ {x : Real}, LT.lt 1 x → Eq (tsum fun n => HDiv.hDiv (ArithmeticFunction. …
    this : ContinuousOn (fun x => (ArithmeticFunction.vonMangoldt.LFunctionResidue …
    C : Real
    hC : ∀ {x : Real}, Membership.mem (Set.Icc 1 2) x → LE.le C (ArithmeticFunctio …
    x : Real
    hx : Membership.mem (Set.Ioc 1 2) x
    ⊢ LE.le (HSub.hSub (HDiv.hDiv (Inv.inv ↑q.totient) (HSub.hSub x 1)) (Neg.neg C …
  -/
  rw [H hx.1, add_comm, sub_neg_eq_add, add_le_add_iff_left]
  /-
    case intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : ∀ {x : Real}, LT.lt 1 x → Eq (tsum fun n => HDiv.hDiv (ArithmeticFunction. …
    this : ContinuousOn (fun x => (ArithmeticFunction.vonMangoldt.LFunctionResidue …
    C : Real
    hC : ∀ {x : Real}, Membership.mem (Set.Icc 1 2) x → LE.le C (ArithmeticFunctio …
    x : Real
    hx : Membership.mem (Set.Ioc 1 2) x
    ⊢ LE.le C (ArithmeticFunction.vonMangoldt.LFunctionResidueClassAux a ↑x).re
  -/
  exact hC <| Set.mem_Icc_of_Ioc hx
  /-
    🎉 no goals
  -/


open vonMangoldt Filter Topology in
/-- The function `n ↦ Λ n / n` restricted to primes in an invertible residue class
is not summable. This then implies that there must be infinitely many such primes. -/
lemma not_summable_residueClass_prime_div (ha : IsUnit a) :
    ¬ Summable fun n : ℕ ↦ (if n.Prime then residueClass a n else 0) / n := by
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    ⊢ Not (Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonM …
  -/
  intro H
  have key : Summable fun n : ℕ ↦ residueClass a n / n := by
    convert (summable_residueClass_non_primes_div a).add H using 2 with n
    simp only [← add_div, ite_add_ite, zero_add, add_zero, ite_self]
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
    key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
    ⊢ False
  -/
  let C := ∑' n, residueClass a n / n
  have H₁ {x : ℝ} (hx : 1 < x) : ∑' n, residueClass a n / (n : ℝ) ^ x ≤ C := by
    refine tsum_le_tsum (fun n ↦ ?_) ?_ key
    · rcases n.eq_zero_or_pos with rfl | hn
      · simp only [Nat.cast_zero, Real.zero_rpow (zero_lt_one.trans hx).ne', div_zero, le_refl]
      · refine div_le_div_of_nonneg_left (residueClass_nonneg a _) (mod_cast hn) ?_
        conv_lhs => rw [← Real.rpow_one n]
        exact Real.rpow_le_rpow_of_exponent_le (by norm_cast) hx.le
    · exact summable_real_of_abscissaOfAbsConv_lt <|
        (abscissaOfAbsConv_residueClass_le_one a).trans_lt <| mod_cast hx
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
    key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
    C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
    H₁ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunct …
    ⊢ False
  -/
  obtain ⟨C', hC'⟩ := LSeries_residueClass_lower_bound ha
  have H₁ {x} (hx : x ∈ Set.Ioc 1 2) : (q.totient : ℝ)⁻¹ ≤ (C + C') * (x - 1) :=
    (div_le_iff₀ <| sub_pos.mpr hx.1).mp <|
      sub_le_iff_le_add.mp <| (hC' hx).trans (H₁ hx.1)
  /-
    case intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
    key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
    C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
    H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
    C' : Real
    hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
    H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
    ⊢ False
  -/
  have hq : 0 < (q.totient : ℝ)⁻¹ := inv_pos.mpr (mod_cast q.totient.pos_of_neZero)
  /-
    case intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
    key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
    C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
    H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
    C' : Real
    hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
    H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
    hq : LT.lt 0 (Inv.inv ↑q.totient)
    ⊢ False
  -/
  rcases le_or_lt (C + C') 0 with h₀ | h₀
    /-
      case intro.inl
      q : Nat
      inst✝ : NeZero q
      a : ZMod q
      ha : IsUnit a
      H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
      key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
      C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
      H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
      C' : Real
      hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
      H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
      hq : LT.lt 0 (Inv.inv ↑q.totient)
      h₀ : LE.le (HAdd.hAdd C C') 0
      ⊢ False
    -/
  · have := hq.trans_le (H₁ (Set.right_mem_Ioc.mpr one_lt_two))
    /-
      case intro.inl
      q : Nat
      inst✝ : NeZero q
      a : ZMod q
      ha : IsUnit a
      H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
      key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
      C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
      H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
      C' : Real
      hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
      H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
      hq : LT.lt 0 (Inv.inv ↑q.totient)
      h₀ : LE.le (HAdd.hAdd C C') 0
      this : LT.lt 0 (HMul.hMul (HAdd.hAdd C C') (HSub.hSub 2 1))
      ⊢ False
    -/
    rw [show (2 : ℝ) - 1 = 1 by norm_num, mul_one] at this
    /-
      case intro.inl
      q : Nat
      inst✝ : NeZero q
      a : ZMod q
      ha : IsUnit a
      H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
      key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
      C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
      H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
      C' : Real
      hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
      H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
      hq : LT.lt 0 (Inv.inv ↑q.totient)
      h₀ : LE.le (HAdd.hAdd C C') 0
      this : LT.lt 0 (HAdd.hAdd C C')
      ⊢ False
    -/
    exact (this.trans_le h₀).false
    /-
      🎉 no goals
    -/
  · obtain ⟨ξ, hξ₁, hξ₂⟩ : ∃ ξ ∈ Set.Ioc 1 2, (C + C') * (ξ - 1) < (q.totient : ℝ)⁻¹ := by
      refine ⟨min (1 + (q.totient : ℝ)⁻¹ / (C + C') / 2) 2, ⟨?_, min_le_right ..⟩, ?_⟩
      · simpa only [lt_inf_iff, lt_add_iff_pos_right, Nat.ofNat_pos, div_pos_iff_of_pos_right,
          Nat.one_lt_ofNat, and_true] using div_pos hq h₀
      · rw [← min_sub_sub_right, add_sub_cancel_left, ← lt_div_iff₀' h₀]
        exact (min_le_left ..).trans_lt <| div_lt_self (div_pos hq h₀) one_lt_two
    /-
      case intro.inr.intro.intro
      q : Nat
      inst✝ : NeZero q
      a : ZMod q
      ha : IsUnit a
      H : Summable fun n => HDiv.hDiv (ite (Nat.Prime n) (ArithmeticFunction.vonMang …
      key : Summable fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueClass …
      C : Real := tsum fun n => HDiv.hDiv (ArithmeticFunction.vonMangoldt.residueCla …
      H₁✝ : ∀ {x : Real}, LT.lt 1 x → LE.le (tsum fun n => HDiv.hDiv (ArithmeticFunc …
      C' : Real
      hC' : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (HSub.hSub (HDiv.hD …
      H₁ : ∀ {x : Real}, Membership.mem (Set.Ioc 1 2) x → LE.le (Inv.inv ↑q.totient) …
      hq : LT.lt 0 (Inv.inv ↑q.totient)
      h₀ : LT.lt 0 (HAdd.hAdd C C')
      ξ : Real
      hξ₁ : Membership.mem (Set.Ioc 1 2) ξ
      hξ₂ : LT.lt (HMul.hMul (HAdd.hAdd C C') (HSub.hSub ξ 1)) (Inv.inv ↑q.totient)
      ⊢ False
    -/
    exact ((H₁ hξ₁).trans_lt hξ₂).false
    /-
      🎉 no goals
    -/


/-- **Dirichlet's Theorem** on primes in arithmetic progression: if `q` is a positive
integer and `a : ZMod q` is a unit, then there are infintely many prime numbers `p`
such that `(p : ZMod q) = a`. -/
theorem setOf_prime_and_eq_mod_infinite (ha : IsUnit a) :
    {p : ℕ | p.Prime ∧ (p : ZMod q) = a}.Infinite := by
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    ⊢ (setOf fun p => And (Nat.Prime p) (Eq (↑p) a)).Infinite
  -/
  by_contra H
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    H : Not (setOf fun p => And (Nat.Prime p) (Eq (↑p) a)).Infinite
    ⊢ False
  -/
  rw [Set.not_infinite] at H
  exact not_summable_residueClass_prime_div ha <|
    summable_of_finite_support <| support_residueClass_prime_div a ▸ H


/-- **Dirichlet's Theorem** on primes in arithmetic progression: if `q` is a positive
integer and `a : ZMod q` is a unit, then there are infintely many prime numbers `p`
such that `(p : ZMod q) = a`. -/
theorem forall_exists_prime_gt_and_eq_mod (ha : IsUnit a) (n : ℕ) :
    ∃ p > n, p.Prime ∧ (p : ZMod q) = a := by
  /-
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    n : Nat
    ⊢ Exists fun p => And (GT.gt p n) (And (Nat.Prime p) (Eq (↑p) a))
  -/
  obtain ⟨p, hp₁, hp₂⟩ := Set.infinite_iff_exists_gt.mp (setOf_prime_and_eq_mod_infinite ha) n
  /-
    case intro.intro
    q : Nat
    inst✝ : NeZero q
    a : ZMod q
    ha : IsUnit a
    n p : Nat
    hp₁ : Membership.mem (setOf fun p => And (Nat.Prime p) (Eq (↑p) a)) p
    hp₂ : LT.lt n p
    ⊢ Exists fun p => And (GT.gt p n) (And (Nat.Prime p) (Eq (↑p) a))
  -/
  exact ⟨p, hp₂.gt, Set.mem_setOf.mp hp₁⟩
  /-
    🎉 no goals
  -/


