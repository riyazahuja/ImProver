/-- In a seminormed group `A`, given `n : ℕ` and `δ : ℝ`, `approxOrderOf A n δ` is the set of
elements within a distance `δ` of a point of order `n`. -/
@[to_additive "In a seminormed additive group `A`, given `n : ℕ` and `δ : ℝ`,
`approxAddOrderOf A n δ` is the set of elements within a distance `δ` of a point of order `n`."]
def approxOrderOf (A : Type*) [SeminormedGroup A] (n : ℕ) (δ : ℝ) : Set A :=
  thickening δ {y | orderOf y = n}


@[to_additive mem_approx_add_orderOf_iff]
theorem mem_approxOrderOf_iff {A : Type*} [SeminormedGroup A] {n : ℕ} {δ : ℝ} {a : A} :
    a ∈ approxOrderOf A n δ ↔ ∃ b : A, orderOf b = n ∧ a ∈ ball b δ := by
  /-
    A : Type u_1
    inst✝ : SeminormedGroup A
    n : Nat
    δ : Real
    a : A
    ⊢ Iff (Membership.mem (approxOrderOf A n δ) a) (Exists fun b => And (Eq (order …
  -/
  simp only [approxOrderOf, thickening_eq_biUnion_ball, mem_iUnion₂, mem_setOf_eq, exists_prop]
  /-
    🎉 no goals
  -/


/-- In a seminormed group `A`, given a sequence of distances `δ₁, δ₂, ...`, `wellApproximable A δ`
is the limsup as `n → ∞` of the sets `approxOrderOf A n δₙ`. Thus, it is the set of points that
lie in infinitely many of the sets `approxOrderOf A n δₙ`. -/
@[to_additive addWellApproximable "In a seminormed additive group `A`, given a sequence of
distances `δ₁, δ₂, ...`, `addWellApproximable A δ` is the limsup as `n → ∞` of the sets
`approxAddOrderOf A n δₙ`. Thus, it is the set of points that lie in infinitely many of the sets
`approxAddOrderOf A n δₙ`."]
def wellApproximable (A : Type*) [SeminormedGroup A] (δ : ℕ → ℝ) : Set A :=
  blimsup (fun n => approxOrderOf A n (δ n)) atTop fun n => 0 < n


@[to_additive mem_add_wellApproximable_iff]
theorem mem_wellApproximable_iff {A : Type*} [SeminormedGroup A] {δ : ℕ → ℝ} {a : A} :
    a ∈ wellApproximable A δ ↔
      a ∈ blimsup (fun n => approxOrderOf A n (δ n)) atTop fun n => 0 < n :=
  Iff.rfl


@[to_additive]
theorem image_pow_subset_of_coprime (hm : 0 < m) (hmn : n.Coprime m) :
    (fun (y : A) => y ^ m) '' approxOrderOf A n δ ⊆ approxOrderOf A n (m * δ) := by
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m n : Nat
    δ : Real
    hm : LT.lt 0 m
    hmn : n.Coprime m
    ⊢ HasSubset.Subset (Set.image (fun y => HPow.hPow y m) (approxOrderOf A n δ))  …
  -/
  rintro - ⟨a, ha, rfl⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m n : Nat
    δ : Real
    hm : LT.lt 0 m
    hmn : n.Coprime m
    a : A
    ha : Membership.mem (approxOrderOf A n δ) a
    ⊢ Membership.mem (approxOrderOf A n (HMul.hMul (↑m) δ)) ((fun y => HPow.hPow y …
  -/
  obtain ⟨b, hb, hab⟩ := mem_approxOrderOf_iff.mp ha
  replace hb : b ^ m ∈ {u : A | orderOf u = n} := by
    rw [← hb] at hmn ⊢; exact hmn.orderOf_pow
  /-
    case intro.intro.intro.intro
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m n : Nat
    δ : Real
    hm : LT.lt 0 m
    hmn : n.Coprime m
    a : A
    ha : Membership.mem (approxOrderOf A n δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun u => Eq (orderOf u) n) (HPow.hPow b m)
    ⊢ Membership.mem (approxOrderOf A n (HMul.hMul (↑m) δ)) ((fun y => HPow.hPow y …
  -/
  apply ball_subset_thickening hb ((m : ℝ) • δ)
  /-
    case intro.intro.intro.intro.a
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m n : Nat
    δ : Real
    hm : LT.lt 0 m
    hmn : n.Coprime m
    a : A
    ha : Membership.mem (approxOrderOf A n δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun u => Eq (orderOf u) n) (HPow.hPow b m)
    ⊢ Membership.mem (Metric.ball (HPow.hPow b m) (HSMul.hSMul (↑m) δ)) ((fun y => …
  -/
  convert pow_mem_ball hm hab using 1
  /-
    case h.e'_4
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m n : Nat
    δ : Real
    hm : LT.lt 0 m
    hmn : n.Coprime m
    a : A
    ha : Membership.mem (approxOrderOf A n δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun u => Eq (orderOf u) n) (HPow.hPow b m)
    ⊢ Eq (Metric.ball (HPow.hPow b m) (HSMul.hSMul (↑m) δ)) (Metric.ball (HPow.hPo …
  -/
  simp only [nsmul_eq_mul, Algebra.id.smul_eq_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem image_pow_subset (n : ℕ) (hm : 0 < m) :
    (fun (y : A) => y ^ m) '' approxOrderOf A (n * m) δ ⊆ approxOrderOf A n (m * δ) := by
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m : Nat
    δ : Real
    n : Nat
    hm : LT.lt 0 m
    ⊢ HasSubset.Subset (Set.image (fun y => HPow.hPow y m) (approxOrderOf A (HMul. …
  -/
  rintro - ⟨a, ha, rfl⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m : Nat
    δ : Real
    n : Nat
    hm : LT.lt 0 m
    a : A
    ha : Membership.mem (approxOrderOf A (HMul.hMul n m) δ) a
    ⊢ Membership.mem (approxOrderOf A n (HMul.hMul (↑m) δ)) ((fun y => HPow.hPow y …
  -/
  obtain ⟨b, hb : orderOf b = n * m, hab : a ∈ ball b δ⟩ := mem_approxOrderOf_iff.mp ha
  replace hb : b ^ m ∈ {y : A | orderOf y = n} := by
    rw [mem_setOf_eq, orderOf_pow' b hm.ne', hb, Nat.gcd_mul_left_left, n.mul_div_cancel hm]
  /-
    case intro.intro.intro.intro
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m : Nat
    δ : Real
    n : Nat
    hm : LT.lt 0 m
    a : A
    ha : Membership.mem (approxOrderOf A (HMul.hMul n m) δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun y => Eq (orderOf y) n) (HPow.hPow b m)
    ⊢ Membership.mem (approxOrderOf A n (HMul.hMul (↑m) δ)) ((fun y => HPow.hPow y …
  -/
  apply ball_subset_thickening hb (m * δ)
  /-
    case intro.intro.intro.intro.a
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m : Nat
    δ : Real
    n : Nat
    hm : LT.lt 0 m
    a : A
    ha : Membership.mem (approxOrderOf A (HMul.hMul n m) δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun y => Eq (orderOf y) n) (HPow.hPow b m)
    ⊢ Membership.mem (Metric.ball (HPow.hPow b m) (HMul.hMul (↑m) δ)) ((fun y => H …
  -/
  convert pow_mem_ball hm hab using 1
  /-
    case h.e'_4
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    m : Nat
    δ : Real
    n : Nat
    hm : LT.lt 0 m
    a : A
    ha : Membership.mem (approxOrderOf A (HMul.hMul n m) δ) a
    b : A
    hab : Membership.mem (Metric.ball b δ) a
    hb : Membership.mem (setOf fun y => Eq (orderOf y) n) (HPow.hPow b m)
    ⊢ Eq (Metric.ball (HPow.hPow b m) (HMul.hMul (↑m) δ)) (Metric.ball (HPow.hPow  …
  -/
  simp only [nsmul_eq_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_subset_of_coprime (han : (orderOf a).Coprime n) :
    a • approxOrderOf A n δ ⊆ approxOrderOf A (orderOf a * n) δ := by
  simp_rw [approxOrderOf, thickening_eq_biUnion_ball, ← image_smul, image_iUnion₂, image_smul,
    smul_ball'', smul_eq_mul, mem_setOf_eq]
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    han : (orderOf a).Coprime n
    ⊢ HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun x => Metric.ball (HMul. …
  -/
  refine iUnion₂_subset_iff.mpr fun b hb c hc => ?_
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    han : (orderOf a).Coprime n
    b : A
    hb : Eq (orderOf b) n
    c : A
    hc : Membership.mem (Metric.ball (HMul.hMul a b) δ) c
    ⊢ Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Metric.ball x δ) c
  -/
  simp only [mem_iUnion, exists_prop]
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    han : (orderOf a).Coprime n
    b : A
    hb : Eq (orderOf b) n
    c : A
    hc : Membership.mem (Metric.ball (HMul.hMul a b) δ) c
    ⊢ Exists fun i => And (Eq (orderOf i) (HMul.hMul (orderOf a) n)) (Membership.m …
  -/
  refine ⟨a * b, ?_, hc⟩
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    han : (orderOf a).Coprime n
    b : A
    hb : Eq (orderOf b) n
    c : A
    hc : Membership.mem (Metric.ball (HMul.hMul a b) δ) c
    ⊢ Eq (orderOf (HMul.hMul a b)) (HMul.hMul (orderOf a) n)
  -/
  rw [← hb] at han ⊢
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    b : A
    han : (orderOf a).Coprime (orderOf b)
    hb : Eq (orderOf b) n
    c : A
    hc : Membership.mem (Metric.ball (HMul.hMul a b) δ) c
    ⊢ Eq (orderOf (HMul.hMul a b)) (HMul.hMul (orderOf a) (orderOf b))
  -/
  exact (Commute.all a b).orderOf_mul_eq_mul_orderOf_of_coprime han
  /-
    🎉 no goals
  -/


@[to_additive vadd_eq_of_mul_dvd]
theorem smul_eq_of_mul_dvd (hn : 0 < n) (han : orderOf a ^ 2 ∣ n) :
    a • approxOrderOf A n δ = approxOrderOf A n δ := by
  simp_rw [approxOrderOf, thickening_eq_biUnion_ball, ← image_smul, image_iUnion₂, image_smul,
    smul_ball'', smul_eq_mul, mem_setOf_eq]
  replace han : ∀ {b : A}, orderOf b = n → orderOf (a * b) = n := by
    intro b hb
    rw [← hb] at han hn
    rw [sq] at han
    rwa [(Commute.all a b).orderOf_mul_eq_right_of_forall_prime_mul_dvd (orderOf_pos_iff.mp hn)
      fun p _ hp' => dvd_trans (mul_dvd_mul_right hp' <| orderOf a) han]
  /-
    A : Type u_1
    inst✝ : SeminormedCommGroup A
    a : A
    n : Nat
    δ : Real
    hn : LT.lt 0 n
    han : ∀ {b : A}, Eq (orderOf b) n → Eq (orderOf (HMul.hMul a b)) n
    ⊢ Eq (Set.iUnion fun i => Set.iUnion fun x => Metric.ball (HMul.hMul a i) δ) ( …
  -/
  let f : {b : A | orderOf b = n} → {b : A | orderOf b = n} := fun b => ⟨a * b, han b.property⟩
  have hf : Surjective f := by
    rintro ⟨b, hb⟩
    refine ⟨⟨a⁻¹ * b, ?_⟩, ?_⟩
    · rw [mem_setOf_eq, ← orderOf_inv, mul_inv_rev, inv_inv, mul_comm]
      apply han
      simpa
    · simp only [f, Subtype.mk_eq_mk, Subtype.coe_mk, mul_inv_cancel_left]
  simpa only [mem_setOf_eq, Subtype.coe_mk, iUnion_coe_set] using
    hf.iUnion_comp fun b => ball (b : A) δ


theorem mem_approxAddOrderOf_iff {δ : ℝ} {x : UnitAddCircle} {n : ℕ} (hn : 0 < n) :
    x ∈ approxAddOrderOf UnitAddCircle n δ ↔ ∃ m < n, gcd m n = 1 ∧ ‖x - ↑((m : ℝ) / n)‖ < δ := by
  simp only [mem_approx_add_orderOf_iff, mem_setOf_eq, ball, exists_prop, dist_eq_norm,
    AddCircle.addOrderOf_eq_pos_iff hn, mul_one]
  /-
    δ : Real
    x : UnitAddCircle
    n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (Exists fun b => And (Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) …
  -/
  constructor
    /-
      case mp
      δ : Real
      x : UnitAddCircle
      n : Nat
      hn : LT.lt 0 n
      ⊢ (Exists fun b => And (Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1)  …
    -/
  · rintro ⟨y, ⟨m, hm₁, hm₂, rfl⟩, hx⟩; exact ⟨m, hm₁, hm₂, hx⟩
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case mpr
      δ : Real
      x : UnitAddCircle
      n : Nat
      hn : LT.lt 0 n
      ⊢ (Exists fun m => And (LT.lt m n) (And (Eq (GCDMonoid.gcd m n) 1) (LT.lt (Nor …
    -/
  · rintro ⟨m, hm₁, hm₂, hx⟩; exact ⟨↑((m : ℝ) / n), ⟨m, hm₁, hm₂, rfl⟩, hx⟩
                              /-
                                🎉 no goals
                              -/


theorem mem_addWellApproximable_iff (δ : ℕ → ℝ) (x : UnitAddCircle) :
    x ∈ addWellApproximable UnitAddCircle δ ↔
      {n : ℕ | ∃ m < n, gcd m n = 1 ∧ ‖x - ↑((m : ℝ) / n)‖ < δ n}.Infinite := by
  simp only [mem_add_wellApproximable_iff, ← Nat.cofinite_eq_atTop, cofinite.blimsup_set_eq,
    mem_setOf_eq]
  /-
    δ : Nat → Real
    x : UnitAddCircle
    ⊢ Iff (setOf fun n => And (LT.lt 0 n) (Membership.mem (approxAddOrderOf UnitAd …
  -/
  refine iff_of_eq (congr_arg Set.Infinite <| ext fun n => ⟨fun hn => ?_, fun hn => ?_⟩)
    /-
      case refine_1
      δ : Nat → Real
      x : UnitAddCircle
      n : Nat
      hn : Membership.mem (setOf fun n => And (LT.lt 0 n) (Membership.mem (approxAdd …
      ⊢ Membership.mem (setOf fun n => Exists fun m => And (LT.lt m n) (And (Eq (GCD …
    -/
  · exact (mem_approxAddOrderOf_iff hn.1).mp hn.2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      δ : Nat → Real
      x : UnitAddCircle
      n : Nat
      hn : Membership.mem (setOf fun n => Exists fun m => And (LT.lt m n) (And (Eq ( …
      ⊢ Membership.mem (setOf fun n => And (LT.lt 0 n) (Membership.mem (approxAddOrd …
    -/
  · have h : 0 < n := by obtain ⟨m, hm₁, _, _⟩ := hn; exact pos_of_gt hm₁
    /-
      case refine_2
      δ : Nat → Real
      x : UnitAddCircle
      n : Nat
      hn : Membership.mem (setOf fun n => Exists fun m => And (LT.lt m n) (And (Eq ( …
      h : LT.lt 0 n
      ⊢ Membership.mem (setOf fun n => And (LT.lt 0 n) (Membership.mem (approxAddOrd …
    -/
    exact ⟨h, (mem_approxAddOrderOf_iff h).mpr hn⟩
    /-
      🎉 no goals
    -/


local notation a "∤" b => ¬a ∣ b


local notation a "∣∣" b => a ∣ b ∧ (a * a)∤b


local notation "𝕊" => AddCircle T


/-- **Gallagher's ergodic theorem** on Diophantine approximation. -/
theorem addWellApproximable_ae_empty_or_univ (δ : ℕ → ℝ) (hδ : Tendsto δ atTop (𝓝 0)) :
    (∀ᵐ x, ¬addWellApproximable 𝕊 δ x) ∨ ∀ᵐ x, addWellApproximable 𝕊 δ x := by
  /- Sketch of proof:

    Let `E := addWellApproximable 𝕊 δ`. For each prime `p : ℕ`, we can partition `E` into three
    pieces `E = (A p) ∪ (B p) ∪ (C p)` where:
      `A p = blimsup (approxAddOrderOf 𝕊 n (δ n)) atTop (fun n => 0 < n ∧ (p ∤ n))`
      `B p = blimsup (approxAddOrderOf 𝕊 n (δ n)) atTop (fun n => 0 < n ∧ (p ∣∣ n))`
      `C p = blimsup (approxAddOrderOf 𝕊 n (δ n)) atTop (fun n => 0 < n ∧ (p*p ∣ n))`.
    In other words, `A p` is the set of points `x` for which there exist infinitely-many `n` such
    that `x` is within a distance `δ n` of a point of order `n` and `p ∤ n`. Similarly for `B`, `C`.

    These sets have the following key properties:
      1. `A p` is almost invariant under the ergodic map `y ↦ p • y`
      2. `B p` is almost invariant under the ergodic map `y ↦ p • y + 1/p`
      3. `C p` is invariant under the map `y ↦ y + 1/p`
    To prove 1 and 2 we need the key result `blimsup_thickening_mul_ae_eq` but 3 is elementary.

    It follows from `AddCircle.ergodic_nsmul_add` and `Ergodic.ae_empty_or_univ_of_image_ae_le` that
    if either `A p` or `B p` is not almost empty for any `p`, then it is almost full and thus so is
    `E`. We may therefore assume that `A p` and `B p` are almost empty for all `p`. We thus have
    `E` is almost equal to `C p` for every prime. Combining this with 3 we find that `E` is almost
    invariant under the map `y ↦ y + 1/p` for every prime `p`. The required result then follows from
    `AddCircle.ae_empty_or_univ_of_forall_vadd_ae_eq_self`. -/
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    ⊢ Or (Filter.Eventually (fun x => Not (addWellApproximable (AddCircle T) δ x)) …
  -/
  letI : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup _
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    ⊢ Or (Filter.Eventually (fun x => Not (addWellApproximable (AddCircle T) δ x)) …
  -/
  set μ : Measure 𝕊 := volume
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    ⊢ Or (Filter.Eventually (fun x => Not (addWellApproximable (AddCircle T) δ x)) …
  -/
  set u : Nat.Primes → 𝕊 := fun p => ↑((↑(1 : ℕ) : ℝ) / ((p : ℕ) : ℝ) * T)
  have hu₀ : ∀ p : Nat.Primes, addOrderOf (u p) = (p : ℕ) := by
    rintro ⟨p, hp⟩; exact addOrderOf_div_of_gcd_eq_one hp.pos (gcd_one_left p)
  have hu : Tendsto (addOrderOf ∘ u) atTop atTop := by
    rw [(funext hu₀ : addOrderOf ∘ u = (↑))]
    have h_mono : Monotone ((↑) : Nat.Primes → ℕ) := fun p q hpq => hpq
    refine h_mono.tendsto_atTop_atTop fun n => ?_
    obtain ⟨p, hp, hp'⟩ := n.exists_infinite_primes
    exact ⟨⟨p, hp'⟩, hp⟩
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    ⊢ Or (Filter.Eventually (fun x => Not (addWellApproximable (AddCircle T) δ x)) …
  -/
  set E := addWellApproximable 𝕊 δ
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    ⊢ Or (Filter.Eventually (fun x => Not (E x)) (MeasureTheory.ae μ)) (Filter.Eve …
  -/
  set X : ℕ → Set 𝕊 := fun n => approxAddOrderOf 𝕊 n (δ n)
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    ⊢ Or (Filter.Eventually (fun x => Not (E x)) (MeasureTheory.ae μ)) (Filter.Eve …
  -/
  set A : ℕ → Set 𝕊 := fun p => blimsup X atTop fun n => 0 < n ∧ p∤n
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    ⊢ Or (Filter.Eventually (fun x => Not (E x)) (MeasureTheory.ae μ)) (Filter.Eve …
  -/
  set B : ℕ → Set 𝕊 := fun p => blimsup X atTop fun n => 0 < n ∧ p∣∣n
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    ⊢ Or (Filter.Eventually (fun x => Not (E x)) (MeasureTheory.ae μ)) (Filter.Eve …
  -/
  set C : ℕ → Set 𝕊 := fun p => blimsup X atTop fun n => 0 < n ∧ p ^ 2 ∣ n
  have hA₀ : ∀ p, MeasurableSet (A p) := fun p =>
    MeasurableSet.measurableSet_blimsup fun n _ => isOpen_thickening.measurableSet
  have hB₀ : ∀ p, MeasurableSet (B p) := fun p =>
    MeasurableSet.measurableSet_blimsup fun n _ => isOpen_thickening.measurableSet
  have hE₀ : NullMeasurableSet E μ := by
    refine (MeasurableSet.measurableSet_blimsup fun n hn =>
      IsOpen.measurableSet ?_).nullMeasurableSet
    exact isOpen_thickening
  have hE₁ : ∀ p, E = A p ∪ B p ∪ C p := by
    intro p
    simp only [E, A, B, C, addWellApproximable, ← blimsup_or_eq_sup, ← and_or_left, ← sup_eq_union,
      sq]
    congr
    ext n
    tauto
  have hE₂ : ∀ p : Nat.Primes, A p =ᵐ[μ] (∅ : Set 𝕊) ∧ B p =ᵐ[μ] (∅ : Set 𝕊) → E =ᵐ[μ] C p := by
    rintro p ⟨hA, hB⟩
    rw [hE₁ p]
    exact union_ae_eq_right_of_ae_eq_empty ((union_ae_eq_right_of_ae_eq_empty hA).trans hB)
  have hA : ∀ p : Nat.Primes, A p =ᵐ[μ] (∅ : Set 𝕊) ∨ A p =ᵐ[μ] univ := by
    rintro ⟨p, hp⟩
    let f : 𝕊 → 𝕊 := fun y => (p : ℕ) • y
    suffices
      f '' A p ⊆ blimsup (fun n => approxAddOrderOf 𝕊 n (p * δ n)) atTop fun n => 0 < n ∧ p∤n by
      apply (ergodic_nsmul hp.one_lt).ae_empty_or_univ_of_image_ae_le (hA₀ p).nullMeasurableSet
      apply (HasSubset.Subset.eventuallyLE this).congr EventuallyEq.rfl
      exact blimsup_thickening_mul_ae_eq μ (fun n => 0 < n ∧ p∤n) (fun n => {y | addOrderOf y = n})
        (Nat.cast_pos.mpr hp.pos) _ hδ
    refine (sSupHom.setImage f).apply_blimsup_le.trans (mono_blimsup fun n hn => ?_)
    replace hn := Nat.coprime_comm.mp (hp.coprime_iff_not_dvd.2 hn.2)
    exact approxAddOrderOf.image_nsmul_subset_of_coprime (δ n) hp.pos hn
  have hB : ∀ p : Nat.Primes, B p =ᵐ[μ] (∅ : Set 𝕊) ∨ B p =ᵐ[μ] univ := by
    rintro ⟨p, hp⟩
    let x := u ⟨p, hp⟩
    let f : 𝕊 → 𝕊 := fun y => p • y + x
    suffices
      f '' B p ⊆ blimsup (fun n => approxAddOrderOf 𝕊 n (p * δ n)) atTop fun n => 0 < n ∧ p∣∣n by
      apply (ergodic_nsmul_add x hp.one_lt).ae_empty_or_univ_of_image_ae_le
        (hB₀ p).nullMeasurableSet
      apply (HasSubset.Subset.eventuallyLE this).congr EventuallyEq.rfl
      exact blimsup_thickening_mul_ae_eq μ (fun n => 0 < n ∧ p∣∣n) (fun n => {y | addOrderOf y = n})
        (Nat.cast_pos.mpr hp.pos) _ hδ
    refine (sSupHom.setImage f).apply_blimsup_le.trans (mono_blimsup ?_)
    rintro n ⟨hn, h_div, h_ndiv⟩
    have h_cop : (addOrderOf x).Coprime (n / p) := by
      obtain ⟨q, rfl⟩ := h_div
      rw [hu₀, Subtype.coe_mk, hp.coprime_iff_not_dvd, q.mul_div_cancel_left hp.pos]
      exact fun contra => h_ndiv (mul_dvd_mul_left p contra)
    replace h_div : n / p * p = n := Nat.div_mul_cancel h_div
    have hf : f = (fun y => x + y) ∘ fun y => p • y := by
      ext; simp [f, add_comm x]
    simp_rw [Function.comp_apply, le_eq_subset]
    rw [sSupHom.setImage_toFun, hf, image_comp]
    have := @monotone_image 𝕊 𝕊 fun y => x + y
    specialize this (approxAddOrderOf.image_nsmul_subset (δ n) (n / p) hp.pos)
    simp only [h_div] at this ⊢
    refine this.trans ?_
    convert approxAddOrderOf.vadd_subset_of_coprime (p * δ n) h_cop
    rw [hu₀, Subtype.coe_mk, mul_comm p, h_div]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    hA₀ : ∀ (p : Nat), MeasurableSet (A p)
    hB₀ : ∀ (p : Nat), MeasurableSet (B p)
    hE₀ : MeasureTheory.NullMeasurableSet E μ
    hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
    hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
    hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
    hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
    ⊢ Or (Filter.Eventually (fun x => Not (E x)) (MeasureTheory.ae μ)) (Filter.Eve …
  -/
  change (∀ᵐ x, x ∉ E) ∨ E ∈ ae volume
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    hA₀ : ∀ (p : Nat), MeasurableSet (A p)
    hB₀ : ∀ (p : Nat), MeasurableSet (B p)
    hE₀ : MeasureTheory.NullMeasurableSet E μ
    hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
    hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
    hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
    hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
    ⊢ Or (Filter.Eventually (fun x => Not (Membership.mem E x)) (MeasureTheory.ae  …
  -/
  rw [← eventuallyEq_empty, ← eventuallyEq_univ]
  have hC : ∀ p : Nat.Primes, u p +ᵥ C p = C p := by
    intro p
    let e := (AddAction.toPerm (u p) : Equiv.Perm 𝕊).toOrderIsoSet
    change e (C p) = C p
    rw [OrderIso.apply_blimsup e, ← hu₀ p]
    exact blimsup_congr (Eventually.of_forall fun n hn =>
      approxAddOrderOf.vadd_eq_of_mul_dvd (δ n) hn.1 hn.2)
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    δ : Nat → Real
    hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
    this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
    hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
    hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
    E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
    X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
    A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
    hA₀ : ∀ (p : Nat), MeasurableSet (A p)
    hB₀ : ∀ (p : Nat), MeasurableSet (B p)
    hE₀ : MeasureTheory.NullMeasurableSet E μ
    hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
    hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
    hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
    hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
    hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
    ⊢ Or ((MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Empt …
  -/
  by_cases h : ∀ p : Nat.Primes, A p =ᵐ[μ] (∅ : Set 𝕊) ∧ B p =ᵐ[μ] (∅ : Set 𝕊)
  · replace h : ∀ p : Nat.Primes, (u p +ᵥ E : Set _) =ᵐ[μ] E := by
      intro p
      replace hE₂ : E =ᵐ[μ] C p := hE₂ p (h p)
      have h_qmp : Measure.QuasiMeasurePreserving (-u p +ᵥ ·) μ μ :=
        (measurePreserving_vadd _ μ).quasiMeasurePreserving
      refine (h_qmp.vadd_ae_eq_of_ae_eq (u p) hE₂).trans (ae_eq_trans ?_ hE₂.symm)
      rw [hC]
    /-
      case pos
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      h : ∀ (p : Nat.Primes), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u p) E …
      ⊢ Or ((MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Empt …
    -/
    exact ae_empty_or_univ_of_forall_vadd_ae_eq_self hE₀ h hu
    /-
      🎉 no goals
    -/
    /-
      case neg
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      h : Not (∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) Emp …
      ⊢ Or ((MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Empt …
    -/
  · right
    /-
      case neg.h
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      h : Not (∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) Emp …
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Set.univ
    -/
    simp only [not_forall, not_and_or] at h
    /-
      case neg.h
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      h : Exists fun x => Or (Not ((MeasureTheory.ae μ).EventuallyEq (A ↑x) EmptyCol …
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Set.univ
    -/
    obtain ⟨p, hp⟩ := h
    /-
      case neg.h.intro
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      p : Nat.Primes
      hp : Or (Not ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCollection.emptyCo …
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq E Set.univ
    -/
    rw [hE₁ p]
    /-
      case neg.h.intro
      T : Real
      hT : Fact (LT.lt 0 T)
      δ : Nat → Real
      hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
      this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
      μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
      u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
      hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
      hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
      E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
      X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
      A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
      hA₀ : ∀ (p : Nat), MeasurableSet (A p)
      hB₀ : ∀ (p : Nat), MeasurableSet (B p)
      hE₀ : MeasureTheory.NullMeasurableSet E μ
      hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
      hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
      hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
      hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
      hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
      p : Nat.Primes
      hp : Or (Not ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCollection.emptyCo …
      ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Union.uni …
    -/
    cases hp
      /-
        case neg.h.intro.inl
        T : Real
        hT : Fact (LT.lt 0 T)
        δ : Nat → Real
        hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
        this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
        μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
        u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
        hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
        hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
        E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
        X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
        A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        hA₀ : ∀ (p : Nat), MeasurableSet (A p)
        hB₀ : ∀ (p : Nat), MeasurableSet (B p)
        hE₀ : MeasureTheory.NullMeasurableSet E μ
        hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
        hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
        hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
        hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
        hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
        p : Nat.Primes
        h✝ : Not ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCollection.emptyCollec …
        ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Union.uni …
      -/
    · cases' hA p with _ h; · contradiction
                              /-
                                🎉 no goals
                              -/
      /-
        case neg.h.intro.inl.inr
        T : Real
        hT : Fact (LT.lt 0 T)
        δ : Nat → Real
        hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
        this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
        μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
        u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
        hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
        hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
        E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
        X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
        A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        hA₀ : ∀ (p : Nat), MeasurableSet (A p)
        hB₀ : ∀ (p : Nat), MeasurableSet (B p)
        hE₀ : MeasureTheory.NullMeasurableSet E μ
        hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
        hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
        hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
        hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
        hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
        p : Nat.Primes
        h✝ : Not ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCollection.emptyCollec …
        h : (MeasureTheory.ae μ).EventuallyEq (A ↑p) Set.univ
        ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Union.uni …
      -/
      simp only [μ, h, union_ae_eq_univ_of_ae_eq_univ_left]
      /-
        🎉 no goals
      -/
      /-
        case neg.h.intro.inr
        T : Real
        hT : Fact (LT.lt 0 T)
        δ : Nat → Real
        hδ : Filter.Tendsto δ Filter.atTop (nhds 0)
        this : SemilatticeSup Nat.Primes := Nat.Subtype.semilatticeSup Irreducible
        μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
        u : Nat.Primes → AddCircle T := fun p => ↑(HMul.hMul (HDiv.hDiv ↑1 ↑↑p) T)
        hu₀ : ∀ (p : Nat.Primes), Eq (addOrderOf (u p)) ↑p
        hu : Filter.Tendsto (Function.comp addOrderOf u) Filter.atTop Filter.atTop
        E : Set (AddCircle T) := addWellApproximable (AddCircle T) δ
        X : Nat → Set (AddCircle T) := fun n => approxAddOrderOf (AddCircle T) n (δ n)
        A : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        B : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        C : Nat → Set (AddCircle T) := fun p => Filter.blimsup X Filter.atTop fun n => …
        hA₀ : ∀ (p : Nat), MeasurableSet (A p)
        hB₀ : ∀ (p : Nat), MeasurableSet (B p)
        hE₀ : MeasureTheory.NullMeasurableSet E μ
        hE₁ : ∀ (p : Nat), Eq E (Union.union (Union.union (A p) (B p)) (C p))
        hE₂ : ∀ (p : Nat.Primes), And ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyC …
        hA : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (A ↑p) EmptyCol …
        hB : ∀ (p : Nat.Primes), Or ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCol …
        hC : ∀ (p : Nat.Primes), Eq (HVAdd.hVAdd (u p) (C ↑p)) (C ↑p)
        p : Nat.Primes
        h✝ : Not ((MeasureTheory.ae μ).EventuallyEq (B ↑p) EmptyCollection.emptyCollec …
        ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Union.uni …
      -/
    · cases' hB p with _ h; · contradiction
                              /-
                                🎉 no goals
                              -/
      simp only [μ, h, union_ae_eq_univ_of_ae_eq_univ_left,
        union_ae_eq_univ_of_ae_eq_univ_right]


/-- A general version of **Dirichlet's approximation theorem**.

See also `AddCircle.exists_norm_nsmul_le`. -/
lemma _root_.NormedAddCommGroup.exists_norm_nsmul_le {A : Type*}
    [NormedAddCommGroup A] [CompactSpace A] [ConnectedSpace A]
    [MeasurableSpace A] [BorelSpace A] {μ : Measure A} [μ.IsAddHaarMeasure]
    (ξ : A) {n : ℕ} (hn : 0 < n) (δ : ℝ) (hδ : μ univ ≤ (n + 1) • μ (closedBall (0 : A) (δ/2))) :
    ∃ j ∈ Icc 1 n, ‖j • ξ‖ ≤ δ := by
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    hδ : LE.le (μ Set.univ) (HSMul.hSMul (HAdd.hAdd n 1) (μ (Metric.closedBall 0 ( …
    ⊢ Exists fun j => And (Membership.mem (Set.Icc 1 n) j) (LE.le (Norm.norm (HSMu …
  -/
  have : IsFiniteMeasure μ := CompactSpace.isFiniteMeasure
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    hδ : LE.le (μ Set.univ) (HSMul.hSMul (HAdd.hAdd n 1) (μ (Metric.closedBall 0 ( …
    this : MeasureTheory.IsFiniteMeasure μ
    ⊢ Exists fun j => And (Membership.mem (Set.Icc 1 n) j) (LE.le (Norm.norm (HSMu …
  -/
  let B : Icc 0 n → Set A := fun j ↦ closedBall ((j : ℕ) • ξ) (δ/2)
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    hδ : LE.le (μ Set.univ) (HSMul.hSMul (HAdd.hAdd n 1) (μ (Metric.closedBall 0 ( …
    this : MeasureTheory.IsFiniteMeasure μ
    B : ↑(Set.Icc 0 n) → Set A := fun j => Metric.closedBall (HSMul.hSMul (↑j) ξ)  …
    ⊢ Exists fun j => And (Membership.mem (Set.Icc 1 n) j) (LE.le (Norm.norm (HSMu …
  -/
  have hB : ∀ j, IsClosed (B j) := fun j ↦ isClosed_ball
  suffices ¬ Pairwise (Disjoint on B) by
    obtain ⟨i, j, hij, x, hx⟩ := exists_lt_mem_inter_of_not_pairwise_disjoint this
    refine ⟨j - i, ⟨le_tsub_of_add_le_left hij, ?_⟩, ?_⟩
    · simpa only [tsub_le_iff_right] using j.property.2.trans le_self_add
    · rw [sub_nsmul _ (Subtype.coe_le_coe.mpr hij.le), ← sub_eq_add_neg, ← dist_eq_norm]
      exact (dist_triangle ((j : ℕ) • ξ) x ((i : ℕ) • ξ)).trans (by
        linarith [mem_closedBall.mp hx.1, mem_closedBall'.mp hx.2])
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    hδ : LE.le (μ Set.univ) (HSMul.hSMul (HAdd.hAdd n 1) (μ (Metric.closedBall 0 ( …
    this : MeasureTheory.IsFiniteMeasure μ
    B : ↑(Set.Icc 0 n) → Set A := fun j => Metric.closedBall (HSMul.hSMul (↑j) ξ)  …
    hB : ∀ (j : ↑(Set.Icc 0 n)), IsClosed (B j)
    ⊢ Not (Pairwise (Function.onFun Disjoint B))
  -/
  by_contra h
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    hδ : LE.le (μ Set.univ) (HSMul.hSMul (HAdd.hAdd n 1) (μ (Metric.closedBall 0 ( …
    this : MeasureTheory.IsFiniteMeasure μ
    B : ↑(Set.Icc 0 n) → Set A := fun j => Metric.closedBall (HSMul.hSMul (↑j) ξ)  …
    hB : ∀ (j : ↑(Set.Icc 0 n)), IsClosed (B j)
    h : Pairwise (Function.onFun Disjoint B)
    ⊢ False
  -/
  apply hn.ne'
  have h' : ⋃ j, B j = univ := by
    rw [← (isClosed_iUnion_of_finite hB).measure_eq_univ_iff_eq (μ := μ)]
    refine le_antisymm (μ.mono (subset_univ _)) ?_
    simp_rw [measure_iUnion h (fun _ ↦ measurableSet_closedBall), tsum_fintype,
      B, μ.addHaar_closedBall_center, Finset.sum_const, Finset.card_univ, Nat.card_fintypeIcc,
      tsub_zero]
    exact hδ
  replace hδ : 0 ≤ δ/2 := by
    by_contra contra
    suffices μ (closedBall 0 (δ/2)) = 0 by
      apply isOpen_univ.measure_ne_zero μ univ_nonempty <| le_zero_iff.mp <| le_trans hδ _
      simp [this]
    rw [not_le, ← closedBall_eq_empty (x := (0 : A))] at contra
    simp [contra]
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    this : MeasureTheory.IsFiniteMeasure μ
    B : ↑(Set.Icc 0 n) → Set A := fun j => Metric.closedBall (HSMul.hSMul (↑j) ξ)  …
    hB : ∀ (j : ↑(Set.Icc 0 n)), IsClosed (B j)
    h : Pairwise (Function.onFun Disjoint B)
    h' : Eq (Set.iUnion fun j => B j) Set.univ
    hδ : LE.le 0 (HDiv.hDiv δ 2)
    ⊢ Eq n 0
  -/
  have h'' : ∀ j, (B j).Nonempty := by intro j; rwa [nonempty_closedBall]
  /-
    A : Type u_1
    inst✝⁵ : NormedAddCommGroup A
    inst✝⁴ : CompactSpace A
    inst✝³ : ConnectedSpace A
    inst✝² : MeasurableSpace A
    inst✝¹ : BorelSpace A
    μ : MeasureTheory.Measure A
    inst✝ : μ.IsAddHaarMeasure
    ξ : A
    n : Nat
    hn : LT.lt 0 n
    δ : Real
    this : MeasureTheory.IsFiniteMeasure μ
    B : ↑(Set.Icc 0 n) → Set A := fun j => Metric.closedBall (HSMul.hSMul (↑j) ξ)  …
    hB : ∀ (j : ↑(Set.Icc 0 n)), IsClosed (B j)
    h : Pairwise (Function.onFun Disjoint B)
    h' : Eq (Set.iUnion fun j => B j) Set.univ
    hδ : LE.le 0 (HDiv.hDiv δ 2)
    h'' : ∀ (j : ↑(Set.Icc 0 n)), (B j).Nonempty
    ⊢ Eq n 0
  -/
  simpa using subsingleton_of_disjoint_isClosed_iUnion_eq_univ h'' h hB h'
  /-
    🎉 no goals
  -/


/-- **Dirichlet's approximation theorem**

See also `Real.exists_rat_abs_sub_le_and_den_le`. -/
lemma exists_norm_nsmul_le (ξ : 𝕊) {n : ℕ} (hn : 0 < n) :
    ∃ j ∈ Icc 1 n, ‖j • ξ‖ ≤ T / ↑(n + 1) := by
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    ξ : AddCircle T
    n : Nat
    hn : LT.lt 0 n
    ⊢ Exists fun j => And (Membership.mem (Set.Icc 1 n) j) (LE.le (Norm.norm (HSMu …
  -/
  apply NormedAddCommGroup.exists_norm_nsmul_le (μ := volume) ξ hn
  rw [AddCircle.measure_univ, volume_closedBall, ← ENNReal.ofReal_nsmul,
    mul_div_cancel₀ _ two_ne_zero, min_eq_right (div_le_self hT.out.le <| by simp), nsmul_eq_mul,
    mul_div_cancel₀ _ (Nat.cast_ne_zero.mpr n.succ_ne_zero)]


