/-- An element `ζ` is a primitive `k`-th root of unity if `ζ ^ k = 1`,
and if `l` satisfies `ζ ^ l = 1` then `k ∣ l`. -/
@[mk_iff IsPrimitiveRoot.iff_def]
structure IsPrimitiveRoot (ζ : M) (k : ℕ) : Prop where
  pow_eq_one : ζ ^ k = 1
  dvd_of_pow_eq_one : ∀ l : ℕ, ζ ^ l = 1 → k ∣ l


/-- Turn a primitive root μ into a member of the `rootsOfUnity` subgroup. -/
@[simps!]
def IsPrimitiveRoot.toRootsOfUnity {μ : M} {n : ℕ} [NeZero n] (h : IsPrimitiveRoot μ n) :
    rootsOfUnity n M :=
  rootsOfUnity.mkOfPowEq μ h.pow_eq_one


open scoped Classical in
/-- `primitiveRoots k R` is the finset of primitive `k`-th roots of unity
in the integral domain `R`. -/
def primitiveRoots (k : ℕ) (R : Type*) [CommRing R] [IsDomain R] : Finset R :=
  (nthRoots k (1 : R)).toFinset.filter fun ζ => IsPrimitiveRoot ζ k


@[simp]
theorem mem_primitiveRoots {ζ : R} (h0 : 0 < k) : ζ ∈ primitiveRoots k R ↔ IsPrimitiveRoot ζ k := by
  classical
  rw [primitiveRoots, mem_filter, Multiset.mem_toFinset, mem_nthRoots h0, and_iff_right_iff_imp]
  exact IsPrimitiveRoot.pow_eq_one


@[simp]
theorem primitiveRoots_zero : primitiveRoots 0 R = ∅ := by
  classical
  rw [primitiveRoots, nthRoots_zero, Multiset.toFinset_zero, Finset.filter_empty]


theorem isPrimitiveRoot_of_mem_primitiveRoots {ζ : R} (h : ζ ∈ primitiveRoots k R) :
    IsPrimitiveRoot ζ k :=
                                     /-
                                       R : Type u_4
                                       k : Nat
                                       inst✝¹ : CommRing R
                                       inst✝ : IsDomain R
                                       ζ : R
                                       h : Membership.mem (primitiveRoots k R) ζ
                                       hk : Eq k 0
                                       ⊢ IsPrimitiveRoot ζ k
                                     -/
  k.eq_zero_or_pos.elim (fun hk ↦ by simp [hk] at h) fun hk ↦ (mem_primitiveRoots hk).1 h
                                     /-
                                       🎉 no goals
                                     -/


theorem mk_of_lt (ζ : M) (hk : 0 < k) (h1 : ζ ^ k = 1) (h : ∀ l : ℕ, 0 < l → l < k → ζ ^ l ≠ 1) :
    IsPrimitiveRoot ζ k := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    ⊢ IsPrimitiveRoot ζ k
  -/
  refine ⟨h1, fun l hl ↦ ?_⟩
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    l : Nat
    hl : Eq (HPow.hPow ζ l) 1
    ⊢ Dvd.dvd k l
  -/
  suffices k.gcd l = k by exact this ▸ k.gcd_dvd_right l
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    l : Nat
    hl : Eq (HPow.hPow ζ l) 1
    ⊢ Eq (k.gcd l) k
  -/
  rw [eq_iff_le_not_lt]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    l : Nat
    hl : Eq (HPow.hPow ζ l) 1
    ⊢ And (LE.le (k.gcd l) k) (Not (LT.lt (k.gcd l) k))
  -/
  refine ⟨Nat.le_of_dvd hk (k.gcd_dvd_left l), ?_⟩
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    l : Nat
    hl : Eq (HPow.hPow ζ l) 1
    ⊢ Not (LT.lt (k.gcd l) k)
  -/
  intro h'; apply h _ (Nat.gcd_pos_of_pos_left _ hk) h'
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h1 : Eq (HPow.hPow ζ k) 1
    h : ∀ (l : Nat), LT.lt 0 l → LT.lt l k → Ne (HPow.hPow ζ l) 1
    l : Nat
    hl : Eq (HPow.hPow ζ l) 1
    h' : LT.lt (k.gcd l) k
    ⊢ Eq (HPow.hPow ζ (k.gcd l)) 1
  -/
  exact pow_gcd_eq_one _ h1 hl
  /-
    🎉 no goals
  -/


@[nontriviality]
theorem of_subsingleton [Subsingleton M] (x : M) : IsPrimitiveRoot x 1 :=
  ⟨Subsingleton.elim _ _, fun _ _ ↦ one_dvd _⟩


theorem pow_eq_one_iff_dvd (h : IsPrimitiveRoot ζ k) (l : ℕ) : ζ ^ l = 1 ↔ k ∣ l :=
  ⟨h.dvd_of_pow_eq_one l, by
    /-
      M : Type u_1
      inst✝ : CommMonoid M
      k : Nat
      ζ : M
      h : IsPrimitiveRoot ζ k
      l : Nat
      ⊢ Dvd.dvd k l → Eq (HPow.hPow ζ l) 1
    -/
    rintro ⟨i, rfl⟩; simp only [pow_mul, h.pow_eq_one, one_pow]⟩
                     /-
                       🎉 no goals
                     -/


theorem isUnit (h : IsPrimitiveRoot ζ k) (h0 : 0 < k) : IsUnit ζ := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    ⊢ IsUnit ζ
  -/
  apply isUnit_of_mul_eq_one ζ (ζ ^ (k - 1))
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    ⊢ Eq (HMul.hMul ζ (HPow.hPow ζ (HSub.hSub k 1))) 1
  -/
  rw [← pow_succ', tsub_add_cancel_of_le h0.nat_succ_le, h.pow_eq_one]
  /-
    🎉 no goals
  -/


theorem pow_ne_one_of_pos_of_lt (h : IsPrimitiveRoot ζ k) (h0 : 0 < l) (hl : l < k) : ζ ^ l ≠ 1 :=
  mt (Nat.le_of_dvd h0 ∘ h.dvd_of_pow_eq_one _) <| not_le_of_lt hl


theorem ne_one (h : IsPrimitiveRoot ζ k) (hk : 1 < k) : ζ ≠ 1 :=
  h.pow_ne_one_of_pos_of_lt zero_lt_one hk ∘ (pow_one ζ).trans


theorem pow_inj (h : IsPrimitiveRoot ζ k) ⦃i j : ℕ⦄ (hi : i < k) (hj : j < k) (H : ζ ^ i = ζ ^ j) :
    i = j := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i j : Nat
    hi : LT.lt i k
    hj : LT.lt j k
    H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
    ⊢ Eq i j
  -/
  wlog hij : i ≤ j generalizing i j
    /-
      case inr
      M : Type u_1
      inst✝ : CommMonoid M
      k : Nat
      ζ : M
      h : IsPrimitiveRoot ζ k
      i j : Nat
      hi : LT.lt i k
      hj : LT.lt j k
      H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
      this : ∀ ⦃i j : Nat⦄, LT.lt i k → LT.lt j k → Eq (HPow.hPow ζ i) (HPow.hPow ζ  …
      hij : Not (LE.le i j)
      ⊢ Eq i j
    -/
  · exact (this hj hi H.symm (le_of_not_le hij)).symm
    /-
      🎉 no goals
    -/
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i j : Nat
    hi : LT.lt i k
    hj : LT.lt j k
    H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
    hij : LE.le i j
    ⊢ Eq i j
  -/
  apply le_antisymm hij
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i j : Nat
    hi : LT.lt i k
    hj : LT.lt j k
    H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
    hij : LE.le i j
    ⊢ LE.le j i
  -/
  rw [← tsub_eq_zero_iff_le]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i j : Nat
    hi : LT.lt i k
    hj : LT.lt j k
    H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
    hij : LE.le i j
    ⊢ Eq (HSub.hSub j i) 0
  -/
  apply Nat.eq_zero_of_dvd_of_lt _ (lt_of_le_of_lt tsub_le_self hj)
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i j : Nat
    hi : LT.lt i k
    hj : LT.lt j k
    H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
    hij : LE.le i j
    ⊢ Dvd.dvd k (HSub.hSub j i)
  -/
  apply h.dvd_of_pow_eq_one
  rw [← ((h.isUnit (lt_of_le_of_lt (Nat.zero_le _) hi)).pow i).mul_left_inj, ← pow_add,
    tsub_add_cancel_of_le hij, H, one_mul]


theorem one : IsPrimitiveRoot (1 : M) 1 :=
  { pow_eq_one := pow_one _
    dvd_of_pow_eq_one := fun _ _ ↦ one_dvd _ }


@[simp]
theorem one_right_iff : IsPrimitiveRoot ζ 1 ↔ ζ = 1 := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ζ : M
    ⊢ Iff (IsPrimitiveRoot ζ 1) (Eq ζ 1)
  -/
  constructor
    /-
      case mp
      M : Type u_1
      inst✝ : CommMonoid M
      ζ : M
      ⊢ IsPrimitiveRoot ζ 1 → Eq ζ 1
    -/
  · intro h; rw [← pow_one ζ, h.pow_eq_one]
             /-
               🎉 no goals
             -/
    /-
      case mpr
      M : Type u_1
      inst✝ : CommMonoid M
      ζ : M
      ⊢ Eq ζ 1 → IsPrimitiveRoot ζ 1
    -/
  · rintro rfl; exact one
                /-
                  🎉 no goals
                -/


@[simp]
theorem coe_submonoidClass_iff {M B : Type*} [CommMonoid M] [SetLike B M] [SubmonoidClass B M]
    {N : B} {ζ : N} : IsPrimitiveRoot (ζ : M) k ↔ IsPrimitiveRoot ζ k := by
  /-
    k : Nat
    M : Type u_7
    B : Type u_8
    inst✝² : CommMonoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    N : B
    ζ : Subtype fun x => Membership.mem N x
    ⊢ Iff (IsPrimitiveRoot (↑ζ) k) (IsPrimitiveRoot ζ k)
  -/
  simp_rw [iff_def]
  /-
    k : Nat
    M : Type u_7
    B : Type u_8
    inst✝² : CommMonoid M
    inst✝¹ : SetLike B M
    inst✝ : SubmonoidClass B M
    N : B
    ζ : Subtype fun x => Membership.mem N x
    ⊢ Iff (And (Eq (HPow.hPow (↑ζ) k) 1) (∀ (l : Nat), Eq (HPow.hPow (↑ζ) l) 1 → D …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_units_iff {ζ : Mˣ} : IsPrimitiveRoot (ζ : M) k ↔ IsPrimitiveRoot ζ k := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : Units M
    ⊢ Iff (IsPrimitiveRoot (↑ζ) k) (IsPrimitiveRoot ζ k)
  -/
  simp only [iff_def, Units.ext_iff, Units.val_pow_eq_pow_val, Units.val_one]
  /-
    🎉 no goals
  -/


lemma isUnit_unit {ζ : M} {n} (hn) (hζ : IsPrimitiveRoot ζ n) :
    IsPrimitiveRoot (hζ.isUnit hn).unit n := coe_units_iff.mp hζ


lemma isUnit_unit' {ζ : G} {n} (hn) (hζ : IsPrimitiveRoot ζ n) :
    IsPrimitiveRoot (hζ.isUnit hn).unit' n := coe_units_iff.mp hζ


theorem pow_of_coprime (h : IsPrimitiveRoot ζ k) (i : ℕ) (hi : i.Coprime k) :
    IsPrimitiveRoot (ζ ^ i) k := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i : Nat
    hi : i.Coprime k
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  by_cases h0 : k = 0
    /-
      case pos
      M : Type u_1
      inst✝ : CommMonoid M
      k : Nat
      ζ : M
      h : IsPrimitiveRoot ζ k
      i : Nat
      hi : i.Coprime k
      h0 : Eq k 0
      ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
    -/
  · subst k; simp_all only [pow_one, Nat.coprime_zero_right]
             /-
               🎉 no goals
             -/
  /-
    case neg
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    i : Nat
    hi : i.Coprime k
    h0 : Not (Eq k 0)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  rcases h.isUnit (Nat.pos_of_ne_zero h0) with ⟨ζ, rfl⟩
  /-
    case neg.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k i : Nat
    hi : i.Coprime k
    h0 : Not (Eq k 0)
    ζ : Units M
    h : IsPrimitiveRoot (↑ζ) k
    ⊢ IsPrimitiveRoot (HPow.hPow (↑ζ) i) k
  -/
  rw [← Units.val_pow_eq_pow_val]
  /-
    case neg.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k i : Nat
    hi : i.Coprime k
    h0 : Not (Eq k 0)
    ζ : Units M
    h : IsPrimitiveRoot (↑ζ) k
    ⊢ IsPrimitiveRoot (↑(HPow.hPow ζ i)) k
  -/
  rw [coe_units_iff] at h ⊢
  refine
    { pow_eq_one := by rw [← pow_mul', pow_mul, h.pow_eq_one, one_pow]
      dvd_of_pow_eq_one := fun l hl ↦ h.dvd_of_pow_eq_one l ?_ }
  rw [← pow_one ζ, ← zpow_natCast ζ, ← hi.gcd_eq_one, Nat.gcd_eq_gcd_ab, zpow_add, mul_pow,
    ← zpow_natCast, ← zpow_mul, mul_right_comm]
  /-
    case neg.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k i : Nat
    hi : i.Coprime k
    h0 : Not (Eq k 0)
    ζ : Units M
    h : IsPrimitiveRoot ζ k
    l : Nat
    hl : Eq (HPow.hPow (HPow.hPow ζ i) l) 1
    ⊢ Eq (HMul.hMul (HPow.hPow ζ (HMul.hMul (HMul.hMul ↑i ↑l) (i.gcdA k))) (HPow.h …
  -/
  simp only [zpow_mul, hl, h.pow_eq_one, one_zpow, one_pow, one_mul, zpow_natCast]
  /-
    🎉 no goals
  -/


theorem pow_of_prime (h : IsPrimitiveRoot ζ k) {p : ℕ} (hprime : Nat.Prime p) (hdiv : ¬p ∣ k) :
    IsPrimitiveRoot (ζ ^ p) k :=
  h.pow_of_coprime p (hprime.coprime_iff_not_dvd.2 hdiv)


theorem pow_iff_coprime (h : IsPrimitiveRoot ζ k) (h0 : 0 < k) (i : ℕ) :
    IsPrimitiveRoot (ζ ^ i) k ↔ i.Coprime k := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i : Nat
    ⊢ Iff (IsPrimitiveRoot (HPow.hPow ζ i) k) (i.Coprime k)
  -/
  refine ⟨fun hi ↦ ?_, h.pow_of_coprime i⟩
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ i) k
    ⊢ i.Coprime k
  -/
  obtain ⟨a, ha⟩ := i.gcd_dvd_left k
  /-
    case intro
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ i) k
    a : Nat
    ha : Eq i (HMul.hMul (i.gcd k) a)
    ⊢ i.Coprime k
  -/
  obtain ⟨b, hb⟩ := i.gcd_dvd_right k
  suffices b = k by
    rwa [this, eq_comm, Nat.mul_left_eq_self_iff h0, ← Nat.coprime_iff_gcd_eq_one] at hb
  /-
    case intro.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ i) k
    a : Nat
    ha : Eq i (HMul.hMul (i.gcd k) a)
    b : Nat
    hb : Eq k (HMul.hMul (i.gcd k) b)
    ⊢ Eq b k
  -/
  rw [ha] at hi
  /-
    case intro.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i a : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ (HMul.hMul (i.gcd k) a)) k
    ha : Eq i (HMul.hMul (i.gcd k) a)
    b : Nat
    hb : Eq k (HMul.hMul (i.gcd k) b)
    ⊢ Eq b k
  -/
  rw [mul_comm] at hb
  /-
    case intro.intro
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i a : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ (HMul.hMul (i.gcd k) a)) k
    ha : Eq i (HMul.hMul (i.gcd k) a)
    b : Nat
    hb : Eq k (HMul.hMul b (i.gcd k))
    ⊢ Eq b k
  -/
  apply Nat.dvd_antisymm ⟨i.gcd k, hb⟩ (hi.dvd_of_pow_eq_one b _)
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    h0 : LT.lt 0 k
    i a : Nat
    hi : IsPrimitiveRoot (HPow.hPow ζ (HMul.hMul (i.gcd k) a)) k
    ha : Eq i (HMul.hMul (i.gcd k) a)
    b : Nat
    hb : Eq k (HMul.hMul b (i.gcd k))
    ⊢ Eq (HPow.hPow (HPow.hPow ζ (HMul.hMul (i.gcd k) a)) b) 1
  -/
  rw [← pow_mul', ← mul_assoc, ← hb, pow_mul, h.pow_eq_one, one_pow]
  /-
    🎉 no goals
  -/


protected theorem orderOf (ζ : M) : IsPrimitiveRoot ζ (orderOf ζ) :=
  ⟨pow_orderOf_eq_one ζ, fun _ ↦ orderOf_dvd_of_pow_eq_one⟩


theorem unique {ζ : M} (hk : IsPrimitiveRoot ζ k) (hl : IsPrimitiveRoot ζ l) : k = l :=
  Nat.dvd_antisymm (hk.2 _ hl.1) (hl.2 _ hk.1)


theorem eq_orderOf (h : IsPrimitiveRoot ζ k) : k = orderOf ζ :=
  h.unique (IsPrimitiveRoot.orderOf ζ)


protected theorem iff (hk : 0 < k) :
    IsPrimitiveRoot ζ k ↔ ζ ^ k = 1 ∧ ∀ l : ℕ, 0 < l → l < k → ζ ^ l ≠ 1 := by
  refine ⟨fun h ↦ ⟨h.pow_eq_one, fun l hl' hl ↦ ?_⟩,
    fun ⟨hζ, hl⟩ ↦ IsPrimitiveRoot.mk_of_lt ζ hk hζ hl⟩
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h : IsPrimitiveRoot ζ k
    l : Nat
    hl' : LT.lt 0 l
    hl : LT.lt l k
    ⊢ Ne (HPow.hPow ζ l) 1
  -/
  rw [h.eq_orderOf] at hl
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    hk : LT.lt 0 k
    h : IsPrimitiveRoot ζ k
    l : Nat
    hl' : LT.lt 0 l
    hl : LT.lt l (orderOf ζ)
    ⊢ Ne (HPow.hPow ζ l) 1
  -/
  exact pow_ne_one_of_lt_orderOf hl'.ne' hl
  /-
    🎉 no goals
  -/


protected theorem not_iff : ¬IsPrimitiveRoot ζ k ↔ orderOf ζ ≠ k :=
  ⟨fun h hk ↦ h <| hk ▸ IsPrimitiveRoot.orderOf ζ,
    fun h hk ↦ h.symm <| hk.unique <| IsPrimitiveRoot.orderOf ζ⟩


theorem pow_mul_pow_lcm {ζ' : M} {k' : ℕ} (hζ : IsPrimitiveRoot ζ k) (hζ' : IsPrimitiveRoot ζ' k')
    (hk : k ≠ 0) (hk' : k' ≠ 0) :
    IsPrimitiveRoot
      (ζ ^ (k / Nat.factorizationLCMLeft k k') * ζ' ^ (k' / Nat.factorizationLCMRight k k'))
      (Nat.lcm k k') := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ ζ' : M
    k' : Nat
    hζ : IsPrimitiveRoot ζ k
    hζ' : IsPrimitiveRoot ζ' k'
    hk : Ne k 0
    hk' : Ne k' 0
    ⊢ IsPrimitiveRoot (HMul.hMul (HPow.hPow ζ (HDiv.hDiv k (k.factorizationLCMLeft …
  -/
  convert IsPrimitiveRoot.orderOf _
  convert ((Commute.all ζ ζ').orderOf_mul_pow_eq_lcm
    (by simpa [← hζ.eq_orderOf]) (by simpa [← hζ'.eq_orderOf])).symm using 2
  /-
    case h.e'_2.h.e'_1
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ ζ' : M
    k' : Nat
    hζ : IsPrimitiveRoot ζ k
    hζ' : IsPrimitiveRoot ζ' k'
    hk : Ne k 0
    hk' : Ne k' 0
    ⊢ Eq k (orderOf ζ)
  -/
  all_goals simp [hζ.eq_orderOf, hζ'.eq_orderOf]
  /-
    🎉 no goals
  -/


theorem pow_of_dvd (h : IsPrimitiveRoot ζ k) {p : ℕ} (hp : p ≠ 0) (hdiv : p ∣ k) :
    IsPrimitiveRoot (ζ ^ p) (k / p) := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    p : Nat
    hp : Ne p 0
    hdiv : Dvd.dvd p k
    ⊢ IsPrimitiveRoot (HPow.hPow ζ p) (HDiv.hDiv k p)
  -/
  rw [h.eq_orderOf] at hdiv ⊢
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    p : Nat
    hp : Ne p 0
    hdiv : Dvd.dvd p (orderOf ζ)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ p) (HDiv.hDiv (orderOf ζ) p)
  -/
  rw [← orderOf_pow_of_dvd hp hdiv]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    k : Nat
    ζ : M
    h : IsPrimitiveRoot ζ k
    p : Nat
    hp : Ne p 0
    hdiv : Dvd.dvd p (orderOf ζ)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ p) (orderOf (HPow.hPow ζ p))
  -/
  exact IsPrimitiveRoot.orderOf _
  /-
    🎉 no goals
  -/


protected theorem mem_rootsOfUnity {ζ : Mˣ} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    ζ ∈ rootsOfUnity n M :=
  h.pow_eq_one


/-- If there is an `n`-th primitive root of unity in `R` and `b` divides `n`,
then there is a `b`-th primitive root of unity in `R`. -/
theorem pow {n : ℕ} {a b : ℕ} (hn : 0 < n) (h : IsPrimitiveRoot ζ n) (hprod : n = a * b) :
    IsPrimitiveRoot (ζ ^ a) b := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ζ : M
    n a b : Nat
    hn : LT.lt 0 n
    h : IsPrimitiveRoot ζ n
    hprod : Eq n (HMul.hMul a b)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ a) b
  -/
  subst n
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ζ : M
    a b : Nat
    hn : LT.lt 0 (HMul.hMul a b)
    h : IsPrimitiveRoot ζ (HMul.hMul a b)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ a) b
  -/
  simp only [iff_def, ← pow_mul, h.pow_eq_one, true_and]
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ζ : M
    a b : Nat
    hn : LT.lt 0 (HMul.hMul a b)
    h : IsPrimitiveRoot ζ (HMul.hMul a b)
    ⊢ ∀ (l : Nat), Eq (HPow.hPow ζ (HMul.hMul a l)) 1 → Dvd.dvd b l
  -/
  intro l hl
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ζ : M
    a b : Nat
    hn : LT.lt 0 (HMul.hMul a b)
    h : IsPrimitiveRoot ζ (HMul.hMul a b)
    l : Nat
    hl : Eq (HPow.hPow ζ (HMul.hMul a l)) 1
    ⊢ Dvd.dvd b l
  -/
  exact Nat.dvd_of_mul_dvd_mul_left (Nat.pos_of_mul_pos_right hn) <| h.dvd_of_pow_eq_one _ hl
  /-
    🎉 no goals
  -/


lemma injOn_pow {n : ℕ} {ζ : M} (hζ : IsPrimitiveRoot ζ n) :
    Set.InjOn (ζ ^ ·) (Finset.range n) := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    n : Nat
    ζ : M
    hζ : IsPrimitiveRoot ζ n
    ⊢ Set.InjOn (fun x => HPow.hPow ζ x) ↑(Finset.range n)
  -/
  intros i hi j hj e
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    n : Nat
    ζ : M
    hζ : IsPrimitiveRoot ζ n
    i : Nat
    hi : Membership.mem (↑(Finset.range n)) i
    j : Nat
    hj : Membership.mem (↑(Finset.range n)) j
    e : Eq ((fun x => HPow.hPow ζ x) i) ((fun x => HPow.hPow ζ x) j)
    ⊢ Eq i j
  -/
  rw [Finset.coe_range, Set.mem_Iio] at hi hj
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    n : Nat
    ζ : M
    hζ : IsPrimitiveRoot ζ n
    i : Nat
    hi : LT.lt i n
    j : Nat
    hj : LT.lt j n
    e : Eq ((fun x => HPow.hPow ζ x) i) ((fun x => HPow.hPow ζ x) j)
    ⊢ Eq i j
  -/
  exact hζ.pow_inj hi hj e
  /-
    🎉 no goals
  -/


theorem map_of_injective [MonoidHomClass F M N] (h : IsPrimitiveRoot ζ k) (hf : Injective f) :
    IsPrimitiveRoot (f ζ) k where
                   /-
                     M : Type u_1
                     N : Type u_2
                     F : Type u_6
                     inst✝³ : CommMonoid M
                     inst✝² : CommMonoid N
                     k : Nat
                     ζ : M
                     f : F
                     inst✝¹ : FunLike F M N
                     inst✝ : MonoidHomClass F M N
                     h : IsPrimitiveRoot ζ k
                     hf : Function.Injective ⇑f
                     ⊢ Eq (HPow.hPow (f ζ) k) 1
                   -/
  pow_eq_one := by rw [← map_pow, h.pow_eq_one, map_one]
                   /-
                     🎉 no goals
                   -/
  dvd_of_pow_eq_one := by
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot ζ k
      hf : Function.Injective ⇑f
      ⊢ ∀ (l : Nat), Eq (HPow.hPow (f ζ) l) 1 → Dvd.dvd k l
    -/
    rw [h.eq_orderOf]
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot ζ k
      hf : Function.Injective ⇑f
      ⊢ ∀ (l : Nat), Eq (HPow.hPow (f ζ) l) 1 → Dvd.dvd (orderOf ζ) l
    -/
    intro l hl
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot ζ k
      hf : Function.Injective ⇑f
      l : Nat
      hl : Eq (HPow.hPow (f ζ) l) 1
      ⊢ Dvd.dvd (orderOf ζ) l
    -/
    rw [← map_pow, ← map_one f] at hl
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot ζ k
      hf : Function.Injective ⇑f
      l : Nat
      hl : Eq (f (HPow.hPow ζ l)) (f 1)
      ⊢ Dvd.dvd (orderOf ζ) l
    -/
    exact orderOf_dvd_of_pow_eq_one (hf hl)
    /-
      🎉 no goals
    -/


theorem of_map_of_injective [MonoidHomClass F M N] (h : IsPrimitiveRoot (f ζ) k)
    (hf : Injective f) : IsPrimitiveRoot ζ k where
                   /-
                     M : Type u_1
                     N : Type u_2
                     F : Type u_6
                     inst✝³ : CommMonoid M
                     inst✝² : CommMonoid N
                     k : Nat
                     ζ : M
                     f : F
                     inst✝¹ : FunLike F M N
                     inst✝ : MonoidHomClass F M N
                     h : IsPrimitiveRoot (f ζ) k
                     hf : Function.Injective ⇑f
                     ⊢ Eq (HPow.hPow ζ k) 1
                   -/
  pow_eq_one := by apply_fun f; rw [map_pow, map_one, h.pow_eq_one]
                                /-
                                  🎉 no goals
                                -/
  dvd_of_pow_eq_one := by
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot (f ζ) k
      hf : Function.Injective ⇑f
      ⊢ ∀ (l : Nat), Eq (HPow.hPow ζ l) 1 → Dvd.dvd k l
    -/
    rw [h.eq_orderOf]
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot (f ζ) k
      hf : Function.Injective ⇑f
      ⊢ ∀ (l : Nat), Eq (HPow.hPow ζ l) 1 → Dvd.dvd (orderOf (f ζ)) l
    -/
    intro l hl
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot (f ζ) k
      hf : Function.Injective ⇑f
      l : Nat
      hl : Eq (HPow.hPow ζ l) 1
      ⊢ Dvd.dvd (orderOf (f ζ)) l
    -/
    apply_fun f at hl
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot (f ζ) k
      hf : Function.Injective ⇑f
      l : Nat
      hl : Eq (f (HPow.hPow ζ l)) (f 1)
      ⊢ Dvd.dvd (orderOf (f ζ)) l
    -/
    rw [map_pow, map_one] at hl
    /-
      M : Type u_1
      N : Type u_2
      F : Type u_6
      inst✝³ : CommMonoid M
      inst✝² : CommMonoid N
      k : Nat
      ζ : M
      f : F
      inst✝¹ : FunLike F M N
      inst✝ : MonoidHomClass F M N
      h : IsPrimitiveRoot (f ζ) k
      hf : Function.Injective ⇑f
      l : Nat
      hl : Eq (HPow.hPow (f ζ) l) 1
      ⊢ Dvd.dvd (orderOf (f ζ)) l
    -/
    exact orderOf_dvd_of_pow_eq_one hl
    /-
      🎉 no goals
    -/


theorem map_iff_of_injective [MonoidHomClass F M N] (hf : Injective f) :
    IsPrimitiveRoot (f ζ) k ↔ IsPrimitiveRoot ζ k :=
  ⟨fun h => h.of_map_of_injective hf, fun h => h.map_of_injective hf⟩


theorem zero [Nontrivial M₀] : IsPrimitiveRoot (0 : M₀) 0 :=
                             /-
                               M₀ : Type u_7
                               inst✝¹ : CommMonoidWithZero M₀
                               inst✝ : Nontrivial M₀
                               l : Nat
                               hl : Eq (HPow.hPow 0 l) 1
                               ⊢ Dvd.dvd 0 l
                             -/
  ⟨pow_zero 0, fun l hl ↦ by simpa [zero_pow_eq] using hl⟩
                             /-
                               🎉 no goals
                             -/


protected theorem ne_zero [Nontrivial M₀] {ζ : M₀} (h : IsPrimitiveRoot ζ k) : k ≠ 0 → ζ ≠ 0 :=
  mt fun hn ↦ h.unique (hn.symm ▸ IsPrimitiveRoot.zero)


lemma injOn_pow_mul {n : ℕ} {ζ : M₀} (hζ : IsPrimitiveRoot ζ n) {α : M₀} (hα : α ≠ 0) :
    Set.InjOn (ζ ^ · * α) (Finset.range n) :=
  fun i hi j hj e ↦
                           /-
                             M₀ : Type u_7
                             inst✝ : CancelCommMonoidWithZero M₀
                             n : Nat
                             ζ : M₀
                             hζ : IsPrimitiveRoot ζ n
                             α : M₀
                             hα : Ne α 0
                             i : Nat
                             hi : Membership.mem (↑(Finset.range n)) i
                             j : Nat
                             hj : Membership.mem (↑(Finset.range n)) j
                             e : Eq ((fun x => HMul.hMul (HPow.hPow ζ x) α) i) ((fun x => HMul.hMul (HPow.h …
                             ⊢ Eq ((fun x => HPow.hPow ζ x) i) ((fun x => HPow.hPow ζ x) j)
                           -/
    hζ.injOn_pow hi hj (by simpa [mul_eq_mul_right_iff, or_iff_left hα] using e)
                           /-
                             🎉 no goals
                           -/


theorem zpow_eq_one (h : IsPrimitiveRoot ζ k) : ζ ^ (k : ℤ) = 1 := by
  /-
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    ⊢ Eq (HPow.hPow ζ ↑k) 1
  -/
  exact_mod_cast h.pow_eq_one
  /-
    🎉 no goals
  -/


theorem zpow_eq_one_iff_dvd (h : IsPrimitiveRoot ζ k) (l : ℤ) : ζ ^ l = 1 ↔ (k : ℤ) ∣ l := by
  /-
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    l : Int
    ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd (↑k) l)
  -/
  by_cases h0 : 0 ≤ l
    /-
      case pos
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : LE.le 0 l
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd (↑k) l)
    -/
  · lift l to ℕ using h0; exact_mod_cast h.pow_eq_one_iff_dvd l
                          /-
                            🎉 no goals
                          -/
    /-
      case neg
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : Not (LE.le 0 l)
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd (↑k) l)
    -/
  · have : 0 ≤ -l := (Int.neg_pos_of_neg <| Int.lt_of_not_ge h0).le
    /-
      case neg
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : Not (LE.le 0 l)
      this : LE.le 0 (Neg.neg l)
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd (↑k) l)
    -/
    lift -l to ℕ using this with l' hl'
    /-
      case neg.intro
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : Not (LE.le 0 l)
      l' : Nat
      hl' : Eq (↑l') (Neg.neg l)
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd (↑k) l)
    -/
    rw [← dvd_neg, ← hl']
    /-
      case neg.intro
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : Not (LE.le 0 l)
      l' : Nat
      hl' : Eq (↑l') (Neg.neg l)
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd ↑k ↑l')
    -/
    norm_cast
    /-
      case neg.intro
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      l : Int
      h0 : Not (LE.le 0 l)
      l' : Nat
      hl' : Eq (↑l') (Neg.neg l)
      ⊢ Iff (Eq (HPow.hPow ζ l) 1) (Dvd.dvd k l')
    -/
    rw [← h.pow_eq_one_iff_dvd, ← inv_inj, ← zpow_neg, ← hl', zpow_natCast, inv_one]
    /-
      🎉 no goals
    -/


theorem inv (h : IsPrimitiveRoot ζ k) : IsPrimitiveRoot ζ⁻¹ k :=
                     /-
                       G : Type u_3
                       inst✝ : DivisionCommMonoid G
                       k : Nat
                       ζ : G
                       h : IsPrimitiveRoot ζ k
                       ⊢ Eq (HPow.hPow (Inv.inv ζ) k) 1
                     -/
  { pow_eq_one := by simp only [h.pow_eq_one, inv_one, eq_self_iff_true, inv_pow]
                     /-
                       🎉 no goals
                     -/
    dvd_of_pow_eq_one := by
      /-
        G : Type u_3
        inst✝ : DivisionCommMonoid G
        k : Nat
        ζ : G
        h : IsPrimitiveRoot ζ k
        ⊢ ∀ (l : Nat), Eq (HPow.hPow (Inv.inv ζ) l) 1 → Dvd.dvd k l
      -/
      intro l hl
      /-
        G : Type u_3
        inst✝ : DivisionCommMonoid G
        k : Nat
        ζ : G
        h : IsPrimitiveRoot ζ k
        l : Nat
        hl : Eq (HPow.hPow (Inv.inv ζ) l) 1
        ⊢ Dvd.dvd k l
      -/
      apply h.dvd_of_pow_eq_one l
      /-
        G : Type u_3
        inst✝ : DivisionCommMonoid G
        k : Nat
        ζ : G
        h : IsPrimitiveRoot ζ k
        l : Nat
        hl : Eq (HPow.hPow (Inv.inv ζ) l) 1
        ⊢ Eq (HPow.hPow ζ l) 1
      -/
      rw [← inv_inj, ← inv_pow, hl, inv_one] }
      /-
        🎉 no goals
      -/


@[simp]
theorem inv_iff : IsPrimitiveRoot ζ⁻¹ k ↔ IsPrimitiveRoot ζ k :=
  ⟨fun h ↦ inv_inv ζ ▸ inv h, fun h ↦ inv h⟩


theorem zpow_of_gcd_eq_one (h : IsPrimitiveRoot ζ k) (i : ℤ) (hi : i.gcd k = 1) :
    IsPrimitiveRoot (ζ ^ i) k := by
  /-
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  by_cases h0 : 0 ≤ i
    /-
      case pos
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      i : Int
      hi : Eq (i.gcd ↑k) 1
      h0 : LE.le 0 i
      ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
    -/
  · lift i to ℕ using h0
    /-
      case pos.intro
      G : Type u_3
      inst✝ : DivisionCommMonoid G
      k : Nat
      ζ : G
      h : IsPrimitiveRoot ζ k
      i : Nat
      hi : Eq ((↑i).gcd ↑k) 1
      ⊢ IsPrimitiveRoot (HPow.hPow ζ ↑i) k
    -/
    exact_mod_cast h.pow_of_coprime i hi
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    h0 : Not (LE.le 0 i)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  have : 0 ≤ -i := (Int.neg_pos_of_neg <| Int.lt_of_not_ge h0).le
  /-
    case neg
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    h0 : Not (LE.le 0 i)
    this : LE.le 0 (Neg.neg i)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  lift -i to ℕ using this with i' hi'
  /-
    case neg.intro
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    h0 : Not (LE.le 0 i)
    i' : Nat
    hi' : Eq (↑i') (Neg.neg i)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
  -/
  rw [← inv_iff, ← zpow_neg, ← hi', zpow_natCast]
  /-
    case neg.intro
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    h0 : Not (LE.le 0 i)
    i' : Nat
    hi' : Eq (↑i') (Neg.neg i)
    ⊢ IsPrimitiveRoot (HPow.hPow ζ i') k
  -/
  apply h.pow_of_coprime
  /-
    case neg.intro.hi
    G : Type u_3
    inst✝ : DivisionCommMonoid G
    k : Nat
    ζ : G
    h : IsPrimitiveRoot ζ k
    i : Int
    hi : Eq (i.gcd ↑k) 1
    h0 : Not (LE.le 0 i)
    i' : Nat
    hi' : Eq (↑i') (Neg.neg i)
    ⊢ i'.Coprime k
  -/
  rwa [Int.gcd, ← Int.natAbs_neg, ← hi'] at hi
  /-
    🎉 no goals
  -/


theorem sub_one_ne_zero (hn : 1 < n) (hζ : IsPrimitiveRoot ζ n) : ζ - 1 ≠ 0 :=
  sub_ne_zero.mpr <| hζ.ne_one hn


@[simp]
theorem primitiveRoots_one : primitiveRoots 1 R = {(1 : R)} := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq (primitiveRoots 1 R) (Singleton.singleton 1)
  -/
  refine Finset.eq_singleton_iff_unique_mem.2 ⟨?_, fun x hx ↦ ?_⟩
    /-
      case refine_1
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ⊢ Membership.mem (primitiveRoots 1 R) 1
    -/
  · simp only [IsPrimitiveRoot.one_right_iff, mem_primitiveRoots zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      x : R
      hx : Membership.mem (primitiveRoots 1 R) x
      ⊢ Eq x 1
    -/
  · rwa [mem_primitiveRoots zero_lt_one, IsPrimitiveRoot.one_right_iff] at hx
    /-
      🎉 no goals
    -/


theorem neZero' {n : ℕ} [NeZero n] (hζ : IsPrimitiveRoot ζ n) : NeZero ((n : ℕ) : R) := by
  /-
    R : Type u_4
    ζ : R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    ⊢ NeZero ↑n
  -/
  let p := ringChar R
  /-
    R : Type u_4
    ζ : R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    p : Nat := ringChar R
    ⊢ NeZero ↑n
  -/
  have hfin := Nat.finiteMultiplicity_iff.2 ⟨CharP.char_ne_one R p, NeZero.pos n⟩
  /-
    R : Type u_4
    ζ : R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    p : Nat := ringChar R
    hfin : FiniteMultiplicity p n
    ⊢ NeZero ↑n
  -/
  obtain ⟨m, hm⟩ := hfin.exists_eq_pow_mul_and_not_dvd
  /-
    case intro
    R : Type u_4
    ζ : R
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    p : Nat := ringChar R
    hfin : FiniteMultiplicity p n
    m : Nat
    hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
    ⊢ NeZero ↑n
  -/
  by_cases hp : p ∣ n
    /-
      case pos
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      ⊢ NeZero ↑n
    -/
  · obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero (multiplicity_pos_of_dvd hp).ne'
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      ⊢ NeZero ↑n
    -/
    have : NeZero p := NeZero.of_pos (Nat.pos_of_dvd_of_pos hp (NeZero.pos n))
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this : NeZero p
      ⊢ NeZero ↑n
    -/
    have hpri : Fact p.Prime := CharP.char_is_prime_of_pos R p
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this : NeZero p
      hpri : Fact (Nat.Prime p)
      ⊢ NeZero ↑n
    -/
    have := hζ.pow_eq_one
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this✝ : NeZero p
      hpri : Fact (Nat.Prime p)
      this : Eq (HPow.hPow ζ n) 1
      ⊢ NeZero ↑n
    -/
    rw [hm.1, hk, pow_succ', mul_assoc, pow_mul', ← frobenius_def, ← frobenius_one p] at this
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this✝ : NeZero p
      hpri : Fact (Nat.Prime p)
      this : Eq ((frobenius R p) (HPow.hPow ζ (HMul.hMul (HPow.hPow p k) m))) ((frob …
      ⊢ NeZero ↑n
    -/
    exfalso
    have hpos : 0 < p ^ k * m :=
      mul_pos (pow_pos hpri.1.pos _) <| Nat.pos_of_ne_zero (fun H ↦ hm.2 <| H ▸ p.dvd_zero)
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this✝ : NeZero p
      hpri : Fact (Nat.Prime p)
      this : Eq ((frobenius R p) (HPow.hPow ζ (HMul.hMul (HPow.hPow p k) m))) ((frob …
      hpos : LT.lt 0 (HMul.hMul (HPow.hPow p k) m)
      ⊢ False
    -/
    refine hζ.pow_ne_one_of_pos_of_lt hpos ?_ (frobenius_inj R p this)
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this✝ : NeZero p
      hpri : Fact (Nat.Prime p)
      this : Eq ((frobenius R p) (HPow.hPow ζ (HMul.hMul (HPow.hPow p k) m))) ((frob …
      hpos : LT.lt 0 (HMul.hMul (HPow.hPow p k) m)
      ⊢ LT.lt (HMul.hMul (HPow.hPow p k) m) n
    -/
    rw [hm.1, hk, pow_succ', mul_assoc, mul_comm p]
    /-
      case pos.intro
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Dvd.dvd p n
      k : Nat
      hk : Eq (multiplicity p n) k.succ
      this✝ : NeZero p
      hpri : Fact (Nat.Prime p)
      this : Eq ((frobenius R p) (HPow.hPow ζ (HMul.hMul (HPow.hPow p k) m))) ((frob …
      hpos : LT.lt 0 (HMul.hMul (HPow.hPow p k) m)
      ⊢ LT.lt (HMul.hMul (HPow.hPow p k) m) (HMul.hMul (HMul.hMul (HPow.hPow p k) m) …
    -/
    exact lt_mul_of_one_lt_right hpos hpri.1.one_lt
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_4
      ζ : R
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      n : Nat
      inst✝ : NeZero n
      hζ : IsPrimitiveRoot ζ n
      p : Nat := ringChar R
      hfin : FiniteMultiplicity p n
      m : Nat
      hm : And (Eq n (HMul.hMul (HPow.hPow p (multiplicity p n)) m)) (Not (Dvd.dvd p …
      hp : Not (Dvd.dvd p n)
      ⊢ NeZero ↑n
    -/
  · exact NeZero.of_not_dvd R hp
    /-
      🎉 no goals
    -/


nonrec theorem mem_nthRootsFinset (hζ : IsPrimitiveRoot ζ k) (hk : 0 < k) :
    ζ ∈ nthRootsFinset k R :=
  (mem_nthRootsFinset hk).2 hζ.pow_eq_one


theorem eq_neg_one_of_two_right [NoZeroDivisors R] {ζ : R} (h : IsPrimitiveRoot ζ 2) : ζ = -1 :=
  (sq_eq_one_iff.mp h.pow_eq_one).resolve_left <| ne_one h one_lt_two


theorem neg_one (p : ℕ) [Nontrivial R] [h : CharP R p] (hp : p ≠ 2) :
    IsPrimitiveRoot (-1 : R) 2 := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : Nontrivial R
    h : CharP R p
    hp : Ne p 2
    ⊢ IsPrimitiveRoot (-1) 2
  -/
  convert IsPrimitiveRoot.orderOf (-1 : R)
  /-
    case h.e'_4
    R : Type u_4
    inst✝¹ : CommRing R
    p : Nat
    inst✝ : Nontrivial R
    h : CharP R p
    hp : Ne p 2
    ⊢ Eq 2 (orderOf (-1))
  -/
  rw [orderOf_neg_one, if_neg <| by rwa [ringChar.eq_iff.mpr h]]
  /-
    🎉 no goals
  -/


/-- If `1 < k` then `(∑ i ∈ range k, ζ ^ i) = 0`. -/
theorem geom_sum_eq_zero [IsDomain R] {ζ : R} (hζ : IsPrimitiveRoot ζ k) (hk : 1 < k) :
    ∑ i ∈ range k, ζ ^ i = 0 := by
  /-
    R : Type u_4
    k : Nat
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    hζ : IsPrimitiveRoot ζ k
    hk : LT.lt 1 k
    ⊢ Eq ((Finset.range k).sum fun i => HPow.hPow ζ i) 0
  -/
  refine eq_zero_of_ne_zero_of_mul_left_eq_zero (sub_ne_zero_of_ne (hζ.ne_one hk).symm) ?_
  /-
    R : Type u_4
    k : Nat
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    hζ : IsPrimitiveRoot ζ k
    hk : LT.lt 1 k
    ⊢ Eq (HMul.hMul (HSub.hSub 1 ζ) ((Finset.range k).sum fun i => HPow.hPow ζ i)) 0
  -/
  rw [mul_neg_geom_sum, hζ.pow_eq_one, sub_self]
  /-
    🎉 no goals
  -/


/-- If `1 < k`, then `ζ ^ k.pred = -(∑ i ∈ range k.pred, ζ ^ i)`. -/
theorem pow_sub_one_eq [IsDomain R] {ζ : R} (hζ : IsPrimitiveRoot ζ k) (hk : 1 < k) :
    ζ ^ k.pred = -∑ i ∈ range k.pred, ζ ^ i := by
  rw [eq_neg_iff_add_eq_zero, add_comm, ← sum_range_succ, ← Nat.succ_eq_add_one,
    Nat.succ_pred_eq_of_pos (pos_of_gt hk), hζ.geom_sum_eq_zero hk]


/-- The (additive) monoid equivalence between `ZMod k`
and the powers of a primitive root of unity `ζ`. -/
def zmodEquivZPowers (h : IsPrimitiveRoot ζ k) : ZMod k ≃+ Additive (Subgroup.zpowers ζ) :=
  AddEquiv.ofBijective
    (AddMonoidHom.liftOfRightInverse (Int.castAddHom <| ZMod k) _ ZMod.intCast_rightInverse
      ⟨{  toFun := fun i ↦ Additive.ofMul (⟨_, i, rfl⟩ : Subgroup.zpowers ζ)
                          /-
                            M : Type u_1
                            N : Type u_2
                            G : Type u_3
                            R : Type u_4
                            S : Type u_5
                            F : Type u_6
                            inst✝³ : CommMonoid M
                            inst✝² : CommMonoid N
                            inst✝¹ : DivisionCommMonoid G
                            k l : Nat
                            inst✝ : CommRing R
                            ζ : Units R
                            h✝ h : IsPrimitiveRoot ζ k
                            ⊢ Eq ((fun i => Additive.ofMul ⟨(fun x => HPow.hPow ζ x) i, ⋯⟩) 0) 0
                          -/
          map_zero' := by simp only [zpow_zero]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/
                         /-
                           M : Type u_1
                           N : Type u_2
                           G : Type u_3
                           R : Type u_4
                           S : Type u_5
                           F : Type u_6
                           inst✝³ : CommMonoid M
                           inst✝² : CommMonoid N
                           inst✝¹ : DivisionCommMonoid G
                           k l : Nat
                           inst✝ : CommRing R
                           ζ : Units R
                           h✝ h : IsPrimitiveRoot ζ k
                           ⊢ ∀ (x y : Int), Eq ({ toFun := fun i => Additive.ofMul ⟨(fun x => HPow.hPow ζ …
                         -/
          map_add' := by intro i j; simp only [zpow_add]; rfl }, fun i hi ↦ by
                                                          /-
                                                            🎉 no goals
                                                          -/
        simp only [AddMonoidHom.mem_ker, CharP.intCast_eq_zero_iff (ZMod k) k, AddMonoidHom.coe_mk,
          Int.coe_castAddHom] at hi ⊢
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : Int
          hi : Dvd.dvd (↑k) i
          ⊢ Eq ({ toFun := fun i => Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩, map_zero' := ⋯ }  …
        -/
        obtain ⟨i, rfl⟩ := hi
        /-
          case intro
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : Int
          ⊢ Eq ({ toFun := fun i => Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩, map_zero' := ⋯ }  …
        -/
        simp [zpow_mul, h.pow_eq_one, one_zpow, zpow_natCast]⟩)
        /-
          🎉 no goals
        -/
    (by
      /-
        M : Type u_1
        N : Type u_2
        G : Type u_3
        R : Type u_4
        S : Type u_5
        F : Type u_6
        inst✝³ : CommMonoid M
        inst✝² : CommMonoid N
        inst✝¹ : DivisionCommMonoid G
        k l : Nat
        inst✝ : CommRing R
        ζ : Units R
        h✝ h : IsPrimitiveRoot ζ k
        ⊢ Function.Bijective ⇑(((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast …
      -/
      constructor
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          ⊢ Function.Injective ⇑(((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast …
        -/
      · rw [injective_iff_map_eq_zero]
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          ⊢ ∀ (a : ZMod k), Eq ((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast …
        -/
        intro i hi
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : ZMod k
          hi : Eq ((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun  …
          ⊢ Eq i 0
        -/
        rw [Subtype.ext_iff] at hi
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : ZMod k
          hi : Eq ↑((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun …
          ⊢ Eq i 0
        -/
        have := (h.zpow_eq_one_iff_dvd _).mp hi
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : ZMod k
          hi : Eq ↑((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun …
          this : Dvd.dvd (↑k) i.cast
          ⊢ Eq i 0
        -/
        rw [← (CharP.intCast_eq_zero_iff (ZMod k) k _).mpr this, eq_comm]
        /-
          case left
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : ZMod k
          hi : Eq ↑((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun …
          this : Dvd.dvd (↑k) i.cast
          ⊢ Eq (↑i.cast) i
        -/
        exact ZMod.intCast_rightInverse i
        /-
          🎉 no goals
        -/
        /-
          case right
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          ⊢ Function.Surjective ⇑(((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cas …
        -/
      · rintro ⟨ξ, i, rfl⟩
        /-
          case right.mk.intro
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : Int
          ⊢ Exists fun a => Eq ((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast …
        -/
        refine ⟨Int.castAddHom (ZMod k) i, ?_⟩
        /-
          case right.mk.intro
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : Int
          ⊢ Eq ((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun :=  …
        -/
        rw [AddMonoidHom.liftOfRightInverse_comp_apply]
        /-
          case right.mk.intro
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝³ : CommMonoid M
          inst✝² : CommMonoid N
          inst✝¹ : DivisionCommMonoid G
          k l : Nat
          inst✝ : CommRing R
          ζ : Units R
          h✝ h : IsPrimitiveRoot ζ k
          i : Int
          ⊢ Eq (↑⟨{ toFun := fun i => Additive.ofMul ⟨(fun x => HPow.hPow ζ x) i, ⋯⟩, ma …
        -/
        rfl)
        /-
          🎉 no goals
        -/


@[simp]
theorem zmodEquivZPowers_apply_coe_int (i : ℤ) :
    h.zmodEquivZPowers i = Additive.ofMul (⟨ζ ^ i, i, rfl⟩ : Subgroup.zpowers ζ) := by
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Int
    ⊢ Eq (h.zmodEquivZPowers ↑i) (Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩)
  -/
  rw [zmodEquivZPowers, AddEquiv.ofBijective_apply] -- Porting note: Original proof didn't have `rw`
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Int
    ⊢ Eq ((((Int.castAddHom (ZMod k)).liftOfRightInverse ZMod.cast ⋯) ⟨{ toFun :=  …
  -/
  exact AddMonoidHom.liftOfRightInverse_comp_apply _ _ ZMod.intCast_rightInverse _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem zmodEquivZPowers_apply_coe_nat (i : ℕ) :
    h.zmodEquivZPowers i = Additive.ofMul (⟨ζ ^ i, i, rfl⟩ : Subgroup.zpowers ζ) := by
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Nat
    ⊢ Eq (h.zmodEquivZPowers ↑i) (Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩)
  -/
  have : (i : ZMod k) = (i : ℤ) := by norm_cast
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Nat
    this : Eq ↑i ↑↑i
    ⊢ Eq (h.zmodEquivZPowers ↑i) (Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩)
  -/
  simp only [this, zmodEquivZPowers_apply_coe_int, zpow_natCast]
  /-
    🎉 no goals
  -/


@[simp]
theorem zmodEquivZPowers_symm_apply_zpow (i : ℤ) :
    h.zmodEquivZPowers.symm (Additive.ofMul (⟨ζ ^ i, i, rfl⟩ : Subgroup.zpowers ζ)) = i := by
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Int
    ⊢ Eq (h.zmodEquivZPowers.symm (Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩)) ↑i
  -/
  rw [← h.zmodEquivZPowers.symm_apply_apply i, zmodEquivZPowers_apply_coe_int]
  /-
    🎉 no goals
  -/


@[simp]
theorem zmodEquivZPowers_symm_apply_zpow' (i : ℤ) : h.zmodEquivZPowers.symm ⟨ζ ^ i, i, rfl⟩ = i :=
  h.zmodEquivZPowers_symm_apply_zpow i


@[simp]
theorem zmodEquivZPowers_symm_apply_pow (i : ℕ) :
    h.zmodEquivZPowers.symm (Additive.ofMul (⟨ζ ^ i, i, rfl⟩ : Subgroup.zpowers ζ)) = i := by
  /-
    R : Type u_4
    k : Nat
    inst✝ : CommRing R
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    i : Nat
    ⊢ Eq (h.zmodEquivZPowers.symm (Additive.ofMul ⟨HPow.hPow ζ i, ⋯⟩)) ↑i
  -/
  rw [← h.zmodEquivZPowers.symm_apply_apply i, zmodEquivZPowers_apply_coe_nat]
  /-
    🎉 no goals
  -/


@[simp]
theorem zmodEquivZPowers_symm_apply_pow' (i : ℕ) : h.zmodEquivZPowers.symm ⟨ζ ^ i, i, rfl⟩ = i :=
  h.zmodEquivZPowers_symm_apply_pow i


theorem zpowers_eq {k : ℕ} [NeZero k] {ζ : Rˣ} (h : IsPrimitiveRoot ζ k) :
    Subgroup.zpowers ζ = rootsOfUnity k R := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    ⊢ Eq (Subgroup.zpowers ζ) (rootsOfUnity k R)
  -/
  apply SetLike.coe_injective
  /-
    case a
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    ⊢ Eq ↑(Subgroup.zpowers ζ) ↑(rootsOfUnity k R)
  -/
  have F : Fintype (Subgroup.zpowers ζ) := Fintype.ofEquiv _ h.zmodEquivZPowers.toEquiv
  refine
    @Set.eq_of_subset_of_card_le Rˣ _ _ F (rootsOfUnity.fintype R k)
      (Subgroup.zpowers_le_of_mem <| show ζ ∈ rootsOfUnity k R from h.pow_eq_one) ?_
  calc
    Fintype.card (rootsOfUnity k R) ≤ k := card_rootsOfUnity R k
    _ = Fintype.card (ZMod k) := (ZMod.card k).symm
    _ = Fintype.card (Subgroup.zpowers ζ) := Fintype.card_congr h.zmodEquivZPowers.toEquiv


lemma map_rootsOfUnity {S F} [CommRing S] [IsDomain S] [FunLike F R S] [MonoidHomClass F R S]
    {ζ : R} {n : ℕ} [NeZero n] (hζ : IsPrimitiveRoot ζ n) {f : F} (hf : Function.Injective f) :
    (rootsOfUnity n R).map (Units.map f) = rootsOfUnity n S := by
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    S : Type u_7
    F : Type u_8
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : FunLike F R S
    inst✝¹ : MonoidHomClass F R S
    ζ : R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    f : F
    hf : Function.Injective ⇑f
    ⊢ Eq (Subgroup.map (Units.map ↑f) (rootsOfUnity n R)) (rootsOfUnity n S)
  -/
  letI : CommMonoid Sˣ := inferInstance
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    S : Type u_7
    F : Type u_8
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : FunLike F R S
    inst✝¹ : MonoidHomClass F R S
    ζ : R
    n : Nat
    inst✝ : NeZero n
    hζ : IsPrimitiveRoot ζ n
    f : F
    hf : Function.Injective ⇑f
    this : CommMonoid (Units S) := inferInstance
    ⊢ Eq (Subgroup.map (Units.map ↑f) (rootsOfUnity n R)) (rootsOfUnity n S)
  -/
  replace hζ := hζ.isUnit_unit <| NeZero.pos n
  rw [← hζ.zpowers_eq,
    ← (hζ.map_of_injective (Units.map_injective (f := (f : R →* S)) hf)).zpowers_eq,
    MonoidHom.map_zpowers]


/-- If `R` contains an `n`-th primitive root, and `S/R` is a ring extension,
then the `n`-th roots of unity in `R` and `S` are isomorphic.
Also see `IsPrimitiveRoot.map_rootsOfUnity` for the equality as `Subgroup Sˣ`. -/
@[simps! (config := .lemmasOnly) apply_coe_val apply_coe_inv_val]
noncomputable
def _root_.rootsOfUnityEquivOfPrimitiveRoots {S F} [CommRing S] [IsDomain S]
    [FunLike F R S] [MonoidHomClass F R S]
    {n : ℕ} [NeZero n] {f : F} (hf : Function.Injective f) (hζ : (primitiveRoots n R).Nonempty) :
    (rootsOfUnity n R) ≃* rootsOfUnity n S :=
  (Subgroup.equivMapOfInjective _ (Units.map f) (Units.map_injective hf)).trans
    (MulEquiv.subgroupCongr <|
      ((mem_primitiveRoots <| NeZero.pos n).mp hζ.choose_spec).map_rootsOfUnity hf)


lemma _root_.rootsOfUnityEquivOfPrimitiveRoots_symm_apply
    {S F} [CommRing S] [IsDomain S] [FunLike F R S] [MonoidHomClass F R S] {n : ℕ} [NeZero n]
    {f : F} (hf : Function.Injective f) (hζ : (primitiveRoots n R).Nonempty) (η) :
    f ((rootsOfUnityEquivOfPrimitiveRoots hf hζ).symm η : Rˣ) = (η : Sˣ) := by
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    S : Type u_7
    F : Type u_8
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : FunLike F R S
    inst✝¹ : MonoidHomClass F R S
    n : Nat
    inst✝ : NeZero n
    f : F
    hf : Function.Injective ⇑f
    hζ : (primitiveRoots n R).Nonempty
    η : Subtype fun x => Membership.mem (rootsOfUnity n S) x
    ⊢ Eq (f ↑↑((rootsOfUnityEquivOfPrimitiveRoots hf hζ).symm η)) ↑↑η
  -/
  obtain ⟨ε, rfl⟩ := (rootsOfUnityEquivOfPrimitiveRoots hf hζ).surjective η
  /-
    case intro
    R : Type u_4
    inst✝⁶ : CommRing R
    inst✝⁵ : IsDomain R
    S : Type u_7
    F : Type u_8
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    inst✝² : FunLike F R S
    inst✝¹ : MonoidHomClass F R S
    n : Nat
    inst✝ : NeZero n
    f : F
    hf : Function.Injective ⇑f
    hζ : (primitiveRoots n R).Nonempty
    ε : Subtype fun x => Membership.mem (rootsOfUnity n R) x
    ⊢ Eq (f ↑↑((rootsOfUnityEquivOfPrimitiveRoots hf hζ).symm ((rootsOfUnityEquivO …
  -/
  rw [MulEquiv.symm_apply_apply, val_rootsOfUnityEquivOfPrimitiveRoots_apply_coe]
  /-
    🎉 no goals
  -/

-- Porting note: rephrased the next few lemmas to avoid `∃ (Prop)`

theorem eq_pow_of_mem_rootsOfUnity {k : ℕ} [NeZero k] {ζ ξ : Rˣ} (h : IsPrimitiveRoot ζ k)
    (hξ : ξ ∈ rootsOfUnity k R) : ∃ i < k, ζ ^ i = ξ := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ ξ : Units R
    h : IsPrimitiveRoot ζ k
    hξ : Membership.mem (rootsOfUnity k R) ξ
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) ξ)
  -/
  obtain ⟨n, rfl⟩ : ∃ n : ℤ, ζ ^ n = ξ := by rwa [← h.zpowers_eq] at hξ
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    n : Int
    hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) (HPow.hPow ζ n))
  -/
  have hk0 : (0 : ℤ) < k := mod_cast NeZero.pos k
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    n : Int
    hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
    hk0 : LT.lt 0 ↑k
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) (HPow.hPow ζ n))
  -/
  let i := n % k
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    n : Int
    hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
    hk0 : LT.lt 0 ↑k
    i : Int := HMod.hMod n ↑k
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) (HPow.hPow ζ n))
  -/
  have hi0 : 0 ≤ i := Int.emod_nonneg _ (ne_of_gt hk0)
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    n : Int
    hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
    hk0 : LT.lt 0 ↑k
    i : Int := HMod.hMod n ↑k
    hi0 : LE.le 0 i
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) (HPow.hPow ζ n))
  -/
  lift i to ℕ using hi0 with i₀ hi₀
  /-
    case intro.intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    n : Int
    hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
    hk0 : LT.lt 0 ↑k
    i : Int := HMod.hMod n ↑k
    i₀ : Nat
    hi₀ : Eq (↑i₀) i
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) (HPow.hPow ζ n))
  -/
  refine ⟨i₀, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ : Units R
      h : IsPrimitiveRoot ζ k
      n : Int
      hξ : Membership.mem (rootsOfUnity k R) (HPow.hPow ζ n)
      hk0 : LT.lt 0 ↑k
      i : Int := HMod.hMod n ↑k
      i₀ : Nat
      hi₀ : Eq (↑i₀) i
      ⊢ LT.lt i₀ k
    -/
  · zify; rw [hi₀]; exact Int.emod_lt_of_pos _ hk0
                    /-
                      🎉 no goals
                    -/
  · rw [← zpow_natCast, hi₀, ← Int.emod_add_ediv n k, zpow_add, zpow_mul, h.zpow_eq_one, one_zpow,
      mul_one]


theorem eq_pow_of_pow_eq_one {k : ℕ} [NeZero k] {ζ ξ : R} (h : IsPrimitiveRoot ζ k)
    (hξ : ξ ^ k = 1) :
    ∃ i < k, ζ ^ i = ξ := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ ξ : R
    h : IsPrimitiveRoot ζ k
    hξ : Eq (HPow.hPow ξ k) 1
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) ξ)
  -/
  lift ζ to Rˣ using h.isUnit <| NeZero.pos k
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ξ : R
    hξ : Eq (HPow.hPow ξ k) 1
    ζ : Units R
    h : IsPrimitiveRoot (↑ζ) k
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow (↑ζ) i) ξ)
  -/
  lift ξ to Rˣ using isUnit_ofPowEqOne hξ <| NeZero.ne k
  /-
    case intro.intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot (↑ζ) k
    ξ : Units R
    hξ : Eq (HPow.hPow (↑ξ) k) 1
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow (↑ζ) i) ↑ξ)
  -/
  simp only [← Units.val_pow_eq_pow_val, ← Units.ext_iff]
  /-
    case intro.intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot (↑ζ) k
    ξ : Units R
    hξ : Eq (HPow.hPow (↑ξ) k) 1
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) ξ)
  -/
  rw [coe_units_iff] at h
  /-
    case intro.intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ : Units R
    h : IsPrimitiveRoot ζ k
    ξ : Units R
    hξ : Eq (HPow.hPow (↑ξ) k) 1
    ⊢ Exists fun i => And (LT.lt i k) (Eq (HPow.hPow ζ i) ξ)
  -/
  exact h.eq_pow_of_mem_rootsOfUnity <| (mem_rootsOfUnity' k ξ).mpr hξ
  /-
    🎉 no goals
  -/


theorem isPrimitiveRoot_iff' {k : ℕ} [NeZero k] {ζ ξ : Rˣ} (h : IsPrimitiveRoot ζ k) :
    IsPrimitiveRoot ξ k ↔ ∃ i < k, i.Coprime k ∧ ζ ^ i = ξ := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ ξ : Units R
    h : IsPrimitiveRoot ζ k
    ⊢ Iff (IsPrimitiveRoot ξ k) (Exists fun i => And (LT.lt i k) (And (i.Coprime k …
  -/
  constructor
    /-
      case mp
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : Units R
      h : IsPrimitiveRoot ζ k
      ⊢ IsPrimitiveRoot ξ k → Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq …
    -/
  · intro hξ
    /-
      case mp
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : Units R
      h : IsPrimitiveRoot ζ k
      hξ : IsPrimitiveRoot ξ k
      ⊢ Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq (HPow.hPow ζ i) ξ))
    -/
    obtain ⟨i, hik, rfl⟩ := h.eq_pow_of_mem_rootsOfUnity hξ.pow_eq_one
    /-
      case mp.intro.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ : Units R
      h : IsPrimitiveRoot ζ k
      i : Nat
      hik : LT.lt i k
      hξ : IsPrimitiveRoot (HPow.hPow ζ i) k
      ⊢ Exists fun i_1 => And (LT.lt i_1 k) (And (i_1.Coprime k) (Eq (HPow.hPow ζ i_ …
    -/
    rw [h.pow_iff_coprime <| NeZero.pos k] at hξ
    /-
      case mp.intro.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ : Units R
      h : IsPrimitiveRoot ζ k
      i : Nat
      hik : LT.lt i k
      hξ : i.Coprime k
      ⊢ Exists fun i_1 => And (LT.lt i_1 k) (And (i_1.Coprime k) (Eq (HPow.hPow ζ i_ …
    -/
    exact ⟨i, hik, hξ, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : Units R
      h : IsPrimitiveRoot ζ k
      ⊢ (Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq (HPow.hPow ζ i) ξ))) …
    -/
  · rintro ⟨i, -, hi, rfl⟩; exact h.pow_of_coprime i hi
                            /-
                              🎉 no goals
                            -/


theorem isPrimitiveRoot_iff {k : ℕ} [NeZero k] {ζ ξ : R} (h : IsPrimitiveRoot ζ k) :
    IsPrimitiveRoot ξ k ↔ ∃ i < k, i.Coprime k ∧ ζ ^ i = ξ := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    k : Nat
    inst✝ : NeZero k
    ζ ξ : R
    h : IsPrimitiveRoot ζ k
    ⊢ Iff (IsPrimitiveRoot ξ k) (Exists fun i => And (LT.lt i k) (And (i.Coprime k …
  -/
  constructor
    /-
      case mp
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : R
      h : IsPrimitiveRoot ζ k
      ⊢ IsPrimitiveRoot ξ k → Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq …
    -/
  · intro hξ
    /-
      case mp
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : R
      h : IsPrimitiveRoot ζ k
      hξ : IsPrimitiveRoot ξ k
      ⊢ Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq (HPow.hPow ζ i) ξ))
    -/
    obtain ⟨i, hik, rfl⟩ := h.eq_pow_of_pow_eq_one hξ.pow_eq_one
    /-
      case mp.intro.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ : R
      h : IsPrimitiveRoot ζ k
      i : Nat
      hik : LT.lt i k
      hξ : IsPrimitiveRoot (HPow.hPow ζ i) k
      ⊢ Exists fun i_1 => And (LT.lt i_1 k) (And (i_1.Coprime k) (Eq (HPow.hPow ζ i_ …
    -/
    rw [h.pow_iff_coprime <| NeZero.pos k] at hξ
    /-
      case mp.intro.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ : R
      h : IsPrimitiveRoot ζ k
      i : Nat
      hik : LT.lt i k
      hξ : i.Coprime k
      ⊢ Exists fun i_1 => And (LT.lt i_1 k) (And (i_1.Coprime k) (Eq (HPow.hPow ζ i_ …
    -/
    exact ⟨i, hik, hξ, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      k : Nat
      inst✝ : NeZero k
      ζ ξ : R
      h : IsPrimitiveRoot ζ k
      ⊢ (Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq (HPow.hPow ζ i) ξ))) …
    -/
  · rintro ⟨i, -, hi, rfl⟩; exact h.pow_of_coprime i hi
                            /-
                              🎉 no goals
                            -/


theorem nthRoots_eq {n : ℕ} {ζ : R} (hζ : IsPrimitiveRoot ζ n) {α a : R} (e : α ^ n = a) :
    nthRoots n a = (Multiset.range n).map (ζ ^ · * α) := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    e : Eq (HPow.hPow α n) a
    ⊢ Eq (Polynomial.nthRoots n a) (Multiset.map (fun x => HMul.hMul (HPow.hPow ζ  …
  -/
  obtain (rfl | hn) := n.eq_zero_or_pos; · simp
                                           /-
                                             🎉 no goals
                                           -/
  /-
    case inr
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    α a : R
    e : Eq (HPow.hPow α n) a
    hn : GT.gt n 0
    ⊢ Eq (Polynomial.nthRoots n a) (Multiset.map (fun x => HMul.hMul (HPow.hPow ζ  …
  -/
  by_cases hα : α = 0
    /-
      case pos
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      α a : R
      e : Eq (HPow.hPow α n) a
      hn : GT.gt n 0
      hα : Eq α 0
      ⊢ Eq (Polynomial.nthRoots n a) (Multiset.map (fun x => HMul.hMul (HPow.hPow ζ  …
    -/
  · rw [hα, zero_pow hn.ne'] at e
    simp only [hα, e.symm, nthRoots_zero_right, mul_zero,
      Finset.range_val, Multiset.map_const', Multiset.card_range]
  classical
  symm; apply Multiset.eq_of_le_of_card_le
  · rw [← Finset.range_val,
      ← Finset.image_val_of_injOn (hζ.injOn_pow_mul hα), Finset.val_le_iff_val_subset]
    intro x hx
    simp only [Finset.image_val, Finset.range_val, Multiset.mem_dedup, Multiset.mem_map,
      Multiset.mem_range] at hx
    obtain ⟨m, _, rfl⟩ := hx
    rw [mem_nthRoots hn, mul_pow, e, ← pow_mul, mul_comm m,
      pow_mul, hζ.pow_eq_one, one_pow, one_mul]
  · simpa only [Multiset.card_map, Multiset.card_range] using card_nthRoots n a


open scoped Classical in
theorem card_nthRoots {n : ℕ} {ζ : R} (hζ : IsPrimitiveRoot ζ n) (a : R) :
    Multiset.card (nthRoots n a) = if ∃ α, α ^ n = a then n else 0 := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    ζ : R
    hζ : IsPrimitiveRoot ζ n
    a : R
    ⊢ Eq (Polynomial.nthRoots n a).card (ite (Exists fun α => Eq (HPow.hPow α n) a …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      a : R
      h : Exists fun α => Eq (HPow.hPow α n) a
      ⊢ Eq (Polynomial.nthRoots n a).card n
    -/
  · obtain ⟨α, hα⟩ := h
    /-
      case pos.intro
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      a α : R
      hα : Eq (HPow.hPow α n) a
      ⊢ Eq (Polynomial.nthRoots n a).card n
    -/
    rw [nthRoots_eq hζ hα, Multiset.card_map, Multiset.card_range]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      a : R
      h : Not (Exists fun α => Eq (HPow.hPow α n) a)
      ⊢ Eq (Polynomial.nthRoots n a).card 0
    -/
  · obtain (rfl|hn) := n.eq_zero_or_pos; · simp
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case neg.inr
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      a : R
      h : Not (Exists fun α => Eq (HPow.hPow α n) a)
      hn : GT.gt n 0
      ⊢ Eq (Polynomial.nthRoots n a).card 0
    -/
    push_neg at h
    /-
      case neg.inr
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      n : Nat
      ζ : R
      hζ : IsPrimitiveRoot ζ n
      a : R
      hn : GT.gt n 0
      h : ∀ (α : R), Ne (HPow.hPow α n) a
      ⊢ Eq (Polynomial.nthRoots n a).card 0
    -/
    simpa only [Multiset.card_eq_zero, Multiset.eq_zero_iff_forall_not_mem, mem_nthRoots hn]
    /-
      🎉 no goals
    -/


/-- A variant of `IsPrimitiveRoot.card_rootsOfUnity` for `ζ : Rˣ`. -/
theorem card_rootsOfUnity' {n : ℕ} [NeZero n] (h : IsPrimitiveRoot ζ n) :
    Fintype.card (rootsOfUnity n R) = n := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    ζ : Units R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (rootsOfUnity n R) x)) n
  -/
  let e := h.zmodEquivZPowers
  /-
    R : Type u_4
    inst✝² : CommRing R
    ζ : Units R
    inst✝¹ : IsDomain R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ n
    e : AddEquiv (ZMod n) (Additive (Subtype fun x => Membership.mem (Subgroup.zpo …
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (rootsOfUnity n R) x)) n
  -/
  have : Fintype (Subgroup.zpowers ζ) := Fintype.ofEquiv _ e.toEquiv
  calc
    Fintype.card (rootsOfUnity n R) = Fintype.card (Subgroup.zpowers ζ) :=
      Fintype.card_congr <| by rw [h.zpowers_eq]
    _ = Fintype.card (ZMod n) := Fintype.card_congr e.toEquiv.symm
    _ = n := ZMod.card n


theorem card_rootsOfUnity {ζ : R} {n : ℕ} [NeZero n] (h : IsPrimitiveRoot ζ n) :
    Fintype.card (rootsOfUnity n R) = n := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    ζ : R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (rootsOfUnity n R) x)) n
  -/
  obtain ⟨ζ, hζ⟩ := h.isUnit <| NeZero.pos n
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    ζ✝ : R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ✝ n
    ζ : Units R
    hζ : Eq (↑ζ) ζ✝
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (rootsOfUnity n R) x)) n
  -/
  rw [← hζ, IsPrimitiveRoot.coe_units_iff] at h
  /-
    case intro
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    ζ✝ : R
    n : Nat
    inst✝ : NeZero n
    ζ : Units R
    h : IsPrimitiveRoot ζ n
    hζ : Eq (↑ζ) ζ✝
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem (rootsOfUnity n R) x)) n
  -/
  exact h.card_rootsOfUnity'
  /-
    🎉 no goals
  -/


/-- The cardinality of the multiset `nthRoots ↑n (1 : R)` is `n`
if there is a primitive root of unity in `R`. -/
theorem card_nthRoots_one {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    Multiset.card (nthRoots n (1 : R)) = n := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.nthRoots n 1).card n
  -/
  rw [card_nthRoots h, if_pos ⟨ζ, h.pow_eq_one⟩]
  /-
    🎉 no goals
  -/


theorem nthRoots_nodup {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) {a : R} (ha : a ≠ 0) :
    (nthRoots n a).Nodup := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    a : R
    ha : Ne a 0
    ⊢ (Polynomial.nthRoots n a).Nodup
  -/
  obtain (rfl | hn) := n.eq_zero_or_pos; · simp
                                           /-
                                             🎉 no goals
                                           -/
  /-
    case inr
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    a : R
    ha : Ne a 0
    hn : GT.gt n 0
    ⊢ (Polynomial.nthRoots n a).Nodup
  -/
  by_cases h : ∃ α, α ^ n = a
    /-
      case pos
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h✝ : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      h : Exists fun α => Eq (HPow.hPow α n) a
      ⊢ (Polynomial.nthRoots n a).Nodup
    -/
  · obtain ⟨α, hα⟩ := h
    /-
      case pos.intro
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      α : R
      hα : Eq (HPow.hPow α n) a
      ⊢ (Polynomial.nthRoots n a).Nodup
    -/
    by_cases hα' : α = 0
      /-
        case pos
        R : Type u_4
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        ζ : R
        n : Nat
        h : IsPrimitiveRoot ζ n
        a : R
        ha : Ne a 0
        hn : GT.gt n 0
        α : R
        hα : Eq (HPow.hPow α n) a
        hα' : Eq α 0
        ⊢ (Polynomial.nthRoots n a).Nodup
      -/
    · exact (ha (by rwa [hα', zero_pow hn.ne', eq_comm] at hα)).elim
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      α : R
      hα : Eq (HPow.hPow α n) a
      hα' : Not (Eq α 0)
      ⊢ (Polynomial.nthRoots n a).Nodup
    -/
    rw [nthRoots_eq h hα, Multiset.nodup_map_iff_inj_on (Multiset.nodup_range n)]
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      α : R
      hα : Eq (HPow.hPow α n) a
      hα' : Not (Eq α 0)
      ⊢ ∀ (x : Nat), Membership.mem (Multiset.range n) x → ∀ (y : Nat), Membership.m …
    -/
    exact h.injOn_pow_mul hα'
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h✝ : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      h : Not (Exists fun α => Eq (HPow.hPow α n) a)
      ⊢ (Polynomial.nthRoots n a).Nodup
    -/
  · suffices nthRoots n a = 0 by simp [this]
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h✝ : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      h : Not (Exists fun α => Eq (HPow.hPow α n) a)
      ⊢ Eq (Polynomial.nthRoots n a) 0
    -/
    push_neg at h
    /-
      case neg
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h✝ : IsPrimitiveRoot ζ n
      a : R
      ha : Ne a 0
      hn : GT.gt n 0
      h : ∀ (α : R), Ne (HPow.hPow α n) a
      ⊢ Eq (Polynomial.nthRoots n a) 0
    -/
    simpa only [Multiset.card_eq_zero, Multiset.eq_zero_iff_forall_not_mem, mem_nthRoots hn]
    /-
      🎉 no goals
    -/


/-- The multiset `nthRoots ↑n (1 : R)` has no repeated elements
if there is a primitive root of unity in `R`. -/
theorem nthRoots_one_nodup {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    (nthRoots n (1 : R)).Nodup :=
  h.nthRoots_nodup one_ne_zero


@[simp]
theorem card_nthRootsFinset {ζ : R} {n : ℕ} (h : IsPrimitiveRoot ζ n) :
    #(nthRootsFinset n R) = n := by
  classical
  rw [nthRootsFinset, ← Multiset.toFinset_eq (nthRoots_one_nodup h), card_mk, h.card_nthRoots_one]


/-- If an integral domain has a primitive `k`-th root of unity, then it has `φ k` of them. -/
theorem card_primitiveRoots {ζ : R} {k : ℕ} (h : IsPrimitiveRoot ζ k) :
    #(primitiveRoots k R) = φ k := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    k : Nat
    h : IsPrimitiveRoot ζ k
    ⊢ Eq (primitiveRoots k R).card k.totient
  -/
  by_cases h0 : k = 0
    /-
      case pos
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Eq k 0
      ⊢ Eq (primitiveRoots k R).card k.totient
    -/
  · simp [h0]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    k : Nat
    h : IsPrimitiveRoot ζ k
    h0 : Not (Eq k 0)
    ⊢ Eq (primitiveRoots k R).card k.totient
  -/
  have : NeZero k := ⟨h0⟩
  /-
    case neg
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    k : Nat
    h : IsPrimitiveRoot ζ k
    h0 : Not (Eq k 0)
    this : NeZero k
    ⊢ Eq (primitiveRoots k R).card k.totient
  -/
  symm
  /-
    case neg
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    k : Nat
    h : IsPrimitiveRoot ζ k
    h0 : Not (Eq k 0)
    this : NeZero k
    ⊢ Eq k.totient (primitiveRoots k R).card
  -/
  refine Finset.card_bij (fun i _ ↦ ζ ^ i) ?_ ?_ ?_
    /-
      case neg.refine_1
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (a : Nat) (ha : Membership.mem (Finset.filter (fun a => k.Coprime a) (Fins …
    -/
  · simp only [and_imp, mem_filter, mem_range, mem_univ]
    /-
      case neg.refine_1
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (a : Nat), LT.lt a k → k.Coprime a → Membership.mem (primitiveRoots k R) ( …
    -/
    rintro i - hi
    /-
      case neg.refine_1
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      i : Nat
      hi : k.Coprime i
      ⊢ Membership.mem (primitiveRoots k R) (HPow.hPow ζ i)
    -/
    rw [mem_primitiveRoots (Nat.pos_of_ne_zero h0)]
    /-
      case neg.refine_1
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      i : Nat
      hi : k.Coprime i
      ⊢ IsPrimitiveRoot (HPow.hPow ζ i) k
    -/
    exact h.pow_of_coprime i hi.symm
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (a₁ : Nat) (ha₁ : Membership.mem (Finset.filter (fun a => k.Coprime a) (Fi …
    -/
  · simp only [and_imp, mem_filter, mem_range, mem_univ]
    /-
      case neg.refine_2
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (a₁ : Nat), LT.lt a₁ k → k.Coprime a₁ → ∀ (a₂ : Nat), LT.lt a₂ k → k.Copri …
    -/
    rintro i hi - j hj - H
    /-
      case neg.refine_2
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      i : Nat
      hi : LT.lt i k
      j : Nat
      hj : LT.lt j k
      H : Eq (HPow.hPow ζ i) (HPow.hPow ζ j)
      ⊢ Eq i j
    -/
    exact h.pow_inj hi hj H
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_3
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (b : R), Membership.mem (primitiveRoots k R) b → Exists fun a => Exists fu …
    -/
  · simp only [exists_prop, mem_filter, mem_range, mem_univ]
    /-
      case neg.refine_3
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ⊢ ∀ (b : R), Membership.mem (primitiveRoots k R) b → Exists fun a => And (And  …
    -/
    intro ξ hξ
    /-
      case neg.refine_3
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ξ : R
      hξ : Membership.mem (primitiveRoots k R) ξ
      ⊢ Exists fun a => And (And (LT.lt a k) (k.Coprime a)) (Eq (HPow.hPow ζ a) ξ)
    -/
    rw [mem_primitiveRoots (Nat.pos_of_ne_zero h0), h.isPrimitiveRoot_iff] at hξ
    /-
      case neg.refine_3
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ξ : R
      hξ : Exists fun i => And (LT.lt i k) (And (i.Coprime k) (Eq (HPow.hPow ζ i) ξ))
      ⊢ Exists fun a => And (And (LT.lt a k) (k.Coprime a)) (Eq (HPow.hPow ζ a) ξ)
    -/
    rcases hξ with ⟨i, hin, hi, H⟩
    /-
      case neg.refine_3.intro.intro.intro
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      k : Nat
      h : IsPrimitiveRoot ζ k
      h0 : Not (Eq k 0)
      this : NeZero k
      ξ : R
      i : Nat
      hin : LT.lt i k
      hi : i.Coprime k
      H : Eq (HPow.hPow ζ i) ξ
      ⊢ Exists fun a => And (And (LT.lt a k) (k.Coprime a)) (Eq (HPow.hPow ζ a) ξ)
    -/
    exact ⟨i, ⟨hin, hi.symm⟩, H⟩
    /-
      🎉 no goals
    -/


/-- The sets `primitiveRoots k R` are pairwise disjoint. -/
theorem disjoint {k l : ℕ} (h : k ≠ l) : Disjoint (primitiveRoots k R) (primitiveRoots l R) :=
  Finset.disjoint_left.2 fun _ hk hl ↦
    h <|
      (isPrimitiveRoot_of_mem_primitiveRoots hk).unique <| isPrimitiveRoot_of_mem_primitiveRoots hl


open scoped Classical in
/-- `nthRoots n` as a `Finset` is equal to the union of `primitiveRoots i R` for `i ∣ n`
if there is a primitive `n`th root of unity in `R`. -/
private -- marking as `private` since `nthRoots_one_eq_biUnion_primitiveRoots` can be used instead
theorem nthRoots_one_eq_biUnion_primitiveRoots' {ζ : R} {n : ℕ} [NeZero n]
    (h : IsPrimitiveRoot ζ n) :
    nthRootsFinset n R = (Nat.divisors n).biUnion fun i ↦ primitiveRoots i R := by
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    ζ : R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.nthRootsFinset n R) (n.divisors.biUnion fun i => primitiveRoo …
  -/
  symm
  /-
    R : Type u_4
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    ζ : R
    n : Nat
    inst✝ : NeZero n
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (n.divisors.biUnion fun i => primitiveRoots i R) (Polynomial.nthRootsFins …
  -/
  apply Finset.eq_of_subset_of_card_le
    /-
      case h
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      ⊢ HasSubset.Subset (n.divisors.biUnion fun i => primitiveRoots i R) (Polynomia …
    -/
  · intro x
    simp only [mem_biUnion, Nat.mem_divisors, Ne, nthRootsFinset,
      ← Multiset.toFinset_eq (nthRoots_one_nodup h), mem_mk, forall_exists_index, and_imp]
    /-
      case h
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      x : R
      ⊢ ∀ (x_1 : Nat), Dvd.dvd x_1 n → Not (Eq n 0) → Membership.mem (primitiveRoots …
    -/
    rintro a ⟨d, hd⟩ hn ha
    have hazero : 0 < a :=
      Nat.pos_of_ne_zero fun ha₀ ↦ hn <| by rwa [ha₀, zero_mul] at hd
    /-
      case h.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      x : R
      a d : Nat
      hd : Eq n (HMul.hMul a d)
      hn : Not (Eq n 0)
      ha : Membership.mem (primitiveRoots a R) x
      hazero : LT.lt 0 a
      ⊢ Membership.mem (Polynomial.nthRoots n 1) x
    -/
    rw [mem_primitiveRoots hazero] at ha
    /-
      case h.intro
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      x : R
      a d : Nat
      hd : Eq n (HMul.hMul a d)
      hn : Not (Eq n 0)
      ha : IsPrimitiveRoot x a
      hazero : LT.lt 0 a
      ⊢ Membership.mem (Polynomial.nthRoots n 1) x
    -/
    rw [mem_nthRoots <| NeZero.pos n, hd, pow_mul, ha.pow_eq_one, one_pow]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      ⊢ LE.le (Polynomial.nthRootsFinset n R).card (n.divisors.biUnion fun i => prim …
    -/
  · apply le_of_eq
    /-
      case h₂.hab
      R : Type u_4
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      ζ : R
      n : Nat
      inst✝ : NeZero n
      h : IsPrimitiveRoot ζ n
      ⊢ Eq (Polynomial.nthRootsFinset n R).card (n.divisors.biUnion fun i => primiti …
    -/
    rw [h.card_nthRootsFinset, Finset.card_biUnion]
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        ⊢ Eq n (n.divisors.sum fun u => (primitiveRoots u R).card)
      -/
    · nth_rw 1 [← Nat.sum_totient n]
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        ⊢ Eq (n.divisors.sum Nat.totient) (n.divisors.sum fun u => (primitiveRoots u R …
      -/
      refine sum_congr rfl ?_
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        ⊢ ∀ (x : Nat), Membership.mem n.divisors x → Eq x.totient (primitiveRoots x R) …
      -/
      simp only [Nat.mem_divisors]
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        ⊢ ∀ (x : Nat), And (Dvd.dvd x n) (Ne n 0) → Eq x.totient (primitiveRoots x R). …
      -/
      rintro k ⟨⟨d, hd⟩, -⟩
      /-
        case h₂.hab.intro.intro
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        k d : Nat
        hd : Eq n (HMul.hMul k d)
        ⊢ Eq k.totient (primitiveRoots k R).card
      -/
      rw [mul_comm] at hd
      /-
        case h₂.hab.intro.intro
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        k d : Nat
        hd : Eq n (HMul.hMul d k)
        ⊢ Eq k.totient (primitiveRoots k R).card
      -/
      rw [(h.pow (NeZero.pos n) hd).card_primitiveRoots]
      /-
        🎉 no goals
      -/
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        ⊢ ∀ (x : Nat), Membership.mem n.divisors x → ∀ (y : Nat), Membership.mem n.div …
      -/
    · intro i _ j _ hdiff
      /-
        case h₂.hab
        R : Type u_4
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        ζ : R
        n : Nat
        inst✝ : NeZero n
        h : IsPrimitiveRoot ζ n
        i : Nat
        a✝¹ : Membership.mem n.divisors i
        j : Nat
        a✝ : Membership.mem n.divisors j
        hdiff : Ne i j
        ⊢ Disjoint (primitiveRoots i R) (primitiveRoots j R)
      -/
      exact disjoint hdiff
      /-
        🎉 no goals
      -/


open scoped Classical in
/-- `nthRoots n` as a `Finset` is equal to the union of `primitiveRoots i R` for `i ∣ n`
if there is a primitive `n`th root of unity in `R`. -/
theorem nthRoots_one_eq_biUnion_primitiveRoots {ζ : R} {n : ℕ}
    (h : IsPrimitiveRoot ζ n) :
    nthRootsFinset n R = (Nat.divisors n).biUnion fun i ↦ primitiveRoots i R := by
  /-
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Polynomial.nthRootsFinset n R) (n.divisors.biUnion fun i => primitiveRoo …
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u_4
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      ζ : R
      n : Nat
      h : IsPrimitiveRoot ζ n
      hn : Eq n 0
      ⊢ Eq (Polynomial.nthRootsFinset n R) (n.divisors.biUnion fun i => primitiveRoo …
    -/
  · simp only [hn, nthRootsFinset_zero, Nat.divisors_zero, biUnion_empty]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    hn : Not (Eq n 0)
    ⊢ Eq (Polynomial.nthRootsFinset n R) (n.divisors.biUnion fun i => primitiveRoo …
  -/
  have : NeZero n := ⟨hn⟩
  /-
    case neg
    R : Type u_4
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ζ : R
    n : Nat
    h : IsPrimitiveRoot ζ n
    hn : Not (Eq n 0)
    this : NeZero n
    ⊢ Eq (Polynomial.nthRootsFinset n R) (n.divisors.biUnion fun i => primitiveRoo …
  -/
  exact nthRoots_one_eq_biUnion_primitiveRoots' h
  /-
    🎉 no goals
  -/


/-- The `MonoidHom` that takes an automorphism to the power of `μ` that `μ` gets mapped to
under it. -/
noncomputable def autToPow [NeZero n] : (S ≃ₐ[R] S) →* (ZMod n)ˣ :=
  let μ' := hμ.toRootsOfUnity
  have ho : orderOf μ' = n := by
    /-
      M : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁷ : CommMonoid M
      inst✝⁶ : CommMonoid N
      inst✝⁵ : DivisionCommMonoid G
      k l : Nat
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      μ : S
      n : Nat
      hμ : IsPrimitiveRoot μ n
      inst✝² : CommRing R
      inst✝¹ : Algebra R S
      inst✝ : NeZero n
      μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
      ⊢ Eq (orderOf μ') n
    -/
    refine Eq.trans ?_ hμ.eq_orderOf.symm -- `rw [hμ.eq_orderOf]` gives "motive not type correct"
    /-
      M : Type u_1
      N : Type u_2
      G : Type u_3
      R : Type u_4
      S : Type u_5
      F : Type u_6
      inst✝⁷ : CommMonoid M
      inst✝⁶ : CommMonoid N
      inst✝⁵ : DivisionCommMonoid G
      k l : Nat
      inst✝⁴ : CommRing S
      inst✝³ : IsDomain S
      μ : S
      n : Nat
      hμ : IsPrimitiveRoot μ n
      inst✝² : CommRing R
      inst✝¹ : Algebra R S
      inst✝ : NeZero n
      μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
      ⊢ Eq (orderOf μ') (orderOf μ)
    -/
    rw [← hμ.val_toRootsOfUnity_coe, orderOf_units, Subgroup.orderOf_coe]
    /-
      🎉 no goals
    -/
  MonoidHom.toHomUnits
    { toFun := fun σ ↦ (map_rootsOfUnity_eq_pow_self σ.toAlgHom μ').choose
      map_one' := by
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          ⊢ Eq ((fun σ => ↑⋯.choose) 1) 1
        -/
        dsimp only
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          ⊢ Eq (↑⋯.choose) 1
        -/
        generalize_proofs h1
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          h1 : Exists fun m => Eq (↑1 ↑↑μ') (HPow.hPow (↑↑μ') m)
          ⊢ Eq (↑h1.choose) 1
        -/
        have h := h1.choose_spec
        replace h : μ' = μ' ^ h1.choose :=
          rootsOfUnity.coe_injective (by simpa only [rootsOfUnity.coe_pow] using h)
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          h1 : Exists fun m => Eq (↑1 ↑↑μ') (HPow.hPow (↑↑μ') m)
          h : Eq μ' (HPow.hPow μ' h1.choose)
          ⊢ Eq (↑h1.choose) 1
        -/
        nth_rw 1 [← pow_one μ'] at h
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          h1 : Exists fun m => Eq (↑1 ↑↑μ') (HPow.hPow (↑↑μ') m)
          h : Eq (HPow.hPow μ' 1) (HPow.hPow μ' h1.choose)
          ⊢ Eq (↑h1.choose) 1
        -/
        convert ho ▸ (ZMod.natCast_eq_natCast_iff ..).mpr (pow_eq_pow_iff_modEq.mp h).symm
        /-
          case h.e'_3
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          h1 : Exists fun m => Eq (↑1 ↑↑μ') (HPow.hPow (↑↑μ') m)
          h : Eq (HPow.hPow μ' 1) (HPow.hPow μ' h1.choose)
          ⊢ Eq 1 ↑1
        -/
        exact Nat.cast_one.symm
        /-
          🎉 no goals
        -/
      map_mul' := by
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          ⊢ ∀ (x y : AlgEquiv R S S), Eq ({ toFun := fun σ => ↑⋯.choose, map_one' := ⋯ } …
        -/
        intro x y
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          ⊢ Eq ({ toFun := fun σ => ↑⋯.choose, map_one' := ⋯ }.toFun (HMul.hMul x y)) (H …
        -/
        dsimp only
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          ⊢ Eq (↑⋯.choose) (HMul.hMul ↑⋯.choose ↑⋯.choose)
        -/
        generalize_proofs hxy' hx' hy'
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          hxy' : Exists fun m => Eq (↑(HMul.hMul x y) ↑↑μ') (HPow.hPow (↑↑μ') m)
          hx' : Exists fun m => Eq (↑x ↑↑μ') (HPow.hPow (↑↑μ') m)
          hy' : Exists fun m => Eq (↑y ↑↑μ') (HPow.hPow (↑↑μ') m)
          ⊢ Eq (↑hxy'.choose) (HMul.hMul ↑hx'.choose ↑hy'.choose)
        -/
        have hxy := hxy'.choose_spec
        replace hxy : x (((μ' : Sˣ) : S) ^ hy'.choose) = ((μ' : Sˣ) : S) ^ hxy'.choose :=
          hy'.choose_spec ▸ hxy
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          hxy' : Exists fun m => Eq (↑(HMul.hMul x y) ↑↑μ') (HPow.hPow (↑↑μ') m)
          hx' : Exists fun m => Eq (↑x ↑↑μ') (HPow.hPow (↑↑μ') m)
          hy' : Exists fun m => Eq (↑y ↑↑μ') (HPow.hPow (↑↑μ') m)
          hxy : Eq (x (HPow.hPow (↑↑μ') hy'.choose)) (HPow.hPow (↑↑μ') hxy'.choose)
          ⊢ Eq (↑hxy'.choose) (HMul.hMul ↑hx'.choose ↑hy'.choose)
        -/
        rw [map_pow] at hxy
        replace hxy : (((μ' : Sˣ) : S) ^ hx'.choose) ^ hy'.choose = ((μ' : Sˣ) : S) ^ hxy'.choose :=
          hx'.choose_spec ▸ hxy
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          hxy' : Exists fun m => Eq (↑(HMul.hMul x y) ↑↑μ') (HPow.hPow (↑↑μ') m)
          hx' : Exists fun m => Eq (↑x ↑↑μ') (HPow.hPow (↑↑μ') m)
          hy' : Exists fun m => Eq (↑y ↑↑μ') (HPow.hPow (↑↑μ') m)
          hxy : Eq (HPow.hPow (HPow.hPow (↑↑μ') hx'.choose) hy'.choose) (HPow.hPow (↑↑μ' …
          ⊢ Eq (↑hxy'.choose) (HMul.hMul ↑hx'.choose ↑hy'.choose)
        -/
        rw [← pow_mul] at hxy
        replace hxy : μ' ^ (hx'.choose * hy'.choose) = μ' ^ hxy'.choose :=
          rootsOfUnity.coe_injective (by simpa only [rootsOfUnity.coe_pow] using hxy)
        /-
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          hxy' : Exists fun m => Eq (↑(HMul.hMul x y) ↑↑μ') (HPow.hPow (↑↑μ') m)
          hx' : Exists fun m => Eq (↑x ↑↑μ') (HPow.hPow (↑↑μ') m)
          hy' : Exists fun m => Eq (↑y ↑↑μ') (HPow.hPow (↑↑μ') m)
          hxy : Eq (HPow.hPow μ' (HMul.hMul hx'.choose hy'.choose)) (HPow.hPow μ' hxy'.c …
          ⊢ Eq (↑hxy'.choose) (HMul.hMul ↑hx'.choose ↑hy'.choose)
        -/
        convert ho ▸ (ZMod.natCast_eq_natCast_iff ..).mpr (pow_eq_pow_iff_modEq.mp hxy).symm
        /-
          case h.e'_3
          M : Type u_1
          N : Type u_2
          G : Type u_3
          R : Type u_4
          S : Type u_5
          F : Type u_6
          inst✝⁷ : CommMonoid M
          inst✝⁶ : CommMonoid N
          inst✝⁵ : DivisionCommMonoid G
          k l : Nat
          inst✝⁴ : CommRing S
          inst✝³ : IsDomain S
          μ : S
          n : Nat
          hμ : IsPrimitiveRoot μ n
          inst✝² : CommRing R
          inst✝¹ : Algebra R S
          inst✝ : NeZero n
          μ' : Subtype fun x => Membership.mem (rootsOfUnity n S) x := hμ.toRootsOfUnity
          ho : Eq (orderOf μ') n
          x y : AlgEquiv R S S
          hxy' : Exists fun m => Eq (↑(HMul.hMul x y) ↑↑μ') (HPow.hPow (↑↑μ') m)
          hx' : Exists fun m => Eq (↑x ↑↑μ') (HPow.hPow (↑↑μ') m)
          hy' : Exists fun m => Eq (↑y ↑↑μ') (HPow.hPow (↑↑μ') m)
          hxy : Eq (HPow.hPow μ' (HMul.hMul hx'.choose hy'.choose)) (HPow.hPow μ' hxy'.c …
          ⊢ Eq (HMul.hMul ↑hx'.choose ↑hy'.choose) ↑(HMul.hMul hx'.choose hy'.choose)
        -/
        exact (Nat.cast_mul ..).symm }
        /-
          🎉 no goals
        -/

-- We are not using @[simps] in `autToPow` to avoid a timeout.

theorem coe_autToPow_apply [NeZero n] (f : S ≃ₐ[R] S) :
    (autToPow R hμ f : ZMod n) =
      ((map_rootsOfUnity_eq_pow_self f hμ.toRootsOfUnity).choose : ZMod n) :=
  rfl


@[simp]
theorem autToPow_spec [NeZero n] (f : S ≃ₐ[R] S) : μ ^ (hμ.autToPow R f : ZMod n).val = f μ := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    ⊢ Eq (HPow.hPow μ (↑((IsPrimitiveRoot.autToPow R hμ) f)).val) (f μ)
  -/
  rw [IsPrimitiveRoot.coe_autToPow_apply]
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    ⊢ Eq (HPow.hPow μ (↑⋯.choose).val) (f μ)
  -/
  generalize_proofs h
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ Eq (HPow.hPow μ (↑h.choose).val) (f μ)
  -/
  refine (?_ : ((hμ.toRootsOfUnity : Sˣ) : S) ^ _ = _).trans h.choose_spec.symm
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ Eq (HPow.hPow (↑↑hμ.toRootsOfUnity) (↑h.choose).val) (HPow.hPow (↑↑hμ.toRoot …
  -/
  rw [← rootsOfUnity.coe_pow, ← rootsOfUnity.coe_pow]
  /-
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ Eq ↑↑(HPow.hPow hμ.toRootsOfUnity (↑h.choose).val) ↑↑(HPow.hPow hμ.toRootsOf …
  -/
  congr 2
  /-
    case e_self.e_self
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ Eq (HPow.hPow hμ.toRootsOfUnity (↑h.choose).val) (HPow.hPow hμ.toRootsOfUnit …
  -/
  rw [pow_eq_pow_iff_modEq, ZMod.val_natCast]
  /-
    case e_self.e_self
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ (orderOf hμ.toRootsOfUnity).ModEq (HMod.hMod h.choose n) h.choose
  -/
  conv => enter [2, 2]; rw [hμ.eq_orderOf]
  /-
    case e_self.e_self
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ (orderOf hμ.toRootsOfUnity).ModEq (HMod.hMod h.choose (orderOf μ)) h.choose
  -/
  rw [← Subgroup.orderOf_coe, ← orderOf_units]
  /-
    case e_self.e_self
    R : Type u_4
    S : Type u_5
    inst✝⁴ : CommRing S
    inst✝³ : IsDomain S
    μ : S
    n : Nat
    hμ : IsPrimitiveRoot μ n
    inst✝² : CommRing R
    inst✝¹ : Algebra R S
    inst✝ : NeZero n
    f : AlgEquiv R S S
    h : Exists fun m => Eq (f ↑↑hμ.toRootsOfUnity) (HPow.hPow (↑↑hμ.toRootsOfUnity …
    ⊢ (orderOf ↑↑hμ.toRootsOfUnity).ModEq (HMod.hMod h.choose (orderOf μ)) h.choose
  -/
  exact Nat.mod_modEq _ _
  /-
    🎉 no goals
  -/


/-- If `G` is cyclic of order `n` and `G'` contains a primitive `n`th root of unity,
then for each `a : G` with `a ≠ 1` there is a homomorphism `φ : G →* G'` such that `φ a ≠ 1`. -/
lemma IsCyclic.exists_apply_ne_one {G G' : Type*} [CommGroup G] [IsCyclic G] [Finite G]
    [CommGroup G'] (hG' : ∃ ζ : G', IsPrimitiveRoot ζ (Nat.card G)) ⦃a : G⦄ (ha : a ≠ 1) :
    ∃ φ : G →* G', φ a ≠ 1 := by
  /-
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    hG' : Exists fun ζ => IsPrimitiveRoot ζ (Nat.card G)
    a : G
    ha : Ne a 1
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  let inst : Fintype G := Fintype.ofFinite _
  /-
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    hG' : Exists fun ζ => IsPrimitiveRoot ζ (Nat.card G)
    a : G
    ha : Ne a 1
    inst : Fintype G := Fintype.ofFinite G
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  obtain ⟨ζ, hζ⟩ := hG'
  -- pick a generator `g` of `G`
  /-
    case intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    ha : Ne a 1
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := G)
  have hζg : orderOf ζ ∣ orderOf g := by
    rw [← hζ.eq_orderOf, orderOf_eq_card_of_forall_mem_zpowers hg, Nat.card_eq_fintype_card]
  -- use the homomorphism `φ` given by `g ↦ ζ`
  /-
    case intro.intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    ha : Ne a 1
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  let φ := monoidHomOfForallMemZpowers hg hζg
  have hφg : IsPrimitiveRoot (φ g) (Nat.card G) := by
    rwa [monoidHomOfForallMemZpowers_apply_gen hg hζg]
  /-
    case intro.intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    ha : Ne a 1
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  use φ
  /-
    case h
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    ha : Ne a 1
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ⊢ Ne (φ a) 1
  -/
  contrapose! ha
  /-
    case h
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ha : Eq (φ a) 1
    ⊢ Eq a 1
  -/
  specialize hg a
  /-
    case h
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg✝ : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg✝ hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ha : Eq (φ a) 1
    hg : Membership.mem (Subgroup.zpowers g) a
    ⊢ Eq a 1
  -/
  rw [← mem_powers_iff_mem_zpowers, Submonoid.mem_powers_iff] at hg
  /-
    case h
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg✝ : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg✝ hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ha : Eq (φ a) 1
    hg : Exists fun n => Eq (HPow.hPow g n) a
    ⊢ Eq a 1
  -/
  obtain ⟨k, hk⟩ := hg
  /-
    case h.intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    ha : Eq (φ a) 1
    k : Nat
    hk : Eq (HPow.hPow g k) a
    ⊢ Eq a 1
  -/
  rw [← hk, map_pow] at ha
  /-
    case h.intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    k : Nat
    ha : Eq (HPow.hPow (φ g) k) 1
    hk : Eq (HPow.hPow g k) a
    ⊢ Eq a 1
  -/
  obtain ⟨l, rfl⟩ := (hφg.pow_eq_one_iff_dvd k).mp ha
  /-
    case h.intro.intro
    G : Type u_7
    G' : Type u_8
    inst✝³ : CommGroup G
    inst✝² : IsCyclic G
    inst✝¹ : Finite G
    inst✝ : CommGroup G'
    a : G
    inst : Fintype G := Fintype.ofFinite G
    ζ : G'
    hζ : IsPrimitiveRoot ζ (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    hζg : Dvd.dvd (orderOf ζ) (orderOf g)
    φ : MonoidHom G G' := monoidHomOfForallMemZpowers hg hζg
    hφg : IsPrimitiveRoot (φ g) (Nat.card G)
    l : Nat
    ha : Eq (HPow.hPow (φ g) (HMul.hMul (Nat.card G) l)) 1
    hk : Eq (HPow.hPow g (HMul.hMul (Nat.card G) l)) a
    ⊢ Eq a 1
  -/
  rw [← hk, pow_mul, Nat.card_eq_fintype_card, pow_card_eq_one, one_pow]
  /-
    🎉 no goals
  -/


/-- If `M` is a commutative group that contains a primitive `n`th root of unity
and `a : ZMod n` is nonzero, then there exists a group homomorphism `φ` from the
additive group `ZMod n` to the multiplicative group `Mˣ` such that `φ a ≠ 1`. -/
lemma ZMod.exists_monoidHom_apply_ne_one {M : Type*} [CommMonoid M] {n : ℕ} [NeZero n]
    (hG : ∃ ζ : M, IsPrimitiveRoot ζ n) {a : ZMod n} (ha : a ≠ 0) :
    ∃ φ : Multiplicative (ZMod n) →* Mˣ, φ (Multiplicative.ofAdd a) ≠ 1 := by
  /-
    M : Type u_7
    inst✝¹ : CommMonoid M
    n : Nat
    inst✝ : NeZero n
    hG : Exists fun ζ => IsPrimitiveRoot ζ n
    a : ZMod n
    ha : Ne a 0
    ⊢ Exists fun φ => Ne (φ (Multiplicative.ofAdd a)) 1
  -/
  obtain ⟨ζ, hζ⟩ := hG
  have hc : n = Nat.card (Multiplicative (ZMod n)) := by
    simp only [Nat.card_eq_fintype_card, Fintype.card_multiplicative, card]
  exact IsCyclic.exists_apply_ne_one
    (hc ▸ ⟨hζ.toRootsOfUnity.val, IsPrimitiveRoot.coe_units_iff.mp hζ⟩) <|
    by simp only [ne_eq, ofAdd_eq_one, ha, not_false_eq_true]


