/-- An ordered additive commutative monoid is called `Archimedean` if for any two elements `x`, `y`
such that `0 < y`, there exists a natural number `n` such that `x ≤ n • y`. -/
class Archimedean (α) [OrderedAddCommMonoid α] : Prop where
  /-- For any two elements `x`, `y` such that `0 < y`, there exists a natural number `n`
  such that `x ≤ n • y`. -/
  arch : ∀ (x : α) {y : α}, 0 < y → ∃ n : ℕ, x ≤ n • y


/-- An ordered commutative monoid is called `MulArchimedean` if for any two elements `x`, `y`
such that `1 < y`, there exists a natural number `n` such that `x ≤ y ^ n`. -/
@[to_additive Archimedean]
class MulArchimedean (α) [OrderedCommMonoid α] : Prop where
  /-- For any two elements `x`, `y` such that `1 < y`, there exists a natural number `n`
  such that `x ≤ y ^ n`. -/
  arch : ∀ (x : α) {y : α}, 1 < y → ∃ n : ℕ, x ≤ y ^ n


@[to_additive]
instance OrderDual.instMulArchimedean [OrderedCommGroup α] [MulArchimedean α] :
    MulArchimedean αᵒᵈ :=
  ⟨fun x y hy =>
    let ⟨n, hn⟩ := MulArchimedean.arch (ofDual x)⁻¹ (inv_lt_one_iff_one_lt.2 hy)
           /-
             α : Type u_1
             inst✝¹ : OrderedCommGroup α
             inst✝ : MulArchimedean α
             x y : OrderDual α
             hy : LT.lt 1 y
             n : Nat
             hn : LE.le (Inv.inv (OrderDual.ofDual x)) (HPow.hPow (Inv.inv y) n)
             ⊢ LE.le x (HPow.hPow y n)
           -/
    ⟨n, by rwa [inv_pow, inv_le_inv_iff] at hn⟩⟩
           /-
             🎉 no goals
           -/


instance Additive.instArchimedean [OrderedCommGroup α] [MulArchimedean α] :
    Archimedean (Additive α) :=
  ⟨fun x _ hy ↦ MulArchimedean.arch x.toMul hy⟩


instance Multiplicative.instMulArchimedean [OrderedAddCommGroup α] [Archimedean α] :
    MulArchimedean (Multiplicative α) :=
  ⟨fun x _ hy ↦ Archimedean.arch x.toAdd hy⟩


@[to_additive]
theorem exists_lt_pow [OrderedCommMonoid M] [MulArchimedean M]
    [MulLeftStrictMono M] {a : M} (ha : 1 < a) (b : M) :
    ∃ n : ℕ, b < a ^ n :=
  let ⟨k, hk⟩ := MulArchimedean.arch b ha
  ⟨k + 1, hk.trans_lt <| pow_lt_pow_right' ha k.lt_succ_self⟩


/-- An archimedean decidable linearly ordered `CommGroup` has a version of the floor: for
`a > 1`, any `g` in the group lies between some two consecutive powers of `a`. -/
@[to_additive "An archimedean decidable linearly ordered `AddCommGroup` has a version of the floor:
for `a > 0`, any `g` in the group lies between some two consecutive multiples of `a`. -/"]
theorem existsUnique_zpow_near_of_one_lt {a : α} (ha : 1 < a) (g : α) :
    ∃! k : ℤ, a ^ k ≤ g ∧ g < a ^ (k + 1) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  let s : Set ℤ := { n : ℤ | a ^ n ≤ g }
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  obtain ⟨k, hk : g⁻¹ ≤ a ^ k⟩ := MulArchimedean.arch g⁻¹ ha
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k : Nat
    hk : LE.le (Inv.inv g) (HPow.hPow a k)
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  have h_ne : s.Nonempty := ⟨-k, by simpa [s] using inv_le_inv' hk⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k : Nat
    hk : LE.le (Inv.inv g) (HPow.hPow a k)
    h_ne : s.Nonempty
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  obtain ⟨k, hk⟩ := MulArchimedean.arch g ha
  have h_bdd : ∀ n ∈ s, n ≤ (k : ℤ) := by
    intro n hn
    apply (zpow_le_zpow_iff_right ha).mp
    rw [← zpow_natCast] at hk
    exact le_trans hn hk
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k✝ : Nat
    hk✝ : LE.le (Inv.inv g) (HPow.hPow a k✝)
    h_ne : s.Nonempty
    k : Nat
    hk : LE.le g (HPow.hPow a k)
    h_bdd : ∀ (n : Int), Membership.mem s n → LE.le n ↑k
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  obtain ⟨m, hm, hm'⟩ := Int.exists_greatest_of_bdd ⟨k, h_bdd⟩ h_ne
  have hm'' : g < a ^ (m + 1) := by
    contrapose! hm'
    exact ⟨m + 1, hm', lt_add_one _⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k✝ : Nat
    hk✝ : LE.le (Inv.inv g) (HPow.hPow a k✝)
    h_ne : s.Nonempty
    k : Nat
    hk : LE.le g (HPow.hPow a k)
    h_bdd : ∀ (n : Int), Membership.mem s n → LE.le n ↑k
    m : Int
    hm : Membership.mem s m
    hm' : ∀ (z : Int), Membership.mem s z → LE.le z m
    hm'' : LT.lt g (HPow.hPow a (HAdd.hAdd m 1))
    ⊢ ExistsUnique fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (H …
  -/
  refine ⟨m, ⟨hm, hm''⟩, fun n hn => (hm' n hn.1).antisymm <| Int.le_of_lt_add_one ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k✝ : Nat
    hk✝ : LE.le (Inv.inv g) (HPow.hPow a k✝)
    h_ne : s.Nonempty
    k : Nat
    hk : LE.le g (HPow.hPow a k)
    h_bdd : ∀ (n : Int), Membership.mem s n → LE.le n ↑k
    m : Int
    hm : Membership.mem s m
    hm' : ∀ (z : Int), Membership.mem s z → LE.le z m
    hm'' : LT.lt g (HPow.hPow a (HAdd.hAdd m 1))
    n : Int
    hn : (fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (HAdd.hAdd  …
    ⊢ LT.lt m (HAdd.hAdd n 1)
  -/
  rw [← zpow_lt_zpow_iff_right ha]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedCommGroup α
    inst✝ : MulArchimedean α
    a : α
    ha : LT.lt 1 a
    g : α
    s : Set Int := setOf fun n => LE.le (HPow.hPow a n) g
    k✝ : Nat
    hk✝ : LE.le (Inv.inv g) (HPow.hPow a k✝)
    h_ne : s.Nonempty
    k : Nat
    hk : LE.le g (HPow.hPow a k)
    h_bdd : ∀ (n : Int), Membership.mem s n → LE.le n ↑k
    m : Int
    hm : Membership.mem s m
    hm' : ∀ (z : Int), Membership.mem s z → LE.le z m
    hm'' : LT.lt g (HPow.hPow a (HAdd.hAdd m 1))
    n : Int
    hn : (fun k => And (LE.le (HPow.hPow a k) g) (LT.lt g (HPow.hPow a (HAdd.hAdd  …
    ⊢ LT.lt (HPow.hPow a m) (HPow.hPow a (HAdd.hAdd n 1))
  -/
  exact lt_of_le_of_lt hm hn.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem existsUnique_zpow_near_of_one_lt' {a : α} (ha : 1 < a) (g : α) :
    ∃! k : ℤ, 1 ≤ g / a ^ k ∧ g / a ^ k < a := by
  simpa only [one_le_div', zpow_add_one, div_lt_iff_lt_mul'] using
    existsUnique_zpow_near_of_one_lt ha g


@[to_additive]
theorem existsUnique_div_zpow_mem_Ico {a : α} (ha : 1 < a) (b c : α) :
    ∃! m : ℤ, b / a ^ m ∈ Set.Ico c (c * a) := by
  simpa only [mem_Ico, le_div_iff_mul_le, one_mul, mul_comm c, div_lt_iff_lt_mul, mul_assoc] using
    existsUnique_zpow_near_of_one_lt' ha (b / c)


@[to_additive]
theorem existsUnique_mul_zpow_mem_Ico {a : α} (ha : 1 < a) (b c : α) :
    ∃! m : ℤ, b * a ^ m ∈ Set.Ico c (c * a) :=
  (Equiv.neg ℤ).bijective.existsUnique_iff.2 <| by
    simpa only [Equiv.neg_apply, mem_Ico, zpow_neg, ← div_eq_mul_inv, le_div_iff_mul_le, one_mul,
      mul_comm c, div_lt_iff_lt_mul, mul_assoc] using existsUnique_zpow_near_of_one_lt' ha (b / c)


@[to_additive]
theorem existsUnique_add_zpow_mem_Ioc {a : α} (ha : 1 < a) (b c : α) :
    ∃! m : ℤ, b * a ^ m ∈ Set.Ioc c (c * a) :=
  (Equiv.addRight (1 : ℤ)).bijective.existsUnique_iff.2 <| by
    simpa only [zpow_add_one, div_lt_iff_lt_mul', le_div_iff_mul_le', ← mul_assoc, and_comm,
      mem_Ioc, Equiv.coe_addRight, mul_le_mul_iff_right] using
      existsUnique_zpow_near_of_one_lt ha (c / b)


@[to_additive]
theorem existsUnique_sub_zpow_mem_Ioc {a : α} (ha : 1 < a) (b c : α) :
    ∃! m : ℤ, b / a ^ m ∈ Set.Ioc c (c * a) :=
  (Equiv.neg ℤ).bijective.existsUnique_iff.2 <| by
    simpa only [Equiv.neg_apply, zpow_neg, div_inv_eq_mul] using
      existsUnique_add_zpow_mem_Ioc ha b c


theorem exists_nat_ge [OrderedSemiring α] [Archimedean α] (x : α) : ∃ n : ℕ, x ≤ n := by
  /-
    α : Type u_1
    inst✝¹ : OrderedSemiring α
    inst✝ : Archimedean α
    x : α
    ⊢ Exists fun n => LE.le x ↑n
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝¹ : OrderedSemiring α
    inst✝ : Archimedean α
    x : α
    a✝ : Nontrivial α
    ⊢ Exists fun n => LE.le x ↑n
  -/
  exact (Archimedean.arch x one_pos).imp fun n h => by rwa [← nsmul_one]
  /-
    🎉 no goals
  -/


instance (priority := 100) [OrderedSemiring α] [Archimedean α] : IsDirected α (· ≤ ·) :=
  ⟨fun x y ↦
    let ⟨m, hm⟩ := exists_nat_ge x; let ⟨n, hn⟩ := exists_nat_ge y
    let ⟨k, hmk, hnk⟩ := exists_ge_ge m n
    ⟨k, hm.trans <| Nat.mono_cast hmk, hn.trans <| Nat.mono_cast hnk⟩⟩


lemma exists_nat_gt (x : α) : ∃ n : ℕ, x < n :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝¹ : StrictOrderedSemiring α
                                                      inst✝ : Archimedean α
                                                      x : α
                                                      n : Nat
                                                      hn : LT.lt x (HSMul.hSMul n 1)
                                                      ⊢ LT.lt x ↑n
                                                    -/
  (exists_lt_nsmul zero_lt_one x).imp fun n hn ↦ by rwa [← nsmul_one]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem add_one_pow_unbounded_of_pos (x : α) (hy : 0 < y) : ∃ n : ℕ, x < (y + 1) ^ n :=
  have : 0 ≤ 1 + y := add_nonneg zero_le_one hy.le
  (Archimedean.arch x hy).imp fun n h ↦
    calc
      x ≤ n • y := h
      _ = n * y := nsmul_eq_mul _ _
      _ < 1 + n * y := lt_one_add _
      _ ≤ (1 + y) ^ n :=
        one_add_mul_le_pow' (mul_nonneg hy.le hy.le) (mul_nonneg this this)
          (add_nonneg zero_le_two hy.le) _
                            /-
                              α : Type u_1
                              inst✝¹ : StrictOrderedSemiring α
                              inst✝ : Archimedean α
                              y x : α
                              hy : LT.lt 0 y
                              this : LE.le 0 (HAdd.hAdd 1 y)
                              n : Nat
                              h : LE.le x (HSMul.hSMul n y)
                              ⊢ Eq (HPow.hPow (HAdd.hAdd 1 y) n) (HPow.hPow (HAdd.hAdd y 1) n)
                            -/
      _ = (y + 1) ^ n := by rw [add_comm]
                            /-
                              🎉 no goals
                            -/


lemma pow_unbounded_of_one_lt [ExistsAddOfLE α] (x : α) (hy1 : 1 < y) : ∃ n : ℕ, x < y ^ n := by
  /-
    α : Type u_1
    inst✝² : StrictOrderedSemiring α
    inst✝¹ : Archimedean α
    y : α
    inst✝ : ExistsAddOfLE α
    x : α
    hy1 : LT.lt 1 y
    ⊢ Exists fun n => LT.lt x (HPow.hPow y n)
  -/
  obtain ⟨z, hz, rfl⟩ := exists_pos_add_of_lt' hy1
  /-
    case intro.intro
    α : Type u_1
    inst✝² : StrictOrderedSemiring α
    inst✝¹ : Archimedean α
    inst✝ : ExistsAddOfLE α
    x z : α
    hz : LT.lt 0 z
    hy1 : LT.lt 1 (HAdd.hAdd 1 z)
    ⊢ Exists fun n => LT.lt x (HPow.hPow (HAdd.hAdd 1 z) n)
  -/
  rw [add_comm]
  /-
    case intro.intro
    α : Type u_1
    inst✝² : StrictOrderedSemiring α
    inst✝¹ : Archimedean α
    inst✝ : ExistsAddOfLE α
    x z : α
    hz : LT.lt 0 z
    hy1 : LT.lt 1 (HAdd.hAdd 1 z)
    ⊢ Exists fun n => LT.lt x (HPow.hPow (HAdd.hAdd z 1) n)
  -/
  exact add_one_pow_unbounded_of_pos _ hz
  /-
    🎉 no goals
  -/


theorem exists_int_ge (x : R) : ∃ n : ℤ, x ≤ n := let ⟨n, h⟩ := exists_nat_ge x; ⟨n, mod_cast h⟩


theorem exists_int_le (x : R) : ∃ n : ℤ, n ≤ x :=
                                            /-
                                              R : Type u_3
                                              inst✝¹ : OrderedRing R
                                              inst✝ : Archimedean R
                                              x : R
                                              n : Int
                                              h : LE.le (Neg.neg x) ↑n
                                              ⊢ LE.le (↑(Neg.neg n)) x
                                            -/
  let ⟨n, h⟩ := exists_int_ge (-x); ⟨-n, by simpa [neg_le] using h⟩
                                            /-
                                              🎉 no goals
                                            -/


instance (priority := 100) : IsDirected R (· ≥ ·) where
  directed a b :=
    let ⟨m, hm⟩ := exists_int_le a; let ⟨n, hn⟩ := exists_int_le b
    ⟨(min m n : ℤ), le_trans (Int.cast_mono <| min_le_left _ _) hm,
      le_trans (Int.cast_mono <| min_le_right _ _) hn⟩


theorem exists_int_gt (x : α) : ∃ n : ℤ, x < n :=
  let ⟨n, h⟩ := exists_nat_gt x
         /-
           α : Type u_1
           inst✝¹ : StrictOrderedRing α
           inst✝ : Archimedean α
           x : α
           n : Nat
           h : LT.lt x ↑n
           ⊢ LT.lt x ↑↑n
         -/
  ⟨n, by rwa [Int.cast_natCast]⟩
         /-
           🎉 no goals
         -/


theorem exists_int_lt (x : α) : ∃ n : ℤ, (n : α) < x :=
  let ⟨n, h⟩ := exists_int_gt (-x)
          /-
            α : Type u_1
            inst✝¹ : StrictOrderedRing α
            inst✝ : Archimedean α
            x : α
            n : Int
            h : LT.lt (Neg.neg x) ↑n
            ⊢ LT.lt (↑(Neg.neg n)) x
          -/
  ⟨-n, by rw [Int.cast_neg]; exact neg_lt.1 h⟩
                             /-
                               🎉 no goals
                             -/


theorem exists_floor (x : α) : ∃ fl : ℤ, ∀ z : ℤ, z ≤ fl ↔ (z : α) ≤ x := by
  /-
    α : Type u_1
    inst✝¹ : StrictOrderedRing α
    inst✝ : Archimedean α
    x : α
    ⊢ Exists fun fl => ∀ (z : Int), Iff (LE.le z fl) (LE.le (↑z) x)
  -/
  haveI := Classical.propDecidable
  have : ∃ ub : ℤ, (ub : α) ≤ x ∧ ∀ z : ℤ, (z : α) ≤ x → z ≤ ub :=
    Int.exists_greatest_of_bdd
      (let ⟨n, hn⟩ := exists_int_gt x
      ⟨n, fun z h' => Int.cast_le.1 <| le_trans h' <| le_of_lt hn⟩)
      (let ⟨n, hn⟩ := exists_int_lt x
      ⟨n, le_of_lt hn⟩)
  /-
    α : Type u_1
    inst✝¹ : StrictOrderedRing α
    inst✝ : Archimedean α
    x : α
    this✝ : (a : Prop) → Decidable a
    this : Exists fun ub => And (LE.le (↑ub) x) (∀ (z : Int), LE.le (↑z) x → LE.le …
    ⊢ Exists fun fl => ∀ (z : Int), Iff (LE.le z fl) (LE.le (↑z) x)
  -/
  refine this.imp fun fl h z => ?_
  /-
    α : Type u_1
    inst✝¹ : StrictOrderedRing α
    inst✝ : Archimedean α
    x : α
    this✝ : (a : Prop) → Decidable a
    this : Exists fun ub => And (LE.le (↑ub) x) (∀ (z : Int), LE.le (↑z) x → LE.le …
    fl : Int
    h : And (LE.le (↑fl) x) (∀ (z : Int), LE.le (↑z) x → LE.le z fl)
    z : Int
    ⊢ Iff (LE.le z fl) (LE.le (↑z) x)
  -/
  cases' h with h₁ h₂
  /-
    case intro
    α : Type u_1
    inst✝¹ : StrictOrderedRing α
    inst✝ : Archimedean α
    x : α
    this✝ : (a : Prop) → Decidable a
    this : Exists fun ub => And (LE.le (↑ub) x) (∀ (z : Int), LE.le (↑z) x → LE.le …
    fl z : Int
    h₁ : LE.le (↑fl) x
    h₂ : ∀ (z : Int), LE.le (↑z) x → LE.le z fl
    ⊢ Iff (LE.le z fl) (LE.le (↑z) x)
  -/
  exact ⟨fun h => le_trans (Int.cast_le.2 h) h₁, h₂ z⟩
  /-
    🎉 no goals
  -/


/-- Every x greater than or equal to 1 is between two successive
natural-number powers of every y greater than one. -/
theorem exists_nat_pow_near (hx : 1 ≤ x) (hy : 1 < y) : ∃ n : ℕ, y ^ n ≤ x ∧ x < y ^ (n + 1) := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedSemiring α
    inst✝¹ : Archimedean α
    inst✝ : ExistsAddOfLE α
    x y : α
    hx : LE.le 1 x
    hy : LT.lt 1 y
    ⊢ Exists fun n => And (LE.le (HPow.hPow y n) x) (LT.lt x (HPow.hPow y (HAdd.hA …
  -/
  have h : ∃ n : ℕ, x < y ^ n := pow_unbounded_of_one_lt _ hy
  classical exact
      let n := Nat.find h
      have hn : x < y ^ n := Nat.find_spec h
      have hnp : 0 < n :=
        pos_iff_ne_zero.2 fun hn0 => by rw [hn0, pow_zero] at hn; exact not_le_of_gt hn hx
      have hnsp : Nat.pred n + 1 = n := Nat.succ_pred_eq_of_pos hnp
      have hltn : Nat.pred n < n := Nat.pred_lt (ne_of_gt hnp)
      ⟨Nat.pred n, le_of_not_lt (Nat.find_min h hltn), by rwa [hnsp]⟩


lemma exists_nat_one_div_lt (hε : 0 < ε) : ∃ n : ℕ, 1 / (n + 1 : α) < ε := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : Archimedean α
    ε : α
    hε : LT.lt 0 ε
    ⊢ Exists fun n => LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)) ε
  -/
  cases' exists_nat_gt (1 / ε) with n hn
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : Archimedean α
    ε : α
    hε : LT.lt 0 ε
    n : Nat
    hn : LT.lt (HDiv.hDiv 1 ε) ↑n
    ⊢ Exists fun n => LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)) ε
  -/
  use n
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrderedSemifield α
    inst✝ : Archimedean α
    ε : α
    hε : LT.lt 0 ε
    n : Nat
    hn : LT.lt (HDiv.hDiv 1 ε) ↑n
    ⊢ LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)) ε
  -/
  rw [div_lt_iff₀, ← div_lt_iff₀' hε]
    /-
      case h
      α : Type u_1
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : Archimedean α
      ε : α
      hε : LT.lt 0 ε
      n : Nat
      hn : LT.lt (HDiv.hDiv 1 ε) ↑n
      ⊢ LT.lt (HDiv.hDiv 1 ε) (HAdd.hAdd (↑n) 1)
    -/
  · apply hn.trans
    /-
      case h
      α : Type u_1
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : Archimedean α
      ε : α
      hε : LT.lt 0 ε
      n : Nat
      hn : LT.lt (HDiv.hDiv 1 ε) ↑n
      ⊢ LT.lt (↑n) (HAdd.hAdd (↑n) 1)
    -/
    simp [zero_lt_one]
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_1
      inst✝¹ : LinearOrderedSemifield α
      inst✝ : Archimedean α
      ε : α
      hε : LT.lt 0 ε
      n : Nat
      hn : LT.lt (HDiv.hDiv 1 ε) ↑n
      ⊢ LT.lt 0 (HAdd.hAdd (↑n) 1)
    -/
  · exact n.cast_add_one_pos
    /-
      🎉 no goals
    -/


/-- Every positive `x` is between two successive integer powers of
another `y` greater than one. This is the same as `exists_mem_Ioc_zpow`,
but with ≤ and < the other way around. -/
theorem exists_mem_Ico_zpow (hx : 0 < x) (hy : 1 < y) : ∃ n : ℤ, x ∈ Ico (y ^ n) (y ^ (n + 1)) := by
  classical exact
      let ⟨N, hN⟩ := pow_unbounded_of_one_lt x⁻¹ hy
      have he : ∃ m : ℤ, y ^ m ≤ x :=
        ⟨-N,
          le_of_lt
            (by
              rw [zpow_neg y ↑N, zpow_natCast]
              exact (inv_lt_comm₀ hx (lt_trans (inv_pos.2 hx) hN)).1 hN)⟩
      let ⟨M, hM⟩ := pow_unbounded_of_one_lt x hy
      have hb : ∃ b : ℤ, ∀ m, y ^ m ≤ x → m ≤ b :=
        ⟨M, fun m hm =>
          le_of_not_lt fun hlt =>
            not_lt_of_ge (zpow_le_zpow_right₀ hy.le hlt.le)
              (lt_of_le_of_lt hm (by rwa [← zpow_natCast] at hM))⟩
      let ⟨n, hn₁, hn₂⟩ := Int.exists_greatest_of_bdd hb he
      ⟨n, hn₁, lt_of_not_ge fun hge => not_le_of_gt (Int.lt_succ _) (hn₂ _ hge)⟩


/-- Every positive `x` is between two successive integer powers of
another `y` greater than one. This is the same as `exists_mem_Ico_zpow`,
but with ≤ and < the other way around. -/
theorem exists_mem_Ioc_zpow (hx : 0 < x) (hy : 1 < y) : ∃ n : ℤ, x ∈ Ioc (y ^ n) (y ^ (n + 1)) :=
  let ⟨m, hle, hlt⟩ := exists_mem_Ico_zpow (inv_pos.2 hx) hy
  have hyp : 0 < y := lt_trans zero_lt_one hy
                /-
                  α : Type u_1
                  inst✝² : LinearOrderedSemifield α
                  inst✝¹ : Archimedean α
                  x y : α
                  inst✝ : ExistsAddOfLE α
                  hx : LT.lt 0 x
                  hy : LT.lt 1 y
                  m : Int
                  hle : LE.le (HPow.hPow y m) (Inv.inv x)
                  hlt : LT.lt (Inv.inv x) (HPow.hPow y (HAdd.hAdd m 1))
                  hyp : LT.lt 0 y
                  ⊢ LT.lt (HPow.hPow y (Neg.neg (HAdd.hAdd m 1))) x
                -/
  ⟨-(m + 1), by rwa [zpow_neg, inv_lt_comm₀ (zpow_pos hyp _) hx], by
                /-
                  🎉 no goals
                -/
    /-
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      hx : LT.lt 0 x
      hy : LT.lt 1 y
      m : Int
      hle : LE.le (HPow.hPow y m) (Inv.inv x)
      hlt : LT.lt (Inv.inv x) (HPow.hPow y (HAdd.hAdd m 1))
      hyp : LT.lt 0 y
      ⊢ LE.le x (HPow.hPow y (HAdd.hAdd (Neg.neg (HAdd.hAdd m 1)) 1))
    -/
    rwa [neg_add, neg_add_cancel_right, zpow_neg, le_inv_comm₀ hx (zpow_pos hyp _)]⟩
    /-
      🎉 no goals
    -/


/-- For any `y < 1` and any positive `x`, there exists `n : ℕ` with `y ^ n < x`. -/
theorem exists_pow_lt_of_lt_one (hx : 0 < x) (hy : y < 1) : ∃ n : ℕ, y ^ n < x := by
  /-
    α : Type u_1
    inst✝² : LinearOrderedSemifield α
    inst✝¹ : Archimedean α
    x y : α
    inst✝ : ExistsAddOfLE α
    hx : LT.lt 0 x
    hy : LT.lt y 1
    ⊢ Exists fun n => LT.lt (HPow.hPow y n) x
  -/
  by_cases y_pos : y ≤ 0
    /-
      case pos
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      hx : LT.lt 0 x
      hy : LT.lt y 1
      y_pos : LE.le y 0
      ⊢ Exists fun n => LT.lt (HPow.hPow y n) x
    -/
  · use 1
    /-
      case h
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      hx : LT.lt 0 x
      hy : LT.lt y 1
      y_pos : LE.le y 0
      ⊢ LT.lt (HPow.hPow y 1) x
    -/
    simp only [pow_one]
    /-
      case h
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      hx : LT.lt 0 x
      hy : LT.lt y 1
      y_pos : LE.le y 0
      ⊢ LT.lt y x
    -/
    exact y_pos.trans_lt hx
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝² : LinearOrderedSemifield α
    inst✝¹ : Archimedean α
    x y : α
    inst✝ : ExistsAddOfLE α
    hx : LT.lt 0 x
    hy : LT.lt y 1
    y_pos : Not (LE.le y 0)
    ⊢ Exists fun n => LT.lt (HPow.hPow y n) x
  -/
  rw [not_le] at y_pos
  /-
    case neg
    α : Type u_1
    inst✝² : LinearOrderedSemifield α
    inst✝¹ : Archimedean α
    x y : α
    inst✝ : ExistsAddOfLE α
    hx : LT.lt 0 x
    hy : LT.lt y 1
    y_pos : LT.lt 0 y
    ⊢ Exists fun n => LT.lt (HPow.hPow y n) x
  -/
  rcases pow_unbounded_of_one_lt x⁻¹ ((one_lt_inv₀ y_pos).2 hy) with ⟨q, hq⟩
  /-
    case neg.intro
    α : Type u_1
    inst✝² : LinearOrderedSemifield α
    inst✝¹ : Archimedean α
    x y : α
    inst✝ : ExistsAddOfLE α
    hx : LT.lt 0 x
    hy : LT.lt y 1
    y_pos : LT.lt 0 y
    q : Nat
    hq : LT.lt (Inv.inv x) (HPow.hPow (Inv.inv y) q)
    ⊢ Exists fun n => LT.lt (HPow.hPow y n) x
  -/
  exact ⟨q, by rwa [inv_pow, inv_lt_inv₀ hx (pow_pos y_pos _)] at hq⟩
  /-
    🎉 no goals
  -/


/-- Given `x` and `y` between `0` and `1`, `x` is between two successive powers of `y`.
This is the same as `exists_nat_pow_near`, but for elements between `0` and `1` -/
theorem exists_nat_pow_near_of_lt_one (xpos : 0 < x) (hx : x ≤ 1) (ypos : 0 < y) (hy : y < 1) :
    ∃ n : ℕ, y ^ (n + 1) < x ∧ x ≤ y ^ n := by
  rcases exists_nat_pow_near (one_le_inv_iff₀.2 ⟨xpos, hx⟩) (one_lt_inv_iff₀.2 ⟨ypos, hy⟩) with
    ⟨n, hn, h'n⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : LinearOrderedSemifield α
    inst✝¹ : Archimedean α
    x y : α
    inst✝ : ExistsAddOfLE α
    xpos : LT.lt 0 x
    hx : LE.le x 1
    ypos : LT.lt 0 y
    hy : LT.lt y 1
    n : Nat
    hn : LE.le (HPow.hPow (Inv.inv y) n) (Inv.inv x)
    h'n : LT.lt (Inv.inv x) (HPow.hPow (Inv.inv y) (HAdd.hAdd n 1))
    ⊢ Exists fun n => And (LT.lt (HPow.hPow y (HAdd.hAdd n 1)) x) (LE.le x (HPow.h …
  -/
  refine ⟨n, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      xpos : LT.lt 0 x
      hx : LE.le x 1
      ypos : LT.lt 0 y
      hy : LT.lt y 1
      n : Nat
      hn : LE.le (HPow.hPow (Inv.inv y) n) (Inv.inv x)
      h'n : LT.lt (Inv.inv x) (HPow.hPow (Inv.inv y) (HAdd.hAdd n 1))
      ⊢ LT.lt (HPow.hPow y (HAdd.hAdd n 1)) x
    -/
  · rwa [inv_pow, inv_lt_inv₀ xpos (pow_pos ypos _)] at h'n
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝² : LinearOrderedSemifield α
      inst✝¹ : Archimedean α
      x y : α
      inst✝ : ExistsAddOfLE α
      xpos : LT.lt 0 x
      hx : LE.le x 1
      ypos : LT.lt 0 y
      hy : LT.lt y 1
      n : Nat
      hn : LE.le (HPow.hPow (Inv.inv y) n) (Inv.inv x)
      h'n : LT.lt (Inv.inv x) (HPow.hPow (Inv.inv y) (HAdd.hAdd n 1))
      ⊢ LE.le x (HPow.hPow y n)
    -/
  · rwa [inv_pow, inv_le_inv₀ (pow_pos ypos _) xpos] at hn
    /-
      🎉 no goals
    -/


theorem archimedean_iff_nat_lt : Archimedean α ↔ ∀ x : α, ∃ n : ℕ, x < n :=
  ⟨@exists_nat_gt α _, fun H =>
    ⟨fun x y y0 =>
                                                /-
                                                  α : Type u_1
                                                  inst✝ : LinearOrderedField α
                                                  H : ∀ (x : α), Exists fun n => LT.lt x ↑n
                                                  x y : α
                                                  y0 : LT.lt 0 y
                                                  n : Nat
                                                  h : LT.lt (HDiv.hDiv x y) ↑n
                                                  ⊢ LT.lt x (HSMul.hSMul n y)
                                                -/
      (H (x / y)).imp fun n h => le_of_lt <| by rwa [div_lt_iff₀ y0, ← nsmul_eq_mul] at h⟩⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem archimedean_iff_nat_le : Archimedean α ↔ ∀ x : α, ∃ n : ℕ, x ≤ n :=
  archimedean_iff_nat_lt.trans
    ⟨fun H x => (H x).imp fun _ => le_of_lt, fun H x =>
      let ⟨n, h⟩ := H x
      ⟨n + 1, lt_of_le_of_lt h (Nat.cast_lt.2 (lt_add_one _))⟩⟩


theorem archimedean_iff_int_lt : Archimedean α ↔ ∀ x : α, ∃ n : ℤ, x < n :=
  ⟨@exists_int_gt α _, by
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      ⊢ (∀ (x : α), Exists fun n => LT.lt x ↑n) → Archimedean α
    -/
    rw [archimedean_iff_nat_lt]
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      ⊢ (∀ (x : α), Exists fun n => LT.lt x ↑n) → ∀ (x : α), Exists fun n => LT.lt x …
    -/
    intro h x
    /-
      α : Type u_1
      inst✝ : LinearOrderedField α
      h : ∀ (x : α), Exists fun n => LT.lt x ↑n
      x : α
      ⊢ Exists fun n => LT.lt x ↑n
    -/
    obtain ⟨n, h⟩ := h x
    /-
      case intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      h✝ : ∀ (x : α), Exists fun n => LT.lt x ↑n
      x : α
      n : Int
      h : LT.lt x ↑n
      ⊢ Exists fun n => LT.lt x ↑n
    -/
    refine ⟨n.toNat, h.trans_le ?_⟩
    /-
      case intro
      α : Type u_1
      inst✝ : LinearOrderedField α
      h✝ : ∀ (x : α), Exists fun n => LT.lt x ↑n
      x : α
      n : Int
      h : LT.lt x ↑n
      ⊢ LE.le ↑n ↑n.toNat
    -/
    exact mod_cast Int.self_le_toNat _⟩
    /-
      🎉 no goals
    -/


theorem archimedean_iff_int_le : Archimedean α ↔ ∀ x : α, ∃ n : ℤ, x ≤ n :=
  archimedean_iff_int_lt.trans
    ⟨fun H x => (H x).imp fun _ => le_of_lt, fun H x =>
      let ⟨n, h⟩ := H x
      ⟨n + 1, lt_of_le_of_lt h (Int.cast_lt.2 (lt_add_one _))⟩⟩


theorem archimedean_iff_rat_lt : Archimedean α ↔ ∀ x : α, ∃ q : ℚ, x < q where
  mp _ x :=
    let ⟨n, h⟩ := exists_nat_gt x
           /-
             α : Type u_1
             inst✝ : LinearOrderedField α
             x✝ : Archimedean α
             x : α
             n : Nat
             h : LT.lt x ↑n
             ⊢ LT.lt x ↑↑n
           -/
    ⟨n, by rwa [Rat.cast_natCast]⟩
           /-
             🎉 no goals
           -/
  mpr H := archimedean_iff_nat_lt.2 fun x ↦
    let ⟨q, h⟩ := H x; ⟨⌈q⌉₊, lt_of_lt_of_le h <| mod_cast Nat.le_ceil _⟩


theorem archimedean_iff_rat_le : Archimedean α ↔ ∀ x : α, ∃ q : ℚ, x ≤ q :=
  archimedean_iff_rat_lt.trans
    ⟨fun H x => (H x).imp fun _ => le_of_lt, fun H x =>
      let ⟨n, h⟩ := H x
      ⟨n + 1, lt_of_le_of_lt h (Rat.cast_lt.2 (lt_add_one _))⟩⟩


instance : Archimedean ℚ :=
                                           /-
                                             α : Type u_1
                                             M : Type u_2
                                             inst✝ : LinearOrderedField α
                                             q : Rat
                                             ⊢ LE.le q ↑q
                                           -/
  archimedean_iff_rat_le.2 fun q => ⟨q, by rw [Rat.cast_id]⟩
                                           /-
                                             🎉 no goals
                                           -/


theorem exists_rat_gt (x : α) : ∃ q : ℚ, x < q := archimedean_iff_rat_lt.mp ‹_› _


theorem exists_rat_lt (x : α) : ∃ q : ℚ, (q : α) < x :=
  let ⟨n, h⟩ := exists_int_lt x
         /-
           α : Type u_1
           inst✝¹ : LinearOrderedField α
           inst✝ : Archimedean α
           x : α
           n : Int
           h : LT.lt (↑n) x
           ⊢ LT.lt (↑↑n) x
         -/
  ⟨n, by rwa [Rat.cast_intCast]⟩
         /-
           🎉 no goals
         -/


theorem exists_rat_btwn {x y : α} (h : x < y) : ∃ q : ℚ, x < q ∧ (q : α) < y := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    ⊢ Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) y)
  -/
  cases' exists_nat_gt (y - x)⁻¹ with n nh
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    n : Nat
    nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
    ⊢ Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) y)
  -/
  cases' exists_floor (x * n) with z zh
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    n : Nat
    nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
    z : Int
    zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
    ⊢ Exists fun q => And (LT.lt x ↑q) (LT.lt (↑q) y)
  -/
  refine ⟨(z + 1 : ℤ) / n, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    n : Nat
    nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
    z : Int
    zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
    ⊢ And (LT.lt x ↑(HDiv.hDiv ↑(HAdd.hAdd z 1) ↑n)) (LT.lt (↑(HDiv.hDiv ↑(HAdd.hA …
  -/
  have n0' := (inv_pos.2 (sub_pos.2 h)).trans nh
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    n : Nat
    nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
    z : Int
    zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
    n0' : LT.lt 0 ↑n
    ⊢ And (LT.lt x ↑(HDiv.hDiv ↑(HAdd.hAdd z 1) ↑n)) (LT.lt (↑(HDiv.hDiv ↑(HAdd.hA …
  -/
  have n0 := Nat.cast_pos.1 n0'
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x y : α
    h : LT.lt x y
    n : Nat
    nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
    z : Int
    zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
    n0' : LT.lt 0 ↑n
    n0 : LT.lt 0 n
    ⊢ And (LT.lt x ↑(HDiv.hDiv ↑(HAdd.hAdd z 1) ↑n)) (LT.lt (↑(HDiv.hDiv ↑(HAdd.hA …
  -/
  rw [Rat.cast_div_of_ne_zero, Rat.cast_natCast, Rat.cast_intCast, div_lt_iff₀ n0']
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ And (LT.lt x (HDiv.hDiv ↑(HAdd.hAdd z 1) ↑n)) (LT.lt (↑(HAdd.hAdd z 1)) (HMu …
    -/
  · refine ⟨(lt_div_iff₀ n0').2 <| (lt_iff_lt_of_le_iff_le (zh _)).1 (lt_add_one _), ?_⟩
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ LT.lt (↑(HAdd.hAdd z 1)) (HMul.hMul y ↑n)
    -/
    rw [Int.cast_add, Int.cast_one]
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ LT.lt (HAdd.hAdd (↑z) 1) (HMul.hMul y ↑n)
    -/
    refine lt_of_le_of_lt (add_le_add_right ((zh _).1 le_rfl) _) ?_
    /-
      case intro.intro
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul x ↑n) 1) (HMul.hMul y ↑n)
    -/
    rwa [← lt_sub_iff_add_lt', ← sub_mul, ← div_lt_iff₀' (sub_pos.2 h), one_div]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hp
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ Ne (↑(↑(HAdd.hAdd z 1)).den) 0
    -/
  · rw [Rat.den_intCast, Nat.cast_one]
    /-
      case intro.intro.hp
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ Ne 1 0
    -/
    exact one_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.hq
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      ⊢ Ne (↑(↑n).num) 0
    -/
  · intro H
    /-
      case intro.intro.hq
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      H : Eq (↑(↑n).num) 0
      ⊢ False
    -/
    rw [Rat.num_natCast, Int.cast_natCast, Nat.cast_eq_zero] at H
    /-
      case intro.intro.hq
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      n : Nat
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑n
      z : Int
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑n))
      n0' : LT.lt 0 ↑n
      n0 : LT.lt 0 n
      H : Eq n 0
      ⊢ False
    -/
    subst H
    /-
      case intro.intro.hq
      α : Type u_1
      inst✝¹ : LinearOrderedField α
      inst✝ : Archimedean α
      x y : α
      h : LT.lt x y
      z : Int
      nh : LT.lt (Inv.inv (HSub.hSub y x)) ↑0
      zh : ∀ (z_1 : Int), Iff (LE.le z_1 z) (LE.le (↑z_1) (HMul.hMul x ↑0))
      n0' : LT.lt 0 ↑0
      n0 : LT.lt 0 0
      ⊢ False
    -/
    cases n0
    /-
      🎉 no goals
    -/


theorem exists_pow_btwn {n : ℕ} (hn : n ≠ 0) {x y : α} (h : x < y) (hy : 0 < y) :
    ∃ q : α, 0 < q ∧ x < q ^ n ∧ q ^ n < y := by
  have ⟨δ, δ_pos, cont⟩ := uniform_continuous_npow_on_bounded (max 1 y)
    (sub_pos.mpr <| max_lt_iff.mpr ⟨h, hy⟩) n
  have ex : ∃ m : ℕ, y ≤ (m * δ) ^ n := by
    have ⟨m, hm⟩ := exists_nat_ge (y / δ + 1 / δ)
    refine ⟨m, le_trans ?_ (le_self_pow₀ ?_ hn)⟩ <;> rw [← div_le_iff₀ δ_pos]
    · exact (lt_add_of_pos_right _ <| by positivity).le.trans hm
    · exact (le_add_of_nonneg_left <| by positivity).trans hm
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    δ : α
    δ_pos : GT.gt δ 0
    cont : ∀ (q r : α), LE.le (abs r) (Max.max 1 y) → LE.le (abs (HSub.hSub q r))  …
    ex : Exists fun m => LE.le y (HPow.hPow (HMul.hMul (↑m) δ) n)
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow q n)) (LT.lt (HPow. …
  -/
  let m := Nat.find ex
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    δ : α
    δ_pos : GT.gt δ 0
    cont : ∀ (q r : α), LE.le (abs r) (Max.max 1 y) → LE.le (abs (HSub.hSub q r))  …
    ex : Exists fun m => LE.le y (HPow.hPow (HMul.hMul (↑m) δ) n)
    m : Nat := Nat.find ex
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow q n)) (LT.lt (HPow. …
  -/
  have m_pos : 0 < m := (Nat.find_pos _).mpr <| by simpa [zero_pow hn] using hy
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    δ : α
    δ_pos : GT.gt δ 0
    cont : ∀ (q r : α), LE.le (abs r) (Max.max 1 y) → LE.le (abs (HSub.hSub q r))  …
    ex : Exists fun m => LE.le y (HPow.hPow (HMul.hMul (↑m) δ) n)
    m : Nat := Nat.find ex
    m_pos : LT.lt 0 m
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow q n)) (LT.lt (HPow. …
  -/
  let q := m.pred * δ
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    δ : α
    δ_pos : GT.gt δ 0
    cont : ∀ (q r : α), LE.le (abs r) (Max.max 1 y) → LE.le (abs (HSub.hSub q r))  …
    ex : Exists fun m => LE.le y (HPow.hPow (HMul.hMul (↑m) δ) n)
    m : Nat := Nat.find ex
    m_pos : LT.lt 0 m
    q : α := HMul.hMul (↑m.pred) δ
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow q n)) (LT.lt (HPow. …
  -/
  have qny : q ^ n < y := lt_of_not_le (Nat.find_min ex <| Nat.pred_lt m_pos.ne')
  have q1y : |q| < max 1 y := (abs_eq_self.mpr <| by positivity).trans_lt <| lt_max_iff.mpr
    (or_iff_not_imp_left.mpr fun q1 ↦ (le_self_pow₀ (le_of_not_lt q1) hn).trans_lt qny)
  have xqn : max x 0 < q ^ n :=
    calc _ = y - (y - max x 0) := by rw [sub_sub_cancel]
      _ ≤ (m * δ) ^ n - (y - max x 0) := sub_le_sub_right (Nat.find_spec ex) _
      _ < (m * δ) ^ n - ((m * δ) ^ n - q ^ n) := by
        refine sub_lt_sub_left ((le_abs_self _).trans_lt <| cont _ _ q1y.le ?_) _
        rw [← Nat.succ_pred_eq_of_pos m_pos, Nat.cast_succ, ← sub_mul,
          add_sub_cancel_left, one_mul, abs_eq_self.mpr (by positivity)]
      _ = q ^ n := sub_sub_cancel ..
  exact ⟨q, lt_of_le_of_ne (by positivity) fun q0 ↦
    (le_sup_right.trans_lt xqn).ne <| q0 ▸ (zero_pow hn).symm, le_sup_left.trans_lt xqn, qny⟩


@[deprecated (since := "2024-12-26")] alias exists_rat_pow_btwn_rat := exists_pow_btwn


/-- There is a rational power between any two positive elements of an archimedean ordered field. -/
theorem exists_rat_pow_btwn {n : ℕ} (hn : n ≠ 0) {x y : α} (h : x < y) (hy : 0 < y) :
    ∃ q : ℚ, 0 < q ∧ x < (q : α) ^ n ∧ (q : α) ^ n < y := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
  obtain ⟨q₂, hx₂, hy₂⟩ := exists_rat_btwn (max_lt h hy)
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    q₂ : Rat
    hx₂ : LT.lt (Max.max x 0) ↑q₂
    hy₂ : LT.lt (↑q₂) y
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
  obtain ⟨q₁, hx₁, hq₁₂⟩ := exists_rat_btwn hx₂
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    q₂ : Rat
    hx₂ : LT.lt (Max.max x 0) ↑q₂
    hy₂ : LT.lt (↑q₂) y
    q₁ : Rat
    hx₁ : LT.lt (Max.max x 0) ↑q₁
    hq₁₂ : LT.lt ↑q₁ ↑q₂
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
  have : (0 : α) < q₂ := (le_max_right _ _).trans_lt hx₂
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    q₂ : Rat
    hx₂ : LT.lt (Max.max x 0) ↑q₂
    hy₂ : LT.lt (↑q₂) y
    q₁ : Rat
    hx₁ : LT.lt (Max.max x 0) ↑q₁
    hq₁₂ : LT.lt ↑q₁ ↑q₂
    this : LT.lt 0 ↑q₂
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
  norm_cast at hq₁₂ this
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    q₂ : Rat
    hx₂ : LT.lt (Max.max x 0) ↑q₂
    hy₂ : LT.lt (↑q₂) y
    q₁ : Rat
    hx₁ : LT.lt (Max.max x 0) ↑q₁
    hq₁₂ : LT.lt q₁ q₂
    this : LT.lt 0 q₂
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
  obtain ⟨q, hq, hq₁, hq₂⟩ := exists_pow_btwn hn hq₁₂ this
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    n : Nat
    hn : Ne n 0
    x y : α
    h : LT.lt x y
    hy : LT.lt 0 y
    q₂ : Rat
    hx₂ : LT.lt (Max.max x 0) ↑q₂
    hy₂ : LT.lt (↑q₂) y
    q₁ : Rat
    hx₁ : LT.lt (Max.max x 0) ↑q₁
    hq₁₂ : LT.lt q₁ q₂
    this : LT.lt 0 q₂
    q : Rat
    hq : LT.lt 0 q
    hq₁ : LT.lt q₁ (HPow.hPow q n)
    hq₂ : LT.lt (HPow.hPow q n) q₂
    ⊢ Exists fun q => And (LT.lt 0 q) (And (LT.lt x (HPow.hPow (↑q) n)) (LT.lt (HP …
  -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  refine ⟨q, hq, (le_max_left _ _).trans_lt <| hx₁.trans ?_, hy₂.trans' ?_⟩ <;> assumption_mod_cast
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem le_of_forall_rat_lt_imp_le (h : ∀ q : ℚ, (q : α) < x → (q : α) ≤ y) : x ≤ y :=
  le_of_not_lt fun hyx =>
    let ⟨_, hy, hx⟩ := exists_rat_btwn hyx
    hy.not_le <| h _ hx


theorem le_of_forall_lt_rat_imp_le (h : ∀ q : ℚ, y < q → x ≤ q) : x ≤ y :=
  le_of_not_lt fun hyx =>
    let ⟨_, hy, hx⟩ := exists_rat_btwn hyx
    hx.not_le <| h _ hy


theorem le_iff_forall_rat_lt_imp_le : x ≤ y ↔ ∀ q : ℚ, (q : α) < x → (q : α) ≤ y :=
  ⟨fun hxy _ hqx ↦ hqx.le.trans hxy, le_of_forall_rat_lt_imp_le⟩


theorem le_iff_forall_lt_rat_imp_le : x ≤ y ↔ ∀ q : ℚ, y < q → x ≤ q :=
  ⟨fun hxy _ hqx ↦ hxy.trans hqx.le, le_of_forall_lt_rat_imp_le⟩


theorem eq_of_forall_rat_lt_iff_lt (h : ∀ q : ℚ, (q : α) < x ↔ (q : α) < y) : x = y :=
  (le_of_forall_rat_lt_imp_le fun q hq => ((h q).1 hq).le).antisymm <|
    le_of_forall_rat_lt_imp_le fun q hq => ((h q).2 hq).le


theorem eq_of_forall_lt_rat_iff_lt (h : ∀ q : ℚ, x < q ↔ y < q) : x = y :=
  (le_of_forall_lt_rat_imp_le fun q hq => ((h q).2 hq).le).antisymm <|
    le_of_forall_lt_rat_imp_le fun q hq => ((h q).1 hq).le


theorem exists_pos_rat_lt {x : α} (x0 : 0 < x) : ∃ q : ℚ, 0 < q ∧ (q : α) < x := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : Archimedean α
    x : α
    x0 : LT.lt 0 x
    ⊢ Exists fun q => And (LT.lt 0 q) (LT.lt (↑q) x)
  -/
  simpa only [Rat.cast_pos] using exists_rat_btwn x0
  /-
    🎉 no goals
  -/


theorem exists_rat_near (x : α) (ε0 : 0 < ε) : ∃ q : ℚ, |x - q| < ε :=
  let ⟨q, h₁, h₂⟩ :=
    exists_rat_btwn <| ((sub_lt_self_iff x).2 ε0).trans ((lt_add_iff_pos_left x).2 ε0)
  ⟨q, abs_sub_lt_iff.2 ⟨sub_lt_comm.1 h₁, sub_lt_iff_lt_add.2 h₂⟩⟩


instance : Archimedean ℕ :=
  ⟨fun n m m0 => ⟨n, by
    /-
      α : Type u_1
      M : Type u_2
      n m : Nat
      m0 : LT.lt 0 m
      ⊢ LE.le n (HSMul.hSMul n m)
    -/
    rw [← mul_one n, smul_eq_mul, mul_assoc, one_mul m]
    /-
      α : Type u_1
      M : Type u_2
      n m : Nat
      m0 : LT.lt 0 m
      ⊢ LE.le (HMul.hMul n 1) (HMul.hMul n m)
    -/
    exact Nat.mul_le_mul_left n (by omega)⟩⟩
    /-
      🎉 no goals
    -/


instance : Archimedean ℤ :=
  ⟨fun n m m0 =>
    ⟨n.toNat,
      le_trans (Int.self_le_toNat _) <| by
        simpa only [nsmul_eq_mul, zero_add, mul_one] using
          mul_le_mul_of_nonneg_left (Int.add_one_le_iff.2 m0) (Int.ofNat_zero_le n.toNat)⟩⟩


instance Nonneg.instArchimedean [OrderedAddCommMonoid α] [Archimedean α] :
    Archimedean { x : α // 0 ≤ x } :=
  ⟨fun x y hy =>
    let ⟨n, hr⟩ := Archimedean.arch (x : α) (hy : (0 : α) < y)
                                                       /-
                                                         α : Type u_1
                                                         M : Type u_2
                                                         inst✝¹ : OrderedAddCommMonoid α
                                                         inst✝ : Archimedean α
                                                         x y : Subtype fun x => LE.le 0 x
                                                         hy : LT.lt 0 y
                                                         n : Nat
                                                         hr : LE.le (↑x) (HSMul.hSMul n ((fun a => ↑a) y))
                                                         ⊢ LE.le ↑x ↑(HSMul.hSMul n y)
                                                       -/
    ⟨n, show (x : α) ≤ (n • y : { x : α // 0 ≤ x }) by simp [*, -nsmul_eq_mul, nsmul_coe]⟩⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


instance Nonneg.instMulArchimedean [StrictOrderedCommSemiring α] [Archimedean α] [ExistsAddOfLE α] :
    MulArchimedean { x : α // 0 ≤ x } :=
  ⟨fun x _ hy ↦ (pow_unbounded_of_one_lt x hy).imp fun _ h ↦ h.le⟩


instance : Archimedean NNRat := Nonneg.instArchimedean

instance : MulArchimedean NNRat := Nonneg.instMulArchimedean


/-- A linear ordered archimedean ring is a floor ring. This is not an `instance` because in some
cases we have a computable `floor` function. -/
noncomputable def Archimedean.floorRing (α) [LinearOrderedRing α] [Archimedean α] : FloorRing α :=
  FloorRing.ofFloor α (fun a => Classical.choose (exists_floor a)) fun z a =>
    (Classical.choose_spec (exists_floor a) z).symm

-- see Note [lower instance priority]

/-- A linear ordered field that is a floor ring is archimedean. -/
instance (priority := 100) FloorRing.archimedean (α) [LinearOrderedField α] [FloorRing α] :
    Archimedean α := by
  /-
    α✝ : Type u_1
    M : Type u_2
    α : Type u_3
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    ⊢ Archimedean α
  -/
  rw [archimedean_iff_int_le]
  /-
    α✝ : Type u_1
    M : Type u_2
    α : Type u_3
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    ⊢ ∀ (x : α), Exists fun n => LE.le x ↑n
  -/
  exact fun x => ⟨⌈x⌉, Int.le_ceil x⟩
  /-
    🎉 no goals
  -/


@[to_additive]
instance Units.instMulArchimedean (α) [OrderedCommMonoid α] [MulArchimedean α] :
    MulArchimedean αˣ :=
  ⟨fun x {_} h ↦ MulArchimedean.arch x.val h⟩

