theorem eq_zero_of_zero_dvd (h : 0 ∣ a) : a = 0 :=
  Dvd.elim h fun c H' => H'.trans (zero_mul c)


/-- Given an element `a` of a commutative semigroup with zero, there exists another element whose
    product with zero equals `a` iff `a` equals zero. -/
@[simp]
theorem zero_dvd_iff : 0 ∣ a ↔ a = 0 :=
  ⟨eq_zero_of_zero_dvd, fun h => by
    /-
      α : Type u_1
      inst✝ : SemigroupWithZero α
      a : α
      h : Eq a 0
      ⊢ Dvd.dvd 0 a
    -/
    rw [h]
    /-
      α : Type u_1
      inst✝ : SemigroupWithZero α
      a : α
      h : Eq a 0
      ⊢ Dvd.dvd 0 0
    -/
    exact ⟨0, by simp⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem dvd_zero (a : α) : a ∣ 0 :=
                  /-
                    α : Type u_1
                    inst✝ : SemigroupWithZero α
                    a : α
                    ⊢ Eq (HMul.hMul a 0) 0
                  -/
  Dvd.intro 0 (by simp)
                  /-
                    🎉 no goals
                  -/


/-- Given two elements `b`, `c` of a `CancelMonoidWithZero` and a nonzero element `a`,
 `a*b` divides `a*c` iff `b` divides `c`. -/
theorem mul_dvd_mul_iff_left [CancelMonoidWithZero α] {a b c : α} (ha : a ≠ 0) :
    a * b ∣ a * c ↔ b ∣ c :=
                           /-
                             α : Type u_1
                             inst✝ : CancelMonoidWithZero α
                             a b c : α
                             ha : Ne a 0
                             d : α
                             ⊢ Iff (Eq (HMul.hMul a c) (HMul.hMul (HMul.hMul a b) d)) (Eq c (HMul.hMul b d))
                           -/
  exists_congr fun d => by rw [mul_assoc, mul_right_inj' ha]
                           /-
                             🎉 no goals
                           -/


/-- Given two elements `a`, `b` of a commutative `CancelMonoidWithZero` and a nonzero
  element `c`, `a*c` divides `b*c` iff `a` divides `b`. -/
theorem mul_dvd_mul_iff_right [CancelCommMonoidWithZero α] {a b c : α} (hc : c ≠ 0) :
    a * c ∣ b * c ↔ a ∣ b :=
                           /-
                             α : Type u_1
                             inst✝ : CancelCommMonoidWithZero α
                             a b c : α
                             hc : Ne c 0
                             d : α
                             ⊢ Iff (Eq (HMul.hMul b c) (HMul.hMul (HMul.hMul a c) d)) (Eq b (HMul.hMul a d))
                           -/
  exists_congr fun d => by rw [mul_right_comm, mul_left_inj' hc]
                           /-
                             🎉 no goals
                           -/


/-- `DvdNotUnit a b` expresses that `a` divides `b` "strictly", i.e. that `b` divided by `a`
is not a unit. -/
def DvdNotUnit (a b : α) : Prop :=
  a ≠ 0 ∧ ∃ x, ¬IsUnit x ∧ b = a * x


theorem dvdNotUnit_of_dvd_of_not_dvd {a b : α} (hd : a ∣ b) (hnd : ¬b ∣ a) : DvdNotUnit a b := by
  /-
    α : Type u_1
    inst✝ : CommMonoidWithZero α
    a b : α
    hd : Dvd.dvd a b
    hnd : Not (Dvd.dvd b a)
    ⊢ DvdNotUnit a b
  -/
  constructor
    /-
      case left
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      a b : α
      hd : Dvd.dvd a b
      hnd : Not (Dvd.dvd b a)
      ⊢ Ne a 0
    -/
  · rintro rfl
    /-
      case left
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      b : α
      hd : Dvd.dvd 0 b
      hnd : Not (Dvd.dvd b 0)
      ⊢ False
    -/
    exact hnd (dvd_zero _)
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      a b : α
      hd : Dvd.dvd a b
      hnd : Not (Dvd.dvd b a)
      ⊢ Exists fun x => And (Not (IsUnit x)) (Eq b (HMul.hMul a x))
    -/
  · rcases hd with ⟨c, rfl⟩
    /-
      case right.intro
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      a c : α
      hnd : Not (Dvd.dvd (HMul.hMul a c) a)
      ⊢ Exists fun x => And (Not (IsUnit x)) (Eq (HMul.hMul a c) (HMul.hMul a x))
    -/
    refine ⟨c, ?_, rfl⟩
    /-
      case right.intro
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      a c : α
      hnd : Not (Dvd.dvd (HMul.hMul a c) a)
      ⊢ Not (IsUnit c)
    -/
    rintro ⟨u, rfl⟩
    /-
      case right.intro.intro
      α : Type u_1
      inst✝ : CommMonoidWithZero α
      a : α
      u : Units α
      hnd : Not (Dvd.dvd (HMul.hMul a ↑u) a)
      ⊢ False
    -/
    simp at hnd
    /-
      🎉 no goals
    -/


theorem isRelPrime_zero_left : IsRelPrime 0 x ↔ IsUnit x :=
  ⟨(· (dvd_zero _) dvd_rfl), IsUnit.isRelPrime_right⟩


theorem isRelPrime_zero_right : IsRelPrime x 0 ↔ IsUnit x :=
  isRelPrime_comm.trans isRelPrime_zero_left


theorem not_isRelPrime_zero_zero [Nontrivial α] : ¬IsRelPrime (0 : α) 0 :=
  mt isRelPrime_zero_right.mp not_isUnit_zero


theorem IsRelPrime.ne_zero_or_ne_zero [Nontrivial α] (h : IsRelPrime x y) : x ≠ 0 ∨ y ≠ 0 :=
                      /-
                        α : Type u_1
                        inst✝¹ : CommMonoidWithZero α
                        x y : α
                        inst✝ : Nontrivial α
                        h : IsRelPrime x y
                        ⊢ Eq x 0 → Ne y 0
                      -/
  not_or_of_imp <| by rintro rfl rfl; exact not_isRelPrime_zero_zero h
                                      /-
                                        🎉 no goals
                                      -/


theorem isRelPrime_of_no_nonunits_factors [MonoidWithZero α] {x y : α} (nonzero : ¬(x = 0 ∧ y = 0))
    (H : ∀ z, ¬ IsUnit z → z ≠ 0 → z ∣ x → ¬z ∣ y) : IsRelPrime x y := by
  /-
    α : Type u_1
    inst✝ : MonoidWithZero α
    x y : α
    nonzero : Not (And (Eq x 0) (Eq y 0))
    H : ∀ (z : α), Not (IsUnit z) → Ne z 0 → Dvd.dvd z x → Not (Dvd.dvd z y)
    ⊢ IsRelPrime x y
  -/
  refine fun z hx hy ↦ by_contra fun h ↦ H z h ?_ hx hy
  /-
    α : Type u_1
    inst✝ : MonoidWithZero α
    x y : α
    nonzero : Not (And (Eq x 0) (Eq y 0))
    H : ∀ (z : α), Not (IsUnit z) → Ne z 0 → Dvd.dvd z x → Not (Dvd.dvd z y)
    z : α
    hx : Dvd.dvd z x
    hy : Dvd.dvd z y
    h : Not (IsUnit z)
    ⊢ Ne z 0
  -/
  rintro rfl; exact nonzero ⟨zero_dvd_iff.1 hx, zero_dvd_iff.1 hy⟩
              /-
                🎉 no goals
              -/


theorem dvd_and_not_dvd_iff [CancelCommMonoidWithZero α] {x y : α} :
    x ∣ y ∧ ¬y ∣ x ↔ DvdNotUnit x y :=
  ⟨fun ⟨⟨d, hd⟩, hyx⟩ =>
                   /-
                     α : Type u_1
                     inst✝ : CancelCommMonoidWithZero α
                     x y : α
                     x✝ : And (Dvd.dvd x y) (Not (Dvd.dvd y x))
                     d : α
                     hd : Eq y (HMul.hMul x d)
                     hyx : Not (Dvd.dvd y x)
                     hx0 : Eq x 0
                     ⊢ False
                   -/
    ⟨fun hx0 => by simp [hx0] at hyx,
                   /-
                     🎉 no goals
                   -/
                                                            /-
                                                              α : Type u_1
                                                              inst✝ : CancelCommMonoidWithZero α
                                                              x y : α
                                                              x✝¹ : And (Dvd.dvd x y) (Not (Dvd.dvd y x))
                                                              d : α
                                                              hd : Eq y (HMul.hMul x d)
                                                              hyx : Not (Dvd.dvd y x)
                                                              x✝ : Dvd.dvd d 1
                                                              e : α
                                                              he : Eq 1 (HMul.hMul d e)
                                                              ⊢ Eq x (HMul.hMul y e)
                                                            -/
      ⟨d, mt isUnit_iff_dvd_one.1 fun ⟨e, he⟩ => hyx ⟨e, by rw [hd, mul_assoc, ← he, mul_one]⟩,
                                                            /-
                                                              🎉 no goals
                                                            -/
        hd⟩⟩,
    fun ⟨hx0, d, hdu, hdx⟩ =>
    ⟨⟨d, hdx⟩, fun ⟨e, he⟩ =>
      hdu
        (isUnit_of_dvd_one
          ⟨e, mul_left_cancel₀ hx0 <| by conv =>
            lhs
            rw [he, hdx]
            simp [mul_assoc]⟩)⟩⟩


theorem ne_zero_of_dvd_ne_zero {p q : α} (h₁ : q ≠ 0) (h₂ : p ∣ q) : p ≠ 0 := by
  /-
    α : Type u_1
    inst✝ : MonoidWithZero α
    p q : α
    h₁ : Ne q 0
    h₂ : Dvd.dvd p q
    ⊢ Ne p 0
  -/
  rcases h₂ with ⟨u, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝ : MonoidWithZero α
    p u : α
    h₁ : Ne (HMul.hMul p u) 0
    ⊢ Ne p 0
  -/
  exact left_ne_zero_of_mul h₁
  /-
    🎉 no goals
  -/


theorem isPrimal_zero : IsPrimal (0 : α) :=
  fun a b h ↦ ⟨a, b, dvd_rfl, dvd_rfl, (zero_dvd_iff.mp h).symm⟩


theorem IsPrimal.mul {α} [CancelCommMonoidWithZero α] {m n : α}
    (hm : IsPrimal m) (hn : IsPrimal n) : IsPrimal (m * n) := by
  /-
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    m n : α
    hm : IsPrimal m
    hn : IsPrimal n
    ⊢ IsPrimal (HMul.hMul m n)
  -/
  obtain rfl | h0 := eq_or_ne m 0; · rwa [zero_mul]
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case inr
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    m n : α
    hm : IsPrimal m
    hn : IsPrimal n
    h0 : Ne m 0
    ⊢ IsPrimal (HMul.hMul m n)
  -/
  intro b c h
  /-
    case inr
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    m n : α
    hm : IsPrimal m
    hn : IsPrimal n
    h0 : Ne m 0
    b c : α
    h : Dvd.dvd (HMul.hMul m n) (HMul.hMul b c)
    ⊢ Exists fun a₁ => Exists fun a₂ => And (Dvd.dvd a₁ b) (And (Dvd.dvd a₂ c) (Eq …
  -/
  obtain ⟨a₁, a₂, ⟨b, rfl⟩, ⟨c, rfl⟩, rfl⟩ := hm (dvd_of_mul_right_dvd h)
  /-
    case inr.intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    n : α
    hn : IsPrimal n
    a₁ a₂ b c : α
    hm : IsPrimal (HMul.hMul a₁ a₂)
    h0 : Ne (HMul.hMul a₁ a₂) 0
    h : Dvd.dvd (HMul.hMul (HMul.hMul a₁ a₂) n) (HMul.hMul (HMul.hMul a₁ b) (HMul. …
    ⊢ Exists fun a₁_1 => Exists fun a₂_1 => And (Dvd.dvd a₁_1 (HMul.hMul a₁ b)) (A …
  -/
  rw [mul_mul_mul_comm, mul_dvd_mul_iff_left h0] at h
  /-
    case inr.intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    n : α
    hn : IsPrimal n
    a₁ a₂ b c : α
    hm : IsPrimal (HMul.hMul a₁ a₂)
    h0 : Ne (HMul.hMul a₁ a₂) 0
    h : Dvd.dvd n (HMul.hMul b c)
    ⊢ Exists fun a₁_1 => Exists fun a₂_1 => And (Dvd.dvd a₁_1 (HMul.hMul a₁ b)) (A …
  -/
  obtain ⟨a₁', a₂', h₁, h₂, rfl⟩ := hn h
  /-
    case inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_2
    inst✝ : CancelCommMonoidWithZero α
    a₁ a₂ b c : α
    hm : IsPrimal (HMul.hMul a₁ a₂)
    h0 : Ne (HMul.hMul a₁ a₂) 0
    a₁' a₂' : α
    h₁ : Dvd.dvd a₁' b
    h₂ : Dvd.dvd a₂' c
    hn : IsPrimal (HMul.hMul a₁' a₂')
    h : Dvd.dvd (HMul.hMul a₁' a₂') (HMul.hMul b c)
    ⊢ Exists fun a₁_1 => Exists fun a₂_1 => And (Dvd.dvd a₁_1 (HMul.hMul a₁ b)) (A …
  -/
  exact ⟨a₁ * a₁', a₂ * a₂', mul_dvd_mul_left _ h₁, mul_dvd_mul_left _ h₂, mul_mul_mul_comm _ _ _ _⟩
  /-
    🎉 no goals
  -/


theorem dvd_antisymm : a ∣ b → b ∣ a → a = b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    a b : α
    inst✝ : Subsingleton (Units α)
    ⊢ Dvd.dvd a b → Dvd.dvd b a → Eq a b
  -/
  rintro ⟨c, rfl⟩ ⟨d, hcd⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    a : α
    inst✝ : Subsingleton (Units α)
    c d : α
    hcd : Eq a (HMul.hMul (HMul.hMul a c) d)
    ⊢ Eq a (HMul.hMul a c)
  -/
  rw [mul_assoc, eq_comm, mul_right_eq_self₀, mul_eq_one] at hcd
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    a : α
    inst✝ : Subsingleton (Units α)
    c d : α
    hcd : Or (And (Eq c 1) (Eq d 1)) (Eq a 0)
    ⊢ Eq a (HMul.hMul a c)
  -/
                                   /-
                                     🎉 no goals
                                   -/
  obtain ⟨rfl, -⟩ | rfl := hcd <;> simp
                                   /-
                                     🎉 no goals
                                   -/


theorem dvd_antisymm' : a ∣ b → b ∣ a → b = a :=
  flip dvd_antisymm


alias Dvd.dvd.antisymm := dvd_antisymm


alias Dvd.dvd.antisymm' := dvd_antisymm'


theorem eq_of_forall_dvd (h : ∀ c, a ∣ c ↔ b ∣ c) : a = b :=
  ((h _).2 dvd_rfl).antisymm <| (h _).1 dvd_rfl


theorem eq_of_forall_dvd' (h : ∀ c, c ∣ a ↔ c ∣ b) : a = b :=
  ((h _).1 dvd_rfl).antisymm <| (h _).2 dvd_rfl


lemma pow_dvd_pow_iff (ha₀ : a ≠ 0) (ha : ¬IsUnit a) : a ^ n ∣ a ^ m ↔ n ≤ m := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    a : α
    m n : Nat
    ha₀ : Ne a 0
    ha : Not (IsUnit a)
    ⊢ Iff (Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)) (LE.le n m)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      ⊢ Dvd.dvd (HPow.hPow a n) (HPow.hPow a m) → LE.le n m
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
      ⊢ LE.le n m
    -/
    rw [← not_lt]
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
      ⊢ Not (LT.lt m n)
    -/
    intro hmn
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
      hmn : LT.lt m n
      ⊢ False
    -/
    apply ha
    have : a ^ m * a ∣ a ^ m * 1 := by
      rw [← pow_succ, mul_one]
      exact (pow_dvd_pow _ (Nat.succ_le_of_lt hmn)).trans h
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
      hmn : LT.lt m n
      this : Dvd.dvd (HMul.hMul (HPow.hPow a m) a) (HMul.hMul (HPow.hPow a m) 1)
      ⊢ IsUnit a
    -/
    rwa [mul_dvd_mul_iff_left, ← isUnit_iff_dvd_one] at this
    /-
      case mp
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      h : Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
      hmn : LT.lt m n
      this : Dvd.dvd (HMul.hMul (HPow.hPow a m) a) (HMul.hMul (HPow.hPow a m) 1)
      ⊢ Ne (HPow.hPow a m) 0
    -/
    apply pow_ne_zero m ha₀
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a : α
      m n : Nat
      ha₀ : Ne a 0
      ha : Not (IsUnit a)
      ⊢ LE.le n m → Dvd.dvd (HPow.hPow a n) (HPow.hPow a m)
    -/
  · apply pow_dvd_pow
    /-
      🎉 no goals
    -/


