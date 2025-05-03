@[simp] lemma Even.neg_pow : Even n → ∀ a : α, (-a) ^ n = a ^ n := by
  /-
    α : Type u_2
    inst✝¹ : Monoid α
    inst✝ : HasDistribNeg α
    n : Nat
    ⊢ Even n → ∀ (a : α), Eq (HPow.hPow (Neg.neg a) n) (HPow.hPow a n)
  -/
  rintro ⟨c, rfl⟩ a
  /-
    case intro
    α : Type u_2
    inst✝¹ : Monoid α
    inst✝ : HasDistribNeg α
    c : Nat
    a : α
    ⊢ Eq (HPow.hPow (Neg.neg a) (HAdd.hAdd c c)) (HPow.hPow a (HAdd.hAdd c c))
  -/
  simp_rw [← two_mul, pow_mul, neg_sq]
  /-
    🎉 no goals
  -/


                                                             /-
                                                               α : Type u_2
                                                               inst✝¹ : Monoid α
                                                               inst✝ : HasDistribNeg α
                                                               n : Nat
                                                               h : Even n
                                                               ⊢ Eq (HPow.hPow (-1) n) 1
                                                             -/
lemma Even.neg_one_pow (h : Even n) : (-1 : α) ^ n = 1 := by rw [h.neg_pow, one_pow]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma Even.neg_zpow : Even n → ∀ a : α, (-a) ^ n = a ^ n := by
  /-
    α : Type u_2
    inst✝¹ : DivisionMonoid α
    inst✝ : HasDistribNeg α
    n : Int
    ⊢ Even n → ∀ (a : α), Eq (HPow.hPow (Neg.neg a) n) (HPow.hPow a n)
  -/
  rintro ⟨c, rfl⟩ a; simp_rw [← Int.two_mul, zpow_mul, zpow_two, neg_mul_neg]
                     /-
                       🎉 no goals
                     -/


                                                              /-
                                                                α : Type u_2
                                                                inst✝¹ : DivisionMonoid α
                                                                inst✝ : HasDistribNeg α
                                                                n : Int
                                                                h : Even n
                                                                ⊢ Eq (HPow.hPow (-1) n) 1
                                                              -/
lemma Even.neg_one_zpow (h : Even n) : (-1 : α) ^ n = 1 := by rw [h.neg_zpow, one_zpow]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] lemma isSquare_zero [MulZeroClass α] : IsSquare (0 : α) := ⟨0, (mul_zero _).symm⟩


                                                              /-
                                                                α : Type u_2
                                                                inst✝ : Semiring α
                                                                a : α
                                                                ⊢ Iff (Even a) (Exists fun b => Eq a (HMul.hMul 2 b))
                                                              -/
lemma even_iff_exists_two_mul : Even a ↔ ∃ b, a = 2 * b := by simp [even_iff_exists_two_nsmul]
                                                              /-
                                                                🎉 no goals
                                                              -/


                                              /-
                                                α : Type u_2
                                                inst✝ : Semiring α
                                                a : α
                                                ⊢ Iff (Even a) (Dvd.dvd 2 a)
                                              -/
lemma even_iff_two_dvd : Even a ↔ 2 ∣ a := by simp [Even, Dvd.dvd, two_mul]
                                              /-
                                                🎉 no goals
                                              -/


alias ⟨Even.two_dvd, _⟩ := even_iff_two_dvd


lemma Even.trans_dvd (ha : Even a) (hab : a ∣ b) : Even b :=
  even_iff_two_dvd.2 <| ha.two_dvd.trans hab


lemma Dvd.dvd.even (hab : a ∣ b) (ha : Even a) : Even b := ha.trans_dvd hab


@[simp] lemma range_two_mul (α) [Semiring α] : Set.range (fun x : α ↦ 2 * x) = {a | Even a} := by
  /-
    α : Type u_4
    inst✝ : Semiring α
    ⊢ Eq (Set.range fun x => HMul.hMul 2 x) (setOf fun a => Even a)
  -/
  ext x
  /-
    case h
    α : Type u_4
    inst✝ : Semiring α
    x : α
    ⊢ Iff (Membership.mem (Set.range fun x => HMul.hMul 2 x) x) (Membership.mem (s …
  -/
  simp [eq_comm, two_mul, Even]
  /-
    🎉 no goals
  -/


                                                /-
                                                  α : Type u_2
                                                  inst✝ : Semiring α
                                                  ⊢ Eq 2 (HAdd.hAdd 1 1)
                                                -/
@[simp] lemma even_two : Even (2 : α) := ⟨1, by rw [one_add_one_eq_two]⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma Even.mul_left (ha : Even a) (b) : Even (b * a) := ha.map (AddMonoidHom.mulLeft _)


@[simp] lemma Even.mul_right (ha : Even a) (b) : Even (a * b) := ha.map (AddMonoidHom.mulRight _)


lemma even_two_mul (a : α) : Even (2 * a) := ⟨a, two_mul _⟩


lemma Even.pow_of_ne_zero (ha : Even a) : ∀ {n : ℕ}, n ≠ 0 → Even (a ^ n)
                   /-
                     α : Type u_2
                     inst✝ : Semiring α
                     a : α
                     ha : Even a
                     n : Nat
                     x✝ : Ne (HAdd.hAdd n 1) 0
                     ⊢ Even (HPow.hPow a (HAdd.hAdd n 1))
                   -/
  | n + 1, _ => by rw [pow_succ]; exact ha.mul_left _
                                  /-
                                    🎉 no goals
                                  -/


/-- An element `a` of a semiring is odd if there exists `k` such `a = 2*k + 1`. -/
def Odd (a : α) : Prop := ∃ k, a = 2 * k + 1


                                                                                  /-
                                                                                    α : Type u_2
                                                                                    inst✝ : Semiring α
                                                                                    a b : α
                                                                                    ⊢ Iff (Eq a (HAdd.hAdd (HMul.hMul 2 b) 1)) (Eq a (HAdd.hAdd (HMul.hMul 2 b) 1))
                                                                                  -/
lemma odd_iff_exists_bit1 : Odd a ↔ ∃ b, a = 2 * b + 1 := exists_congr fun b ↦ by rw [two_mul]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


alias ⟨Odd.exists_bit1, _⟩ := odd_iff_exists_bit1


@[simp] lemma range_two_mul_add_one (α : Type*) [Semiring α] :
                                                          /-
                                                            α : Type u_4
                                                            inst✝ : Semiring α
                                                            ⊢ Eq (Set.range fun x => HAdd.hAdd (HMul.hMul 2 x) 1) (setOf fun a => Odd a)
                                                          -/
    Set.range (fun x : α ↦ 2 * x + 1) = {a | Odd a} := by ext x; simp [Odd, eq_comm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma Even.add_odd : Even a → Odd b → Odd (a + b) := by
  /-
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Even a → Odd b → Odd (HAdd.hAdd a b)
  -/
  rintro ⟨a, rfl⟩ ⟨b, rfl⟩; exact ⟨a + b, by rw [mul_add, ← two_mul, add_assoc]⟩
                            /-
                              🎉 no goals
                            -/


lemma Even.odd_add (ha : Even a) (hb : Odd b) : Odd (b + a) := add_comm a b ▸ ha.add_odd hb

lemma Odd.add_even (ha : Odd a) (hb : Even b) : Odd (a + b) := add_comm a b ▸ hb.add_odd ha


lemma Odd.add_odd : Odd a → Odd b → Even (a + b) := by
  /-
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Odd a → Odd b → Even (HAdd.hAdd a b)
  -/
  rintro ⟨a, rfl⟩ ⟨b, rfl⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Even (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 a) 1) (HAdd.hAdd (HMul.hMul 2 b) 1))
  -/
  refine ⟨a + b + 1, ?_⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 a) 1) (HAdd.hAdd (HMul.hMul 2 b) 1)) ( …
  -/
  rw [two_mul, two_mul]
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd a a) 1) (HAdd.hAdd (HAdd.hAdd b b) 1)) ( …
  -/
  ac_rfl
  /-
    🎉 no goals
  -/


@[simp] lemma odd_one : Odd (1 : α) :=
  ⟨0, (zero_add _).symm.trans (congr_arg (· + (1 : α)) (mul_zero _).symm)⟩


@[simp] lemma Even.add_one (h : Even a) : Odd (a + 1) := h.add_odd odd_one

@[simp] lemma Even.one_add (h : Even a) : Odd (1 + a) := h.odd_add odd_one

@[simp] lemma Odd.add_one (h : Odd a) : Even (a + 1) := h.add_odd odd_one

@[simp] lemma Odd.one_add (h : Odd a) : Even (1 + a) := odd_one.add_odd h


lemma odd_two_mul_add_one (a : α) : Odd (2 * a + 1) := ⟨_, rfl⟩


                                                          /-
                                                            α : Type u_2
                                                            inst✝ : Semiring α
                                                            a : α
                                                            ⊢ Odd (HAdd.hAdd a (HAdd.hAdd a 1))
                                                          -/
@[simp] lemma odd_add_self_one' : Odd (a + (a + 1)) := by simp [← add_assoc]
                                                          /-
                                                            🎉 no goals
                                                          -/

                                                       /-
                                                         α : Type u_2
                                                         inst✝ : Semiring α
                                                         a : α
                                                         ⊢ Odd (HAdd.hAdd (HAdd.hAdd a 1) a)
                                                       -/
@[simp] lemma odd_add_one_self : Odd (a + 1 + a) := by simp [add_comm _ a]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                          /-
                                                            α : Type u_2
                                                            inst✝ : Semiring α
                                                            a : α
                                                            ⊢ Odd (HAdd.hAdd a (HAdd.hAdd 1 a))
                                                          -/
@[simp] lemma odd_add_one_self' : Odd (a + (1 + a)) := by simp [add_comm 1 a]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma Odd.map [FunLike F α β] [RingHomClass F α β] (f : F) : Odd a → Odd (f a) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝³ : Semiring α
    inst✝² : Semiring β
    a : α
    inst✝¹ : FunLike F α β
    inst✝ : RingHomClass F α β
    f : F
    ⊢ Odd a → Odd (f a)
  -/
  rintro ⟨a, rfl⟩; exact ⟨f a, by simp [two_mul]⟩
                   /-
                     🎉 no goals
                   -/


lemma Odd.natCast {R : Type*} [Semiring R] {n : ℕ} (hn : Odd n) : Odd (n : R) :=
  hn.map <| Nat.castRingHom R


@[simp] lemma Odd.mul : Odd a → Odd b → Odd (a * b) := by
  /-
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Odd a → Odd b → Odd (HMul.hMul a b)
  -/
  rintro ⟨a, rfl⟩ ⟨b, rfl⟩
  /-
    case intro.intro
    α : Type u_2
    inst✝ : Semiring α
    a b : α
    ⊢ Odd (HMul.hMul (HAdd.hAdd (HMul.hMul 2 a) 1) (HAdd.hAdd (HMul.hMul 2 b) 1))
  -/
  refine ⟨2 * a * b + b + a, ?_⟩
  rw [mul_add, add_mul, mul_one, ← add_assoc, one_mul, mul_assoc, ← mul_add, ← mul_add, ← mul_assoc,
    ← Nat.cast_two, ← Nat.cast_comm]


lemma Odd.pow (ha : Odd a) : ∀ {n : ℕ}, Odd (a ^ n)
  | 0 => by
    /-
      α : Type u_2
      inst✝ : Semiring α
      a : α
      ha : Odd a
      ⊢ Odd (HPow.hPow a 0)
    -/
    rw [pow_zero]
    /-
      α : Type u_2
      inst✝ : Semiring α
      a : α
      ha : Odd a
      ⊢ Odd 1
    -/
    exact odd_one
    /-
      🎉 no goals
    -/
                /-
                  α : Type u_2
                  inst✝ : Semiring α
                  a : α
                  ha : Odd a
                  n : Nat
                  ⊢ Odd (HPow.hPow a (HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [pow_succ]; exact ha.pow.mul ha
                               /-
                                 🎉 no goals
                               -/


lemma Odd.pow_add_pow_eq_zero [IsCancelAdd α] (hn : Odd n) (hab : a + b = 0) :
    a ^ n + b ^ n = 0 := by
  /-
    α : Type u_2
    inst✝¹ : Semiring α
    a b : α
    n : Nat
    inst✝ : IsCancelAdd α
    hn : Odd n
    hab : Eq (HAdd.hAdd a b) 0
    ⊢ Eq (HAdd.hAdd (HPow.hPow a n) (HPow.hPow b n)) 0
  -/
  obtain ⟨k, rfl⟩ := hn
  /-
    case intro
    α : Type u_2
    inst✝¹ : Semiring α
    a b : α
    inst✝ : IsCancelAdd α
    hab : Eq (HAdd.hAdd a b) 0
    k : Nat
    ⊢ Eq (HAdd.hAdd (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 k) 1)) (HPow.hPow b (HAdd …
  -/
  induction' k with k ih
    /-
      case intro.zero
      α : Type u_2
      inst✝¹ : Semiring α
      a b : α
      inst✝ : IsCancelAdd α
      hab : Eq (HAdd.hAdd a b) 0
      ⊢ Eq (HAdd.hAdd (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 0) 1)) (HPow.hPow b (HAdd …
    -/
  · simpa
    /-
      🎉 no goals
    -/
  have : a ^ 2 = b ^ 2 := add_right_cancel <|
    calc
      a ^ 2 + a * b = 0 := by rw [sq, ← mul_add, hab, mul_zero]
      _ = b ^ 2 + a * b := by rw [sq, ← add_mul, add_comm, hab, zero_mul]
  /-
    case intro.succ
    α : Type u_2
    inst✝¹ : Semiring α
    a b : α
    inst✝ : IsCancelAdd α
    hab : Eq (HAdd.hAdd a b) 0
    k : Nat
    ih : Eq (HAdd.hAdd (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 k) 1)) (HPow.hPow b (H …
    this : Eq (HPow.hPow a 2) (HPow.hPow b 2)
    ⊢ Eq (HAdd.hAdd (HPow.hPow a (HAdd.hAdd (HMul.hMul 2 (HAdd.hAdd k 1)) 1)) (HPo …
  -/
  refine add_right_cancel (b := b ^ (2 * k + 1) * a ^ 2) ?_
  calc
    _ = (a ^ (2 * k + 1) + b ^ (2 * k + 1)) * a ^ 2 + b ^ (2 * k + 3) := by
      rw [add_mul, ← pow_add, add_right_comm]; rfl
    _ = _ := by rw [ih, zero_mul, zero_add, zero_add, this, ← pow_add]


lemma Odd.neg_pow : Odd n → ∀ a : α, (-a) ^ n = -a ^ n := by
  /-
    α : Type u_2
    inst✝¹ : Monoid α
    inst✝ : HasDistribNeg α
    n : Nat
    ⊢ Odd n → ∀ (a : α), Eq (HPow.hPow (Neg.neg a) n) (Neg.neg (HPow.hPow a n))
  -/
  rintro ⟨c, rfl⟩ a; simp_rw [pow_add, pow_mul, neg_sq, pow_one, mul_neg]
                     /-
                       🎉 no goals
                     -/


                                                                    /-
                                                                      α : Type u_2
                                                                      inst✝¹ : Monoid α
                                                                      inst✝ : HasDistribNeg α
                                                                      n : Nat
                                                                      h : Odd n
                                                                      ⊢ Eq (HPow.hPow (-1) n) (-1)
                                                                    -/
@[simp] lemma Odd.neg_one_pow (h : Odd n) : (-1 : α) ^ n = -1 := by rw [h.neg_pow, one_pow]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                         /-
                                           α : Type u_2
                                           inst✝ : Ring α
                                           ⊢ Even (-2)
                                         -/
lemma even_neg_two : Even (-2 : α) := by simp only [even_neg, even_two]
                                         /-
                                           🎉 no goals
                                         -/


lemma Odd.neg (hp : Odd a) : Odd (-a) := by
  /-
    α : Type u_2
    inst✝ : Ring α
    a : α
    hp : Odd a
    ⊢ Odd (Neg.neg a)
  -/
  obtain ⟨k, hk⟩ := hp
  /-
    case intro
    α : Type u_2
    inst✝ : Ring α
    a k : α
    hk : Eq a (HAdd.hAdd (HMul.hMul 2 k) 1)
    ⊢ Odd (Neg.neg a)
  -/
  use -(k + 1)
  rw [mul_neg, mul_add, neg_add, add_assoc, two_mul (1 : α), neg_add, neg_add_cancel_right,
    ← neg_add, hk]


@[simp] lemma odd_neg : Odd (-a) ↔ Odd a := ⟨fun h ↦ neg_neg a ▸ h.neg, Odd.neg⟩


                                       /-
                                         α : Type u_2
                                         inst✝ : Ring α
                                         ⊢ Odd (-1)
                                       -/
lemma odd_neg_one : Odd (-1 : α) := by simp
                                       /-
                                         🎉 no goals
                                       -/


lemma Odd.sub_even (ha : Odd a) (hb : Even b) : Odd (a - b) := by
  /-
    α : Type u_2
    inst✝ : Ring α
    a b : α
    ha : Odd a
    hb : Even b
    ⊢ Odd (HSub.hSub a b)
  -/
  rw [sub_eq_add_neg]; exact ha.add_even hb.neg
                       /-
                         🎉 no goals
                       -/


lemma Even.sub_odd (ha : Even a) (hb : Odd b) : Odd (a - b) := by
  /-
    α : Type u_2
    inst✝ : Ring α
    a b : α
    ha : Even a
    hb : Odd b
    ⊢ Odd (HSub.hSub a b)
  -/
  rw [sub_eq_add_neg]; exact ha.add_odd hb.neg
                       /-
                         🎉 no goals
                       -/


lemma Odd.sub_odd (ha : Odd a) (hb : Odd b) : Even (a - b) := by
  /-
    α : Type u_2
    inst✝ : Ring α
    a b : α
    ha : Odd a
    hb : Odd b
    ⊢ Even (HSub.hSub a b)
  -/
  rw [sub_eq_add_neg]; exact ha.add_odd hb.neg
                       /-
                         🎉 no goals
                       -/


lemma odd_iff : Odd n ↔ n % 2 = 1 :=
                    /-
                      n : Nat
                      x✝ : Odd n
                      m : Nat
                      hm : Eq n (HAdd.hAdd (HMul.hMul 2 m) 1)
                      ⊢ Eq (HMod.hMod n 2) 1
                    -/
                    /-
                      🎉 no goals
                    -/
  ⟨fun ⟨m, hm⟩ ↦ by omega, fun h ↦ ⟨n / 2, (mod_add_div n 2).symm.trans (by rw [h, add_comm])⟩⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


instance : DecidablePred (Odd : ℕ → Prop) := fun _ ↦ decidable_of_iff _ odd_iff.symm


                                             /-
                                               n : Nat
                                               ⊢ Iff (Not (Odd n)) (Eq (HMod.hMod n 2) 0)
                                             -/
lemma not_odd_iff : ¬Odd n ↔ n % 2 = 0 := by rw [odd_iff, mod_two_not_eq_one]
                                             /-
                                               🎉 no goals
                                             -/


                                                       /-
                                                         n : Nat
                                                         ⊢ Iff (Not (Odd n)) (Even n)
                                                       -/
@[simp] lemma not_odd_iff_even : ¬Odd n ↔ Even n := by rw [not_odd_iff, even_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                       /-
                                                         n : Nat
                                                         ⊢ Iff (Not (Even n)) (Odd n)
                                                       -/
@[simp] lemma not_even_iff_odd : ¬Even n ↔ Odd n := by rw [not_even_iff, odd_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp] lemma not_odd_zero : ¬Odd 0 := not_odd_iff.mpr rfl


@[deprecated not_odd_iff_even (since := "2024-08-21")]
                                               /-
                                                 n : Nat
                                                 ⊢ Iff (Even n) (Not (Odd n))
                                               -/
lemma even_iff_not_odd : Even n ↔ ¬Odd n := by rw [not_odd_iff, even_iff]
                                               /-
                                                 🎉 no goals
                                               -/


@[deprecated not_even_iff_odd (since := "2024-08-21")]
                                               /-
                                                 n : Nat
                                                 ⊢ Iff (Odd n) (Not (Even n))
                                               -/
lemma odd_iff_not_even : Odd n ↔ ¬Even n := by rw [not_even_iff, odd_iff]
                                               /-
                                                 🎉 no goals
                                               -/


lemma _root_.Odd.not_two_dvd_nat (h : Odd n) : ¬(2 ∣ n) := by
  /-
    n : Nat
    h : Odd n
    ⊢ Not (Dvd.dvd 2 n)
  -/
  rwa [← even_iff_two_dvd, not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma even_xor_odd (n : ℕ) : Xor' (Even n) (Odd n) := by
  /-
    n : Nat
    ⊢ Xor' (Even n) (Odd n)
  -/
  simp [Xor', ← not_even_iff_odd, Decidable.em (Even n)]
  /-
    🎉 no goals
  -/


lemma even_or_odd (n : ℕ) : Even n ∨ Odd n := (even_xor_odd n).or


lemma even_or_odd' (n : ℕ) : ∃ k, n = 2 * k ∨ n = 2 * k + 1 := by
  /-
    n : Nat
    ⊢ Exists fun k => Or (Eq n (HMul.hMul 2 k)) (Eq n (HAdd.hAdd (HMul.hMul 2 k) 1))
  -/
  simpa only [← two_mul, exists_or, Odd, Even] using even_or_odd n
  /-
    🎉 no goals
  -/


lemma even_xor_odd' (n : ℕ) : ∃ k, Xor' (n = 2 * k) (n = 2 * k + 1) := by
  /-
    n : Nat
    ⊢ Exists fun k => Xor' (Eq n (HMul.hMul 2 k)) (Eq n (HAdd.hAdd (HMul.hMul 2 k) …
  -/
  obtain ⟨k, rfl⟩ | ⟨k, rfl⟩ := even_or_odd n <;> use k
    /-
      case h
      k : Nat
      ⊢ Xor' (Eq (HAdd.hAdd k k) (HMul.hMul 2 k)) (Eq (HAdd.hAdd k k) (HAdd.hAdd (HM …
    -/
  · simpa only [← two_mul, eq_self_iff_true, xor_true] using (succ_ne_self (2 * k)).symm
    /-
      🎉 no goals
    -/
    /-
      case h
      k : Nat
      ⊢ Xor' (Eq (HAdd.hAdd (HMul.hMul 2 k) 1) (HMul.hMul 2 k)) (Eq (HAdd.hAdd (HMul …
    -/
  · simpa only [xor_true, xor_comm] using (succ_ne_self _)
    /-
      🎉 no goals
    -/


lemma odd_add_one {n : ℕ} : Odd (n + 1) ↔ ¬ Odd n := by
  /-
    n : Nat
    ⊢ Iff (Odd (HAdd.hAdd n 1)) (Not (Odd n))
  -/
  rw [← not_even_iff_odd, Nat.even_add_one, not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma mod_two_add_add_odd_mod_two (m : ℕ) {n : ℕ} (hn : Odd n) : m % 2 + (m + n) % 2 = 1 :=
                                    /-
                                      m n : Nat
                                      hn : Odd n
                                      hm : Even m
                                      ⊢ Eq (HAdd.hAdd (HMod.hMod m 2) (HMod.hMod (HAdd.hAdd m n) 2)) 1
                                    -/
  ((even_or_odd m).elim fun hm ↦ by rw [even_iff.1 hm, odd_iff.1 (hm.add_odd hn)]) fun hm ↦ by
                                    /-
                                      🎉 no goals
                                    -/
    /-
      m n : Nat
      hn : Odd n
      hm : Odd m
      ⊢ Eq (HAdd.hAdd (HMod.hMod m 2) (HMod.hMod (HAdd.hAdd m n) 2)) 1
    -/
    rw [odd_iff.1 hm, even_iff.1 (hm.add_odd hn)]
    /-
      🎉 no goals
    -/


@[simp] lemma mod_two_add_succ_mod_two (m : ℕ) : m % 2 + (m + 1) % 2 = 1 :=
  mod_two_add_add_odd_mod_two m odd_one


@[simp] lemma succ_mod_two_add_mod_two (m : ℕ) : (m + 1) % 2 + m % 2 = 1 := by
  /-
    m : Nat
    ⊢ Eq (HAdd.hAdd (HMod.hMod (HAdd.hAdd m 1) 2) (HMod.hMod m 2)) 1
  -/
  rw [add_comm, mod_two_add_succ_mod_two]
  /-
    🎉 no goals
  -/


lemma even_add' : Even (m + n) ↔ (Odd m ↔ Odd n) := by
  /-
    m n : Nat
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Odd m) (Odd n))
  -/
  rw [even_add, ← not_odd_iff_even, ← not_odd_iff_even, not_iff_not]
  /-
    🎉 no goals
  -/


                                                              /-
                                                                n : Nat
                                                                ⊢ Not (Even (HAdd.hAdd (HMul.hMul 2 n) 1))
                                                              -/
@[simp] lemma not_even_bit1 (n : ℕ) : ¬Even (2 * n + 1) := by simp [parity_simps]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma not_even_two_mul_add_one (n : ℕ) : ¬ Even (2 * n + 1) :=
  not_even_iff_odd.2 <| odd_two_mul_add_one n


lemma even_sub' (h : n ≤ m) : Even (m - n) ↔ (Odd m ↔ Odd n) := by
  /-
    m n : Nat
    h : LE.le n m
    ⊢ Iff (Even (HSub.hSub m n)) (Iff (Odd m) (Odd n))
  -/
  rw [even_sub h, ← not_odd_iff_even, ← not_odd_iff_even, not_iff_not]
  /-
    🎉 no goals
  -/


lemma Odd.sub_odd (hm : Odd m) (hn : Odd n) : Even (m - n) :=
                                  /-
                                    m n : Nat
                                    hm : Odd m
                                    hn : Odd n
                                    h : LE.le n m
                                    ⊢ Even (HSub.hSub m n)
                                  -/
  (le_total n m).elim (fun h ↦ by simp only [even_sub' h, *]) fun h ↦ by
                                  /-
                                    🎉 no goals
                                  -/
    /-
      m n : Nat
      hm : Odd m
      hn : Odd n
      h : LE.le m n
      ⊢ Even (HSub.hSub m n)
    -/
    simp only [Nat.sub_eq_zero_iff_le.2 h, even_zero]
    /-
      🎉 no goals
    -/


alias _root_.Odd.tsub_odd := Nat.Odd.sub_odd


                                                  /-
                                                    m n : Nat
                                                    ⊢ Iff (Odd (HMul.hMul m n)) (And (Odd m) (Odd n))
                                                  -/
lemma odd_mul : Odd (m * n) ↔ Odd m ∧ Odd n := by simp [not_or, even_mul, ← not_even_iff_odd]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma Odd.of_mul_left (h : Odd (m * n)) : Odd m :=
  (odd_mul.mp h).1


lemma Odd.of_mul_right (h : Odd (m * n)) : Odd n :=
  (odd_mul.mp h).2


lemma even_div : Even (m / n) ↔ m % (2 * n) / n = 0 := by
  /-
    m n : Nat
    ⊢ Iff (Even (HDiv.hDiv m n)) (Eq (HDiv.hDiv (HMod.hMod m (HMul.hMul 2 n)) n) 0)
  -/
  rw [even_iff_two_dvd, dvd_iff_mod_eq_zero, ← Nat.mod_mul_right_div_self, mul_comm]
  /-
    🎉 no goals
  -/


@[parity_simps] lemma odd_add : Odd (m + n) ↔ (Odd m ↔ Even n) := by
  /-
    m n : Nat
    ⊢ Iff (Odd (HAdd.hAdd m n)) (Iff (Odd m) (Even n))
  -/
  rw [← not_even_iff_odd, even_add, not_iff, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


                                                      /-
                                                        m n : Nat
                                                        ⊢ Iff (Odd (HAdd.hAdd m n)) (Iff (Odd n) (Even m))
                                                      -/
lemma odd_add' : Odd (m + n) ↔ (Odd n ↔ Even m) := by rw [add_comm, odd_add]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                    /-
                                                      m n : Nat
                                                      h : Odd (HAdd.hAdd m n)
                                                      ⊢ Ne m n
                                                    -/
lemma ne_of_odd_add (h : Odd (m + n)) : m ≠ n := by rintro rfl; simp [← not_even_iff_odd] at h
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[parity_simps] lemma odd_sub (h : n ≤ m) : Odd (m - n) ↔ (Odd m ↔ Even n) := by
  /-
    m n : Nat
    h : LE.le n m
    ⊢ Iff (Odd (HSub.hSub m n)) (Iff (Odd m) (Even n))
  -/
  rw [← not_even_iff_odd, even_sub h, not_iff, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma Odd.sub_even (h : n ≤ m) (hm : Odd m) (hn : Even n) : Odd (m - n) :=
  (odd_sub h).mpr <| iff_of_true hm hn


lemma odd_sub' (h : n ≤ m) : Odd (m - n) ↔ (Odd n ↔ Even m) := by
  /-
    m n : Nat
    h : LE.le n m
    ⊢ Iff (Odd (HSub.hSub m n)) (Iff (Odd n) (Even m))
  -/
  rw [← not_even_iff_odd, even_sub h, not_iff, not_iff_comm, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma Even.sub_odd (h : n ≤ m) (hm : Even m) (hn : Odd n) : Odd (m - n) :=
  (odd_sub' h).mpr <| iff_of_true hn hm


lemma two_mul_div_two_add_one_of_odd (h : Odd n) : 2 * (n / 2) + 1 = n := by
  /-
    n : Nat
    h : Odd n
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv n 2)) 1) n
  -/
  rw [← odd_iff.mp h, div_add_mod]
  /-
    🎉 no goals
  -/


lemma div_two_mul_two_add_one_of_odd (h : Odd n) : n / 2 * 2 + 1 = n := by
  /-
    n : Nat
    h : Odd n
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv n 2) 2) 1) n
  -/
  rw [← odd_iff.mp h, div_add_mod']
  /-
    🎉 no goals
  -/


lemma one_add_div_two_mul_two_of_odd (h : Odd n) : 1 + n / 2 * 2 = n := by
  /-
    n : Nat
    h : Odd n
    ⊢ Eq (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv n 2) 2)) n
  -/
  rw [← odd_iff.mp h, mod_add_div']
  /-
    🎉 no goals
  -/


lemma iterate_bit0 (hf : Involutive f) (n : ℕ) : f^[2 * n] = id := by
  /-
    α : Type u_4
    f : α → α
    hf : Function.Involutive f
    n : Nat
    ⊢ Eq (Nat.iterate f (HMul.hMul 2 n)) id
  -/
  rw [iterate_mul, involutive_iff_iter_2_eq_id.1 hf, iterate_id]
  /-
    🎉 no goals
  -/


lemma iterate_bit1 (hf : Involutive f) (n : ℕ) : f^[2 * n + 1] = f := by
  /-
    α : Type u_4
    f : α → α
    hf : Function.Involutive f
    n : Nat
    ⊢ Eq (Nat.iterate f (HAdd.hAdd (HMul.hMul 2 n) 1)) f
  -/
  rw [← succ_eq_add_one, iterate_succ, hf.iterate_bit0, id_comp]
  /-
    🎉 no goals
  -/


lemma iterate_two_mul (hf : Involutive f) (n : ℕ) : f^[2 * n] = id := by
  /-
    α : Type u_4
    f : α → α
    hf : Function.Involutive f
    n : Nat
    ⊢ Eq (Nat.iterate f (HMul.hMul 2 n)) id
  -/
  rw [iterate_mul, involutive_iff_iter_2_eq_id.1 hf, iterate_id]
  /-
    🎉 no goals
  -/


lemma iterate_even (hf : Involutive f) (hn : Even n) : f^[n] = id := by
  /-
    α : Type u_4
    f : α → α
    n : Nat
    hf : Function.Involutive f
    hn : Even n
    ⊢ Eq (Nat.iterate f n) id
  -/
  obtain ⟨m, rfl⟩ := hn
  /-
    case intro
    α : Type u_4
    f : α → α
    hf : Function.Involutive f
    m : Nat
    ⊢ Eq (Nat.iterate f (HAdd.hAdd m m)) id
  -/
  rw [← two_mul, hf.iterate_two_mul]
  /-
    🎉 no goals
  -/


lemma iterate_odd (hf : Involutive f) (hn : Odd n) : f^[n] = f := by
  /-
    α : Type u_4
    f : α → α
    n : Nat
    hf : Function.Involutive f
    hn : Odd n
    ⊢ Eq (Nat.iterate f n) f
  -/
  obtain ⟨m, rfl⟩ := hn
  /-
    case intro
    α : Type u_4
    f : α → α
    hf : Function.Involutive f
    m : Nat
    ⊢ Eq (Nat.iterate f (HAdd.hAdd (HMul.hMul 2 m) 1)) f
  -/
  rw [iterate_add, hf.iterate_two_mul, id_comp, iterate_one]
  /-
    🎉 no goals
  -/


lemma iterate_eq_self (hf : Involutive f) (hne : f ≠ id) : f^[n] = f ↔ Odd n :=
                                                 /-
                                                   α : Type u_4
                                                   f : α → α
                                                   n : Nat
                                                   hf : Function.Involutive f
                                                   hne : Ne f id
                                                   H : Eq (Nat.iterate f n) f
                                                   hn : Even n
                                                   ⊢ Eq f id
                                                 -/
  ⟨fun H ↦ not_even_iff_odd.1 fun hn ↦ hne <| by rwa [hf.iterate_even hn, eq_comm] at H,
                                                 /-
                                                   🎉 no goals
                                                 -/
    hf.iterate_odd⟩


lemma iterate_eq_id (hf : Involutive f) (hne : f ≠ id) : f^[n] = id ↔ Even n :=
                                                 /-
                                                   α : Type u_4
                                                   f : α → α
                                                   n : Nat
                                                   hf : Function.Involutive f
                                                   hne : Ne f id
                                                   H : Eq (Nat.iterate f n) id
                                                   hn : Odd n
                                                   ⊢ Eq f id
                                                 -/
  ⟨fun H ↦ not_odd_iff_even.1 fun hn ↦ hne <| by rwa [hf.iterate_odd hn] at H, hf.iterate_even⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma neg_one_pow_eq_ite {R : Type*} [Monoid R] [HasDistribNeg R] {n : ℕ} :
    (-1 : R) ^ n = ite (Even n) 1 (-1) := by
  cases even_or_odd n with
  | inl h => rw [h.neg_one_pow, if_pos h]
  | inr h => rw [h.neg_one_pow, if_neg (by simpa using h)]


lemma neg_one_pow_eq_one_iff_even {R : Type*} [Monoid R] [HasDistribNeg R] {n : ℕ}
                                                         /-
                                                           R : Type u_4
                                                           inst✝¹ : Monoid R
                                                           inst✝ : HasDistribNeg R
                                                           n : Nat
                                                           h : Ne (-1) 1
                                                           ⊢ Iff (Eq (HPow.hPow (-1) n) 1) (Even n)
                                                         -/
    (h : (-1 : R) ≠ 1) : (-1 : R) ^ n = 1 ↔ Even n := by simp [neg_one_pow_eq_ite, h]
                                                         /-
                                                           🎉 no goals
                                                         -/


private theorem natCast_eq_zero_or_one_of_two_eq_zero' (n : ℕ) (h : (2 : R) = 0) :
    (Even n → (n : R) = 0) ∧ (Odd n → (n : R) = 1) := by
  induction n using Nat.twoStepInduction with
  | zero => simp
  | one => simp
  | more n _ _ => simpa [add_assoc, Nat.even_add_one, Nat.odd_add_one, h]


theorem natCast_eq_zero_of_even_of_two_eq_zero {n : ℕ} (hn : Even n) (h : (2 : R) = 0) :
    (n : R) = 0 :=
  (natCast_eq_zero_or_one_of_two_eq_zero' n h).1 hn


theorem natCast_eq_one_of_odd_of_two_eq_zero {n : ℕ} (hn : Odd n) (h : (2 : R) = 0) :
    (n : R) = 1 :=
  (natCast_eq_zero_or_one_of_two_eq_zero' n h).2 hn


theorem natCast_eq_zero_or_one_of_two_eq_zero (n : ℕ) (h : (2 : R) = 0) :
    (n : R) = 0 ∨ (n : R) = 1 := by
  /-
    R : Type u_4
    inst✝ : AddMonoidWithOne R
    n : Nat
    h : Eq 2 0
    ⊢ Or (Eq (↑n) 0) (Eq (↑n) 1)
  -/
  obtain hn | hn := Nat.even_or_odd n
    /-
      case inl
      R : Type u_4
      inst✝ : AddMonoidWithOne R
      n : Nat
      h : Eq 2 0
      hn : Even n
      ⊢ Or (Eq (↑n) 0) (Eq (↑n) 1)
    -/
  · exact Or.inl <| natCast_eq_zero_of_even_of_two_eq_zero hn h
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_4
      inst✝ : AddMonoidWithOne R
      n : Nat
      h : Eq 2 0
      hn : Odd n
      ⊢ Or (Eq (↑n) 0) (Eq (↑n) 1)
    -/
  · exact Or.inr <| natCast_eq_one_of_odd_of_two_eq_zero hn h
    /-
      🎉 no goals
    -/


