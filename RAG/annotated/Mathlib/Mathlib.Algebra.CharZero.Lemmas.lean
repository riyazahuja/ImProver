/-- `Nat.cast` as an embedding into monoids of characteristic `0`. -/
@[simps]
def castEmbedding : ℕ ↪ R :=
  ⟨Nat.cast, cast_injective⟩


@[simp]
theorem cast_pow_eq_one {R : Type*} [Semiring R] [CharZero R] (q : ℕ) (n : ℕ) (hn : n ≠ 0) :
    (q : R) ^ n = 1 ↔ q = 1 := by
  /-
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CharZero R
    q n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq (HPow.hPow (↑q) n) 1) (Eq q 1)
  -/
  rw [← cast_pow, cast_eq_one]
  /-
    R : Type u_2
    inst✝¹ : Semiring R
    inst✝ : CharZero R
    q n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq (HPow.hPow q n) 1) (Eq q 1)
  -/
  exact pow_eq_one_iff hn
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_div_charZero {k : Type*} [DivisionSemiring k] [CharZero k] {m n : ℕ} (n_dvd : n ∣ m) :
    ((m / n : ℕ) : k) = m / n := by
  /-
    k : Type u_2
    inst✝¹ : DivisionSemiring k
    inst✝ : CharZero k
    m n : Nat
    n_dvd : Dvd.dvd n m
    ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      k : Type u_2
      inst✝¹ : DivisionSemiring k
      inst✝ : CharZero k
      m : Nat
      n_dvd : Dvd.dvd 0 m
      ⊢ Eq (↑(HDiv.hDiv m 0)) (HDiv.hDiv ↑m ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_2
      inst✝¹ : DivisionSemiring k
      inst✝ : CharZero k
      m n : Nat
      n_dvd : Dvd.dvd n m
      hn : Ne n 0
      ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
    -/
  · exact cast_div n_dvd (cast_ne_zero.2 hn)
    /-
      🎉 no goals
    -/


instance CharZero.NeZero.two : NeZero (2 : M) :=
  ⟨by
    /-
      α : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoidWithOne M
      inst✝ : CharZero M
      n : Nat
      ⊢ Ne 2 0
    -/
    have : ((2 : ℕ) : M) ≠ 0 := Nat.cast_ne_zero.2 (by decide)
    /-
      α : Type u_1
      M : Type u_2
      inst✝¹ : AddMonoidWithOne M
      inst✝ : CharZero M
      n : Nat
      this : Ne (↑2) 0
      ⊢ Ne 2 0
    -/
    rwa [Nat.cast_two] at this⟩
    /-
      🎉 no goals
    -/


lemma support_natCast (hn : n ≠ 0) : support (n : α → M) = univ :=
  support_const <| Nat.cast_ne_zero.2 hn


@[deprecated (since := "2024-04-17")]
alias support_nat_cast := support_natCast


lemma mulSupport_natCast (hn : n ≠ 1) : mulSupport (n : α → M) = univ :=
  mulSupport_const <| Nat.cast_ne_one.2 hn


@[deprecated (since := "2024-04-17")]
alias mulSupport_nat_cast := mulSupport_natCast


@[simp]
theorem add_self_eq_zero {a : R} : a + a = 0 ↔ a = 0 := by
  /-
    R : Type u_1
    inst✝² : NonAssocSemiring R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    a : R
    ⊢ Iff (Eq (HAdd.hAdd a a) 0) (Eq a 0)
  -/
  simp only [(two_mul a).symm, mul_eq_zero, two_ne_zero, false_or]
  /-
    🎉 no goals
  -/


@[scoped simp] theorem CharZero.neg_eq_self_iff {a : R} : -a = a ↔ a = 0 :=
  neg_eq_iff_add_eq_zero.trans add_self_eq_zero


@[scoped simp] theorem CharZero.eq_neg_self_iff {a : R} : a = -a ↔ a = 0 :=
  eq_neg_iff_add_eq_zero.trans add_self_eq_zero


theorem nat_mul_inj {n : ℕ} {a b : R} (h : (n : R) * a = (n : R) * b) : n = 0 ∨ a = b := by
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    n : Nat
    a b : R
    h : Eq (HMul.hMul (↑n) a) (HMul.hMul (↑n) b)
    ⊢ Or (Eq n 0) (Eq a b)
  -/
  rw [← sub_eq_zero, ← mul_sub, mul_eq_zero, sub_eq_zero] at h
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    n : Nat
    a b : R
    h : Or (Eq (↑n) 0) (Eq a b)
    ⊢ Or (Eq n 0) (Eq a b)
  -/
  exact mod_cast h
  /-
    🎉 no goals
  -/


theorem nat_mul_inj' {n : ℕ} {a b : R} (h : (n : R) * a = (n : R) * b) (w : n ≠ 0) : a = b := by
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : CharZero R
    n : Nat
    a b : R
    h : Eq (HMul.hMul (↑n) a) (HMul.hMul (↑n) b)
    w : Ne n 0
    ⊢ Eq a b
  -/
  simpa [w] using nat_mul_inj h
  /-
    🎉 no goals
  -/


@[simp] lemma add_self_div_two (a : R) : (a + a) / 2 = a := by
  /-
    R : Type u_1
    inst✝¹ : DivisionSemiring R
    inst✝ : NeZero 2
    a : R
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd a a) 2) a
  -/
  rw [← mul_two, mul_div_cancel_right₀ a two_ne_zero]
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-07-16")] alias half_add_self := add_self_div_two



@[simp]
                                                     /-
                                                       R : Type u_1
                                                       inst✝¹ : DivisionSemiring R
                                                       inst✝ : NeZero 2
                                                       a : R
                                                       ⊢ Eq (HAdd.hAdd (HDiv.hDiv a 2) (HDiv.hDiv a 2)) a
                                                     -/
theorem add_halves (a : R) : a / 2 + a / 2 = a := by rw [← add_div, add_self_div_two]
                                                     /-
                                                       🎉 no goals
                                                     -/

@[deprecated (since := "2024-07-16")] alias add_halves' := add_halves


                                                   /-
                                                     R : Type u_1
                                                     inst✝¹ : DivisionRing R
                                                     inst✝ : CharZero R
                                                     a : R
                                                     ⊢ Eq (HSub.hSub a (HDiv.hDiv a 2)) (HDiv.hDiv a 2)
                                                   -/
theorem sub_half (a : R) : a - a / 2 = a / 2 := by rw [sub_eq_iff_eq_add, add_halves]
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                      /-
                                                        R : Type u_1
                                                        inst✝¹ : DivisionRing R
                                                        inst✝ : CharZero R
                                                        a : R
                                                        ⊢ Eq (HSub.hSub (HDiv.hDiv a 2) a) (Neg.neg (HDiv.hDiv a 2))
                                                      -/
theorem half_sub (a : R) : a / 2 - a = -(a / 2) := by rw [← neg_sub, sub_half]
                                                      /-
                                                        🎉 no goals
                                                      -/


instance {R : Type*} [AddMonoidWithOne R] [CharZero R] :
    CharZero (WithTop R) where
  cast_injective m n h := by
    /-
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      inst✝ : CharZero R
      m n : Nat
      h : Eq ↑m ↑n
      ⊢ Eq m n
    -/
    rwa [← coe_natCast, ← coe_natCast n, coe_eq_coe, Nat.cast_inj] at h
    /-
      🎉 no goals
    -/


instance {R : Type*} [AddMonoidWithOne R] [CharZero R] :
    CharZero (WithBot R) where
  cast_injective m n h := by
    /-
      R : Type u_1
      inst✝¹ : AddMonoidWithOne R
      inst✝ : CharZero R
      m n : Nat
      h : Eq ↑m ↑n
      ⊢ Eq m n
    -/
    rwa [← coe_natCast, ← coe_natCast n, coe_eq_coe, Nat.cast_inj] at h
    /-
      🎉 no goals
    -/


theorem RingHom.charZero (ϕ : R →+* S) [CharZero S] : CharZero R :=
                                                     /-
                                                       R : Type u_1
                                                       S : Type u_2
                                                       inst✝² : NonAssocSemiring R
                                                       inst✝¹ : NonAssocSemiring S
                                                       ϕ : RingHom R S
                                                       inst✝ : CharZero S
                                                       a b : Nat
                                                       h : Eq ↑a ↑b
                                                       ⊢ Eq ↑a ↑b
                                                     -/
  ⟨fun a b h => CharZero.cast_injective (R := S) (by rw [← map_natCast ϕ, ← map_natCast ϕ, h])⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem RingHom.charZero_iff {ϕ : R →+* S} (hϕ : Function.Injective ϕ) : CharZero R ↔ CharZero S :=
  ⟨fun hR =>
        /-
          R : Type u_1
          S : Type u_2
          inst✝¹ : NonAssocSemiring R
          inst✝ : NonAssocSemiring S
          ϕ : RingHom R S
          hϕ : Function.Injective ⇑ϕ
          hR : CharZero R
          ⊢ Function.Injective Nat.cast
        -/
    ⟨by intro a b h; rwa [← @Nat.cast_inj R, ← hϕ.eq_iff, map_natCast ϕ, map_natCast ϕ]⟩,
                     /-
                       🎉 no goals
                     -/
    fun _ => ϕ.charZero⟩


theorem RingHom.injective_nat (f : ℕ →+* R) [CharZero R] : Function.Injective f :=
  Subsingleton.elim (Nat.castRingHom _) f ▸ Nat.cast_injective


@[simp]
theorem units_ne_neg_self (u : Rˣ) : u ≠ -u := by
  simp_rw [ne_eq, Units.ext_iff, Units.val_neg, eq_neg_iff_add_eq_zero, ← two_mul,
    Units.mul_left_eq_zero, two_ne_zero, not_false_iff]


@[simp]
theorem neg_units_ne_self (u : Rˣ) : -u ≠ u := (units_ne_neg_self u).symm


