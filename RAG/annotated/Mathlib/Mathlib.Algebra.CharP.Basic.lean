lemma natCast_injOn_Iio : (Set.Iio p).InjOn ((↑) : ℕ → R) :=
  fun _a ha _b hb hab ↦ ((natCast_eq_natCast _ _).1 hab).eq_of_lt_of_lt ha hb


lemma intCast_injOn_Ico [IsRightCancelAdd R] : InjOn (Int.cast : ℤ → R) (Ico 0 p) := by
  /-
    R : Type u_1
    inst✝² : AddGroupWithOne R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : IsRightCancelAdd R
    ⊢ Set.InjOn Int.cast (Set.Ico 0 ↑p)
  -/
  rintro a ⟨ha₀, ha⟩ b ⟨hb₀, hb⟩ hab
  /-
    case intro.intro
    R : Type u_1
    inst✝² : AddGroupWithOne R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : IsRightCancelAdd R
    a : Int
    ha₀ : LE.le 0 a
    ha : LT.lt a ↑p
    b : Int
    hb₀ : LE.le 0 b
    hb : LT.lt b ↑p
    hab : Eq ↑a ↑b
    ⊢ Eq a b
  -/
  lift a to ℕ using ha₀
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : AddGroupWithOne R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : IsRightCancelAdd R
    b : Int
    hb₀ : LE.le 0 b
    hb : LT.lt b ↑p
    a : Nat
    ha : LT.lt ↑a ↑p
    hab : Eq ↑↑a ↑b
    ⊢ Eq (↑a) b
  -/
  lift b to ℕ using hb₀
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : AddGroupWithOne R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : IsRightCancelAdd R
    a : Nat
    ha : LT.lt ↑a ↑p
    b : Nat
    hb : LT.lt ↑b ↑p
    hab : Eq ↑↑a ↑↑b
    ⊢ Eq ↑a ↑b
  -/
  norm_cast at *
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : AddGroupWithOne R
    p : Nat
    inst✝¹ : CharP R p
    inst✝ : IsRightCancelAdd R
    a b : Nat
    ha : LT.lt a p
    hb : LT.lt b p
    hab : Eq ↑a ↑b
    ⊢ Eq a b
  -/
  exact natCast_injOn_Iio _ _ ha hb hab
  /-
    🎉 no goals
  -/


lemma RingHom.charP_iff_charP {K L : Type*} [DivisionRing K] [Semiring L] [Nontrivial L]
    (f : K →+* L) (p : ℕ) : CharP K p ↔ CharP L p := by
  /-
    K : Type u_2
    L : Type u_3
    inst✝² : DivisionRing K
    inst✝¹ : Semiring L
    inst✝ : Nontrivial L
    f : RingHom K L
    p : Nat
    ⊢ Iff (CharP K p) (CharP L p)
  -/
  simp only [charP_iff, ← f.injective.eq_iff, map_natCast f, map_zero f]
  /-
    🎉 no goals
  -/


variable (R) in
/-- If a ring `R` is of characteristic `p`, then for any prime number `q` different from `p`,
it is not zero in `R`. -/
lemma cast_ne_zero_of_ne_of_prime [Nontrivial R]
    {p q : ℕ} [CharP R p] (hq : q.Prime) (hneq : p ≠ q) : (q : R) ≠ 0 := fun h ↦ by
  /-
    R : Type u_1
    inst✝² : NonAssocSemiring R
    inst✝¹ : Nontrivial R
    p q : Nat
    inst✝ : CharP R p
    hq : Nat.Prime q
    hneq : Ne p q
    h : Eq (↑q) 0
    ⊢ False
  -/
  rw [cast_eq_zero_iff R p q] at h
  /-
    R : Type u_1
    inst✝² : NonAssocSemiring R
    inst✝¹ : Nontrivial R
    p q : Nat
    inst✝ : CharP R p
    hq : Nat.Prime q
    hneq : Ne p q
    h : Dvd.dvd p q
    ⊢ False
  -/
  rcases hq.eq_one_or_self_of_dvd _ h with h | h
    /-
      case inl
      R : Type u_1
      inst✝² : NonAssocSemiring R
      inst✝¹ : Nontrivial R
      p q : Nat
      inst✝ : CharP R p
      hq : Nat.Prime q
      hneq : Ne p q
      h✝ : Dvd.dvd p q
      h : Eq p 1
      ⊢ False
    -/
  · subst h
    /-
      case inl
      R : Type u_1
      inst✝² : NonAssocSemiring R
      inst✝¹ : Nontrivial R
      q : Nat
      hq : Nat.Prime q
      inst✝ : CharP R 1
      hneq : Ne 1 q
      h : Dvd.dvd 1 q
      ⊢ False
    -/
    exact false_of_nontrivial_of_char_one (R := R)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝² : NonAssocSemiring R
      inst✝¹ : Nontrivial R
      p q : Nat
      inst✝ : CharP R p
      hq : Nat.Prime q
      hneq : Ne p q
      h✝ : Dvd.dvd p q
      h : Eq p q
      ⊢ False
    -/
  · exact hneq h
    /-
      🎉 no goals
    -/


lemma ringChar_of_prime_eq_zero [Nontrivial R] {p : ℕ} (hprime : Nat.Prime p)
    (hp0 : (p : R) = 0) : ringChar R = p :=
  Or.resolve_left ((Nat.dvd_prime hprime).1 (ringChar.dvd hp0)) ringChar_ne_one


lemma charP_iff_prime_eq_zero [Nontrivial R] {p : ℕ} (hp : p.Prime) :
    CharP R p ↔ (p : R) = 0 :=
  ⟨fun _ => cast_eq_zero R p,
   fun hp0 => (ringChar_of_prime_eq_zero hp hp0) ▸ inferInstance⟩


/-- We have `2 ≠ 0` in a nontrivial ring whose characteristic is not `2`. -/
protected lemma Ring.two_ne_zero {R : Type*} [NonAssocSemiring R] [Nontrivial R]
    (hR : ringChar R ≠ 2) : (2 : R) ≠ 0 := by
  /-
    R : Type u_2
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    hR : Ne (ringChar R) 2
    ⊢ Ne 2 0
  -/
  rw [Ne, (by norm_cast : (2 : R) = (2 : ℕ)), ringChar.spec, Nat.dvd_prime Nat.prime_two]
  /-
    R : Type u_2
    inst✝¹ : NonAssocSemiring R
    inst✝ : Nontrivial R
    hR : Ne (ringChar R) 2
    ⊢ Not (Or (Eq (ringChar R) 1) (Eq (ringChar R) 2))
  -/
  exact mt (or_iff_left hR).mp CharP.ringChar_ne_one
  /-
    🎉 no goals
  -/

-- We have `CharP.neg_one_ne_one`, which assumes `[Ring R] (p : ℕ) [CharP R p] [Fact (2 < p)]`.
-- This is a version using `ringChar` instead.

/-- Characteristic `≠ 2` and nontrivial implies that `-1 ≠ 1`. -/
lemma Ring.neg_one_ne_one_of_char_ne_two {R : Type*} [NonAssocRing R] [Nontrivial R]
    (hR : ringChar R ≠ 2) : (-1 : R) ≠ 1 := fun h =>
  Ring.two_ne_zero hR (one_add_one_eq_two (R := R) ▸ neg_eq_iff_add_eq_zero.mp h)


/-- Characteristic `≠ 2` in a domain implies that `-a = a` iff `a = 0`. -/
lemma Ring.eq_self_iff_eq_zero_of_char_ne_two {R : Type*} [NonAssocRing R] [Nontrivial R]
    [NoZeroDivisors R] (hR : ringChar R ≠ 2) {a : R} : -a = a ↔ a = 0 :=
  ⟨fun h =>
    (mul_eq_zero.mp <| (two_mul a).trans <| neg_eq_iff_add_eq_zero.mp h).resolve_left
      (Ring.two_ne_zero hR),
    fun h => ((congr_arg (fun x => -x) h).trans neg_zero).trans h.symm⟩


/-- The characteristic of the product of rings is the least common multiple of the
characteristics of the two rings. -/
instance Nat.lcm.charP [CharP S q] : CharP (R × S) (Nat.lcm p q) where
  cast_eq_zero_iff' := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : AddMonoidWithOne R
      inst✝² : AddMonoidWithOne S
      p q : Nat
      inst✝¹ : CharP R p
      inst✝ : CharP S q
      ⊢ ∀ (x : Nat), Iff (Eq (↑x) 0) (Dvd.dvd (p.lcm q) x)
    -/
    simp [Prod.ext_iff, CharP.cast_eq_zero_iff R p, CharP.cast_eq_zero_iff S q, Nat.lcm_dvd_iff]
    /-
      🎉 no goals
    -/


/-- The characteristic of the product of two rings of the same characteristic
  is the same as the characteristic of the rings -/
instance Prod.charP [CharP S p] : CharP (R × S) p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : AddMonoidWithOne R
    inst✝² : AddMonoidWithOne S
    p q : Nat
    inst✝¹ : CharP R p
    inst✝ : CharP S p
    ⊢ CharP (Prod R S) p
  -/
  convert Nat.lcm.charP R S p p; simp
                                 /-
                                   🎉 no goals
                                 -/


instance Prod.charZero_of_left [CharZero R] : CharZero (R × S) where
  cast_injective _ _ h := CharZero.cast_injective congr(Prod.fst $h)


instance Prod.charZero_of_right [CharZero S] : CharZero (R × S) where
  cast_injective _ _ h := CharZero.cast_injective congr(Prod.snd $h)


instance ULift.charP [AddMonoidWithOne R] (p : ℕ) [CharP R p] : CharP (ULift R) p where
  cast_eq_zero_iff' n := Iff.trans ULift.ext_iff <| CharP.cast_eq_zero_iff R p n


instance MulOpposite.charP [AddMonoidWithOne R] (p : ℕ) [CharP R p] : CharP Rᵐᵒᵖ p where
  cast_eq_zero_iff' n := MulOpposite.unop_inj.symm.trans <| CharP.cast_eq_zero_iff R p n


/-- If two integers from `{0, 1, -1}` result in equal elements in a ring `R`
that is nontrivial and of characteristic not `2`, then they are equal. -/
lemma Int.cast_injOn_of_ringChar_ne_two {R : Type*} [NonAssocRing R] [Nontrivial R]
    (hR : ringChar R ≠ 2) : ({0, 1, -1} : Set ℤ).InjOn ((↑) : ℤ → R) := by
  /-
    R : Type u_2
    inst✝¹ : NonAssocRing R
    inst✝ : Nontrivial R
    hR : Ne (ringChar R) 2
    ⊢ Set.InjOn Int.cast (Insert.insert 0 (Insert.insert 1 (Singleton.singleton (- …
  -/
  rintro _ (rfl | rfl | rfl) _ (rfl | rfl | rfl) h <;>
  simp only
    [cast_neg, cast_one, cast_zero, neg_eq_zero, one_ne_zero, zero_ne_one, zero_eq_neg] at h ⊢
    /-
      case inr.inl.inr.inr
      R : Type u_2
      inst✝¹ : NonAssocRing R
      inst✝ : Nontrivial R
      hR : Ne (ringChar R) 2
      h : Eq 1 (-1)
      ⊢ Eq 1 (-1)
    -/
  · exact ((Ring.neg_one_ne_one_of_char_ne_two hR).symm h).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr.inl
      R : Type u_2
      inst✝¹ : NonAssocRing R
      inst✝ : Nontrivial R
      hR : Ne (ringChar R) 2
      h : Eq (-1) 1
      ⊢ Eq (-1) 1
    -/
  · exact ((Ring.neg_one_ne_one_of_char_ne_two hR) h).elim
    /-
      🎉 no goals
    -/


lemma charZero_iff_forall_prime_ne_zero [NonAssocRing R] [NoZeroDivisors R] [Nontrivial R] :
    CharZero R ↔ ∀ p : ℕ, p.Prime → (p : R) ≠ 0 := by
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    ⊢ Iff (CharZero R) (∀ (p : Nat), Nat.Prime p → Ne (↑p) 0)
  -/
  refine ⟨fun h p hp => by simp [hp.ne_zero], fun h => ?_⟩
  /-
    R : Type u_1
    inst✝² : NonAssocRing R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    h : ∀ (p : Nat), Nat.Prime p → Ne (↑p) 0
    ⊢ CharZero R
  -/
  let p := ringChar R
  cases CharP.char_is_prime_or_zero R p with
  | inl hp => simpa using h p hp
  | inr h => have : CharP R 0 := h ▸ inferInstance; exact CharP.charP_to_charZero R


/-- The characteristic of `F_p` is `p`. -/
@[stacks 09FS "First part. We don't require `p` to be a prime in mathlib."]
instance charP (n : ℕ) [NeZero n] : CharP (Fin n) n where cast_eq_zero_iff' _ := natCast_eq_zero


instance (S : Type*) [Semiring S] (p) [ExpChar R p] [ExpChar S p] : ExpChar (R × S) p := by
  /-
    R : Type u_1
    inst✝³ : AddMonoidWithOne R
    S : Type u_2
    inst✝² : Semiring S
    p : Nat
    inst✝¹ : ExpChar R p
    inst✝ : ExpChar S p
    ⊢ ExpChar (Prod R S) p
  -/
  obtain hp | ⟨hp⟩ := ‹ExpChar R p›
    /-
      case zero
      R : Type u_1
      inst✝⁴ : AddMonoidWithOne R
      S : Type u_2
      inst✝³ : Semiring S
      inst✝² : CharZero R
      inst✝¹ : ExpChar R 1
      inst✝ : ExpChar S 1
      ⊢ ExpChar (Prod R S) 1
    -/
  · have := Prod.charZero_of_left R S; exact .zero
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case prime
    R : Type u_1
    inst✝³ : AddMonoidWithOne R
    S : Type u_2
    inst✝² : Semiring S
    p : Nat
    inst✝¹ : ExpChar R p
    inst✝ : ExpChar S p
    hp : Nat.Prime p
    hchar✝ : CharP R p
    ⊢ ExpChar (Prod R S) p
  -/
  obtain _ | _ := ‹ExpChar S p›
    /-
      case prime.zero
      R : Type u_1
      inst✝⁴ : AddMonoidWithOne R
      S : Type u_2
      inst✝³ : Semiring S
      inst✝² : CharZero S
      inst✝¹ : ExpChar R 1
      inst✝ : ExpChar S 1
      hp : Nat.Prime 1
      hchar✝ : CharP R 1
      ⊢ ExpChar (Prod R S) 1
    -/
  · exact (Nat.not_prime_one hp).elim
    /-
      🎉 no goals
    -/
    /-
      case prime.prime
      R : Type u_1
      inst✝³ : AddMonoidWithOne R
      S : Type u_2
      inst✝² : Semiring S
      p : Nat
      inst✝¹ : ExpChar R p
      inst✝ : ExpChar S p
      hp : Nat.Prime p
      hchar✝¹ : CharP R p
      hprime✝ : Nat.Prime p
      hchar✝ : CharP S p
      ⊢ ExpChar (Prod R S) p
    -/
  · have := Prod.charP R S p; exact .prime hp
                              /-
                                🎉 no goals
                              -/


