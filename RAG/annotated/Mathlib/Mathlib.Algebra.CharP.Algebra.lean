/-- If a ring homomorphism `R →+* A` is injective then `A` has the same characteristic as `R`. -/
theorem charP_of_injective_ringHom {R A : Type*} [NonAssocSemiring R] [NonAssocSemiring A]
    {f : R →+* A} (h : Function.Injective f) (p : ℕ) [CharP R p] : CharP A p where
  cast_eq_zero_iff' x := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝² : NonAssocSemiring R
      inst✝¹ : NonAssocSemiring A
      f : RingHom R A
      h : Function.Injective ⇑f
      p : Nat
      inst✝ : CharP R p
      x : Nat
      ⊢ Iff (Eq (↑x) 0) (Dvd.dvd p x)
    -/
    rw [← CharP.cast_eq_zero_iff R p x, ← map_natCast f x, map_eq_zero_iff f h]
    /-
      🎉 no goals
    -/


/-- If the algebra map `R →+* A` is injective then `A` has the same characteristic as `R`. -/
theorem charP_of_injective_algebraMap {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A]
    (h : Function.Injective (algebraMap R A)) (p : ℕ) [CharP R p] : CharP A p :=
  charP_of_injective_ringHom h p


theorem charP_of_injective_algebraMap' (R A : Type*) [Field R] [Semiring A] [Algebra R A]
    [Nontrivial A] (p : ℕ) [CharP R p] : CharP A p :=
  charP_of_injective_algebraMap (algebraMap R A).injective p


/-- If a ring homomorphism `R →+* A` is injective and `R` has characteristic zero
then so does `A`. -/
theorem charZero_of_injective_ringHom {R A : Type*} [NonAssocSemiring R] [NonAssocSemiring A]
    {f : R →+* A} (h : Function.Injective f) [CharZero R] : CharZero A where
                                                             /-
                                                               R : Type u_1
                                                               A : Type u_2
                                                               inst✝² : NonAssocSemiring R
                                                               inst✝¹ : NonAssocSemiring A
                                                               f : RingHom R A
                                                               h : Function.Injective ⇑f
                                                               inst✝ : CharZero R
                                                               x✝² x✝¹ : Nat
                                                               x✝ : Eq ↑x✝² ↑x✝¹
                                                               ⊢ Eq (f ↑x✝²) (f ↑x✝¹)
                                                             -/
  cast_injective _ _ _ := CharZero.cast_injective <| h <| by simpa only [map_natCast f]
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- If the algebra map `R →+* A` is injective and `R` has characteristic zero then so does `A`. -/
theorem charZero_of_injective_algebraMap {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A]
    (h : Function.Injective (algebraMap R A)) [CharZero R] : CharZero A :=
  charZero_of_injective_ringHom h


/-- If `R →+* A` is injective, and `A` is of characteristic `p`, then `R` is also of
characteristic `p`. Similar to `RingHom.charZero`. -/
theorem RingHom.charP {R A : Type*} [NonAssocSemiring R] [NonAssocSemiring A] (f : R →+* A)
    (H : Function.Injective f) (p : ℕ) [CharP A p] : CharP R p := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : NonAssocSemiring R
    inst✝¹ : NonAssocSemiring A
    f : RingHom R A
    H : Function.Injective ⇑f
    p : Nat
    inst✝ : CharP A p
    ⊢ CharP R p
  -/
  obtain ⟨q, h⟩ := CharP.exists R
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝² : NonAssocSemiring R
    inst✝¹ : NonAssocSemiring A
    f : RingHom R A
    H : Function.Injective ⇑f
    p : Nat
    inst✝ : CharP A p
    q : Nat
    h : CharP R q
    ⊢ CharP R p
  -/
  exact CharP.eq _ (charP_of_injective_ringHom H q) ‹CharP A p› ▸ h
  /-
    🎉 no goals
  -/


/-- If `R →+* A` is injective, then `R` is of characteristic `p` if and only if `A` is also of
characteristic `p`. Similar to `RingHom.charZero_iff`. -/
theorem RingHom.charP_iff {R A : Type*} [NonAssocSemiring R] [NonAssocSemiring A] (f : R →+* A)
    (H : Function.Injective f) (p : ℕ) : CharP R p ↔ CharP A p :=
  ⟨fun _ ↦ charP_of_injective_ringHom H p, fun _ ↦ f.charP H p⟩


/-- If a ring homomorphism `R →+* A` is injective then `A` has the same exponential characteristic
as `R`. -/
lemma expChar_of_injective_ringHom {R A : Type*}
    [Semiring R] [Semiring A] {f : R →+* A} (h : Function.Injective f)
    (q : ℕ) [hR : ExpChar R q] : ExpChar A q := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring A
    f : RingHom R A
    h : Function.Injective ⇑f
    q : Nat
    hR : ExpChar R q
    ⊢ ExpChar A q
  -/
  rcases hR with _ | hprime
    /-
      case zero
      R : Type u_1
      A : Type u_2
      inst✝² : Semiring R
      inst✝¹ : Semiring A
      f : RingHom R A
      h : Function.Injective ⇑f
      inst✝ : CharZero R
      ⊢ ExpChar A 1
    -/
  · haveI := charZero_of_injective_ringHom h; exact .zero
                                              /-
                                                🎉 no goals
                                              -/
  /-
    case prime
    R : Type u_1
    A : Type u_2
    inst✝¹ : Semiring R
    inst✝ : Semiring A
    f : RingHom R A
    h : Function.Injective ⇑f
    q : Nat
    hprime : Nat.Prime q
    hchar✝ : CharP R q
    ⊢ ExpChar A q
  -/
  haveI := charP_of_injective_ringHom h q; exact .prime hprime
                                           /-
                                             🎉 no goals
                                           -/


/-- If `R →+* A` is injective, and `A` is of exponential characteristic `p`, then `R` is also of
exponential characteristic `p`. Similar to `RingHom.charZero`. -/
lemma RingHom.expChar {R A : Type*} [Semiring R] [Semiring A] (f : R →+* A)
    (H : Function.Injective f) (p : ℕ) [ExpChar A p] : ExpChar R p := by
  cases ‹ExpChar A p› with
  | zero => haveI := f.charZero; exact .zero
  | prime hp => haveI := f.charP H p; exact .prime hp


/-- If `R →+* A` is injective, then `R` is of exponential characteristic `p` if and only if `A` is
also of exponential characteristic `p`. Similar to `RingHom.charZero_iff`. -/
lemma RingHom.expChar_iff {R A : Type*} [Semiring R] [Semiring A] (f : R →+* A)
    (H : Function.Injective f) (p : ℕ) : ExpChar R p ↔ ExpChar A p :=
  ⟨fun _ ↦ expChar_of_injective_ringHom H p, fun _ ↦ f.expChar H p⟩


/-- If the algebra map `R →+* A` is injective then `A` has the same exponential characteristic
as `R`. -/
lemma expChar_of_injective_algebraMap {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A]
    (h : Function.Injective (algebraMap R A)) (q : ℕ) [ExpChar R q] : ExpChar A q :=
  expChar_of_injective_ringHom h q


/-- A nontrivial `ℚ`-algebra has `CharP` equal to zero.

This cannot be a (local) instance because it would immediately form a loop with the
instance `DivisionRing.toRatAlgebra`. It's probably easier to go the other way: prove `CharZero R`
and automatically receive an `Algebra ℚ R` instance.
-/
theorem algebraRat.charP_zero [Semiring R] [Algebra ℚ R] : CharP R 0 :=
  charP_of_injective_algebraMap (algebraMap ℚ R).injective 0


/-- A nontrivial `ℚ`-algebra has characteristic zero.

This cannot be a (local) instance because it would immediately form a loop with the
instance `DivisionRing.toRatAlgebra`. It's probably easier to go the other way: prove `CharZero R`
and automatically receive an `Algebra ℚ R` instance.
-/
theorem algebraRat.charZero [Ring R] [Algebra ℚ R] : CharZero R :=
  @CharP.charP_to_charZero R _ (algebraRat.charP_zero R)


theorem Algebra.charP_iff (p : ℕ) : CharP K p ↔ CharP L p :=
  (algebraMap K L).charP_iff_charP p


theorem Algebra.ringChar_eq : ringChar K = ringChar L := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : CommSemiring L
    inst✝¹ : Nontrivial L
    inst✝ : Algebra K L
    ⊢ Eq (ringChar K) (ringChar L)
  -/
  rw [ringChar.eq_iff, Algebra.charP_iff K L]
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : CommSemiring L
    inst✝¹ : Nontrivial L
    inst✝ : Algebra K L
    ⊢ CharP L (ringChar L)
  -/
  apply ringChar.charP
  /-
    🎉 no goals
  -/


/-- If `R` has characteristic `p`, then so does `FreeAlgebra R X`. -/
instance charP [CharP R p] : CharP (FreeAlgebra R X) p :=
  charP_of_injective_algebraMap FreeAlgebra.algebraMap_leftInverse.injective p


/-- If `R` has characteristic `0`, then so does `FreeAlgebra R X`. -/
instance charZero [CharZero R] : CharZero (FreeAlgebra R X) :=
  charZero_of_injective_algebraMap FreeAlgebra.algebraMap_leftInverse.injective


/-- If `R` has characteristic `p`, then so does Frac(R). -/
theorem charP_of_isFractionRing [CharP R p] : CharP K p :=
  charP_of_injective_algebraMap (IsFractionRing.injective R K) p


/-- If `R` has characteristic `0`, then so does Frac(R). -/
theorem charZero_of_isFractionRing [CharZero R] : CharZero K :=
  @CharP.charP_to_charZero K _ (charP_of_isFractionRing R 0)


/-- If `R` has characteristic `p`, then so does `FractionRing R`. -/
instance charP [CharP R p] : CharP (FractionRing R) p :=
  charP_of_isFractionRing R p


/-- If `R` has characteristic `0`, then so does `FractionRing R`. -/
instance charZero [CharZero R] : CharZero (FractionRing R) :=
  charZero_of_isFractionRing R


