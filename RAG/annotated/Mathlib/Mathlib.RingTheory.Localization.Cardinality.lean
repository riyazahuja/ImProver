theorem lift_cardinalMk_le (S : Submonoid R) [IsLocalization S L] :
    Cardinal.lift.{u} #L ≤ Cardinal.lift.{v} #R := by
  /-
    R : Type u
    inst✝³ : CommSemiring R
    L : Type v
    inst✝² : CommSemiring L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk L)) (Cardinal.lift.{v, u} (Cardinal …
  -/
  have := Localization.cardinalMk_le S
  /-
    R : Type u
    inst✝³ : CommSemiring R
    L : Type v
    inst✝² : CommSemiring L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    this : LE.le (Cardinal.mk (Localization S)) (Cardinal.mk R)
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk L)) (Cardinal.lift.{v, u} (Cardinal …
  -/
  rwa [← lift_le.{v}, lift_mk_eq'.2 ⟨(Localization.algEquiv S L).toEquiv⟩] at this
  /-
    🎉 no goals
  -/


/-- A localization always has cardinality less than or equal to the base ring. -/
theorem cardinalMk_le {L : Type u} [CommSemiring L] [Algebra R L]
    (S : Submonoid R) [IsLocalization S L] : #L ≤ #R := by
  /-
    R : Type u
    inst✝³ : CommSemiring R
    L : Type u
    inst✝² : CommSemiring L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    ⊢ LE.le (Cardinal.mk L) (Cardinal.mk R)
  -/
  simpa using lift_cardinalMk_le (L := L) S
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-30")] alias card_le := cardinalMk_le


theorem cardinalMk {S : Submonoid R} (hS : S ≤ R⁰) : #(Localization S) = #R := by
  /-
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    ⊢ Eq (Cardinal.mk (Localization S)) (Cardinal.mk R)
  -/
  apply OreLocalization.cardinalMk
  /-
    case hS
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    ⊢ LE.le S (nonZeroDivisorsRight R)
  -/
  convert hS using 1
  /-
    case h.e'_4
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    ⊢ Eq (nonZeroDivisorsRight R) (nonZeroDivisors R)
  -/
  ext x
  /-
    case h.e'_4.h
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    x : R
    ⊢ Iff (Membership.mem (nonZeroDivisorsRight R) x) (Membership.mem (nonZeroDivi …
  -/
  rw [mem_nonZeroDivisorsRight_iff, mem_nonZeroDivisors_iff]
  /-
    case h.e'_4.h
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    x : R
    ⊢ Iff (∀ (y : R), Eq (HMul.hMul x y) 0 → Eq y 0) (∀ (x_1 : R), Eq (HMul.hMul x …
  -/
  congr! 3
  /-
    case h.e'_4.h.a.h.h.h.e'_2
    R : Type u
    inst✝ : CommRing R
    S : Submonoid R
    hS : LE.le S (nonZeroDivisors R)
    x a✝ : R
    ⊢ Eq (HMul.hMul x a✝) (HMul.hMul a✝ x)
  -/
  rw [mul_comm]
  /-
    🎉 no goals
  -/


theorem lift_cardinalMk (S : Submonoid R) [IsLocalization S L] (hS : S ≤ R⁰) :
    Cardinal.lift.{u} #L = Cardinal.lift.{v} #R := by
  /-
    R : Type u
    inst✝³ : CommRing R
    L : Type v
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    hS : LE.le S (nonZeroDivisors R)
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk L)) (Cardinal.lift.{v, u} (Cardinal.mk …
  -/
  have := Localization.cardinalMk hS
  /-
    R : Type u
    inst✝³ : CommRing R
    L : Type v
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    hS : LE.le S (nonZeroDivisors R)
    this : Eq (Cardinal.mk (Localization S)) (Cardinal.mk R)
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk L)) (Cardinal.lift.{v, u} (Cardinal.mk …
  -/
  rwa [← lift_inj.{u, v}, lift_mk_eq'.2 ⟨(Localization.algEquiv S L).toEquiv⟩] at this
  /-
    🎉 no goals
  -/


/-- If you do not localize at any zero-divisors, localization preserves cardinality. -/
theorem cardinalMk (L : Type u) [CommRing L] [Algebra R L]
    (S : Submonoid R) [IsLocalization S L] (hS : S ≤ R⁰) : #L = #R := by
  /-
    R : Type u
    inst✝³ : CommRing R
    L : Type u
    inst✝² : CommRing L
    inst✝¹ : Algebra R L
    S : Submonoid R
    inst✝ : IsLocalization S L
    hS : LE.le S (nonZeroDivisors R)
    ⊢ Eq (Cardinal.mk L) (Cardinal.mk R)
  -/
  simpa using lift_cardinalMk L S hS
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-30")] alias card := cardinalMk


@[simp]
theorem Cardinal.mk_fractionRing (R : Type u) [CommRing R] : #(FractionRing R) = #R :=
  IsLocalization.cardinalMk (FractionRing R) R⁰ le_rfl


alias FractionRing.cardinalMk := Cardinal.mk_fractionRing


theorem lift_cardinalMk [IsFractionRing R L] : Cardinal.lift.{u} #L = Cardinal.lift.{v} #R :=
  IsLocalization.lift_cardinalMk L _ le_rfl


theorem cardinalMk (L : Type u) [CommRing L] [Algebra R L] [IsFractionRing R L] : #L = #R :=
  IsLocalization.cardinalMk L _ le_rfl


