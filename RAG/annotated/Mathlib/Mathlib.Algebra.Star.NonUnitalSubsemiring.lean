/-- A sub star semigroup is a subset of a magma which is closed under the `star`-/
structure SubStarSemigroup (M : Type v) [Mul M] [Star M] extends Subsemigroup M : Type v where
  /-- The `carrier` of a `StarSubset` is closed under the `star` operation. -/
  star_mem' : ∀ {a : M} (_ha : a ∈ carrier), star a ∈ carrier


/-- A non-unital star subsemiring is a non-unital subsemiring which also is closed under the
`star` operation. -/
structure NonUnitalStarSubsemiring (R : Type v) [NonUnitalNonAssocSemiring R] [Star R]
    extends NonUnitalSubsemiring R : Type v where
  /-- The `carrier` of a `NonUnitalStarSubsemiring` is closed under the `star` operation. -/
  star_mem' : ∀ {a : R} (_ha : a ∈ carrier), star a ∈ carrier


instance instSetLike : SetLike (NonUnitalStarSubsemiring R) R where
  coe {s} := s.carrier
                             /-
                               A : Type v
                               B : Type w
                               C : Type w'
                               R : Type v
                               inst✝¹ : NonUnitalNonAssocSemiring R
                               inst✝ : StarRing R
                               p q : NonUnitalStarSubsemiring R
                               h : Eq (fun {s} => s.carrier) fun {s} => s.carrier
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance instNonUnitalSubsemiringClass : NonUnitalSubsemiringClass (NonUnitalStarSubsemiring R) R
    where
  add_mem {s} := s.add_mem'
  mul_mem {s} := s.mul_mem'
  zero_mem {s} := s.zero_mem'


instance instStarMemClass : StarMemClass (NonUnitalStarSubsemiring R) R where
  star_mem {s} := s.star_mem'


theorem mem_carrier {s : NonUnitalStarSubsemiring R} {x : R} : x ∈ s.carrier ↔ x ∈ s :=
  Iff.rfl


/-- Copy of a non-unital star subsemiring with a new `carrier` equal to the old one.
Useful to fix definitional equalities. -/
protected def copy (S : NonUnitalStarSubsemiring R) (s : Set R) (hs : s = ↑S) :
    NonUnitalStarSubsemiring R :=
  { S.toNonUnitalSubsemiring.copy s hs with
    star_mem' := fun {x} (hx : x ∈ s) => by
      /-
        A : Type v
        B : Type w
        C : Type w'
        R : Type v
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : StarRing R
        S : NonUnitalStarSubsemiring R
        s : Set R
        hs : Eq s ↑S
        x : R
        hx : Membership.mem s x
        ⊢ Membership.mem __src✝.carrier (Star.star x)
      -/
      show star x ∈ s
      /-
        A : Type v
        B : Type w
        C : Type w'
        R : Type v
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : StarRing R
        S : NonUnitalStarSubsemiring R
        s : Set R
        hs : Eq s ↑S
        x : R
        hx : Membership.mem s x
        ⊢ Membership.mem s (Star.star x)
      -/
      rw [hs] at hx ⊢
      /-
        A : Type v
        B : Type w
        C : Type w'
        R : Type v
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : StarRing R
        S : NonUnitalStarSubsemiring R
        s : Set R
        hs : Eq s ↑S
        x : R
        hx : Membership.mem (↑S) x
        ⊢ Membership.mem (↑S) (Star.star x)
      -/
      exact S.star_mem' hx }
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_copy (S : NonUnitalStarSubsemiring R) (s : Set R) (hs : s = ↑S) :
    (S.copy s hs : Set R) = s :=
  rfl


theorem copy_eq (S : NonUnitalStarSubsemiring R) (s : Set R) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


/-- The center of a non-unital non-associative semiring `R` is the set of elements that
commute and associate with everything in `R`, here realized as non-unital star
subsemiring. -/
def center (R) [NonUnitalNonAssocSemiring R] [StarRing R] : NonUnitalStarSubsemiring R where
  toNonUnitalSubsemiring := NonUnitalSubsemiring.center R
  star_mem' := Set.star_mem_center


