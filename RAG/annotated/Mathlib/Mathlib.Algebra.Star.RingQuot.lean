theorem Rel.star (hr : ∀ a b, r a b → r (star a) (star b))
    ⦃a b : R⦄ (h : Rel r a b) : Rel r (star a) (star b) := by
  induction h with
  | of h          => exact Rel.of (hr _ _ h)
  | add_left _ h  => rw [star_add, star_add]
                     exact Rel.add_left h
  | mul_left _ h  => rw [star_mul, star_mul]
                     exact Rel.mul_right h
  | mul_right _ h => rw [star_mul, star_mul]
                     exact Rel.mul_left h


private irreducible_def star' (hr : ∀ a b, r a b → r (star a) (star b)) : RingQuot r → RingQuot r
  | ⟨a⟩ => ⟨Quot.map (star : R → R) (Rel.star r hr) a⟩


theorem star'_quot (hr : ∀ a b, r a b → r (star a) (star b)) {a} :
    (star' r hr ⟨Quot.mk _ a⟩ : RingQuot r) = ⟨Quot.mk _ (star a)⟩ := star'_def _ _ _


/-- Transfer a star_ring instance through a quotient, if the quotient is invariant to `star` -/
def starRing {R : Type u} [Semiring R] [StarRing R] (r : R → R → Prop)
    (hr : ∀ a b, r a b → r (star a) (star b)) : StarRing (RingQuot r) where
  star := star' r hr
  star_involutive := by
    /-
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      ⊢ Function.Involutive Star.star
    -/
    rintro ⟨⟨⟩⟩
    /-
      case mk.mk
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      toQuot✝ : Quot (RingQuot.Rel r)
      a✝ : R
      ⊢ Eq (Star.star (Star.star { toQuot := Quot.mk (RingQuot.Rel r) a✝ })) { toQuo …
    -/
    simp [star'_quot]
    /-
      🎉 no goals
    -/
  star_mul := by
    /-
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      ⊢ ∀ (r_1 s : RingQuot r), Eq (Star.star (HMul.hMul r_1 s)) (HMul.hMul (Star.st …
    -/
    rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩
    /-
      case mk.mk.mk.mk
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      toQuot✝¹ : Quot (RingQuot.Rel r)
      a✝¹ : R
      toQuot✝ : Quot (RingQuot.Rel r)
      a✝ : R
      ⊢ Eq (Star.star (HMul.hMul { toQuot := Quot.mk (RingQuot.Rel r) a✝¹ } { toQuot …
    -/
    simp [star'_quot, mul_quot, star_mul]
    /-
      🎉 no goals
    -/
  star_add := by
    /-
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      ⊢ ∀ (r_1 s : RingQuot r), Eq (Star.star (HAdd.hAdd r_1 s)) (HAdd.hAdd (Star.st …
    -/
    rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩
    /-
      case mk.mk.mk.mk
      R✝ : Type u
      inst✝³ : Semiring R✝
      r✝ : R✝ → R✝ → Prop
      inst✝² : StarRing R✝
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : StarRing R
      r : R → R → Prop
      hr : ∀ (a b : R), r a b → r (Star.star a) (Star.star b)
      toQuot✝¹ : Quot (RingQuot.Rel r)
      a✝¹ : R
      toQuot✝ : Quot (RingQuot.Rel r)
      a✝ : R
      ⊢ Eq (Star.star (HAdd.hAdd { toQuot := Quot.mk (RingQuot.Rel r) a✝¹ } { toQuot …
    -/
    simp [star'_quot, add_quot, star_add]
    /-
      🎉 no goals
    -/


