theorem infinite_of_charZero (R A : Type*) [CommRing R] [IsDomain R] [Ring A] [Algebra R A]
    [CharZero A] : { x : A | IsAlgebraic R x }.Infinite :=
  infinite_of_injective_forall_mem Nat.cast_injective isAlgebraic_nat


theorem aleph0_le_cardinalMk_of_charZero (R A : Type*) [CommRing R] [IsDomain R] [Ring A]
    [Algebra R A] [CharZero A] : ℵ₀ ≤ #{ x : A // IsAlgebraic R x } :=
  infinite_iff.1 (Set.infinite_coe_iff.2 <| infinite_of_charZero R A)


@[deprecated (since := "2024-11-10")]
alias aleph0_le_cardinal_mk_of_charZero := aleph0_le_cardinalMk_of_charZero


theorem cardinalMk_lift_le_mul :
    Cardinal.lift.{u} #{ x : A // IsAlgebraic R x } ≤ Cardinal.lift.{v} #R[X] * ℵ₀ := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (Subtype fun x => IsAlgebraic R x)) …
  -/
  rw [← mk_uLift, ← mk_uLift]
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.mk (ULift.{u, v} (Subtype fun x => IsAlgebraic R x))) (HMul. …
  -/
  choose g hg₁ hg₂ using fun x : { x : A | IsAlgebraic R x } => x.coe_prop
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    g : ↑(setOf fun x => IsAlgebraic R x) → Polynomial R
    hg₁ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Ne (g x) 0
    hg₂ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Eq ((Polynomial.aeval ↑x) (g  …
    ⊢ LE.le (Cardinal.mk (ULift.{u, v} (Subtype fun x => IsAlgebraic R x))) (HMul. …
  -/
  refine lift_mk_le_lift_mk_mul_of_lift_mk_preimage_le g fun f => ?_
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    g : ↑(setOf fun x => IsAlgebraic R x) → Polynomial R
    hg₁ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Ne (g x) 0
    hg₂ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Eq ((Polynomial.aeval ↑x) (g  …
    f : Polynomial R
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Set.preimage g (Singleton.singlet …
  -/
  rw [lift_le_aleph0, le_aleph0_iff_set_countable]
  suffices MapsTo (↑) (g ⁻¹' {f}) (f.rootSet A) from
    this.countable_of_injOn Subtype.coe_injective.injOn (f.rootSet_finite A).countable
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    g : ↑(setOf fun x => IsAlgebraic R x) → Polynomial R
    hg₁ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Ne (g x) 0
    hg₂ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Eq ((Polynomial.aeval ↑x) (g  …
    f : Polynomial R
    ⊢ Set.MapsTo Subtype.val (Set.preimage g (Singleton.singleton f)) (f.rootSet A)
  -/
  rintro x (rfl : g x = f)
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    g : ↑(setOf fun x => IsAlgebraic R x) → Polynomial R
    hg₁ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Ne (g x) 0
    hg₂ : ∀ (x : ↑(setOf fun x => IsAlgebraic R x)), Eq ((Polynomial.aeval ↑x) (g  …
    x : ↑(setOf fun x => IsAlgebraic R x)
    ⊢ Membership.mem ((g x).rootSet A) ↑x
  -/
  exact mem_rootSet.2 ⟨hg₁ x, hg₂ x⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_lift_le_mul := cardinalMk_lift_le_mul


theorem cardinalMk_lift_le_max :
    Cardinal.lift.{u} #{ x : A // IsAlgebraic R x } ≤ max (Cardinal.lift.{v} #R) ℵ₀ :=
  (cardinalMk_lift_le_mul R A).trans <|
                                                                    /-
                                                                      R : Type u
                                                                      A : Type v
                                                                      inst✝⁴ : CommRing R
                                                                      inst✝³ : CommRing A
                                                                      inst✝² : IsDomain A
                                                                      inst✝¹ : Algebra R A
                                                                      inst✝ : NoZeroSMulDivisors R A
                                                                      ⊢ LE.le (HMul.hMul (Cardinal.lift.{v, u} (Max.max (Cardinal.mk R) Cardinal.ale …
                                                                    -/
    (mul_le_mul_right' (lift_le.2 cardinalMk_le_max) _).trans <| by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_lift_le_max := cardinalMk_lift_le_max


@[simp]
theorem cardinalMk_lift_of_infinite [Infinite R] :
    Cardinal.lift.{u} #{ x : A // IsAlgebraic R x } = Cardinal.lift.{v} #R :=
  ((cardinalMk_lift_le_max R A).trans_eq (max_eq_left <| aleph0_le_mk _)).antisymm <|
    lift_mk_le'.2 ⟨⟨fun x => ⟨algebraMap R A x, isAlgebraic_algebraMap _⟩, fun _ _ h =>
      NoZeroSMulDivisors.algebraMap_injective R A (Subtype.ext_iff.1 h)⟩⟩


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_lift_of_infinite := cardinalMk_lift_of_infinite


@[simp]
protected theorem countable : Set.Countable { x : A | IsAlgebraic R x } := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Algebra R A
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : Countable R
    ⊢ (setOf fun x => IsAlgebraic R x).Countable
  -/
  rw [← le_aleph0_iff_set_countable, ← lift_le_aleph0]
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Algebra R A
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : Countable R
    ⊢ LE.le (Cardinal.lift.{?u.19682, v} (Cardinal.mk ↑(setOf fun x => IsAlgebraic …
  -/
  apply (cardinalMk_lift_le_max R A).trans
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing A
    inst✝³ : IsDomain A
    inst✝² : Algebra R A
    inst✝¹ : NoZeroSMulDivisors R A
    inst✝ : Countable R
    ⊢ LE.le (Max.max (Cardinal.lift.{v, u} (Cardinal.mk R)) Cardinal.aleph0) Cardi …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem cardinalMk_of_countable_of_charZero [CharZero A] [IsDomain R] :
    #{ x : A // IsAlgebraic R x } = ℵ₀ :=
  (Algebraic.countable R A).le_aleph0.antisymm (aleph0_le_cardinalMk_of_charZero R A)


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_of_countable_of_charZero := cardinalMk_of_countable_of_charZero


theorem cardinalMk_le_mul : #{ x : A // IsAlgebraic R x } ≤ #R[X] * ℵ₀ := by
  /-
    R A : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.mk (Subtype fun x => IsAlgebraic R x)) (HMul.hMul (Cardinal. …
  -/
  rw [← lift_id #_, ← lift_id #R[X]]
  /-
    R A : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.lift.{u, u} (Cardinal.mk (Subtype fun x => IsAlgebraic R x)) …
  -/
  exact cardinalMk_lift_le_mul R A
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_mul := cardinalMk_le_mul


@[stacks 09GK]
theorem cardinalMk_le_max : #{ x : A // IsAlgebraic R x } ≤ max #R ℵ₀ := by
  /-
    R A : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.mk (Subtype fun x => IsAlgebraic R x)) (Max.max (Cardinal.mk …
  -/
  rw [← lift_id #_, ← lift_id #R]
  /-
    R A : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : IsDomain A
    inst✝¹ : Algebra R A
    inst✝ : NoZeroSMulDivisors R A
    ⊢ LE.le (Cardinal.lift.{u, u} (Cardinal.mk (Subtype fun x => IsAlgebraic R x)) …
  -/
  exact cardinalMk_lift_le_max R A
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")] alias cardinal_mk_le_max := cardinalMk_le_max


@[simp]
theorem cardinalMk_of_infinite [Infinite R] : #{ x : A // IsAlgebraic R x } = #R :=
  lift_inj.1 <| cardinalMk_lift_of_infinite R A


@[deprecated (since := "2024-11-10")] alias cardinal_mk_of_infinite := cardinalMk_of_infinite


