/-- The cardinality of a nontrivial module over a ring is at least the cardinality of the ring if
there are no zero divisors (for instance if the ring is a field) -/
theorem mk_le_of_module (R : Type u) (E : Type v)
    [AddCommGroup E] [Ring R] [Module R E] [Nontrivial E] [NoZeroSMulDivisors R E] :
    Cardinal.lift.{v} (#R) ≤ Cardinal.lift.{u} (#E) := by
  /-
    R : Type u
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Ring R
    inst✝² : Module R E
    inst✝¹ : Nontrivial E
    inst✝ : NoZeroSMulDivisors R E
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Cardinal …
  -/
  obtain ⟨x, hx⟩ : ∃ (x : E), x ≠ 0 := exists_ne 0
  /-
    case intro
    R : Type u
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Ring R
    inst✝² : Module R E
    inst✝¹ : Nontrivial E
    inst✝ : NoZeroSMulDivisors R E
    x : E
    hx : Ne x 0
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Cardinal …
  -/
  have : Injective (fun k ↦ k • x) := smul_left_injective R hx
  /-
    case intro
    R : Type u
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Ring R
    inst✝² : Module R E
    inst✝¹ : Nontrivial E
    inst✝ : NoZeroSMulDivisors R E
    x : E
    hx : Ne x 0
    this : Function.Injective fun k => HSMul.hSMul k x
    ⊢ LE.le (Cardinal.lift.{v, u} (Cardinal.mk R)) (Cardinal.lift.{u, v} (Cardinal …
  -/
  exact lift_mk_le_lift_mk_of_injective this
  /-
    🎉 no goals
  -/


