/-- The collection of elements of a `MonoidWithZero` that are not left zero divisors form a
`Submonoid`. -/
def nonZeroDivisorsLeft : Submonoid M₀ where
  carrier := {x | ∀ y, y * x = 0 → y = 0}
                 /-
                   M₀ : Type u_1
                   inst✝ : MonoidWithZero M₀
                   ⊢ Membership.mem { carrier := setOf fun x => ∀ (y : M₀), Eq (HMul.hMul y x) 0  …
                 -/
  one_mem' := by simp
                 /-
                   🎉 no goals
                 -/
  mul_mem' {x} {y} hx hy := fun z hz ↦ hx _ <| hy _ (mul_assoc z x y ▸ hz)


@[simp] lemma mem_nonZeroDivisorsLeft_iff {x : M₀} :
    x ∈ nonZeroDivisorsLeft M₀ ↔ ∀ y, y * x = 0 → y = 0 :=
  Iff.rfl


lemma nmem_nonZeroDivisorsLeft_iff {r : M₀} :
    r ∉ nonZeroDivisorsLeft M₀ ↔ {s | s * r = 0 ∧ s ≠ 0}.Nonempty := by
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    r : M₀
    ⊢ Iff (Not (Membership.mem (nonZeroDivisorsLeft M₀) r)) (setOf fun s => And (E …
  -/
  simpa [mem_nonZeroDivisorsLeft_iff] using Set.nonempty_def.symm
  /-
    🎉 no goals
  -/


/-- The collection of elements of a `MonoidWithZero` that are not right zero divisors form a
`Submonoid`. -/
def nonZeroDivisorsRight : Submonoid M₀ where
  carrier := {x | ∀ y, x * y = 0 → y = 0}
                 /-
                   M₀ : Type u_1
                   inst✝ : MonoidWithZero M₀
                   ⊢ Membership.mem { carrier := setOf fun x => ∀ (y : M₀), Eq (HMul.hMul x y) 0  …
                 -/
  one_mem' := by simp
                 /-
                   🎉 no goals
                 -/
  mul_mem' := fun {x} {y} hx hy z hz ↦ hy _ (hx _ ((mul_assoc x y z).symm ▸ hz))


@[simp] lemma mem_nonZeroDivisorsRight_iff {x : M₀} :
    x ∈ nonZeroDivisorsRight M₀ ↔ ∀ y, x * y = 0 → y = 0 :=
  Iff.rfl


lemma nmem_nonZeroDivisorsRight_iff {r : M₀} :
    r ∉ nonZeroDivisorsRight M₀ ↔ {s | r * s = 0 ∧ s ≠ 0}.Nonempty := by
  /-
    M₀ : Type u_1
    inst✝ : MonoidWithZero M₀
    r : M₀
    ⊢ Iff (Not (Membership.mem (nonZeroDivisorsRight M₀) r)) (setOf fun s => And ( …
  -/
  simpa [mem_nonZeroDivisorsRight_iff] using Set.nonempty_def.symm
  /-
    🎉 no goals
  -/


lemma nonZeroDivisorsLeft_eq_right (M₀ : Type*) [CommMonoidWithZero M₀] :
    nonZeroDivisorsLeft M₀ = nonZeroDivisorsRight M₀ := by
  /-
    M₀ : Type u_2
    inst✝ : CommMonoidWithZero M₀
    ⊢ Eq (nonZeroDivisorsLeft M₀) (nonZeroDivisorsRight M₀)
  -/
  ext x; simp [mul_comm x]
         /-
           🎉 no goals
         -/


@[simp] lemma coe_nonZeroDivisorsLeft_eq [NoZeroDivisors M₀] [Nontrivial M₀] :
    nonZeroDivisorsLeft M₀ = {x : M₀ | x ≠ 0} := by
  /-
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    ⊢ Eq (↑(nonZeroDivisorsLeft M₀)) (setOf fun x => Ne x 0)
  -/
  ext x
  simp only [SetLike.mem_coe, mem_nonZeroDivisorsLeft_iff, mul_eq_zero, forall_eq_or_imp, true_and,
    Set.mem_setOf_eq]
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    ⊢ Iff (∀ (a : M₀), Eq x 0 → Eq a 0) (Ne x 0)
  -/
  refine ⟨fun h ↦ ?_, fun hx y hx' ↦ by contradiction⟩
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    h : ∀ (a : M₀), Eq x 0 → Eq a 0
    ⊢ Ne x 0
  -/
  contrapose! h
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    h : Eq x 0
    ⊢ Exists fun a => And (Eq x 0) (Ne a 0)
  -/
  exact ⟨1, h, one_ne_zero⟩
  /-
    🎉 no goals
  -/


@[simp] lemma coe_nonZeroDivisorsRight_eq [NoZeroDivisors M₀] [Nontrivial M₀] :
    nonZeroDivisorsRight M₀ = {x : M₀ | x ≠ 0} := by
  /-
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    ⊢ Eq (↑(nonZeroDivisorsRight M₀)) (setOf fun x => Ne x 0)
  -/
  ext x
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    ⊢ Iff (Membership.mem (↑(nonZeroDivisorsRight M₀)) x) (Membership.mem (setOf f …
  -/
  simp only [SetLike.mem_coe, mem_nonZeroDivisorsRight_iff, mul_eq_zero, Set.mem_setOf_eq]
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    ⊢ Iff (∀ (y : M₀), Or (Eq x 0) (Eq y 0) → Eq y 0) (Ne x 0)
  -/
  refine ⟨fun h ↦ ?_, fun hx y hx' ↦ by aesop⟩
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    h : ∀ (y : M₀), Or (Eq x 0) (Eq y 0) → Eq y 0
    ⊢ Ne x 0
  -/
  contrapose! h
  /-
    case h
    M₀ : Type u_1
    inst✝² : MonoidWithZero M₀
    inst✝¹ : NoZeroDivisors M₀
    inst✝ : Nontrivial M₀
    x : M₀
    h : Eq x 0
    ⊢ Exists fun y => And (Or (Eq x 0) (Eq y 0)) (Ne y 0)
  -/
  exact ⟨1, Or.inl h, one_ne_zero⟩
  /-
    🎉 no goals
  -/


/-- The submonoid of non-zero-divisors of a `MonoidWithZero` `R`. -/
def nonZeroDivisors (R : Type*) [MonoidWithZero R] : Submonoid R where
  carrier := { x | ∀ z, z * x = 0 → z = 0 }
                      /-
                        R : Type u_1
                        inst✝ : MonoidWithZero R
                        x✝ : R
                        hz : Eq (HMul.hMul x✝ 1) 0
                        ⊢ Eq x✝ 0
                      -/
  one_mem' _ hz := by rwa [mul_one] at hz
    /-
      R : Type u_1
      inst✝ : MonoidWithZero R
      a✝ b✝ : R
      hx₁ : Membership.mem (setOf fun x => ∀ (z : R), Eq (HMul.hMul z x) 0 → Eq z 0) …
      hx₂ : Membership.mem (setOf fun x => ∀ (z : R), Eq (HMul.hMul z x) 0 → Eq z 0) …
      x✝ : R
      hz : Eq (HMul.hMul x✝ (HMul.hMul a✝ b✝)) 0
      ⊢ Eq x✝ 0
    -/
                      /-
                        🎉 no goals
                      -/
    /-
      R : Type u_1
      inst✝ : MonoidWithZero R
      a✝ b✝ : R
      hx₁ : Membership.mem (setOf fun x => ∀ (z : R), Eq (HMul.hMul z x) 0 → Eq z 0) …
      hx₂ : Membership.mem (setOf fun x => ∀ (z : R), Eq (HMul.hMul z x) 0 → Eq z 0) …
      x✝ : R
      hz : Eq (HMul.hMul (HMul.hMul x✝ a✝) b✝) 0
      ⊢ Eq x✝ 0
    -/
  mul_mem' hx₁ hx₂ _ hz := by
    /-
      🎉 no goals
    -/
    rw [← mul_assoc] at hz
    exact hx₁ _ (hx₂ _ hz)


/-- The notation for the submonoid of non-zerodivisors. -/
scoped[nonZeroDivisors] notation:9000 R "⁰" => nonZeroDivisors R


/-- Let `R` be a monoid with zero and `M` an additive monoid with an `R`-action, then the collection
of non-zero smul-divisors forms a submonoid. These elements are also called `M`-regular. -/
def nonZeroSMulDivisors (R : Type*) [MonoidWithZero R] (M : Type _) [Zero M] [MulAction R M] :
    Submonoid R where
  carrier := { r | ∀ m : M, r • m = 0 → m = 0}
  one_mem' m h := (one_smul R m) ▸ h
  mul_mem' {r₁ r₂} h₁ h₂ m H := h₂ _ <| h₁ _ <| mul_smul r₁ r₂ m ▸ H


/-- The notation for the submonoid of non-zero smul-divisors. -/
scoped[nonZeroSMulDivisors] notation:9000 R "⁰[" M "]" => nonZeroSMulDivisors R M


theorem mem_nonZeroDivisors_iff {r : M} : r ∈ M⁰ ↔ ∀ x, x * r = 0 → x = 0 := Iff.rfl


lemma nmem_nonZeroDivisors_iff {r : M} : r ∉ M⁰ ↔ {s | s * r = 0 ∧ s ≠ 0}.Nonempty := by
  /-
    M : Type u_1
    inst✝ : MonoidWithZero M
    r : M
    ⊢ Iff (Not (Membership.mem (nonZeroDivisors M) r)) (setOf fun s => And (Eq (HM …
  -/
  simpa [mem_nonZeroDivisors_iff] using Set.nonempty_def.symm
  /-
    🎉 no goals
  -/


theorem mul_right_mem_nonZeroDivisors_eq_zero_iff {x r : M} (hr : r ∈ M⁰) : x * r = 0 ↔ x = 0 :=
            /-
              M : Type u_1
              inst✝ : MonoidWithZero M
              x r : M
              hr : Membership.mem (nonZeroDivisors M) r
              ⊢ Eq x 0 → Eq (HMul.hMul x r) 0
            -/
  ⟨hr _, by simp +contextual⟩
            /-
              🎉 no goals
            -/

@[simp]
theorem mul_right_coe_nonZeroDivisors_eq_zero_iff {x : M} {c : M⁰} : x * c = 0 ↔ x = 0 :=
  mul_right_mem_nonZeroDivisors_eq_zero_iff c.prop


theorem mul_left_mem_nonZeroDivisors_eq_zero_iff {r x : M₁} (hr : r ∈ M₁⁰) : r * x = 0 ↔ x = 0 := by
  /-
    M₁ : Type u_3
    inst✝ : CommMonoidWithZero M₁
    r x : M₁
    hr : Membership.mem (nonZeroDivisors M₁) r
    ⊢ Iff (Eq (HMul.hMul r x) 0) (Eq x 0)
  -/
  rw [mul_comm, mul_right_mem_nonZeroDivisors_eq_zero_iff hr]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_left_coe_nonZeroDivisors_eq_zero_iff {c : M₁⁰} {x : M₁} : (c : M₁) * x = 0 ↔ x = 0 :=
  mul_left_mem_nonZeroDivisors_eq_zero_iff c.prop


theorem mul_cancel_right_mem_nonZeroDivisors {x y r : R} (hr : r ∈ R⁰) : x * r = y * r ↔ x = y := by
  /-
    R : Type u_4
    inst✝ : Ring R
    x y r : R
    hr : Membership.mem (nonZeroDivisors R) r
    ⊢ Iff (Eq (HMul.hMul x r) (HMul.hMul y r)) (Eq x y)
  -/
  refine ⟨fun h ↦ ?_, congrArg (· * r)⟩
  /-
    R : Type u_4
    inst✝ : Ring R
    x y r : R
    hr : Membership.mem (nonZeroDivisors R) r
    h : Eq (HMul.hMul x r) (HMul.hMul y r)
    ⊢ Eq x y
  -/
  rw [← sub_eq_zero, ← mul_right_mem_nonZeroDivisors_eq_zero_iff hr, sub_mul, h, sub_self]
  /-
    🎉 no goals
  -/


theorem mul_cancel_right_coe_nonZeroDivisors {x y : R} {c : R⁰} : x * c = y * c ↔ x = y :=
  mul_cancel_right_mem_nonZeroDivisors c.prop


@[simp]
theorem mul_cancel_left_mem_nonZeroDivisors {x y r : R'} (hr : r ∈ R'⁰) :
    r * x = r * y ↔ x = y := by
  /-
    R' : Type u_5
    inst✝ : CommRing R'
    x y r : R'
    hr : Membership.mem (nonZeroDivisors R') r
    ⊢ Iff (Eq (HMul.hMul r x) (HMul.hMul r y)) (Eq x y)
  -/
  simp_rw [mul_comm r, mul_cancel_right_mem_nonZeroDivisors hr]
  /-
    🎉 no goals
  -/


theorem mul_cancel_left_coe_nonZeroDivisors {x y : R'} {c : R'⁰} : (c : R') * x = c * y ↔ x = y :=
  mul_cancel_left_mem_nonZeroDivisors c.prop


theorem dvd_cancel_right_mem_nonZeroDivisors {x y r : R'} (hr : r ∈ R'⁰) : x * r ∣ y * r ↔ x ∣ y :=
  ⟨fun ⟨z, _⟩ ↦ ⟨z, by rwa [← mul_cancel_right_mem_nonZeroDivisors hr, mul_assoc, mul_comm z r,
                                       /-
                                         R' : Type u_5
                                         inst✝ : CommRing R'
                                         x y r : R'
                                         hr : Membership.mem (nonZeroDivisors R') r
                                         x✝ : Dvd.dvd x y
                                         z : R'
                                         h : Eq y (HMul.hMul x z)
                                         ⊢ Eq (HMul.hMul y r) (HMul.hMul (HMul.hMul x r) z)
                                       -/
    ← mul_assoc]⟩, fun ⟨z, h⟩ ↦ ⟨z, by rw [h, mul_assoc, mul_comm z r, ← mul_assoc]⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem dvd_cancel_right_coe_nonZeroDivisors {x y : R'} {c : R'⁰} : x * c ∣ y * c ↔ x ∣ y :=
  dvd_cancel_right_mem_nonZeroDivisors c.prop


theorem dvd_cancel_left_mem_nonZeroDivisors {x y r : R'} (hr : r ∈ R'⁰) : r * x ∣ r * y ↔ x ∣ y :=
                       /-
                         R' : Type u_5
                         inst✝ : CommRing R'
                         x y r : R'
                         hr : Membership.mem (nonZeroDivisors R') r
                         x✝ : Dvd.dvd (HMul.hMul r x) (HMul.hMul r y)
                         z : R'
                         h✝ : Eq (HMul.hMul r y) (HMul.hMul (HMul.hMul r x) z)
                         ⊢ Eq y (HMul.hMul x z)
                       -/
  ⟨fun ⟨z, _⟩ ↦ ⟨z, by rwa [← mul_cancel_left_mem_nonZeroDivisors hr, ← mul_assoc]⟩,
                       /-
                         🎉 no goals
                       -/
                        /-
                          R' : Type u_5
                          inst✝ : CommRing R'
                          x y r : R'
                          hr : Membership.mem (nonZeroDivisors R') r
                          x✝ : Dvd.dvd x y
                          z : R'
                          h : Eq y (HMul.hMul x z)
                          ⊢ Eq (HMul.hMul r y) (HMul.hMul (HMul.hMul r x) z)
                        -/
    fun ⟨z, h⟩ ↦ ⟨z, by rw [h, ← mul_assoc]⟩⟩
                        /-
                          🎉 no goals
                        -/


theorem dvd_cancel_left_coe_nonZeroDivisors {x y : R'} {c : R'⁰} : c * x ∣ c * y ↔ x ∣ y :=
  dvd_cancel_left_mem_nonZeroDivisors c.prop


theorem zero_not_mem_nonZeroDivisors [Nontrivial M] : 0 ∉ M⁰ :=
  fun h ↦ one_ne_zero <| h 1 <| mul_zero _


theorem nonZeroDivisors.ne_zero [Nontrivial M] {x} (hx : x ∈ M⁰) : x ≠ 0 :=
  ne_of_mem_of_not_mem hx zero_not_mem_nonZeroDivisors


@[simp]
theorem nonZeroDivisors.coe_ne_zero [Nontrivial M] (x : M⁰) : (x : M) ≠ 0 :=
  nonZeroDivisors.ne_zero x.2


theorem mul_mem_nonZeroDivisors {a b : M₁} : a * b ∈ M₁⁰ ↔ a ∈ M₁⁰ ∧ b ∈ M₁⁰ := by
  /-
    M₁ : Type u_3
    inst✝ : CommMonoidWithZero M₁
    a b : M₁
    ⊢ Iff (Membership.mem (nonZeroDivisors M₁) (HMul.hMul a b)) (And (Membership.m …
  -/
  constructor
    /-
      case mp
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      ⊢ Membership.mem (nonZeroDivisors M₁) (HMul.hMul a b) → And (Membership.mem (n …
    -/
  · intro h
    /-
      case mp
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      h : Membership.mem (nonZeroDivisors M₁) (HMul.hMul a b)
      ⊢ And (Membership.mem (nonZeroDivisors M₁) a) (Membership.mem (nonZeroDivisors …
    -/
    constructor <;> intro x h' <;> apply h
      /-
        case mp.left.a
        M₁ : Type u_3
        inst✝ : CommMonoidWithZero M₁
        a b : M₁
        h : Membership.mem (nonZeroDivisors M₁) (HMul.hMul a b)
        x : M₁
        h' : Eq (HMul.hMul x a) 0
        ⊢ Eq (HMul.hMul x (HMul.hMul a b)) 0
      -/
    · rw [← mul_assoc, h', zero_mul]
      /-
        🎉 no goals
      -/
      /-
        case mp.right.a
        M₁ : Type u_3
        inst✝ : CommMonoidWithZero M₁
        a b : M₁
        h : Membership.mem (nonZeroDivisors M₁) (HMul.hMul a b)
        x : M₁
        h' : Eq (HMul.hMul x b) 0
        ⊢ Eq (HMul.hMul x (HMul.hMul a b)) 0
      -/
    · rw [mul_comm a b, ← mul_assoc, h', zero_mul]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      ⊢ And (Membership.mem (nonZeroDivisors M₁) a) (Membership.mem (nonZeroDivisors …
    -/
  · rintro ⟨ha, hb⟩ x hx
    /-
      case mpr.intro
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      ha : Membership.mem (nonZeroDivisors M₁) a
      hb : Membership.mem (nonZeroDivisors M₁) b
      x : M₁
      hx : Eq (HMul.hMul x (HMul.hMul a b)) 0
      ⊢ Eq x 0
    -/
    apply ha
    /-
      case mpr.intro.a
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      ha : Membership.mem (nonZeroDivisors M₁) a
      hb : Membership.mem (nonZeroDivisors M₁) b
      x : M₁
      hx : Eq (HMul.hMul x (HMul.hMul a b)) 0
      ⊢ Eq (HMul.hMul x a) 0
    -/
    apply hb
    /-
      case mpr.intro.a.a
      M₁ : Type u_3
      inst✝ : CommMonoidWithZero M₁
      a b : M₁
      ha : Membership.mem (nonZeroDivisors M₁) a
      hb : Membership.mem (nonZeroDivisors M₁) b
      x : M₁
      hx : Eq (HMul.hMul x (HMul.hMul a b)) 0
      ⊢ Eq (HMul.hMul (HMul.hMul x a) b) 0
    -/
    rw [mul_assoc, hx]
    /-
      🎉 no goals
    -/


lemma IsUnit.mem_nonZeroDivisors {a : M} (ha : IsUnit a) : a ∈ M⁰ :=
  fun _ h ↦ ha.mul_left_eq_zero.mp h


theorem eq_zero_of_ne_zero_of_mul_right_eq_zero [NoZeroDivisors M] {x y : M} (hnx : x ≠ 0)
    (hxy : y * x = 0) : y = 0 :=
  Or.resolve_right (eq_zero_or_eq_zero_of_mul_eq_zero hxy) hnx


theorem eq_zero_of_ne_zero_of_mul_left_eq_zero [NoZeroDivisors M] {x y : M} (hnx : x ≠ 0)
    (hxy : x * y = 0) : y = 0 :=
  Or.resolve_left (eq_zero_or_eq_zero_of_mul_eq_zero hxy) hnx


theorem mem_nonZeroDivisors_of_ne_zero [NoZeroDivisors M] {x : M} (hx : x ≠ 0) : x ∈ M⁰ := fun _ ↦
  eq_zero_of_ne_zero_of_mul_right_eq_zero hx


@[simp] lemma mem_nonZeroDivisors_iff_ne_zero [NoZeroDivisors M] [Nontrivial M] {x : M} :
    x ∈ M⁰ ↔ x ≠ 0 := ⟨nonZeroDivisors.ne_zero, mem_nonZeroDivisors_of_ne_zero⟩


theorem map_ne_zero_of_mem_nonZeroDivisors [Nontrivial M] [ZeroHomClass F M M'] (g : F)
    (hg : Function.Injective (g : M → M')) {x : M} (h : x ∈ M⁰) : g x ≠ 0 := fun h0 ↦
  one_ne_zero (h 1 ((one_mul x).symm ▸ hg (h0.trans (map_zero g).symm)))


theorem map_mem_nonZeroDivisors [Nontrivial M] [NoZeroDivisors M'] [ZeroHomClass F M M'] (g : F)
    (hg : Function.Injective g) {x : M} (h : x ∈ M⁰) : g x ∈ M'⁰ := fun _ hz ↦
  eq_zero_of_ne_zero_of_mul_right_eq_zero (map_ne_zero_of_mem_nonZeroDivisors g hg h) hz


theorem MulEquivClass.map_nonZeroDivisors {R S F : Type*} [MonoidWithZero R] [MonoidWithZero S]
    [EquivLike F R S] [MulEquivClass F R S] (h : F) :
    Submonoid.map h (nonZeroDivisors R) = nonZeroDivisors S := by
  /-
    R : Type u_7
    S : Type u_8
    F : Type u_9
    inst✝³ : MonoidWithZero R
    inst✝² : MonoidWithZero S
    inst✝¹ : EquivLike F R S
    inst✝ : MulEquivClass F R S
    h : F
    ⊢ Eq (Submonoid.map h (nonZeroDivisors R)) (nonZeroDivisors S)
  -/
  let h : R ≃* S := h
  /-
    R : Type u_7
    S : Type u_8
    F : Type u_9
    inst✝³ : MonoidWithZero R
    inst✝² : MonoidWithZero S
    inst✝¹ : EquivLike F R S
    inst✝ : MulEquivClass F R S
    h✝ : F
    h : MulEquiv R S := ↑h✝
    ⊢ Eq (Submonoid.map h✝ (nonZeroDivisors R)) (nonZeroDivisors S)
  -/
  show Submonoid.map h.toMonoidHom _ = _
  /-
    R : Type u_7
    S : Type u_8
    F : Type u_9
    inst✝³ : MonoidWithZero R
    inst✝² : MonoidWithZero S
    inst✝¹ : EquivLike F R S
    inst✝ : MulEquivClass F R S
    h✝ : F
    h : MulEquiv R S := ↑h✝
    ⊢ Eq (Submonoid.map h.toMonoidHom (nonZeroDivisors R)) (nonZeroDivisors S)
  -/
  ext
  simp_rw [Submonoid.map_equiv_eq_comap_symm, Submonoid.mem_comap, mem_nonZeroDivisors_iff,
    ← h.symm.forall_congr_right, h.symm.coe_toMonoidHom, h.symm.toEquiv_eq_coe, h.symm.coe_toEquiv,
    ← map_mul, map_eq_zero_iff _ h.symm.injective]


theorem le_nonZeroDivisors_of_noZeroDivisors [NoZeroDivisors M] {S : Submonoid M}
    (hS : (0 : M) ∉ S) : S ≤ M⁰ := fun _ hx _ hy ↦
  Or.recOn (eq_zero_or_eq_zero_of_mul_eq_zero hy) id fun h ↦
    absurd (h ▸ hx : (0 : M) ∈ S) hS


theorem powers_le_nonZeroDivisors_of_noZeroDivisors [NoZeroDivisors M] {a : M} (ha : a ≠ 0) :
    Submonoid.powers a ≤ M⁰ :=
  le_nonZeroDivisors_of_noZeroDivisors fun h ↦ absurd (h.recOn fun _ hn ↦ pow_eq_zero hn) ha


theorem map_le_nonZeroDivisors_of_injective [NoZeroDivisors M'] [MonoidWithZeroHomClass F M M']
    (f : F) (hf : Function.Injective f) {S : Submonoid M} (hS : S ≤ M⁰) : S.map f ≤ M'⁰ := by
  /-
    M : Type u_1
    M' : Type u_2
    F : Type u_6
    inst✝⁴ : MonoidWithZero M
    inst✝³ : MonoidWithZero M'
    inst✝² : FunLike F M M'
    inst✝¹ : NoZeroDivisors M'
    inst✝ : MonoidWithZeroHomClass F M M'
    f : F
    hf : Function.Injective ⇑f
    S : Submonoid M
    hS : LE.le S (nonZeroDivisors M)
    ⊢ LE.le (Submonoid.map f S) (nonZeroDivisors M')
  -/
  cases subsingleton_or_nontrivial M
    /-
      case inl
      M : Type u_1
      M' : Type u_2
      F : Type u_6
      inst✝⁴ : MonoidWithZero M
      inst✝³ : MonoidWithZero M'
      inst✝² : FunLike F M M'
      inst✝¹ : NoZeroDivisors M'
      inst✝ : MonoidWithZeroHomClass F M M'
      f : F
      hf : Function.Injective ⇑f
      S : Submonoid M
      hS : LE.le S (nonZeroDivisors M)
      h✝ : Subsingleton M
      ⊢ LE.le (Submonoid.map f S) (nonZeroDivisors M')
    -/
  · simp [Subsingleton.elim S ⊥]
    /-
      🎉 no goals
    -/
  · exact le_nonZeroDivisors_of_noZeroDivisors fun h ↦
      let ⟨x, hx, hx0⟩ := h
      zero_ne_one (hS (hf (hx0.trans (map_zero f).symm) ▸ hx : 0 ∈ S) 1 (mul_zero 1)).symm


theorem nonZeroDivisors_le_comap_nonZeroDivisors_of_injective [NoZeroDivisors M']
    [MonoidWithZeroHomClass F M M'] (f : F) (hf : Function.Injective f) : M⁰ ≤ M'⁰.comap f :=
  Submonoid.le_comap_of_map_le _ (map_le_nonZeroDivisors_of_injective _ hf le_rfl)


/-- In a finite ring, an element is a unit iff it is a non-zero-divisor. -/
lemma isUnit_iff_mem_nonZeroDivisors_of_finite [Finite R] {a : R} :
    IsUnit a ↔ a ∈ nonZeroDivisors R := by
  /-
    R : Type u_4
    inst✝¹ : Ring R
    inst✝ : Finite R
    a : R
    ⊢ Iff (IsUnit a) (Membership.mem (nonZeroDivisors R) a)
  -/
  refine ⟨IsUnit.mem_nonZeroDivisors, fun ha ↦ ?_⟩
  /-
    R : Type u_4
    inst✝¹ : Ring R
    inst✝ : Finite R
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    ⊢ IsUnit a
  -/
  rw [IsUnit.isUnit_iff_mulRight_bijective, ← Finite.injective_iff_bijective]
  /-
    R : Type u_4
    inst✝¹ : Ring R
    inst✝ : Finite R
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    ⊢ Function.Injective fun x => HMul.hMul x a
  -/
  intro b c hbc
  /-
    R : Type u_4
    inst✝¹ : Ring R
    inst✝ : Finite R
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    b c : R
    hbc : Eq ((fun x => HMul.hMul x a) b) ((fun x => HMul.hMul x a) c)
    ⊢ Eq b c
  -/
  rw [← sub_eq_zero, ← sub_mul] at hbc
  /-
    R : Type u_4
    inst✝¹ : Ring R
    inst✝ : Finite R
    a : R
    ha : Membership.mem (nonZeroDivisors R) a
    b c : R
    hbc : Eq (HMul.hMul (HSub.hSub b c) a) 0
    ⊢ Eq b c
  -/
  exact sub_eq_zero.mp (ha _ hbc)
  /-
    🎉 no goals
  -/


/-- Canonical isomorphism between the non-zero-divisors and units of a group with zero. -/
@[simps]
noncomputable def nonZeroDivisorsEquivUnits : G₀⁰ ≃* G₀ˣ where
  toFun u := .mk0 _ <| mem_nonZeroDivisors_iff_ne_zero.1 u.2
  invFun u := ⟨u, u.isUnit.mem_nonZeroDivisors⟩
  left_inv u := rfl
                    /-
                      G₀ : Type u_1
                      inst✝ : GroupWithZero G₀
                      x : G₀
                      u : Units G₀
                      ⊢ Eq ((fun u => Units.mk0 ↑u ⋯) ((fun u => ⟨↑u, ⋯⟩) u)) u
                    -/
  right_inv u := by simp
                    /-
                      🎉 no goals
                    -/
                     /-
                       G₀ : Type u_1
                       inst✝ : GroupWithZero G₀
                       x : G₀
                       u v : Subtype fun x => Membership.mem (nonZeroDivisors G₀) x
                       ⊢ Eq ({ toFun := fun u => Units.mk0 ↑u ⋯, invFun := fun u => ⟨↑u, ⋯⟩, left_inv …
                     -/
  map_mul' u v := by simp
                     /-
                       🎉 no goals
                     -/


lemma isUnit_of_mem_nonZeroDivisors (hx : x ∈ nonZeroDivisors G₀) : IsUnit x :=
  (nonZeroDivisorsEquivUnits ⟨x, hx⟩).isUnit


lemma mem_nonZeroSMulDivisors_iff {x : R} : x ∈ R⁰[M] ↔ ∀ (m : M), x • m = 0 → m = 0 := Iff.rfl


@[simp]
lemma unop_nonZeroSMulDivisors_mulOpposite_eq_nonZeroDivisors :
    (Rᵐᵒᵖ ⁰[R]).unop = R⁰ := rfl


/-- The non-zero `•`-divisors with `•` as right multiplication correspond with the non-zero
divisors. Note that the `MulOpposite` is needed because we defined `nonZeroDivisors` with
multiplication on the right. -/
lemma nonZeroSMulDivisors_mulOpposite_eq_op_nonZeroDivisors :
    Rᵐᵒᵖ ⁰[R] = R⁰.op := rfl


/-- The units of the monoid of non-zero divisors of `M₀` are equivalent to the units of `M₀`. -/
@[simps]
def unitsNonZeroDivisorsEquiv : M₀⁰ˣ ≃* M₀ˣ where
  __ := Units.map M₀⁰.subtype
  invFun u := ⟨⟨u, u.isUnit.mem_nonZeroDivisors⟩, ⟨(u⁻¹ : M₀ˣ), u⁻¹.isUnit.mem_nonZeroDivisors⟩,
       /-
         M₀ : Type ?u.63735
         inst✝ : MonoidWithZero M₀
         a b : Subtype fun x => Membership.mem (nonZeroDivisors M₀) x
         u : Units M₀
         ⊢ Eq (HMul.hMul ⟨↑u, ⋯⟩ ⟨↑(Inv.inv u), ⋯⟩) 1
       -/
       /-
         🎉 no goals
       -/
    by simp, by simp⟩
                /-
                  🎉 no goals
                -/
  left_inv _ := rfl
  right_inv _ := rfl


@[simp, norm_cast] lemma nonZeroDivisors.associated_coe : Associated (a : M₀) b ↔ Associated a b :=
                                                               /-
                                                                 M₀ : Type u_1
                                                                 inst✝ : MonoidWithZero M₀
                                                                 a b : Subtype fun x => Membership.mem (nonZeroDivisors M₀) x
                                                                 ⊢ Iff (Exists fun b_1 => Eq (HMul.hMul ↑a ↑(unitsNonZeroDivisorsEquiv.symm.sym …
                                                               -/
  unitsNonZeroDivisorsEquiv.symm.exists_congr_left.trans <| by simp [Associated]; norm_cast
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem mk_mem_nonZeroDivisors_associates : Associates.mk a ∈ (Associates M₀)⁰ ↔ a ∈ M₀⁰ := by
  /-
    M₀ : Type u_1
    inst✝ : CommMonoidWithZero M₀
    a : M₀
    ⊢ Iff (Membership.mem (nonZeroDivisors (Associates M₀)) (Associates.mk a)) (Me …
  -/
  rw [mem_nonZeroDivisors_iff, mem_nonZeroDivisors_iff, ← not_iff_not]
  /-
    M₀ : Type u_1
    inst✝ : CommMonoidWithZero M₀
    a : M₀
    ⊢ Iff (Not (∀ (x : Associates M₀), Eq (HMul.hMul x (Associates.mk a)) 0 → Eq x …
  -/
  push_neg
  /-
    M₀ : Type u_1
    inst✝ : CommMonoidWithZero M₀
    a : M₀
    ⊢ Iff (Exists fun x => And (Eq (HMul.hMul x (Associates.mk a)) 0) (Ne x 0)) (E …
  -/
  constructor
    /-
      case mp
      M₀ : Type u_1
      inst✝ : CommMonoidWithZero M₀
      a : M₀
      ⊢ (Exists fun x => And (Eq (HMul.hMul x (Associates.mk a)) 0) (Ne x 0)) → Exis …
    -/
  · rintro ⟨⟨x⟩, hx₁, hx₂⟩
    /-
      case mp.intro.mk.intro
      M₀ : Type u_1
      inst✝ : CommMonoidWithZero M₀
      a : M₀
      w✝ : Associates M₀
      x : M₀
      hx₁ : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M₀)) x) (Associates.mk a)) 0
      hx₂ : Ne (Quot.mk (⇑(Associated.setoid M₀)) x) 0
      ⊢ Exists fun x => And (Eq (HMul.hMul x a) 0) (Ne x 0)
    -/
    refine ⟨x, ?_, ?_⟩
      /-
        case mp.intro.mk.intro.refine_1
        M₀ : Type u_1
        inst✝ : CommMonoidWithZero M₀
        a : M₀
        w✝ : Associates M₀
        x : M₀
        hx₁ : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M₀)) x) (Associates.mk a)) 0
        hx₂ : Ne (Quot.mk (⇑(Associated.setoid M₀)) x) 0
        ⊢ Eq (HMul.hMul x a) 0
      -/
    · rwa [← Associates.mk_eq_zero, ← Associates.mk_mul_mk, ← Associates.quot_mk_eq_mk]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.mk.intro.refine_2
        M₀ : Type u_1
        inst✝ : CommMonoidWithZero M₀
        a : M₀
        w✝ : Associates M₀
        x : M₀
        hx₁ : Eq (HMul.hMul (Quot.mk (⇑(Associated.setoid M₀)) x) (Associates.mk a)) 0
        hx₂ : Ne (Quot.mk (⇑(Associated.setoid M₀)) x) 0
        ⊢ Ne x 0
      -/
    · rwa [← Associates.mk_ne_zero, ← Associates.quot_mk_eq_mk]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      M₀ : Type u_1
      inst✝ : CommMonoidWithZero M₀
      a : M₀
      ⊢ (Exists fun x => And (Eq (HMul.hMul x a) 0) (Ne x 0)) → Exists fun x => And  …
    -/
  · refine fun ⟨b, hb₁, hb₂⟩ ↦ ⟨Associates.mk b, ?_, by rwa [Associates.mk_ne_zero]⟩
    /-
      case mpr
      M₀ : Type u_1
      inst✝ : CommMonoidWithZero M₀
      a : M₀
      x✝ : Exists fun x => And (Eq (HMul.hMul x a) 0) (Ne x 0)
      b : M₀
      hb₁ : Eq (HMul.hMul b a) 0
      hb₂ : Ne b 0
      ⊢ Eq (HMul.hMul (Associates.mk b) (Associates.mk a)) 0
    -/
    rw [Associates.mk_mul_mk, hb₁, Associates.mk_zero]
    /-
      🎉 no goals
    -/


/-- The non-zero divisors of associates of a monoid with zero `M₀` are isomorphic to the associates
of the non-zero divisors of `M₀` under the map `⟨⟦a⟧, _⟩ ↦ ⟦⟨a, _⟩⟧`. -/
def associatesNonZeroDivisorsEquiv : (Associates M₀)⁰ ≃* Associates M₀⁰ where
  toEquiv := .subtypeQuotientEquivQuotientSubtype _ (s₂ := Associated.setoid _)
    (· ∈ nonZeroDivisors _)
        /-
          M₀✝ : ?m.70260
          M₀ : Type u_1
          inst✝ : CommMonoidWithZero M₀
          a : M₀
          ⊢ ∀ (a : M₀), Iff (Membership.mem (nonZeroDivisors M₀) a) ((fun x => Membershi …
        -/
    (by simp [mem_nonZeroDivisors_iff, Quotient.forall, Associates.mk_mul_mk])
        /-
          🎉 no goals
        -/
        /-
          M₀✝ : ?m.70260
          M₀ : Type u_1
          inst✝ : CommMonoidWithZero M₀
          a : M₀
          ⊢ ∀ (x y : Subtype fun x => Membership.mem (nonZeroDivisors M₀) x), Iff ((Asso …
        -/
    (by simp [Associated.setoid])
        /-
          🎉 no goals
        -/
                 /-
                   M₀✝ : ?m.70260
                   M₀ : Type u_1
                   inst✝ : CommMonoidWithZero M₀
                   a : M₀
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem (nonZeroDivisors (Associates M₀)) x …
                 -/
  map_mul' := by simp [Quotient.forall, Associates.mk_mul_mk]
                 /-
                   🎉 no goals
                 -/


@[simp]
lemma associatesNonZeroDivisorsEquiv_mk_mk (a : M₀) (ha) :
    associatesNonZeroDivisorsEquiv ⟨⟦a⟧, ha⟩ = ⟦⟨a, mk_mem_nonZeroDivisors_associates.1 ha⟩⟧ := rfl


@[simp]
lemma associatesNonZeroDivisorsEquiv_symm_mk_mk (a : M₀) (ha) :
    associatesNonZeroDivisorsEquiv.symm ⟦⟨a, ha⟩⟧ = ⟨⟦a⟧, mk_mem_nonZeroDivisors_associates.2 ha⟩ :=
  rfl


