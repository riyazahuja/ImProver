/-- Transfer `One` across an `Equiv` -/
@[to_additive "Transfer `Zero` across an `Equiv`"]
protected abbrev one [One β] : One α :=
  ⟨e.symm 1⟩


@[to_additive]
theorem one_def [One β] :
    letI := e.one
    1 = e.symm 1 :=
  rfl


@[to_additive]
noncomputable instance [Small.{v} α] [One α] : One (Shrink.{v} α) :=
  (equivShrink α).symm.one


/-- Transfer `Mul` across an `Equiv` -/
@[to_additive "Transfer `Add` across an `Equiv`"]
protected abbrev mul [Mul β] : Mul α :=
  ⟨fun x y => e.symm (e x * e y)⟩


@[to_additive]
theorem mul_def [Mul β] (x y : α) :
    letI := Equiv.mul e
    x * y = e.symm (e x * e y) :=
  rfl


@[to_additive]
noncomputable instance [Small.{v} α] [Mul α] : Mul (Shrink.{v} α) :=
  (equivShrink α).symm.mul


/-- Transfer `Div` across an `Equiv` -/
@[to_additive "Transfer `Sub` across an `Equiv`"]
protected abbrev div [Div β] : Div α :=
  ⟨fun x y => e.symm (e x / e y)⟩


@[to_additive]
theorem div_def [Div β] (x y : α) :
    letI := Equiv.div e
    x / y = e.symm (e x / e y) :=
  rfl


@[to_additive]
noncomputable instance [Small.{v} α] [Div α] : Div (Shrink.{v} α) :=
  (equivShrink α).symm.div

-- Porting note: this should be called `inv`,
-- but we already have an `Equiv.inv` (which perhaps should move to `Perm.inv`?)

/-- Transfer `Inv` across an `Equiv` -/
@[to_additive "Transfer `Neg` across an `Equiv`"]
protected abbrev Inv [Inv β] : Inv α :=
  ⟨fun x => e.symm (e x)⁻¹⟩


@[to_additive]
theorem inv_def [Inv β] (x : α) :
    letI := Equiv.Inv e
    x⁻¹ = e.symm (e x)⁻¹ :=
  rfl


@[to_additive]
noncomputable instance [Small.{v} α] [Inv α] : Inv (Shrink.{v} α) :=
  (equivShrink α).symm.Inv


/-- Transfer `SMul` across an `Equiv` -/
protected abbrev smul (R : Type*) [SMul R β] : SMul R α :=
  ⟨fun r x => e.symm (r • e x)⟩


theorem smul_def {R : Type*} [SMul R β] (r : R) (x : α) :
    letI := e.smul R
    r • x = e.symm (r • e x) :=
  rfl


noncomputable instance [Small.{v} α] (R : Type*) [SMul R α] : SMul R (Shrink.{v} α) :=
  (equivShrink α).symm.smul R


/-- Transfer `Pow` across an `Equiv` -/
@[reducible, to_additive existing smul]
protected def pow (N : Type*) [Pow β N] : Pow α N :=
  ⟨fun x n => e.symm (e x ^ n)⟩


theorem pow_def {N : Type*} [Pow β N] (n : N) (x : α) :
    letI := e.pow N
    x ^ n = e.symm (e x ^ n) :=
  rfl


noncomputable instance [Small.{v} α] (N : Type*) [Pow α N] : Pow (Shrink.{v} α) N :=
  (equivShrink α).symm.pow N


/-- An equivalence `e : α ≃ β` gives a multiplicative equivalence `α ≃* β` where
the multiplicative structure on `α` is the one obtained by transporting a multiplicative structure
on `β` back along `e`. -/
@[to_additive "An equivalence `e : α ≃ β` gives an additive equivalence `α ≃+ β` where
the additive structure on `α` is the one obtained by transporting an additive structure
on `β` back along `e`."]
def mulEquiv (e : α ≃ β) [Mul β] :
    let _ := Equiv.mul e
    α ≃* β := by
  /-
    α : Type u
    β : Type v
    e✝ e : Equiv α β
    inst✝ : Mul β
    ⊢ let x := e.mul;
      MulEquiv α β
  -/
  intros
  exact
    { e with
      map_mul' := fun x y => by
        apply e.symm.injective
        simp [mul_def] }


@[to_additive (attr := simp)]
theorem mulEquiv_apply (e : α ≃ β) [Mul β] (a : α) : (mulEquiv e) a = e a :=
  rfl


@[to_additive]
theorem mulEquiv_symm_apply (e : α ≃ β) [Mul β] (b : β) :
    letI := Equiv.mul e
    (mulEquiv e).symm b = e.symm b :=
  rfl


/-- Shrink `α` to a smaller universe preserves multiplication. -/
@[to_additive "Shrink `α` to a smaller universe preserves addition."]
noncomputable def _root_.Shrink.mulEquiv [Small.{v} α] [Mul α] : Shrink.{v} α ≃* α :=
  (equivShrink α).symm.mulEquiv


/-- An equivalence `e : α ≃ β` gives a ring equivalence `α ≃+* β`
where the ring structure on `α` is
the one obtained by transporting a ring structure on `β` back along `e`.
-/
def ringEquiv (e : α ≃ β) [Add β] [Mul β] : by
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      ⊢ Sort ?u.6355
    -/
    let add := Equiv.add e
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      add : Add α := e.add
      ⊢ Sort ?u.6355
    -/
    let mul := Equiv.mul e
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      add : Add α := e.add
      mul : Mul α := e.mul
      ⊢ Sort ?u.6355
    -/
    exact α ≃+* β := by
    /-
      🎉 no goals
    -/
  /-
    α : Type u
    β : Type v
    e✝ e : Equiv α β
    inst✝¹ : Add β
    inst✝ : Mul β
    ⊢ let add := e.add;
      let mul := e.mul;
      RingEquiv α β
  -/
  intros
  exact
    { e with
      map_add' := fun x y => by
        apply e.symm.injective
        simp [add_def]
      map_mul' := fun x y => by
        apply e.symm.injective
        simp [mul_def] }


@[simp]
theorem ringEquiv_apply (e : α ≃ β) [Add β] [Mul β] (a : α) : (ringEquiv e) a = e a :=
  rfl


theorem ringEquiv_symm_apply (e : α ≃ β) [Add β] [Mul β] (b : β) : by
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      b : β
      ⊢ Sort ?u.10845
    -/
    letI := Equiv.add e
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      b : β
      this : Add α := e.add
      ⊢ Sort ?u.10845
    -/
    letI := Equiv.mul e
    /-
      α : Type u
      β : Type v
      e✝ e : Equiv α β
      inst✝¹ : Add β
      inst✝ : Mul β
      b : β
      this✝ : Add α := e.add
      this : Mul α := e.mul
      ⊢ Sort ?u.10845
    -/
    exact (ringEquiv e).symm b = e.symm b := rfl
    /-
      🎉 no goals
    -/


variable (α) in
/-- Shrink `α` to a smaller universe preserves ring structure. -/
noncomputable def _root_.Shrink.ringEquiv [Small.{v} α] [Add α] [Mul α] : Shrink.{v} α ≃+* α :=
  (equivShrink α).symm.ringEquiv


/-- Transfer `Semigroup` across an `Equiv` -/
@[to_additive "Transfer `add_semigroup` across an `Equiv`"]
protected abbrev semigroup [Semigroup β] : Semigroup α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semigroup β
    ⊢ Semigroup α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semigroup β
    mul : Mul α := e.mul
    ⊢ Semigroup α
  -/
  apply e.injective.semigroup _; intros; exact e.apply_symm_apply _
                                         /-
                                           🎉 no goals
                                         -/


@[to_additive]
noncomputable instance [Small.{v} α] [Semigroup α] : Semigroup (Shrink.{v} α) :=
  (equivShrink α).symm.semigroup


/-- Transfer `SemigroupWithZero` across an `Equiv` -/
protected abbrev semigroupWithZero [SemigroupWithZero β] : SemigroupWithZero α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : SemigroupWithZero β
    ⊢ SemigroupWithZero α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : SemigroupWithZero β
    mul : Mul α := e.mul
    ⊢ SemigroupWithZero α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : SemigroupWithZero β
    mul : Mul α := e.mul
    zero : Zero α := e.zero
    ⊢ SemigroupWithZero α
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  apply e.injective.semigroupWithZero _ <;> intros <;> exact e.apply_symm_apply _
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive]
noncomputable instance [Small.{v} α] [SemigroupWithZero α] : SemigroupWithZero (Shrink.{v} α) :=
  (equivShrink α).symm.semigroupWithZero


/-- Transfer `CommSemigroup` across an `Equiv` -/
@[to_additive "Transfer `AddCommSemigroup` across an `Equiv`"]
protected abbrev commSemigroup [CommSemigroup β] : CommSemigroup α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemigroup β
    ⊢ CommSemigroup α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemigroup β
    mul : Mul α := e.mul
    ⊢ CommSemigroup α
  -/
  apply e.injective.commSemigroup _; intros; exact e.apply_symm_apply _
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive]
noncomputable instance [Small.{v} α] [CommSemigroup α] : CommSemigroup (Shrink.{v} α) :=
  (equivShrink α).symm.commSemigroup


/-- Transfer `MulZeroClass` across an `Equiv` -/
protected abbrev mulZeroClass [MulZeroClass β] : MulZeroClass α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroClass β
    ⊢ MulZeroClass α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroClass β
    zero : Zero α := e.zero
    ⊢ MulZeroClass α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroClass β
    zero : Zero α := e.zero
    mul : Mul α := e.mul
    ⊢ MulZeroClass α
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  apply e.injective.mulZeroClass _ <;> intros <;> exact e.apply_symm_apply _
                                                  /-
                                                    🎉 no goals
                                                  -/


noncomputable instance [Small.{v} α] [MulZeroClass α] : MulZeroClass (Shrink.{v} α) :=
  (equivShrink α).symm.mulZeroClass


/-- Transfer `MulOneClass` across an `Equiv` -/
@[to_additive "Transfer `AddZeroClass` across an `Equiv`"]
protected abbrev mulOneClass [MulOneClass β] : MulOneClass α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulOneClass β
    ⊢ MulOneClass α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulOneClass β
    one : One α := e.one
    ⊢ MulOneClass α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulOneClass β
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ MulOneClass α
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  apply e.injective.mulOneClass _ <;> intros <;> exact e.apply_symm_apply _
                                                 /-
                                                   🎉 no goals
                                                 -/


@[to_additive]
noncomputable instance [Small.{v} α] [MulOneClass α] : MulOneClass (Shrink.{v} α) :=
  (equivShrink α).symm.mulOneClass


/-- Transfer `MulZeroOneClass` across an `Equiv` -/
protected abbrev mulZeroOneClass [MulZeroOneClass β] : MulZeroOneClass α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroOneClass β
    ⊢ MulZeroOneClass α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroOneClass β
    zero : Zero α := e.zero
    ⊢ MulZeroOneClass α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroOneClass β
    zero : Zero α := e.zero
    one : One α := e.one
    ⊢ MulZeroOneClass α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : MulZeroOneClass β
    zero : Zero α := e.zero
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ MulZeroOneClass α
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  apply e.injective.mulZeroOneClass _ <;> intros <;> exact e.apply_symm_apply _
                                                     /-
                                                       🎉 no goals
                                                     -/


noncomputable instance [Small.{v} α] [MulZeroOneClass α] : MulZeroOneClass (Shrink.{v} α) :=
  (equivShrink α).symm.mulZeroOneClass


/-- Transfer `Monoid` across an `Equiv` -/
@[to_additive "Transfer `AddMonoid` across an `Equiv`"]
protected abbrev monoid [Monoid β] : Monoid α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Monoid β
    ⊢ Monoid α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Monoid β
    one : One α := e.one
    ⊢ Monoid α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Monoid β
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ Monoid α
  -/
  let pow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Monoid β
    one : One α := e.one
    mul : Mul α := e.mul
    pow : Pow α Nat := e.pow Nat
    ⊢ Monoid α
  -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  apply e.injective.monoid _ <;> intros <;> exact e.apply_symm_apply _
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive]
noncomputable instance [Small.{v} α] [Monoid α] : Monoid (Shrink.{v} α) :=
  (equivShrink α).symm.monoid


/-- Transfer `CommMonoid` across an `Equiv` -/
@[to_additive "Transfer `AddCommMonoid` across an `Equiv`"]
protected abbrev commMonoid [CommMonoid β] : CommMonoid α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommMonoid β
    ⊢ CommMonoid α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommMonoid β
    one : One α := e.one
    ⊢ CommMonoid α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommMonoid β
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ CommMonoid α
  -/
  let pow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommMonoid β
    one : One α := e.one
    mul : Mul α := e.mul
    pow : Pow α Nat := e.pow Nat
    ⊢ CommMonoid α
  -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  apply e.injective.commMonoid _ <;> intros <;> exact e.apply_symm_apply _
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
noncomputable instance [Small.{v} α] [CommMonoid α] : CommMonoid (Shrink.{v} α) :=
  (equivShrink α).symm.commMonoid


/-- Transfer `Group` across an `Equiv` -/
@[to_additive "Transfer `AddGroup` across an `Equiv`"]
protected abbrev group [Group β] : Group α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    ⊢ Group α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    ⊢ Group α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ Group α
  -/
  let inv := e.Inv
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    ⊢ Group α
  -/
  let div := e.div
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    ⊢ Group α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    npow : Pow α Nat := e.pow Nat
    ⊢ Group α
  -/
  let zpow := e.pow ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Group β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    ⊢ Group α
  -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  apply e.injective.group _ <;> intros <;> exact e.apply_symm_apply _
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive]
noncomputable instance [Small.{v} α] [Group α] : Group (Shrink.{v} α) :=
  (equivShrink α).symm.group


/-- Transfer `CommGroup` across an `Equiv` -/
@[to_additive "Transfer `AddCommGroup` across an `Equiv`"]
protected abbrev commGroup [CommGroup β] : CommGroup α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    ⊢ CommGroup α
  -/
  let one := e.one
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    ⊢ CommGroup α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    mul : Mul α := e.mul
    ⊢ CommGroup α
  -/
  let inv := e.Inv
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    ⊢ CommGroup α
  -/
  let div := e.div
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    ⊢ CommGroup α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    npow : Pow α Nat := e.pow Nat
    ⊢ CommGroup α
  -/
  let zpow := e.pow ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommGroup β
    one : One α := e.one
    mul : Mul α := e.mul
    inv : Inv α := e.Inv
    div : Div α := e.div
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    ⊢ CommGroup α
  -/
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  apply e.injective.commGroup _ <;> intros <;> exact e.apply_symm_apply _
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
noncomputable instance [Small.{v} α] [CommGroup α] : CommGroup (Shrink.{v} α) :=
  (equivShrink α).symm.commGroup


/-- Transfer `NonUnitalNonAssocSemiring` across an `Equiv` -/
protected abbrev nonUnitalNonAssocSemiring [NonUnitalNonAssocSemiring β] :
    NonUnitalNonAssocSemiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocSemiring β
    ⊢ NonUnitalNonAssocSemiring α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocSemiring β
    zero : Zero α := e.zero
    ⊢ NonUnitalNonAssocSemiring α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalNonAssocSemiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalNonAssocSemiring α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalNonAssocSemiring α
  -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  apply e.injective.nonUnitalNonAssocSemiring _ <;> intros <;> exact e.apply_symm_apply _
                                                               /-
                                                                 🎉 no goals
                                                               -/


noncomputable instance [Small.{v} α] [NonUnitalNonAssocSemiring α] :
    NonUnitalNonAssocSemiring (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalNonAssocSemiring


/-- Transfer `NonUnitalSemiring` across an `Equiv` -/
protected abbrev nonUnitalSemiring [NonUnitalSemiring β] : NonUnitalSemiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalSemiring β
    ⊢ NonUnitalSemiring α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalSemiring β
    zero : Zero α := e.zero
    ⊢ NonUnitalSemiring α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalSemiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalSemiring α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalSemiring α
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  apply e.injective.nonUnitalSemiring _ <;> intros <;> exact e.apply_symm_apply _
                                                       /-
                                                         🎉 no goals
                                                       -/


noncomputable instance [Small.{v} α] [NonUnitalSemiring α] : NonUnitalSemiring (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalSemiring


/-- Transfer `AddMonoidWithOne` across an `Equiv` -/
protected abbrev addMonoidWithOne [AddMonoidWithOne β] : AddMonoidWithOne α :=
  { e.addMonoid, e.one with
    natCast := fun n => e.symm n
                                    /-
                                      α : Type u
                                      β : Type v
                                      e : Equiv α β
                                      inst✝ : AddMonoidWithOne β
                                      ⊢ Eq (e (NatCast.natCast 0)) (e 0)
                                    -/
    natCast_zero := e.injective (by simp [zero_def])
                                    /-
                                      🎉 no goals
                                    -/
                                             /-
                                               α : Type u
                                               β : Type v
                                               e : Equiv α β
                                               inst✝ : AddMonoidWithOne β
                                               n : Nat
                                               ⊢ Eq (e (NatCast.natCast (HAdd.hAdd n 1))) (e (HAdd.hAdd (NatCast.natCast n) 1))
                                             -/
    natCast_succ := fun n => e.injective (by simp [add_def, one_def]) }
                                             /-
                                               🎉 no goals
                                             -/


noncomputable instance [Small.{v} α] [AddMonoidWithOne α] : AddMonoidWithOne (Shrink.{v} α) :=
  (equivShrink α).symm.addMonoidWithOne


/-- Transfer `AddGroupWithOne` across an `Equiv` -/
protected abbrev addGroupWithOne [AddGroupWithOne β] : AddGroupWithOne α :=
  { e.addMonoidWithOne,
    e.addGroup with
    intCast := fun n => e.symm n
                                 /-
                                   α : Type u
                                   β : Type v
                                   e : Equiv α β
                                   inst✝ : AddGroupWithOne β
                                   n : Nat
                                   ⊢ Eq (IntCast.intCast ↑n) ↑n
                                 -/
    intCast_ofNat := fun n => by simp only [Int.cast_natCast]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/
    intCast_negSucc := fun _ =>
      congr_arg e.symm <| (Int.cast_negSucc _).trans <| congr_arg _ (e.apply_symm_apply _).symm }


noncomputable instance [Small.{v} α] [AddGroupWithOne α] : AddGroupWithOne (Shrink.{v} α) :=
  (equivShrink α).symm.addGroupWithOne


/-- Transfer `NonAssocSemiring` across an `Equiv` -/
protected abbrev nonAssocSemiring [NonAssocSemiring β] : NonAssocSemiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocSemiring β
    ⊢ NonAssocSemiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocSemiring β
    mul : Mul α := e.mul
    ⊢ NonAssocSemiring α
  -/
  let add_monoid_with_one := e.addMonoidWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocSemiring β
    mul : Mul α := e.mul
    add_monoid_with_one : AddMonoidWithOne α := e.addMonoidWithOne
    ⊢ NonAssocSemiring α
  -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  apply e.injective.nonAssocSemiring _ <;> intros <;> exact e.apply_symm_apply _
                                                      /-
                                                        🎉 no goals
                                                      -/


noncomputable instance [Small.{v} α] [NonAssocSemiring α] : NonAssocSemiring (Shrink.{v} α) :=
  (equivShrink α).symm.nonAssocSemiring


/-- Transfer `Semiring` across an `Equiv` -/
protected abbrev semiring [Semiring β] : Semiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semiring β
    ⊢ Semiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semiring β
    mul : Mul α := e.mul
    ⊢ Semiring α
  -/
  let add_monoid_with_one := e.addMonoidWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semiring β
    mul : Mul α := e.mul
    add_monoid_with_one : AddMonoidWithOne α := e.addMonoidWithOne
    ⊢ Semiring α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Semiring β
    mul : Mul α := e.mul
    add_monoid_with_one : AddMonoidWithOne α := e.addMonoidWithOne
    npow : Pow α Nat := e.pow Nat
    ⊢ Semiring α
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  apply e.injective.semiring _ <;> intros <;> exact e.apply_symm_apply _
                                              /-
                                                🎉 no goals
                                              -/


noncomputable instance [Small.{v} α] [Semiring α] : Semiring (Shrink.{v} α) :=
  (equivShrink α).symm.semiring


/-- Transfer `NonUnitalCommSemiring` across an `Equiv` -/
protected abbrev nonUnitalCommSemiring [NonUnitalCommSemiring β] : NonUnitalCommSemiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommSemiring β
    ⊢ NonUnitalCommSemiring α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommSemiring β
    zero : Zero α := e.zero
    ⊢ NonUnitalCommSemiring α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalCommSemiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalCommSemiring α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommSemiring β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalCommSemiring α
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  apply e.injective.nonUnitalCommSemiring _ <;> intros <;> exact e.apply_symm_apply _
                                                           /-
                                                             🎉 no goals
                                                           -/


noncomputable instance [Small.{v} α] [NonUnitalCommSemiring α] :
    NonUnitalCommSemiring (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalCommSemiring


/-- Transfer `CommSemiring` across an `Equiv` -/
protected abbrev commSemiring [CommSemiring β] : CommSemiring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemiring β
    ⊢ CommSemiring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemiring β
    mul : Mul α := e.mul
    ⊢ CommSemiring α
  -/
  let add_monoid_with_one := e.addMonoidWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemiring β
    mul : Mul α := e.mul
    add_monoid_with_one : AddMonoidWithOne α := e.addMonoidWithOne
    ⊢ CommSemiring α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommSemiring β
    mul : Mul α := e.mul
    add_monoid_with_one : AddMonoidWithOne α := e.addMonoidWithOne
    npow : Pow α Nat := e.pow Nat
    ⊢ CommSemiring α
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  apply e.injective.commSemiring _ <;> intros <;> exact e.apply_symm_apply _
                                                  /-
                                                    🎉 no goals
                                                  -/


noncomputable instance [Small.{v} α] [CommSemiring α] : CommSemiring (Shrink.{v} α) :=
  (equivShrink α).symm.commSemiring


/-- Transfer `NonUnitalNonAssocRing` across an `Equiv` -/
protected abbrev nonUnitalNonAssocRing [NonUnitalNonAssocRing β] : NonUnitalNonAssocRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    ⊢ NonUnitalNonAssocRing α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    ⊢ NonUnitalNonAssocRing α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalNonAssocRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalNonAssocRing α
  -/
  let neg := e.Neg
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    ⊢ NonUnitalNonAssocRing α
  -/
  let sub := e.sub
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    ⊢ NonUnitalNonAssocRing α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalNonAssocRing α
  -/
  let zsmul := e.smul ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalNonAssocRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    zsmul : SMul Int α := e.smul Int
    ⊢ NonUnitalNonAssocRing α
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  apply e.injective.nonUnitalNonAssocRing _ <;> intros <;> exact e.apply_symm_apply _
                                                           /-
                                                             🎉 no goals
                                                           -/


noncomputable instance [Small.{v} α] [NonUnitalNonAssocRing α] :
    NonUnitalNonAssocRing (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalNonAssocRing


/-- Transfer `NonUnitalRing` across an `Equiv` -/
protected abbrev nonUnitalRing [NonUnitalRing β] : NonUnitalRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    ⊢ NonUnitalRing α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    ⊢ NonUnitalRing α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalRing α
  -/
  let neg := e.Neg
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    ⊢ NonUnitalRing α
  -/
  let sub := e.sub
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    ⊢ NonUnitalRing α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalRing α
  -/
  let zsmul := e.smul ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    zsmul : SMul Int α := e.smul Int
    ⊢ NonUnitalRing α
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  apply e.injective.nonUnitalRing _ <;> intros <;> exact e.apply_symm_apply _
                                                   /-
                                                     🎉 no goals
                                                   -/


noncomputable instance [Small.{v} α] [NonUnitalRing α] : NonUnitalRing (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalRing


/-- Transfer `NonAssocRing` across an `Equiv` -/
protected abbrev nonAssocRing [NonAssocRing β] : NonAssocRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocRing β
    ⊢ NonAssocRing α
  -/
  let add_group_with_one := e.addGroupWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    ⊢ NonAssocRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonAssocRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    mul : Mul α := e.mul
    ⊢ NonAssocRing α
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  apply e.injective.nonAssocRing _ <;> intros <;> exact e.apply_symm_apply _
                                                  /-
                                                    🎉 no goals
                                                  -/


noncomputable instance [Small.{v} α] [NonAssocRing α] : NonAssocRing (Shrink.{v} α) :=
  (equivShrink α).symm.nonAssocRing


/-- Transfer `Ring` across an `Equiv` -/
protected abbrev ring [Ring β] : Ring α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Ring β
    ⊢ Ring α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Ring β
    mul : Mul α := e.mul
    ⊢ Ring α
  -/
  let add_group_with_one := e.addGroupWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Ring β
    mul : Mul α := e.mul
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    ⊢ Ring α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Ring β
    mul : Mul α := e.mul
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    npow : Pow α Nat := e.pow Nat
    ⊢ Ring α
  -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  apply e.injective.ring _ <;> intros <;> exact e.apply_symm_apply _
                                          /-
                                            🎉 no goals
                                          -/


noncomputable instance [Small.{v} α] [Ring α] : Ring (Shrink.{v} α) :=
  (equivShrink α).symm.ring


/-- Transfer `NonUnitalCommRing` across an `Equiv` -/
protected abbrev nonUnitalCommRing [NonUnitalCommRing β] : NonUnitalCommRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    ⊢ NonUnitalCommRing α
  -/
  let zero := e.zero
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    ⊢ NonUnitalCommRing α
  -/
  let add := e.add
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    ⊢ NonUnitalCommRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    ⊢ NonUnitalCommRing α
  -/
  let neg := e.Neg
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    ⊢ NonUnitalCommRing α
  -/
  let sub := e.sub
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    ⊢ NonUnitalCommRing α
  -/
  let nsmul := e.smul ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    ⊢ NonUnitalCommRing α
  -/
  let zsmul := e.smul ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : NonUnitalCommRing β
    zero : Zero α := e.zero
    add : Add α := e.add
    mul : Mul α := e.mul
    neg : Neg α := e.Neg
    sub : Sub α := e.sub
    nsmul : SMul Nat α := e.smul Nat
    zsmul : SMul Int α := e.smul Int
    ⊢ NonUnitalCommRing α
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  apply e.injective.nonUnitalCommRing _ <;> intros <;> exact e.apply_symm_apply _
                                                       /-
                                                         🎉 no goals
                                                       -/


noncomputable instance [Small.{v} α] [NonUnitalCommRing α] : NonUnitalCommRing (Shrink.{v} α) :=
  (equivShrink α).symm.nonUnitalCommRing


/-- Transfer `CommRing` across an `Equiv` -/
protected abbrev commRing [CommRing β] : CommRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommRing β
    ⊢ CommRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommRing β
    mul : Mul α := e.mul
    ⊢ CommRing α
  -/
  let add_group_with_one := e.addGroupWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommRing β
    mul : Mul α := e.mul
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    ⊢ CommRing α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : CommRing β
    mul : Mul α := e.mul
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    npow : Pow α Nat := e.pow Nat
    ⊢ CommRing α
  -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  apply e.injective.commRing _ <;> intros <;> exact e.apply_symm_apply _
                                              /-
                                                🎉 no goals
                                              -/


noncomputable instance [Small.{v} α] [CommRing α] : CommRing (Shrink.{v} α) :=
  (equivShrink α).symm.commRing


include e in
/-- Transfer `Nontrivial` across an `Equiv` -/
protected theorem nontrivial [Nontrivial β] : Nontrivial α :=
  e.surjective.nontrivial


noncomputable instance [Small.{v} α] [Nontrivial α] : Nontrivial (Shrink.{v} α) :=
  (equivShrink α).symm.nontrivial


/-- Transfer `IsDomain` across an `Equiv` -/
protected theorem isDomain [Semiring α] [Semiring β] [IsDomain β] (e : α ≃+* β) : IsDomain α :=
  Function.Injective.isDomain e.toRingHom e.injective


noncomputable instance [Small.{v} α] [Semiring α] [IsDomain α] : IsDomain (Shrink.{v} α) :=
  Equiv.isDomain (Shrink.ringEquiv α)


/-- Transfer `NNRatCast` across an `Equiv` -/
protected abbrev nnratCast [NNRatCast β] : NNRatCast α where nnratCast q := e.symm q


/-- Transfer `RatCast` across an `Equiv` -/
protected abbrev ratCast [RatCast β] : RatCast α where ratCast n := e.symm n


noncomputable instance _root_.Shrink.instNNRatCast [Small.{v} α] [NNRatCast α] :
    NNRatCast (Shrink.{v} α) := (equivShrink α).symm.nnratCast


noncomputable instance _root_.Shrink.instRatCast [Small.{v} α] [RatCast α] :
    RatCast (Shrink.{v} α) := (equivShrink α).symm.ratCast


/-- Transfer `DivisionRing` across an `Equiv` -/
protected abbrev divisionRing [DivisionRing β] : DivisionRing α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    ⊢ DivisionRing α
  -/
  let add_group_with_one := e.addGroupWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    ⊢ DivisionRing α
  -/
  let inv := e.Inv
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    ⊢ DivisionRing α
  -/
  let div := e.div
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    ⊢ DivisionRing α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    ⊢ DivisionRing α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    ⊢ DivisionRing α
  -/
  let zpow := e.pow ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    ⊢ DivisionRing α
  -/
  let nnratCast := e.nnratCast
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ⊢ DivisionRing α
  -/
  let ratCast := e.ratCast
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    ⊢ DivisionRing α
  -/
  let nnqsmul := e.smul ℚ≥0
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    nnqsmul : SMul NNRat α := e.smul NNRat
    ⊢ DivisionRing α
  -/
  let qsmul := e.smul ℚ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : DivisionRing β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    nnqsmul : SMul NNRat α := e.smul NNRat
    qsmul : SMul Rat α := e.smul Rat
    ⊢ DivisionRing α
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  apply e.injective.divisionRing _ <;> intros <;> exact e.apply_symm_apply _
                                                  /-
                                                    🎉 no goals
                                                  -/


noncomputable instance [Small.{v} α] [DivisionRing α] : DivisionRing (Shrink.{v} α) :=
  (equivShrink α).symm.divisionRing


/-- Transfer `Field` across an `Equiv` -/
protected abbrev field [Field β] : Field α := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    ⊢ Field α
  -/
  let add_group_with_one := e.addGroupWithOne
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    ⊢ Field α
  -/
  let neg := e.Neg
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    ⊢ Field α
  -/
  let inv := e.Inv
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    ⊢ Field α
  -/
  let div := e.div
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    ⊢ Field α
  -/
  let mul := e.mul
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    ⊢ Field α
  -/
  let npow := e.pow ℕ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    ⊢ Field α
  -/
  let zpow := e.pow ℤ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    ⊢ Field α
  -/
  let nnratCast := e.nnratCast
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ⊢ Field α
  -/
  let ratCast := e.ratCast
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    ⊢ Field α
  -/
  let nnqsmul := e.smul ℚ≥0
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    nnqsmul : SMul NNRat α := e.smul NNRat
    ⊢ Field α
  -/
  let qsmul := e.smul ℚ
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    inst✝ : Field β
    add_group_with_one : AddGroupWithOne α := e.addGroupWithOne
    neg : Neg α := e.Neg
    inv : Inv α := e.Inv
    div : Div α := e.div
    mul : Mul α := e.mul
    npow : Pow α Nat := e.pow Nat
    zpow : Pow α Int := e.pow Int
    nnratCast : NNRatCast α := e.nnratCast
    ratCast : RatCast α := e.ratCast
    nnqsmul : SMul NNRat α := e.smul NNRat
    qsmul : SMul Rat α := e.smul Rat
    ⊢ Field α
  -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  apply e.injective.field _ <;> intros <;> exact e.apply_symm_apply _
                                           /-
                                             🎉 no goals
                                           -/


noncomputable instance [Small.{v} α] [Field α] : Field (Shrink.{v} α) :=
  (equivShrink α).symm.field


/-- Transfer `MulAction` across an `Equiv` -/
protected abbrev mulAction (e : α ≃ β) [MulAction R β] : MulAction R α :=
  { e.smul R with
                   /-
                     α : Type u
                     β : Type v
                     e✝ : Equiv α β
                     R : Type u_1
                     inst✝¹ : Monoid R
                     e : Equiv α β
                     inst✝ : MulAction R β
                     ⊢ ∀ (b : α), Eq (HSMul.hSMul 1 b) b
                   -/
    one_smul := by simp [smul_def]
                   /-
                     🎉 no goals
                   -/
                   /-
                     α : Type u
                     β : Type v
                     e✝ : Equiv α β
                     R : Type u_1
                     inst✝¹ : Monoid R
                     e : Equiv α β
                     inst✝ : MulAction R β
                     ⊢ ∀ (x y : R) (b : α), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul x (HSMu …
                   -/
    mul_smul := by simp [smul_def, mul_smul] }
                   /-
                     🎉 no goals
                   -/


noncomputable instance [Small.{v} α] [MulAction R α] : MulAction R (Shrink.{v} α) :=
  (equivShrink α).symm.mulAction R


/-- Transfer `DistribMulAction` across an `Equiv` -/
protected abbrev distribMulAction (e : α ≃ β) [AddCommMonoid β] :
    letI := Equiv.addCommMonoid e
    ∀ [DistribMulAction R β], DistribMulAction R α := by
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝¹ : Monoid R
    e : Equiv α β
    inst✝ : AddCommMonoid β
    ⊢ [inst : DistribMulAction R β] → DistribMulAction R α
  -/
  intros
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝² : Monoid R
    e : Equiv α β
    inst✝¹ : AddCommMonoid β
    inst✝ : DistribMulAction R β
    ⊢ DistribMulAction R α
  -/
  letI := Equiv.addCommMonoid e
  exact
    ({ Equiv.mulAction R e with
        smul_zero := by simp [zero_def, smul_def]
        smul_add := by simp [add_def, smul_def, smul_add] } :
      DistribMulAction R α)


noncomputable instance [Small.{v} α] [AddCommMonoid α] [DistribMulAction R α] :
    DistribMulAction R (Shrink.{v} α) :=
  (equivShrink α).symm.distribMulAction R


/-- Transfer `Module` across an `Equiv` -/
protected abbrev module (e : α ≃ β) [AddCommMonoid β] :
    let _ := Equiv.addCommMonoid e
    ∀ [Module R β], Module R α := by
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝¹ : Semiring R
    e : Equiv α β
    inst✝ : AddCommMonoid β
    ⊢ let x := e.addCommMonoid;
      [inst : Module R β] → Module R α
  -/
  intros
  exact
    ({ Equiv.distribMulAction R e with
        zero_smul := by simp [smul_def, zero_smul, zero_def]
        add_smul := by simp [add_def, smul_def, add_smul] } :
      Module R α)


noncomputable instance [Small.{v} α] [AddCommMonoid α] [Module R α] : Module R (Shrink.{v} α) :=
  (equivShrink α).symm.module R


/-- An equivalence `e : α ≃ β` gives a linear equivalence `α ≃ₗ[R] β`
where the `R`-module structure on `α` is
the one obtained by transporting an `R`-module structure on `β` back along `e`.
-/
def linearEquiv (e : α ≃ β) [AddCommMonoid β] [Module R β] : by
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : Semiring R
      e : Equiv α β
      inst✝¹ : AddCommMonoid β
      inst✝ : Module R β
      ⊢ Sort ?u.91326
    -/
    let addCommMonoid := Equiv.addCommMonoid e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : Semiring R
      e : Equiv α β
      inst✝¹ : AddCommMonoid β
      inst✝ : Module R β
      addCommMonoid : AddCommMonoid α := e.addCommMonoid
      ⊢ Sort ?u.91326
    -/
    let module := Equiv.module R e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : Semiring R
      e : Equiv α β
      inst✝¹ : AddCommMonoid β
      inst✝ : Module R β
      addCommMonoid : AddCommMonoid α := e.addCommMonoid
      module : Module R α := Equiv.module R e
      ⊢ Sort ?u.91326
    -/
    exact α ≃ₗ[R] β := by
    /-
      🎉 no goals
    -/
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝² : Semiring R
    e : Equiv α β
    inst✝¹ : AddCommMonoid β
    inst✝ : Module R β
    ⊢ let addCommMonoid := e.addCommMonoid;
      let module := Equiv.module R e;
      LinearEquiv (RingHom.id R) α β
  -/
  intros
  exact
    { Equiv.addEquiv e with
      map_smul' := fun r x => by
        apply e.symm.injective
        simp only [toFun_as_coe, RingHom.id_apply, EmbeddingLike.apply_eq_iff_eq]
        exact Iff.mpr (apply_eq_iff_eq_symm_apply _) rfl }


variable (α) in
/-- Shrink `α` to a smaller universe preserves module structure. -/
@[simps!]
noncomputable def _root_.Shrink.linearEquiv [Small.{v} α] [AddCommMonoid α] [Module R α] :
    Shrink.{v} α ≃ₗ[R] α :=
  Equiv.linearEquiv _ (equivShrink α).symm


/-- Transfer `Algebra` across an `Equiv` -/
protected abbrev algebra (e : α ≃ β) [Semiring β] :
    let _ := Equiv.semiring e
    ∀ [Algebra R β], Algebra R α := by
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝¹ : CommSemiring R
    e : Equiv α β
    inst✝ : Semiring β
    ⊢ let x := e.semiring;
      [inst : Algebra R β] → Algebra R α
  -/
  intros
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    x✝ : Semiring α := e.semiring
    inst✝ : Algebra R β
    ⊢ Algebra R α
  -/
  letI : Module R α := e.module R
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    x✝ : Semiring α := e.semiring
    inst✝ : Algebra R β
    this : Module R α := Equiv.module R e
    ⊢ Algebra R α
  -/
  fapply Algebra.ofModule
    /-
      case h₁
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      ⊢ ∀ (r : R) (x y : α), Eq (HMul.hMul (HSMul.hSMul r x) y) (HSMul.hSMul r (HMul …
    -/
  · intro r x y
    /-
      case h₁
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      r : R
      x y : α
      ⊢ Eq (HMul.hMul (HSMul.hSMul r x) y) (HSMul.hSMul r (HMul.hMul x y))
    -/
    show e.symm (e (e.symm (r • e x)) * e y) = e.symm (r • e.ringEquiv (x * y))
    /-
      case h₁
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      r : R
      x y : α
      ⊢ Eq (e.symm (HMul.hMul (e (e.symm (HSMul.hSMul r (e x)))) (e y))) (e.symm (HS …
    -/
    simp only [apply_symm_apply, Algebra.smul_mul_assoc, map_mul, ringEquiv_apply]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      ⊢ ∀ (r : R) (x y : α), Eq (HMul.hMul x (HSMul.hSMul r y)) (HSMul.hSMul r (HMul …
    -/
  · intro r x y
    /-
      case h₂
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      r : R
      x y : α
      ⊢ Eq (HMul.hMul x (HSMul.hSMul r y)) (HSMul.hSMul r (HMul.hMul x y))
    -/
    show e.symm (e x * e (e.symm (r • e y))) = e.symm (r • e (e.symm (e x * e y)))
    /-
      case h₂
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      x✝ : Semiring α := e.semiring
      inst✝ : Algebra R β
      this : Module R α := Equiv.module R e
      r : R
      x y : α
      ⊢ Eq (e.symm (HMul.hMul (e x) (e (e.symm (HSMul.hSMul r (e y)))))) (e.symm (HS …
    -/
    simp only [apply_symm_apply, Algebra.mul_smul_comm]
    /-
      🎉 no goals
    -/


lemma algebraMap_def (e : α ≃ β) [Semiring β] [Algebra R β] (r : R) :
    (@algebraMap R α _ (Equiv.semiring e) (Equiv.algebra R e)) r = e.symm ((algebraMap R β) r) := by
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    r : R
    ⊢ Eq ((algebraMap R α) r) (e.symm ((algebraMap R β) r))
  -/
  let _ := Equiv.semiring e
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    r : R
    x✝ : Semiring α := e.semiring
    ⊢ Eq ((algebraMap R α) r) (e.symm ((algebraMap R β) r))
  -/
  let _ := Equiv.algebra R e
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    r : R
    x✝¹ : Semiring α := e.semiring
    x✝ : Algebra R α := Equiv.algebra R e
    ⊢ Eq ((algebraMap R α) r) (e.symm ((algebraMap R β) r))
  -/
  simp only [Algebra.algebraMap_eq_smul_one]
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    r : R
    x✝¹ : Semiring α := e.semiring
    x✝ : Algebra R α := Equiv.algebra R e
    ⊢ Eq (HSMul.hSMul r 1) (e.symm (HSMul.hSMul r 1))
  -/
  show e.symm (r • e 1) = e.symm (r • 1)
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    r : R
    x✝¹ : Semiring α := e.semiring
    x✝ : Algebra R α := Equiv.algebra R e
    ⊢ Eq (e.symm (HSMul.hSMul r (e 1))) (e.symm (HSMul.hSMul r 1))
  -/
  simp only [Equiv.one_def, apply_symm_apply]
  /-
    🎉 no goals
  -/


noncomputable instance [Small.{v} α] [Semiring α] [Algebra R α] :
    Algebra R (Shrink.{v} α) :=
  (equivShrink α).symm.algebra _


/-- An equivalence `e : α ≃ β` gives an algebra equivalence `α ≃ₐ[R] β`
where the `R`-algebra structure on `α` is
the one obtained by transporting an `R`-algebra structure on `β` back along `e`.
-/
def algEquiv (e : α ≃ β) [Semiring β] [Algebra R β] : by
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      ⊢ Sort ?u.98425
    -/
    let semiring := Equiv.semiring e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      semiring : Semiring α := e.semiring
      ⊢ Sort ?u.98425
    -/
    let algebra := Equiv.algebra R e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      semiring : Semiring α := e.semiring
      algebra : Algebra R α := Equiv.algebra R e
      ⊢ Sort ?u.98425
    -/
    exact α ≃ₐ[R] β := by
    /-
      🎉 no goals
    -/
  /-
    α : Type u
    β : Type v
    e✝ : Equiv α β
    R : Type u_1
    inst✝² : CommSemiring R
    e : Equiv α β
    inst✝¹ : Semiring β
    inst✝ : Algebra R β
    ⊢ let semiring := e.semiring;
      let algebra := Equiv.algebra R e;
      AlgEquiv R α β
  -/
  intros
  exact
    { Equiv.ringEquiv e with
      commutes' := fun r => by
        apply e.symm.injective
        simp only [RingEquiv.toEquiv_eq_coe, toFun_as_coe, EquivLike.coe_coe, ringEquiv_apply,
          symm_apply_apply, algebraMap_def] }


@[simp]
theorem algEquiv_apply (e : α ≃ β) [Semiring β] [Algebra R β] (a : α) : (algEquiv R e) a = e a :=
  rfl


theorem algEquiv_symm_apply (e : α ≃ β) [Semiring β] [Algebra R β] (b : β) : by
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      b : β
      ⊢ Sort ?u.99993
    -/
    letI := Equiv.semiring e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      b : β
      this : Semiring α := e.semiring
      ⊢ Sort ?u.99993
    -/
    letI := Equiv.algebra R e
    /-
      α : Type u
      β : Type v
      e✝ : Equiv α β
      R : Type u_1
      inst✝² : CommSemiring R
      e : Equiv α β
      inst✝¹ : Semiring β
      inst✝ : Algebra R β
      b : β
      this✝ : Semiring α := e.semiring
      this : Algebra R α := Equiv.algebra R e
      ⊢ Sort ?u.99993
    -/
    exact (algEquiv R e).symm b = e.symm b := rfl
    /-
      🎉 no goals
    -/


variable (α) in
/-- Shrink `α` to a smaller universe preserves algebra structure. -/
@[simps!]
noncomputable def _root_.Shrink.algEquiv [Small.{v} α] [Semiring α] [Algebra R α] :
    Shrink.{v} α ≃ₐ[R] α :=
  Equiv.algEquiv _ (equivShrink α).symm


/-- Any finite group in universe `u` is equivalent to some finite group in universe `v`. -/
lemma exists_type_univ_nonempty_mulEquiv (G : Type u) [Group G] [Finite G] :
    ∃ (G' : Type v) (_ : Group G') (_ : Fintype G'), Nonempty (G ≃* G') := by
  /-
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    ⊢ Exists fun G' => Exists fun x => Exists fun x_1 => Nonempty (MulEquiv G G')
  -/
  obtain ⟨n, ⟨e⟩⟩ := Finite.exists_equiv_fin G
  /-
    case intro.intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    n : Nat
    e : Equiv G (Fin n)
    ⊢ Exists fun G' => Exists fun x => Exists fun x_1 => Nonempty (MulEquiv G G')
  -/
  let f : Fin n ≃ ULift (Fin n) := Equiv.ulift.symm
  /-
    case intro.intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    n : Nat
    e : Equiv G (Fin n)
    f : Equiv (Fin n) (ULift.{?u.101337, 0} (Fin n)) := Equiv.ulift.symm
    ⊢ Exists fun G' => Exists fun x => Exists fun x_1 => Nonempty (MulEquiv G G')
  -/
  let e : G ≃ ULift (Fin n) := e.trans f
  /-
    case intro.intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    n : Nat
    e✝ : Equiv G (Fin n)
    f : Equiv (Fin n) (ULift.{?u.101337, 0} (Fin n)) := Equiv.ulift.symm
    e : Equiv G (ULift.{?u.101337, 0} (Fin n)) := e✝.trans f
    ⊢ Exists fun G' => Exists fun x => Exists fun x_1 => Nonempty (MulEquiv G G')
  -/
  letI groupH : Group (ULift (Fin n)) := e.symm.group
  /-
    case intro.intro
    G : Type u
    inst✝¹ : Group G
    inst✝ : Finite G
    n : Nat
    e✝ : Equiv G (Fin n)
    f : Equiv (Fin n) (ULift.{?u.101337, 0} (Fin n)) := Equiv.ulift.symm
    e : Equiv G (ULift.{?u.101337, 0} (Fin n)) := e✝.trans f
    groupH : Group (ULift.{?u.101337, 0} (Fin n)) := e.symm.group
    ⊢ Exists fun G' => Exists fun x => Exists fun x_1 => Nonempty (MulEquiv G G')
  -/
  exact ⟨ULift (Fin n), groupH, inferInstance, ⟨MulEquiv.symm <| e.symm.mulEquiv⟩⟩
  /-
    🎉 no goals
  -/


/-- Transport a module instance via an isomorphism of the underlying abelian groups.
This has better definitional properties than `Equiv.module` since here
the abelian group structure remains unmodified. -/
abbrev AddEquiv.module (e : α ≃+ β) :
    Module A α where
  toSMul := e.toEquiv.smul A
                 /-
                   α : Type u
                   β : Type v
                   R : Type u_1
                   inst✝⁵ : CommSemiring R
                   A : Type u_2
                   inst✝⁴ : Semiring A
                   inst✝³ : Algebra R A
                   inst✝² : AddCommMonoid α
                   inst✝¹ : AddCommMonoid β
                   inst✝ : Module A β
                   e : AddEquiv α β
                   ⊢ ∀ (b : α), Eq (HSMul.hSMul 1 b) b
                 -/
  one_smul := by simp [Equiv.smul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u
                   β : Type v
                   R : Type u_1
                   inst✝⁵ : CommSemiring R
                   A : Type u_2
                   inst✝⁴ : Semiring A
                   inst✝³ : Algebra R A
                   inst✝² : AddCommMonoid α
                   inst✝¹ : AddCommMonoid β
                   inst✝ : Module A β
                   e : AddEquiv α β
                   ⊢ ∀ (x y : A) (b : α), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul x (HSMu …
                 -/
  mul_smul := by simp [Equiv.smul_def, mul_smul]
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u
                    β : Type v
                    R : Type u_1
                    inst✝⁵ : CommSemiring R
                    A : Type u_2
                    inst✝⁴ : Semiring A
                    inst✝³ : Algebra R A
                    inst✝² : AddCommMonoid α
                    inst✝¹ : AddCommMonoid β
                    inst✝ : Module A β
                    e : AddEquiv α β
                    ⊢ ∀ (a : A), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_zero := by simp [Equiv.smul_def]
                  /-
                    🎉 no goals
                  -/
                 /-
                   α : Type u
                   β : Type v
                   R : Type u_1
                   inst✝⁵ : CommSemiring R
                   A : Type u_2
                   inst✝⁴ : Semiring A
                   inst✝³ : Algebra R A
                   inst✝² : AddCommMonoid α
                   inst✝¹ : AddCommMonoid β
                   inst✝ : Module A β
                   e : AddEquiv α β
                   ⊢ ∀ (a : A) (x y : α), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hS …
                 -/
  smul_add := by simp [Equiv.smul_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   α : Type u
                   β : Type v
                   R : Type u_1
                   inst✝⁵ : CommSemiring R
                   A : Type u_2
                   inst✝⁴ : Semiring A
                   inst✝³ : Algebra R A
                   inst✝² : AddCommMonoid α
                   inst✝¹ : AddCommMonoid β
                   inst✝ : Module A β
                   e : AddEquiv α β
                   ⊢ ∀ (r s : A) (x : α), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hS …
                 -/
  add_smul := by simp [Equiv.smul_def, add_smul]
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u
                    β : Type v
                    R : Type u_1
                    inst✝⁵ : CommSemiring R
                    A : Type u_2
                    inst✝⁴ : Semiring A
                    inst✝³ : Algebra R A
                    inst✝² : AddCommMonoid α
                    inst✝¹ : AddCommMonoid β
                    inst✝ : Module A β
                    e : AddEquiv α β
                    ⊢ ∀ (x : α), Eq (HSMul.hSMul 0 x) 0
                  -/
  zero_smul := by simp [Equiv.smul_def]
                  /-
                    🎉 no goals
                  -/


/-- The module instance from `AddEquiv.module` is compatible with the `R`-module structures,
if the `AddEquiv` is induced by an `R`-module isomorphism. -/
lemma LinearEquiv.isScalarTower [Module R α] [Module R β] [IsScalarTower R A β]
    (e : α ≃ₗ[R] β) :
    letI := e.toAddEquiv.module A
    IsScalarTower R A α := by
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝⁸ : CommSemiring R
    A : Type u_2
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : AddCommMonoid β
    inst✝³ : Module A β
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : IsScalarTower R A β
    e : LinearEquiv (RingHom.id R) α β
    ⊢ IsScalarTower R A α
  -/
  letI := e.toAddEquiv.module A
  /-
    α : Type u
    β : Type v
    R : Type u_1
    inst✝⁸ : CommSemiring R
    A : Type u_2
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : AddCommMonoid β
    inst✝³ : Module A β
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : IsScalarTower R A β
    e : LinearEquiv (RingHom.id R) α β
    this : Module A α := AddEquiv.module A e.toAddEquiv
    ⊢ IsScalarTower R A α
  -/
  constructor
  /-
    case smul_assoc
    α : Type u
    β : Type v
    R : Type u_1
    inst✝⁸ : CommSemiring R
    A : Type u_2
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : AddCommMonoid β
    inst✝³ : Module A β
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : IsScalarTower R A β
    e : LinearEquiv (RingHom.id R) α β
    this : Module A α := AddEquiv.module A e.toAddEquiv
    ⊢ ∀ (x : R) (y : A) (z : α), Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul …
  -/
  intro x y z
  /-
    case smul_assoc
    α : Type u
    β : Type v
    R : Type u_1
    inst✝⁸ : CommSemiring R
    A : Type u_2
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : AddCommMonoid β
    inst✝³ : Module A β
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : IsScalarTower R A β
    e : LinearEquiv (RingHom.id R) α β
    this : Module A α := AddEquiv.module A e.toAddEquiv
    x : R
    y : A
    z : α
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
  -/
  simp only [Equiv.smul_def, AddEquiv.toEquiv_eq_coe, smul_assoc]
  /-
    case smul_assoc
    α : Type u
    β : Type v
    R : Type u_1
    inst✝⁸ : CommSemiring R
    A : Type u_2
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid α
    inst✝⁴ : AddCommMonoid β
    inst✝³ : Module A β
    inst✝² : Module R α
    inst✝¹ : Module R β
    inst✝ : IsScalarTower R A β
    e : LinearEquiv (RingHom.id R) α β
    this : Module A α := AddEquiv.module A e.toAddEquiv
    x : R
    y : A
    z : α
    ⊢ Eq ({ toFun := (↑e).toFun, invFun := e.invFun, left_inv := ⋯, right_inv := ⋯ …
  -/
  apply e.symm.map_smul
  /-
    🎉 no goals
  -/


