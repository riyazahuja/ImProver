/-- A predicate to express that a (semi)ring is a (semi)field.

This is mainly useful because such a predicate does not contain data,
and can therefore be easily transported along ring isomorphisms.
Additionally, this is useful when trying to prove that
a particular ring structure extends to a (semi)field. -/
structure IsField (R : Type u) [Semiring R] : Prop where
  /-- For a semiring to be a field, it must have two distinct elements. -/
  exists_pair_ne : ∃ x y : R, x ≠ y
  /-- Fields are commutative. -/
  mul_comm : ∀ x y : R, x * y = y * x
  /-- Nonzero elements have multiplicative inverses. -/
  mul_inv_cancel : ∀ {a : R}, a ≠ 0 → ∃ b, a * b = 1


/-- Transferring from `Semifield` to `IsField`. -/
theorem Semifield.toIsField (R : Type u) [Semifield R] : IsField R where
  __ := ‹Semifield R›
  mul_inv_cancel {a} ha := ⟨a⁻¹, mul_inv_cancel₀ ha⟩


/-- Transferring from `Field` to `IsField`. -/
theorem Field.toIsField (R : Type u) [Field R] : IsField R :=
  Semifield.toIsField _


@[simp]
theorem IsField.nontrivial {R : Type u} [Semiring R] (h : IsField R) : Nontrivial R :=
  ⟨h.exists_pair_ne⟩


@[simp]
theorem not_isField_of_subsingleton (R : Type u) [Semiring R] [Subsingleton R] : ¬IsField R :=
  fun h =>
  let ⟨_, _, h⟩ := h.exists_pair_ne
  h (Subsingleton.elim _ _)


open Classical in
/-- Transferring from `IsField` to `Semifield`. -/
noncomputable def IsField.toSemifield {R : Type u} [Semiring R] (h : IsField R) : Semifield R where
  __ := ‹Semiring R›
  __ := h
  inv a := if ha : a = 0 then 0 else Classical.choose (h.mul_inv_cancel ha)
  inv_zero := dif_pos rfl
                            /-
                              R : Type u
                              inst✝ : Semiring R
                              h : IsField R
                              a : R
                              ha : Ne a 0
                              ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
                            -/
  mul_inv_cancel a ha := by convert Classical.choose_spec (h.mul_inv_cancel ha); exact dif_neg ha
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  nnqsmul := _
  nnqsmul_def _ _ := rfl


/-- Transferring from `IsField` to `Field`. -/
noncomputable def IsField.toField {R : Type u} [Ring R] (h : IsField R) : Field R :=
  { ‹Ring R›, IsField.toSemifield h with
    qsmul := _
    qsmul_def := fun _ _ => rfl }


/-- For each field, and for each nonzero element of said field, there is a unique inverse.
Since `IsField` doesn't remember the data of an `inv` function and as such,
a lemma that there is a unique inverse could be useful.
-/
theorem uniq_inv_of_isField (R : Type u) [Ring R] (hf : IsField R) :
    ∀ x : R, x ≠ 0 → ∃! y : R, x * y = 1 := by
  /-
    R : Type u
    inst✝ : Ring R
    hf : IsField R
    ⊢ ∀ (x : R), Ne x 0 → ExistsUnique fun y => Eq (HMul.hMul x y) 1
  -/
  intro x hx
  /-
    R : Type u
    inst✝ : Ring R
    hf : IsField R
    x : R
    hx : Ne x 0
    ⊢ ExistsUnique fun y => Eq (HMul.hMul x y) 1
  -/
  apply existsUnique_of_exists_of_unique
    /-
      case hex
      R : Type u
      inst✝ : Ring R
      hf : IsField R
      x : R
      hx : Ne x 0
      ⊢ Exists fun x_1 => Eq (HMul.hMul x x_1) 1
    -/
  · exact hf.mul_inv_cancel hx
    /-
      🎉 no goals
    -/
    /-
      case hunique
      R : Type u
      inst✝ : Ring R
      hf : IsField R
      x : R
      hx : Ne x 0
      ⊢ ∀ (y₁ y₂ : R), Eq (HMul.hMul x y₁) 1 → Eq (HMul.hMul x y₂) 1 → Eq y₁ y₂
    -/
  · intro y z hxy hxz
    calc
      y = y * (x * z) := by rw [hxz, mul_one]
      _ = x * y * z := by rw [← mul_assoc, hf.mul_comm y x]
      _ = z := by rw [hxy, one_mul]


