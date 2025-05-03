/-- The canonical map from the additive to the multiplicative circle, as an `AddChar`. -/
noncomputable def AddCircle.toCircle_addChar {T : ℝ} : AddChar (AddCircle T) Circle where
  toFun := toCircle
  map_zero_eq_one' := toCircle_zero
  map_add_eq_mul' := toCircle_add


/-- The additive character from `ZMod N` to the unit circle in `ℂ`, sending `j mod N` to
`exp (2 * π * I * j / N)`. -/
noncomputable def toCircle : AddChar (ZMod N) Circle :=
  toCircle_addChar.compAddMonoidHom toAddCircle


lemma toCircle_intCast (j : ℤ) :
    toCircle (j : ZMod N) = exp (2 * π * I * j / N) := by
  rw [toCircle, AddChar.compAddMonoidHom_apply, toCircle_addChar, AddChar.coe_mk,
    AddCircle.toCircle, toAddCircle_intCast, Function.Periodic.lift_coe, Circle.coe_exp]
  /-
    N : Nat
    inst✝ : NeZero N
    j : Int
    ⊢ Eq (Complex.exp (HMul.hMul (↑(HMul.hMul (HDiv.hDiv (HMul.hMul 2 Real.pi) 1)  …
  -/
  push_cast
  /-
    N : Nat
    inst✝ : NeZero N
    j : Int
    ⊢ Eq (Complex.exp (HMul.hMul (HMul.hMul (HDiv.hDiv (HMul.hMul 2 ↑Real.pi) 1) ( …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma toCircle_natCast (j : ℕ) :
    toCircle (j : ZMod N) = exp (2 * π * I * j / N) := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : Nat
    ⊢ Eq (↑(ZMod.toCircle ↑j)) (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul …
  -/
  simpa using toCircle_intCast (N := N) j
  /-
    🎉 no goals
  -/


/--
Explicit formula for `toCircle j`. Note that this is "evil" because it uses `ZMod.val`. Where
possible, it is recommended to lift `j` to `ℤ` and use `toCircle_intCast` instead.
-/
lemma toCircle_apply (j : ZMod N) :
    toCircle j = exp (2 * π * I * j.val / N) := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    ⊢ Eq (↑(ZMod.toCircle j)) (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul. …
  -/
  rw [← toCircle_natCast, natCast_zmod_val]
  /-
    🎉 no goals
  -/


lemma injective_toCircle : Injective (toCircle : ZMod N → Circle) :=
  (AddCircle.injective_toCircle one_ne_zero).comp (toAddCircle_injective N)


/-- The additive character from `ZMod N` to `ℂ`, sending `j mod N` to `exp (2 * π * I * j / N)`. -/
noncomputable def stdAddChar : AddChar (ZMod N) ℂ := Circle.coeHom.compAddChar toCircle


lemma stdAddChar_coe (j : ℤ) :
                                                            /-
                                                              N : Nat
                                                              inst✝ : NeZero N
                                                              j : Int
                                                              ⊢ Eq (ZMod.stdAddChar ↑j) (Complex.exp (HDiv.hDiv (HMul.hMul (HMul.hMul (HMul. …
                                                            -/
    stdAddChar (j : ZMod N) = exp (2 * π * I * j / N) := by simp [stdAddChar, toCircle_intCast]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma stdAddChar_apply (j : ZMod N) : stdAddChar j = ↑(toCircle j) := rfl


lemma injective_stdAddChar : Injective (stdAddChar : AddChar (ZMod N) ℂ) :=
  Subtype.coe_injective.comp injective_toCircle


/-- The standard additive character `ZMod N → ℂ` is primitive. -/
lemma isPrimitive_stdAddChar (N : ℕ) [NeZero N] :
    (stdAddChar (N := N)).IsPrimitive := by
  /-
    N : Nat
    inst✝ : NeZero N
    ⊢ ZMod.stdAddChar.IsPrimitive
  -/
  refine AddChar.zmod_char_primitive_of_eq_one_only_at_zero _ _ (fun t ht ↦ ?_)
  /-
    N : Nat
    inst✝ : NeZero N
    t : ZMod N
    ht : Eq (ZMod.stdAddChar t) 1
    ⊢ Eq t 0
  -/
  rwa [← (stdAddChar (N := N)).map_zero_eq_one, injective_stdAddChar.eq_iff] at ht
  /-
    🎉 no goals
  -/


