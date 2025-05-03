/-- In a `GroupWithZero` `G₀`, the unit group `G₀ˣ` is equivalent to the subtype of nonzero
elements. -/
@[simps] def _root_.unitsEquivNeZero : G₀ˣ ≃ {a : G₀ // a ≠ 0} where
  toFun a := ⟨a, a.ne_zero⟩
  invFun a := Units.mk0 _ a.prop
  left_inv _ := Units.ext rfl
  right_inv _ := rfl


/-- Left multiplication by a nonzero element in a `GroupWithZero` is a permutation of the
underlying type. -/
@[simps! (config := .asFn)]
protected def mulLeft₀ (a : G₀) (ha : a ≠ 0) : Perm G₀ :=
  (Units.mk0 a ha).mulLeft


theorem _root_.mulLeft_bijective₀ (a : G₀) (ha : a ≠ 0) : Function.Bijective (a * · : G₀ → G₀) :=
  (Equiv.mulLeft₀ a ha).bijective


/-- Right multiplication by a nonzero element in a `GroupWithZero` is a permutation of the
underlying type. -/
@[simps! (config := .asFn)]
protected def mulRight₀ (a : G₀) (ha : a ≠ 0) : Perm G₀ :=
  (Units.mk0 a ha).mulRight


theorem _root_.mulRight_bijective₀ (a : G₀) (ha : a ≠ 0) : Function.Bijective ((· * a) : G₀ → G₀) :=
  (Equiv.mulRight₀ a ha).bijective


/-- Right division by a nonzero element in a `GroupWithZero` is a permutation of the
underlying type. -/
@[simps! (config := { simpRhs := true })]
def divRight₀ (a : G₀) (ha : a ≠ 0) : Perm G₀ where
  toFun := (· / a)
  invFun := (· * a)
                   /-
                     G₀ : Type u_1
                     inst✝ : GroupWithZero G₀
                     a : G₀
                     ha : Ne a 0
                     x✝ : G₀
                     ⊢ Eq ((fun x => HMul.hMul x a) ((fun x => HDiv.hDiv x a) x✝)) x✝
                   -/
  left_inv _ := by simp [ha]
                   /-
                     🎉 no goals
                   -/
                    /-
                      G₀ : Type u_1
                      inst✝ : GroupWithZero G₀
                      a : G₀
                      ha : Ne a 0
                      x✝ : G₀
                      ⊢ Eq ((fun x => HDiv.hDiv x a) ((fun x => HMul.hMul x a) x✝)) x✝
                    -/
  right_inv _ := by simp [ha]
                    /-
                      🎉 no goals
                    -/


/-- Left division by a nonzero element in a `CommGroupWithZero` is a permutation of the underlying
type. -/
@[simps! (config := { simpRhs := true })]
def divLeft₀ (a : G₀) (ha : a ≠ 0) : Perm G₀ where
  toFun := (a / ·)
  invFun := (a / ·)
                   /-
                     G₀ : Type u_1
                     inst✝ : CommGroupWithZero G₀
                     a : G₀
                     ha : Ne a 0
                     x✝ : G₀
                     ⊢ Eq ((fun x => HDiv.hDiv a x) ((fun x => HDiv.hDiv a x) x✝)) x✝
                   -/
  left_inv _ := by simp [ha]
                   /-
                     🎉 no goals
                   -/
                    /-
                      G₀ : Type u_1
                      inst✝ : CommGroupWithZero G₀
                      a : G₀
                      ha : Ne a 0
                      x✝ : G₀
                      ⊢ Eq ((fun x => HDiv.hDiv a x) ((fun x => HDiv.hDiv a x) x✝)) x✝
                    -/
  right_inv _ := by simp [ha]
                    /-
                      🎉 no goals
                    -/


