/-- The automorphism group of `ZMod n` is isomorphic to the group of units of `ZMod n`. -/
@[simps]
def AddAutEquivUnits : AddAut (ZMod n) ≃* (ZMod n)ˣ :=
  have h (f : AddAut (ZMod n)) (x : ZMod n) : f 1 * x = f x := by
    /-
      n : Nat
      f : AddAut (ZMod n)
      x : ZMod n
      ⊢ Eq (HMul.hMul (f 1) x) (f x)
    -/
    rw [mul_comm, ← x.intCast_zmod_cast, ← zsmul_eq_mul, ← map_zsmul, zsmul_one]
    /-
      🎉 no goals
    -/
  { toFun := fun f ↦ Units.mkOfMulEqOne (f 1) (f⁻¹ 1) ((h f _).trans (f.inv_apply_self _ _))
    invFun := AddAut.mulLeft
                           /-
                             n : Nat
                             h : ∀ (f : AddAut (ZMod n)) (x : ZMod n), Eq (HMul.hMul (f 1) x) (f x)
                             f : AddAut (ZMod n)
                             ⊢ Eq (AddAut.mulLeft ((fun f => Units.mkOfMulEqOne (f 1) ((Inv.inv f) 1) ⋯) f) …
                           -/
    left_inv := fun f ↦ by simp [DFunLike.ext_iff, Units.smul_def, h]
                           /-
                             🎉 no goals
                           -/
                            /-
                              n : Nat
                              h : ∀ (f : AddAut (ZMod n)) (x : ZMod n), Eq (HMul.hMul (f 1) x) (f x)
                              x : Units (ZMod n)
                              ⊢ Eq ((fun f => Units.mkOfMulEqOne (f 1) ((Inv.inv f) 1) ⋯) (AddAut.mulLeft x) …
                            -/
    right_inv := fun x ↦ by simp [Units.ext_iff, Units.smul_def]
                            /-
                              🎉 no goals
                            -/
                             /-
                               n : Nat
                               h : ∀ (f : AddAut (ZMod n)) (x : ZMod n), Eq (HMul.hMul (f 1) x) (f x)
                               f g : AddAut (ZMod n)
                               ⊢ Eq ({ toFun := fun f => Units.mkOfMulEqOne (f 1) ((Inv.inv f) 1) ⋯, invFun : …
                             -/
    map_mul' := fun f g ↦ by simp [Units.ext_iff, h] }
                             /-
                               🎉 no goals
                             -/


