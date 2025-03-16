/-- `Equiv.mulLeft₀` as an order isomorphism. -/
@[simps! (config := { simpRhs := true })]
def mulLeft₀ (a : G₀) (ha : 0 < a) : G₀ ≃o G₀ where
  toEquiv := .mulLeft₀ a ha.ne'
  map_rel_iff' := mul_le_mul_left ha


lemma mulLeft₀_symm (a : G₀) (ha : 0 < a) : (mulLeft₀ a ha).symm = mulLeft₀ a⁻¹ (inv_pos.2 ha) := by
  /-
    G₀ : Type u_1
    inst✝⁴ : GroupWithZero G₀
    inst✝³ : PartialOrder G₀
    inst✝² : PosMulMono G₀
    inst✝¹ : PosMulReflectLE G₀
    inst✝ : ZeroLEOneClass G₀
    a : G₀
    ha : LT.lt 0 a
    ⊢ Eq (OrderIso.mulLeft₀ a ha).symm (OrderIso.mulLeft₀ (Inv.inv a) ⋯)
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- `Equiv.mulRight₀` as an order isomorphism. -/
@[simps! (config := { simpRhs := true })]
def mulRight₀ (a : G₀) (ha : 0 < a) : G₀ ≃o G₀ where
  toEquiv := .mulRight₀ a ha.ne'
  map_rel_iff' := mul_le_mul_right ha


lemma mulRight₀_symm (a : G₀) (ha : 0 < a) :
                                                               /-
                                                                 G₀ : Type u_1
                                                                 inst✝⁵ : GroupWithZero G₀
                                                                 inst✝⁴ : PartialOrder G₀
                                                                 inst✝³ : MulPosMono G₀
                                                                 inst✝² : MulPosReflectLE G₀
                                                                 inst✝¹ : ZeroLEOneClass G₀
                                                                 inst✝ : PosMulReflectLT G₀
                                                                 a : G₀
                                                                 ha : LT.lt 0 a
                                                                 ⊢ Eq (OrderIso.mulRight₀ a ha).symm (OrderIso.mulRight₀ (Inv.inv a) ⋯)
                                                               -/
    (mulRight₀ a ha).symm = mulRight₀ a⁻¹ (inv_pos.2 ha) := by ext; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `Equiv.divRight₀` as an order isomorphism. -/
@[simps! (config := { simpRhs := true })]
def divRight₀ [ZeroLEOneClass G₀] [MulPosStrictMono G₀] [MulPosReflectLE G₀] [PosMulReflectLT G₀]
    (a : G₀) (ha : 0 < a) : G₀ ≃o G₀ where
  toEquiv := .divRight₀ a ha.ne'
  map_rel_iff' {b c} := by
    /-
      G₀ : Type ?u.4332
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : PartialOrder G₀
      inst✝³ : ZeroLEOneClass G₀
      inst✝² : MulPosStrictMono G₀
      inst✝¹ : MulPosReflectLE G₀
      inst✝ : PosMulReflectLT G₀
      a : G₀
      ha : LT.lt 0 a
      b c : G₀
      ⊢ Iff (LE.le ((Equiv.divRight₀ a ⋯) b) ((Equiv.divRight₀ a ⋯) c)) (LE.le b c)
    -/
    simp only [Equiv.divRight₀_apply, div_eq_mul_inv]
    /-
      G₀ : Type ?u.4332
      inst✝⁵ : GroupWithZero G₀
      inst✝⁴ : PartialOrder G₀
      inst✝³ : ZeroLEOneClass G₀
      inst✝² : MulPosStrictMono G₀
      inst✝¹ : MulPosReflectLE G₀
      inst✝ : PosMulReflectLT G₀
      a : G₀
      ha : LT.lt 0 a
      b c : G₀
      ⊢ Iff (LE.le (HMul.hMul b (Inv.inv a)) (HMul.hMul c (Inv.inv a))) (LE.le b c)
    -/
    exact mul_le_mul_right (a := a⁻¹) (inv_pos.mpr ha)
    /-
      🎉 no goals
    -/


