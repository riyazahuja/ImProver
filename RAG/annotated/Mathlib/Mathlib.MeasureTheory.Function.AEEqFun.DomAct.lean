@[to_additive]
instance : SMul Mᵈᵐᵃ (α →ₘ[μ] β) where
  smul c f := f.compMeasurePreserving (mk.symm c • ·) (measurePreserving_smul _ _)


@[to_additive]
theorem smul_aeeqFun_aeeq (c : Mᵈᵐᵃ) (f : α →ₘ[μ] β) :
    c • f =ᵐ[μ] (f <| mk.symm c • ·) :=
  f.coeFn_compMeasurePreserving _


@[to_additive (attr := simp)]
theorem mk_smul_mk_aeeqFun (c : M) (f : α → β) (hf : AEStronglyMeasurable f μ) :
    mk c • AEEqFun.mk f hf = AEEqFun.mk (f <| c • ·)
      (hf.comp_measurePreserving (measurePreserving_smul _ _)) :=
  rfl


@[to_additive (attr := simp)]
theorem smul_aeeqFun_const (c : Mᵈᵐᵃ) (b : β) :
    c • (AEEqFun.const α b : α →ₘ[μ] β) = AEEqFun.const α b :=
  rfl


instance [SMul N β] [ContinuousConstSMul N β] : SMulCommClass Mᵈᵐᵃ N (α →ₘ[μ] β) where
                  /-
                    M : Type u_3
                    N : Type u_1
                    α : Type u_4
                    β : Type u_2
                    inst✝⁸ : MeasurableSpace M
                    inst✝⁷ : MeasurableSpace N
                    inst✝⁶ : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    inst✝⁵ : TopologicalSpace β
                    inst✝⁴ : SMul M α
                    inst✝³ : MeasurableSMul M α
                    inst✝² : MeasureTheory.SMulInvariantMeasure M α μ
                    inst✝¹ : SMul N β
                    inst✝ : ContinuousConstSMul N β
                    ⊢ ∀ (m : DomMulAct M) (n : N) (a : MeasureTheory.AEEqFun α β μ), Eq (HSMul.hSM …
                  -/
  smul_comm := by rintro _ _ ⟨_⟩; rfl
                                  /-
                                    🎉 no goals
                                  -/


instance [SMul N β] [ContinuousConstSMul N β] : SMulCommClass N Mᵈᵐᵃ (α →ₘ[μ] β) :=
  .symm _ _ _


@[to_additive]
instance [SMul N α] [MeasurableSMul N α] [SMulInvariantMeasure N α μ] [SMulCommClass M N α] :
    SMulCommClass Mᵈᵐᵃ Nᵈᵐᵃ (α →ₘ[μ] β) where
  smul_comm := mk.surjective.forall.2 fun c₁ ↦ mk.surjective.forall.2 fun c₂ ↦
    (AEEqFun.induction_on · fun f hf ↦ by simp only [mk_smul_mk_aeeqFun, smul_comm])


instance [Zero β] : SMulZeroClass Mᵈᵐᵃ (α →ₘ[μ] β) where
  smul_zero _ := rfl

-- TODO: add `AEEqFun.addZeroClass`

instance [AddMonoid β] [ContinuousAdd β] : DistribSMul Mᵈᵐᵃ (α →ₘ[μ] β) where
                 /-
                   M : Type ?u.9044
                   N : Type ?u.9047
                   α : Type ?u.9050
                   β : Type ?u.9060
                   inst✝⁸ : MeasurableSpace M
                   inst✝⁷ : MeasurableSpace N
                   inst✝⁶ : MeasurableSpace α
                   μ : MeasureTheory.Measure α
                   inst✝⁵ : TopologicalSpace β
                   inst✝⁴ : SMul M α
                   inst✝³ : MeasurableSMul M α
                   inst✝² : MeasureTheory.SMulInvariantMeasure M α μ
                   inst✝¹ : AddMonoid β
                   inst✝ : ContinuousAdd β
                   ⊢ ∀ (a : DomMulAct M) (x y : MeasureTheory.AEEqFun α β μ), Eq (HSMul.hSMul a ( …
                 -/
  smul_add := by rintro _ ⟨⟩ ⟨⟩; rfl
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive]
instance : MulAction Mᵈᵐᵃ (α →ₘ[μ] β) where
  one_smul := (AEEqFun.induction_on · fun _ _ ↦ by
    simp only [← mk_one, mk_smul_mk_aeeqFun, one_smul])
  mul_smul := mk.surjective.forall.2 fun _ ↦ mk.surjective.forall.2 fun _ ↦
    (AEEqFun.induction_on · fun _ _ ↦ by simp only [← mk_mul, mk_smul_mk_aeeqFun, mul_smul])


instance [Monoid β] [ContinuousMul β] : MulDistribMulAction Mᵈᵐᵃ (α →ₘ[μ] β) where
  smul_one _ := rfl
                 /-
                   M : Type ?u.21654
                   N : Type ?u.21657
                   α : Type ?u.21660
                   β : Type ?u.21670
                   inst✝⁹ : MeasurableSpace M
                   inst✝⁸ : MeasurableSpace N
                   inst✝⁷ : MeasurableSpace α
                   μ : MeasureTheory.Measure α
                   inst✝⁶ : TopologicalSpace β
                   inst✝⁵ : Monoid M
                   inst✝⁴ : MulAction M α
                   inst✝³ : MeasurableSMul M α
                   inst✝² : MeasureTheory.SMulInvariantMeasure M α μ
                   inst✝¹ : Monoid β
                   inst✝ : ContinuousMul β
                   ⊢ ∀ (r : DomMulAct M) (x y : MeasureTheory.AEEqFun α β μ), Eq (HSMul.hSMul r ( …
                 -/
  smul_mul := by rintro _ ⟨⟩ ⟨⟩; rfl
                                 /-
                                   🎉 no goals
                                 -/


instance [AddMonoid β] [ContinuousAdd β] : DistribMulAction Mᵈᵐᵃ (α →ₘ[μ] β) where
  smul_zero := smul_zero
  smul_add := smul_add


