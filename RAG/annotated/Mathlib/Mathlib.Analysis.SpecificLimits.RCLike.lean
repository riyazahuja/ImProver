theorem RCLike.tendsto_inverse_atTop_nhds_zero_nat :
    Tendsto (fun n : ℕ => (n : 𝕜)⁻¹) atTop (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
  -/
  convert tendsto_algebraMap_inverse_atTop_nhds_zero_nat 𝕜
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    x✝ : Nat
    ⊢ Eq (Inv.inv ↑x✝) (Function.comp (⇑(algebraMap Real 𝕜)) (fun n => Inv.inv ↑n) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem RCLike.tendsto_add_mul_div_add_mul_atTop_nhds (a b c : 𝕜) {d : 𝕜} (hd : d ≠ 0) :
    Tendsto (fun k : ℕ ↦ (a + c * k) / (b + d * k)) atTop (𝓝 (c / d)) := by
  /-
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    a b c d : 𝕜
    hd : Ne d 0
    ⊢ Filter.Tendsto (fun k => HDiv.hDiv (HAdd.hAdd a (HMul.hMul c ↑k)) (HAdd.hAdd …
  -/
  apply Filter.Tendsto.congr'
  /-
    case hl
    𝕜 : Type u_1
    inst✝ : RCLike 𝕜
    a b c d : 𝕜
    hd : Ne d 0
    ⊢ Filter.atTop.EventuallyEq ?f₁ fun k => HDiv.hDiv (HAdd.hAdd a (HMul.hMul c ↑ …
  -/
  case f₁ => exact fun k ↦ (a * (↑k)⁻¹ + c) / (b * (↑k)⁻¹ + d)
    /-
      case hl
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      a b c d : 𝕜
      hd : Ne d 0
      ⊢ Filter.atTop.EventuallyEq (fun k => HDiv.hDiv (HAdd.hAdd (HMul.hMul a (Inv.i …
    -/
  · refine (eventually_ne_atTop 0).mp (Eventually.of_forall ?_)
    /-
      case hl
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      a b c d : 𝕜
      hd : Ne d 0
      ⊢ ∀ (x : Nat), Ne x 0 → Eq ((fun k => HDiv.hDiv (HAdd.hAdd (HMul.hMul a (Inv.i …
    -/
    intro h hx
    /-
      case hl
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      a b c d : 𝕜
      hd : Ne d 0
      h : Nat
      hx : Ne h 0
      ⊢ Eq ((fun k => HDiv.hDiv (HAdd.hAdd (HMul.hMul a (Inv.inv ↑k)) c) (HAdd.hAdd  …
    -/
    field_simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case h
      𝕜 : Type u_1
      inst✝ : RCLike 𝕜
      a b c d : 𝕜
      hd : Ne d 0
      ⊢ Filter.Tendsto (fun k => HDiv.hDiv (HAdd.hAdd (HMul.hMul a (Inv.inv ↑k)) c)  …
    -/
  · apply Filter.Tendsto.div _ _ hd
    all_goals
      apply zero_add (_ : 𝕜) ▸ Filter.Tendsto.add_const _ _
      apply mul_zero (_ : 𝕜) ▸ Filter.Tendsto.const_mul _ _
      exact RCLike.tendsto_inverse_atTop_nhds_zero_nat 𝕜

