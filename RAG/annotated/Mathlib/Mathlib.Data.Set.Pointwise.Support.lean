theorem mulSupport_comp_inv_smul [One γ] (c : α) (f : β → γ) :
    (mulSupport fun x ↦ f (c⁻¹ • x)) = c • mulSupport f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Group α
    inst✝¹ : MulAction α β
    inst✝ : One γ
    c : α
    f : β → γ
    ⊢ Eq (Function.mulSupport fun x => f (HSMul.hSMul (Inv.inv c) x)) (HSMul.hSMul …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Group α
    inst✝¹ : MulAction α β
    inst✝ : One γ
    c : α
    f : β → γ
    x : β
    ⊢ Iff (Membership.mem (Function.mulSupport fun x => f (HSMul.hSMul (Inv.inv c) …
  -/
  simp only [mem_smul_set_iff_inv_smul_mem, mem_mulSupport]
  /-
    🎉 no goals
  -/

/- Note: to_additive also automatically translates `SMul` to `VAdd`, so we give the additive version
manually. -/

theorem support_comp_inv_smul [Zero γ] (c : α) (f : β → γ) :
    (support fun x ↦ f (c⁻¹ • x)) = c • support f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Group α
    inst✝¹ : MulAction α β
    inst✝ : Zero γ
    c : α
    f : β → γ
    ⊢ Eq (Function.support fun x => f (HSMul.hSMul (Inv.inv c) x)) (HSMul.hSMul c  …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : Group α
    inst✝¹ : MulAction α β
    inst✝ : Zero γ
    c : α
    f : β → γ
    x : β
    ⊢ Iff (Membership.mem (Function.support fun x => f (HSMul.hSMul (Inv.inv c) x) …
  -/
  simp only [mem_smul_set_iff_inv_smul_mem, mem_support]
  /-
    🎉 no goals
  -/


theorem mulSupport_comp_inv_smul₀ [One γ] {c : α} (hc : c ≠ 0) (f : β → γ) :
    (mulSupport fun x ↦ f (c⁻¹ • x)) = c • mulSupport f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : GroupWithZero α
    inst✝¹ : MulAction α β
    inst✝ : One γ
    c : α
    hc : Ne c 0
    f : β → γ
    ⊢ Eq (Function.mulSupport fun x => f (HSMul.hSMul (Inv.inv c) x)) (HSMul.hSMul …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : GroupWithZero α
    inst✝¹ : MulAction α β
    inst✝ : One γ
    c : α
    hc : Ne c 0
    f : β → γ
    x : β
    ⊢ Iff (Membership.mem (Function.mulSupport fun x => f (HSMul.hSMul (Inv.inv c) …
  -/
  simp only [mem_smul_set_iff_inv_smul_mem₀ hc, mem_mulSupport]
  /-
    🎉 no goals
  -/

/- Note: to_additive also automatically translates `SMul` to `VAdd`, so we give the additive version
manually. -/

theorem support_comp_inv_smul₀ [Zero γ] {c : α} (hc : c ≠ 0) (f : β → γ) :
    (support fun x ↦ f (c⁻¹ • x)) = c • support f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : GroupWithZero α
    inst✝¹ : MulAction α β
    inst✝ : Zero γ
    c : α
    hc : Ne c 0
    f : β → γ
    ⊢ Eq (Function.support fun x => f (HSMul.hSMul (Inv.inv c) x)) (HSMul.hSMul c  …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : GroupWithZero α
    inst✝¹ : MulAction α β
    inst✝ : Zero γ
    c : α
    hc : Ne c 0
    f : β → γ
    x : β
    ⊢ Iff (Membership.mem (Function.support fun x => f (HSMul.hSMul (Inv.inv c) x) …
  -/
  simp only [mem_smul_set_iff_inv_smul_mem₀ hc, mem_support]
  /-
    🎉 no goals
  -/


