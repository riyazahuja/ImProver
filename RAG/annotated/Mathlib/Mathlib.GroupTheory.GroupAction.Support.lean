/-- A set `s` supports `b` if `g • b = b` whenever `g • a = a` for all `a ∈ s`. -/
@[to_additive "A set `s` supports `b` if `g +ᵥ b = b` whenever `g +ᵥ a = a` for all `a ∈ s`."]
def Supports (s : Set α) (b : β) :=
  ∀ g : G, (∀ ⦃a⦄, a ∈ s → g • a = a) → g • b = b


@[to_additive]
theorem supports_of_mem (ha : a ∈ s) : Supports G s a := fun _ h => h ha


@[to_additive]
theorem Supports.mono (h : s ⊆ t) (hs : Supports G s b) : Supports G t b := fun _ hg =>
  (hs _) fun _ ha => hg <| h ha


@[to_additive]
theorem Supports.smul (g : H) (h : Supports G s b) : Supports G (g • s) (g • b) := by
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁶ : Group H
    inst✝⁵ : SMul G α
    inst✝⁴ : SMul G β
    inst✝³ : MulAction H α
    inst✝² : SMul H β
    inst✝¹ : SMulCommClass G H β
    inst✝ : SMulCommClass G H α
    s : Set α
    b : β
    g : H
    h : MulAction.Supports G s b
    ⊢ MulAction.Supports G (HSMul.hSMul g s) (HSMul.hSMul g b)
  -/
  rintro g' hg'
  /-
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁶ : Group H
    inst✝⁵ : SMul G α
    inst✝⁴ : SMul G β
    inst✝³ : MulAction H α
    inst✝² : SMul H β
    inst✝¹ : SMulCommClass G H β
    inst✝ : SMulCommClass G H α
    s : Set α
    b : β
    g : H
    h : MulAction.Supports G s b
    g' : G
    hg' : ∀ ⦃a : α⦄, Membership.mem (HSMul.hSMul g s) a → Eq (HSMul.hSMul g' a) a
    ⊢ Eq (HSMul.hSMul g' (HSMul.hSMul g b)) (HSMul.hSMul g b)
  -/
  rw [smul_comm, h]
  /-
    case a
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁶ : Group H
    inst✝⁵ : SMul G α
    inst✝⁴ : SMul G β
    inst✝³ : MulAction H α
    inst✝² : SMul H β
    inst✝¹ : SMulCommClass G H β
    inst✝ : SMulCommClass G H α
    s : Set α
    b : β
    g : H
    h : MulAction.Supports G s b
    g' : G
    hg' : ∀ ⦃a : α⦄, Membership.mem (HSMul.hSMul g s) a → Eq (HSMul.hSMul g' a) a
    ⊢ ∀ ⦃a : α⦄, Membership.mem s a → Eq (HSMul.hSMul g' a) a
  -/
  rintro a ha
  /-
    case a
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁶ : Group H
    inst✝⁵ : SMul G α
    inst✝⁴ : SMul G β
    inst✝³ : MulAction H α
    inst✝² : SMul H β
    inst✝¹ : SMulCommClass G H β
    inst✝ : SMulCommClass G H α
    s : Set α
    b : β
    g : H
    h : MulAction.Supports G s b
    g' : G
    hg' : ∀ ⦃a : α⦄, Membership.mem (HSMul.hSMul g s) a → Eq (HSMul.hSMul g' a) a
    a : α
    ha : Membership.mem s a
    ⊢ Eq (HSMul.hSMul g' a) a
  -/
  have := Set.forall_mem_image.1 hg' ha
  /-
    case a
    G : Type u_1
    H : Type u_2
    α : Type u_3
    β : Type u_4
    inst✝⁶ : Group H
    inst✝⁵ : SMul G α
    inst✝⁴ : SMul G β
    inst✝³ : MulAction H α
    inst✝² : SMul H β
    inst✝¹ : SMulCommClass G H β
    inst✝ : SMulCommClass G H α
    s : Set α
    b : β
    g : H
    h : MulAction.Supports G s b
    g' : G
    hg' : ∀ ⦃a : α⦄, Membership.mem (HSMul.hSMul g s) a → Eq (HSMul.hSMul g' a) a
    a : α
    ha : Membership.mem s a
    this : Eq (HSMul.hSMul g' (HSMul.hSMul g a)) (HSMul.hSMul g a)
    ⊢ Eq (HSMul.hSMul g' a) a
  -/
  rwa [smul_comm, smul_left_cancel_iff] at this
  /-
    🎉 no goals
  -/


