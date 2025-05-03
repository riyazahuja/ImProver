/-- The `n = 1` case of the Ahlswede-Daykin inequality. Note that we can't just expand everything
out and bound termwise since `c₀ * d₁` appears twice on the RHS of the assumptions while `c₁ * d₀`
does not appear. -/
private lemma ineq [ExistsAddOfLE β] {a₀ a₁ b₀ b₁ c₀ c₁ d₀ d₁ : β}
    (ha₀ : 0 ≤ a₀) (ha₁ : 0 ≤ a₁) (hb₀ : 0 ≤ b₀) (hb₁ : 0 ≤ b₁)
    (hc₀ : 0 ≤ c₀) (hc₁ : 0 ≤ c₁) (hd₀ : 0 ≤ d₀) (hd₁ : 0 ≤ d₁)
    (h₀₀ : a₀ * b₀ ≤ c₀ * d₀) (h₁₀ : a₁ * b₀ ≤ c₀ * d₁)
    (h₀₁ : a₀ * b₁ ≤ c₀ * d₁) (h₁₁ : a₁ * b₁ ≤ c₁ * d₁) :
    (a₀ + a₁) * (b₀ + b₁) ≤ (c₀ + c₁) * (d₀ + d₁) := by
  calc
    _ = a₀ * b₀ + (a₀ * b₁ + a₁ * b₀) + a₁ * b₁ := by ring
    _ ≤ c₀ * d₀ + (c₀ * d₁ + c₁ * d₀) + c₁ * d₁ := add_le_add_three h₀₀ ?_ h₁₁
    _ = (c₀ + c₁) * (d₀ + d₁) := by ring
  /-
    β : Type u_2
    inst✝¹ : LinearOrderedCommSemiring β
    inst✝ : ExistsAddOfLE β
    a₀ a₁ b₀ b₁ c₀ c₁ d₀ d₁ : β
    ha₀ : LE.le 0 a₀
    ha₁ : LE.le 0 a₁
    hb₀ : LE.le 0 b₀
    hb₁ : LE.le 0 b₁
    hc₀ : LE.le 0 c₀
    hc₁ : LE.le 0 c₁
    hd₀ : LE.le 0 d₀
    hd₁ : LE.le 0 d₁
    h₀₀ : LE.le (HMul.hMul a₀ b₀) (HMul.hMul c₀ d₀)
    h₁₀ : LE.le (HMul.hMul a₁ b₀) (HMul.hMul c₀ d₁)
    h₀₁ : LE.le (HMul.hMul a₀ b₁) (HMul.hMul c₀ d₁)
    h₁₁ : LE.le (HMul.hMul a₁ b₁) (HMul.hMul c₁ d₁)
    ⊢ LE.le (HAdd.hAdd (HMul.hMul a₀ b₁) (HMul.hMul a₁ b₀)) (HAdd.hAdd (HMul.hMul  …
  -/
  obtain hcd | hcd := (mul_nonneg hc₀ hd₁).eq_or_gt
    /-
      case inl
      β : Type u_2
      inst✝¹ : LinearOrderedCommSemiring β
      inst✝ : ExistsAddOfLE β
      a₀ a₁ b₀ b₁ c₀ c₁ d₀ d₁ : β
      ha₀ : LE.le 0 a₀
      ha₁ : LE.le 0 a₁
      hb₀ : LE.le 0 b₀
      hb₁ : LE.le 0 b₁
      hc₀ : LE.le 0 c₀
      hc₁ : LE.le 0 c₁
      hd₀ : LE.le 0 d₀
      hd₁ : LE.le 0 d₁
      h₀₀ : LE.le (HMul.hMul a₀ b₀) (HMul.hMul c₀ d₀)
      h₁₀ : LE.le (HMul.hMul a₁ b₀) (HMul.hMul c₀ d₁)
      h₀₁ : LE.le (HMul.hMul a₀ b₁) (HMul.hMul c₀ d₁)
      h₁₁ : LE.le (HMul.hMul a₁ b₁) (HMul.hMul c₁ d₁)
      hcd : Eq (HMul.hMul c₀ d₁) 0
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a₀ b₁) (HMul.hMul a₁ b₀)) (HAdd.hAdd (HMul.hMul  …
    -/
  · rw [hcd] at h₀₁ h₁₀
    /-
      case inl
      β : Type u_2
      inst✝¹ : LinearOrderedCommSemiring β
      inst✝ : ExistsAddOfLE β
      a₀ a₁ b₀ b₁ c₀ c₁ d₀ d₁ : β
      ha₀ : LE.le 0 a₀
      ha₁ : LE.le 0 a₁
      hb₀ : LE.le 0 b₀
      hb₁ : LE.le 0 b₁
      hc₀ : LE.le 0 c₀
      hc₁ : LE.le 0 c₁
      hd₀ : LE.le 0 d₀
      hd₁ : LE.le 0 d₁
      h₀₀ : LE.le (HMul.hMul a₀ b₀) (HMul.hMul c₀ d₀)
      h₁₀ : LE.le (HMul.hMul a₁ b₀) 0
      h₀₁ : LE.le (HMul.hMul a₀ b₁) 0
      h₁₁ : LE.le (HMul.hMul a₁ b₁) (HMul.hMul c₁ d₁)
      hcd : Eq (HMul.hMul c₀ d₁) 0
      ⊢ LE.le (HAdd.hAdd (HMul.hMul a₀ b₁) (HMul.hMul a₁ b₀)) (HAdd.hAdd (HMul.hMul  …
    -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
    rw [h₀₁.antisymm, h₁₀.antisymm, add_zero] <;> positivity
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case inr
    β : Type u_2
    inst✝¹ : LinearOrderedCommSemiring β
    inst✝ : ExistsAddOfLE β
    a₀ a₁ b₀ b₁ c₀ c₁ d₀ d₁ : β
    ha₀ : LE.le 0 a₀
    ha₁ : LE.le 0 a₁
    hb₀ : LE.le 0 b₀
    hb₁ : LE.le 0 b₁
    hc₀ : LE.le 0 c₀
    hc₁ : LE.le 0 c₁
    hd₀ : LE.le 0 d₀
    hd₁ : LE.le 0 d₁
    h₀₀ : LE.le (HMul.hMul a₀ b₀) (HMul.hMul c₀ d₀)
    h₁₀ : LE.le (HMul.hMul a₁ b₀) (HMul.hMul c₀ d₁)
    h₀₁ : LE.le (HMul.hMul a₀ b₁) (HMul.hMul c₀ d₁)
    h₁₁ : LE.le (HMul.hMul a₁ b₁) (HMul.hMul c₁ d₁)
    hcd : LT.lt 0 (HMul.hMul c₀ d₁)
    ⊢ LE.le (HAdd.hAdd (HMul.hMul a₀ b₁) (HMul.hMul a₁ b₀)) (HAdd.hAdd (HMul.hMul  …
  -/
  refine le_of_mul_le_mul_right ?_ hcd
  calc (a₀ * b₁ + a₁ * b₀) * (c₀ * d₁)
      = a₀ * b₁ * (c₀ * d₁) + c₀ * d₁ * (a₁ * b₀) := by ring
    _ ≤ a₀ * b₁ * (a₁ * b₀) + c₀ * d₁ * (c₀ * d₁) := mul_add_mul_le_mul_add_mul h₀₁ h₁₀
    _ = a₀ * b₀ * (a₁ * b₁) + c₀ * d₁ * (c₀ * d₁) := by ring
    _ ≤ c₀ * d₀ * (c₁ * d₁) + c₀ * d₁ * (c₀ * d₁) :=
        add_le_add_right (mul_le_mul h₀₀ h₁₁ (by positivity) <| by positivity) _
    _ = (c₀ * d₁ + c₁ * d₀) * (c₀ * d₁) := by ring


private def collapse (𝒜 : Finset (Finset α)) (a : α) (f : Finset α → β) (s : Finset α) : β :=
  ∑ t ∈ 𝒜 with t.erase a = s, f t


private lemma erase_eq_iff (hs : a ∉ s) : t.erase a = s ↔ t = s ∨ t = insert a s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s t : Finset α
    hs : Not (Membership.mem s a)
    ⊢ Iff (Eq (t.erase a) s) (Or (Eq t s) (Eq t (Insert.insert a s)))
  -/
  by_cases ht : a ∈ t <;>
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s t : Finset α
      hs : Not (Membership.mem s a)
      ht : Membership.mem t a
      ⊢ Iff (Eq (t.erase a) s) (Or (Eq t s) (Eq t (Insert.insert a s)))
    -/
    /-
      case pos
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s t : Finset α
      hs : Not (Membership.mem s a)
      ht : Membership.mem t a
      ⊢ Eq t s → Eq t (Insert.insert a s)
    -/
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s t : Finset α
      hs : Not (Membership.mem s a)
      ht : Not (Membership.mem t a)
      ⊢ Eq t (Insert.insert a s) → Eq t s
    -/
    aesop
    /-
      🎉 no goals
    -/


private lemma filter_collapse_eq (ha : a ∉ s) (𝒜 : Finset (Finset α)) :
    {t ∈ 𝒜 | t.erase a = s} =
      if s ∈ 𝒜 then
        (if insert a s ∈ 𝒜 then {s, insert a s} else {s})
      else
        (if insert a s ∈ 𝒜 then {insert a s} else ∅) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    𝒜 : Finset (Finset α)
    ⊢ Eq (Finset.filter (fun t => Eq (t.erase a) s) 𝒜) (ite (Membership.mem 𝒜 s) ( …
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
  ext t; split_ifs <;> simp [erase_eq_iff ha] <;> aesop
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma collapse_eq (ha : a ∉ s) (𝒜 : Finset (Finset α)) (f : Finset α → β) :
    collapse 𝒜 a f s = (if s ∈ 𝒜 then f s else 0) +
      if insert a s ∈ 𝒜 then f (insert a s) else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    𝒜 : Finset (Finset α)
    f : Finset α → β
    ⊢ Eq (collapse 𝒜 a f s) (HAdd.hAdd (ite (Membership.mem 𝒜 s) (f s) 0) (ite (Me …
  -/
  rw [collapse, filter_collapse_eq ha]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    a : α
    s : Finset α
    ha : Not (Membership.mem s a)
    𝒜 : Finset (Finset α)
    f : Finset α → β
    ⊢ Eq ((ite (Membership.mem 𝒜 s) (ite (Membership.mem 𝒜 (Insert.insert a s)) (I …
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
  split_ifs <;> simp [(ne_of_mem_of_not_mem' (mem_insert_self a s) ha).symm, *]
                /-
                  🎉 no goals
                -/


lemma collapse_of_mem (ha : a ∉ s) (ht : t ∈ 𝒜) (hu : u ∈ 𝒜) (hts : t = s)
    (hus : u = insert a s) : collapse 𝒜 a f s = f t + f u := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    s t u : Finset α
    ha : Not (Membership.mem s a)
    ht : Membership.mem 𝒜 t
    hu : Membership.mem 𝒜 u
    hts : Eq t s
    hus : Eq u (Insert.insert a s)
    ⊢ Eq (collapse 𝒜 a f s) (HAdd.hAdd (f t) (f u))
  -/
  subst hts; subst hus; simp_rw [collapse_eq ha, if_pos ht, if_pos hu]
                        /-
                          🎉 no goals
                        -/


lemma le_collapse_of_mem (ha : a ∉ s) (hf : 0 ≤ f) (hts : t = s) (ht : t ∈ 𝒜) :
    f t ≤ collapse 𝒜 a f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    s t : Finset α
    ha : Not (Membership.mem s a)
    hf : LE.le 0 f
    hts : Eq t s
    ht : Membership.mem 𝒜 t
    ⊢ LE.le (f t) (collapse 𝒜 a f s)
  -/
  subst hts
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    t : Finset α
    hf : LE.le 0 f
    ht : Membership.mem 𝒜 t
    ha : Not (Membership.mem t a)
    ⊢ LE.le (f t) (collapse 𝒜 a f t)
  -/
  rw [collapse_eq ha, if_pos ht]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    t : Finset α
    hf : LE.le 0 f
    ht : Membership.mem 𝒜 t
    ha : Not (Membership.mem t a)
    ⊢ LE.le (f t) (HAdd.hAdd (f t) (ite (Membership.mem 𝒜 (Insert.insert a t)) (f  …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      t : Finset α
      hf : LE.le 0 f
      ht : Membership.mem 𝒜 t
      ha : Not (Membership.mem t a)
      h✝ : Membership.mem 𝒜 (Insert.insert a t)
      ⊢ LE.le (f t) (HAdd.hAdd (f t) (f (Insert.insert a t)))
    -/
  · exact le_add_of_nonneg_right <| hf _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      t : Finset α
      hf : LE.le 0 f
      ht : Membership.mem 𝒜 t
      ha : Not (Membership.mem t a)
      h✝ : Not (Membership.mem 𝒜 (Insert.insert a t))
      ⊢ LE.le (f t) (HAdd.hAdd (f t) 0)
    -/
  · rw [add_zero]
    /-
      🎉 no goals
    -/


lemma le_collapse_of_insert_mem (ha : a ∉ s) (hf : 0 ≤ f) (hts : t = insert a s) (ht : t ∈ 𝒜) :
    f t ≤ collapse 𝒜 a f s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    s t : Finset α
    ha : Not (Membership.mem s a)
    hf : LE.le 0 f
    hts : Eq t (Insert.insert a s)
    ht : Membership.mem 𝒜 t
    ⊢ LE.le (f t) (collapse 𝒜 a f s)
  -/
  rw [collapse_eq ha, ← hts, if_pos ht]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : LinearOrderedCommSemiring β
    𝒜 : Finset (Finset α)
    a : α
    f : Finset α → β
    s t : Finset α
    ha : Not (Membership.mem s a)
    hf : LE.le 0 f
    hts : Eq t (Insert.insert a s)
    ht : Membership.mem 𝒜 t
    ⊢ LE.le (f t) (HAdd.hAdd (ite (Membership.mem 𝒜 s) (f s) 0) (f t))
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      s t : Finset α
      ha : Not (Membership.mem s a)
      hf : LE.le 0 f
      hts : Eq t (Insert.insert a s)
      ht : Membership.mem 𝒜 t
      h✝ : Membership.mem 𝒜 s
      ⊢ LE.le (f t) (HAdd.hAdd (f s) (f t))
    -/
  · exact le_add_of_nonneg_left <| hf _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      s t : Finset α
      ha : Not (Membership.mem s a)
      hf : LE.le 0 f
      hts : Eq t (Insert.insert a s)
      ht : Membership.mem 𝒜 t
      h✝ : Not (Membership.mem 𝒜 s)
      ⊢ LE.le (f t) (HAdd.hAdd 0 (f t))
    -/
  · rw [zero_add]
    /-
      🎉 no goals
    -/


lemma collapse_nonneg (hf : 0 ≤ f) : 0 ≤ collapse 𝒜 a f := fun _s ↦ sum_nonneg fun _t _ ↦ hf _


lemma collapse_modular [ExistsAddOfLE β]
    (hu : a ∉ u) (h₁ : 0 ≤ f₁) (h₂ : 0 ≤ f₂) (h₃ : 0 ≤ f₃) (h₄ : 0 ≤ f₄)
    (h : ∀ ⦃s⦄, s ⊆ insert a u → ∀ ⦃t⦄, t ⊆ insert a u →  f₁ s * f₂ t ≤ f₃ (s ∩ t) * f₄ (s ∪ t))
    (𝒜 ℬ : Finset (Finset α)) :
    ∀ ⦃s⦄, s ⊆ u → ∀ ⦃t⦄, t ⊆ u → collapse 𝒜 a f₁ s * collapse ℬ a f₂ t ≤
      collapse (𝒜 ⊼ ℬ) a f₃ (s ∩ t) * collapse (𝒜 ⊻ ℬ) a f₄ (s ∪ t) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    ⊢ ∀ ⦃s : Finset α⦄, HasSubset.Subset s u → ∀ ⦃t : Finset α⦄, HasSubset.Subset  …
  -/
  rintro s hsu t htu
  -- Gather a bunch of facts we'll need a lot
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have := hsu.trans <| subset_insert a _
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this : HasSubset.Subset s (Insert.insert a u)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have := htu.trans <| subset_insert a _
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝ : HasSubset.Subset s (Insert.insert a u)
    this : HasSubset.Subset t (Insert.insert a u)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have := insert_subset_insert a hsu
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝¹ : HasSubset.Subset s (Insert.insert a u)
    this✝ : HasSubset.Subset t (Insert.insert a u)
    this : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have := insert_subset_insert a htu
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝² : HasSubset.Subset s (Insert.insert a u)
    this✝¹ : HasSubset.Subset t (Insert.insert a u)
    this✝ : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have has := not_mem_mono hsu hu
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝² : HasSubset.Subset s (Insert.insert a u)
    this✝¹ : HasSubset.Subset t (Insert.insert a u)
    this✝ : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    has : Not (Membership.mem s a)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have hat := not_mem_mono htu hu
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝² : HasSubset.Subset s (Insert.insert a u)
    this✝¹ : HasSubset.Subset t (Insert.insert a u)
    this✝ : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    has : Not (Membership.mem s a)
    hat : Not (Membership.mem t a)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have : a ∉ s ∩ t := not_mem_mono (inter_subset_left.trans hsu) hu
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝³ : HasSubset.Subset s (Insert.insert a u)
    this✝² : HasSubset.Subset t (Insert.insert a u)
    this✝¹ : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this✝ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    has : Not (Membership.mem s a)
    hat : Not (Membership.mem t a)
    this : Not (Membership.mem (Inter.inter s t) a)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  have := not_mem_union.2 ⟨has, hat⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝⁴ : HasSubset.Subset s (Insert.insert a u)
    this✝³ : HasSubset.Subset t (Insert.insert a u)
    this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    has : Not (Membership.mem s a)
    hat : Not (Membership.mem t a)
    this✝ : Not (Membership.mem (Inter.inter s t) a)
    this : Not (Membership.mem (Union.union s t) a)
    ⊢ LE.le (HMul.hMul (collapse 𝒜 a f₁ s) (collapse ℬ a f₂ t)) (HMul.hMul (collap …
  -/
  rw [collapse_eq has]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    a : α
    f₁ f₂ f₃ f₄ : Finset α → β
    u : Finset α
    inst✝ : ExistsAddOfLE β
    hu : Not (Membership.mem u a)
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    s : Finset α
    hsu : HasSubset.Subset s u
    t : Finset α
    htu : HasSubset.Subset t u
    this✝⁴ : HasSubset.Subset s (Insert.insert a u)
    this✝³ : HasSubset.Subset t (Insert.insert a u)
    this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
    this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
    has : Not (Membership.mem s a)
    hat : Not (Membership.mem t a)
    this✝ : Not (Membership.mem (Inter.inter s t) a)
    this : Not (Membership.mem (Union.union s t) a)
    ⊢ LE.le (HMul.hMul (HAdd.hAdd (ite (Membership.mem 𝒜 s) (f₁ s) 0) (ite (Member …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Membership.mem 𝒜 s
      h✝ : Membership.mem 𝒜 (Insert.insert a s)
      ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (collapse ℬ a f …
    -/
  · rw [collapse_eq hat]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Membership.mem 𝒜 s
      h✝ : Membership.mem 𝒜 (Insert.insert a s)
      ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (HAdd.hAdd (ite …
    -/
    split_ifs
    · rw [collapse_of_mem ‹_› (inter_mem_infs ‹_› ‹_›) (inter_mem_infs ‹_› ‹_›) rfl
        (insert_inter_distrib _ _ _).symm, collapse_of_mem ‹_› (union_mem_sups ‹_› ‹_›)
        (union_mem_sups ‹_› ‹_›) rfl (insert_union_distrib _ _ _).symm]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Membership.mem ℬ t
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (HAdd.hAdd (f₂  …
      -/
      refine ineq (h₁ _) (h₁ _) (h₂ _) (h₂ _) (h₃ _) (h₃ _) (h₄ _) (h₄ _) (h ‹_› ‹_›) ?_ ?_ ?_
        /-
          case pos.refine_1
          α : Type u_1
          β : Type u_2
          inst✝² : DecidableEq α
          inst✝¹ : LinearOrderedCommSemiring β
          a : α
          f₁ f₂ f₃ f₄ : Finset α → β
          u : Finset α
          inst✝ : ExistsAddOfLE β
          hu : Not (Membership.mem u a)
          h₁ : LE.le 0 f₁
          h₂ : LE.le 0 f₂
          h₃ : LE.le 0 f₃
          h₄ : LE.le 0 f₄
          h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
          𝒜 ℬ : Finset (Finset α)
          s : Finset α
          hsu : HasSubset.Subset s u
          t : Finset α
          htu : HasSubset.Subset t u
          this✝⁴ : HasSubset.Subset s (Insert.insert a u)
          this✝³ : HasSubset.Subset t (Insert.insert a u)
          this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
          this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
          has : Not (Membership.mem s a)
          hat : Not (Membership.mem t a)
          this✝ : Not (Membership.mem (Inter.inter s t) a)
          this : Not (Membership.mem (Union.union s t) a)
          h✝³ : Membership.mem 𝒜 s
          h✝² : Membership.mem 𝒜 (Insert.insert a s)
          h✝¹ : Membership.mem ℬ t
          h✝ : Membership.mem ℬ (Insert.insert a t)
          ⊢ LE.le (HMul.hMul (f₁ (Insert.insert a s)) (f₂ t)) (HMul.hMul (f₃ (Inter.inte …
        -/
      · simpa [*] using h ‹insert a s ⊆ _› ‹t ⊆ _›
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_2
          α : Type u_1
          β : Type u_2
          inst✝² : DecidableEq α
          inst✝¹ : LinearOrderedCommSemiring β
          a : α
          f₁ f₂ f₃ f₄ : Finset α → β
          u : Finset α
          inst✝ : ExistsAddOfLE β
          hu : Not (Membership.mem u a)
          h₁ : LE.le 0 f₁
          h₂ : LE.le 0 f₂
          h₃ : LE.le 0 f₃
          h₄ : LE.le 0 f₄
          h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
          𝒜 ℬ : Finset (Finset α)
          s : Finset α
          hsu : HasSubset.Subset s u
          t : Finset α
          htu : HasSubset.Subset t u
          this✝⁴ : HasSubset.Subset s (Insert.insert a u)
          this✝³ : HasSubset.Subset t (Insert.insert a u)
          this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
          this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
          has : Not (Membership.mem s a)
          hat : Not (Membership.mem t a)
          this✝ : Not (Membership.mem (Inter.inter s t) a)
          this : Not (Membership.mem (Union.union s t) a)
          h✝³ : Membership.mem 𝒜 s
          h✝² : Membership.mem 𝒜 (Insert.insert a s)
          h✝¹ : Membership.mem ℬ t
          h✝ : Membership.mem ℬ (Insert.insert a t)
          ⊢ LE.le (HMul.hMul (f₁ s) (f₂ (Insert.insert a t))) (HMul.hMul (f₃ (Inter.inte …
        -/
      · simpa [*] using h ‹s ⊆ _› ‹insert a t ⊆ _›
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_3
          α : Type u_1
          β : Type u_2
          inst✝² : DecidableEq α
          inst✝¹ : LinearOrderedCommSemiring β
          a : α
          f₁ f₂ f₃ f₄ : Finset α → β
          u : Finset α
          inst✝ : ExistsAddOfLE β
          hu : Not (Membership.mem u a)
          h₁ : LE.le 0 f₁
          h₂ : LE.le 0 f₂
          h₃ : LE.le 0 f₃
          h₄ : LE.le 0 f₄
          h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
          𝒜 ℬ : Finset (Finset α)
          s : Finset α
          hsu : HasSubset.Subset s u
          t : Finset α
          htu : HasSubset.Subset t u
          this✝⁴ : HasSubset.Subset s (Insert.insert a u)
          this✝³ : HasSubset.Subset t (Insert.insert a u)
          this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
          this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
          has : Not (Membership.mem s a)
          hat : Not (Membership.mem t a)
          this✝ : Not (Membership.mem (Inter.inter s t) a)
          this : Not (Membership.mem (Union.union s t) a)
          h✝³ : Membership.mem 𝒜 s
          h✝² : Membership.mem 𝒜 (Insert.insert a s)
          h✝¹ : Membership.mem ℬ t
          h✝ : Membership.mem ℬ (Insert.insert a t)
          ⊢ LE.le (HMul.hMul (f₁ (Insert.insert a s)) (f₂ (Insert.insert a t))) (HMul.hM …
        -/
      · simpa [*] using h ‹insert a s ⊆ _› ‹insert a t ⊆ _›
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Membership.mem ℬ t
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (HAdd.hAdd (f₂  …
      -/
    · rw [add_zero, add_mul]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Membership.mem ℬ t
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) (f₂ t)) (HMul.hMul (f₁ (Insert.insert a s …
      -/
      refine (add_le_add (h ‹_› ‹_›) <| h ‹_› ‹_›).trans ?_
      rw [collapse_of_mem ‹_› (union_mem_sups ‹_› ‹_›) (union_mem_sups ‹_› ‹_›) rfl
        (insert_union _ _ _), insert_inter_of_not_mem ‹_›, ← mul_add]
      exact mul_le_mul_of_nonneg_right (le_collapse_of_mem ‹_› h₃ rfl <| inter_mem_infs ‹_› ‹_›) <|
        add_nonneg (h₄ _) <| h₄ _
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (HAdd.hAdd 0 (f …
      -/
    · rw [zero_add, add_mul]
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) (f₂ (Insert.insert a t))) (HMul.hMul (f₁  …
      -/
      refine (add_le_add (h ‹_› ‹_›) <| h ‹_› ‹_›).trans ?_
      rw [collapse_of_mem ‹_› (inter_mem_infs ‹_› ‹_›) (inter_mem_infs ‹_› ‹_›)
        (inter_insert_of_not_mem ‹_›) (insert_inter_distrib _ _ _).symm, union_insert,
        insert_union_distrib, ← add_mul]
      exact mul_le_mul_of_nonneg_left (le_collapse_of_insert_mem ‹_› h₄
        (insert_union_distrib _ _ _).symm <| union_mem_sups ‹_› ‹_›) <| add_nonneg (h₃ _) <| h₃ _
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) (f₁ (Insert.insert a s))) (HAdd.hAdd 0 0) …
      -/
    · rw [add_zero, mul_zero]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le 0 (HMul.hMul (collapse (HasInfs.infs 𝒜 ℬ) a f₃ (Inter.inter s t)) (col …
      -/
      exact mul_nonneg (collapse_nonneg h₃ _) <| collapse_nonneg h₄ _
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Membership.mem 𝒜 s
      h✝ : Not (Membership.mem 𝒜 (Insert.insert a s))
      ⊢ LE.le (HMul.hMul (HAdd.hAdd (f₁ s) 0) (collapse ℬ a f₂ t)) (HMul.hMul (colla …
    -/
  · rw [add_zero, collapse_eq hat, mul_add]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Membership.mem 𝒜 s
      h✝ : Not (Membership.mem 𝒜 (Insert.insert a s))
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) (ite (Membership.mem ℬ t) (f₂ t) 0)) (HMu …
    -/
    split_ifs
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Not (Membership.mem 𝒜 (Insert.insert a s))
        h✝¹ : Membership.mem ℬ t
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) (f₂ t)) (HMul.hMul (f₁ s) (f₂ (Insert.ins …
      -/
    · refine (add_le_add (h ‹_› ‹_›) <| h ‹_› ‹_›).trans ?_
      rw [collapse_of_mem ‹_› (union_mem_sups ‹_› ‹_›) (union_mem_sups ‹_› ‹_›) rfl
        (union_insert _ _ _), inter_insert_of_not_mem ‹_›, ← mul_add]
      exact mul_le_mul_of_nonneg_right (le_collapse_of_mem ‹_› h₃ rfl <| inter_mem_infs ‹_› ‹_›) <|
        add_nonneg (h₄ _) <| h₄ _
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Not (Membership.mem 𝒜 (Insert.insert a s))
        h✝¹ : Membership.mem ℬ t
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) (f₂ t)) (HMul.hMul (f₁ s) 0)) (HMul.hMul  …
      -/
    · rw [mul_zero, add_zero]
      exact (h ‹_› ‹_›).trans <| mul_le_mul (le_collapse_of_mem ‹_› h₃ rfl <|
        inter_mem_infs ‹_› ‹_›) (le_collapse_of_mem ‹_› h₄ rfl <| union_mem_sups ‹_› ‹_›)
        (h₄ _) <| collapse_nonneg h₃ _
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Not (Membership.mem 𝒜 (Insert.insert a s))
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) 0) (HMul.hMul (f₁ s) (f₂ (Insert.insert a …
      -/
    · rw [mul_zero, zero_add]
      refine (h ‹_› ‹_›).trans <| mul_le_mul ?_ (le_collapse_of_insert_mem ‹_› h₄
        (union_insert _ _ _) <| union_mem_sups ‹_› ‹_›) (h₄ _) <| collapse_nonneg h₃ _
      exact le_collapse_of_mem (not_mem_mono inter_subset_left ‹_›) h₃
        (inter_insert_of_not_mem ‹_›) <| inter_mem_infs ‹_› ‹_›
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Not (Membership.mem 𝒜 (Insert.insert a s))
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ s) 0) (HMul.hMul (f₁ s) 0)) (HMul.hMul (coll …
      -/
    · simp_rw [mul_zero, add_zero]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Membership.mem 𝒜 s
        h✝² : Not (Membership.mem 𝒜 (Insert.insert a s))
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le 0 (HMul.hMul (collapse (HasInfs.infs 𝒜 ℬ) a f₃ (Inter.inter s t)) (col …
      -/
      exact mul_nonneg (collapse_nonneg h₃ _) <| collapse_nonneg h₄ _
      /-
        🎉 no goals
      -/
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Not (Membership.mem 𝒜 s)
      h✝ : Membership.mem 𝒜 (Insert.insert a s)
      ⊢ LE.le (HMul.hMul (HAdd.hAdd 0 (f₁ (Insert.insert a s))) (collapse ℬ a f₂ t)) …
    -/
  · rw [zero_add, collapse_eq hat, mul_add]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Not (Membership.mem 𝒜 s)
      h✝ : Membership.mem 𝒜 (Insert.insert a s)
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ (Insert.insert a s)) (ite (Membership.mem ℬ  …
    -/
    split_ifs
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Not (Membership.mem 𝒜 s)
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Membership.mem ℬ t
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ (Insert.insert a s)) (f₂ t)) (HMul.hMul (f₁  …
      -/
    · refine (add_le_add (h ‹_› ‹_›) <| h ‹_› ‹_›).trans ?_
      rw [collapse_of_mem ‹_› (inter_mem_infs ‹_› ‹_›) (inter_mem_infs ‹_› ‹_›)
        (insert_inter_of_not_mem ‹_›) (insert_inter_distrib _ _ _).symm,
        insert_inter_of_not_mem ‹_›, ← insert_inter_distrib, insert_union, insert_union_distrib,
        ← add_mul]
      exact mul_le_mul_of_nonneg_left (le_collapse_of_insert_mem ‹_› h₄
        (insert_union_distrib _ _ _).symm <| union_mem_sups ‹_› ‹_›) <| add_nonneg (h₃ _) <| h₃ _
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Not (Membership.mem 𝒜 s)
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Membership.mem ℬ t
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ (Insert.insert a s)) (f₂ t)) (HMul.hMul (f₁  …
      -/
    · rw [mul_zero, add_zero]
      refine (h ‹_› ‹_›).trans <| mul_le_mul (le_collapse_of_mem ‹_› h₃
        (insert_inter_of_not_mem ‹_›) <| inter_mem_infs ‹_› ‹_›) (le_collapse_of_insert_mem ‹_› h₄
        (insert_union _ _ _) <| union_mem_sups ‹_› ‹_›) (h₄ _) <| collapse_nonneg h₃ _
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Not (Membership.mem 𝒜 s)
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Membership.mem ℬ (Insert.insert a t)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ (Insert.insert a s)) 0) (HMul.hMul (f₁ (Inse …
      -/
    · rw [mul_zero, zero_add]
      exact (h ‹_› ‹_›).trans <| mul_le_mul (le_collapse_of_insert_mem ‹_› h₃
        (insert_inter_distrib _ _ _).symm <| inter_mem_infs ‹_› ‹_›) (le_collapse_of_insert_mem ‹_›
        h₄ (insert_union_distrib _ _ _).symm <| union_mem_sups ‹_› ‹_›) (h₄ _) <|
        collapse_nonneg h₃ _
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Not (Membership.mem 𝒜 s)
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (f₁ (Insert.insert a s)) 0) (HMul.hMul (f₁ (Inse …
      -/
    · simp_rw [mul_zero, add_zero]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝² : DecidableEq α
        inst✝¹ : LinearOrderedCommSemiring β
        a : α
        f₁ f₂ f₃ f₄ : Finset α → β
        u : Finset α
        inst✝ : ExistsAddOfLE β
        hu : Not (Membership.mem u a)
        h₁ : LE.le 0 f₁
        h₂ : LE.le 0 f₂
        h₃ : LE.le 0 f₃
        h₄ : LE.le 0 f₄
        h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
        𝒜 ℬ : Finset (Finset α)
        s : Finset α
        hsu : HasSubset.Subset s u
        t : Finset α
        htu : HasSubset.Subset t u
        this✝⁴ : HasSubset.Subset s (Insert.insert a u)
        this✝³ : HasSubset.Subset t (Insert.insert a u)
        this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
        this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
        has : Not (Membership.mem s a)
        hat : Not (Membership.mem t a)
        this✝ : Not (Membership.mem (Inter.inter s t) a)
        this : Not (Membership.mem (Union.union s t) a)
        h✝³ : Not (Membership.mem 𝒜 s)
        h✝² : Membership.mem 𝒜 (Insert.insert a s)
        h✝¹ : Not (Membership.mem ℬ t)
        h✝ : Not (Membership.mem ℬ (Insert.insert a t))
        ⊢ LE.le 0 (HMul.hMul (collapse (HasInfs.infs 𝒜 ℬ) a f₃ (Inter.inter s t)) (col …
      -/
      exact mul_nonneg (collapse_nonneg h₃ _) <| collapse_nonneg h₄ _
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Not (Membership.mem 𝒜 s)
      h✝ : Not (Membership.mem 𝒜 (Insert.insert a s))
      ⊢ LE.le (HMul.hMul (HAdd.hAdd 0 0) (collapse ℬ a f₂ t)) (HMul.hMul (collapse ( …
    -/
  · simp_rw [add_zero, zero_mul]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      a : α
      f₁ f₂ f₃ f₄ : Finset α → β
      u : Finset α
      inst✝ : ExistsAddOfLE β
      hu : Not (Membership.mem u a)
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
      𝒜 ℬ : Finset (Finset α)
      s : Finset α
      hsu : HasSubset.Subset s u
      t : Finset α
      htu : HasSubset.Subset t u
      this✝⁴ : HasSubset.Subset s (Insert.insert a u)
      this✝³ : HasSubset.Subset t (Insert.insert a u)
      this✝² : HasSubset.Subset (Insert.insert a s) (Insert.insert a u)
      this✝¹ : HasSubset.Subset (Insert.insert a t) (Insert.insert a u)
      has : Not (Membership.mem s a)
      hat : Not (Membership.mem t a)
      this✝ : Not (Membership.mem (Inter.inter s t) a)
      this : Not (Membership.mem (Union.union s t) a)
      h✝¹ : Not (Membership.mem 𝒜 s)
      h✝ : Not (Membership.mem 𝒜 (Insert.insert a s))
      ⊢ LE.le 0 (HMul.hMul (collapse (HasInfs.infs 𝒜 ℬ) a f₃ (Inter.inter s t)) (col …
    -/
    exact mul_nonneg (collapse_nonneg h₃ _) <| collapse_nonneg h₄ _
    /-
      🎉 no goals
    -/


lemma sum_collapse (h𝒜 : 𝒜 ⊆ (insert a u).powerset) (hu : a ∉ u) :
    ∑ s ∈ u.powerset, collapse 𝒜 a f s = ∑ s ∈ 𝒜, f s := by
  calc
    _ = ∑ s ∈ u.powerset ∩ 𝒜, f s + ∑ s ∈ u.powerset.image (insert a) ∩ 𝒜, f s := ?_
    _ = ∑ s ∈ u.powerset ∩ 𝒜, f s + ∑ s ∈ ((insert a u).powerset \ u.powerset) ∩ 𝒜, f s := ?_
    _ = ∑ s ∈ 𝒜, f s := ?_
    /-
      case calc_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      u : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
      hu : Not (Membership.mem u a)
      ⊢ Eq (u.powerset.sum fun s => collapse 𝒜 a f s) (HAdd.hAdd ((Inter.inter u.pow …
    -/
  · rw [← Finset.sum_ite_mem, ← Finset.sum_ite_mem, sum_image, ← sum_add_distrib]
      /-
        case calc_1
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        ⊢ Eq (u.powerset.sum fun s => collapse 𝒜 a f s) (u.powerset.sum fun x => HAdd. …
      -/
    · exact sum_congr rfl fun s hs ↦ collapse_eq (not_mem_mono (mem_powerset.1 hs) hu) _ _
      /-
        🎉 no goals
      -/
      /-
        case calc_1
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        ⊢ ∀ (x : Finset α), Membership.mem u.powerset x → ∀ (y : Finset α), Membership …
      -/
    · exact (insert_erase_invOn.2.injOn).mono fun s hs ↦ not_mem_mono (mem_powerset.1 hs) hu
      /-
        🎉 no goals
      -/
    /-
      case calc_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      u : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
      hu : Not (Membership.mem u a)
      ⊢ Eq (HAdd.hAdd ((Inter.inter u.powerset 𝒜).sum fun s => f s) ((Inter.inter (F …
    -/
  · congr with s
    /-
      case calc_2.e_a.e_s.e_a.h
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      u : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
      hu : Not (Membership.mem u a)
      s : Finset α
      ⊢ Iff (Membership.mem (Finset.image (Insert.insert a) u.powerset) s) (Membersh …
    -/
    simp only [mem_image, mem_powerset, mem_sdiff, subset_insert_iff]
    /-
      case calc_2.e_a.e_s.e_a.h
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : LinearOrderedCommSemiring β
      𝒜 : Finset (Finset α)
      a : α
      f : Finset α → β
      u : Finset α
      h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
      hu : Not (Membership.mem u a)
      s : Finset α
      ⊢ Iff (Exists fun a_1 => And (HasSubset.Subset a_1 u) (Eq (Insert.insert a a_1 …
    -/
    refine ⟨?_, fun h ↦ ⟨_, h.1, ?_⟩⟩
      /-
        case calc_2.e_a.e_s.e_a.h.refine_1
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        s : Finset α
        ⊢ (Exists fun a_1 => And (HasSubset.Subset a_1 u) (Eq (Insert.insert a a_1) s) …
      -/
    · rintro ⟨s, hs, rfl⟩
      exact ⟨subset_insert_iff.1 <| insert_subset_insert _ hs, fun h ↦
        hu <| h <| mem_insert_self _ _⟩
      /-
        case calc_2.e_a.e_s.e_a.h.refine_2
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        s : Finset α
        h : And (HasSubset.Subset (s.erase a) u) (Not (HasSubset.Subset s u))
        ⊢ Eq (Insert.insert a (s.erase a)) s
      -/
    · rw [insert_erase (erase_ne_self.1 fun hs ↦ ?_)]
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        s : Finset α
        h : And (HasSubset.Subset (s.erase a) u) (Not (HasSubset.Subset s u))
        hs : Eq (s.erase a) s
        ⊢ False
      -/
      rw [hs] at h
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : LinearOrderedCommSemiring β
        𝒜 : Finset (Finset α)
        a : α
        f : Finset α → β
        u : Finset α
        h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
        hu : Not (Membership.mem u a)
        s : Finset α
        h : And (HasSubset.Subset s u) (Not (HasSubset.Subset s u))
        hs : Eq (s.erase a) s
        ⊢ False
      -/
      exact h.2 h.1
      /-
        🎉 no goals
      -/
  · rw [← sum_union (disjoint_sdiff_self_right.mono inf_le_left inf_le_left),
      ← union_inter_distrib_right, union_sdiff_of_subset (powerset_mono.2 <| subset_insert _ _),
      inter_eq_right.2 h𝒜]


/-- The **Four Functions Theorem** on a powerset algebra. See `four_functions_theorem` for the
finite distributive lattice generalisation. -/
protected lemma Finset.four_functions_theorem (u : Finset α)
    (h₁ : 0 ≤ f₁) (h₂ : 0 ≤ f₂) (h₃ : 0 ≤ f₃) (h₄ : 0 ≤ f₄)
    (h : ∀ ⦃s⦄, s ⊆ u → ∀ ⦃t⦄, t ⊆ u → f₁ s * f₂ t ≤ f₃ (s ∩ t) * f₄ (s ∪ t))
    {𝒜 ℬ : Finset (Finset α)} (h𝒜 : 𝒜 ⊆ u.powerset) (hℬ : ℬ ⊆ u.powerset) :
    (∑ s ∈ 𝒜, f₁ s) * ∑ s ∈ ℬ, f₂ s ≤ (∑ s ∈ 𝒜 ⊼ ℬ, f₃ s) * ∑ s ∈ 𝒜 ⊻ ℬ, f₄ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    f₁ f₂ f₃ f₄ : Finset α → β
    inst✝ : ExistsAddOfLE β
    u : Finset α
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s u → ∀ ⦃t : Finset α⦄, HasSubset.Subse …
    𝒜 ℬ : Finset (Finset α)
    h𝒜 : HasSubset.Subset 𝒜 u.powerset
    hℬ : HasSubset.Subset ℬ u.powerset
    ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
  -/
  induction' u using Finset.induction with a u hu ih generalizing f₁ f₂ f₃ f₄ 𝒜 ℬ
    /-
      case empty
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      inst✝ : ExistsAddOfLE β
      f₁ f₂ f₃ f₄ : Finset α → β
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s EmptyCollection.emptyCollection → ∀ ⦃ …
      𝒜 ℬ : Finset (Finset α)
      h𝒜 : HasSubset.Subset 𝒜 EmptyCollection.emptyCollection.powerset
      hℬ : HasSubset.Subset ℬ EmptyCollection.emptyCollection.powerset
      ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
    -/
  · simp only [Finset.powerset_empty, Finset.subset_singleton_iff] at h𝒜 hℬ
    /-
      case empty
      α : Type u_1
      β : Type u_2
      inst✝² : DecidableEq α
      inst✝¹ : LinearOrderedCommSemiring β
      inst✝ : ExistsAddOfLE β
      f₁ f₂ f₃ f₄ : Finset α → β
      h₁ : LE.le 0 f₁
      h₂ : LE.le 0 f₂
      h₃ : LE.le 0 f₃
      h₄ : LE.le 0 f₄
      h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s EmptyCollection.emptyCollection → ∀ ⦃ …
      𝒜 ℬ : Finset (Finset α)
      h𝒜 : Or (Eq 𝒜 EmptyCollection.emptyCollection) (Eq 𝒜 (Singleton.singleton Empt …
      hℬ : Or (Eq ℬ EmptyCollection.emptyCollection) (Eq ℬ (Singleton.singleton Empt …
      ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
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
    obtain rfl | rfl := h𝒜 <;> obtain rfl | rfl := hℬ <;> simp; exact h (subset_refl ∅) subset_rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/
  specialize ih (collapse_nonneg h₁) (collapse_nonneg h₂) (collapse_nonneg h₃) (collapse_nonneg h₄)
    (collapse_modular hu h₁ h₂ h₃ h₄ h 𝒜 ℬ) Subset.rfl Subset.rfl
  /-
    case insert
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    inst✝ : ExistsAddOfLE β
    a : α
    u : Finset α
    hu : Not (Membership.mem u a)
    f₁ f₂ f₃ f₄ : Finset α → β
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
    hℬ : HasSubset.Subset ℬ (Insert.insert a u).powerset
    ih : LE.le (HMul.hMul (u.powerset.sum fun s => collapse 𝒜 a f₁ s) (u.powerset. …
    ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
  -/
  have : 𝒜 ⊼ ℬ ⊆ powerset (insert a u) := by simpa using infs_subset h𝒜 hℬ
  /-
    case insert
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : LinearOrderedCommSemiring β
    inst✝ : ExistsAddOfLE β
    a : α
    u : Finset α
    hu : Not (Membership.mem u a)
    f₁ f₂ f₃ f₄ : Finset α → β
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ ⦃s : Finset α⦄, HasSubset.Subset s (Insert.insert a u) → ∀ ⦃t : Finset α …
    𝒜 ℬ : Finset (Finset α)
    h𝒜 : HasSubset.Subset 𝒜 (Insert.insert a u).powerset
    hℬ : HasSubset.Subset ℬ (Insert.insert a u).powerset
    ih : LE.le (HMul.hMul (u.powerset.sum fun s => collapse 𝒜 a f₁ s) (u.powerset. …
    this : HasSubset.Subset (HasInfs.infs 𝒜 ℬ) (Insert.insert a u).powerset
    ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
  -/
  have : 𝒜 ⊻ ℬ ⊆ powerset (insert a u) := by simpa using sups_subset h𝒜 hℬ
  simpa only [powerset_sups_powerset_self, powerset_infs_powerset_self, sum_collapse,
    not_false_eq_true, *] using ih


private lemma four_functions_theorem_aux (h₁ : 0 ≤ f₁) (h₂ : 0 ≤ f₂) (h₃ : 0 ≤ f₃) (h₄ : 0 ≤ f₄)
    (h : ∀ s t, f₁ s * f₂ t ≤ f₃ (s ∩ t) * f₄ (s ∪ t)) (𝒜 ℬ : Finset (Finset α)) :
    (∑ s ∈ 𝒜, f₁ s) * ∑ s ∈ ℬ, f₂ s ≤ (∑ s ∈ 𝒜 ⊼ ℬ, f₃ s) * ∑ s ∈ 𝒜 ⊻ ℬ, f₄ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : LinearOrderedCommSemiring β
    f₁ f₂ f₃ f₄ : Finset α → β
    inst✝¹ : ExistsAddOfLE β
    inst✝ : Fintype α
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ (s t : Finset α), LE.le (HMul.hMul (f₁ s) (f₂ t)) (HMul.hMul (f₃ (Inter. …
    𝒜 ℬ : Finset (Finset α)
    ⊢ LE.le (HMul.hMul (𝒜.sum fun s => f₁ s) (ℬ.sum fun s => f₂ s)) (HMul.hMul ((H …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  refine univ.four_functions_theorem h₁ h₂ h₃ h₄ ?_ ?_ ?_ <;> simp [h]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- The **Four Functions Theorem**, aka **Ahlswede-Daykin Inequality**. -/
lemma four_functions_theorem [DecidableEq α] (h₁ : 0 ≤ f₁) (h₂ : 0 ≤ f₂) (h₃ : 0 ≤ f₃) (h₄ : 0 ≤ f₄)
    (h : ∀ a b, f₁ a * f₂ b ≤ f₃ (a ⊓ b) * f₄ (a ⊔ b)) (s t : Finset α) :
    (∑ a ∈ s, f₁ a) * ∑ a ∈ t, f₂ a ≤ (∑ a ∈ s ⊼ t, f₃ a) * ∑ a ∈ s ⊻ t, f₄ a := by
  classical
  set L : Sublattice α := ⟨latticeClosure (s ∪ t), isSublattice_latticeClosure.1,
    isSublattice_latticeClosure.2⟩
  have : Finite L := (s.finite_toSet.union t.finite_toSet).latticeClosure.to_subtype
  set s' : Finset L := s.preimage (↑) Subtype.coe_injective.injOn
  set t' : Finset L := t.preimage (↑) Subtype.coe_injective.injOn
  have hs' : s'.map ⟨L.subtype, Subtype.coe_injective⟩ = s := by
    simp [s', map_eq_image, image_preimage, filter_eq_self]
    exact fun a ha ↦ subset_latticeClosure <| Set.subset_union_left ha
  have ht' : t'.map ⟨L.subtype, Subtype.coe_injective⟩ = t := by
    simp [t', map_eq_image, image_preimage, filter_eq_self]
    exact fun a ha ↦ subset_latticeClosure <| Set.subset_union_right ha
  clear_value s' t'
  obtain ⟨β, _, _, g, hg⟩ := exists_birkhoff_representation L
  have := four_functions_theorem_aux (extend g (f₁ ∘ (↑)) 0) (extend g (f₂ ∘ (↑)) 0)
    (extend g (f₃ ∘ (↑)) 0) (extend g (f₄ ∘ (↑)) 0) (extend_nonneg (fun _ ↦ h₁ _) le_rfl)
    (extend_nonneg (fun _ ↦ h₂ _) le_rfl) (extend_nonneg (fun _ ↦ h₃ _) le_rfl)
    (extend_nonneg (fun _ ↦ h₄ _) le_rfl) ?_ (s'.map ⟨g, hg⟩) (t'.map ⟨g, hg⟩)
  · simpa only [← hs', ← ht', ← map_sups, ← map_infs, sum_map, Embedding.coeFn_mk, hg.extend_apply]
      using this
  rintro s t
  classical
  obtain ⟨a, rfl⟩ | hs := em (∃ a, g a = s)
  · obtain ⟨b, rfl⟩ | ht := em (∃ b, g b = t)
    · simp_rw [← sup_eq_union, ← inf_eq_inter, ← map_sup, ← map_inf, hg.extend_apply]
      exact h _ _
    · simpa [extend_apply' _ _ _ ht] using mul_nonneg
        (extend_nonneg (fun a : L ↦ h₃ a) le_rfl _) (extend_nonneg (fun a : L ↦ h₄ a) le_rfl _)
  · simpa [extend_apply' _ _ _ hs] using mul_nonneg
      (extend_nonneg (fun a : L ↦ h₃ a) le_rfl _) (extend_nonneg (fun a : L ↦ h₄ a) le_rfl _)


/-- An inequality of Daykin. Interestingly, any lattice in which this inequality holds is
distributive. -/
lemma Finset.le_card_infs_mul_card_sups [DecidableEq α] (s t : Finset α) :
    #s * #t ≤ #(s ⊼ t) * #(s ⊻ t) := by
  simpa using four_functions_theorem (1 : α → ℕ) 1 1 1 zero_le_one zero_le_one zero_le_one
    zero_le_one (fun _ _ ↦ le_rfl) s t


/-- Special case of the **Four Functions Theorem** when `s = t = univ`. -/
lemma four_functions_theorem_univ (h₁ : 0 ≤ f₁) (h₂ : 0 ≤ f₂) (h₃ : 0 ≤ f₃) (h₄ : 0 ≤ f₄)
    (h : ∀ a b, f₁ a * f₂ b ≤ f₃ (a ⊓ b) * f₄ (a ⊔ b)) :
    (∑ a, f₁ a) * ∑ a, f₂ a ≤ (∑ a, f₃ a) * ∑ a, f₄ a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DistribLattice α
    inst✝² : LinearOrderedCommSemiring β
    inst✝¹ : ExistsAddOfLE β
    f₁ f₂ f₃ f₄ : α → β
    inst✝ : Fintype α
    h₁ : LE.le 0 f₁
    h₂ : LE.le 0 f₂
    h₃ : LE.le 0 f₃
    h₄ : LE.le 0 f₄
    h : ∀ (a b : α), LE.le (HMul.hMul (f₁ a) (f₂ b)) (HMul.hMul (f₃ (Min.min a b)) …
    ⊢ LE.le (HMul.hMul (Finset.univ.sum fun a => f₁ a) (Finset.univ.sum fun a => f …
  -/
  classical simpa using four_functions_theorem f₁ f₂ f₃ f₄ h₁ h₂ h₃ h₄ h univ univ
  /-
    🎉 no goals
  -/


/-- The **Holley Inequality**. -/
lemma holley (hμ₀ : 0 ≤ μ) (hf : 0 ≤ f) (hg : 0 ≤ g) (hμ : Monotone μ)
    (hfg : ∑ a, f a = ∑ a, g a) (h : ∀ a b, f a * g b ≤ f (a ⊓ b) * g (a ⊔ b)) :
    ∑ a, μ a * f a ≤ ∑ a, μ a * g a := by
  classical
  obtain rfl | hf := hf.eq_or_lt
  · simp only [Pi.zero_apply, sum_const_zero, eq_comm, Fintype.sum_eq_zero_iff_of_nonneg hg] at hfg
    simp [hfg]
  obtain rfl | hg := hg.eq_or_lt
  · simp only [Pi.zero_apply, sum_const_zero, Fintype.sum_eq_zero_iff_of_nonneg hf.le] at hfg
    simp [hfg]
  have := four_functions_theorem g (μ * f) f (μ * g) hg.le (mul_nonneg hμ₀ hf.le) hf.le
    (mul_nonneg hμ₀ hg.le) (fun a b ↦ ?_) univ univ
  · simpa [hfg, sum_pos hg] using this
  · simp_rw [Pi.mul_apply, mul_left_comm _ (μ _), mul_comm (g _)]
    rw [sup_comm, inf_comm]
    exact mul_le_mul (hμ le_sup_left) (h _ _) (mul_nonneg (hf.le _) <| hg.le _) <| hμ₀ _


/-- The **Fortuin-Kastelyn-Ginibre Inequality**. -/
lemma fkg (hμ₀ : 0 ≤ μ) (hf₀ : 0 ≤ f) (hg₀ : 0 ≤ g) (hf : Monotone f) (hg : Monotone g)
    (hμ : ∀ a b, μ a * μ b ≤ μ (a ⊓ b) * μ (a ⊔ b)) :
    (∑ a, μ a * f a) * ∑ a, μ a * g a ≤ (∑ a, μ a) * ∑ a, μ a * (f a * g a) := by
  refine four_functions_theorem_univ (μ * f) (μ * g) μ _ (mul_nonneg hμ₀ hf₀) (mul_nonneg hμ₀ hg₀)
    hμ₀ (mul_nonneg hμ₀ <| mul_nonneg hf₀ hg₀) (fun a b ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DistribLattice α
    inst✝² : LinearOrderedCommSemiring β
    inst✝¹ : ExistsAddOfLE β
    f g μ : α → β
    inst✝ : Fintype α
    hμ₀ : LE.le 0 μ
    hf₀ : LE.le 0 f
    hg₀ : LE.le 0 g
    hf : Monotone f
    hg : Monotone g
    hμ : ∀ (a b : α), LE.le (HMul.hMul (μ a) (μ b)) (HMul.hMul (μ (Min.min a b)) ( …
    a b : α
    ⊢ LE.le (HMul.hMul (HMul.hMul μ f a) (HMul.hMul μ g b)) (HMul.hMul (μ (Min.min …
  -/
  dsimp
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : DistribLattice α
    inst✝² : LinearOrderedCommSemiring β
    inst✝¹ : ExistsAddOfLE β
    f g μ : α → β
    inst✝ : Fintype α
    hμ₀ : LE.le 0 μ
    hf₀ : LE.le 0 f
    hg₀ : LE.le 0 g
    hf : Monotone f
    hg : Monotone g
    hμ : ∀ (a b : α), LE.le (HMul.hMul (μ a) (μ b)) (HMul.hMul (μ (Min.min a b)) ( …
    a b : α
    ⊢ LE.le (HMul.hMul (HMul.hMul (μ a) (f a)) (HMul.hMul (μ b) (g b))) (HMul.hMul …
  -/
  rw [mul_mul_mul_comm, ← mul_assoc (μ (a ⊓ b))]
  exact mul_le_mul (hμ _ _) (mul_le_mul (hf le_sup_left) (hg le_sup_right) (hg₀ _) <| hf₀ _)
    (mul_nonneg (hf₀ _) <| hg₀ _) <| mul_nonneg (hμ₀ _) <| hμ₀ _


/-- A slight generalisation of the **Marica-Schönheim Inequality**. -/
lemma Finset.le_card_diffs_mul_card_diffs (s t : Finset α) :
    #s * #t ≤ #(s \\ t) * #(t \\ s) := by
  have : ∀ s t : Finset α, (s \\ t).map ⟨_, liftLatticeHom_injective⟩ =
      s.map ⟨_, liftLatticeHom_injective⟩ \\ t.map ⟨_, liftLatticeHom_injective⟩ := by
    rintro s t
    simp_rw [map_eq_image]
    exact image_image₂_distrib fun a b ↦ rfl
  simpa [← card_compls (_ ⊻ _), ← map_sup, ← map_inf, ← this] using
    (s.map ⟨_, liftLatticeHom_injective⟩).le_card_infs_mul_card_sups
      (t.map ⟨_, liftLatticeHom_injective⟩)ᶜˢ


/-- The **Marica-Schönheim Inequality**. -/
lemma Finset.card_le_card_diffs (s : Finset α) : #s ≤ #(s \\ s) :=
  le_of_pow_le_pow_left₀ two_ne_zero (zero_le _) <| by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : GeneralizedBooleanAlgebra α
      s : Finset α
      ⊢ LE.le (HPow.hPow s.card 2) (HPow.hPow (s.diffs s).card 2)
    -/
    simpa [← sq] using s.le_card_diffs_mul_card_diffs s
    /-
      🎉 no goals
    -/

