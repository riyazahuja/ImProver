instance : TopologicalSpace SignType :=
  ⊥


instance : DiscreteTopology SignType :=
  ⟨rfl⟩


theorem continuousAt_sign_of_pos {a : α} (h : 0 < a) : ContinuousAt SignType.sign a := by
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt 0 a
    ⊢ ContinuousAt (⇑SignType.sign) a
  -/
  refine (continuousAt_const : ContinuousAt (fun _ => (1 : SignType)) a).congr ?_
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt 0 a
    ⊢ (nhds a).EventuallyEq (fun x => 1) ⇑SignType.sign
  -/
  rw [Filter.EventuallyEq, eventually_nhds_iff]
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt 0 a
    ⊢ Exists fun t => And (∀ (y : α), Membership.mem t y → Eq 1 (SignType.sign y)) …
  -/
  exact ⟨{ x | 0 < x }, fun x hx => (sign_pos hx).symm, isOpen_lt' 0, h⟩
  /-
    🎉 no goals
  -/


theorem continuousAt_sign_of_neg {a : α} (h : a < 0) : ContinuousAt SignType.sign a := by
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt a 0
    ⊢ ContinuousAt (⇑SignType.sign) a
  -/
  refine (continuousAt_const : ContinuousAt (fun x => (-1 : SignType)) a).congr ?_
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt a 0
    ⊢ (nhds a).EventuallyEq (fun x => -1) ⇑SignType.sign
  -/
  rw [Filter.EventuallyEq, eventually_nhds_iff]
  /-
    α : Type u_1
    inst✝⁴ : Zero α
    inst✝³ : TopologicalSpace α
    inst✝² : PartialOrder α
    inst✝¹ : DecidableRel fun x1 x2 => LT.lt x1 x2
    inst✝ : OrderTopology α
    a : α
    h : LT.lt a 0
    ⊢ Exists fun t => And (∀ (y : α), Membership.mem t y → Eq (-1) (SignType.sign  …
  -/
  exact ⟨{ x | x < 0 }, fun x hx => (sign_neg hx).symm, isOpen_gt' 0, h⟩
  /-
    🎉 no goals
  -/


theorem continuousAt_sign_of_ne_zero {a : α} (h : a ≠ 0) : ContinuousAt SignType.sign a := by
  /-
    α : Type u_1
    inst✝³ : Zero α
    inst✝² : TopologicalSpace α
    inst✝¹ : LinearOrder α
    inst✝ : OrderTopology α
    a : α
    h : Ne a 0
    ⊢ ContinuousAt (⇑SignType.sign) a
  -/
  rcases h.lt_or_lt with (h_neg | h_pos)
    /-
      case inl
      α : Type u_1
      inst✝³ : Zero α
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      h : Ne a 0
      h_neg : LT.lt a 0
      ⊢ ContinuousAt (⇑SignType.sign) a
    -/
  · exact continuousAt_sign_of_neg h_neg
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝³ : Zero α
      inst✝² : TopologicalSpace α
      inst✝¹ : LinearOrder α
      inst✝ : OrderTopology α
      a : α
      h : Ne a 0
      h_pos : LT.lt 0 a
      ⊢ ContinuousAt (⇑SignType.sign) a
    -/
  · exact continuousAt_sign_of_pos h_pos
    /-
      🎉 no goals
    -/


