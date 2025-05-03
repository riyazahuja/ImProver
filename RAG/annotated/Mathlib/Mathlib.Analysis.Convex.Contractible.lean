/-- A non-empty star convex set is a contractible space. -/
protected theorem StarConvex.contractibleSpace (h : StarConvex ℝ x s) (hne : s.Nonempty) :
    ContractibleSpace s := by
  refine
    (contractible_iff_id_nullhomotopic s).2
      ⟨⟨x, h.mem hne⟩,
        ⟨⟨⟨fun p => ⟨p.1.1 • x + (1 - p.1.1) • (p.2 : E), ?_⟩, ?_⟩, fun x => ?_, fun x => ?_⟩⟩⟩
    /-
      case refine_1
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      s : Set E
      x : E
      h : StarConvex Real x s
      hne : s.Nonempty
      p : Prod ↑unitInterval ↑s
      ⊢ Membership.mem s (HAdd.hAdd (HSMul.hSMul (↑p.1) x) (HSMul.hSMul (HSub.hSub 1 …
    -/
  · exact h p.2.2 p.1.2.1 (sub_nonneg.2 p.1.2.2) (add_sub_cancel _ _)
    /-
      🎉 no goals
    -/
  · exact
      ((continuous_subtype_val.fst'.smul continuous_const).add
            ((continuous_const.sub continuous_subtype_val.fst').smul
              continuous_subtype_val.snd')).subtype_mk
        _
    /-
      case refine_3
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      s : Set E
      x✝ : E
      h : StarConvex Real x✝ s
      hne : s.Nonempty
      x : ↑s
      ⊢ Eq ({ toFun := fun p => ⟨HAdd.hAdd (HSMul.hSMul (↑p.1) x✝) (HSMul.hSMul (HSu …
    -/
  · ext1
    /-
      case refine_3.a
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      s : Set E
      x✝ : E
      h : StarConvex Real x✝ s
      hne : s.Nonempty
      x : ↑s
      ⊢ Eq ↑({ toFun := fun p => ⟨HAdd.hAdd (HSMul.hSMul (↑p.1) x✝) (HSMul.hSMul (HS …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      s : Set E
      x✝ : E
      h : StarConvex Real x✝ s
      hne : s.Nonempty
      x : ↑s
      ⊢ Eq ({ toFun := fun p => ⟨HAdd.hAdd (HSMul.hSMul (↑p.1) x✝) (HSMul.hSMul (HSu …
    -/
  · ext1
    /-
      case refine_4.a
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      s : Set E
      x✝ : E
      h : StarConvex Real x✝ s
      hne : s.Nonempty
      x : ↑s
      ⊢ Eq ↑({ toFun := fun p => ⟨HAdd.hAdd (HSMul.hSMul (↑p.1) x✝) (HSMul.hSMul (HS …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A non-empty convex set is a contractible space. -/
protected theorem Convex.contractibleSpace (hs : Convex ℝ s) (hne : s.Nonempty) :
    ContractibleSpace s :=
  let ⟨_, hx⟩ := hne
  (hs.starConvex hx).contractibleSpace hne


instance (priority := 100) RealTopologicalVectorSpace.contractibleSpace : ContractibleSpace E :=
  (Homeomorph.Set.univ E).contractibleSpace_iff.mp <|
    convex_univ.contractibleSpace Set.univ_nonempty

