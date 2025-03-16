@[to_additive]
instance Lp.instContinuousSMulDomMulAct : ContinuousSMul Mᵈᵐᵃ (Lp E p μ) where
  continuous_smul :=
    let g : C(Mᵈᵐᵃ × Lp E p μ, C(X, X)) :=
      (ContinuousMap.mk (fun a : M × X ↦ a.1 • a.2) continuous_smul).curry.comp <|
               /-
                 X : Type u_1
                 M : Type u_2
                 E : Type u_3
                 inst✝¹⁴ : TopologicalSpace X
                 inst✝¹³ : R1Space X
                 inst✝¹² : MeasurableSpace X
                 inst✝¹¹ : BorelSpace X
                 inst✝¹⁰ : Monoid M
                 inst✝⁹ : TopologicalSpace M
                 inst✝⁸ : MeasurableSpace M
                 inst✝⁷ : OpensMeasurableSpace M
                 inst✝⁶ : SMul M X
                 inst✝⁵ : ContinuousSMul M X
                 inst✝⁴ : NormedAddCommGroup E
                 μ : MeasureTheory.Measure X
                 inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
                 inst✝² : μ.InnerRegularCompactLTTop
                 inst✝¹ : MeasureTheory.SMulInvariantMeasure M X μ
                 p : ENNReal
                 inst✝ : Fact (LE.le 1 p)
                 hp : Fact (Ne p Top.top)
                 ⊢ Continuous ⇑DomMulAct.mk.symm
               -/
        .comp (.mk DomMulAct.mk.symm) ContinuousMap.fst
               /-
                 🎉 no goals
               -/
    continuous_snd.compMeasurePreservingLp g.continuous _ Fact.out


