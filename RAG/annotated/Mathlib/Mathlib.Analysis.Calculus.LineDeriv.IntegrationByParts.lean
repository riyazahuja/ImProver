lemma integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable_aux1 [SigmaFinite μ]
    {f f' : E × ℝ → F} {g g' : E × ℝ → G} {B : F →L[ℝ] G →L[ℝ] W}
    (hf'g : Integrable (fun x ↦ B (f' x) (g x)) (μ.prod volume))
    (hfg' : Integrable (fun x ↦ B (f x) (g' x)) (μ.prod volume))
    (hfg : Integrable (fun x ↦ B (f x) (g x)) (μ.prod volume))
    (hf : ∀ x, HasLineDerivAt ℝ f (f' x) x (0, 1)) (hg : ∀ x, HasLineDerivAt ℝ g (g' x) x (0, 1)) :
    ∫ x, B (f x) (g' x) ∂(μ.prod volume) = - ∫ x, B (f' x) (g x) ∂(μ.prod volume) := calc
  ∫ x, B (f x) (g' x) ∂(μ.prod volume)
    = ∫ x, (∫ t, B (f (x, t)) (g' (x, t))) ∂μ := integral_prod _ hfg'
  _ = ∫ x, (- ∫ t, B (f' (x, t)) (g (x, t))) ∂μ := by
    /-
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Real E
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace Real F
      inst✝⁵ : NormedAddCommGroup G
      inst✝⁴ : NormedSpace Real G
      inst✝³ : NormedAddCommGroup W
      inst✝² : NormedSpace Real W
      inst✝¹ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝ : MeasureTheory.SigmaFinite μ
      f f' : Prod E Real → F
      g g' : Prod E Real → G
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
      hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
      hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
      ⊢ Eq (MeasureTheory.integral μ fun x => MeasureTheory.integral MeasureTheory.M …
    -/
    apply integral_congr_ae
    filter_upwards [hf'g.prod_right_ae, hfg'.prod_right_ae, hfg.prod_right_ae]
      with x hf'gx hfg'x hfgx
    /-
      case h
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Real E
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : NormedSpace Real F
      inst✝⁵ : NormedAddCommGroup G
      inst✝⁴ : NormedSpace Real G
      inst✝³ : NormedAddCommGroup W
      inst✝² : NormedSpace Real W
      inst✝¹ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝ : MeasureTheory.SigmaFinite μ
      f f' : Prod E Real → F
      g g' : Prod E Real → G
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
      hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
      hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
      x : E
      hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
      hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
      hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun t => (B (f  …
    -/
    apply integral_bilinear_hasDerivAt_right_eq_neg_left_of_integrable ?_ ?_ hfg'x hf'gx hfgx
      /-
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Real E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace Real F
        inst✝⁵ : NormedAddCommGroup G
        inst✝⁴ : NormedSpace Real G
        inst✝³ : NormedAddCommGroup W
        inst✝² : NormedSpace Real W
        inst✝¹ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝ : MeasureTheory.SigmaFinite μ
        f f' : Prod E Real → F
        g g' : Prod E Real → G
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
        hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
        hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
        x : E
        hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
        hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
        hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
        ⊢ ∀ (x_1 : Real), HasDerivAt (fun x_2 => f { fst := x, snd := x_2 }) (f' { fst …
      -/
    · intro t
      /-
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Real E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace Real F
        inst✝⁵ : NormedAddCommGroup G
        inst✝⁴ : NormedSpace Real G
        inst✝³ : NormedAddCommGroup W
        inst✝² : NormedSpace Real W
        inst✝¹ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝ : MeasureTheory.SigmaFinite μ
        f f' : Prod E Real → F
        g g' : Prod E Real → G
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
        hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
        hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
        x : E
        hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
        hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
        hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
        t : Real
        ⊢ HasDerivAt (fun x_1 => f { fst := x, snd := x_1 }) (f' { fst := x, snd := t  …
      -/
      convert (hf (x, t)).scomp_of_eq t ((hasDerivAt_id t).add (hasDerivAt_const t (-t))) (by simp)
            /-
              case h.e'_8.h
              E : Type u_1
              F : Type u_2
              G : Type u_3
              W : Type u_4
              inst✝⁹ : NormedAddCommGroup E
              inst✝⁸ : NormedSpace Real E
              inst✝⁷ : NormedAddCommGroup F
              inst✝⁶ : NormedSpace Real F
              inst✝⁵ : NormedAddCommGroup G
              inst✝⁴ : NormedSpace Real G
              inst✝³ : NormedAddCommGroup W
              inst✝² : NormedSpace Real W
              inst✝¹ : MeasurableSpace E
              μ : MeasureTheory.Measure E
              inst✝ : MeasureTheory.SigmaFinite μ
              f f' : Prod E Real → F
              g g' : Prod E Real → G
              B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
              hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
              hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
              hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
              hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
              hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
              x : E
              hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
              hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
              hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
              t x✝ : Real
              ⊢ Eq (f { fst := x, snd := x✝ }) (Function.comp (fun t_1 => f (HAdd.hAdd { fst …
            -/
            /-
              🎉 no goals
            -/
        <;> simp
            /-
              🎉 no goals
            -/
      /-
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Real E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace Real F
        inst✝⁵ : NormedAddCommGroup G
        inst✝⁴ : NormedSpace Real G
        inst✝³ : NormedAddCommGroup W
        inst✝² : NormedSpace Real W
        inst✝¹ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝ : MeasureTheory.SigmaFinite μ
        f f' : Prod E Real → F
        g g' : Prod E Real → G
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
        hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
        hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
        x : E
        hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
        hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
        hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
        ⊢ ∀ (x_1 : Real), HasDerivAt (fun x_2 => g { fst := x, snd := x_2 }) (g' { fst …
      -/
    · intro t
      /-
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Real E
        inst✝⁷ : NormedAddCommGroup F
        inst✝⁶ : NormedSpace Real F
        inst✝⁵ : NormedAddCommGroup G
        inst✝⁴ : NormedSpace Real G
        inst✝³ : NormedAddCommGroup W
        inst✝² : NormedSpace Real W
        inst✝¹ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝ : MeasureTheory.SigmaFinite μ
        f f' : Prod E Real → F
        g g' : Prod E Real → G
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
        hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
        hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
        x : E
        hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
        hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
        hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
        t : Real
        ⊢ HasDerivAt (fun x_1 => g { fst := x, snd := x_1 }) (g' { fst := x, snd := t  …
      -/
      convert (hg (x, t)).scomp_of_eq t ((hasDerivAt_id t).add (hasDerivAt_const t (-t))) (by simp)
            /-
              case h.e'_8.h
              E : Type u_1
              F : Type u_2
              G : Type u_3
              W : Type u_4
              inst✝⁹ : NormedAddCommGroup E
              inst✝⁸ : NormedSpace Real E
              inst✝⁷ : NormedAddCommGroup F
              inst✝⁶ : NormedSpace Real F
              inst✝⁵ : NormedAddCommGroup G
              inst✝⁴ : NormedSpace Real G
              inst✝³ : NormedAddCommGroup W
              inst✝² : NormedSpace Real W
              inst✝¹ : MeasurableSpace E
              μ : MeasureTheory.Measure E
              inst✝ : MeasureTheory.SigmaFinite μ
              f f' : Prod E Real → F
              g g' : Prod E Real → G
              B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
              hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
              hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
              hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
              hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
              hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
              x : E
              hf'gx : MeasureTheory.Integrable (fun y => (B (f' { fst := x, snd := y })) (g  …
              hfg'x : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g'  …
              hfgx : MeasureTheory.Integrable (fun y => (B (f { fst := x, snd := y })) (g {  …
              t x✝ : Real
              ⊢ Eq (g { fst := x, snd := x✝ }) (Function.comp (fun t_1 => g (HAdd.hAdd { fst …
            -/
            /-
              🎉 no goals
            -/
        <;> simp
            /-
              🎉 no goals
            -/
                                                   /-
                                                     E : Type u_1
                                                     F : Type u_2
                                                     G : Type u_3
                                                     W : Type u_4
                                                     inst✝⁹ : NormedAddCommGroup E
                                                     inst✝⁸ : NormedSpace Real E
                                                     inst✝⁷ : NormedAddCommGroup F
                                                     inst✝⁶ : NormedSpace Real F
                                                     inst✝⁵ : NormedAddCommGroup G
                                                     inst✝⁴ : NormedSpace Real G
                                                     inst✝³ : NormedAddCommGroup W
                                                     inst✝² : NormedSpace Real W
                                                     inst✝¹ : MeasurableSpace E
                                                     μ : MeasureTheory.Measure E
                                                     inst✝ : MeasureTheory.SigmaFinite μ
                                                     f f' : Prod E Real → F
                                                     g g' : Prod E Real → G
                                                     B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
                                                     hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (μ.prod MeasureThe …
                                                     hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (μ.prod MeasureThe …
                                                     hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (μ.prod MeasureTheor …
                                                     hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
                                                     hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
                                                     ⊢ Eq (MeasureTheory.integral μ fun x => Neg.neg (MeasureTheory.integral Measur …
                                                   -/
  _ = - ∫ x, B (f' x) (g x) ∂(μ.prod volume) := by rw [integral_neg, integral_prod _ hf'g]
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable_aux2
    [FiniteDimensional ℝ E] {μ : Measure (E × ℝ)} [IsAddHaarMeasure μ]
    {f f' : E × ℝ → F} {g g' : E × ℝ → G} {B : F →L[ℝ] G →L[ℝ] W}
    (hf'g : Integrable (fun x ↦ B (f' x) (g x)) μ)
    (hfg' : Integrable (fun x ↦ B (f x) (g' x)) μ)
    (hfg : Integrable (fun x ↦ B (f x) (g x)) μ)
    (hf : ∀ x, HasLineDerivAt ℝ f (f' x) x (0, 1)) (hg : ∀ x, HasLineDerivAt ℝ g (g' x) x (0, 1)) :
    ∫ x, B (f x) (g' x) ∂μ = - ∫ x, B (f' x) (g x) ∂μ := by
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure (Prod E Real)
    inst✝ : μ.IsAddHaarMeasure
    f f' : Prod E Real → F
    g g' : Prod E Real → G
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
    hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  let ν : Measure E := addHaar
  have A : ν.prod volume = (addHaarScalarFactor (ν.prod volume) μ) • μ :=
    isAddLeftInvariant_eq_smul _ _
  have Hf'g : Integrable (fun x ↦ B (f' x) (g x)) (ν.prod volume) := by
    rw [A]; exact hf'g.smul_measure_nnreal
  have Hfg' : Integrable (fun x ↦ B (f x) (g' x)) (ν.prod volume) := by
    rw [A]; exact hfg'.smul_measure_nnreal
  have Hfg : Integrable (fun x ↦ B (f x) (g x)) (ν.prod volume) := by
    rw [A]; exact hfg.smul_measure_nnreal
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure (Prod E Real)
    inst✝ : μ.IsAddHaarMeasure
    f f' : Prod E Real → F
    g g' : Prod E Real → G
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
    hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
    ν : MeasureTheory.Measure E := MeasureTheory.Measure.addHaar
    A : Eq (ν.prod MeasureTheory.MeasureSpace.volume) (HSMul.hSMul ((ν.prod Measur …
    Hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (ν.prod MeasureThe …
    Hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (ν.prod MeasureThe …
    Hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (ν.prod MeasureTheor …
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  rw [isAddLeftInvariant_eq_smul μ (ν.prod volume)]
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    μ : MeasureTheory.Measure (Prod E Real)
    inst✝ : μ.IsAddHaarMeasure
    f f' : Prod E Real → F
    g g' : Prod E Real → G
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : Prod E Real), HasLineDerivAt Real f (f' x) x { fst := 0, snd := 1 }
    hg : ∀ (x : Prod E Real), HasLineDerivAt Real g (g' x) x { fst := 0, snd := 1 }
    ν : MeasureTheory.Measure E := MeasureTheory.Measure.addHaar
    A : Eq (ν.prod MeasureTheory.MeasureSpace.volume) (HSMul.hSMul ((ν.prod Measur …
    Hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) (ν.prod MeasureThe …
    Hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) (ν.prod MeasureThe …
    Hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) (ν.prod MeasureTheor …
    ⊢ Eq (MeasureTheory.integral (HSMul.hSMul (μ.addHaarScalarFactor (ν.prod Measu …
  -/
  simp [integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable_aux1 Hf'g Hfg' Hfg hf hg]
  /-
    🎉 no goals
  -/


/-- **Integration by parts for line derivatives**
Version with a general bilinear form `B`.
If `B f g` is integrable, as well as `B f' g` and `B f g'` where `f'` and `g'` are derivatives
of `f` and `g` in a given direction `v`, then `∫ B f g' = - ∫ B f' g`. -/
theorem integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable
    {f f' : E → F} {g g' : E → G} {v : E} {B : F →L[ℝ] G →L[ℝ] W}
    (hf'g : Integrable (fun x ↦ B (f' x) (g x)) μ) (hfg' : Integrable (fun x ↦ B (f x) (g' x)) μ)
    (hfg : Integrable (fun x ↦ B (f x) (g x)) μ)
    (hf : ∀ x, HasLineDerivAt ℝ f (f' x) x v) (hg : ∀ x, HasLineDerivAt ℝ g (g' x) x v) :
    ∫ x, B (f x) (g' x) ∂μ = - ∫ x, B (f' x) (g x) ∂μ := by
  /-
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  by_cases hW : CompleteSpace W; swap
    /-
      case neg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : Not (CompleteSpace W)
      ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
    -/
  · simp [integral, hW]
    /-
      🎉 no goals
    -/
  /-
    case pos
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  rcases eq_or_ne v 0 with rfl|hv
  · have Hf' x : f' x = 0 := by
      simpa [(hasLineDerivAt_zero (f := f) (x := x)).lineDeriv] using (hf x).lineDeriv.symm
    have Hg' x : g' x = 0 := by
      simpa [(hasLineDerivAt_zero (f := g) (x := x)).lineDeriv] using (hg x).lineDeriv.symm
    /-
      case pos.inl
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hW : CompleteSpace W
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x 0
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x 0
      Hf' : ∀ (x : E), Eq (f' x) 0
      Hg' : ∀ (x : E), Eq (g' x) 0
      ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
    -/
    simp [Hf', Hg']
    /-
      🎉 no goals
    -/
  /-
    case pos.inr
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  have : Nontrivial E := nontrivial_iff.2 ⟨v, 0, hv⟩
  /-
    case pos.inr
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    this : Nontrivial E
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  let n := finrank ℝ E
  /-
    case pos.inr
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    this : Nontrivial E
    n : Nat := Module.finrank Real E
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  let E' := Fin (n - 1) → ℝ
  obtain ⟨L, hL⟩ : ∃ L : E ≃L[ℝ] (E' × ℝ), L v = (0, 1) := by
    have : finrank ℝ (E' × ℝ) = n := by simpa [this, E'] using Nat.sub_add_cancel finrank_pos
    have L₀ : E ≃L[ℝ] (E' × ℝ) := (ContinuousLinearEquiv.ofFinrankEq this).symm
    obtain ⟨M, hM⟩ : ∃ M : (E' × ℝ) ≃L[ℝ] (E' × ℝ), M (L₀ v) = (0, 1) := by
      apply SeparatingDual.exists_continuousLinearEquiv_apply_eq
      · simpa using hv
      · simp
    exact ⟨L₀.trans M, by simp [hM]⟩
  /-
    case pos.inr.intro
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    this : Nontrivial E
    n : Nat := Module.finrank Real E
    E' : Type := Fin (HSub.hSub n 1) → Real
    L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
    hL : Eq (L v) { fst := 0, snd := 1 }
    ⊢ Eq (MeasureTheory.integral μ fun x => (B (f x)) (g' x)) (Neg.neg (MeasureThe …
  -/
  let ν := Measure.map L μ
  suffices H : ∫ (x : E' × ℝ), (B (f (L.symm x))) (g' (L.symm x)) ∂ν =
      -∫ (x : E' × ℝ), (B (f' (L.symm x))) (g (L.symm x)) ∂ν by
    have : μ = Measure.map L.symm ν := by
      simp [ν, Measure.map_map L.symm.continuous.measurable L.continuous.measurable]
    have hL : IsClosedEmbedding L.symm := L.symm.toHomeomorph.isClosedEmbedding
    simpa [this, hL.integral_map] using H
  /-
    case pos.inr.intro
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    this : Nontrivial E
    n : Nat := Module.finrank Real E
    E' : Type := Fin (HSub.hSub n 1) → Real
    L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
    hL : Eq (L v) { fst := 0, snd := 1 }
    ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
    ⊢ Eq (MeasureTheory.integral ν fun x => (B (f (L.symm x))) (g' (L.symm x))) (N …
  -/
  have L_emb : MeasurableEmbedding L := L.toHomeomorph.measurableEmbedding
  /-
    case pos.inr.intro
    E : Type u_1
    F : Type u_2
    G : Type u_3
    W : Type u_4
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Real E
    inst✝⁹ : NormedAddCommGroup F
    inst✝⁸ : NormedSpace Real F
    inst✝⁷ : NormedAddCommGroup G
    inst✝⁶ : NormedSpace Real G
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    inst✝³ : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝² : BorelSpace E
    inst✝¹ : FiniteDimensional Real E
    inst✝ : μ.IsAddHaarMeasure
    f f' : E → F
    g g' : E → G
    v : E
    B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
    hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
    hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
    hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
    hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
    hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
    hW : CompleteSpace W
    hv : Ne v 0
    this : Nontrivial E
    n : Nat := Module.finrank Real E
    E' : Type := Fin (HSub.hSub n 1) → Real
    L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
    hL : Eq (L v) { fst := 0, snd := 1 }
    ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
    L_emb : MeasurableEmbedding ⇑L
    ⊢ Eq (MeasureTheory.integral ν fun x => (B (f (L.symm x))) (g' (L.symm x))) (N …
  -/
  apply integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable_aux2
    /-
      case pos.inr.intro.hf'g
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      ⊢ MeasureTheory.Integrable (fun x => (B (f' (L.symm x))) (g (L.symm x))) ν
    -/
  · simpa [ν, L_emb.integrable_map_iff, Function.comp_def] using hf'g
    /-
      🎉 no goals
    -/
    /-
      case pos.inr.intro.hfg'
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      ⊢ MeasureTheory.Integrable (fun x => (B (f (L.symm x))) (g' (L.symm x))) ν
    -/
  · simpa [ν, L_emb.integrable_map_iff, Function.comp_def] using hfg'
    /-
      🎉 no goals
    -/
    /-
      case pos.inr.intro.hfg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      ⊢ MeasureTheory.Integrable (fun x => (B (f (L.symm x))) (g (L.symm x))) ν
    -/
  · simpa [ν, L_emb.integrable_map_iff, Function.comp_def] using hfg
    /-
      🎉 no goals
    -/
    /-
      case pos.inr.intro.hf
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      ⊢ ∀ (x : Prod E' Real), HasLineDerivAt Real (fun x => f (L.symm x)) (f' (L.sym …
    -/
  · intro x
    /-
      case pos.inr.intro.hf
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      ⊢ HasLineDerivAt Real (fun x => f (L.symm x)) (f' (L.symm x)) x { fst := 0, sn …
    -/
    have : f = (f ∘ L.symm) ∘ (L : E →ₗ[ℝ] (E' × ℝ)) := by ext y; simp
    /-
      case pos.inr.intro.hf
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq f (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L)
      ⊢ HasLineDerivAt Real (fun x => f (L.symm x)) (f' (L.symm x)) x { fst := 0, sn …
    -/
    specialize hf (L.symm x)
    /-
      case pos.inr.intro.hf
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq f (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L)
      hf : HasLineDerivAt Real f (f' (L.symm x)) (L.symm x) v
      ⊢ HasLineDerivAt Real (fun x => f (L.symm x)) (f' (L.symm x)) x { fst := 0, sn …
    -/
    rw [this] at hf
    /-
      case pos.inr.intro.hf
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq f (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L)
      hf : HasLineDerivAt Real (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L) (f' (L …
      ⊢ HasLineDerivAt Real (fun x => f (L.symm x)) (f' (L.symm x)) x { fst := 0, sn …
    -/
    convert hf.of_comp using 1
      /-
        case h.e'_11
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace Real E
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : NormedAddCommGroup G
        inst✝⁶ : NormedSpace Real G
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        inst✝³ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        f f' : E → F
        g g' : E → G
        v : E
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
        hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
        hW : CompleteSpace W
        hv : Ne v 0
        this✝ : Nontrivial E
        n : Nat := Module.finrank Real E
        E' : Type := Fin (HSub.hSub n 1) → Real
        L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
        hL : Eq (L v) { fst := 0, snd := 1 }
        ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
        L_emb : MeasurableEmbedding ⇑L
        x : Prod E' Real
        this : Eq f (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L)
        hf : HasLineDerivAt Real (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L) (f' (L …
        ⊢ Eq x (↑↑L (L.symm x))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.e'_12
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace Real E
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : NormedAddCommGroup G
        inst✝⁶ : NormedSpace Real G
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        inst✝³ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        f f' : E → F
        g g' : E → G
        v : E
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
        hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
        hW : CompleteSpace W
        hv : Ne v 0
        this✝ : Nontrivial E
        n : Nat := Module.finrank Real E
        E' : Type := Fin (HSub.hSub n 1) → Real
        L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
        hL : Eq (L v) { fst := 0, snd := 1 }
        ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
        L_emb : MeasurableEmbedding ⇑L
        x : Prod E' Real
        this : Eq f (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L)
        hf : HasLineDerivAt Real (Function.comp (Function.comp f ⇑L.symm) ⇑↑↑L) (f' (L …
        ⊢ Eq { fst := 0, snd := 1 } (↑↑L v)
      -/
    · simp [← hL]
      /-
        🎉 no goals
      -/
    /-
      case pos.inr.intro.hg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      ⊢ ∀ (x : Prod E' Real), HasLineDerivAt Real (fun x => g (L.symm x)) (g' (L.sym …
    -/
  · intro x
    /-
      case pos.inr.intro.hg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      ⊢ HasLineDerivAt Real (fun x => g (L.symm x)) (g' (L.symm x)) x { fst := 0, sn …
    -/
    have : g = (g ∘ L.symm) ∘ (L : E →ₗ[ℝ] (E' × ℝ)) := by ext y; simp
    /-
      case pos.inr.intro.hg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hg : ∀ (x : E), HasLineDerivAt Real g (g' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq g (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L)
      ⊢ HasLineDerivAt Real (fun x => g (L.symm x)) (g' (L.symm x)) x { fst := 0, sn …
    -/
    specialize hg (L.symm x)
    /-
      case pos.inr.intro.hg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq g (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L)
      hg : HasLineDerivAt Real g (g' (L.symm x)) (L.symm x) v
      ⊢ HasLineDerivAt Real (fun x => g (L.symm x)) (g' (L.symm x)) x { fst := 0, sn …
    -/
    rw [this] at hg
    /-
      case pos.inr.intro.hg
      E : Type u_1
      F : Type u_2
      G : Type u_3
      W : Type u_4
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : NormedSpace Real E
      inst✝⁹ : NormedAddCommGroup F
      inst✝⁸ : NormedSpace Real F
      inst✝⁷ : NormedAddCommGroup G
      inst✝⁶ : NormedSpace Real G
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      inst✝³ : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝² : BorelSpace E
      inst✝¹ : FiniteDimensional Real E
      inst✝ : μ.IsAddHaarMeasure
      f f' : E → F
      g g' : E → G
      v : E
      B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
      hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
      hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
      hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
      hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
      hW : CompleteSpace W
      hv : Ne v 0
      this✝ : Nontrivial E
      n : Nat := Module.finrank Real E
      E' : Type := Fin (HSub.hSub n 1) → Real
      L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
      hL : Eq (L v) { fst := 0, snd := 1 }
      ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
      L_emb : MeasurableEmbedding ⇑L
      x : Prod E' Real
      this : Eq g (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L)
      hg : HasLineDerivAt Real (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L) (g' (L …
      ⊢ HasLineDerivAt Real (fun x => g (L.symm x)) (g' (L.symm x)) x { fst := 0, sn …
    -/
    convert hg.of_comp using 1
      /-
        case h.e'_11
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace Real E
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : NormedAddCommGroup G
        inst✝⁶ : NormedSpace Real G
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        inst✝³ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        f f' : E → F
        g g' : E → G
        v : E
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
        hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
        hW : CompleteSpace W
        hv : Ne v 0
        this✝ : Nontrivial E
        n : Nat := Module.finrank Real E
        E' : Type := Fin (HSub.hSub n 1) → Real
        L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
        hL : Eq (L v) { fst := 0, snd := 1 }
        ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
        L_emb : MeasurableEmbedding ⇑L
        x : Prod E' Real
        this : Eq g (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L)
        hg : HasLineDerivAt Real (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L) (g' (L …
        ⊢ Eq x (↑↑L (L.symm x))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.e'_12
        E : Type u_1
        F : Type u_2
        G : Type u_3
        W : Type u_4
        inst✝¹¹ : NormedAddCommGroup E
        inst✝¹⁰ : NormedSpace Real E
        inst✝⁹ : NormedAddCommGroup F
        inst✝⁸ : NormedSpace Real F
        inst✝⁷ : NormedAddCommGroup G
        inst✝⁶ : NormedSpace Real G
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        inst✝³ : MeasurableSpace E
        μ : MeasureTheory.Measure E
        inst✝² : BorelSpace E
        inst✝¹ : FiniteDimensional Real E
        inst✝ : μ.IsAddHaarMeasure
        f f' : E → F
        g g' : E → G
        v : E
        B : ContinuousLinearMap (RingHom.id Real) F (ContinuousLinearMap (RingHom.id R …
        hf'g : MeasureTheory.Integrable (fun x => (B (f' x)) (g x)) μ
        hfg' : MeasureTheory.Integrable (fun x => (B (f x)) (g' x)) μ
        hfg : MeasureTheory.Integrable (fun x => (B (f x)) (g x)) μ
        hf : ∀ (x : E), HasLineDerivAt Real f (f' x) x v
        hW : CompleteSpace W
        hv : Ne v 0
        this✝ : Nontrivial E
        n : Nat := Module.finrank Real E
        E' : Type := Fin (HSub.hSub n 1) → Real
        L : ContinuousLinearEquiv (RingHom.id Real) E (Prod E' Real)
        hL : Eq (L v) { fst := 0, snd := 1 }
        ν : MeasureTheory.Measure (Prod E' Real) := MeasureTheory.Measure.map (⇑L) μ
        L_emb : MeasurableEmbedding ⇑L
        x : Prod E' Real
        this : Eq g (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L)
        hg : HasLineDerivAt Real (Function.comp (Function.comp g ⇑L.symm) ⇑↑↑L) (g' (L …
        ⊢ Eq { fst := 0, snd := 1 } (↑↑L v)
      -/
    · simp [← hL]
      /-
        🎉 no goals
      -/


/-- **Integration by parts for Fréchet derivatives**
Version with a general bilinear form `B`.
If `B f g` is integrable, as well as `B f' g` and `B f g'` where `f'` and `g'` are derivatives
of `f` and `g` in a given direction `v`, then `∫ B f g' = - ∫ B f' g`. -/
theorem integral_bilinear_hasFDerivAt_right_eq_neg_left_of_integrable
    {f : E → F} {f' : E → (E →L[ℝ] F)}
    {g : E → G} {g' : E → (E →L[ℝ] G)} {v : E} {B : F →L[ℝ] G →L[ℝ] W}
    (hf'g : Integrable (fun x ↦ B (f' x v) (g x)) μ)
    (hfg' : Integrable (fun x ↦ B (f x) (g' x v)) μ)
    (hfg : Integrable (fun x ↦ B (f x) (g x)) μ)
    (hf : ∀ x, HasFDerivAt f (f' x) x) (hg : ∀ x, HasFDerivAt g (g' x) x) :
    ∫ x, B (f x) (g' x v) ∂μ = - ∫ x, B (f' x v) (g x) ∂μ :=
  integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable hf'g hfg' hfg
    (fun x ↦ (hf x).hasLineDerivAt v) (fun x ↦ (hg x).hasLineDerivAt v)


/-- **Integration by parts for Fréchet derivatives**
Version with a general bilinear form `B`.
If `B f g` is integrable, as well as `B f' g` and `B f g'` where `f'` and `g'` are the derivatives
of `f` and `g` in a given direction `v`, then `∫ B f g' = - ∫ B f' g`. -/
theorem integral_bilinear_fderiv_right_eq_neg_left_of_integrable
    {f : E → F} {g : E → G} {v : E} {B : F →L[ℝ] G →L[ℝ] W}
    (hf'g : Integrable (fun x ↦ B (fderiv ℝ f x v) (g x)) μ)
    (hfg' : Integrable (fun x ↦ B (f x) (fderiv ℝ g x v)) μ)
    (hfg : Integrable (fun x ↦ B (f x) (g x)) μ)
    (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
    ∫ x, B (f x) (fderiv ℝ g x v) ∂μ = - ∫ x, B (fderiv ℝ f x v) (g x) ∂μ :=
  integral_bilinear_hasFDerivAt_right_eq_neg_left_of_integrable hf'g hfg' hfg
    (fun x ↦ (hf x).hasFDerivAt) (fun x ↦ (hg x).hasFDerivAt)


/-- **Integration by parts for Fréchet derivatives**
Version with a scalar function: `∫ f • g' = - ∫ f' • g` when `f • g'` and `f' • g` and `f • g`
are integrable, where `f'` and `g'` are the derivatives of `f` and `g` in a given direction `v`. -/
theorem integral_smul_fderiv_eq_neg_fderiv_smul_of_integrable
    {f : E → 𝕜} {g : E → G} {v : E}
    (hf'g : Integrable (fun x ↦ fderiv ℝ f x v • g x) μ)
    (hfg' : Integrable (fun x ↦ f x • fderiv ℝ g x v) μ)
    (hfg : Integrable (fun x ↦ f x • g x) μ)
    (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
    ∫ x, f x • fderiv ℝ g x v ∂μ = - ∫ x, fderiv ℝ f x v • g x ∂μ :=
  integral_bilinear_fderiv_right_eq_neg_left_of_integrable
    (B := ContinuousLinearMap.lsmul ℝ 𝕜) hf'g hfg' hfg hf hg


/-- **Integration by parts for Fréchet derivatives**
Version with two scalar functions: `∫ f * g' = - ∫ f' * g` when `f * g'` and `f' * g` and `f * g`
are integrable, where `f'` and `g'` are the derivatives of `f` and `g` in a given direction `v`. -/
theorem integral_mul_fderiv_eq_neg_fderiv_mul_of_integrable
    {f : E → 𝕜} {g : E → 𝕜} {v : E}
    (hf'g : Integrable (fun x ↦ fderiv ℝ f x v * g x) μ)
    (hfg' : Integrable (fun x ↦ f x * fderiv ℝ g x v) μ)
    (hfg : Integrable (fun x ↦ f x * g x) μ)
    (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
    ∫ x, f x * fderiv ℝ g x v ∂μ = - ∫ x, fderiv ℝ f x v * g x ∂μ :=
  integral_bilinear_fderiv_right_eq_neg_left_of_integrable
    (B := ContinuousLinearMap.mul ℝ 𝕜) hf'g hfg' hfg hf hg

