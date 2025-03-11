local macro:arg t:term:max noWs "ⁿ" : term => `(Fin n → $t)


local macro:arg t:term:max noWs "ⁿ⁺¹" : term => `(Fin (n + 1) → $t)


local notation "e " i => Pi.single i 1


/-- An auxiliary lemma for
`MeasureTheory.integral_divergence_of_hasFDerivWithinAt_off_countable`. This is exactly
`BoxIntegral.hasIntegral_GP_divergence_of_forall_hasDerivWithinAt` reformulated for the
Bochner integral. -/
theorem integral_divergence_of_hasFDerivWithinAt_off_countable_aux₁ (I : Box (Fin (n + 1)))
    (f : ℝⁿ⁺¹ → Eⁿ⁺¹)
    (f' : ℝⁿ⁺¹ → ℝⁿ⁺¹ →L[ℝ] Eⁿ⁺¹) (s : Set ℝⁿ⁺¹)
    (hs : s.Countable) (Hc : ContinuousOn f (Box.Icc I))
    (Hd : ∀ x ∈ (Box.Icc I) \ s, HasFDerivWithinAt f (f' x) (Box.Icc I) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            n : Nat
            I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
            f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
            f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
            s : Set (Fin (HAdd.hAdd n 1) → Real)
            hs : s.Countable
            Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
            Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
            ⊢ MeasureTheory.Measure (Fin (HAdd.hAdd n 1) → Real)
          -/
    (Hi : IntegrableOn (fun x => ∑ i, f' x (e i) i) (Box.Icc I)) :
          /-
            🎉 no goals
          -/
    (∫ x in Box.Icc I, ∑ i, f' x (e i) i) =
      ∑ i : Fin (n + 1),
        ((∫ x in Box.Icc (I.face i), f (i.insertNth (I.upper i) x) i) -
          ∫ x in Box.Icc (I.face i), f (i.insertNth (I.lower i) x) i) := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  simp only [← setIntegral_congr_set (Box.coe_ae_eq_Icc _)]
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ↑I) f …
  -/
  have A := (Hi.mono_set Box.coe_subset_Icc).hasBoxIntegral ⊥ rfl
  have B :=
    hasIntegral_GP_divergence_of_forall_hasDerivWithinAt I f f' (s ∩ Box.Icc I)
      (hs.mono inter_subset_left) (fun x hx => Hc _ hx.2) fun x hx =>
      Hd _ ⟨hx.1, fun h => hx.2 ⟨h, hx.1⟩⟩
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
    B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ↑I) f …
  -/
  rw [continuousOn_pi] at Hc
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
    B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict ↑I) f …
  -/
  refine (A.unique B).trans (sum_congr rfl fun i _ => ?_)
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
    B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
    i : Fin (HAdd.hAdd n 1)
    x✝ : Membership.mem Finset.univ i
    ⊢ Eq (HSub.hSub (BoxIntegral.integral (I.face i) BoxIntegral.IntegrationParams …
  -/
  refine congr_arg₂ Sub.sub ?_ ?_
    /-
      case refine_1
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
      B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      ⊢ Eq (BoxIntegral.integral (I.face i) BoxIntegral.IntegrationParams.GP (fun x  …
    -/
  · have := Box.continuousOn_face_Icc (Hc i) (Set.right_mem_Icc.2 (I.lower_le_upper i))
    have := (this.integrableOn_compact (μ := volume) (Box.isCompact_Icc _)).mono_set
      Box.coe_subset_Icc
    /-
      case refine_1
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
      B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      this✝ : ContinuousOn (Function.comp (fun y => f y i) (i.insertNth (I.upper i)) …
      this : MeasureTheory.IntegrableOn (Function.comp (fun y => f y i) (i.insertNth …
      ⊢ Eq (BoxIntegral.integral (I.face i) BoxIntegral.IntegrationParams.GP (fun x  …
    -/
    exact (this.hasBoxIntegral ⊥ rfl).integral_eq
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
      B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      ⊢ Eq (BoxIntegral.integral (I.face i) BoxIntegral.IntegrationParams.GP (fun x  …
    -/
  · have := Box.continuousOn_face_Icc (Hc i) (Set.left_mem_Icc.2 (I.lower_le_upper i))
    have := (this.integrableOn_compact (μ := volume) (Box.isCompact_Icc _)).mono_set
      Box.coe_subset_Icc
    /-
      case refine_2
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (fun y => f y i) (BoxIntegral.B …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      A : BoxIntegral.HasIntegral I Bot.bot (fun x => Finset.univ.sum fun i => (f' x …
      B : BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finse …
      i : Fin (HAdd.hAdd n 1)
      x✝ : Membership.mem Finset.univ i
      this✝ : ContinuousOn (Function.comp (fun y => f y i) (i.insertNth (I.lower i)) …
      this : MeasureTheory.IntegrableOn (Function.comp (fun y => f y i) (i.insertNth …
      ⊢ Eq (BoxIntegral.integral (I.face i) BoxIntegral.IntegrationParams.GP (fun x  …
    -/
    exact (this.hasBoxIntegral ⊥ rfl).integral_eq
    /-
      🎉 no goals
    -/


/-- An auxiliary lemma for
`MeasureTheory.integral_divergence_of_hasFDerivWithinAt_off_countable`. Compared to the previous
lemma, here we drop the assumption of differentiability on the boundary of the box. -/
theorem integral_divergence_of_hasFDerivWithinAt_off_countable_aux₂ (I : Box (Fin (n + 1)))
    (f : ℝⁿ⁺¹ → Eⁿ⁺¹)
    (f' : ℝⁿ⁺¹ → ℝⁿ⁺¹ →L[ℝ] Eⁿ⁺¹)
    (s : Set ℝⁿ⁺¹) (hs : s.Countable) (Hc : ContinuousOn f (Box.Icc I))
    (Hd : ∀ x ∈ Box.Ioo I \ s, HasFDerivAt f (f' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            n : Nat
            I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
            f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
            f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
            s : Set (Fin (HAdd.hAdd n 1) → Real)
            hs : s.Countable
            Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
            Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
            ⊢ MeasureTheory.Measure (Fin (HAdd.hAdd n 1) → Real)
          -/
    (Hi : IntegrableOn (∑ i, f' · (e i) i) (Box.Icc I)) :
          /-
            🎉 no goals
          -/
    (∫ x in Box.Icc I, ∑ i, f' x (e i) i) =
      ∑ i : Fin (n + 1),
        ((∫ x in Box.Icc (I.face i), f (i.insertNth (I.upper i) x) i) -
          ∫ x in Box.Icc (I.face i), f (i.insertNth (I.lower i) x) i) := by
  /- Choose a monotone sequence `J k` of subboxes that cover the interior of `I` and prove that
    these boxes satisfy the assumptions of the previous lemma. -/
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  rcases I.exists_seq_mono_tendsto with ⟨J, hJ_sub, hJl, hJu⟩
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  have hJ_sub' : ∀ k, Box.Icc (J k) ⊆ Box.Icc I := fun k => (hJ_sub k).trans I.Ioo_subset_Icc
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  have hJ_le : ∀ k, J k ≤ I := fun k => Box.le_iff_Icc.2 (hJ_sub' k)
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  have HcJ : ∀ k, ContinuousOn f (Box.Icc (J k)) := fun k => Hc.mono (hJ_sub' k)
  have HdJ : ∀ (k), ∀ x ∈ (Box.Icc (J k)) \ s, HasFDerivWithinAt f (f' x) (Box.Icc (J k)) x :=
    fun k x hx => (Hd x ⟨hJ_sub k hx.1, hx.2⟩).hasFDerivWithinAt
  have HiJ : ∀ k, IntegrableOn (∑ i, f' · (e i) i) (Box.Icc (J k)) volume := fun k =>
    Hi.mono_set (hJ_sub' k)
  -- Apply the previous lemma to `J k`.
  have HJ_eq := fun k =>
    integral_divergence_of_hasFDerivWithinAt_off_countable_aux₁ (J k) f f' s hs (HcJ k) (HdJ k)
      (HiJ k)
  -- Note that the LHS of `HJ_eq k` tends to the LHS of the goal as `k → ∞`.
  have hI_tendsto :
    Tendsto (fun k => ∫ x in Box.Icc (J k), ∑ i, f' x (e i) i) atTop
      (𝓝 (∫ x in Box.Icc I, ∑ i, f' x (e i) i)) := by
    simp only [IntegrableOn, ← Measure.restrict_congr_set (Box.Ioo_ae_eq_Icc _)] at Hi ⊢
    rw [← Box.iUnion_Ioo_of_tendsto J.monotone hJl hJu] at Hi ⊢
    exact tendsto_setIntegral_of_monotone (fun k => (J k).measurableSet_Ioo)
      (Box.Ioo.comp J).monotone Hi
  -- Thus it suffices to prove the same about the RHS.
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    hI_tendsto : Filter.Tendsto (fun k => MeasureTheory.integral (MeasureTheory.Me …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (BoxI …
  -/
  refine tendsto_nhds_unique_of_eventuallyEq hI_tendsto ?_ (Eventually.of_forall HJ_eq)
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    hI_tendsto : Filter.Tendsto (fun k => MeasureTheory.integral (MeasureTheory.Me …
    ⊢ Filter.Tendsto (fun x => Finset.univ.sum fun i => HSub.hSub (MeasureTheory.i …
  -/
  clear hI_tendsto
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : Filter.Tendsto (Function.comp BoxIntegral.Box.lower ⇑J) Filter.atTop (nh …
    hJu : Filter.Tendsto (Function.comp BoxIntegral.Box.upper ⇑J) Filter.atTop (nh …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    ⊢ Filter.Tendsto (fun x => Finset.univ.sum fun i => HSub.hSub (MeasureTheory.i …
  -/
  rw [tendsto_pi_nhds] at hJl hJu
  /- We'll need to prove a similar statement about the integrals over the front sides and the
    integrals over the back sides. In order to avoid repeating ourselves, we formulate a lemma. -/
  suffices ∀ (i : Fin (n + 1)) (c : ℕ → ℝ) (d), (∀ k, c k ∈ Icc (I.lower i) (I.upper i)) →
    Tendsto c atTop (𝓝 d) →
      Tendsto (fun k => ∫ x in Box.Icc ((J k).face i), f (i.insertNth (c k) x) i) atTop
        (𝓝 <| ∫ x in Box.Icc (I.face i), f (i.insertNth d x) i) by
    rw [Box.Icc_eq_pi] at hJ_sub'
    refine tendsto_finset_sum _ fun i _ => (this _ _ _ ?_ (hJu _)).sub (this _ _ _ ?_ (hJl _))
    exacts [fun k => hJ_sub' k (J k).upper_mem_Icc _ trivial, fun k =>
      hJ_sub' k (J k).lower_mem_Icc _ trivial]
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJu : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    ⊢ ∀ (i : Fin (HAdd.hAdd n 1)) (c : Nat → Real) (d : Real), (∀ (k : Nat), Membe …
  -/
  intro i c d hc hcd
  /- First we prove that the integrals of the restriction of `f` to `{x | x i = d}` over increasing
    boxes `((J k).face i).Icc` tend to the desired limit. The proof mostly repeats the one above. -/
  have hd : d ∈ Icc (I.lower i) (I.upper i) :=
    isClosed_Icc.mem_of_tendsto hcd (Eventually.of_forall hc)
  have Hic : ∀ k, IntegrableOn (fun x => f (i.insertNth (c k) x) i) (Box.Icc (I.face i)) := fun k =>
    (Box.continuousOn_face_Icc ((continuous_apply i).comp_continuousOn Hc) (hc k)).integrableOn_Icc
  have Hid : IntegrableOn (fun x => f (i.insertNth d x) i) (Box.Icc (I.face i)) :=
    (Box.continuousOn_face_Icc ((continuous_apply i).comp_continuousOn Hc) hd).integrableOn_Icc
  have H :
    Tendsto (fun k => ∫ x in Box.Icc ((J k).face i), f (i.insertNth d x) i) atTop
      (𝓝 <| ∫ x in Box.Icc (I.face i), f (i.insertNth d x) i) := by
    have hIoo : (⋃ k, Box.Ioo ((J k).face i)) = Box.Ioo (I.face i) :=
      Box.iUnion_Ioo_of_tendsto ((Box.monotone_face i).comp J.monotone)
        (tendsto_pi_nhds.2 fun _ => hJl _) (tendsto_pi_nhds.2 fun _ => hJu _)
    simp only [IntegrableOn, ← Measure.restrict_congr_set (Box.Ioo_ae_eq_Icc _), ← hIoo] at Hid ⊢
    exact tendsto_setIntegral_of_monotone (fun k => ((J k).face i).measurableSet_Ioo)
      (Box.Ioo.monotone.comp ((Box.monotone_face i).comp J.monotone)) Hid
  /- Thus it suffices to show that the distance between the integrals of the restrictions of `f` to
    `{x | x i = c k}` and `{x | x i = d}` over `((J k).face i).Icc` tends to zero as `k → ∞`. Choose
    `ε > 0`. -/
  /-
    case intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJu : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    i : Fin (HAdd.hAdd n 1)
    c : Nat → Real
    d : Real
    hc : ∀ (k : Nat), Membership.mem (Set.Icc (I.lower i) (I.upper i)) (c k)
    hcd : Filter.Tendsto c Filter.atTop (nhds d)
    hd : Membership.mem (Set.Icc (I.lower i) (I.upper i)) d
    Hic : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => f (i.insertNth (c k) x …
    Hid : MeasureTheory.IntegrableOn (fun x => f (i.insertNth d x) i) (BoxIntegral …
    H : Filter.Tendsto (fun k => MeasureTheory.integral (MeasureTheory.MeasureSpac …
    ⊢ Filter.Tendsto (fun k => MeasureTheory.integral (MeasureTheory.MeasureSpace. …
  -/
  refine H.congr_dist (Metric.nhds_basis_closedBall.tendsto_right_iff.2 fun ε εpos => ?_)
  have hvol_pos : ∀ J : Box (Fin n), 0 < ∏ j, (J.upper j - J.lower j) := fun J =>
    prod_pos fun j hj => sub_pos.2 <| J.lower_lt_upper _
  /- Choose `δ > 0` such that for any `x y ∈ I.Icc` at distance at most `δ`, the distance between
    `f x` and `f y` is at most `ε / volume (I.face i).Icc`, then the distance between the integrals
    is at most `(ε / volume (I.face i).Icc) * volume ((J k).face i).Icc ≤ ε`. -/
  rcases Metric.uniformContinuousOn_iff_le.1 (I.isCompact_Icc.uniformContinuousOn_of_continuous Hc)
      (ε / ∏ j, ((I.face i).upper j - (I.face i).lower j)) (div_pos εpos (hvol_pos (I.face i)))
    with ⟨δ, δpos, hδ⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    J : OrderHom Nat (BoxIntegral.Box (Fin (HAdd.hAdd n 1)))
    hJ_sub : ∀ (n_1 : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J n_1)) (BoxInt …
    hJl : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJu : ∀ (x : Fin (HAdd.hAdd n 1)), Filter.Tendsto (fun i => Function.comp BoxI …
    hJ_sub' : ∀ (k : Nat), HasSubset.Subset (BoxIntegral.Box.Icc (J k)) (BoxIntegr …
    hJ_le : ∀ (k : Nat), LE.le (J k) I
    HcJ : ∀ (k : Nat), ContinuousOn f (BoxIntegral.Box.Icc (J k))
    HdJ : ∀ (k : Nat) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdif …
    HiJ : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i  …
    HJ_eq : ∀ (k : Nat), Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.vo …
    i : Fin (HAdd.hAdd n 1)
    c : Nat → Real
    d : Real
    hc : ∀ (k : Nat), Membership.mem (Set.Icc (I.lower i) (I.upper i)) (c k)
    hcd : Filter.Tendsto c Filter.atTop (nhds d)
    hd : Membership.mem (Set.Icc (I.lower i) (I.upper i)) d
    Hic : ∀ (k : Nat), MeasureTheory.IntegrableOn (fun x => f (i.insertNth (c k) x …
    Hid : MeasureTheory.IntegrableOn (fun x => f (i.insertNth d x) i) (BoxIntegral …
    H : Filter.Tendsto (fun k => MeasureTheory.integral (MeasureTheory.MeasureSpac …
    ε : Real
    εpos : LT.lt 0 ε
    hvol_pos : ∀ (J : BoxIntegral.Box (Fin n)), LT.lt 0 (Finset.univ.prod fun j => …
    δ : Real
    δpos : GT.gt δ 0
    hδ : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (BoxIntegral.Box.Icc I …
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.closedBall 0 ε) (Dist.dis …
  -/
  refine (hcd.eventually (Metric.ball_mem_nhds _ δpos)).mono fun k hk => ?_
  have Hsub : Box.Icc ((J k).face i) ⊆ Box.Icc (I.face i) :=
    Box.le_iff_Icc.1 (Box.face_mono (hJ_le _) i)
  rw [mem_closedBall_zero_iff, Real.norm_eq_abs, abs_of_nonneg dist_nonneg, dist_eq_norm,
    ← integral_sub (Hid.mono_set Hsub) ((Hic _).mono_set Hsub)]
  calc
    ‖∫ x in Box.Icc ((J k).face i), f (i.insertNth d x) i - f (i.insertNth (c k) x) i‖ ≤
        (ε / ∏ j, ((I.face i).upper j - (I.face i).lower j)) *
          (volume (Box.Icc ((J k).face i))).toReal := by
      refine norm_setIntegral_le_of_norm_le_const' (((J k).face i).measure_Icc_lt_top _)
        ((J k).face i).measurableSet_Icc fun x hx => ?_
      rw [← dist_eq_norm]
      calc
        dist (f (i.insertNth d x) i) (f (i.insertNth (c k) x) i) ≤
            dist (f (i.insertNth d x)) (f (i.insertNth (c k) x)) :=
          dist_le_pi_dist (f (i.insertNth d x)) (f (i.insertNth (c k) x)) i
        _ ≤ ε / ∏ j, ((I.face i).upper j - (I.face i).lower j) :=
          hδ _ (I.mapsTo_insertNth_face_Icc hd <| Hsub hx) _
            (I.mapsTo_insertNth_face_Icc (hc _) <| Hsub hx) ?_
      rw [Fin.dist_insertNth_insertNth, dist_self, dist_comm]
      exact max_le hk.le δpos.lt.le
    _ ≤ ε := by
      rw [Box.Icc_def, Real.volume_Icc_pi_toReal ((J k).face i).lower_le_upper,
        ← le_div_iff₀ (hvol_pos _)]
      gcongr
      exacts [hvol_pos _, fun _ _ ↦ sub_nonneg.2 (Box.lower_le_upper _ _),
        (hJ_sub' _ (J _).upper_mem_Icc).2 _, (hJ_sub' _ (J _).lower_mem_Icc).1 _]


local notation "face " i => Set.Icc (a ∘ Fin.succAbove i) (b ∘ Fin.succAbove i)

local notation:max "frontFace " i:arg => Fin.insertNth i (b i)

local notation:max "backFace " i:arg => Fin.insertNth i (a i)


/-- **Divergence theorem** for Bochner integral. If `f : ℝⁿ⁺¹ → Eⁿ⁺¹` is continuous on a rectangular
box `[a, b] : Set ℝⁿ⁺¹`, `a ≤ b`, is differentiable on its interior with derivative
`f' : ℝⁿ⁺¹ → ℝⁿ⁺¹ →L[ℝ] Eⁿ⁺¹` and the divergence `fun x ↦ ∑ i, f' x eᵢ i` is integrable on `[a, b]`,
where `eᵢ = Pi.single i 1` is the `i`-th basis vector, then its integral is equal to the sum of
integrals of `f` over the faces of `[a, b]`, taken with appropriate signs.

Moreover, the same is true if the function is not differentiable at countably many
points of the interior of `[a, b]`.

We represent both faces `x i = a i` and `x i = b i` as the box
`face i = [a ∘ Fin.succAbove i, b ∘ Fin.succAbove i]` in `ℝⁿ`, where
`Fin.succAbove : Fin n ↪o Fin (n + 1)` is the order embedding with range `{i}ᶜ`. The restrictions
of `f : ℝⁿ⁺¹ → Eⁿ⁺¹` to these faces are given by `f ∘ backFace i` and `f ∘ frontFace i`, where
`backFace i = Fin.insertNth i (a i)` and `frontFace i = Fin.insertNth i (b i)` are embeddings
`ℝⁿ → ℝⁿ⁺¹` that take `y : ℝⁿ` and insert `a i` (resp., `b i`) as `i`-th coordinate. -/
theorem integral_divergence_of_hasFDerivWithinAt_off_countable (hle : a ≤ b)
    (f : ℝⁿ⁺¹ → Eⁿ⁺¹)
    (f' : ℝⁿ⁺¹ → ℝⁿ⁺¹ →L[ℝ] Eⁿ⁺¹)
    (s : Set ℝⁿ⁺¹) (hs : s.Countable) (Hc : ContinuousOn f (Icc a b))
    (Hd : ∀ x ∈ (Set.pi univ fun i => Ioo (a i) (b i)) \ s, HasFDerivAt f (f' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            n : Nat
            a b : Fin (HAdd.hAdd n 1) → Real
            hle : LE.le a b
            f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
            f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
            s : Set (Fin (HAdd.hAdd n 1) → Real)
            hs : s.Countable
            Hc : ContinuousOn f (Set.Icc a b)
            Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
            ⊢ MeasureTheory.Measure (Fin (HAdd.hAdd n 1) → Real)
          -/
    (Hi : IntegrableOn (fun x => ∑ i, f' x (e i) i) (Icc a b)) :
          /-
            🎉 no goals
          -/
    (∫ x in Icc a b, ∑ i, f' x (e i) i) =
      ∑ i : Fin (n + 1),
        ((∫ x in face i, f (frontFace i x) i) - ∫ x in face i, f (backFace i x) i) := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    n : Nat
    a b : Fin (HAdd.hAdd n 1) → Real
    hle : LE.le a b
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hc : ContinuousOn f (Set.Icc a b)
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
    Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
    ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
  -/
  rcases em (∃ i, a i = b i) with (⟨i, hi⟩ | hne)
  · -- First we sort out the trivial case `∃ i, a i = b i`.
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
    rw [volume_pi, ← setIntegral_congr_set Measure.univ_pi_Ioc_ae_eq_Icc]
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.pi fun x => MeasureTheory …
    -/
    have hi' : Ioc (a i) (b i) = ∅ := Ioc_eq_empty hi.not_lt
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
      ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.pi fun x => MeasureTheory …
    -/
    have : (pi Set.univ fun j => Ioc (a j) (b j)) = ∅ := univ_pi_eq_empty hi'
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
      this : Eq (Set.univ.pi fun j => Set.Ioc (a j) (b j)) EmptyCollection.emptyColl …
      ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.pi fun x => MeasureTheory …
    -/
    rw [this, setIntegral_empty, sum_eq_zero]
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
      this : Eq (Set.univ.pi fun j => Set.Ioc (a j) (b j)) EmptyCollection.emptyColl …
      ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem Finset.univ x → Eq (HSub.hSub (M …
    -/
    rintro j -
    /-
      case inl.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      i : Fin (HAdd.hAdd n 1)
      hi : Eq (a i) (b i)
      hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
      this : Eq (Set.univ.pi fun j => Set.Ioc (a j) (b j)) EmptyCollection.emptyColl …
      j : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
    rcases eq_or_ne i j with (rfl | hne)
      /-
        case inl.intro.inl
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        n : Nat
        a b : Fin (HAdd.hAdd n 1) → Real
        hle : LE.le a b
        f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hc : ContinuousOn f (Set.Icc a b)
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
        Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
        i : Fin (HAdd.hAdd n 1)
        hi : Eq (a i) (b i)
        hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
        this : Eq (Set.univ.pi fun j => Set.Ioc (a j) (b j)) EmptyCollection.emptyColl …
        ⊢ Eq (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
      -/
    · simp [hi]
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.inr
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        n : Nat
        a b : Fin (HAdd.hAdd n 1) → Real
        hle : LE.le a b
        f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hc : ContinuousOn f (Set.Icc a b)
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
        Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
        i : Fin (HAdd.hAdd n 1)
        hi : Eq (a i) (b i)
        hi' : Eq (Set.Ioc (a i) (b i)) EmptyCollection.emptyCollection
        this : Eq (Set.univ.pi fun j => Set.Ioc (a j) (b j)) EmptyCollection.emptyColl …
        j : Fin (HAdd.hAdd n 1)
        hne : Ne i j
        ⊢ Eq (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
      -/
    · rcases Fin.exists_succAbove_eq hne with ⟨i, rfl⟩
      have : Icc (a ∘ j.succAbove) (b ∘ j.succAbove) =ᵐ[volume] (∅ : Set ℝⁿ) := by
        rw [ae_eq_empty, Real.volume_Icc_pi, prod_eq_zero (Finset.mem_univ i)]
        simp [hi]
      rw [setIntegral_congr_set this, setIntegral_congr_set this, setIntegral_empty,
        setIntegral_empty, sub_self]
  · -- In the non-trivial case `∀ i, a i < b i`, we apply a lemma we proved above.
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      n : Nat
      a b : Fin (HAdd.hAdd n 1) → Real
      hle : LE.le a b
      f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
      Hi : MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => (f' x) (Pi. …
      hne : Not (Exists fun i => Eq (a i) (b i))
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
    have hlt : ∀ i, a i < b i := fun i => (hle i).lt_of_ne fun hi => hne ⟨i, hi⟩
    exact integral_divergence_of_hasFDerivWithinAt_off_countable_aux₂ ⟨a, b, hlt⟩ f f' s hs Hc
      Hd Hi


/-- **Divergence theorem** for a family of functions `f : Fin (n + 1) → ℝⁿ⁺¹ → E`. See also
`MeasureTheory.integral_divergence_of_hasFDerivWithinAt_off_countable'` for a version formulated
in terms of a vector-valued function `f : ℝⁿ⁺¹ → Eⁿ⁺¹`. -/
theorem integral_divergence_of_hasFDerivWithinAt_off_countable' (hle : a ≤ b)
    (f : Fin (n + 1) → ℝⁿ⁺¹ → E)
    (f' : Fin (n + 1) → ℝⁿ⁺¹ → ℝⁿ⁺¹ →L[ℝ] E) (s : Set ℝⁿ⁺¹)
    (hs : s.Countable) (Hc : ∀ i, ContinuousOn (f i) (Icc a b))
    (Hd : ∀ x ∈ (pi Set.univ fun i => Ioo (a i) (b i)) \ s, ∀ (i), HasFDerivAt (f i) (f' i x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            n : Nat
            a b : Fin (HAdd.hAdd n 1) → Real
            hle : LE.le a b
            f : Fin (HAdd.hAdd n 1) → (Fin (HAdd.hAdd n 1) → Real) → E
            f' : Fin (HAdd.hAdd n 1) → (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap  …
            s : Set (Fin (HAdd.hAdd n 1) → Real)
            hs : s.Countable
            Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
            Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ …
            ⊢ MeasureTheory.Measure (Fin (HAdd.hAdd n 1) → Real)
          -/
    (Hi : IntegrableOn (fun x => ∑ i, f' i x (e i)) (Icc a b)) :
          /-
            🎉 no goals
          -/
    (∫ x in Icc a b, ∑ i, f' i x (e i)) =
      ∑ i : Fin (n + 1), ((∫ x in face i, f i (frontFace i x)) -
        ∫ x in face i, f i (backFace i x)) :=
  integral_divergence_of_hasFDerivWithinAt_off_countable a b hle (fun x i => f i x)
    (fun x => ContinuousLinearMap.pi fun i => f' i x) s hs (continuousOn_pi.2 Hc)
    (fun x hx => hasFDerivAt_pi.2 (Hd x hx)) Hi


/-- An auxiliary lemma that is used to specialize the general divergence theorem to spaces that do
not have the form `Fin n → ℝ`. -/
theorem integral_divergence_of_hasFDerivWithinAt_off_countable_of_equiv {F : Type*}
    [NormedAddCommGroup F] [NormedSpace ℝ F] [PartialOrder F] [MeasureSpace F] [BorelSpace F]
    (eL : F ≃L[ℝ] ℝⁿ⁺¹) (he_ord : ∀ x y, eL x ≤ eL y ↔ x ≤ y)
    (he_vol : MeasurePreserving eL volume volume) (f : Fin (n + 1) → F → E)
    (f' : Fin (n + 1) → F → F →L[ℝ] E) (s : Set F) (hs : s.Countable) (a b : F) (hle : a ≤ b)
    (Hc : ∀ i, ContinuousOn (f i) (Icc a b))
    (Hd : ∀ x ∈ interior (Icc a b) \ s, ∀ (i), HasFDerivAt (f i) (f' i x) x) (DF : F → E)
                                                           /-
                                                             E : Type u
                                                             inst✝⁷ : NormedAddCommGroup E
                                                             inst✝⁶ : NormedSpace Real E
                                                             inst✝⁵ : CompleteSpace E
                                                             n : Nat
                                                             F : Type u_1
                                                             inst✝⁴ : NormedAddCommGroup F
                                                             inst✝³ : NormedSpace Real F
                                                             inst✝² : PartialOrder F
                                                             inst✝¹ : MeasureTheory.MeasureSpace F
                                                             inst✝ : BorelSpace F
                                                             eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
                                                             he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
                                                             he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
                                                             f : Fin (HAdd.hAdd n 1) → F → E
                                                             f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
                                                             s : Set F
                                                             hs : s.Countable
                                                             a b : F
                                                             hle : LE.le a b
                                                             Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
                                                             Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
                                                             DF : F → E
                                                             hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
                                                             ⊢ MeasureTheory.Measure F
                                                           -/
    (hDF : ∀ x, DF x = ∑ i, f' i x (eL.symm <| e i)) (Hi : IntegrableOn DF (Icc a b)) :
                                                           /-
                                                             🎉 no goals
                                                           -/
    ∫ x in Icc a b, DF x =
      ∑ i : Fin (n + 1),
        ((∫ x in Icc (eL a ∘ i.succAbove) (eL b ∘ i.succAbove),
            f i (eL.symm <| i.insertNth (eL b i) x)) -
          ∫ x in Icc (eL a ∘ i.succAbove) (eL b ∘ i.succAbove),
            f i (eL.symm <| i.insertNth (eL a i) x)) :=
  have he_emb : MeasurableEmbedding eL := eL.toHomeomorph.measurableEmbedding
  have hIcc : eL ⁻¹' Icc (eL a) (eL b) = Icc a b := by
    /-
      E : Type u
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      inst✝⁵ : CompleteSpace E
      n : Nat
      F : Type u_1
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      inst✝² : PartialOrder F
      inst✝¹ : MeasureTheory.MeasureSpace F
      inst✝ : BorelSpace F
      eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
      he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
      he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
      f : Fin (HAdd.hAdd n 1) → F → E
      f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
      s : Set F
      hs : s.Countable
      a b : F
      hle : LE.le a b
      Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
      Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
      DF : F → E
      hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
      Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
      he_emb : MeasurableEmbedding ⇑eL
      ⊢ Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
    -/
    ext1 x; simp only [Set.mem_preimage, Set.mem_Icc, he_ord]
            /-
              🎉 no goals
            -/
                                                             /-
                                                               E : Type u
                                                               inst✝⁷ : NormedAddCommGroup E
                                                               inst✝⁶ : NormedSpace Real E
                                                               inst✝⁵ : CompleteSpace E
                                                               n : Nat
                                                               F : Type u_1
                                                               inst✝⁴ : NormedAddCommGroup F
                                                               inst✝³ : NormedSpace Real F
                                                               inst✝² : PartialOrder F
                                                               inst✝¹ : MeasureTheory.MeasureSpace F
                                                               inst✝ : BorelSpace F
                                                               eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
                                                               he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
                                                               he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
                                                               f : Fin (HAdd.hAdd n 1) → F → E
                                                               f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
                                                               s : Set F
                                                               hs : s.Countable
                                                               a b : F
                                                               hle : LE.le a b
                                                               Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
                                                               Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
                                                               DF : F → E
                                                               hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
                                                               Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
                                                               he_emb : MeasurableEmbedding ⇑eL
                                                               hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
                                                               ⊢ Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
                                                             -/
  have hIcc' : Icc (eL a) (eL b) = eL.symm ⁻¹' Icc a b := by rw [← hIcc, eL.symm_preimage_preimage]
                                                             /-
                                                               🎉 no goals
                                                             -/
  calc
                                                                              /-
                                                                                E : Type u
                                                                                inst✝⁷ : NormedAddCommGroup E
                                                                                inst✝⁶ : NormedSpace Real E
                                                                                inst✝⁵ : CompleteSpace E
                                                                                n : Nat
                                                                                F : Type u_1
                                                                                inst✝⁴ : NormedAddCommGroup F
                                                                                inst✝³ : NormedSpace Real F
                                                                                inst✝² : PartialOrder F
                                                                                inst✝¹ : MeasureTheory.MeasureSpace F
                                                                                inst✝ : BorelSpace F
                                                                                eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
                                                                                he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
                                                                                he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
                                                                                f : Fin (HAdd.hAdd n 1) → F → E
                                                                                f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
                                                                                s : Set F
                                                                                hs : s.Countable
                                                                                a b : F
                                                                                hle : LE.le a b
                                                                                Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
                                                                                Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
                                                                                DF : F → E
                                                                                hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
                                                                                Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
                                                                                he_emb : MeasurableEmbedding ⇑eL
                                                                                hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
                                                                                hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
                                                                                ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
                                                                              -/
    ∫ x in Icc a b, DF x = ∫ x in Icc a b, ∑ i, f' i x (eL.symm <| e i) := by simp only [hDF]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    _ = ∫ x in Icc (eL a) (eL b), ∑ i, f' i (eL.symm x) (eL.symm <| e i) := by
      /-
        E : Type u
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : CompleteSpace E
        n : Nat
        F : Type u_1
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real F
        inst✝² : PartialOrder F
        inst✝¹ : MeasureTheory.MeasureSpace F
        inst✝ : BorelSpace F
        eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
        he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
        he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
        f : Fin (HAdd.hAdd n 1) → F → E
        f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
        s : Set F
        hs : s.Countable
        a b : F
        hle : LE.le a b
        Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
        Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
        DF : F → E
        hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
        Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
        he_emb : MeasurableEmbedding ⇑eL
        hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
        hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
      -/
      rw [← he_vol.setIntegral_preimage_emb he_emb]
      /-
        E : Type u
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace Real E
        inst✝⁵ : CompleteSpace E
        n : Nat
        F : Type u_1
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace Real F
        inst✝² : PartialOrder F
        inst✝¹ : MeasureTheory.MeasureSpace F
        inst✝ : BorelSpace F
        eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
        he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
        he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
        f : Fin (HAdd.hAdd n 1) → F → E
        f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
        s : Set F
        hs : s.Countable
        a b : F
        hle : LE.le a b
        Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
        Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
        DF : F → E
        hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
        Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
        he_emb : MeasurableEmbedding ⇑eL
        hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
        hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
      -/
      simp only [hIcc, eL.symm_apply_apply]
      /-
        🎉 no goals
      -/
    _ = ∑ i : Fin (n + 1),
          ((∫ x in Icc (eL a ∘ i.succAbove) (eL b ∘ i.succAbove),
              f i (eL.symm <| i.insertNth (eL b i) x)) -
            ∫ x in Icc (eL a ∘ i.succAbove) (eL b ∘ i.succAbove),
              f i (eL.symm <| i.insertNth (eL a i) x)) := by
      refine integral_divergence_of_hasFDerivWithinAt_off_countable' (eL a) (eL b)
        ((he_ord _ _).2 hle) (fun i x => f i (eL.symm x))
        (fun i x => f' i (eL.symm x) ∘L (eL.symm : ℝⁿ⁺¹ →L[ℝ] F)) (eL.symm ⁻¹' s)
        (hs.preimage eL.symm.injective) ?_ ?_ ?_
        /-
          case refine_1
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn ((fun i x => f i (eL.symm x)) i) ( …
        -/
      · exact fun i => (Hc i).comp eL.symm.continuousOn hIcc'.subset
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          ⊢ ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (Set.univ.pi …
        -/
      · refine fun x hx i => (Hd (eL.symm x) ⟨?_, hx.2⟩ i).comp x eL.symm.hasFDerivAt
        /-
          case refine_2
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          x : Fin (HAdd.hAdd n 1) → Real
          hx : Membership.mem (SDiff.sdiff (Set.univ.pi fun i => Set.Ioo (eL a i) (eL b  …
          i : Fin (HAdd.hAdd n 1)
          ⊢ Membership.mem (interior (Set.Icc a b)) (eL.symm x)
        -/
        rw [← hIcc]
        /-
          case refine_2
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          x : Fin (HAdd.hAdd n 1) → Real
          hx : Membership.mem (SDiff.sdiff (Set.univ.pi fun i => Set.Ioo (eL a i) (eL b  …
          i : Fin (HAdd.hAdd n 1)
          ⊢ Membership.mem (interior (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b)))) (eL.s …
        -/
        refine preimage_interior_subset_interior_preimage eL.continuous ?_
        simpa only [Set.mem_preimage, eL.apply_symm_apply, ← pi_univ_Icc,
          interior_pi_set (@finite_univ (Fin _) _), interior_Icc] using hx.1
        /-
          case refine_3
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          ⊢ MeasureTheory.IntegrableOn (fun x => Finset.univ.sum fun i => ((fun i x => ( …
        -/
      · rw [← he_vol.integrableOn_comp_preimage he_emb, hIcc]
        /-
          case refine_3
          E : Type u
          inst✝⁷ : NormedAddCommGroup E
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : CompleteSpace E
          n : Nat
          F : Type u_1
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace Real F
          inst✝² : PartialOrder F
          inst✝¹ : MeasureTheory.MeasureSpace F
          inst✝ : BorelSpace F
          eL : ContinuousLinearEquiv (RingHom.id Real) F (Fin (HAdd.hAdd n 1) → Real)
          he_ord : ∀ (x y : F), Iff (LE.le (eL x) (eL y)) (LE.le x y)
          he_vol : MeasureTheory.MeasurePreserving (⇑eL) MeasureTheory.MeasureSpace.volu …
          f : Fin (HAdd.hAdd n 1) → F → E
          f' : Fin (HAdd.hAdd n 1) → F → ContinuousLinearMap (RingHom.id Real) F E
          s : Set F
          hs : s.Countable
          a b : F
          hle : LE.le a b
          Hc : ∀ (i : Fin (HAdd.hAdd n 1)), ContinuousOn (f i) (Set.Icc a b)
          Hd : ∀ (x : F), Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x → ∀  …
          DF : F → E
          hDF : ∀ (x : F), Eq (DF x) (Finset.univ.sum fun i => (f' i x) (eL.symm (Pi.sin …
          Hi : MeasureTheory.IntegrableOn DF (Set.Icc a b) MeasureTheory.MeasureSpace.vo …
          he_emb : MeasurableEmbedding ⇑eL
          hIcc : Eq (Set.preimage (⇑eL) (Set.Icc (eL a) (eL b))) (Set.Icc a b)
          hIcc' : Eq (Set.Icc (eL a) (eL b)) (Set.preimage (⇑eL.symm) (Set.Icc a b))
          ⊢ MeasureTheory.IntegrableOn (Function.comp (fun x => Finset.univ.sum fun i => …
        -/
        simp [← hDF, Function.comp_def, Hi]
        /-
          🎉 no goals
        -/


local macro:arg t:term:max noWs "¹" : term => `(Fin 1 → $t)

local macro:arg t:term:max noWs "²" : term => `(Fin 2 → $t)


/-- **Fundamental theorem of calculus, part 2**. This version assumes that `f` is continuous on the
interval and is differentiable off a countable set `s`.

See also

* `intervalIntegral.integral_eq_sub_of_hasDeriv_right_of_le` for a version that only assumes right
differentiability of `f`;

* `MeasureTheory.integral_eq_of_hasDerivWithinAt_off_countable` for a version that works both
  for `a ≤ b` and `b ≤ a` at the expense of using unordered intervals instead of `Set.Icc`. -/
theorem integral_eq_of_hasDerivWithinAt_off_countable_of_le (f f' : ℝ → E) {a b : ℝ}
    (hle : a ≤ b) {s : Set ℝ} (hs : s.Countable) (Hc : ContinuousOn f (Icc a b))
    (Hd : ∀ x ∈ Ioo a b \ s, HasDerivAt f (f' x) x) (Hi : IntervalIntegrable f' volume a b) :
    ∫ x in a..b, f' x = f b - f a := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f f' : Real → E
    a b : Real
    hle : LE.le a b
    s : Set Real
    hs : s.Countable
    Hc : ContinuousOn f (Set.Icc a b)
    Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo a b) s) x → HasDerivAt …
    Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
  -/
  set e : ℝ ≃L[ℝ] ℝ¹ := (ContinuousLinearEquiv.funUnique (Fin 1) ℝ ℝ).symm
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f f' : Real → E
    a b : Real
    hle : LE.le a b
    s : Set Real
    hs : s.Countable
    Hc : ContinuousOn f (Set.Icc a b)
    Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo a b) s) x → HasDerivAt …
    Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    e : ContinuousLinearEquiv (RingHom.id Real) Real (Fin 1 → Real) := (Continuous …
    ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
  -/
  have e_symm : ∀ x, e.symm x = x 0 := fun x => rfl
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f f' : Real → E
    a b : Real
    hle : LE.le a b
    s : Set Real
    hs : s.Countable
    Hc : ContinuousOn f (Set.Icc a b)
    Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo a b) s) x → HasDerivAt …
    Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    e : ContinuousLinearEquiv (RingHom.id Real) Real (Fin 1 → Real) := (Continuous …
    e_symm : ∀ (x : Fin 1 → Real), Eq (e.symm x) (x 0)
    ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
  -/
  set F' : ℝ → ℝ →L[ℝ] E := fun x => smulRight (1 : ℝ →L[ℝ] ℝ) (f' x)
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f f' : Real → E
    a b : Real
    hle : LE.le a b
    s : Set Real
    hs : s.Countable
    Hc : ContinuousOn f (Set.Icc a b)
    Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo a b) s) x → HasDerivAt …
    Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    e : ContinuousLinearEquiv (RingHom.id Real) Real (Fin 1 → Real) := (Continuous …
    e_symm : ∀ (x : Fin 1 → Real), Eq (e.symm x) (x 0)
    F' : Real → ContinuousLinearMap (RingHom.id Real) Real E := fun x => Continuou …
    ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
  -/
  have hF' : ∀ x y, F' x y = y • f' x := fun x y => rfl
  calc
    ∫ x in a..b, f' x = ∫ x in Icc a b, f' x := by
      rw [intervalIntegral.integral_of_le hle, setIntegral_congr_set Ioc_ae_eq_Icc]
    _ = ∑ i : Fin 1,
          ((∫ x in Icc (e a ∘ i.succAbove) (e b ∘ i.succAbove),
              f (e.symm <| i.insertNth (e b i) x)) -
            ∫ x in Icc (e a ∘ i.succAbove) (e b ∘ i.succAbove),
              f (e.symm <| i.insertNth (e a i) x)) := by
      simp only [← interior_Icc] at Hd
      refine
        integral_divergence_of_hasFDerivWithinAt_off_countable_of_equiv e ?_ ?_ (fun _ => f)
          (fun _ => F') s hs a b hle (fun _ => Hc) (fun x hx _ => Hd x hx) _ ?_ ?_
      · exact fun x y => (OrderIso.funUnique (Fin 1) ℝ).symm.le_iff_le
      · exact (volume_preserving_funUnique (Fin 1) ℝ).symm _
      · intro x; rw [Fin.sum_univ_one, hF', e_symm, Pi.single_eq_same, one_smul]
      · rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hle] at Hi
        exact Hi.congr_set_ae Ioc_ae_eq_Icc.symm
    _ = f b - f a := by
      simp only [e, Fin.sum_univ_one, e_symm]
      have : ∀ c : ℝ, const (Fin 0) c = isEmptyElim := fun c => Subsingleton.elim _ _
      simp [this, volume_pi, Measure.pi_of_empty fun _ : Fin 0 => volume]


/-- **Fundamental theorem of calculus, part 2**. This version assumes that `f` is continuous on the
interval and is differentiable off a countable set `s`.

See also `intervalIntegral.integral_eq_sub_of_hasDeriv_right` for a version that
only assumes right differentiability of `f`.
-/
theorem integral_eq_of_hasDerivWithinAt_off_countable (f f' : ℝ → E) {a b : ℝ} {s : Set ℝ}
    (hs : s.Countable) (Hc : ContinuousOn f [[a, b]])
    (Hd : ∀ x ∈ Ioo (min a b) (max a b) \ s, HasDerivAt f (f' x) x)
    (Hi : IntervalIntegrable f' volume a b) : ∫ x in a..b, f' x = f b - f a := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f f' : Real → E
    a b : Real
    s : Set Real
    hs : s.Countable
    Hc : ContinuousOn f (Set.uIcc a b)
    Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo (Min.min a b) (Max.max …
    Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
    ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
  -/
  rcases le_total a b with hab | hab
    /-
      case inl
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f f' : Real → E
      a b : Real
      s : Set Real
      hs : s.Countable
      Hc : ContinuousOn f (Set.uIcc a b)
      Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo (Min.min a b) (Max.max …
      Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hab : LE.le a b
      ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
    -/
  · simp only [uIcc_of_le hab, min_eq_left hab, max_eq_right hab] at *
    /-
      case inl
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f f' : Real → E
      a b : Real
      s : Set Real
      hs : s.Countable
      Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hab : LE.le a b
      Hc : ContinuousOn f (Set.Icc a b)
      Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo a b) s) x → HasDerivAt …
      ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
    -/
    exact integral_eq_of_hasDerivWithinAt_off_countable_of_le f f' hab hs Hc Hd Hi
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f f' : Real → E
      a b : Real
      s : Set Real
      hs : s.Countable
      Hc : ContinuousOn f (Set.uIcc a b)
      Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo (Min.min a b) (Max.max …
      Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hab : LE.le b a
      ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
    -/
  · simp only [uIcc_of_ge hab, min_eq_right hab, max_eq_left hab] at *
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f f' : Real → E
      a b : Real
      s : Set Real
      hs : s.Countable
      Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hab : LE.le b a
      Hc : ContinuousOn f (Set.Icc b a)
      Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo b a) s) x → HasDerivAt …
      ⊢ Eq (intervalIntegral (fun x => f' x) a b MeasureTheory.MeasureSpace.volume)  …
    -/
    rw [intervalIntegral.integral_symm, neg_eq_iff_eq_neg, neg_sub]
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f f' : Real → E
      a b : Real
      s : Set Real
      hs : s.Countable
      Hi : IntervalIntegrable f' MeasureTheory.MeasureSpace.volume a b
      hab : LE.le b a
      Hc : ContinuousOn f (Set.Icc b a)
      Hd : ∀ (x : Real), Membership.mem (SDiff.sdiff (Set.Ioo b a) s) x → HasDerivAt …
      ⊢ Eq (intervalIntegral (fun x => f' x) b a MeasureTheory.MeasureSpace.volume)  …
    -/
    exact integral_eq_of_hasDerivWithinAt_off_countable_of_le f f' hab hs Hc Hd Hi.symm
    /-
      🎉 no goals
    -/


/-- **Divergence theorem** for functions on the plane along rectangles. It is formulated in terms of
two functions `f g : ℝ × ℝ → E` and an integral over `Icc a b = [a.1, b.1] × [a.2, b.2]`, where
`a b : ℝ × ℝ`, `a ≤ b`. When thinking of `f` and `g` as the two coordinates of a single function
`F : ℝ × ℝ → E × E` and when `E = ℝ`, this is the usual statement that the integral of the
divergence of `F` inside the rectangle equals the integral of the normal derivative of `F` along the
boundary.

See also `MeasureTheory.integral2_divergence_prod_of_hasFDerivWithinAt_off_countable` for a
version that does not assume `a ≤ b` and uses iterated interval integral instead of the integral
over `Icc a b`. -/
theorem integral_divergence_prod_Icc_of_hasFDerivWithinAt_off_countable_of_le (f g : ℝ × ℝ → E)
    (f' g' : ℝ × ℝ → ℝ × ℝ →L[ℝ] E) (a b : ℝ × ℝ) (hle : a ≤ b) (s : Set (ℝ × ℝ)) (hs : s.Countable)
    (Hcf : ContinuousOn f (Icc a b)) (Hcg : ContinuousOn g (Icc a b))
    (Hdf : ∀ x ∈ Ioo a.1 b.1 ×ˢ Ioo a.2 b.2 \ s, HasFDerivAt f (f' x) x)
    (Hdg : ∀ x ∈ Ioo a.1 b.1 ×ˢ Ioo a.2 b.2 \ s, HasFDerivAt g (g' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            f g : Prod Real Real → E
            f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
            a b : Prod Real Real
            hle : LE.le a b
            s : Set (Prod Real Real)
            hs : s.Countable
            Hcf : ContinuousOn f (Set.Icc a b)
            Hcg : ContinuousOn g (Set.Icc a b)
            Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
            Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
            ⊢ MeasureTheory.Measure (Prod Real Real)
          -/
    (Hi : IntegrableOn (fun x => f' x (1, 0) + g' x (0, 1)) (Icc a b)) :
          /-
            🎉 no goals
          -/
    (∫ x in Icc a b, f' x (1, 0) + g' x (0, 1)) =
      (((∫ x in a.1..b.1, g (x, b.2)) - ∫ x in a.1..b.1, g (x, a.2)) +
          ∫ y in a.2..b.2, f (b.1, y)) -
        ∫ y in a.2..b.2, f (a.1, y) :=
  let e : (ℝ × ℝ) ≃L[ℝ] ℝ² := (ContinuousLinearEquiv.finTwoArrow ℝ ℝ).symm
  calc
    (∫ x in Icc a b, f' x (1, 0) + g' x (0, 1)) =
        ∑ i : Fin 2,
          ((∫ x in Icc (e a ∘ i.succAbove) (e b ∘ i.succAbove),
              ![f, g] i (e.symm <| i.insertNth (e b i) x)) -
            ∫ x in Icc (e a ∘ i.succAbove) (e b ∘ i.succAbove),
              ![f, g] i (e.symm <| i.insertNth (e a i) x)) := by
      refine integral_divergence_of_hasFDerivWithinAt_off_countable_of_equiv e ?_ ?_ ![f, g]
        ![f', g'] s hs a b hle ?_ (fun x hx => ?_) _ ?_ Hi
        /-
          case refine_1
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          ⊢ ∀ (x y : Prod Real Real), Iff (LE.le (e x) (e y)) (LE.le x y)
        -/
      · exact fun x y => (OrderIso.finTwoArrowIso ℝ).symm.le_iff_le
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          ⊢ MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume Measu …
        -/
      · exact (volume_preserving_finTwoArrow ℝ).symm _
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          ⊢ ∀ (i : Fin (HAdd.hAdd 1 1)), ContinuousOn (Matrix.vecCons f (Matrix.vecCons  …
        -/
      · exact Fin.forall_fin_two.2 ⟨Hcf, Hcg⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_4
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          x : Prod Real Real
          hx : Membership.mem (SDiff.sdiff (interior (Set.Icc a b)) s) x
          ⊢ ∀ (i : Fin (HAdd.hAdd 1 1)), HasFDerivAt (Matrix.vecCons f (Matrix.vecCons g …
        -/
      · rw [Icc_prod_eq, interior_prod_eq, interior_Icc, interior_Icc] at hx
        /-
          case refine_4
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          x : Prod Real Real
          hx : Membership.mem (SDiff.sdiff (SProd.sprod (Set.Ioo a.1 b.1) (Set.Ioo a.2 b …
          ⊢ ∀ (i : Fin (HAdd.hAdd 1 1)), HasFDerivAt (Matrix.vecCons f (Matrix.vecCons g …
        -/
        exact Fin.forall_fin_two.2 ⟨Hdf x hx, Hdg x hx⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_5
          E : Type u
          inst✝² : NormedAddCommGroup E
          inst✝¹ : NormedSpace Real E
          inst✝ : CompleteSpace E
          f g : Prod Real Real → E
          f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
          a b : Prod Real Real
          hle : LE.le a b
          s : Set (Prod Real Real)
          hs : s.Countable
          Hcf : ContinuousOn f (Set.Icc a b)
          Hcg : ContinuousOn g (Set.Icc a b)
          Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
          Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
          e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
          ⊢ ∀ (x : Prod Real Real), Eq (HAdd.hAdd ((f' x) { fst := 1, snd := 0 }) ((g' x …
        -/
      · intro x; rw [Fin.sum_univ_two]; rfl
                                        /-
                                          🎉 no goals
                                        -/
    _ = ((∫ y in Icc a.2 b.2, f (b.1, y)) - ∫ y in Icc a.2 b.2, f (a.1, y)) +
          ((∫ x in Icc a.1 b.1, g (x, b.2)) - ∫ x in Icc a.1 b.1, g (x, a.2)) := by
      have : ∀ (a b : ℝ¹) (f : ℝ¹ → E),
          ∫ x in Icc a b, f x = ∫ x in Icc (a 0) (b 0), f fun _ => x := fun a b f ↦ by
        convert (((volume_preserving_funUnique (Fin 1) ℝ).symm _).setIntegral_preimage_emb
          (MeasurableEquiv.measurableEmbedding _) f _).symm
        exact ((OrderIso.funUnique (Fin 1) ℝ).symm.preimage_Icc a b).symm
      /-
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        f g : Prod Real Real → E
        f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
        a b : Prod Real Real
        hle : LE.le a b
        s : Set (Prod Real Real)
        hs : s.Countable
        Hcf : ContinuousOn f (Set.Icc a b)
        Hcg : ContinuousOn g (Set.Icc a b)
        Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
        e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
        this : ∀ (a b : Fin 1 → Real) (f : (Fin 1 → Real) → E), Eq (MeasureTheory.inte …
        ⊢ Eq (Finset.univ.sum fun i => HSub.hSub (MeasureTheory.integral (MeasureTheor …
      -/
      simp only [Fin.sum_univ_two, this]
      /-
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        f g : Prod Real Real → E
        f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
        a b : Prod Real Real
        hle : LE.le a b
        s : Set (Prod Real Real)
        hs : s.Countable
        Hcf : ContinuousOn f (Set.Icc a b)
        Hcg : ContinuousOn g (Set.Icc a b)
        Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
        e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
        this : ∀ (a b : Fin 1 → Real) (f : (Fin 1 → Real) → E), Eq (MeasureTheory.inte …
        ⊢ Eq (HAdd.hAdd (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace …
      -/
      rfl
      /-
        🎉 no goals
      -/
    _ = (((∫ x in a.1..b.1, g (x, b.2)) - ∫ x in a.1..b.1, g (x, a.2)) +
            ∫ y in a.2..b.2, f (b.1, y)) - ∫ y in a.2..b.2, f (a.1, y) := by
      simp only [intervalIntegral.integral_of_le hle.1, intervalIntegral.integral_of_le hle.2,
        setIntegral_congr_set (Ioc_ae_eq_Icc (α := ℝ) (μ := volume))]
      /-
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        f g : Prod Real Real → E
        f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
        a b : Prod Real Real
        hle : LE.le a b
        s : Set (Prod Real Real)
        hs : s.Countable
        Hcf : ContinuousOn f (Set.Icc a b)
        Hcg : ContinuousOn g (Set.Icc a b)
        Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
        Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
        e : ContinuousLinearEquiv (RingHom.id Real) (Prod Real Real) (Fin 2 → Real) := …
        ⊢ Eq (HAdd.hAdd (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace …
      -/
      /-
        🎉 no goals
      -/
      abel
      /-
        🎉 no goals
      -/


/-- **Divergence theorem** for functions on the plane. It is formulated in terms of two functions
`f g : ℝ × ℝ → E` and iterated integral `∫ x in a₁..b₁, ∫ y in a₂..b₂, _`, where
`a₁ a₂ b₁ b₂ : ℝ`. When thinking of `f` and `g` as the two coordinates of a single function
`F : ℝ × ℝ → E × E` and when `E = ℝ`, this is the usual statement that the integral of the
divergence of `F` inside the rectangle with vertices `(aᵢ, bⱼ)`, `i, j =1,2`, equals the integral of
the normal derivative of `F` along the boundary.

See also `MeasureTheory.integral_divergence_prod_Icc_of_hasFDerivWithinAt_off_countable_of_le`
for a version that uses an integral over `Icc a b`, where `a b : ℝ × ℝ`, `a ≤ b`. -/
theorem integral2_divergence_prod_of_hasFDerivWithinAt_off_countable (f g : ℝ × ℝ → E)
    (f' g' : ℝ × ℝ → ℝ × ℝ →L[ℝ] E) (a₁ a₂ b₁ b₂ : ℝ) (s : Set (ℝ × ℝ)) (hs : s.Countable)
    (Hcf : ContinuousOn f ([[a₁, b₁]] ×ˢ [[a₂, b₂]]))
    (Hcg : ContinuousOn g ([[a₁, b₁]] ×ˢ [[a₂, b₂]]))
    (Hdf : ∀ x ∈ Ioo (min a₁ b₁) (max a₁ b₁) ×ˢ Ioo (min a₂ b₂) (max a₂ b₂) \ s,
      HasFDerivAt f (f' x) x)
    (Hdg : ∀ x ∈ Ioo (min a₁ b₁) (max a₁ b₁) ×ˢ Ioo (min a₂ b₂) (max a₂ b₂) \ s,
      HasFDerivAt g (g' x) x)
          /-
            E : Type u
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Real E
            inst✝ : CompleteSpace E
            f g : Prod Real Real → E
            f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
            a₁ a₂ b₁ b₂ : Real
            s : Set (Prod Real Real)
            hs : s.Countable
            Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
            Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
            Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
            Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
            ⊢ MeasureTheory.Measure (Prod Real Real)
          -/
    (Hi : IntegrableOn (fun x => f' x (1, 0) + g' x (0, 1)) ([[a₁, b₁]] ×ˢ [[a₂, b₂]])) :
          /-
            🎉 no goals
          -/
    (∫ x in a₁..b₁, ∫ y in a₂..b₂, f' (x, y) (1, 0) + g' (x, y) (0, 1)) =
      (((∫ x in a₁..b₁, g (x, b₂)) - ∫ x in a₁..b₁, g (x, a₂)) + ∫ y in a₂..b₂, f (b₁, y)) -
        ∫ y in a₂..b₂, f (a₁, y) := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Prod Real Real → E
    f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
    a₁ a₂ b₁ b₂ : Real
    s : Set (Prod Real Real)
    hs : s.Countable
    Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
    ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
  -/
  wlog h₁ : a₁ ≤ b₁ generalizing a₁ b₁
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₁ a₂ b₁ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      this : ∀ (a₁ b₁ : Real), ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIc …
      h₁ : Not (LE.le a₁ b₁)
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
  · specialize this b₁ a₁
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₁ a₂ b₁ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : Not (LE.le a₁ b₁)
      this : ContinuousOn f (SProd.sprod (Set.uIcc b₁ a₁) (Set.uIcc a₂ b₂)) → Contin …
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
    rw [uIcc_comm b₁ a₁, min_comm b₁ a₁, max_comm b₁ a₁] at this
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₁ a₂ b₁ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : Not (LE.le a₁ b₁)
      this : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂)) → Contin …
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
    simp only [intervalIntegral.integral_symm b₁ a₁]
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₁ a₂ b₁ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : Not (LE.le a₁ b₁)
      this : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂)) → Contin …
      ⊢ Eq (Neg.neg (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd …
    -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    refine (congr_arg Neg.neg (this Hcf Hcg Hdf Hdg Hi (le_of_not_le h₁))).trans ?_; abel
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Prod Real Real → E
    f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
    a₂ b₂ : Real
    s : Set (Prod Real Real)
    hs : s.Countable
    a₁ b₁ : Real
    Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
    h₁ : LE.le a₁ b₁
    ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
  -/
  wlog h₂ : a₂ ≤ b₂ generalizing a₂ b₂
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₂ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      a₁ b₁ : Real
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : LE.le a₁ b₁
      this : ∀ (a₂ b₂ : Real), ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIc …
      h₂ : Not (LE.le a₂ b₂)
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
  · specialize this b₂ a₂
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₂ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      a₁ b₁ : Real
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : LE.le a₁ b₁
      h₂ : Not (LE.le a₂ b₂)
      this : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc b₂ a₂)) → Contin …
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
    rw [uIcc_comm b₂ a₂, min_comm b₂ a₂, max_comm b₂ a₂] at this
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₂ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      a₁ b₁ : Real
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : LE.le a₁ b₁
      h₂ : Not (LE.le a₂ b₂)
      this : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂)) → Contin …
      ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
    -/
    simp only [intervalIntegral.integral_symm b₂ a₂, intervalIntegral.integral_neg]
    /-
      case inr
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      f g : Prod Real Real → E
      f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
      a₂ b₂ : Real
      s : Set (Prod Real Real)
      hs : s.Countable
      a₁ b₁ : Real
      Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
      Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
      Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
      h₁ : LE.le a₁ b₁
      h₂ : Not (LE.le a₂ b₂)
      this : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂)) → Contin …
      ⊢ Eq (Neg.neg (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd …
    -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    refine (congr_arg Neg.neg (this Hcf Hcg Hdf Hdg Hi (le_of_not_le h₂))).trans ?_; abel
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    f g : Prod Real Real → E
    f' g' : Prod Real Real → ContinuousLinearMap (RingHom.id Real) (Prod Real Real …
    s : Set (Prod Real Real)
    hs : s.Countable
    a₁ b₁ : Real
    h₁ : LE.le a₁ b₁
    a₂ b₂ : Real
    Hcf : ContinuousOn f (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hcg : ContinuousOn g (SProd.sprod (Set.uIcc a₁ b₁) (Set.uIcc a₂ b₂))
    Hdf : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hdg : ∀ (x : Prod Real Real), Membership.mem (SDiff.sdiff (SProd.sprod (Set.Io …
    Hi : MeasureTheory.IntegrableOn (fun x => HAdd.hAdd ((f' x) { fst := 1, snd := …
    h₂ : LE.le a₂ b₂
    ⊢ Eq (intervalIntegral (fun x => intervalIntegral (fun y => HAdd.hAdd ((f' { f …
  -/
  simp only [uIcc_of_le h₁, uIcc_of_le h₂, min_eq_left, max_eq_right, h₁, h₂] at Hcf Hcg Hdf Hdg Hi
  calc
    (∫ x in a₁..b₁, ∫ y in a₂..b₂, f' (x, y) (1, 0) + g' (x, y) (0, 1)) =
        ∫ x in Icc a₁ b₁, ∫ y in Icc a₂ b₂, f' (x, y) (1, 0) + g' (x, y) (0, 1) := by
      simp only [intervalIntegral.integral_of_le, h₁, h₂,
        setIntegral_congr_set (Ioc_ae_eq_Icc (α := ℝ) (μ := volume))]
    _ = ∫ x in Icc a₁ b₁ ×ˢ Icc a₂ b₂, f' x (1, 0) + g' x (0, 1) := (setIntegral_prod _ Hi).symm
    _ = (((∫ x in a₁..b₁, g (x, b₂)) - ∫ x in a₁..b₁, g (x, a₂)) + ∫ y in a₂..b₂, f (b₁, y)) -
          ∫ y in a₂..b₂, f (a₁, y) := by
      rw [Icc_prod_Icc] at *
      apply integral_divergence_prod_Icc_of_hasFDerivWithinAt_off_countable_of_le f g f' g'
        (a₁, a₂) (b₁, b₂) ⟨h₁, h₂⟩ s <;> assumption


