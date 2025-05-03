/-- A monotone function continuous at the supremum of a nonempty set sends this supremum to
the supremum of the image of this set. -/
theorem MonotoneOn.map_csSup_of_continuousWithinAt {f : α → β} {A : Set α}
    (Cf : ContinuousWithinAt f A (sSup A))
    (Mf : MonotoneOn f A) (A_nonemp : A.Nonempty) (A_bdd : BddAbove A := by bddDefault) :
    f (sSup A) = sSup (f '' A) :=
  --This is a particular case of the more general `IsLUB.isLUB_of_tendsto`
  .symm <| ((isLUB_csSup A_nonemp A_bdd).isLUB_of_tendsto Mf A_nonemp <|
    Cf.mono_left fun ⦃_⦄ a ↦ a).csSup_eq (A_nonemp.image f)


/-- A monotone function continuous at the supremum of a nonempty set sends this supremum to
the supremum of the image of this set. -/
theorem Monotone.map_csSup_of_continuousAt {f : α → β} {A : Set α}
    (Cf : ContinuousAt f (sSup A)) (Mf : Monotone f) (A_nonemp : A.Nonempty)
    (A_bdd : BddAbove A := by bddDefault) : f (sSup A) = sSup (f '' A) :=
  MonotoneOn.map_csSup_of_continuousWithinAt Cf.continuousWithinAt
    (Mf.monotoneOn _) A_nonemp A_bdd


@[deprecated (since := "2024-08-26")] alias Monotone.map_sSup_of_continuousAt' :=
  Monotone.map_csSup_of_continuousAt


/-- A monotone function continuous at the indexed supremum over a nonempty `Sort` sends this indexed
supremum to the indexed supremum of the composition. -/
theorem Monotone.map_ciSup_of_continuousAt {ι : Sort*} [Nonempty ι] {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iSup g)) (Mf : Monotone f)
    (bdd : BddAbove (range g) := by bddDefault) : f (⨆ i, g i) = ⨆ i, f (g i) := by
  rw [iSup, Monotone.map_csSup_of_continuousAt Cf Mf (range_nonempty g) bdd, ← range_comp, iSup,
    comp_def]


@[deprecated (since := "2024-08-26")] alias Monotone.map_iSup_of_continuousAt' :=
  Monotone.map_ciSup_of_continuousAt


/-- A monotone function continuous at the infimum of a nonempty set sends this infimum to
the infimum of the image of this set. -/
theorem MonotoneOn.map_csInf_of_continuousWithinAt {f : α → β} {A : Set α}
    (Cf : ContinuousWithinAt f A (sInf A))
    (Mf : MonotoneOn f A) (A_nonemp : A.Nonempty) (A_bdd : BddBelow A := by bddDefault) :
    f (sInf A) = sInf (f '' A) :=
  MonotoneOn.map_csSup_of_continuousWithinAt (α := αᵒᵈ) (β := βᵒᵈ) Cf Mf.dual A_nonemp A_bdd


/-- A monotone function continuous at the infimum of a nonempty set sends this infimum to
the infimum of the image of this set. -/
theorem Monotone.map_csInf_of_continuousAt {f : α → β} {A : Set α} (Cf : ContinuousAt f (sInf A))
    (Mf : Monotone f) (A_nonemp : A.Nonempty) (A_bdd : BddBelow A := by bddDefault) :
    f (sInf A) = sInf (f '' A) :=
  Monotone.map_csSup_of_continuousAt (α := αᵒᵈ) (β := βᵒᵈ) Cf Mf.dual A_nonemp A_bdd


@[deprecated (since := "2024-08-26")] alias Monotone.map_sInf_of_continuousAt' :=
  Monotone.map_csInf_of_continuousAt


/-- A monotone function continuous at the indexed infimum over a nonempty `Sort` sends this indexed
infimum to the indexed infimum of the composition. -/
theorem Monotone.map_ciInf_of_continuousAt {ι : Sort*} [Nonempty ι] {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iInf g)) (Mf : Monotone f)
    (bdd : BddBelow (range g) := by bddDefault) : f (⨅ i, g i) = ⨅ i, f (g i) := by
  rw [iInf, Monotone.map_csInf_of_continuousAt Cf Mf (range_nonempty g) bdd, ← range_comp, iInf,
    comp_def]


@[deprecated (since := "2024-08-26")] alias Monotone.map_iInf_of_continuousAt' :=
  Monotone.map_ciInf_of_continuousAt


/-- An antitone function continuous at the infimum of a nonempty set sends this infimum to
the supremum of the image of this set. -/
theorem AntitoneOn.map_csInf_of_continuousWithinAt {f : α → β} {A : Set α}
    (Cf : ContinuousWithinAt f A (sInf A))
    (Af : AntitoneOn f A) (A_nonemp : A.Nonempty) (A_bdd : BddBelow A := by bddDefault) :
    f (sInf A) = sSup (f '' A) :=
  MonotoneOn.map_csInf_of_continuousWithinAt (β := βᵒᵈ) Cf Af.dual_right A_nonemp A_bdd


/-- An antitone function continuous at the infimum of a nonempty set sends this infimum to
the supremum of the image of this set. -/
theorem Antitone.map_csInf_of_continuousAt {f : α → β} {A : Set α} (Cf : ContinuousAt f (sInf A))
    (Af : Antitone f) (A_nonemp : A.Nonempty) (A_bdd : BddBelow A := by bddDefault) :
    f (sInf A) = sSup (f '' A) :=
  Monotone.map_csInf_of_continuousAt (β := βᵒᵈ) Cf Af.dual_right A_nonemp A_bdd


@[deprecated (since := "2024-08-26")] alias Antitone.map_sInf_of_continuousAt' :=
  Antitone.map_csInf_of_continuousAt


/-- An antitone function continuous at the indexed infimum over a nonempty `Sort` sends this indexed
infimum to the indexed supremum of the composition. -/
theorem Antitone.map_ciInf_of_continuousAt {ι : Sort*} [Nonempty ι] {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iInf g)) (Af : Antitone f)
    (bdd : BddBelow (range g) := by bddDefault) : f (⨅ i, g i) = ⨆ i, f (g i) := by
  rw [iInf, Antitone.map_csInf_of_continuousAt Cf Af (range_nonempty g) bdd, ← range_comp, iSup,
    comp_def]


@[deprecated (since := "2024-08-26")] alias Antitone.map_iInf_of_continuousAt' :=
  Antitone.map_ciInf_of_continuousAt


/-- An antitone function continuous at the supremum of a nonempty set sends this supremum to
the infimum of the image of this set. -/
theorem AntitoneOn.map_csSup_of_continuousWithinAt {f : α → β} {A : Set α}
    (Cf : ContinuousWithinAt f A (sSup A))
    (Af : AntitoneOn f A) (A_nonemp : A.Nonempty) (A_bdd : BddAbove A := by bddDefault) :
    f (sSup A) = sInf (f '' A) :=
  MonotoneOn.map_csSup_of_continuousWithinAt (β := βᵒᵈ) Cf Af.dual_right A_nonemp A_bdd


/-- An antitone function continuous at the supremum of a nonempty set sends this supremum to
the infimum of the image of this set. -/
theorem Antitone.map_csSup_of_continuousAt {f : α → β} {A : Set α} (Cf : ContinuousAt f (sSup A))
    (Af : Antitone f) (A_nonemp : A.Nonempty) (A_bdd : BddAbove A := by bddDefault) :
    f (sSup A) = sInf (f '' A) :=
  Monotone.map_csSup_of_continuousAt (β := βᵒᵈ) Cf Af.dual_right A_nonemp A_bdd


@[deprecated (since := "2024-08-26")] alias Antitone.map_sSup_of_continuousAt' :=
  Antitone.map_csSup_of_continuousAt


/-- An antitone function continuous at the indexed supremum over a nonempty `Sort` sends this
indexed supremum to the indexed infimum of the composition. -/
theorem Antitone.map_ciSup_of_continuousAt {ι : Sort*} [Nonempty ι] {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iSup g)) (Af : Antitone f)
    (bdd : BddAbove (range g) := by bddDefault) : f (⨆ i, g i) = ⨅ i, f (g i) := by
  rw [iSup, Antitone.map_csSup_of_continuousAt Cf Af (range_nonempty g) bdd, ← range_comp, iInf,
    comp_def]


@[deprecated (since := "2024-08-26")] alias Antitone.map_iSup_of_continuousAt' :=
  Antitone.map_ciSup_of_continuousAt


theorem sSup_mem_closure {s : Set α} (hs : s.Nonempty) : sSup s ∈ closure s :=
  (isLUB_sSup s).mem_closure hs


theorem sInf_mem_closure {s : Set α} (hs : s.Nonempty) : sInf s ∈ closure s :=
  (isGLB_sInf s).mem_closure hs


theorem IsClosed.sSup_mem {s : Set α} (hs : s.Nonempty) (hc : IsClosed s) : sSup s ∈ s :=
  (isLUB_sSup s).mem_of_isClosed hs hc


theorem IsClosed.sInf_mem {s : Set α} (hs : s.Nonempty) (hc : IsClosed s) : sInf s ∈ s :=
  (isGLB_sInf s).mem_of_isClosed hs hc


/-- A monotone function `f` sending `bot` to `bot` and continuous at the supremum of a set sends
this supremum to the supremum of the image of this set. -/
theorem MonotoneOn.map_sSup_of_continuousWithinAt {f : α → β} {s : Set α}
    (Cf : ContinuousWithinAt f s (sSup s))
    (Mf : MonotoneOn f s) (fbot : f ⊥ = ⊥) : f (sSup s) = sSup (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : CompleteLinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : CompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderClosedTopology β
    f : α → β
    s : Set α
    Cf : ContinuousWithinAt f s (SupSet.sSup s)
    Mf : MonotoneOn f s
    fbot : Eq (f Bot.bot) Bot.bot
    ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.image f s))
  -/
  rcases s.eq_empty_or_nonempty with h | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁵ : CompleteLinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : CompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderClosedTopology β
      f : α → β
      s : Set α
      Cf : ContinuousWithinAt f s (SupSet.sSup s)
      Mf : MonotoneOn f s
      fbot : Eq (f Bot.bot) Bot.bot
      h : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.image f s))
    -/
  · simp [h, fbot]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝⁵ : CompleteLinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : CompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderClosedTopology β
      f : α → β
      s : Set α
      Cf : ContinuousWithinAt f s (SupSet.sSup s)
      Mf : MonotoneOn f s
      fbot : Eq (f Bot.bot) Bot.bot
      h : s.Nonempty
      ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.image f s))
    -/
  · exact Mf.map_csSup_of_continuousWithinAt Cf h
    /-
      🎉 no goals
    -/


/-- A monotone function `f` sending `bot` to `bot` and continuous at the supremum of a set sends
this supremum to the supremum of the image of this set. -/
theorem Monotone.map_sSup_of_continuousAt {f : α → β} {s : Set α} (Cf : ContinuousAt f (sSup s))
    (Mf : Monotone f) (fbot : f ⊥ = ⊥) : f (sSup s) = sSup (f '' s) :=
  MonotoneOn.map_sSup_of_continuousWithinAt Cf.continuousWithinAt (Mf.monotoneOn _) fbot


/-- If a monotone function sending `bot` to `bot` is continuous at the indexed supremum over
a `Sort`, then it sends this indexed supremum to the indexed supremum of the composition. -/
theorem Monotone.map_iSup_of_continuousAt {ι : Sort*} {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iSup g)) (Mf : Monotone f) (fbot : f ⊥ = ⊥) :
    f (⨆ i, g i) = ⨆ i, f (g i) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : CompleteLinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : CompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderClosedTopology β
    ι : Sort u_3
    f : α → β
    g : ι → α
    Cf : ContinuousAt f (iSup g)
    Mf : Monotone f
    fbot : Eq (f Bot.bot) Bot.bot
    ⊢ Eq (f (iSup fun i => g i)) (iSup fun i => f (g i))
  -/
  rw [iSup, Mf.map_sSup_of_continuousAt Cf fbot, ← range_comp, iSup, comp_def]
  /-
    🎉 no goals
  -/


/-- A monotone function `f` sending `top` to `top` and continuous at the infimum of a set sends
this infimum to the infimum of the image of this set. -/
theorem MonotoneOn.map_sInf_of_continuousWithinAt {f : α → β} {s : Set α}
    (Cf : ContinuousWithinAt f s (sInf s)) (Mf : MonotoneOn f s) (ftop : f ⊤ = ⊤) :
    f (sInf s) = sInf (f '' s) :=
  MonotoneOn.map_sSup_of_continuousWithinAt (α := αᵒᵈ) (β := βᵒᵈ) Cf Mf.dual ftop


/-- A monotone function `f` sending `top` to `top` and continuous at the infimum of a set sends
this infimum to the infimum of the image of this set. -/
theorem Monotone.map_sInf_of_continuousAt {f : α → β} {s : Set α} (Cf : ContinuousAt f (sInf s))
    (Mf : Monotone f) (ftop : f ⊤ = ⊤) : f (sInf s) = sInf (f '' s) :=
  Monotone.map_sSup_of_continuousAt (α := αᵒᵈ) (β := βᵒᵈ) Cf Mf.dual ftop


/-- If a monotone function sending `top` to `top` is continuous at the indexed infimum over
a `Sort`, then it sends this indexed infimum to the indexed infimum of the composition. -/
theorem Monotone.map_iInf_of_continuousAt {ι : Sort*} {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iInf g)) (Mf : Monotone f) (ftop : f ⊤ = ⊤) : f (iInf g) = iInf (f ∘ g) :=
  Monotone.map_iSup_of_continuousAt (α := αᵒᵈ) (β := βᵒᵈ) Cf Mf.dual ftop


/-- An antitone function `f` sending `bot` to `top` and continuous at the supremum of a set sends
this supremum to the infimum of the image of this set. -/
theorem AntitoneOn.map_sSup_of_continuousWithinAt {f : α → β} {s : Set α}
    (Cf : ContinuousWithinAt f s (sSup s)) (Af : AntitoneOn f s) (fbot : f ⊥ = ⊤) :
    f (sSup s) = sInf (f '' s) :=
  MonotoneOn.map_sSup_of_continuousWithinAt
    (show ContinuousWithinAt (OrderDual.toDual ∘ f) s (sSup s) from Cf) Af fbot


/-- An antitone function `f` sending `bot` to `top` and continuous at the supremum of a set sends
this supremum to the infimum of the image of this set. -/
theorem Antitone.map_sSup_of_continuousAt {f : α → β} {s : Set α} (Cf : ContinuousAt f (sSup s))
    (Af : Antitone f) (fbot : f ⊥ = ⊤) : f (sSup s) = sInf (f '' s) :=
  Monotone.map_sSup_of_continuousAt (show ContinuousAt (OrderDual.toDual ∘ f) (sSup s) from Cf) Af
    fbot


/-- An antitone function sending `bot` to `top` is continuous at the indexed supremum over
a `Sort`, then it sends this indexed supremum to the indexed supremum of the composition. -/
theorem Antitone.map_iSup_of_continuousAt {ι : Sort*} {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iSup g)) (Af : Antitone f) (fbot : f ⊥ = ⊤) :
    f (⨆ i, g i) = ⨅ i, f (g i) :=
  Monotone.map_iSup_of_continuousAt (show ContinuousAt (OrderDual.toDual ∘ f) (iSup g) from Cf) Af
    fbot


/-- An antitone function `f` sending `top` to `bot` and continuous at the infimum of a set sends
this infimum to the supremum of the image of this set. -/
theorem AntitoneOn.map_sInf_of_continuousWithinAt {f : α → β} {s : Set α}
    (Cf : ContinuousWithinAt f s (sInf s)) (Af : AntitoneOn f s) (ftop : f ⊤ = ⊥) :
    f (sInf s) = sSup (f '' s) :=
  MonotoneOn.map_sInf_of_continuousWithinAt
    (show ContinuousWithinAt (OrderDual.toDual ∘ f) s (sInf s) from Cf) Af ftop


/-- An antitone function `f` sending `top` to `bot` and continuous at the infimum of a set sends
this infimum to the supremum of the image of this set. -/
theorem Antitone.map_sInf_of_continuousAt {f : α → β} {s : Set α} (Cf : ContinuousAt f (sInf s))
    (Af : Antitone f) (ftop : f ⊤ = ⊥) : f (sInf s) = sSup (f '' s) :=
  Monotone.map_sInf_of_continuousAt (show ContinuousAt (OrderDual.toDual ∘ f) (sInf s) from Cf) Af
    ftop


/-- If an antitone function sending `top` to `bot` is continuous at the indexed infimum over
a `Sort`, then it sends this indexed infimum to the indexed supremum of the composition. -/
theorem Antitone.map_iInf_of_continuousAt {ι : Sort*} {f : α → β} {g : ι → α}
    (Cf : ContinuousAt f (iInf g)) (Af : Antitone f) (ftop : f ⊤ = ⊥) : f (iInf g) = iSup (f ∘ g) :=
  Monotone.map_iInf_of_continuousAt (show ContinuousAt (OrderDual.toDual ∘ f) (iInf g) from Cf) Af
    ftop


theorem csSup_mem_closure {s : Set α} (hs : s.Nonempty) (B : BddAbove s) : sSup s ∈ closure s :=
  (isLUB_csSup hs B).mem_closure hs


theorem csInf_mem_closure {s : Set α} (hs : s.Nonempty) (B : BddBelow s) : sInf s ∈ closure s :=
  (isGLB_csInf hs B).mem_closure hs


theorem IsClosed.csSup_mem {s : Set α} (hc : IsClosed s) (hs : s.Nonempty) (B : BddAbove s) :
    sSup s ∈ s :=
  (isLUB_csSup hs B).mem_of_isClosed hs hc


theorem IsClosed.csInf_mem {s : Set α} (hc : IsClosed s) (hs : s.Nonempty) (B : BddBelow s) :
    sInf s ∈ s :=
  (isGLB_csInf hs B).mem_of_isClosed hs hc


theorem IsClosed.isLeast_csInf {s : Set α} (hc : IsClosed s) (hs : s.Nonempty) (B : BddBelow s) :
    IsLeast s (sInf s) :=
  ⟨hc.csInf_mem hs B, (isGLB_csInf hs B).1⟩


theorem IsClosed.isGreatest_csSup {s : Set α} (hc : IsClosed s) (hs : s.Nonempty) (B : BddAbove s) :
    IsGreatest s (sSup s) :=
  IsClosed.isLeast_csInf (α := αᵒᵈ) hc hs B


lemma MonotoneOn.tendsto_nhdsWithin_Ioo_left {α β : Type*} [LinearOrder α] [TopologicalSpace α]
    [OrderTopology α] [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β]
    {f : α → β} {x y : α} (h_nonempty : (Ioo y x).Nonempty) (Mf : MonotoneOn f (Ioo y x))
    (h_bdd : BddAbove (f '' Ioo y x)) :
    Tendsto f (𝓝[<] x) (𝓝 (sSup (f '' Ioo y x))) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    x y : α
    h_nonempty : (Set.Ioo y x).Nonempty
    Mf : MonotoneOn f (Set.Ioo y x)
    h_bdd : BddAbove (Set.image f (Set.Ioo y x))
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (SupSet.sSup (Set.image f  …
  -/
  refine tendsto_order.2 ⟨fun l hl => ?_, fun m hm => ?_⟩
  · obtain ⟨z, ⟨yz, zx⟩, lz⟩ : ∃ a : α, a ∈ Ioo y x ∧ l < f a := by
      simpa only [mem_image, exists_prop, exists_exists_and_eq_and] using
        exists_lt_of_lt_csSup (h_nonempty.image _) hl
    /-
      case refine_1.intro.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo y x).Nonempty
      Mf : MonotoneOn f (Set.Ioo y x)
      h_bdd : BddAbove (Set.image f (Set.Ioo y x))
      l : β
      hl : LT.lt l (SupSet.sSup (Set.image f (Set.Ioo y x)))
      z : α
      lz : LT.lt l (f z)
      yz : LT.lt y z
      zx : LT.lt z x
      ⊢ Filter.Eventually (fun b => LT.lt l (f b)) (nhdsWithin x (Set.Iio x))
    -/
    filter_upwards [Ioo_mem_nhdsLT zx] with w hw
    /-
      case h
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo y x).Nonempty
      Mf : MonotoneOn f (Set.Ioo y x)
      h_bdd : BddAbove (Set.image f (Set.Ioo y x))
      l : β
      hl : LT.lt l (SupSet.sSup (Set.image f (Set.Ioo y x)))
      z : α
      lz : LT.lt l (f z)
      yz : LT.lt y z
      zx : LT.lt z x
      w : α
      hw : Membership.mem (Set.Ioo z x) w
      ⊢ LT.lt l (f w)
    -/
    exact lz.trans_le <| Mf ⟨yz, zx⟩ ⟨yz.trans_le hw.1.le, hw.2⟩ hw.1.le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo y x).Nonempty
      Mf : MonotoneOn f (Set.Ioo y x)
      h_bdd : BddAbove (Set.image f (Set.Ioo y x))
      m : β
      hm : GT.gt m (SupSet.sSup (Set.image f (Set.Ioo y x)))
      ⊢ Filter.Eventually (fun b => LT.lt (f b) m) (nhdsWithin x (Set.Iio x))
    -/
  · rcases h_nonempty with ⟨_, hy, hx⟩
    /-
      case refine_2.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      Mf : MonotoneOn f (Set.Ioo y x)
      h_bdd : BddAbove (Set.image f (Set.Ioo y x))
      m : β
      hm : GT.gt m (SupSet.sSup (Set.image f (Set.Ioo y x)))
      w✝ : α
      hy : LT.lt y w✝
      hx : LT.lt w✝ x
      ⊢ Filter.Eventually (fun b => LT.lt (f b) m) (nhdsWithin x (Set.Iio x))
    -/
    filter_upwards [Ioo_mem_nhdsLT (hy.trans hx)] with w hw
    /-
      case h
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      Mf : MonotoneOn f (Set.Ioo y x)
      h_bdd : BddAbove (Set.image f (Set.Ioo y x))
      m : β
      hm : GT.gt m (SupSet.sSup (Set.image f (Set.Ioo y x)))
      w✝ : α
      hy : LT.lt y w✝
      hx : LT.lt w✝ x
      w : α
      hw : Membership.mem (Set.Ioo y x) w
      ⊢ LT.lt (f w) m
    -/
    exact (le_csSup h_bdd (mem_image_of_mem _ hw)).trans_lt hm
    /-
      🎉 no goals
    -/


lemma MonotoneOn.tendsto_nhdsWithin_Ioo_right {α β : Type*} [LinearOrder α] [TopologicalSpace α]
    [OrderTopology α] [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β]
    {f : α → β} {x y : α} (h_nonempty : (Ioo x y).Nonempty) (Mf : MonotoneOn f (Ioo x y))
    (h_bdd : BddBelow (f '' Ioo x y)) :
    Tendsto f (𝓝[>] x) (𝓝 (sInf (f '' Ioo x y))) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    x y : α
    h_nonempty : (Set.Ioo x y).Nonempty
    Mf : MonotoneOn f (Set.Ioo x y)
    h_bdd : BddBelow (Set.image f (Set.Ioo x y))
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Ioi x)) (nhds (InfSet.sInf (Set.image f  …
  -/
  refine tendsto_order.2 ⟨fun l hl => ?_, fun m hm => ?_⟩
    /-
      case refine_1
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo x y).Nonempty
      Mf : MonotoneOn f (Set.Ioo x y)
      h_bdd : BddBelow (Set.image f (Set.Ioo x y))
      l : β
      hl : LT.lt l (InfSet.sInf (Set.image f (Set.Ioo x y)))
      ⊢ Filter.Eventually (fun b => LT.lt l (f b)) (nhdsWithin x (Set.Ioi x))
    -/
  · rcases h_nonempty with ⟨p, hy, hx⟩
    /-
      case refine_1.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      Mf : MonotoneOn f (Set.Ioo x y)
      h_bdd : BddBelow (Set.image f (Set.Ioo x y))
      l : β
      hl : LT.lt l (InfSet.sInf (Set.image f (Set.Ioo x y)))
      p : α
      hy : LT.lt x p
      hx : LT.lt p y
      ⊢ Filter.Eventually (fun b => LT.lt l (f b)) (nhdsWithin x (Set.Ioi x))
    -/
    filter_upwards [Ioo_mem_nhdsGT (hy.trans hx)] with w hw
    /-
      case h
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      Mf : MonotoneOn f (Set.Ioo x y)
      h_bdd : BddBelow (Set.image f (Set.Ioo x y))
      l : β
      hl : LT.lt l (InfSet.sInf (Set.image f (Set.Ioo x y)))
      p : α
      hy : LT.lt x p
      hx : LT.lt p y
      w : α
      hw : Membership.mem (Set.Ioo x y) w
      ⊢ LT.lt l (f w)
    -/
    exact hl.trans_le <| csInf_le h_bdd (mem_image_of_mem _ hw)
    /-
      🎉 no goals
    -/
  · obtain ⟨z, ⟨xz, zy⟩, zm⟩ : ∃ a : α, a ∈ Ioo x y ∧ f a < m := by
      simpa [mem_image, exists_prop, exists_exists_and_eq_and] using
        exists_lt_of_csInf_lt (h_nonempty.image _) hm
    /-
      case refine_2.intro.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo x y).Nonempty
      Mf : MonotoneOn f (Set.Ioo x y)
      h_bdd : BddBelow (Set.image f (Set.Ioo x y))
      m : β
      hm : GT.gt m (InfSet.sInf (Set.image f (Set.Ioo x y)))
      z : α
      zm : LT.lt (f z) m
      xz : LT.lt x z
      zy : LT.lt z y
      ⊢ Filter.Eventually (fun b => LT.lt (f b) m) (nhdsWithin x (Set.Ioi x))
    -/
    filter_upwards [Ioo_mem_nhdsGT xz] with w hw
    /-
      case h
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x y : α
      h_nonempty : (Set.Ioo x y).Nonempty
      Mf : MonotoneOn f (Set.Ioo x y)
      h_bdd : BddBelow (Set.image f (Set.Ioo x y))
      m : β
      hm : GT.gt m (InfSet.sInf (Set.image f (Set.Ioo x y)))
      z : α
      zm : LT.lt (f z) m
      xz : LT.lt x z
      zy : LT.lt z y
      w : α
      hw : Membership.mem (Set.Ioo x z) w
      ⊢ LT.lt (f w) m
    -/
    exact (Mf ⟨hw.1, hw.2.trans zy⟩ ⟨xz, zy⟩ hw.2.le).trans_lt zm
    /-
      🎉 no goals
    -/


lemma MonotoneOn.tendsto_nhdsLT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β} {x : α}
    (Mf : MonotoneOn f (Iio x)) (h_bdd : BddAbove (f '' Iio x)) :
    Tendsto f (𝓝[<] x) (𝓝 (sSup (f '' Iio x))) := by
  /-
    α : Type u_3
    β : Type u_4
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    x : α
    Mf : MonotoneOn f (Set.Iio x)
    h_bdd : BddAbove (Set.image f (Set.Iio x))
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (SupSet.sSup (Set.image f  …
  -/
  rcases eq_empty_or_nonempty (Iio x) with (h | h); · simp [h]
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    case inr
    α : Type u_3
    β : Type u_4
    inst✝⁵ : LinearOrder α
    inst✝⁴ : TopologicalSpace α
    inst✝³ : OrderTopology α
    inst✝² : ConditionallyCompleteLinearOrder β
    inst✝¹ : TopologicalSpace β
    inst✝ : OrderTopology β
    f : α → β
    x : α
    Mf : MonotoneOn f (Set.Iio x)
    h_bdd : BddAbove (Set.image f (Set.Iio x))
    h : (Set.Iio x).Nonempty
    ⊢ Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds (SupSet.sSup (Set.image f  …
  -/
  refine tendsto_order.2 ⟨fun l hl => ?_, fun m hm => ?_⟩
  · obtain ⟨z, zx, lz⟩ : ∃ a : α, a < x ∧ l < f a := by
      simpa only [mem_image, exists_prop, exists_exists_and_eq_and] using
        exists_lt_of_lt_csSup (h.image _) hl
    /-
      case inr.refine_1.intro.intro
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x : α
      Mf : MonotoneOn f (Set.Iio x)
      h_bdd : BddAbove (Set.image f (Set.Iio x))
      h : (Set.Iio x).Nonempty
      l : β
      hl : LT.lt l (SupSet.sSup (Set.image f (Set.Iio x)))
      z : α
      zx : LT.lt z x
      lz : LT.lt l (f z)
      ⊢ Filter.Eventually (fun b => LT.lt l (f b)) (nhdsWithin x (Set.Iio x))
    -/
    filter_upwards [Ioo_mem_nhdsLT zx] with y hy using lz.trans_le (Mf zx hy.2 hy.1.le)
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x : α
      Mf : MonotoneOn f (Set.Iio x)
      h_bdd : BddAbove (Set.image f (Set.Iio x))
      h : (Set.Iio x).Nonempty
      m : β
      hm : GT.gt m (SupSet.sSup (Set.image f (Set.Iio x)))
      ⊢ Filter.Eventually (fun b => LT.lt (f b) m) (nhdsWithin x (Set.Iio x))
    -/
  · refine mem_of_superset self_mem_nhdsWithin fun y hy => lt_of_le_of_lt ?_ hm
    /-
      case inr.refine_2
      α : Type u_3
      β : Type u_4
      inst✝⁵ : LinearOrder α
      inst✝⁴ : TopologicalSpace α
      inst✝³ : OrderTopology α
      inst✝² : ConditionallyCompleteLinearOrder β
      inst✝¹ : TopologicalSpace β
      inst✝ : OrderTopology β
      f : α → β
      x : α
      Mf : MonotoneOn f (Set.Iio x)
      h_bdd : BddAbove (Set.image f (Set.Iio x))
      h : (Set.Iio x).Nonempty
      m : β
      hm : GT.gt m (SupSet.sSup (Set.image f (Set.Iio x)))
      y : α
      hy : Membership.mem (Set.Iio x) y
      ⊢ LE.le (f y) (SupSet.sSup (Set.image f (Set.Iio x)))
    -/
    exact le_csSup h_bdd (mem_image_of_mem _ hy)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-22")]
alias MonotoneOn.tendsto_nhdsWithin_Iio := MonotoneOn.tendsto_nhdsLT


lemma MonotoneOn.tendsto_nhdsGT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β} {x : α}
    (Mf : MonotoneOn f (Ioi x)) (h_bdd : BddBelow (f '' Ioi x)) :
    Tendsto f (𝓝[>] x) (𝓝 (sInf (f '' Ioi x))) :=
  MonotoneOn.tendsto_nhdsLT (α := αᵒᵈ) (β := βᵒᵈ) Mf.dual h_bdd


@[deprecated (since := "2024-12-22")]
alias MonotoneOn.tendsto_nhdsWithin_Ioi := MonotoneOn.tendsto_nhdsGT


/-- A monotone map has a limit to the left of any point `x`, equal to `sSup (f '' (Iio x))`. -/
theorem Monotone.tendsto_nhdsLT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β}
    (Mf : Monotone f) (x : α) : Tendsto f (𝓝[<] x) (𝓝 (sSup (f '' Iio x))) :=
  MonotoneOn.tendsto_nhdsLT (Mf.monotoneOn _) (Mf.map_bddAbove bddAbove_Iio)


@[deprecated (since := "2024-12-22")]
alias Monotone.tendsto_nhdsWithin_Iio := Monotone.tendsto_nhdsLT


/-- A monotone map has a limit to the right of any point `x`, equal to `sInf (f '' (Ioi x))`. -/
theorem Monotone.tendsto_nhdsGT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β}
    (Mf : Monotone f) (x : α) : Tendsto f (𝓝[>] x) (𝓝 (sInf (f '' Ioi x))) :=
  Monotone.tendsto_nhdsLT (α := αᵒᵈ) (β := βᵒᵈ) Mf.dual x


@[deprecated (since := "2024-12-22")]
alias Monotone.tendsto_nhdsWithin_Ioi := Monotone.tendsto_nhdsGT


lemma AntitoneOn.tendsto_nhdsWithin_Ioo_left {α β : Type*} [LinearOrder α] [TopologicalSpace α]
    [OrderTopology α] [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β]
    {f : α → β} {x y : α} (h_nonempty : (Ioo y x).Nonempty) (Af : AntitoneOn f (Ioo y x))
    (h_bdd : BddBelow (f '' Ioo y x)) :
    Tendsto f (𝓝[<] x) (𝓝 (sInf (f '' Ioo y x))) :=
  MonotoneOn.tendsto_nhdsWithin_Ioo_left h_nonempty Af.dual_right h_bdd


lemma AntitoneOn.tendsto_nhdsWithin_Ioo_right {α β : Type*} [LinearOrder α] [TopologicalSpace α]
    [OrderTopology α] [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β]
    {f : α → β} {x y : α} (h_nonempty : (Ioo x y).Nonempty) (Af : AntitoneOn f (Ioo x y))
    (h_bdd : BddAbove (f '' Ioo x y)) :
    Tendsto f (𝓝[>] x) (𝓝 (sSup (f '' Ioo x y))) :=
  MonotoneOn.tendsto_nhdsWithin_Ioo_right h_nonempty Af.dual_right h_bdd


lemma AntitoneOn.tendsto_nhdsLT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β} {x : α}
    (Af : AntitoneOn f (Iio x)) (h_bdd : BddBelow (f '' Iio x)) :
    Tendsto f (𝓝[<] x) (𝓝 (sInf (f '' Iio x))) :=
  MonotoneOn.tendsto_nhdsLT Af.dual_right h_bdd


@[deprecated (since := "2024-12-22")]
alias AntitoneOn.tendsto_nhdsWithin_Iio := AntitoneOn.tendsto_nhdsLT


lemma AntitoneOn.tendsto_nhdsGT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β} {x : α}
    (Af : AntitoneOn f (Ioi x)) (h_bdd : BddAbove (f '' Ioi x)) :
    Tendsto f (𝓝[>] x) (𝓝 (sSup (f '' Ioi x))) :=
  MonotoneOn.tendsto_nhdsGT Af.dual_right h_bdd


@[deprecated (since := "2024-12-22")]
alias AntitoneOn.tendsto_nhdsWithin_Ioi := AntitoneOn.tendsto_nhdsGT


/-- An antitone map has a limit to the left of any point `x`, equal to `sInf (f '' (Iio x))`. -/
theorem Antitone.tendsto_nhdsLT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β}
    (Af : Antitone f) (x : α) : Tendsto f (𝓝[<] x) (𝓝 (sInf (f '' Iio x))) :=
  Monotone.tendsto_nhdsLT Af.dual_right x


@[deprecated (since := "2024-12-22")]
alias Antitone.tendsto_nhdsWithin_Iio := Antitone.tendsto_nhdsLT


/-- An antitone map has a limit to the right of any point `x`, equal to `sSup (f '' (Ioi x))`. -/
theorem Antitone.tendsto_nhdsGT {α β : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
    [ConditionallyCompleteLinearOrder β] [TopologicalSpace β] [OrderTopology β] {f : α → β}
    (Af : Antitone f) (x : α) : Tendsto f (𝓝[>] x) (𝓝 (sSup (f '' Ioi x))) :=
  Monotone.tendsto_nhdsGT Af.dual_right x


@[deprecated (since := "2024-12-22")]
alias Antitone.tendsto_nhdsWithin_Ioi := Antitone.tendsto_nhdsGT


